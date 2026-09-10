/* QEMU's `microbit` machine — an nRF51822, Cortex-M0, 256 KB flash, 16 KB SRAM.
 *
 * The part is chosen for being *small*: a generous machine would run the
 * harness without proving anything about the footprint claim, and 16 KB of SRAM
 * makes the streaming sink in `dm_vm.h` load-bearing here rather than
 * theoretical.
 *
 * **Everything about time is stubbed, and that is the honest answer rather than
 * a gap.** QEMU models neither flash wait states nor the prefetch buffer, and
 * both are a vendor choice rather than a core property — the same binary on two
 * M0+ parts at one clock does not take the same number of cycles. So this
 * machine reports that it has no cycle counter, the harness refuses to run its
 * cycle mode here, and `docs/claim4.md` keeps saying the board is what settles
 * it. Returning a plausible number would be the shape of every instrument fault
 * this project has had.
 *
 * There is deliberately no clock setup either. The claim under test is that the
 * interpreter behaves identically as Thumb, so anything else running on the part
 * is a confound.
 *
 * I/O is ARM semihosting: there is a debugger on the other end of this wire, so
 * the corpus is a file the emulator opens on the host and the trace is another
 * one it writes back. `port/pico/machine.c` is the same contract with no
 * debugger available, and looks nothing like this.
 */

#include <stdint.h>

#include "dm_io.h"
#include "dm_machine.h"
#include "dm_vm.h"
#include "semihost.h"

const char *const dm_machine_name = "qemu-microbit-cortex-m0";

void dm_machine_init(void)
{
}

uint32_t dm_clk_hz(void)
{
    return 0;
}

int dm_cycles_available(void)
{
    return 0;
}

int dm_clock_source_valid(void)
{
    return 0;
}

void dm_cycles_start(void)
{
}

uint32_t dm_cycles_stop(void)
{
    return DM_CYCLES_OVERFLOW;
}

uint64_t dm_us_now(void)
{
    return 0;
}

/* -- I/O ------------------------------------------------------------------ */

static int32_t frames_handle = -1;
static int32_t out_handle = -1;
static uint32_t fuel_setting;
static uint32_t reps_setting;
static int quiet_setting;

static uint32_t str_len(const char *s)
{
    uint32_t n = 0;
    while (s[n]) {
        n++;
    }
    return n;
}

static int32_t open_named(const char *name, uint32_t mode)
{
    return sh_open(name, str_len(name), mode);
}

/* `sh_read` returns bytes *not* transferred, so a short read at EOF is a
 * positive return rather than an error -- the opposite of read(2), and the kind
 * of inversion that reads as success while doing nothing. */
static uint32_t read_exact(int32_t handle, void *buffer, uint32_t length)
{
    const int32_t remaining = sh_read(handle, buffer, length);
    if (remaining < 0) {
        return 0;
    }
    return length - (uint32_t)remaining;
}

/* A small decimal file, or `fallback` when it is absent or empty. Three of the
 * harness's four inputs are one number in a file, which is what a machine with
 * no argv and no environment can be given. */
static uint32_t read_number(const char *name, uint32_t fallback)
{
    const int32_t handle = open_named(name, SH_MODE_RB);
    if (handle < 0) {
        return fallback;
    }
    uint8_t text[16] = {0};
    const uint32_t length = read_exact(handle, text, sizeof text - 1);
    sh_close(handle);

    uint32_t value = 0;
    for (uint32_t i = 0; i < length && text[i] >= '0' && text[i] <= '9'; i++) {
        value = value * 10u + (uint32_t)(text[i] - '0');
    }
    /* Zero means "not given" for every caller here: a zero fuel budget faults
     * before the first instruction and zero repetitions measures nothing, so
     * neither is a request anybody makes on purpose. */
    return value ? value : fallback;
}

void dm_io_begin(void)
{
    /* A file rather than the emulator's `:tt`. Semihosting stdout is the
     * debugger's stdout, and a tool that logs to it would interleave with the
     * trace -- so the differ would be parsing both. */
    out_handle = open_named("trace.txt", SH_MODE_WB);
    if (out_handle < 0) {
        sh_exit(72);
    }
    fuel_setting = read_number("fuel.txt", DM_DEFAULT_FUEL);
    reps_setting = read_number("cycles.txt", 0);

    const int32_t quiet_handle = open_named("quiet.txt", SH_MODE_RB);
    quiet_setting = quiet_handle >= 0;
    if (quiet_setting) {
        sh_close(quiet_handle);
    }
}

uint32_t dm_config_fuel(void)
{
    return fuel_setting;
}

uint32_t dm_config_reps(void)
{
    return reps_setting;
}

int dm_config_quiet(void)
{
    return quiet_setting;
}

uint32_t dm_config_batch(void)
{
    /* The host opened the corpus and is still holding it. There is one batch
     * and nothing to reconcile afterwards. */
    return 0;
}

/* Reopening rather than seeking. `dm_io_again` is zero here so this happens
 * exactly once, and a close/open pair is two semihosting calls against
 * `SYS_SEEK`'s one plus the question of whether the monitor implements it. */
void dm_corpus_rewind(void)
{
    if (frames_handle >= 0) {
        sh_close(frames_handle);
    }
    frames_handle = open_named("frames.bin", SH_MODE_RB);
    if (frames_handle < 0) {
        static const char message[] = "FAILED to open frames.bin\n";
        (void)sh_write(out_handle, message, sizeof message - 1);
        sh_exit(73);
    }
}

uint32_t dm_corpus_next(uint8_t *dst, uint32_t max)
{
    uint8_t header[4];
    if (read_exact(frames_handle, header, sizeof header) != sizeof header) {
        return DM_CORPUS_END; /* clean EOF */
    }
    const uint32_t length = (uint32_t)header[0] | ((uint32_t)header[1] << 8) |
                            ((uint32_t)header[2] << 16) | ((uint32_t)header[3] << 24);
    if (length > max) {
        return length; /* the harness reports it; a truncation would not */
    }
    if (read_exact(frames_handle, dst, length) != length) {
        return max + 1u; /* short read: also over-long, also a fault */
    }
    return length;
}

void dm_io_write(const char *data, uint32_t length)
{
    (void)sh_write(out_handle, data, length);
}

int dm_io_again(void)
{
    /* The host was present for the whole run, so one pass is the run. */
    return 0;
}

void dm_io_end(int status)
{
    if (frames_handle >= 0) {
        sh_close(frames_handle);
    }
    sh_close(out_handle);
    sh_exit(status);
}
