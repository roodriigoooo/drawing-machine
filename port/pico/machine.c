/* Raspberry Pi Pico — RP2040, dual Cortex-M0+, 264 KB SRAM.
 *
 * The machine backend behind `port/bare/dm_machine.h`: bring the part up on a
 * frequency that is *exact*, start the cycle counter and a second timebase, and hand the
 * harness a cycle counter. Nothing else runs here — no second core, no
 * interrupts, no peripherals the measurement does not need — because anything
 * else executing on the part is a confound in a number whose whole point is
 * that it is core-limited.
 *
 * ## Three decisions, each with a reason that outlives it
 *
 * **The interpreter runs from SRAM, not from XIP flash.** That is a property of
 * `port/pico/link.ld` rather than of this file, but it is the reason the number
 * this machine produces means anything: RP2040 executes flash over a QSPI
 * interface with a cache and a vendor-chosen wait-state configuration, so a
 * cycle count taken from XIP measures Raspberry Pi's flash setup as much as the
 * interpreter. `docs/claim4.md` says at length that flash wait states and the
 * prefetch buffer are a vendor choice and not a core property; running from
 * SRAM is how that confound is removed rather than reasoned about. An
 * XIP-resident arm is a *separate* measurement worth having, and it is not this
 * one.
 *
 * **clk_sys is the crystal, undivided and un-multiplied: 12 MHz exactly.** The
 * PLL would give the 125 MHz a Pico normally runs at, and it would also put a
 * VCO, two dividers and a lock loop between the number and the crystal. Cycles
 * per instruction does not depend on frequency at all — a Cortex-M0+ has no
 * cache, no branch prediction and a single-cycle multiplier, so the same code
 * takes the same cycles at any clock — so the *one* thing the clock has to be
 * is known, and 12 MHz from a crystal is known to its ppm rating without a
 * calibration step. Energy per drawing is the number that does care about the
 * operating point, and it is measured at a named clock rather than inferred.
 *
 * **Two timebases, deliberately redundant.** SysTick counts core cycles; the
 * TIMER block counts microseconds from the same crystal through a separate
 * divider. Their ratio is useful as a secondary consistency check, but it is
 * not source proof because both could follow the same wrong mux selection. The
 * harness separately requires `dm_clock_source_valid()` before measuring.
 */

#include <stdint.h>

#include "dm_io.h"
#include "dm_machine.h"
#include "dm_vm.h"
#include "rp2040.h"

const char *const dm_machine_name = "rp2040-cortex-m0plus-sram";

/* ## Bring-up bisection, and why this part needs one
 *
 * Every other machine in `port/` can say what went wrong: the native harness
 * writes to stderr and the QEMU one has semihosting. This one has a UART pin and
 * nothing else, so a firmware that dies before `uart_start()` — anywhere in the
 * clock bring-up, or in a fault handler that cannot report — is indistinguishable
 * from a disconnected wire. Both are silence, and the two have opposite repairs.
 *
 * There is exactly one other observable: **the board can reboot itself.**
 * `reset_usb_boot` lives in the bootrom, needs no clock this file configured and
 * no peripheral out of reset, so `RPI-RP2` reappearing on the host is a message
 * from the firmware saying "I reached here." Setting `DM_BRINGUP_STAGE` to N
 * stops after the Nth step and sends that message; the first N that fails to
 * come back is the step that hangs.
 *
 * Zero — the default — compiles every checkpoint out, so the measured image is
 * byte-identical to one built without this. `tests/test_pico.py` asserts that.
 */
#ifndef DM_BRINGUP_STAGE
#define DM_BRINGUP_STAGE 0
#endif

static void reboot_to_bootsel(void);

#if DM_BRINGUP_STAGE
static void bringup_reached(unsigned stage)
{
    if (stage == (unsigned)DM_BRINGUP_STAGE) {
        reboot_to_bootsel();
        for (;;) {
        }
    }
}
#define DM_BRINGUP(stage) bringup_reached(stage)
#else
#define DM_BRINGUP(stage) ((void)0)
#endif

#if DM_BRINGUP_STAGE
/* The shared harness uses this only for its optional buffer microscope. Keep
 * the reboot primitive in the Pico backend so the normal harness remains
 * machine-neutral when no diagnostic stage is compiled. */
void dm_bringup_stage(unsigned stage)
{
    bringup_reached(stage);
}
#endif

/* The crystal on every Pico-form board. The selected-source readback below is
 * the direct hardware assertion; this constant supplies the UART divisor and
 * the expected cycle frequency in the protocol. */
#define XOSC_HZ 12000000u

/* §2.16.3: the startup delay counts in units of 256 crystal cycles and must
 * cover the crystal's own start-up time. 12 MHz needs ~1 ms, which is 47; this
 * is five times that, because the cost of being generous is 5 ms once per run
 * and the cost of being tight is an intermittent hang. */
#define XOSC_STARTUP_DELAY 256u

static void xosc_start(void)
{
    XOSC_CTRL = XOSC_CTRL_FREQ_RANGE_1_15MHZ;
    XOSC_STARTUP = XOSC_STARTUP_DELAY;
    XOSC_CTRL_SET = XOSC_CTRL_ENABLE;
    while (!(XOSC_STATUS & XOSC_STATUS_STABLE)) {
    }
}

/* The order matters and is the datasheet's (§2.15.3.2). clk_sys is moved onto
 * the glitchless reference *first*, so that switching the reference underneath
 * it carries the core across without ever selecting an aux source; then the
 * reference moves from the ring oscillator to the crystal. Doing it the other
 * way round runs the core off an unswitched mux for the duration.
 *
 * Both waits are on `SELECTED`, which is the hardware acknowledging the switch.
 * A glitchless mux does not change until its current source is running, so a
 * write with no wait is a request that can silently not have happened. */
static void clocks_to_xosc(void)
{
    CLK_SYS_CTRL = CLK_SYS_SRC_REF;
    while (!(CLK_SYS_SELECTED & (1u << CLK_SYS_SRC_REF))) {
    }
    CLK_REF_CTRL = CLK_REF_SRC_XOSC;
    while (!(CLK_REF_SELECTED & (1u << CLK_REF_SRC_XOSC))) {
    }
    CLK_REF_DIV = CLK_DIV_1;
    CLK_SYS_DIV = CLK_DIV_1;
}

/* The microsecond timer is fed by the watchdog's tick generator, which divides
 * clk_ref. With clk_ref at the crystal, one tick per 12 reference cycles is one
 * microsecond exactly — no rounding, which is why the crystal is the reference
 * rather than the ring oscillator. */
static void timer_start(void)
{
    RESETS_RESET_CLR = RESETS_BIT_TIMER;
    while (!(RESETS_RESET_DONE & RESETS_BIT_TIMER)) {
    }
    WATCHDOG_TICK = (XOSC_HZ / 1000000u) | WATCHDOG_TICK_ENABLE;
}

/* Free-running, no interrupt, clocked by the processor. The reload is the full
 * 24 bits so a single measurement has the longest span the counter can give:
 * 16,777,215 cycles, which is 1.4 s at 12 MHz against a drawing's few thousand.
 */
static void systick_start(void)
{
    SYST_CSR = 0;
    SYST_RVR = SYST_MASK;
    SYST_CVR = 0;
    SYST_CSR = SYST_CSR_CLKSOURCE | SYST_CSR_ENABLE;
}

void dm_machine_init(void)
{
    /* 1 proves the bootrom launched the image, the vector table and SRAM link
     * are right, and `startup.c` reached C — before anything here touches a
     * clock. Everything after it is one register block each. */
    DM_BRINGUP(1);
    xosc_start();
    DM_BRINGUP(2);
    clocks_to_xosc();
    DM_BRINGUP(3);
    timer_start();
    DM_BRINGUP(4);
    systick_start();
    DM_BRINGUP(5);
}

uint32_t dm_clk_hz(void)
{
    return XOSC_HZ;
}

int dm_cycles_available(void)
{
    return 1;
}

int dm_clock_source_valid(void)
{
    /* This is a direct assertion of the mux state, not a second measurement
     * derived from the same clock. `SELECTED` is the hardware acknowledgement
     * for each glitchless mux; the control and divider readbacks close the
     * remaining ways to report a nominal 12 MHz while running elsewhere. */
    return (XOSC_STATUS & XOSC_STATUS_STABLE) != 0u &&
           (CLK_REF_CTRL & CLK_REF_SRC_MASK) == CLK_REF_SRC_XOSC &&
           CLK_REF_SELECTED == (1u << CLK_REF_SRC_XOSC) &&
           (CLK_SYS_CTRL & CLK_SYS_SRC_MASK) == CLK_SYS_SRC_REF &&
           CLK_SYS_SELECTED == (1u << CLK_SYS_SRC_REF) &&
           CLK_REF_DIV == CLK_DIV_1 && CLK_SYS_DIV == CLK_DIV_1;
}

static uint32_t cycles_at_start;

void dm_cycles_start(void)
{
    (void)SYST_CSR; /* reading CSR clears COUNTFLAG, so the span starts clean */
    cycles_at_start = SYST_CVR;
}

uint32_t dm_cycles_stop(void)
{
    const uint32_t now = SYST_CVR;
    const uint32_t csr = SYST_CSR;

    /* SysTick counts *down*, so elapsed is start minus now. COUNTFLAG says the
     * counter passed zero at least once since it was last read, and a wrapped
     * 24-bit difference is indistinguishable from a small one — so this reports
     * the wrap instead of returning a number that looks like a measurement.
     * The harness turns that into a failure rather than a data point. */
    if (csr & SYST_CSR_COUNTFLAG) {
        return DM_CYCLES_OVERFLOW;
    }
    return (cycles_at_start - now) & SYST_MASK;
}

uint64_t dm_us_now(void)
{
    /* §4.6.2: reading `TIMELR` latches `TIMEHR`, so this order is the whole
     * atomicity guarantee and reversing it is a real bug once every 71 minutes.
     */
    const uint32_t low = TIMER_LR;
    const uint32_t high = TIMER_HR;
    return ((uint64_t)high << 32) | low;
}

/* -- I/O: corpus already in SRAM, trace out a pin -------------------------- */

/* ## Why there is no channel inward
 *
 * A Pico with one USB cable and no debugger has no way for a host to hand it
 * anything at run time. The bootrom's USB interface exists only in BOOTSEL mode,
 * and the moment the image starts, that is gone.
 *
 * So the corpus does not arrive: **it is already here.** `scripts/uf2.py` packs
 * the programs into the same UF2 as the image, addressed at `_corpus_base`,
 * and the bootrom's drag-and-drop loader writes both into SRAM before the core
 * executes an instruction. Configuration that a semihosted machine reads from
 * `fuel.txt` and `cycles.txt` rides in the header at the same address.
 *
 * The consequence that shapes everything else: **this firmware cannot know when
 * a host attached to its UART.** So it runs the corpus forever, and the harness
 * brackets each pass with `BEGIN`/`END` so a reader can discard the partial
 * pass it walked in on and keep the next whole one.
 */

/* A pointer into SRAM, supplied by `pico/link.ld`, which also asserts that the
 * image's own `.bss` ends below it. */
extern uint8_t _corpus_base;

/* 'D','C','M','0'. A magic number rather than a length check, because an image
 * flashed without a corpus finds whatever the last run left in SRAM -- which is
 * a *valid* corpus, and would be silently measured. */
#define DM_CORPUS_MAGIC 0x304D4344u
#define DM_CORPUS_VERSION 1u
#define DM_CORPUS_HEADER_WORDS 9u

enum {
    HDR_MAGIC = 0,
    HDR_VERSION = 1,
    HDR_FUEL = 2,
    HDR_REPS = 3,
    HDR_QUIET = 4,
    HDR_N_PROGRAMS = 5,
    HDR_BYTES = 6,
    HDR_PASSES = 7, /* 0 runs forever; N reboots to BOOTSEL after N passes */
    HDR_BATCH = 8,
};

static const uint32_t *header;
static const uint8_t *payload;
static uint32_t payload_bytes;
static uint32_t cursor;
static uint32_t passes_left;

/* 115200 rather than the 750,000 this clock could reach exactly. The output is
 * read by whatever USB-serial adapter the user has, and 115200 is the one rate
 * every driver supports; `termios` on macOS does not even name a constant for
 * the faster one. The measured spans exclude the UART entirely, so the rate
 * costs wall-clock time and nothing else. */
#ifndef DM_UART_BAUD
#define DM_UART_BAUD 115200u
#endif

/* §4.2.7.1's divisor, folded at compile time because both terms are constants
 * here -- which also keeps `__aeabi_uidiv` out of bring-up. */
#define DM_UART_DIV ((8u * XOSC_HZ + DM_UART_BAUD / 2u) / DM_UART_BAUD)

/* A fault before UART setup must not touch UART registers that are still in
 * reset. This is safe in the diagnostic build because the corrected startup
 * path clears BSS before any C code can fault. */
static volatile int uart_ready;

static void uart_start(void)
{
    /* **clk_peri is enabled first, and the order is the whole function.**
     *
     * `RESET_DONE` for UART0 does not assert until the block has a clock, so
     * releasing its reset and enabling clk_peri afterwards spins forever on a
     * handshake that cannot complete — measured 2026-08-15, on this board, as a
     * silent hang with no output and no reboot. The pico-sdk never meets this
     * because `clocks_init()` runs long before any peripheral touches `RESETS`;
     * reading the SDK therefore does not tell you the ordering is load-bearing,
     * and nothing in QEMU models a reset that waits on a clock.
     *
     * Disabled and then enabled rather than written once, because clk_peri's mux
     * is **not** glitchless: changing `AUXSRC` while the clock is running can
     * stop it. AUXSRC 0 is clk_sys, i.e. the crystal.
     */
    CLK_PERI_CTRL = 0;
    CLK_PERI_CTRL = CLK_PERI_CTRL_ENABLE;

    RESETS_RESET_CLR = RESETS_BIT_UART0 | RESETS_BIT_IO_BANK0 | RESETS_BIT_PADS_BANK0;
    while ((RESETS_RESET_DONE & (RESETS_BIT_UART0 | RESETS_BIT_IO_BANK0 |
                                 RESETS_BIT_PADS_BANK0)) !=
           (RESETS_BIT_UART0 | RESETS_BIT_IO_BANK0 | RESETS_BIT_PADS_BANK0)) {
    }
    DM_BRINGUP(6);

    UART0_CR = 0;
    UART0_IBRD = DM_UART_DIV >> 7;
    UART0_FBRD = ((DM_UART_DIV & 0x7fu) + 1u) >> 1;
    /* Writing LCR_H is what latches IBRD/FBRD, so it must come after them. */
    UART0_LCR_H = UART0_LCR_H_8BIT | UART0_LCR_H_FIFO;
    UART0_CR = UART0_CR_UARTEN | UART0_CR_TXE;

    /* GPIO0 becomes UART0 TX. The pad wants its input buffer enabled and its
     * output disable cleared; the function select is what actually connects it. */
    PADS_BANK0_GPIO0 = (PADS_BANK0_GPIO0 | PADS_BANK0_IE) & ~PADS_BANK0_OD;
    IO_BANK0_GPIO0_CTRL = IO_BANK0_FUNC_UART;
    uart_ready = 1;
    DM_BRINGUP(7);
}

static void uart_put(char c)
{
    while (UART0_FR & UART0_FR_TXFF) {
    }
    UART0_DR = (uint32_t)(uint8_t)c;
}

static void uart_puts(const char *s)
{
    while (*s) {
        uart_put(*s++);
    }
}

static void halt_forever(const char *why)
{
    for (;;) {
        uart_puts(why);
        uart_put('\n');
    }
}

/* -- what a fault says, on a part with no debugger ------------------------- */

/* `port/qemu/startup.c` reports a fault over semihosting, which is right there
 * and catastrophic here: semihosting is a `BKPT`, and on ARMv6-M with no
 * debugger attached `BKPT` escalates to HardFault. So the fault handler faults,
 * and the board locks up in silence — every bug downgraded to "it hung", which
 * is the least useful thing a measurement instrument can say.
 *
 * This reports over the UART instead, and reports the **stacked PC**, because
 * "something faulted" and "it faulted at 0x2000_0d3a" are different amounts of
 * information: the second one is `llvm-objdump -d` away from a source line.
 *
 * The first byte is emitted before dereferencing the exception frame. Stage 17
 * deliberately faults after `uart_start()`, so this makes exception entry
 * observable even if the frame pointer or its contents are the next problem.
 * The handler drains the UART and reboots only in that diagnostic stage; an
 * ordinary fault stays halted with its evidence on the wire.
 *
 * Called from `startup.c`'s naked handler with the exception frame, which
 * ARMv6-M pushes as {R0,R1,R2,R3,R12,LR,PC,xPSR}.
 */
void dm_pico_fault(const uint32_t *frame)
{
    static const char digits[] = "0123456789abcdef";

    if (!uart_ready) {
        for (;;) {
        }
    }

    /* Stage 17 reaches this handler only after UART setup. Keep this byte
     * before any frame access: it proves the vector and exception entry path
     * independently of the C-side frame decode. */
    uart_put('!');
    uart_put('\n');
    uart_puts("FAULT pc=0x");
    for (int shift = 28; shift >= 0; shift -= 4) {
        uart_put(digits[(frame[6] >> shift) & 0xfu]);
    }
    uart_puts(" lr=0x");
    for (int shift = 28; shift >= 0; shift -= 4) {
        uart_put(digits[(frame[5] >> shift) & 0xfu]);
    }
    uart_put('\n');
    while (UART0_FR & UART0_FR_BUSY) {
    }
#if DM_BRINGUP_STAGE == 17
    reboot_to_bootsel();
#endif
    for (;;) {
    }
}

#if DM_BRINGUP_STAGE == 17
/* Keep the deliberate fault in a named, no-prologue function. The host reads
 * this symbol from the stage ELF and compares it with the stacked PC, so a
 * plausible FAULT line from some unrelated exception cannot pass the test. */
__attribute__((naked, noinline, used)) void dm_stage17_udf(void)
{
    __asm__ volatile(".short 0xde00");
}
#endif

#if DM_BRINGUP_STAGE == 9
/* The only checkpoint that tests a fact facing the *wire*.
 *
 * Stages 1-8 prove registers were written and a reset handshake completed. None
 * of them proves a bit reached the pin, and the difference is the whole
 * remaining question: a PL011 with a stopped clock accepts every write, fills
 * its 32-byte FIFO and stalls, which from the host is byte-for-byte the same
 * observation as a dupont wire in the wrong socket.
 *
 * So this one puts a byte in the transmitter and reports back **only if the
 * transmitter drained it**. Reaching 9 puts the fault outside the chip; failing
 * to reach it keeps the fault inside. Re-seating connectors cannot separate
 * those two and this can.
 */
static void bringup_transmit(void)
{
    uint32_t spins = 0;

    uart_put('U'); /* 0x55 -- alternating bits, the worst case for a bad divisor */
    while (UART0_FR & UART0_FR_BUSY) {
        if (++spins > 8000000u) {
            for (;;) { /* never drained; must not reboot and report success */
            }
        }
    }
}
#endif

#if DM_BRINGUP_STAGE == 10 || DM_BRINGUP_STAGE == 11
/* Stages 10 and 11 cross the FIFO boundary, which stage 9 never did.
 *
 * Stage 9 sent one byte and passed; the smoke corpus sends about fifty and
 * hangs, and the transmit FIFO is 32 deep. So the interesting region is not "can
 * this UART send a byte" but "does the FIFO drain" — and the two have different
 * causes. `uart_put` spins on `TXFF`, so a FIFO that fills and never empties is
 * a lock-up with no output, which is the observation.
 *
 * Bounded, unlike `uart_put`: a checkpoint that hangs reports nothing, and the
 * whole point is to distinguish where it stopped.
 *
 * 11 repeats it with the FIFO **disabled**, so the transmitter is exercised one
 * byte at a time through the holding register instead. 10 failing and 11 passing
 * isolates the fault to the FIFO configuration; both failing puts it in the
 * transmitter or its clock.
 */
static void bringup_sustain(int with_fifo)
{
    if (!with_fifo) {
        UART0_CR = 0;
        UART0_LCR_H = UART0_LCR_H_8BIT; /* no FEN */
        UART0_CR = UART0_CR_UARTEN | UART0_CR_TXE;
    }
    for (uint32_t n = 0; n < 64u; n++) {
        uint32_t spins = 0;
        while (UART0_FR & UART0_FR_TXFF) {
            if (++spins > 8000000u) {
                for (;;) { /* the FIFO never drained */
                }
            }
        }
        UART0_DR = (uint32_t)(uint8_t)('0' + (n % 10u));
    }
    /* Drain before rebooting, or `reset_usb_boot` truncates the evidence. */
    uint32_t spins = 0;
    while (UART0_FR & UART0_FR_BUSY) {
        if (++spins > 8000000u) {
            for (;;) {
            }
        }
    }
}
#endif

void dm_io_begin(void)
{
    uart_start();
#if DM_BRINGUP_STAGE == 9
    bringup_transmit();
#endif
#if DM_BRINGUP_STAGE == 17
    /* Stage 17 tests the *instrument*, not the port.
     *
     * Everything from stage 12 on has been read as "it hung", on the grounds
     * that no `FAULT pc=` line appeared. That inference is only sound if the
     * fault path works — and on ARMv6-M a fault taken inside the HardFault
     * handler, or an exception whose own register push faults, puts the core in
     * LOCKUP with no output at all. A silent fault and a hang are then the same
     * observation, and the whole bisection has been reading one as the other.
     *
     * `UDF #0` is the architecturally-defined undefined instruction, so this is
     * a guaranteed HardFault at a known PC. If `dm_pico_fault` prints, silence
     * elsewhere really does mean a hang. If it does not, every "hang" in this
     * log is suspect and the fault path is the bug.
     */
    dm_stage17_udf();
#endif
    DM_BRINGUP(17);
    DM_BRINGUP(9);
#if DM_BRINGUP_STAGE == 10
    bringup_sustain(1);
#endif
    DM_BRINGUP(10);
#if DM_BRINGUP_STAGE == 11
    bringup_sustain(0);
#endif
    DM_BRINGUP(11);

    header = (const uint32_t *)(const void *)&_corpus_base;
    if (header[HDR_MAGIC] != DM_CORPUS_MAGIC) {
        halt_forever("FAILED: no corpus at _corpus_base -- pack one with scripts/uf2.py");
    }
    if (header[HDR_VERSION] != DM_CORPUS_VERSION) {
        halt_forever("FAILED: corpus header version is not the one this image reads");
    }
    payload = &_corpus_base + DM_CORPUS_HEADER_WORDS * 4u;
    payload_bytes = header[HDR_BYTES];
    cursor = 0;
    passes_left = header[HDR_PASSES];
    /* 8 says the UF2 landed where `link.ld` reserved it and the header survived
     * the trip — the packer and the linker agreeing on silicon rather than in a
     * test. Past this only the interpreter itself is left. */
    DM_BRINGUP(8);
}

/* §2.8.3.1.3 — the bootrom's public functions, reached through a table rather
 * than at fixed addresses, because the addresses are not stable across bootrom
 * revisions and a hard-coded one is a jump into whatever the next revision put
 * there.
 *
 * The one that matters is `reset_usb_boot`, and it is what makes a multi-batch
 * sweep bearable: the corpus region holds 180 KB, the full conformance corpus is
 * several times that, and without this every batch would need someone to hold
 * the BOOTSEL button and replug the cable. With it, the board returns to BOOTSEL
 * by itself and the host writes the next batch — one press for a whole sweep.
 */
typedef void *(*rom_lookup_fn)(uint16_t *table, uint32_t code);
typedef void (*rom_reset_usb_boot_fn)(uint32_t gpio_mask, uint32_t disable_mask);

#define ROM_FUNC_TABLE (*(uint16_t *)0x00000014u)
#define ROM_TABLE_LOOKUP (*(uint16_t *)0x00000018u)
#define ROM_CODE(a, b) ((uint32_t)(a) | ((uint32_t)(b) << 8))

static void reboot_to_bootsel(void)
{
    const rom_lookup_fn lookup = (rom_lookup_fn)(uintptr_t)ROM_TABLE_LOOKUP;
    uint16_t *const table = (uint16_t *)(uintptr_t)ROM_FUNC_TABLE;
    const rom_reset_usb_boot_fn reset_usb_boot =
        (rom_reset_usb_boot_fn)lookup(table, ROM_CODE('U', 'B'));

    if (reset_usb_boot) {
        reset_usb_boot(0, 0); /* no activity LED, both interfaces enabled */
    }
}

uint32_t dm_config_fuel(void)
{
    return header[HDR_FUEL] ? header[HDR_FUEL] : DM_DEFAULT_FUEL;
}

uint32_t dm_config_reps(void)
{
    return header[HDR_REPS];
}

int dm_config_quiet(void)
{
    return header[HDR_QUIET] != 0;
}

uint32_t dm_config_batch(void)
{
    /* 13 is the harness calling back immediately after `puts_("BEGIN ")`, so
     * reaching it means those six `put`s completed. */
    DM_BRINGUP(13);
    /* Detailed stage-13 microscope: all six `puts_("BEGIN ")` writes have
     * completed, and the backend can now reboot without entering the rest of
     * the pass. Stage 13 remains the historical checkpoint for compatibility. */
    DM_BRINGUP(22);
    /* Echoed in every `BEGIN` line. The corpus region is smaller than a full
     * conformance sweep, so a sweep is several UF2s in sequence; matching the
     * captured passes to their references by *position* would silently diff one
     * batch against another's programs the first time one was retried. */
    return header[HDR_BATCH];
}

void dm_corpus_rewind(void)
{
    cursor = 0;
    /* 12 is the first thing `main`'s pass loop calls, and it is the only hook
     * this machine has inside the shared harness. Everything below 12 is
     * `machine.c`'s own bring-up; reaching 12 means `dm_io_begin` returned and
     * the harness started a pass, which no earlier checkpoint can show because
     * they all reboot before returning. */
    DM_BRINGUP(12);
}

uint32_t dm_corpus_next(uint8_t *dst, uint32_t max)
{
    /* 15 is past `trace_corpus`'s header — `frac_bits`/`curve_steps` and the
     * `put_i32` that formats them — and just before the interpreter runs. */
    DM_BRINGUP(15);
    if (cursor + 4u > payload_bytes) {
        return DM_CORPUS_END;
    }
    const uint32_t length = (uint32_t)payload[cursor] |
                            ((uint32_t)payload[cursor + 1u] << 8) |
                            ((uint32_t)payload[cursor + 2u] << 16) |
                            ((uint32_t)payload[cursor + 3u] << 24);
    cursor += 4u;
    if (length > max || cursor + length > payload_bytes) {
        return max + 1u; /* over-long or truncated; the harness reports it */
    }
    for (uint32_t i = 0; i < length; i++) {
        dst[i] = payload[cursor + i];
    }
    cursor += length;
    return length;
}

void dm_io_write(const char *data, uint32_t length)
{
    /* 14 is the harness's first `flush`. At `--out-max 1` that lands on the
     * second `put` of `"BEGIN "`, so it is the earliest possible proof that the
     * buffered output path — not the raw transmitter — carries a byte. */
    DM_BRINGUP(14);
    for (uint32_t i = 0; i < length; i++) {
        uart_put(data[i]);
    }
}

int dm_io_again(void)
{
    /* 16 is a whole pass: corpus walked, every program interpreted, the trace
     * formatted and flushed. Past it only `dm_io_end` remains. */
    DM_BRINGUP(16);
    /* Zero passes means forever. See the note above `_corpus_base`: with no
     * channel inward, a firmware that stops after one pass is only readable by
     * a host that was already listening — which a scripted run is, and a person
     * with a terminal is not. */
    if (passes_left == 0u) {
        return 1;
    }
    passes_left--;
    return passes_left != 0u;
}

void dm_io_end(int status)
{
    /* The status is not printed, and not because it is unimportant: the harness
     * has already written `FAILED: <reason>` through `dm_io_write`, which says
     * more than a number would. Formatting it here would also pull
     * `__aeabi_idiv` into the image for one diagnostic -- ARMv6-M has no divide
     * instruction, so `status / 10` is a function call. */
    while (UART0_FR & UART0_FR_BUSY) {
    }
    /* A clean end to a finite run hands the board back to the bootrom, so the
     * host can write the next batch without anyone touching the button.
     *
     * A *failure* deliberately does not: rebooting into BOOTSEL after a fault
     * would erase the state that caused it and hand the next batch to a board
     * that just failed, which is how a sweep reports a pass it did not get. It
     * stops instead, quiet, with its reason already on the wire. */
    if (status == 0 && header[HDR_PASSES] != 0u) {
        reboot_to_bootsel();
    }
    for (;;) {
    }
}
