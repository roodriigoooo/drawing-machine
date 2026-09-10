/* The conformance harness, as bare-metal firmware — one copy, every machine.
 *
 * Same job as `host/dm_trace.c` and deliberately the same output, byte for
 * byte, so `scripts/conformance.py` diffs a *Thumb* run against
 * `dm/vm/interp.py` using the code path it already had. What changes between
 * machines is where the corpus comes from and where the trace goes, which is
 * `dm_io.h`, and how the part is brought up, which is `dm_machine.h`.
 *
 * **There is exactly one of these on purpose.** Claim 4's value is that the same
 * differ compares the same protocol whether the trace came from an emulator or
 * from silicon; a second harness would let the two drift and then agree about
 * different things. `port/qemu/machine.c` and `port/pico/machine.c` are the only
 * files that differ, and neither of them can see the line protocol.
 *
 * There is no libc here. A printf would drag in several kilobytes and its own
 * integer formatting, and then a divergence could live in the harness rather
 * than the interpreter. Everything below `dm_vm_run` is ~60 lines that can be
 * read in one sitting.
 */

#include <stdint.h>

#include "dm_io.h"
#include "dm_machine.h"
#include "dm_vm.h"

/* One program at a time, never the whole corpus. Not a nicety: the smallest
 * machine here has 16 KB of SRAM and the corpora run to hundreds of KB. Holding
 * one program is also what a device would do, which keeps the firmware's memory
 * profile honest against the footprint claim.
 *
 * `max_len` in the training configs is 3072, so 4 KB admits every program the
 * project has and diagnoses anything longer instead of truncating it. */
#ifndef DM_PROGRAM_MAX
#define DM_PROGRAM_MAX 4096u
#endif

/* How much output to accumulate before handing it to `dm_io_write`. On a
 * semihosted machine every write is a `bkpt` the debugger traps; on a UART it
 * is bytes down a pin. The machine's Makefile sets this, and the default is the
 * small one because a machine that forgets to choose should not silently
 * allocate. */
#ifndef DM_OUT_MAX
#define DM_OUT_MAX 1024u
#endif

static uint8_t program[DM_PROGRAM_MAX];

static char out[DM_OUT_MAX];
static uint32_t out_len;

#if defined(DM_BRINGUP_STAGE) && DM_BRINGUP_STAGE != 0
extern void dm_bringup_stage(unsigned stage);
#define DM_HARNESS_BRINGUP(stage) dm_bringup_stage(stage)
#else
#define DM_HARNESS_BRINGUP(stage) ((void)0)
#endif

#ifdef DM_BSS_PREFILL
/* `port/pico/startup.c` fills the complete BSS span with a nonzero pattern
 * before its real clear in this diagnostic build. Scan the whole linker-defined
 * range: checking one convenient global can certify a partial clear by
 * accident. The caller keeps the reduction in a volatile automatic so the
 * result survives machine/UART initialisation on the stack. */
extern uint32_t _sbss, _ebss;

static uint32_t bss_nonzero(void)
{
    const volatile uint32_t *cursor = &_sbss;
    const volatile uint32_t *const end = &_ebss;
    uint32_t nonzero = 0;
    while (cursor < end) {
        nonzero |= *cursor++;
    }
    return nonzero;
}
#endif

static const char *const FAULT_NAME[] = {
    [DM_FAULT_UNKNOWN_OPCODE] = "unknown_opcode",
    [DM_FAULT_TRUNCATED] = "truncated",
    [DM_FAULT_UNMATCHED_ENDREP] = "unmatched_endrep",
    [DM_FAULT_UNTERMINATED_REPEAT] = "unterminated_repeat",
    [DM_FAULT_DEPTH_OVERFLOW] = "depth_overflow",
    [DM_FAULT_ZERO_REPEAT] = "zero_repeat",
    [DM_FAULT_OUT_OF_FUEL] = "out_of_fuel",
    [DM_FAULT_NO_HALT] = "no_halt",
    [DM_FAULT_CALL_UNSUPPORTED] = "call_unsupported",
};

static const char *const PATH_NAME[] = {
    [DM_PATH_DISCARDED] = "discarded",
    [DM_PATH_STROKE] = "stroke",
    [DM_PATH_REGION] = "region",
};

/* -- output, buffered so the channel is not touched per character ---------- */

static void flush(void)
{
    if (out_len) {
        dm_io_write(out, out_len);
        out_len = 0;
    }
}

static void put(char c)
{
    if (out_len == DM_OUT_MAX) {
        flush();
    }
    out[out_len++] = c;
}

static void puts_(const char *s)
{
    while (*s) {
        put(*s++);
    }
}

/* u / 10 and u %% 10 without a divide instruction: ARMv6-M has `MULS` and no
 * divider, so the reciprocal is built from shifts and the remainder recovered
 * with one multiply. Cheaper than calling `__aeabi_uidiv` 10 times per number,
 * and it keeps the formatter off the division helper altogether. */
static uint32_t divmod10(uint32_t u, uint32_t *remainder)
{
    uint32_t q = (u >> 1) + (u >> 2);
    q += q >> 4;
    q += q >> 8;
    q += q >> 16;
    q >>= 3;
    uint32_t r = u - q * 10u;
    if (r >= 10u) {
        q++;
        r -= 10u;
    }
    *remainder = r;
    return q;
}

static void put_u32(uint32_t u)
{
    char digits[12];
    uint32_t n = 0;

    do {
        uint32_t digit;
        u = divmod10(u, &digit);
        digits[n++] = (char)('0' + digit);
    } while (u);
    while (n) {
        put(digits[--n]);
    }
}

static void put_i32(int32_t v)
{
    uint32_t u = (uint32_t)v;

    if (v < 0) {
        put('-');
        u = ~u + 1u; /* two's complement negate; -INT32_MIN is UB as an int */
    }
    put_u32(u);
}

/* Nothing here formats a 64-bit value, and that is a decision rather than an
 * omission: `uint64_t / 10` on ARMv6-M is a call into `__aeabi_uldivmod`, a
 * helper the interpreter does not need and the runtime does not carry. The
 * microsecond timebase is 64-bit at its source and is narrowed to an elapsed
 * `uint32_t` here, which covers 71 minutes; summing per-program cycles is left
 * to the host, where arithmetic is free. */

/* -- sink ----------------------------------------------------------------- */

static void on_path_begin(void *ctx)
{
    (void)ctx;
    puts_("b\n");
}

static void on_path_point(void *ctx, dm_fx x, dm_fx y)
{
    (void)ctx;
    put('p');
    put(' ');
    put_i32(x);
    put(' ');
    put_i32(y);
    put('\n');
}

static void on_path_end(void *ctx, dm_path_kind kind, uint32_t n, uint8_t width)
{
    (void)ctx;
    puts_("e ");
    puts_(PATH_NAME[kind]);
    put(' ');
    put_i32((int32_t)n);
    put(' ');
    put_i32(width);
    put('\n');
}

static void on_disc(void *ctx, dm_fx cx, dm_fx cy, uint8_t radius, uint8_t width)
{
    (void)ctx;
    put('d');
    put(' ');
    put_i32(cx);
    put(' ');
    put_i32(cy);
    put(' ');
    put_i32(radius);
    put(' ');
    put_i32(width);
    put('\n');
}

static void on_fault(void *ctx, dm_fault_kind kind, uint32_t pc)
{
    (void)ctx;
    put('f');
    put(' ');
    puts_(FAULT_NAME[kind]);
    put(' ');
    put_i32((int32_t)pc);
    put('\n');
}

static const dm_sink REPORTING = {
    .ctx = 0,
    .path_begin = on_path_begin,
    .path_point = on_path_point,
    .path_end = on_path_end,
    .disc = on_disc,
    .fault = on_fault,
};

/* `dm_vm.h` permits every callback to be NULL, in which case that class of
 * output is discarded and the interpreter still runs. Both a cycle count and an
 * instruction count taken with the reporting sink attached would mostly measure
 * this file's decimal formatter and the channel under it, so every measurement
 * uses this one -- and running it also exercises the all-NULL contract, which
 * nothing else does. */
static const dm_sink SILENT = {0};

static void die(const char *why, int status)
{
    puts_("FAILED: ");
    puts_(why);
    put('\n');
    flush();
    dm_io_end(status);
}

static uint32_t next_program(void)
{
    const uint32_t length = dm_corpus_next(program, DM_PROGRAM_MAX);
    if (length != DM_CORPUS_END && length > DM_PROGRAM_MAX) {
        die("a program is longer than the buffer", 74);
    }
    return length;
}

/* -- the two modes -------------------------------------------------------- */

static uint32_t trace_corpus(uint32_t fuel, int quiet)
{
    const dm_sink *const sink = quiet ? &SILENT : &REPORTING;

    if (!quiet) {
        puts_("frac_bits ");
        put_i32(DM_FRAC_BITS);
        puts_("\ncurve_steps ");
        put_i32(DM_CURVE_STEPS);
        put('\n');
    }

    uint32_t index = 0;
    for (;; index++) {
        const uint32_t length = next_program();
        if (length == DM_CORPUS_END) {
            return index;
        }
        if (!quiet) {
            put('#');
            put_u32(index);
            put('\n');
        }
        const dm_result result = dm_vm_run(program, length, sink, fuel);
        if (!quiet) {
            puts_("= ");
            put_u32(result.steps);
            put(' ');
            put_i32(result.halted);
            put(' ');
            put_u32(result.n_faults);
            put('\n');
        }
    }
}

/* Claim 4's fourth number. Reported per program rather than aggregated, because
 * `docs/claim4.md` already establishes that the work is proportional to what a
 * program *expands to* and not to its length -- an aggregate cannot separate
 * those, and the corpora are chosen to make them disagree.
 *
 * The statistic over repetitions is the **minimum**, not the mean. A Cortex-M0+
 * has no cache and no branch prediction, so the same program takes the same
 * number of cycles every time and anything above the minimum is interference
 * from outside the measurement. A mean would fold that in and call it cost.
 */
static uint32_t cycle_corpus(uint32_t fuel, uint32_t reps)
{
    if (!dm_cycles_available()) {
        die("this machine has no cycle counter", 75);
    }
    if (!dm_clock_source_valid()) {
        die("clock source is not verified as XOSC", 76);
    }

    /* The instrument's own cost, measured the same way it measures, so the host
     * subtracts a number rather than an assumption. */
    uint32_t overhead = DM_CYCLES_OVERFLOW;
    for (uint32_t r = 0; r < reps; r++) {
        dm_cycles_start();
        const uint32_t empty = dm_cycles_stop();
        if (empty < overhead) {
            overhead = empty;
        }
    }

    puts_("clock_source xosc\n");
    puts_("machine ");
    puts_(dm_machine_name);
    puts_("\nclk_hz ");
    put_u32(dm_clk_hz());
    puts_("\nreps ");
    put_u32(reps);
    puts_("\noverhead ");
    put_u32(overhead);
    put('\n');

    /* Accumulated across the *measured* spans only, never across the output in
     * between. The point of this number is to recover the core frequency from a
     * second, independent timebase, and a span that included a UART draining at
     * 115200 baud would recover something four orders of magnitude too low and
     * make the check vacuous. */
    uint64_t measured_us = 0;
    uint32_t index = 0;

    for (;; index++) {
        const uint32_t length = next_program();
        if (length == DM_CORPUS_END) {
            break;
        }
        uint32_t best = DM_CYCLES_OVERFLOW;
        uint32_t steps = 0;

        const uint64_t started = dm_us_now();
        for (uint32_t r = 0; r < reps; r++) {
            dm_cycles_start();
            const dm_result result = dm_vm_run(program, length, &SILENT, fuel);
            const uint32_t cycles = dm_cycles_stop();
            steps = result.steps;
            if (cycles < best) {
                best = cycles;
            }
        }
        measured_us += dm_us_now() - started;

        put('c');
        put(' ');
        put_u32(index);
        put(' ');
        put_u32(best);
        put(' ');
        put_u32(steps);
        put('\n');
    }

    puts_("elapsed_us ");
    if (measured_us >> 32) {
        puts_("overflow");
    } else {
        put_u32((uint32_t)measured_us);
    }
    put('\n');
    return index;
}

int main(void)
{
#ifdef DM_BSS_PREFILL
    volatile uint32_t bss_bad = bss_nonzero();
#endif
    dm_machine_init();
    dm_io_begin();
#ifdef DM_BSS_PREFILL
    if (bss_bad != 0u) {
        static const char failure[] = "FAILED: .bss was not cleared\n";
        dm_io_write(failure, sizeof failure - 1u);
        dm_io_end(76);
    }
    {
        static const char passed[] = "BSS_CLEAR ok\n";
        dm_io_write(passed, sizeof passed - 1u);
    }
#endif

    const uint32_t fuel = dm_config_fuel();
    const uint32_t reps = dm_config_reps();
    const int quiet = dm_config_quiet();

    /* `BEGIN`/`END` bracket every pass. On a machine that repeats forever -- a
     * Pico has no way to know when a host attached to its UART -- this is what
     * lets the reader discard a partial pass and keep a whole one, instead of
     * parsing a trace that begins in the middle of a drawing and is well-formed
     * all the way to the end. */
    do {
        dm_corpus_rewind();
#if defined(DM_BRINGUP_STAGE) && \
    (DM_BRINGUP_STAGE >= 18 && DM_BRINGUP_STAGE <= 21)
        /* Stage 13's failure surface is deliberately split here. The volatile
         * accesses prevent the compiler from replacing the probe with a known
         * constant, which is the memory operation the diagnostic is meant to
         * exercise. */
        volatile uint32_t *const length_probe = &out_len;
        const uint32_t observed_out_len = *length_probe;
        DM_HARNESS_BRINGUP(18);
        if (observed_out_len != 0u) {
            static const char failure[] = "FAILED: out_len was not zero\n";
            dm_io_write(failure, (uint32_t)(sizeof failure - 1u));
            dm_io_end(77);
        }
        DM_HARNESS_BRINGUP(19);

        volatile char *const output_probe = out;
        const char canary = 'Q';
        output_probe[0] = canary;
        if (output_probe[0] != canary) {
            static const char failure[] = "FAILED: out canary mismatch\n";
            dm_io_write(failure, (uint32_t)(sizeof failure - 1u));
            dm_io_end(78);
        }
        DM_HARNESS_BRINGUP(20);
#endif
        puts_("BEGIN ");
#if defined(DM_BRINGUP_STAGE) && \
    (DM_BRINGUP_STAGE >= 18 && DM_BRINGUP_STAGE <= 21)
        DM_HARNESS_BRINGUP(21);
#endif
        put_u32(dm_config_batch());
        put('\n');
        const uint32_t n = reps ? cycle_corpus(fuel, reps) : trace_corpus(fuel, quiet);
        puts_("END ");
        put_u32(n);
        put('\n');
        flush();
    } while (dm_io_again());

    dm_io_end(0);
    return 0;
}
