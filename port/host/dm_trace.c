/* Host harness: run bytecode through the device interpreter and print the trace
 * in a form `scripts/conformance.py` can diff against `dm/vm/interp.py`.
 *
 * Reads length-prefixed programs from stdin -- `uint32` little-endian length,
 * then that many bytes, repeated to EOF -- so a corpus of thousands costs one
 * process instead of thousands.
 *
 * Coordinates are printed as raw fixed-point integers. Not a formatting
 * preference: the claim being checked is bit-exact agreement with the
 * reference, and printing decimals would first have to round, which is the one
 * thing the port exists to avoid.
 *
 * The tool holds no path buffer. Points are printed as they stream and the
 * verdict follows, which is the sink contract in `dm_vm.h` exercised as
 * written rather than papered over with a host-side array.
 *
 *     printf '\x05\x00\x00\x00\x01\x10\x10\x02\x20' | dm_trace
 *
 * Usage: dm_trace [fuel]
 */

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "dm_vm.h"

/* Longest program the harness accepts. `max_len` in the training configs is
 * 3072 bytes; this is an order of magnitude above the largest corpus and the
 * limit is diagnosed rather than silently truncating a program into a
 * different one. */
#define DM_TRACE_MAX_PROGRAM 65536

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

static void on_path_begin(void *ctx)
{
    (void)ctx;
    fputs("b\n", stdout);
}

static void on_path_point(void *ctx, dm_fx x, dm_fx y)
{
    (void)ctx;
    printf("p %ld %ld\n", (long)x, (long)y);
}

static void on_path_end(void *ctx, dm_path_kind kind, uint32_t n_points, uint8_t width)
{
    (void)ctx;
    printf("e %s %lu %u\n", PATH_NAME[kind], (unsigned long)n_points, width);
}

static void on_disc(void *ctx, dm_fx cx, dm_fx cy, uint8_t radius, uint8_t width)
{
    (void)ctx;
    printf("d %ld %ld %u %u\n", (long)cx, (long)cy, radius, width);
}

static void on_fault(void *ctx, dm_fault_kind kind, uint32_t pc)
{
    (void)ctx;
    printf("f %s %lu\n", FAULT_NAME[kind], (unsigned long)pc);
}

/* Little-endian on the wire regardless of host order, so the harness cannot
 * disagree with the Python side about framing on some future machine. */
static int read_length(uint32_t *out)
{
    unsigned char header[4];

    if (fread(header, 1, sizeof header, stdin) != sizeof header) {
        return 0;
    }
    *out = (uint32_t)header[0] | ((uint32_t)header[1] << 8) | ((uint32_t)header[2] << 16) |
           ((uint32_t)header[3] << 24);
    return 1;
}

int main(int argc, char **argv)
{
    static uint8_t program[DM_TRACE_MAX_PROGRAM];
    const dm_sink sink = {
        .ctx = NULL,
        .path_begin = on_path_begin,
        .path_point = on_path_point,
        .path_end = on_path_end,
        .disc = on_disc,
        .fault = on_fault,
    };
    uint32_t fuel = DM_DEFAULT_FUEL;
    uint32_t length;
    unsigned long index = 0;

    if (argc > 2) {
        fprintf(stderr, "usage: %s [fuel]\n", argv[0]);
        return 2;
    }
    if (argc == 2) {
        fuel = (uint32_t)strtoul(argv[1], NULL, 10);
    }

    /* `BEGIN`/`END` bracket the run on all three harnesses. Here and under QEMU
     * the framing is redundant -- the stream begins where the run began -- but
     * on a Pico there is no channel inward, so the firmware repeats the corpus
     * forever and the reader keeps whichever pass it catches whole. One protocol
     * for all three is the entire reason `scripts/conformance.py` can diff a
     * silicon trace with the parser it already had, so the brackets are emitted
     * here too rather than tolerated there. */
    printf("BEGIN 0\nfrac_bits %d\ncurve_steps %d\n", DM_FRAC_BITS, DM_CURVE_STEPS);

    while (read_length(&length)) {
        if (length > sizeof program) {
            fprintf(stderr, "program %lu is %lu bytes, over the %zu-byte harness limit\n", index,
                    (unsigned long)length, sizeof program);
            return 1;
        }
        if (fread(program, 1, length, stdin) != length) {
            fprintf(stderr, "program %lu truncated on stdin\n", index);
            return 1;
        }

        printf("#%lu\n", index);
        const dm_result result = dm_vm_run(program, length, &sink, fuel);
        printf("= %lu %u %lu\n", (unsigned long)result.steps, result.halted,
               (unsigned long)result.n_faults);
        index++;
    }

    printf("END %lu\n", index);
    return ferror(stdout) ? 1 : 0;
}
