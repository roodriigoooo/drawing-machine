/* Drawing VM for Cortex-M0+ -- the device port of `dm/vm/interp.py`.
 *
 * The Python file is the specification and this is required to match it
 * *exactly*, not approximately. Two design decisions make that achievable, and
 * both are properties of the ISA rather than concessions:
 *
 * ## 1. There is no floating point, and nothing is lost by removing it
 *
 * The reference VM computes in `double`, so a port is normally a choice between
 * soft-float (kilobytes of flash and hundreds of cycles per operation on a chip
 * with no FPU) and fixed point (fast, but a different answer). Here it is
 * neither, because the reference's geometry is *already integral* at a scale of
 * `DM_ONE`:
 *
 *   - Every corner point is an integer. `place()` clamps an 8-bit operand plus
 *     an integer offset, and the only writer of the current point is `place()`
 *     or a curve's own endpoint, which is a `place()` result.
 *   - Curve samples are taken at `t = i/S` for `i = 1..S`. With `S` a power of
 *     two the Bernstein coefficients `u^3, 3u^2t, 3ut^2, t^3` are exactly
 *     `A/S^3, B/S^3, C/S^3, D/S^3` for integers `A+B+C+D = S^3`.
 *
 * So each emitted coordinate is `(A*x0 + B*x1 + C*x2 + D*x3) / S^3` with an
 * integer numerator, and `DM_FRAC_BITS = 3*log2(S)` makes that numerator the
 * fixed-point value itself -- no rounding, no division, no shift. The reference
 * computes the same quantity in `double`, where every intermediate is a small
 * multiple of a power of two and therefore also exact. **The two agree bit for
 * bit, and `scripts/conformance.py` asserts equality rather than a tolerance.**
 *
 * The bound is `S^3 * DM_COORD_MAX` = 1,044,480 at the default `S = 16`, so
 * `int32_t` carries it with 2000x of headroom and `int16_t` cannot carry it at
 * all -- it needs 21 bits.
 *
 * This exactness is conditional on `S` being a power of two. At `S = 10` the
 * coefficients are not binary fractions, the reference itself starts rounding,
 * and no integer port can match it. `DM_CURVE_STEPS_LOG2` is the knob; there is
 * deliberately no way to ask for a non-power-of-two.
 *
 * ## 2. Geometry is streamed, so peak SRAM does not depend on the program
 *
 * The reference accumulates a `path` list and turns it into a stroke or a
 * region when it ends. Ported literally that is unbounded: a 3 KB program of
 * curves is ~7,000 points, ~56 KB, which does not fit the part this project
 * claims to target. The port emits each point as it is produced and reports
 * what the path turned out to be in `path_end`. A consumer that must know
 * up-front can buffer; a plotter or a rasteriser that can defer does not have
 * to, and the interpreter's own footprint becomes a constant.
 *
 * What remains resident is the frame stack (`DM_MAX_REPEAT_DEPTH` entries) and
 * a handful of scalars. There is no recursion and no allocation, so worst-case
 * stack is a sum over a straight-line call graph rather than an estimate.
 */

#ifndef DM_VM_H
#define DM_VM_H

#include <stdint.h>

#include "dm_isa.h"

/* Curve flattening. `DM_CURVE_STEPS` must equal `CURVE_STEPS` in
 * `dm/vm/interp.py` for the port to agree with the reference, and must be a
 * power of two for the agreement to be exact (see above). */
#ifndef DM_CURVE_STEPS_LOG2
#define DM_CURVE_STEPS_LOG2 4
#endif

#define DM_CURVE_STEPS (1 << DM_CURVE_STEPS_LOG2)

/* Fixed-point scale of every emitted coordinate: one canvas unit is `DM_ONE`.
 * Chosen as the curve denominator rather than for convenience, because that is
 * what makes flattening exact and division-free. */
#define DM_FRAC_BITS (3 * DM_CURVE_STEPS_LOG2)
#define DM_ONE (1 << DM_FRAC_BITS)

/* `A*x0 + B*x1 + C*x2 + D*x3 <= S^3 * DM_COORD_MAX` must fit a signed 32-bit
 * accumulator. That is the only constraint on the knob. */
_Static_assert(DM_CURVE_STEPS_LOG2 >= 0 && DM_CURVE_STEPS_LOG2 <= 7,
               "DM_CURVE_STEPS_LOG2 outside the range where flattening fits int32_t");

/* Canvas coordinate, Q(31-DM_FRAC_BITS).DM_FRAC_BITS. */
typedef int32_t dm_fx;

/* Every condition the reference reports, with the same names. Faults are
 * reported, never raised: a generated program is expected to be malformed
 * sometimes, and the validity rate is one of the project's headline metrics. */
typedef enum {
    DM_FAULT_UNKNOWN_OPCODE = 0,
    DM_FAULT_TRUNCATED,
    DM_FAULT_UNMATCHED_ENDREP,
    DM_FAULT_UNTERMINATED_REPEAT,
    DM_FAULT_DEPTH_OVERFLOW,
    DM_FAULT_ZERO_REPEAT,
    DM_FAULT_OUT_OF_FUEL,
    DM_FAULT_NO_HALT,
    DM_FAULT_CALL_UNSUPPORTED
} dm_fault_kind;

/* What a path turned out to be, delivered at `path_end`.
 *
 * DISCARDED is not an error. The reference drops a path of fewer than two
 * points on a flush and fewer than three on a FILL, so a consumer that rendered
 * the points as they arrived must undo them. Streaming buys bounded memory and
 * this is what it costs. */
typedef enum {
    DM_PATH_DISCARDED = 0,
    DM_PATH_STROKE,
    DM_PATH_REGION
} dm_path_kind;

/* Geometry out. Any callback may be NULL, in which case that class of output is
 * discarded and the interpreter still runs -- useful for counting faults or
 * timing execution without a consumer attached. */
typedef struct {
    void *ctx;
    void (*path_begin)(void *ctx);
    void (*path_point)(void *ctx, dm_fx x, dm_fx y);
    void (*path_end)(void *ctx, dm_path_kind kind, uint32_t n_points, uint8_t width);
    void (*disc)(void *ctx, dm_fx cx, dm_fx cy, uint8_t radius, uint8_t width);
    void (*fault)(void *ctx, dm_fault_kind kind, uint32_t pc);
} dm_sink;

typedef struct {
    uint32_t steps;     /* instructions executed; the reference's `Trace.steps` */
    uint32_t n_faults;  /* callbacks made to `sink->fault` */
    uint8_t halted;     /* reached HALT. A halted program can still have faulted. */
} dm_result;

/* Instruction budget matching the reference's `DEFAULT_FUEL`. */
#define DM_DEFAULT_FUEL 100000u

/* Execute `program`. Re-entrant and allocation-free; the only state is the
 * caller's `sink` and this call's own frame. */
dm_result dm_vm_run(const uint8_t *program, uint32_t length, const dm_sink *sink, uint32_t fuel);

#endif /* DM_VM_H */
