/* Device interpreter. See `dm_vm.h` for why there is no floating point and why
 * geometry is streamed; this file is the line-for-line counterpart of
 * `dm/vm/interp.py` and is meant to be read beside it.
 *
 * Ordering is part of the specification and several orderings here look
 * arbitrary until you check the reference:
 *
 *   - WIDTH flushes *before* it changes the width, so the flushed stroke keeps
 *     the old one.
 *   - A zero-count REPEAT at maximum depth reports ZERO_REPEAT *and*
 *     DEPTH_OVERFLOW, in that order.
 *   - HALT inside an open REPEAT reports UNTERMINATED_REPEAT and still halts,
 *     so a program can be both halted and faulted.
 *   - FILL drops a path of fewer than three points instead of flushing it as a
 *     stroke.
 *   - COLOR is a known opcode with no behaviour. It advances the program
 *     counter and does nothing else.
 *
 * Nothing here allocates, recurses, or calls libc.
 */

#include "dm_vm.h"

/* An element of D4 and a translation: `p -> d4(p) + (dx, dy)`, the group
 * `dm/isa/transform.py` defines. `code` packs D4 into three bits -- bit 0 is
 * mirror, bits 1-2 are quarter turns -- and is the wire format, so it is read
 * from the operand unchanged.
 *
 * The translation is `int32_t` and the reason is arithmetic rather than caution:
 * a `REPEATX` count is `u8`, and iteration k runs under the step composed k
 * times, so a step carrying the extreme +-127 shift reaches about +-32,000 by
 * the last iteration. An `int16_t` would hold that and nothing beyond it. */
typedef struct {
    int32_t dx, dy;
    uint8_t code;
} dm_xform;

typedef struct {
    uint32_t body;      /* pc of the first instruction inside the loop */
    uint32_t count;     /* iterations requested; >= 1 once ZERO_REPEAT is applied */
    uint32_t iteration; /* completed iterations */
    int32_t dx;         /* per-iteration offset, sign-extended from the operand */
    int32_t dy;
    int32_t base_x;     /* offset in force when the loop was entered */
    int32_t base_y;

    /* The transform tier's four bytes, and they are four rather than a whole
     * `dm_xform` on purpose. A step's operands are `u8` and `i8` and never
     * accumulate -- what accumulates is its running power, which lives in the
     * transform stack where it is needed per point. Storing the step wide would
     * cost 8 bytes in every frame to hold values that cannot exceed a byte. */
    uint8_t step_code;
    int8_t step_dx, step_dy;
    uint8_t xf_slot;  /* transform-stack entry this loop owns, or DM_NO_SLOT */
    uint8_t xf_floor; /* scope floor in force before this loop opened */
} dm_frame;

#define DM_NO_SLOT 0xFFu

typedef struct {
    const dm_sink *sink;
    dm_result result;

    int32_t cur_x, cur_y; /* current point, integer canvas units (never fractional) */
    int32_t off_x, off_y; /* REPEAT translation in force */
    uint8_t width;

    uint32_t path_n; /* points emitted in the open path; 0 means no open path */

    dm_frame stack[DM_MAX_REPEAT_DEPTH];
    uint32_t depth;

    /* Open transform scopes, and their composition. `xf_active` is a cache: a
     * coordinate needs the whole stack applied, and recomputing a four-deep
     * composition per point would put a loop on the hottest path in the
     * interpreter. It is 12 bytes to make every placement two adds and a
     * conditional swap, and it is refreshed only when the stack changes --
     * which is once per scope, not once per point. */
    dm_xform xf_stack[DM_MAX_XFORM_DEPTH];
    dm_xform xf_active;
    uint32_t xf_depth;
    /* Entries below this index belong to an open `REPEATX` and cannot be closed
     * by an `ENDX`. A loop owns its slot, so a body that closed it would leave
     * the frame pointing past the end of the stack. `dm/isa/unroll.py` already
     * refuses that program as a crossing; the conformance fuzzer found it in the
     * *reference* as a crash, and both now report it as a fault. */
    uint32_t xf_floor;
} dm_state;

/* D4 composition in closed form, which is why the device carries no 8x8 table.
 *
 * Writing an element as `s^m r^t` -- t quarter turns, then mirror, the order
 * `dm/isa/transform.py:D4.point` applies them -- and using `r^t s = s r^-t`:
 *
 *     (s^m2 r^t2) (s^m1 r^t1) = s^(m1^m2) r^(t1 + (m1 ? -t2 : t2))
 *
 * `tests/test_port.py` checks all 64 products against the reference, which
 * derives its table from the action on the canvas rather than by transcription.
 */
static uint8_t dm_d4_then(uint8_t first, uint8_t second)
{
    const uint8_t m1 = first & 1u, m2 = second & 1u;
    const int32_t t1 = (first >> 1) & 3, t2 = (second >> 1) & 3;
    const int32_t turns = (m1 ? t1 - t2 : t1 + t2) & 3;

    return (uint8_t)((turns << 1) | (m1 ^ m2));
}

/* The action on a canvas *position*: a quarter turn is (x, y) -> (y, MAX - x)
 * and a mirror is x -> MAX - x. Every element maps the canvas onto itself with
 * integer arithmetic and no half-pixel centre, which is what keeps a transformed
 * repeat exact. */
static void dm_d4_point(uint8_t code, int32_t *x, int32_t *y)
{
    for (uint8_t turns = (code >> 1) & 3; turns > 0; turns--) {
        const int32_t px = *x;

        *x = *y;
        *y = DM_COORD_MAX - px;
    }
    if (code & 1u) {
        *x = DM_COORD_MAX - *x;
    }
}

/* The action on a *vector* -- a displacement, not a position, so the canvas
 * constant does not appear. The same distinction `dm/data/augment.py` draws
 * between a `COORD` and a `DELTA`, and getting it wrong is invisible on the
 * identity and wrong everywhere a rotation meets an offset. */
static void dm_d4_linear(uint8_t code, int32_t *x, int32_t *y)
{
    for (uint8_t turns = (code >> 1) & 3; turns > 0; turns--) {
        const int32_t px = *x;

        *x = *y;
        *y = -px;
    }
    if (code & 1u) {
        *x = -*x;
    }
}

/* `first` applied, then `second`. Composing the affine parts gives
 * `B(A p + t_A) + t_B = (BA) p + (B t_A + t_B)`, so the accumulated translation
 * takes the *linear* part of `second` -- the same rule a delta operand follows,
 * for the same reason. */
static void dm_xform_then(dm_xform *first, const dm_xform *second)
{
    dm_d4_linear(second->code, &first->dx, &first->dy);
    first->dx += second->dx;
    first->dy += second->dy;
    first->code = dm_d4_then(first->code, second->code);
}

/* The open scopes as one transform, innermost acting first.
 *
 * `xf_stack[0]` is the outermost scope and acts *last*, which is every graphics
 * stack's convention and the only fold under which "a scope transforms
 * everything inside it" holds for nested scopes. D4 is not commutative, so the
 * other order is a different drawing rather than a different spelling. */
static void dm_xform_recompose(dm_state *st)
{
    dm_xform out = {0};

    for (uint32_t i = st->xf_depth; i-- > 0;) {
        dm_xform_then(&out, &st->xf_stack[i]);
    }
    st->xf_active = out;
}

/* One canvas unit is DM_ONE, and the current point is always integral, so
 * placing it costs a shift rather than a conversion. */
static dm_fx dm_to_fx(int32_t canvas_units)
{
    return (dm_fx)(canvas_units << DM_FRAC_BITS);
}

static int32_t dm_clamp(int32_t value)
{
    if (value < 0) {
        return 0;
    }
    if (value > DM_COORD_MAX) {
        return DM_COORD_MAX;
    }
    return value;
}

/* `place()` in the reference: translate by the active REPEAT offset, map through
 * the open transform scopes, then clamp to the canvas. The sum needs more than
 * eight bits -- nesting to DM_MAX_REPEAT_DEPTH with the extreme delta reaches
 * about +-130,000 -- which is why the offsets are int32_t and not int16_t.
 *
 * **Two axes at once, and that is forced rather than tidy**: a rotation mixes x
 * into y, so the per-axis form this replaced cannot express the tier at all. The
 * two coordinates return in registers under AAPCS, so the shape costs nothing.
 *
 * The order is the specification: offset first, then transform, then clamp. A
 * `REPEAT` step is a translation in the *untransformed* frame, and clamping
 * before the transform would fold a canvas edge into the middle of a rotated
 * drawing. `PLAN.md`'s direction-item-2 decision 4 fixed it before the first run
 * rather than after.
 *
 * An empty scope stack is the common case -- every L0 and L1 program ever
 * measured -- so it is one predictable branch away from the arithmetic this
 * function did before the tier existed. */
typedef struct {
    int32_t x, y;
} dm_point;

static dm_point dm_place(const dm_state *st, uint8_t vx, uint8_t vy)
{
    int32_t x = (int32_t)vx + st->off_x;
    int32_t y = (int32_t)vy + st->off_y;

    if (st->xf_depth != 0) {
        dm_d4_point(st->xf_active.code, &x, &y);
        x += st->xf_active.dx;
        y += st->xf_active.dy;
    }
    return (dm_point){dm_clamp(x), dm_clamp(y)};
}

static void dm_fault(dm_state *st, dm_fault_kind kind, uint32_t pc)
{
    st->result.n_faults++;
    if (st->sink->fault) {
        st->sink->fault(st->sink->ctx, kind, pc);
    }
}

static void dm_path_push(dm_state *st, dm_fx x, dm_fx y)
{
    if (st->path_n == 0 && st->sink->path_begin) {
        st->sink->path_begin(st->sink->ctx);
    }
    st->path_n++;
    if (st->sink->path_point) {
        st->sink->path_point(st->sink->ctx, x, y);
    }
}

/* Close the open path, if there is one, as `kind` when it is long enough and as
 * DM_PATH_DISCARDED when it is not. `min_points` is 2 for a stroke and 3 for a
 * region -- the reference's two different thresholds, named rather than
 * duplicated. */
static void dm_path_close(dm_state *st, dm_path_kind kind, uint32_t min_points)
{
    if (st->path_n == 0) {
        return;
    }
    if (st->sink->path_end) {
        st->sink->path_end(st->sink->ctx,
                           st->path_n >= min_points ? kind : DM_PATH_DISCARDED,
                           st->path_n, st->width);
    }
    st->path_n = 0;
}

static void dm_flush(dm_state *st)
{
    dm_path_close(st, DM_PATH_STROKE, 2);
}

/* Cubic Bezier, flattened at t = i/S for i = 1..S, so the endpoint is emitted
 * and the start point is not -- the caller has already placed it.
 *
 * The Bernstein coefficients at those t are exactly A/S^3, B/S^3, C/S^3, D/S^3
 * for the integers computed below, and A+B+C+D = S^3 = DM_ONE. With integer
 * corner points the numerator *is* the fixed-point coordinate, so this is the
 * whole of the port's arithmetic: no division, no rounding, no float, and the
 * same value the reference obtains in double. `dm_vm.h` carries the argument
 * and `scripts/conformance.py` checks it against every corpus in the project.
 */
static void dm_emit_bezier(dm_state *st, int32_t x0, int32_t y0, int32_t x1, int32_t y1,
                           int32_t x2, int32_t y2, int32_t x3, int32_t y3)
{
    for (int32_t i = 1; i <= DM_CURVE_STEPS; i++) {
        const int32_t u = DM_CURVE_STEPS - i;
        const int32_t a = u * u * u;
        const int32_t b = 3 * u * u * i;
        const int32_t c = 3 * u * i * i;
        const int32_t d = i * i * i;

        dm_path_push(st, a * x0 + b * x1 + c * x2 + d * x3,
                     a * y0 + b * y1 + c * y2 + d * y3);
    }
}

dm_result dm_vm_run(const uint8_t *program, uint32_t length, const dm_sink *sink, uint32_t fuel)
{
    dm_state st = {0};
    uint32_t pc = 0;

    st.sink = sink;
    st.width = 1;

    while (pc < length) {
        if (st.result.steps >= fuel) {
            dm_fault(&st, DM_FAULT_OUT_OF_FUEL, pc);
            goto finished;
        }
        st.result.steps++;

        const uint8_t opcode = program[pc];
        if (!dm_is_opcode(opcode)) {
            dm_fault(&st, DM_FAULT_UNKNOWN_OPCODE, pc);
            goto finished;
        }

        const uint8_t size = dm_instr_size[opcode];
        /* `length - pc` rather than `pc + size` so a program at the top of the
         * address space cannot wrap the comparison. */
        if (size > length - pc) {
            dm_fault(&st, DM_FAULT_TRUNCATED, pc);
            goto finished;
        }

        const uint8_t *operand = &program[pc + 1];
        uint32_t next = pc + size;

        switch ((dm_op)opcode) {
        case DM_OP_HALT:
            dm_flush(&st);
            /* An unclosed transform scope is an unclosed scope, and halting
             * inside one would leave the last strokes mapped by something the
             * program never closed. */
            if (st.depth > 0 || st.xf_depth > 0) {
                dm_fault(&st, DM_FAULT_UNTERMINATED_REPEAT, pc);
            }
            st.result.halted = 1;
            goto finished;

        case DM_OP_MOVE: {
            dm_flush(&st);
            const dm_point at = dm_place(&st, operand[0], operand[1]);

            st.cur_x = at.x;
            st.cur_y = at.y;
            dm_path_push(&st, dm_to_fx(st.cur_x), dm_to_fx(st.cur_y));
            break;
        }

        case DM_OP_LINE: {
            if (st.path_n == 0) {
                dm_path_push(&st, dm_to_fx(st.cur_x), dm_to_fx(st.cur_y));
            }
            const dm_point at = dm_place(&st, operand[0], operand[1]);

            st.cur_x = at.x;
            st.cur_y = at.y;
            dm_path_push(&st, dm_to_fx(st.cur_x), dm_to_fx(st.cur_y));
            break;
        }

        case DM_OP_CURVE: {
            if (st.path_n == 0) {
                dm_path_push(&st, dm_to_fx(st.cur_x), dm_to_fx(st.cur_y));
            }
            const dm_point c1 = dm_place(&st, operand[0], operand[1]);
            const dm_point c2 = dm_place(&st, operand[2], operand[3]);
            const dm_point end = dm_place(&st, operand[4], operand[5]);

            dm_emit_bezier(&st, st.cur_x, st.cur_y, c1.x, c1.y, c2.x, c2.y, end.x, end.y);
            st.cur_x = end.x;
            st.cur_y = end.y;
            break;
        }

        case DM_OP_CIRCLE:
            /* Placed at the current point and does not disturb the open path. */
            if (st.sink->disc) {
                st.sink->disc(st.sink->ctx, dm_to_fx(st.cur_x), dm_to_fx(st.cur_y),
                              operand[0], st.width);
            }
            break;

        case DM_OP_WIDTH:
            dm_flush(&st);
            st.width = operand[0] > 1 ? operand[0] : 1;
            break;

        case DM_OP_FILL:
            dm_path_close(&st, DM_PATH_REGION, 3);
            break;

        /* One path for both loops, because `REPEATX` *is* `REPEAT` with a
         * transform: two loop bodies would be two places for the iteration
         * accounting to drift, and that accounting is what claim 2 measures.
         *
         * They differ only in where the operands go. `REPEAT` puts its step in
         * the translation offset; `REPEATX` puts its whole step -- rotation,
         * mirror and shift -- into one transform and takes nothing through the
         * offset, because only a whole transform conjugates exactly and a split
         * one does not commute with translating the program. */
        case DM_OP_REPEAT:
        case DM_OP_REPEATX: {
            const int transformed = opcode == DM_OP_REPEATX;
            uint32_t count = operand[0];

            if (count == 0) {
                dm_fault(&st, DM_FAULT_ZERO_REPEAT, pc);
                count = 1;
            }
            if (st.depth >= DM_MAX_REPEAT_DEPTH) {
                dm_fault(&st, DM_FAULT_DEPTH_OVERFLOW, pc);
                goto finished;
            }
            uint8_t slot = DM_NO_SLOT;
            if (transformed) {
                /* The transform stack has its own bound and shares
                 * DEPTH_OVERFLOW: the two are the same structural failure, and
                 * a fault kind is a name mirrored in `dm_vm.h`. */
                if (st.xf_depth >= DM_MAX_XFORM_DEPTH) {
                    dm_fault(&st, DM_FAULT_DEPTH_OVERFLOW, pc);
                    goto finished;
                }
                slot = (uint8_t)st.xf_depth;
                st.xf_stack[st.xf_depth++] = (dm_xform){0}; /* iteration 0 */
                dm_xform_recompose(&st);
            }
            st.stack[st.depth++] = (dm_frame){
                .body = next,
                .count = count,
                .iteration = 0,
                /* DELTA is the only signed kind; this and the two transform
                 * instructions are the only places it occurs. */
                .dx = transformed ? 0 : (int8_t)operand[1],
                .dy = transformed ? 0 : (int8_t)operand[2],
                .base_x = st.off_x,
                .base_y = st.off_y,
                .step_code = transformed ? (uint8_t)(operand[1] & DM_D4_MASK) : 0,
                .step_dx = transformed ? (int8_t)operand[2] : 0,
                .step_dy = transformed ? (int8_t)operand[3] : 0,
                .xf_slot = slot,
                .xf_floor = (uint8_t)st.xf_floor,
            };
            if (transformed) {
                st.xf_floor = (uint32_t)slot + 1u;
            }
            break;
        }

        case DM_OP_ENDREP: {
            if (st.depth == 0) {
                dm_fault(&st, DM_FAULT_UNMATCHED_ENDREP, pc);
                goto finished;
            }
            dm_frame *const frame = &st.stack[st.depth - 1];
            frame->iteration++;
            if (frame->iteration < frame->count) {
                dm_flush(&st);
                st.off_x = frame->base_x + frame->dx * (int32_t)frame->iteration;
                st.off_y = frame->base_y + frame->dy * (int32_t)frame->iteration;
                if (frame->xf_slot != DM_NO_SLOT) {
                    /* Iteration k runs under the step composed k times, built
                     * one composition per iteration rather than raised from
                     * scratch. Both are exact and equal; this one is O(1) where
                     * a power is O(k), and `count` reaches 255. */
                    const dm_xform step = {frame->step_dx, frame->step_dy,
                                           frame->step_code};

                    dm_xform_then(&st.xf_stack[frame->xf_slot], &step);
                    dm_xform_recompose(&st);
                }
                next = frame->body;
            } else {
                st.off_x = frame->base_x;
                st.off_y = frame->base_y;
                if (frame->xf_slot != DM_NO_SLOT) {
                    /* Truncate to the slot, which also discards any scope the
                     * body opened and never closed -- the reference's
                     * `del xforms[slot:]`, and the reason a leaked `XFORM`
                     * inside a loop does not survive the loop. */
                    st.xf_depth = frame->xf_slot;
                    dm_xform_recompose(&st);
                }
                st.xf_floor = frame->xf_floor;
                st.depth--;
            }
            break;
        }

        case DM_OP_XFORM: {
            /* The depth test precedes the flush, because the reference faults
             * before it closes the open path and a flush is observable. */
            if (st.xf_depth >= DM_MAX_XFORM_DEPTH) {
                dm_fault(&st, DM_FAULT_DEPTH_OVERFLOW, pc);
                goto finished;
            }
            dm_flush(&st);
            st.xf_stack[st.xf_depth++] = (dm_xform){
                (int8_t)operand[1], (int8_t)operand[2],
                (uint8_t)(operand[0] & DM_D4_MASK),
            };
            dm_xform_recompose(&st);
            break;
        }

        case DM_OP_ENDX:
            /* A closer with nothing open is `ENDREP`'s failure exactly, and it
             * shares the fault kind for the same reason DEPTH_OVERFLOW is
             * shared: the two are one structural error. */
            /* `<=` rather than `== 0`: an `ENDX` may not close a scope it did
             * not open, and a `REPEATX`'s slot belongs to the loop. */
            if (st.xf_depth <= st.xf_floor) {
                dm_fault(&st, DM_FAULT_UNMATCHED_ENDREP, pc);
                goto finished;
            }
            dm_flush(&st);
            st.xf_depth--;
            dm_xform_recompose(&st);
            break;

        case DM_OP_CALL:
            dm_fault(&st, DM_FAULT_CALL_UNSUPPORTED, pc);
            goto finished;

        case DM_OP_COLOR:
            /* Allocated, not implemented. Deliberately a no-op rather than a
             * fault: the reference has no branch for it, and Tier.RESERVED
             * opcodes must not make a program invalid. */
            break;
        }

        pc = next;
    }

finished:
    dm_flush(&st);
    if (!st.result.halted && st.result.n_faults == 0) {
        dm_fault(&st, DM_FAULT_NO_HALT, length);
    }
    return st.result;
}
