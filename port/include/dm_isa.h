/* Drawing ISA v1 -- the opcode table, mirrored for the device.
 *
 * `dm/isa/spec.py` is the single source of truth. This header is a mirror, and
 * a mirror is a convention unless something checks it, so `tests/test_port.py`
 * parses this file and asserts it agrees with `spec.py` on every opcode, every
 * instruction size, and the two structural assumptions the interpreter makes:
 *
 *   1. Opcodes are contiguous from 0x00, so `byte < DM_N_OPCODES` is a complete
 *      validity test and the device needs no search and no sentinel table.
 *   2. `Kind.DELTA` -- the only signed operand alphabet -- occurs in `REPEAT`,
 *      `XFORM` and `REPEATX` and nowhere else, so the interpreter sign-extends
 *      at three named places instead of carrying a per-operand kind table into
 *      flash.
 *
 * Both are true of ISA v2 and neither is guaranteed by it. If a future opcode
 * breaks one, the test fails here rather than the geometry diverging on device.
 * Assumption 2 was a single site under v1 and the transform tier is what widened
 * it; `tests/test_port.py` asserts the exact set rather than the count, so a
 * fourth signed instruction fails rather than joining silently.
 */

#ifndef DM_ISA_H
#define DM_ISA_H

#include <stdint.h>

/* 8-bit coordinates over a 256x256 canvas (`spec.py`: CANVAS, COORD_BITS). */
#define DM_CANVAS 256
#define DM_COORD_MAX (DM_CANVAS - 1)

/* Nesting bound for REPEAT. The interpreter's frame stack is exactly this deep
 * and is a fixed-size array, which is what keeps its stack usage a constant
 * rather than a function of the program. */
#ifndef DM_MAX_REPEAT_DEPTH
#define DM_MAX_REPEAT_DEPTH 4
#endif

/* Nesting bound for open transform scopes, and the same number for the same
 * reason. It is a separate name because the two stacks are separate -- a
 * `REPEATX` occupies one entry in each -- so a device that wanted shallower
 * transforms than loops could lower this alone. `dm/vm/interp.py` bounds both by
 * `MAX_REPEAT_DEPTH`, and the two must agree or a program that faults on one
 * side runs on the other. */
#ifndef DM_MAX_XFORM_DEPTH
#define DM_MAX_XFORM_DEPTH DM_MAX_REPEAT_DEPTH
#endif

typedef enum {
    DM_OP_HALT = 0x00,
    DM_OP_MOVE = 0x01,
    DM_OP_LINE = 0x02,
    DM_OP_CURVE = 0x03,
    DM_OP_CIRCLE = 0x04,
    DM_OP_WIDTH = 0x05,
    DM_OP_FILL = 0x06,
    DM_OP_REPEAT = 0x07,
    DM_OP_ENDREP = 0x08,
    DM_OP_CALL = 0x09,
    DM_OP_COLOR = 0x0A,
    /* ISA v2: the transform tier. */
    DM_OP_XFORM = 0x0B,
    DM_OP_ENDX = 0x0C,
    DM_OP_REPEATX = 0x0D
} dm_op;

#define DM_N_OPCODES 14

/* Elements of D4, and the mask an operand is read through. A generated program
 * can carry any byte, so the interpreter masks rather than validates -- the
 * reference does the same, and an assembler is where an out-of-range code is
 * refused. */
#define DM_D4_ORDER 8
#define DM_D4_MASK (DM_D4_ORDER - 1)

/* Encoded length in bytes, opcode included, indexed by opcode. 14 bytes of
 * flash, and the only ISA table the device carries. */
extern const uint8_t dm_instr_size[DM_N_OPCODES];

/* Complete because opcodes are contiguous from zero -- assumption 1 above. */
static inline int dm_is_opcode(uint8_t byte)
{
    return byte < DM_N_OPCODES;
}

#endif /* DM_ISA_H */
