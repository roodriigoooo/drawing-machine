#include "dm_isa.h"

/* Ordered by opcode value, which `dm_instr_size[op]` depends on. Kept as an
 * explicit designated-initialiser table so a reordering cannot silently shift
 * the sizes, and so the mirror test can read the pairs out of the source. */
const uint8_t dm_instr_size[DM_N_OPCODES] = {
    [DM_OP_HALT] = 1,   /* -- */
    [DM_OP_MOVE] = 3,   /* COORD COORD */
    [DM_OP_LINE] = 3,   /* COORD COORD */
    [DM_OP_CURVE] = 7,  /* COORD x6 */
    [DM_OP_CIRCLE] = 2, /* SCALAR */
    [DM_OP_WIDTH] = 2,  /* SCALAR */
    [DM_OP_FILL] = 1,   /* -- */
    [DM_OP_REPEAT] = 4, /* COUNT DELTA DELTA */
    [DM_OP_ENDREP] = 1, /* -- */
    [DM_OP_CALL] = 2,   /* ID */
    [DM_OP_COLOR] = 2,  /* SCALAR */
    [DM_OP_XFORM] = 4,  /* XF DELTA DELTA */
    [DM_OP_ENDX] = 1,   /* -- */
    [DM_OP_REPEATX] = 5 /* COUNT XF DELTA DELTA */
};
