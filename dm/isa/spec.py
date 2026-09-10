"""Drawing ISA v1 -- opcode table, operand kinds, encoding rules.

Single source of truth. The assembler, the three codecs, the reference VM and
the eventual C port all derive their tables from here.

Coordinates are 8-bit unsigned over a 256x256 canvas. That width is uniform on
purpose: it keeps the byte stream naturally aligned, and it makes the token,
byte and bit representations carry *identical* information, so the
representation ablation has no quantization confound.

Instructions are variable-length but byte-aligned: one opcode byte followed by
a fixed operand count determined by the opcode.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

CANVAS = 256
COORD_BITS = 8
MAX_REPEAT_DEPTH = 4


class Op(IntEnum):
    HALT = 0x00
    MOVE = 0x01
    LINE = 0x02
    CURVE = 0x03
    CIRCLE = 0x04
    WIDTH = 0x05
    FILL = 0x06
    REPEAT = 0x07
    ENDREP = 0x08
    CALL = 0x09
    COLOR = 0x0A
    # ISA v2, 2026-08-11. The transform tier: `XFORM`/`ENDX` scope a
    # D4-and-translation over the coordinates a body emits, and `REPEATX` is
    # `REPEAT` whose iteration k applies that transform k times -- the one of
    # the three with a compression ceiling, because a motif and its mirror cost
    # `body + 5` bytes instead of two bodies. See `dm/isa/transform.py`.
    XFORM = 0x0B
    ENDX = 0x0C
    REPEATX = 0x0D


class Kind(IntEnum):
    """Operand alphabets. The distinction matters only to the typed token codec."""

    COORD = 0   # u8, canvas coordinate
    DELTA = 1   # i8 stored two's complement, per-iteration offset
    COUNT = 2   # u8, repeat count, >= 1
    SCALAR = 3  # u8, radius / stroke width
    ID = 4      # u8, subroutine index
    #: u8 packing one element of D4: bit 0 mirror, bits 1-2 quarter turns.
    #:
    #: A `Kind` of its own rather than a `COUNT`, and the reason is not
    #: bookkeeping: `dm/data/augment.py` transforms operands **by kind**, and
    #: under an augmentation `A` a transform operand `T` must **conjugate** to
    #: `A T A^-1` where a count is invariant and a delta takes the linear part.
    #: D4 is non-abelian, so an `XF` operand augmented as anything else is
    #: silently wrong. It is also what makes the typed-token axis non-degenerate
    #: on an L2 corpus, which every L0-only corpus proved it cannot be.
    XF = 5


class Tier(IntEnum):
    L0 = 0        # flat trace: structurally valid by construction
    L1 = 1        # control flow: REPEAT / ENDREP
    L2 = 2        # learned library: CALL
    RESERVED = 3  # allocated, not yet in any experiment


@dataclass(frozen=True)
class InstrSpec:
    op: Op
    mnemonic: str
    operands: tuple[Kind, ...]
    tier: Tier

    @property
    def size(self) -> int:
        """Encoded length in bytes, opcode included."""
        return 1 + len(self.operands)


_SPECS = (
    InstrSpec(Op.HALT, "HALT", (), Tier.L0),
    InstrSpec(Op.MOVE, "MOVE", (Kind.COORD, Kind.COORD), Tier.L0),
    InstrSpec(Op.LINE, "LINE", (Kind.COORD, Kind.COORD), Tier.L0),
    InstrSpec(Op.CURVE, "CURVE", (Kind.COORD,) * 6, Tier.L0),
    InstrSpec(Op.CIRCLE, "CIRCLE", (Kind.SCALAR,), Tier.L0),
    InstrSpec(Op.WIDTH, "WIDTH", (Kind.SCALAR,), Tier.L0),
    InstrSpec(Op.FILL, "FILL", (), Tier.L0),
    InstrSpec(Op.REPEAT, "REPEAT", (Kind.COUNT, Kind.DELTA, Kind.DELTA), Tier.L1),
    InstrSpec(Op.ENDREP, "ENDREP", (), Tier.L1),
    InstrSpec(Op.CALL, "CALL", (Kind.ID,), Tier.L2),
    InstrSpec(Op.COLOR, "COLOR", (Kind.SCALAR,), Tier.RESERVED),
    # Appended, never inserted. The token codecs index an opcode by its
    # *position* in this tuple, so inserting v2 opcodes above `COLOR` would have
    # moved `COLOR` from index 10 to 13 and made the legacy layout unreproducible
    # for it. New rows go at the end, always.
    InstrSpec(Op.XFORM, "XFORM", (Kind.XF, Kind.DELTA, Kind.DELTA), Tier.L2),
    InstrSpec(Op.ENDX, "ENDX", (), Tier.L2),
    InstrSpec(Op.REPEATX, "REPEATX", (Kind.COUNT, Kind.XF, Kind.DELTA, Kind.DELTA),
              Tier.L2),
)

SPECS: dict[Op, InstrSpec] = {s.op: s for s in _SPECS}
BY_MNEMONIC: dict[str, InstrSpec] = {s.mnemonic: s for s in _SPECS}
N_OPCODES = len(_SPECS)

#: Slots the token alphabets reserve for opcodes, **fixed and larger than
#: `N_OPCODES` on purpose**.
#:
#: `TokenCodec` lays out `[specials][opcodes][values]`, so before this existed
#: every new opcode shifted the value region and changed `vocab_size` -- which
#: meant a checkpoint trained under one opcode table decoded its own generated
#: symbols to *different opcodes* under the next, silently, because the model is
#: rebuilt from the vocabulary in its blob while the codec is rebuilt from
#: today's table. Reserving the region once makes ISA v2 and everything after it
#: additive.
#:
#: The 2026-08-11 migration to 16 slots is the one deliberate break:
#: `token` 269 -> 274 and `token_typed` 1293 -> 1373. `byte` and `bit` are
#: unaffected at any table size, which is why no claim-2, claim-3 or claim-4
#: number moved. A pre-v2 checkpoint is reconstructed exactly with
#: `TokenCodec(opcode_slots=11)`, pinned by
#: `test_the_pre_v2_token_layout_can_still_be_reconstructed`.
OPCODE_SLOTS = 16
assert N_OPCODES <= OPCODE_SLOTS, "the reserved opcode region has run out"


class ISAError(Exception):
    pass


class UnknownOpcode(ISAError):
    def __init__(self, byte: int) -> None:
        super().__init__(f"unknown opcode 0x{byte:02x}")
        self.byte = byte


def spec_for(byte: int) -> InstrSpec:
    try:
        return SPECS[Op(byte)]
    except ValueError as exc:
        raise UnknownOpcode(byte) from exc


def is_opcode(byte: int) -> bool:
    return byte in SPECS


def ops_for_tier(max_tier: Tier) -> tuple[Op, ...]:
    """Opcodes available at or below `max_tier`, excluding RESERVED."""
    return tuple(s.op for s in _SPECS if s.tier <= max_tier)


def encode_operand(kind: Kind, value: int) -> int:
    """Python int -> operand byte."""
    if kind is Kind.DELTA:
        if not -128 <= value <= 127:
            raise ISAError(f"delta {value} out of range [-128, 127]")
        return value & 0xFF
    if kind is Kind.XF and not 0 <= value < 8:
        # Authoring is strict where execution is permissive: `VM.run` masks a
        # transform byte to three bits so a *generated* program stays total,
        # but an assembler that quietly turned `XFORM 9` into `XFORM 1` would
        # let a corpus generator write one drawing and mean another.
        raise ISAError(f"transform code {value} outside D4 (0..7)")
    if not 0 <= value <= 255:
        raise ISAError(f"{kind.name} operand {value} out of range [0, 255]")
    return value


def decode_operand(kind: Kind, byte: int) -> int:
    """Operand byte -> Python int."""
    if kind is Kind.DELTA:
        return byte - 256 if byte >= 128 else byte
    return byte
