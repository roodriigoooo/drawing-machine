"""The relative view of a program: same bytecode, coordinates re-expressed as
deltas from the pen.

This is a **codec-level** transform and deliberately not an ISA change. The
bytecode on disk, the VM, `REPEAT`'s coordinate semantics and the C port stay
absolute -- section 2 of `PLAN.md` chose absolute coordinates so that `REPEAT`
has a well-defined transform and drift stays out of the VM, and none of that is
being reopened. What changes is only what the *model* is shown, in the same way
the byte and bit alphabets change what the model is shown without changing the
program.

The map is exact and bijective. Coordinates are u8 and the delta is taken
mod 256, so `to_absolute(to_relative(p)) == p` for every well-formed program and
the two views carry *identical* information -- the same invariant the four
alphabets already satisfy, and the reason relativity is a fourth ablation axis
rather than a different experiment.

**Why it is owed.** Tier C has no budget at which both convergence guards pass
(4,613 programs), and the obvious fix was refused by measurement: x18
integer-affine augmentation left the best at 156.11 against an un-augmented
156.75, still drifting +9.6. The diagnosis is in the representation, not in the
policy -- **under absolute coordinates a translated icon shares no operand byte
with its original, so augmentation multiplies the task exactly as fast as it
multiplies the corpus.** Under this view a translation changes the *first*
coordinate pair and nothing else, which `test_relative.py` pins as a byte count
rather than as a claim.

Two consequences that have to be stated before any number is read:

- **Mirrors and quarter turns are still full-price.** They act on the deltas
  themselves, so they remain a genuine task multiplier. Only translation
  becomes cheap, and a policy that leans on the other two has learnt nothing
  from the x18 result.
- **It changes what claim 2's recovery metric means.** A translational repeat
  is, in this view, a *literally repeated symbol sequence* -- later copies
  differ from the first only in the delta that steps between them. Recovery
  then scores copy detection, which is a far easier problem than the structure
  discovery the metric was built for. Claim 2's headline number stays on the
  absolute view; the relative one is a contrast arm and must be labelled as
  one (`dm/eval/recovery.py`).
"""

from __future__ import annotations

from .spec import Kind, SPECS, UnknownOpcode, spec_for

#: Coordinates are (x, y) pairs everywhere in the ISA, and the rewrite below
#: consumes them two at a time. A future instruction with an odd run of COORD
#: operands would silently pair an x with the next instruction's y, so the
#: invariant is checked once at import rather than trusted.
for _spec in SPECS.values():
    _run = 0
    for _kind in (*_spec.operands, Kind.SCALAR):  # sentinel closes a trailing run
        if _kind is Kind.COORD:
            _run += 1
            continue
        if _run % 2:
            raise AssertionError(
                f"{_spec.mnemonic} has an odd run of COORD operands; "
                "dm.isa.relative pairs them and would desynchronise"
            )
        _run = 0


def _rewrite(program: bytes, relative: bool) -> bytes:
    """Walk the instruction stream, rewriting COORD operands in place.

    Permissive in the same way and for the same reason as
    `dm.isa.codec.opcode_mask`: on an unknown opcode or a truncated instruction
    it stops and copies the remainder verbatim, so a structurally impossible
    stream still round-trips to bytes the VM can fault on. Raising here would
    move validity measurement out of the VM and into the codec for one arm
    only, which is exactly what the four-codec design exists to avoid.

    Only `Kind.COORD` moves. A `CIRCLE` radius and a `WIDTH` are lengths, and a
    `REPEAT`'s dx/dy is already a displacement -- rewriting either against the
    pen would corrupt it. That is the same by-Kind rule `dm.data.augment` uses,
    and it is why both live off `spec.operands` rather than off operand index.
    """
    out = bytearray(program)
    pen_x = pen_y = 0
    pc = 0
    while pc < len(program):
        try:
            spec = spec_for(program[pc])
        except UnknownOpcode:
            break
        if pc + spec.size > len(program):
            break
        index = 0
        while index < len(spec.operands):
            if spec.operands[index] is not Kind.COORD:
                index += 1
                continue
            # Read from `program` and write to `out`: the source values are the
            # ones being converted, and an in-place read would consume a value
            # this loop has already overwritten.
            x, y = program[pc + 1 + index], program[pc + 2 + index]
            if relative:
                out[pc + 1 + index] = (x - pen_x) & 0xFF
                out[pc + 2 + index] = (y - pen_y) & 0xFF
                pen_x, pen_y = x, y
            else:
                pen_x, pen_y = (pen_x + x) & 0xFF, (pen_y + y) & 0xFF
                out[pc + 1 + index], out[pc + 2 + index] = pen_x, pen_y
            index += 2
        pc += spec.size
    return bytes(out)


def to_relative(program: bytes) -> bytes:
    """Absolute bytecode -> the relative view. Inverse of `to_absolute`."""
    return _rewrite(program, relative=True)


def to_absolute(program: bytes) -> bytes:
    """The relative view -> absolute bytecode. Inverse of `to_relative`."""
    return _rewrite(program, relative=False)
