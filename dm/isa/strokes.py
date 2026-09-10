"""Programs as sequences of strokes, and the fixed-width summary of one.

Claim 3's factorisation -- non-autoregressive over strokes, autoregressive over
bits -- needs a stroke boundary and a stroke *summary*, and both have to be
exact in a specific sense:

- **`join(split(p)) == p`, byte for byte.** The planner's bits/drawing is only
  comparable with the AR baseline's if the two models are transmitting the same
  program. A split that dropped a `WIDTH` or normalised a `HALT` would make the
  comparison meaningless while still looking like a drawing.
- **The summary is a deterministic function of the stroke.** That is what makes
  the factorisation a likelihood decomposition rather than an analogy: with
  `s = f(x)`, `p(s) * p(x | s)` evaluated at `s = f(x)` lower-bounds `p(x)`, so
  its negative log *upper*-bounds the true cost -- the conservative direction,
  and the one that lets a planner result be believed. `dm/models/planner.py`
  carries the argument in full.

The summary is six bytes because six is what the composition level needs to
place a stroke without describing it: where it starts, how big it is, how much
of the program it costs, and whether it ends the drawing. Shape is the stroke decoder's job and QuickDraw's
level; layout is the composition level's job and Tier D's
(`docs/diffusion-strategy.md` 5.2).
"""

from __future__ import annotations

from typing import NamedTuple

from .spec import CANVAS, Kind, Op, SPECS, UnknownOpcode, spec_for

#: Bytes per stroke summary. Fixed width, so the composition level is a grid of
#: slots and a stroke count is a property of the *contents* rather than of the
#: tensor -- which is what lets one denoiser handle drawings of any length.
SUMMARY_BYTES = 6


class Summary(NamedTuple):
    """Where a stroke starts, how far it extends, what it costs, and whether it
    ends the drawing.

    All six fields are u8, so a summary is ISA-native: the composition level
    predicts bytes on the same 0-255 canvas the ISA uses, and no quantisation
    stage sits between the two levels to round repeats apart (`PLAN.md` 10).

    `halts` is the sixth because **termination is the AR baseline's standing
    failure** -- it over-assigns HALT at instruction boundaries by 1.5-2.6x and
    undershoots generated length by 22% with no sampling involved (`PLAN.md`
    9.5a). Without it a stroke decoder cannot know whether it is drawing the
    last stroke, so it would have to guess HALT from stroke shape, which is the
    same guess made worse. With it, the composition level decides where the
    drawing ends and the decoder is told -- and `halts` stays a deterministic
    function of the stroke's own bytes, because a stroke is terminal exactly
    when it contains HALT.
    """

    x0: int
    y0: int
    width: int
    height: int
    length: int
    halts: int

    def to_bytes(self) -> bytes:
        return bytes(self)

    @classmethod
    def of(cls, stroke: bytes) -> "Summary":
        """The summary of one stroke. Total: never raises, never rejects.

        A malformed stroke still gets a summary, because the planner is scored
        on generated programs too and validity is measured in the VM for every
        arm (`dm/vm/interp.py`). Coordinates are read by `Kind`, so a `CIRCLE`
        radius and a `REPEAT` delta do not enter the bounding box.
        """
        xs, ys = [], []
        halts, pc = 0, 0
        while pc < len(stroke):
            try:
                spec = spec_for(stroke[pc])
            except UnknownOpcode:
                break
            if pc + spec.size > len(stroke):
                break
            halts |= spec.op is Op.HALT
            index = 0
            while index < len(spec.operands):
                if spec.operands[index] is Kind.COORD:
                    xs.append(stroke[pc + 1 + index])
                    ys.append(stroke[pc + 2 + index])
                    index += 2
                else:
                    index += 1
            pc += spec.size
        length = min(len(stroke), CANVAS - 1)
        if not xs:
            return cls(0, 0, 0, 0, length, int(halts))
        return cls(
            xs[0], ys[0],
            max(xs) - min(xs), max(ys) - min(ys),
            length, int(halts),
        )


def split(program: bytes, max_strokes: int | None = None) -> list[bytes]:
    """The program as strokes, split at `MOVE` boundaries.

    A stroke is a `MOVE` and everything up to the next one, so anything before
    the first `MOVE` (a `WIDTH`, typically) joins the first stroke and the
    trailing `HALT` joins the last. That keeps `join(split(p)) == p` without a
    preamble or a terminator field, and both would be one more thing to model.

    `max_strokes` **merges** the overflow into the final stroke rather than
    dropping it. Dropping would remove bytes from the program and make
    bits/drawing incomparable with the AR baseline on precisely the drawings
    that are hardest; merging costs the last slot a coarse summary and keeps
    every byte in the likelihood. The count of merged programs is the caller's
    to report -- `dm.train_planner` does, in the same column
    `ProgramDataset.length_stats` uses for truncation.
    """
    starts, pc, opened = [], 0, False
    while pc < len(program):
        try:
            spec = spec_for(program[pc])
        except UnknownOpcode:
            break
        if spec.op is Op.MOVE:
            # The *first* MOVE does not open a stroke, it joins the one already
            # being built. Otherwise a leading `WIDTH` becomes a stroke of its
            # own -- a slot with no coordinates, a degenerate summary, and one
            # fewer slot for a stroke that has some.
            if opened:
                starts.append(pc)
            opened = True
        pc += spec.size
        if pc > len(program):
            break
    if max_strokes is not None:
        starts = starts[: max(0, max_strokes - 1)]
    bounds = [0, *starts, len(program)]
    return [program[a:b] for a, b in zip(bounds, bounds[1:]) if b > a] or [program]


def join(strokes: list[bytes]) -> bytes:
    return b"".join(strokes)


def to_boundary(program: bytes) -> bytes:
    """The longest prefix of `program` that is a whole number of instructions.

    The AR arm never emits a mid-instruction tail: `HaltMonitor` carries a parse
    state and stops each row *at a boundary*, with the stated invariant that the
    trace of the truncated stream equals the trace of the full one. Claim 3's
    sampler had no equivalent -- it cut each stroke at exactly the byte count
    the composition level asked for, wherever that landed -- so a plan that
    named an odd length produced a program the VM reports as `truncated`.

    This is parity, not a new liberty: the same rule the AR arm has had since
    the halt-symbol artifact, applied where the planner decides its lengths.
    An unknown opcode stops the walk for the same reason `VM.run` breaks there
    -- every later byte is dead, so keeping them would change the trace.
    """
    pc = 0
    while pc < len(program):
        try:
            spec = spec_for(program[pc])
        except UnknownOpcode:
            break
        if pc + spec.size > len(program):
            break
        pc += spec.size
    return program[:pc]


def summaries(program: bytes, max_strokes: int | None = None) -> list[Summary]:
    return [Summary.of(s) for s in split(program, max_strokes)]


def summary_bytes(program: bytes, max_strokes: int | None = None) -> bytes:
    """Every summary concatenated -- the composition level's whole input."""
    return b"".join(s.to_bytes() for s in summaries(program, max_strokes))


def stroke_count(program: bytes) -> int:
    return len(split(program))


#: Sanity: `SUMMARY_BYTES` and `Summary`'s field count are one number. Asserted
#: rather than derived, because the tensor layout in `dm/models/planner.py`
#: reads `SUMMARY_BYTES` and a mismatch would show up as a silent reshape.
assert len(Summary._fields) == SUMMARY_BYTES
assert all(len(s.operands) + 1 == s.size for s in SPECS.values())
