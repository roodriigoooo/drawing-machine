"""Integer-affine augmentation: more samples, and not one repeat lost.

Tier C is 4,613 programs and has **no budget at which both convergence guards
pass** -- at 800 steps every arm is short of its minimum by 14-103 bits/1k, and
by 1,000 some are past it. The corpus cannot support an asymptote, so the cheap
lever is to raise the effective sample count.

**The group is integer affine, and that is not a stylistic choice.** `REPEAT
n dx dy` is lossless only under an exact translation, and rounding commutes with
integer translation and with nothing else. Measured against Tabler's 3.88%
oracle ceiling (`scripts/repeat_oracle.py`):

    integer translate / mirror / rotate 90   3.97%   preserved exactly
    scale x1.07                              1.70%   destroys 57% +-3% of it
    jitter +-1                               0.02%   destroys 99.6% of it

**The two destructive rows were re-measured on 2026-08-13 and the old figures
(77% and 65%) were single *rounding phases*.** A scale is a non-integer map
followed by rounding, and where the rounding lands is a parameter the operation
never names; on a grid-authored corpus the phases resonate rather than average,
so the same x1.07 destroys 81% about the origin and 65% about each drawing's own
centre. The numbers above are means over 24 low-discrepancy sub-pixel phases
(`docs/bezier.md` 5.1). **The conclusion is unchanged and is now stronger**: the
group is D4 <| integer translation, and the jitter row is far worse than it
looked.

The last two rows are the SVG-Icons8 failure reproduced on demand: a pipeline
that transforms in floats and *then* rounds turns authored repeats into
near-repeats, which a lossless `REPEAT` cannot emit. So a deforming policy would
destroy exactly the structure claim 2 exists to measure.

Operands are transformed **by `Kind`, never positionally**. A `CIRCLE`'s radius
and a `WIDTH`'s stroke width are lengths, invariant under this group; a
`REPEAT`'s `dx`/`dy` are a *vector*, so they take the linear part of the
transform and not the translation. Treating every operand as a coordinate would
silently corrupt all three.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

from ..isa.asm import assemble, parse
from ..isa.spec import BY_MNEMONIC, CANVAS, Kind
from ..isa.transform import D4, Transform


@dataclass(frozen=True)
class Affine:
    """An integer affine map on the canvas: rotate, then mirror, then translate."""

    dx: int = 0
    dy: int = 0
    mirror: bool = False
    quarter_turns: int = 0

    def linear(self, x: int, y: int) -> tuple[int, int]:
        """The rotation/mirror part, which is what a DELTA operand takes."""
        for _ in range(self.quarter_turns % 4):
            x, y = y, -x
        return (-x, y) if self.mirror else (x, y)

    def point(self, x: int, y: int) -> tuple[int, int]:
        # Rotation about the canvas centre keeps the result on the canvas, so
        # the only rejections come from the translation.
        for _ in range(self.quarter_turns % 4):
            x, y = y, CANVAS - 1 - x
        if self.mirror:
            x = CANVAS - 1 - x
        return x + self.dx, y + self.dy

    @property
    def is_identity(self) -> bool:
        return not (self.dx or self.dy or self.mirror or self.quarter_turns % 4)

    @classmethod
    def of(cls, transform: Transform) -> Affine:
        """The ISA's spelling as this one. The inverse of `as_transform`."""
        return cls(transform.dx, transform.dy,
                   transform.d4.mirror, transform.d4.turns)

    def as_transform(self) -> Transform:
        """The same element, spelled the way the ISA spells it.

        `Affine` and `dm.isa.transform.Transform` are two spellings of one
        group -- this class predates the opcode. Converting in one place is what
        keeps them from drifting into two groups that agree on seven elements;
        `tests/test_transform.py` pins the action of both on all eight.
        """
        return Transform(D4(self.quarter_turns % 4, self.mirror), self.dx, self.dy)


def apply(program: bytes, transform: Affine) -> bytes | None:
    """Transform every coordinate. Returns None if the drawing leaves the canvas.

    Rejected rather than clamped: clamping is a deformation, and a deformation
    is what this module exists to avoid.

    **`XFORM` is rewritten as a whole instruction, not operand by operand.** A
    scope's three operands are one affine map, and transforming a program by `A`
    has to conjugate that map -- `A S A^-1`, `Transform.conjugate` -- or the
    inner scope keeps acting in the *old* frame and the augmented program draws
    something else. `REPEATX` does not need the coupled rule: its `dx`/`dy` are
    a per-iteration displacement rather than part of a map, so the `XF` operand
    conjugates on its own and the deltas take `linear`, which is what every
    other delta in the ISA takes. The derivation is in `Transform.conjugate`.
    """
    outer = transform.as_transform()
    lines: list[str] = []
    for instr in parse(program):
        spec = BY_MNEMONIC[instr.mnemonic]
        args = list(instr.args)

        if instr.mnemonic in ("XFORM", "REPEATX"):
            # Both carry a transform as *one* thing, so both are rewritten as one
            # thing. `REPEATX` keeps its count, which no transform touches.
            head = args[:-3]
            scope = Transform.of(*args[-3:]).conjugate(outer)
            if not (-128 <= scope.dx <= 127 and -128 <= scope.dy <= 127):
                return None      # the conjugated shift has no i8 spelling
            fields = [*head, scope.d4.code, scope.dx, scope.dy]
            lines.append(" ".join([instr.mnemonic, *map(str, fields)]))
            continue

        out: list[int] = []
        index = 0
        while index < len(args):
            kind = spec.operands[index]
            if kind is Kind.COORD:
                x, y = transform.point(args[index], args[index + 1])
                if not (0 <= x < CANVAS and 0 <= y < CANVAS):
                    return None
                out += [x, y]
                index += 2
            elif kind is Kind.DELTA:
                dx, dy = transform.linear(args[index], args[index + 1])
                if not (-128 <= dx <= 127 and -128 <= dy <= 127):
                    return None
                out += [dx, dy]
                index += 2
            else:                      # COUNT, SCALAR, ID: invariant here
                out.append(args[index])
                index += 1
        lines.append(" ".join([instr.mnemonic, *map(str, out)]))
    return assemble("\n".join(lines))


def policies(
    shifts: tuple[int, ...] = (-8, 8),
    mirror: bool = True,
    quarter_turns: tuple[int, ...] = (0,),
) -> list[Affine]:
    """The identity first, then every combination of the requested generators."""
    out = [Affine()]
    for dx, dy, flip, turns in product(shifts + (0,), shifts + (0,),
                                       (False, True) if mirror else (False,),
                                       quarter_turns):
        candidate = Affine(dx, dy, flip, turns)
        if not candidate.is_identity:
            out.append(candidate)
    return out


def expand(programs: list[bytes], transforms: list[Affine]) -> list[bytes]:
    """Every program under every transform, dropping those that leave the canvas.

    Deterministic and order-stable, so the corpus is a function of its policy and
    a run record naming the policy names the data.
    """
    out: list[bytes] = []
    for program in programs:
        for transform in transforms:
            augmented = apply(program, transform)
            if augmented:
                out.append(augmented)
    return out
