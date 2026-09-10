"""D4 and integer translation: the group `XFORM` puts in the ISA.

**The group is forced by measurement, not chosen for elegance.** `REPEAT n dx
dy` is lossless only under an exact translation, and rounding commutes with
integer translation and with nothing else. Against Tabler's 3.88% repeat ceiling
(`scripts/repeat_oracle.py`): integer translate / mirror / 90° rotate preserve it
exactly, ×1.07 scale destroys 57% ±3% of it, ±1 jitter 99.6% (re-measured over
24 sub-pixel rounding phases on 2026-08-13; the 77% and 65% that stood here were
single phases -- `docs/bezier.md` §5.1). So the transform tier is
**D4 ⋉ integer translation** -- eight symmetries and a shift, nothing continuous
-- and anything larger reopens the quantisation confound the 8-bit ISA exists to
eliminate (`dm/data/augment.py` carries the same argument for augmentation).

An element is `(code, dx, dy)` where `code` packs D4 into one byte: bit 0 is
mirror, bits 1-2 are quarter turns. That is `Kind.XF`, and the packing is the
wire format -- the VM, the augmenter, the corpus generator and `port/src/dm_vm.c`
all read it, so it lives here rather than in any one of them.

**The action is `dm.data.augment.Affine.point`'s, exactly.** A quarter turn is
`(x, y) -> (y, CANVAS - 1 - x)` and a mirror is `x -> CANVAS - 1 - x`, both of
which map the canvas onto itself with integer arithmetic and no half-pixel
centre. Sharing the convention is what lets an augmented L2 program and an
`XFORM` program mean the same thing; two conventions that agreed on seven of the
eight elements would be a fault nothing in the pipeline could see.

Composition is a table lookup and an add. No multiply and no division, which is
the property that makes the device cost a constant rather than a per-point
computation.
"""

from __future__ import annotations

from typing import NamedTuple

from .spec import CANVAS, ISAError

#: Elements of D4: four rotations, each with and without a mirror.
D4_ORDER = 8


class D4(NamedTuple):
    """One symmetry of the square, as the ISA spells it.

    `turns` is applied first, then `mirror` -- the order `Affine.point` uses.
    Swapping them is a different (and still valid) parameterisation of the same
    eight elements, which is exactly why the order has to be written down once
    and shared rather than re-derived per call site.
    """

    turns: int = 0
    mirror: bool = False

    @property
    def code(self) -> int:
        return (self.turns % 4) << 1 | int(self.mirror)

    @classmethod
    def of(cls, code: int) -> D4:
        if not 0 <= code < D4_ORDER:
            raise ISAError(f"transform code {code} outside D4 (0..7)")
        return cls((code >> 1) & 3, bool(code & 1))

    def point(self, x: int, y: int) -> tuple[int, int]:
        """The action on a canvas coordinate. Integer in, integer out."""
        for _ in range(self.turns % 4):
            x, y = y, CANVAS - 1 - x
        return (CANVAS - 1 - x, y) if self.mirror else (x, y)

    def linear(self, x: int, y: int) -> tuple[int, int]:
        """The action on a *vector* -- a `REPEAT` step, not a position.

        The same distinction `dm/data/augment.py` makes: a delta takes the
        linear part and never the translation, or a repeated body walks off in
        the wrong direction under a rotation.
        """
        for _ in range(self.turns % 4):
            x, y = y, -x
        return (-x, y) if self.mirror else (x, y)

    def then(self, other: D4) -> D4:
        """`other ∘ self` -- self applied first. An 8×8 table, computed once."""
        return _COMPOSE[self.code][other.code]

    def inverse(self) -> D4:
        return _INVERSE[self.code]

    def conjugate(self, outer: D4) -> D4:
        """`outer ∘ self ∘ outer^-1` -- this element, seen from `outer`'s frame.

        What a `REPEATX` step operand does under an augmentation: the step is
        applied per iteration to a body whose coordinates the augmentation has
        already moved, so it has to be re-expressed in the new frame. Non-trivial
        precisely because D4 is non-abelian -- a rotation conjugated by a mirror
        is the *opposite* rotation.
        """
        return outer.inverse().then(self).then(outer)


def _compose_by_action(a: D4, b: D4) -> D4:
    """Which element acts as `b` after `a`, found by trying all eight.

    Derived from the action rather than from a hand-written multiplication
    table, because a hand-written table for a non-abelian group is a
    transcription error waiting to happen and this one is checked against every
    canvas corner. `test_transform.py` pins the result against brute force over
    the whole canvas.
    """
    probes = ((0, 0), (1, 0), (0, 1), (CANVAS - 1, 0), (0, CANVAS - 1))
    want = [b.point(*a.point(*p)) for p in probes]
    for code in range(D4_ORDER):
        candidate = D4.of(code)
        if [candidate.point(*p) for p in probes] == want:
            return candidate
    raise ISAError("D4 is not closed, which cannot happen")  # pragma: no cover


_ALL = tuple(D4.of(code) for code in range(D4_ORDER))
_COMPOSE = tuple(tuple(_compose_by_action(a, b) for b in _ALL) for a in _ALL)
_INVERSE = tuple(
    next(b for b in _ALL if a.then(b) == D4()) for a in _ALL
)


class Transform(NamedTuple):
    """A D4 element and a translation: `p -> d4(p) + (dx, dy)`.

    The affine part is kept separate from the linear one because they compose
    differently, and the composition rule is the whole reason this is a type
    rather than three ints.
    """

    d4: D4 = D4()
    dx: int = 0
    dy: int = 0

    @classmethod
    def of(cls, code: int, dx: int, dy: int) -> Transform:
        return cls(D4.of(code), dx, dy)

    @property
    def is_identity(self) -> bool:
        return self.d4 == D4() and not (self.dx or self.dy)

    def point(self, x: int, y: int) -> tuple[int, int]:
        x, y = self.d4.point(x, y)
        return x + self.dx, y + self.dy

    def then(self, other: Transform) -> Transform:
        """`other ∘ self`: this transform first, then `other`.

        `other(self(p)) = B(A p + t_A) + t_B = (BA) p + (B t_A + t_B)`, so the
        translation takes the *linear* part of `other` -- the same rule a delta
        operand follows, for the same reason.
        """
        dx, dy = other.d4.linear(self.dx, self.dy)
        return Transform(self.d4.then(other.d4), dx + other.dx, dy + other.dy)

    def inverse(self) -> Transform:
        """The map that undoes this one. `(A, t)^-1 = (A^-1, -A^-1 t)`."""
        d4 = self.d4.inverse()
        dx, dy = d4.linear(self.dx, self.dy)
        return Transform(d4, -dx, -dy)

    def conjugate(self, outer: Transform) -> Transform:
        """`outer ∘ self ∘ outer^-1` -- this scope, seen from `outer`'s frame.

        **This is what an augmentation does to an `XFORM` instruction**, and it
        is not optional: D4 is non-abelian, so a program transformed by `A`
        whose inner scopes were left alone draws something else entirely.

        Writing `A(p) = L p + a`, the result is
        `d4' = L d4 L^-1` and `t' = L t + a - d4'(a)` -- so the translation
        picks up a term from `outer`'s *own* translation that a plain vector
        would not. That is the difference between `XFORM`'s `dx`/`dy`, which are
        part of an affine map, and `REPEATX`'s, which are a per-iteration
        displacement and take `linear` alone. Both wear `Kind.DELTA`; only one
        of them is a vector.
        """
        return outer.inverse().then(self).then(outer)

    def power(self, k: int) -> Transform:
        """This transform applied `k` times, which is what `REPEATX` iteration
        `k` runs under.

        By repeated composition rather than by a closed form: `(A, t)^k` has a
        geometric-series translation that is only exact when accumulated the way
        the C port will accumulate it, and a formula that agreed in Python and
        drifted in C is the whole failure mode `scripts/conformance.py` exists
        to catch.
        """
        out = Transform()
        for _ in range(max(0, k)):
            out = out.then(self)
        return out


IDENTITY = Transform()


def compose_all(stack: list[Transform]) -> Transform:
    """The whole scope stack as one transform: **innermost acts first.**

    `stack[0]` is the outermost open scope and acts *last*, which is every
    graphics stack's convention -- SVG, PostScript, OpenGL -- and the only one
    under which "an `XFORM` transforms everything inside it" is true of nested
    scopes rather than only of coordinates. D4 is non-abelian, so the other fold
    is a different drawing and not a different spelling of this one.

    It is also what makes `dm/isa/unroll.py` composable. Unrolling works on
    *bytes*: a scope's body is expanded first and the enclosing transform is
    then applied to the result, so the flattener's natural order has to be the
    interpreter's or the two would disagree on exactly the nested programs the
    transform tier exists to express.

    `place()` applies the single `(A, t)` this returns, so depth costs nothing
    per point -- it is paid once, when the stack changes. That is what makes
    §2.4's prediction ("~nothing to the decode term, a small constant to the
    per-point term") checkable rather than hopeful.
    """
    out = IDENTITY
    for item in reversed(stack):
        out = out.then(item)
    return out
