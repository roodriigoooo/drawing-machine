"""Cubic control points as the geometry primitive, fitted to source polylines.

`Op.CURVE` has existed since ISA v1 and only Tabler ever emits it, because
Tabler's source is already cubics. Every other corpus arrives as a polyline and
leaves as `MOVE`/`LINE`, so the project's whole geometry story is "RDP, then
three bytes per surviving vertex". `docs/direction.md` §7.3 asks what changes if
a stroke is spelled as control points instead, and it is worth being precise
about which of the three answers this module is built to decide.

1. **Length.** Fewer bytes per stroke at equal fidelity. A compression question,
   measurable with no training, and the weakest of the three.
2. **Commutation with the transform group.** An affine image of a Bézier is the
   Bézier of the transformed control points, so `XFORM` and `REPEATX` would act
   on a handful of numbers per stroke rather than on all of them.
3. **Whether *scale* can join the transform group at all.** Scale is excluded by
   measurement, not by taste: ×1.07 destroys most of Tabler's repeat ceiling
   because a repeat needs every coordinate of a body to round the same way, and
   `(1 - p)^n` collapses as `n` grows. Shrink `n` and the same `p` gives a very
   different answer. That was the prediction on record and the reason this file
   exists. **Measured 2026-08-13 (`docs/bezier.md` §5): it fires and it is not
   enough.** A control-point corpus retains 61.5% ±9.8% where a matched polyline
   retains 42.3% ±3.8%, in 24 of 24 paired rounding phases -- and 61.5% is not
   lossless, on a ceiling 38% smaller, so scale stays out. What survives is (1):
   -13.4% bytes at matched fidelity on icons, and a null on QuickDraw.

**Two properties are load-bearing and both are enforced here rather than
assumed.**

**Control points are integers on the canvas, and the error is measured through
the VM's own flattener.** `port/include/dm_vm.h` argues that a `CURVE`'s geometry
is exactly integral at a scale of `CURVE_STEPS ** 3`, which is what lets
`scripts/conformance.py` compare the C port with `==` and carry no tolerance
anywhere. That argument holds for *integer* corner points and for nothing else,
so a fitter that emitted fractional controls would silently cost the project its
one exact device claim. It also means the honest reconstruction error is the
distance to the 16-segment polyline the device draws, not to the ideal cubic --
so `fit_polyline` scores every candidate through `dm.vm.interp.flatten_cubic`,
the same function `VM.run` calls.

**A control point outside the canvas is a deformation.** `VM.place` clamps to
`[0, CANVAS)`, and clamping a control point moves the curve. Rather than let
that happen silently the fit *includes* the clamp before scoring, so a segment
whose ideal controls do not fit is simply a segment that fails its tolerance and
gets split. The corollary is that this fitter can always terminate: two points
are a `LINE`, and a `LINE` between quantised endpoints is what the polyline
spelling would have emitted anyway.
"""

from __future__ import annotations

import math
from typing import NamedTuple

import numpy as np

from ..vm.interp import CURVE_STEPS, flatten_cubic
from .spec import CANVAS, Op

#: Samples used to *re-parameterise* a candidate fit. Higher than `CURVE_STEPS`
#: on purpose: the flattening is the specification of what gets drawn, while the
#: projection that assigns each source point a curve parameter is an internal
#: search step and gains from a finer grid. Keeping the two separate is what
#: stops a search detail from changing the geometry.
PROJECT_STEPS = 64

#: Re-parameterisation passes. Schneider's algorithm uses Newton-Raphson on the
#: analytic derivative; projecting onto the flattened curve instead is robust
#: where the derivative vanishes (a cusp, or a stroke that doubles back) and
#: costs one more pass to reach the same place. Two is where the error stops
#: moving on this corpus.
REPARAM_PASSES = 2

#: The smallest error any spelling on an integer grid can have: half the
#: diagonal of a pixel. A tolerance below this is not achievable by *any*
#: primitive, so `fit_polyline`'s guarantee is `error <= max(tol, this)` and the
#: two arms are compared on their **measured** error rather than on `tol`.
QUANTISATION_FLOOR = 2.0 ** -0.5


def round_half_up(value: float) -> int:
    """Round with ties away from the lower integer, **never** Python's `round`.

    This looks like a style preference and it is a correctness requirement.
    Python's `round` and `numpy.rint` both round half to *even*, so the tie-break
    depends on the parity of the integer part -- `round(2.5) == 2` and
    `round(3.5) == 4`. That makes rounding **stop commuting with integer
    translation** at exactly the half-pixel coordinates:
    `round(x + k) != round(x) + k` whenever `frac(x) == 0.5` and `k` is odd.

    Every exact-repeat result in this project rests on rounding commuting with
    integer translation (`dm/data/augment.py`, `docs/tier-c.md`). A fitter using
    banker's rounding would therefore spell two exact copies of a motif
    differently depending on where they sat, destroying repeats the primitive
    had nothing to do with -- which is a difference between the arms that no
    reader could see. `floor(x + 0.5)` has no parity term and commutes exactly.
    """
    return math.floor(value + 0.5)


class Segment(NamedTuple):
    """One emitted instruction's worth of geometry, with integer operands.

    `kind` is `Op.LINE` or `Op.CURVE`; `controls` is empty for a line and the
    two interior control points for a cubic. The start point is implicit -- it
    is the previous segment's end, exactly as the VM reads it -- which is what
    keeps a fitted stroke a *chain* rather than a list of independent pieces
    that could drift apart by a rounding unit each.
    """

    kind: Op
    controls: tuple[tuple[int, int], ...]
    end: tuple[int, int]

    @property
    def numbers(self) -> int:
        """Operand count. The `n` in `(1 - p)^n`, per instruction."""
        return 2 * (len(self.controls) + 1)

    def to_bytes(self) -> bytes:
        flat: list[int] = []
        for point in (*self.controls, self.end):
            flat.extend(point)
        return bytes((int(self.kind), *flat))


# ---------------------------------------------------------------------------
# geometry helpers


def _quantise(point) -> tuple[int, int]:
    x, y = point
    return (
        min(CANVAS - 1, max(0, round_half_up(float(x)))),
        min(CANVAS - 1, max(0, round_half_up(float(y)))),
    )


def _place(anchor: tuple[int, int], offset) -> tuple[int, int]:
    """`anchor` plus a rounded offset, clamped. Integer-add, never float-add.

    The whole fit works relative to an integer anchor so that a translated copy
    of a stroke performs bitwise-identical arithmetic; re-absolutising through a
    float add would put the magnitude back in and undo it.
    """
    return (
        min(CANVAS - 1, max(0, anchor[0] + round_half_up(float(offset[0])))),
        min(CANVAS - 1, max(0, anchor[1] + round_half_up(float(offset[1])))),
    )


def dedup(points: np.ndarray) -> np.ndarray:
    """Drop consecutive duplicates, which carry no direction and break tangents."""
    pts = np.asarray(points, dtype=np.float64)
    if len(pts) < 2:
        return pts
    keep = np.insert(np.any(np.diff(pts, axis=0) != 0.0, axis=1), 0, True)
    return pts[keep]


def _point_to_polyline(points: np.ndarray, poly: np.ndarray) -> np.ndarray:
    """Distance from every point to the nearest point *on* `poly`'s segments.

    Segment-wise rather than vertex-wise: a 16-step flattening of a gentle curve
    has vertices up to several pixels apart, and scoring against vertices alone
    would report an error that the drawn geometry does not have. The projection
    is the standard clamped parameter along each segment.
    """
    if len(poly) == 1:
        return np.linalg.norm(points - poly[0], axis=1)
    a, b = poly[:-1], poly[1:]
    seg = b - a                                     # (m, 2)
    denom = np.einsum("ij,ij->i", seg, seg)
    denom = np.where(denom > 0.0, denom, 1.0)
    rel = points[:, None, :] - a[None, :, :]        # (n, m, 2)
    t = np.clip(np.einsum("nmj,mj->nm", rel, seg) / denom, 0.0, 1.0)
    # `rel - t * seg`, not `points - (a + t * seg)`. The two are equal in exact
    # arithmetic and not in float64: the second form adds an absolute coordinate
    # to a small offset, so its last bits depend on where the stroke sits, and a
    # *tie* in the split pivot below then breaks differently for two translated
    # copies of one shape. Symmetric shapes tie exactly and are common in icons,
    # so this is not a corner case -- it is how the arm under test would have
    # quietly lost planted repeats. `rel` is a difference of like magnitudes and
    # is exact.
    return np.linalg.norm(rel - t[..., None] * seg[None, :, :], axis=2).min(axis=1)


def hausdorff(a: np.ndarray, b: np.ndarray) -> float:
    """Symmetric worst-case distance between two polylines, in canvas pixels.

    One-sided would be the wrong metric and wrong in the flattering direction: a
    curve is free to bulge between two source points and a source-to-curve
    maximum never sees it. Both directions are taken because the tolerance this
    feeds is what the byte comparison is *matched on*, and a fit scored by a
    lenient metric buys its bytes from the metric rather than from the primitive.
    """
    if len(a) == 0 or len(b) == 0:
        return float("inf")
    return max(
        float(_point_to_polyline(a, b).max()),
        float(_point_to_polyline(b, a).max()),
    )


def draw(start: tuple[int, int], segments: list[Segment],
         steps: int = CURVE_STEPS) -> np.ndarray:
    """The polyline the VM would draw for this chain. The scoring reference."""
    out: list[tuple[float, float]] = [(float(start[0]), float(start[1]))]
    cursor = out[0]
    for segment in segments:
        end = (float(segment.end[0]), float(segment.end[1]))
        if segment.kind is Op.LINE:
            out.append(end)
        else:
            c1, c2 = segment.controls
            out.extend(
                flatten_cubic(cursor, (float(c1[0]), float(c1[1])),
                              (float(c2[0]), float(c2[1])), end, steps)
            )
        cursor = end
    return np.asarray(out, dtype=np.float64)


# ---------------------------------------------------------------------------
# the fit


def _tangent(points: np.ndarray, at_start: bool) -> np.ndarray:
    """Unit tangent at an end of the span, averaged over a short run.

    A single difference is the textbook choice and it is noisy on quantised
    input, where the first two vertices can be one pixel apart in a direction
    the stroke does not go. Averaging the first few normalised differences costs
    nothing and stops one rounding unit from choosing the shape of a whole
    segment.
    """
    pts = points if at_start else points[::-1]
    limit = min(4, len(pts) - 1)
    acc = np.zeros(2, dtype=np.float64)
    for i in range(1, limit + 1):
        step = pts[i] - pts[0]
        norm = float(np.linalg.norm(step))
        if norm > 0.0:
            acc += step / norm
    norm = float(np.linalg.norm(acc))
    return acc / norm if norm > 0.0 else np.array([0.0, 0.0])


def _chord_parameters(points: np.ndarray) -> np.ndarray:
    step = np.linalg.norm(np.diff(points, axis=0), axis=1)
    walked = np.concatenate([[0.0], np.cumsum(step)])
    return walked / walked[-1] if walked[-1] > 0.0 else np.linspace(0.0, 1.0, len(points))


def _solve_alphas(local: np.ndarray, u: np.ndarray, p3: np.ndarray,
                  t1: np.ndarray, t2: np.ndarray) -> tuple[float, float]:
    """Least-squares control-point distances along the two end tangents.

    The classical normal equations for a cubic with both endpoints and both
    tangent *directions* fixed, leaving two scalars. Endpoints are fixed because
    they are shared with the neighbouring segments and with the quantised grid;
    tangents are fixed because a fit free to choose them produces chains that
    kink at every junction, which reads as reconstruction error the primitive
    did not cause.

    **Everything here is relative to the span's own start point**, which is not
    presentation. In absolute coordinates the residual carries a
    `p0 * (b0 + b1)` term, and multiplying a coordinate near 200 by a Bernstein
    weight rounds differently from multiplying one near 20 -- so the fitted
    alphas, and occasionally the quantised control points, would depend on
    *where the stroke sits*. Two exact copies of a motif would then be spelled
    differently and the re-spelled corpus would lose repeats the primitive had
    nothing to do with. In the local frame `p0` is exactly zero, the term
    vanishes, and the arithmetic a translated copy performs is the identical
    arithmetic.
    """
    u = np.clip(u, 0.0, 1.0)
    one = 1.0 - u
    b1 = 3.0 * u * one ** 2
    b2, b3 = 3.0 * u ** 2 * one, u ** 3
    a1 = t1[None, :] * b1[:, None]
    a2 = t2[None, :] * b2[:, None]
    # The `p0 * (b0 + b1)` term of the residual is identically zero in this
    # frame, which is the point of working in it.
    residual = local - p3[None, :] * (b2 + b3)[:, None]
    c11 = float(np.einsum("ij,ij->", a1, a1))
    c12 = float(np.einsum("ij,ij->", a1, a2))
    c22 = float(np.einsum("ij,ij->", a2, a2))
    x1 = float(np.einsum("ij,ij->", residual, a1))
    x2 = float(np.einsum("ij,ij->", residual, a2))
    det = c11 * c22 - c12 * c12
    chord = float(np.linalg.norm(p3))       # the start point is the origin here
    if abs(det) < 1e-12:
        # Wu and Barsky's fallback: a third of the chord in each direction. Used
        # when the two tangents are parallel, which is exactly a straight span --
        # where any alpha draws the same line.
        return chord / 3.0, chord / 3.0
    alpha1 = (x1 * c22 - c12 * x2) / det
    alpha2 = (c11 * x2 - x1 * c12) / det
    floor = 1e-6 * chord
    if alpha1 < floor or alpha2 < floor:
        return chord / 3.0, chord / 3.0
    return alpha1, alpha2


def _candidate(points: np.ndarray, start: tuple[int, int], end: tuple[int, int],
               u: np.ndarray) -> tuple[Segment, float, np.ndarray]:
    """One cubic at these parameters: the segment, its error, and new parameters.

    Returns the re-projection as well as the fit, because the caller iterates and
    the projection is most of the cost. Everything is scored *after* quantisation
    and clamping, so the error is the drawn one.
    """
    origin = np.asarray(start, dtype=np.float64)
    p3 = np.asarray(end, dtype=np.float64) - origin
    t1, t2 = _tangent(points, True), _tangent(points, False)
    alpha1, alpha2 = _solve_alphas(points - origin, u, p3, t1, t2)
    # Quantise the *offset* and add the integer start, rather than quantising
    # the absolute point: `floor(x + k + 0.5) == floor(x + 0.5) + k` for integer
    # `k` in exact arithmetic, and doing it as an integer add makes it true in
    # floating point too. Clamping is the one step that cannot be
    # translation-invariant, and it should not be: the canvas has edges.
    c1 = _place(start, alpha1 * t1)
    c2 = _place(end, alpha2 * t2)
    segment = Segment(Op.CURVE, (c1, c2), end)
    drawn = draw(start, [segment])
    error = hausdorff(points, drawn)
    fine = draw(start, [segment], steps=PROJECT_STEPS)
    # Re-parameterise by arclength position of each point's projection on the
    # candidate. Chord length on the *source* is a good first guess and a poor
    # final one, because the source is sampled unevenly by RDP.
    walked = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(fine, axis=0), axis=1))])
    if walked[-1] <= 0.0:
        return segment, error, u
    nearest = np.argmin(
        np.linalg.norm(points[:, None, :] - fine[None, :, :], axis=2), axis=1
    )
    return segment, error, walked[nearest] / walked[-1]


def _fit_span(points: np.ndarray, start: tuple[int, int], tol: float,
              depth: int, allow_curve: bool) -> list[Segment]:
    origin = np.asarray(start, dtype=np.float64)
    end = _place(start, points[-1] - origin)
    line = Segment(Op.LINE, (), end)
    line_error = hausdorff(points, draw(start, [line]))
    if len(points) == 2 or line_error <= tol:
        return [line]

    if not allow_curve:
        # The control arm. With cubics forbidden this recursion *is*
        # Ramer-Douglas-Peucker, with two deliberate differences from
        # `quickdraw._rdp`: the criterion is the same quantised, both-ways
        # Hausdorff distance the curve arm is scored by, and the endpoints are
        # the same integers. That is what makes the two arms differ in the
        # primitive and in nothing else -- a polyline simplified by a
        # perpendicular-distance rule against unquantised vertices would be a
        # second knob wearing the first one's name.
        if depth <= 0:
            return [Segment(Op.LINE, (), _place(start, p - origin)) for p in points[1:]]
        pivot = int(np.argmax(_point_to_polyline(points, draw(start, [line]))))
        pivot = min(max(pivot, 1), len(points) - 2)
        left = _fit_span(points[: pivot + 1], start, tol, depth - 1, allow_curve)
        return left + _fit_span(points[pivot:], left[-1].end, tol, depth - 1, allow_curve)

    u = _chord_parameters(points)
    best, best_error, u = _candidate(points, start, end, u)
    for _ in range(REPARAM_PASSES):
        if best_error <= tol:
            break
        segment, error, u = _candidate(points, start, end, u)
        if error < best_error:
            best, best_error = segment, error

    # A cubic that is no better than the straight line is four wasted bytes.
    if best_error <= tol and best_error < line_error:
        return [best]
    if line_error <= tol:
        return [line]

    if depth <= 0:
        # Out of splits: fall back to the polyline spelling of what is left,
        # which is bounded by quantisation alone and is what the other arm would
        # have emitted here anyway. Never the failed cubic -- a fitter that
        # returns geometry it knows misses tolerance turns a byte comparison
        # into an unmatched-fidelity one.
        return [Segment(Op.LINE, (), _quantise(p)) for p in points[1:]]

    # Split where the *fit* is worst, which is the point the two halves most
    # need on their shared boundary.
    pivot = int(np.argmax(_point_to_polyline(points, draw(start, [best]))))
    pivot = min(max(pivot, 1), len(points) - 2)
    left = _fit_span(points[: pivot + 1], start, tol, depth - 1, allow_curve)
    return left + _fit_span(points[pivot:], left[-1].end, tol, depth - 1, allow_curve)


def fit_polyline(points: np.ndarray, tol: float, allow_curve: bool = True,
                 max_depth: int = 32) -> tuple[tuple[int, int], list[Segment]]:
    """Fit one stroke. Returns its quantised start point and the chain after it.

    `tol` is a worst-case (Hausdorff) distance in canvas pixels between the
    source polyline and the polyline the VM draws, so it is the same kind of
    number `rdp_eps` is -- and, because `allow_curve=False` runs the identical
    recursion with cubics forbidden, it is the *same* number for both arms. The
    two spellings of a corpus can therefore be compared at matched fidelity
    rather than at matched knob settings, which is the only comparison a byte
    count means anything in.

    The guarantee is `error <= max(tol, QUANTISATION_FLOOR)`. Below half a
    pixel diagonal no spelling on an integer grid can do better, whatever
    primitive it uses, so a tolerance under that floor is a request no arm can
    honour and would silently favour whichever arm happened to miss by less.
    """
    pts = dedup(points)
    if len(pts) < 2:
        return ((0, 0), [])
    start = _quantise(pts[0])
    return start, _fit_span(pts, start, tol, max_depth, allow_curve)


def stroke_bytes(start: tuple[int, int], segments: list[Segment]) -> bytes:
    """`MOVE` to the start, then one instruction per segment."""
    if not segments:
        return b""
    out = bytearray((int(Op.MOVE), start[0], start[1]))
    for segment in segments:
        out += segment.to_bytes()
    return bytes(out)


__all__ = [
    "CURVE_STEPS",
    "PROJECT_STEPS",
    "QUANTISATION_FLOOR",
    "REPARAM_PASSES",
    "Segment",
    "dedup",
    "draw",
    "fit_polyline",
    "hausdorff",
    "round_half_up",
    "stroke_bytes",
]
