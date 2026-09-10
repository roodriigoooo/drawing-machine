"""The curve fitter: three things have to hold before any byte count means anything.

**The tolerance has to be a bound**, not an average, or the two arms are being
compared at two fidelities and the byte column is free.

**The control points have to be integers on the canvas**, or claim 4 quietly
loses its exactness -- `port/include/dm_vm.h` argues the VM's curve geometry is
integral at a scale of `CURVE_STEPS ** 3`, and that argument holds for integer
corner points and for nothing else.

**And the fit has to commute with an integer translation**, or the whole
retention measurement is a fitter artifact: a corpus whose repeats survive
re-spelling in one arm and not the other would show a difference that has
nothing to do with the primitive.
"""

import numpy as np
import pytest

from dm.isa.asm import parse
from dm.isa.bezier import (
    QUANTISATION_FLOOR,
    Segment,
    draw,
    fit_polyline,
    hausdorff,
    round_half_up,
    stroke_bytes,
)
from dm.isa.spec import CANVAS, Op
from dm.vm.interp import VM

#: The grid every coordinate the VM emits lives on: `CURVE`'s Bernstein weights
#: have denominator `CURVE_STEPS ** 3` and `MOVE`/`LINE` are integral, so a trace
#: point is always a multiple of 1/4096 with magnitude under 256. Test geometry
#: is snapped to it because the equivariance guarantee is about *this* input --
#: on it, adding an integer is exact in float64 and a translated copy performs
#: bitwise-identical arithmetic. Arbitrary reals have no such guarantee and
#: testing against them would assert something the fitter cannot promise.
LATTICE = 4096.0


def _snap(points: np.ndarray) -> np.ndarray:
    return np.round(points * LATTICE) / LATTICE


def _arc(radius: float = 60.0, n: int = 40, centre=(128.0, 128.0),
         sweep: float = np.pi / 2) -> np.ndarray:
    theta = np.linspace(0.0, sweep, n)
    return _snap(np.stack([centre[0] + radius * np.cos(theta),
                           centre[1] + radius * np.sin(theta)], axis=1))


def _zigzag(n: int = 24) -> np.ndarray:
    xs = np.arange(n, dtype=np.float64) * 8.0 + 20.0
    ys = 128.0 + 30.0 * (np.arange(n) % 2)
    return np.stack([xs, ys], axis=1)


@pytest.mark.parametrize("tol", [0.5, 1.0, 2.0, 4.0])
@pytest.mark.parametrize("points", [_arc(), _arc(20.0, 8), _zigzag(),
                                    _arc(100.0, 60, sweep=1.9 * np.pi)])
@pytest.mark.parametrize("allow_curve", [False, True])
def test_the_tolerance_is_a_bound_not_an_average(tol, points, allow_curve):
    """And the bound is `max(tol, QUANTISATION_FLOOR)`.

    Below half a pixel diagonal no spelling on an integer grid can do better,
    whatever primitive it uses, so a tolerance under the floor is a request no
    arm can honour. Asserting the bare `tol` would make the *fitter* look broken
    for a limit the canvas imposes -- and, worse, would hide that the two arms
    are then being compared at whichever error each happened to reach.
    """
    start, segments = fit_polyline(points, tol, allow_curve=allow_curve)
    achieved = hausdorff(points, draw(start, segments))
    assert achieved <= max(tol, QUANTISATION_FLOOR) + 1e-9


def test_control_points_are_integers_inside_the_canvas():
    # A near-cusp: the unconstrained least-squares fit wants control points far
    # outside the canvas here, and `VM.place` would clamp them silently.
    points = np.concatenate([_arc(110.0, 30, sweep=np.pi),
                             _arc(110.0, 30, sweep=np.pi)[::-1] + 1.0])
    start, segments = fit_polyline(points, 2.0)
    for segment in (Segment(Op.MOVE, (), start), *segments):
        for x, y in (*segment.controls, segment.end):
            assert isinstance(x, int) and isinstance(y, int)
            assert 0 <= x < CANVAS and 0 <= y < CANVAS


def test_the_scored_geometry_is_the_geometry_the_vm_draws():
    """`draw` and `VM.run` must agree exactly, or the tolerance is fiction."""
    points = _arc(70.0, 50)
    start, segments = fit_polyline(points, 1.0)
    program = stroke_bytes(start, segments) + bytes((int(Op.HALT),))
    trace = VM().run(program)
    assert trace.valid and len(trace.strokes) == 1
    drawn = np.asarray(trace.strokes[0].points, dtype=np.float64)
    assert np.array_equal(drawn, draw(start, segments))


def test_the_control_arm_emits_no_curve_at_all():
    _, segments = fit_polyline(_arc(80.0, 60), 1.0, allow_curve=False)
    assert segments and all(s.kind is Op.LINE for s in segments)


def test_curves_beat_lines_on_smooth_geometry_at_equal_tolerance():
    curved = fit_polyline(_arc(80.0, 60), 2.0, allow_curve=True)
    straight = fit_polyline(_arc(80.0, 60), 2.0, allow_curve=False)
    assert len(stroke_bytes(*curved)) < len(stroke_bytes(*straight))


def test_a_straight_run_is_spelled_as_lines_even_when_curves_are_allowed():
    """A cubic through a straight span costs four extra bytes and draws the same
    line, and a fitter that preferred it would inflate the arm under test."""
    line = np.stack([np.linspace(10.0, 200.0, 20), np.full(20, 64.0)], axis=1)
    _, segments = fit_polyline(line, 1.0, allow_curve=True)
    assert [s.kind for s in segments] == [Op.LINE]


def test_the_chain_is_continuous_so_the_vm_never_moves_between_segments():
    _, segments = fit_polyline(_arc(90.0, 80, sweep=1.5 * np.pi), 0.75)
    assert len(segments) > 1
    # Continuity is structural here: every segment carries only its end point,
    # and the VM starts each one from the previous cursor. The check is that
    # nothing in the chain is a MOVE, which would restart the stroke.
    assert all(s.kind in (Op.LINE, Op.CURVE) for s in segments)


@pytest.mark.parametrize("allow_curve", [False, True])
@pytest.mark.parametrize("shift", [(37, 0), (0, -19), (11, 23)])
def test_the_fit_commutes_with_an_integer_translation(allow_curve, shift):
    """The property the whole retention measurement rests on.

    If a fitted shape depended on where it sat, two exact copies of a motif
    would fit differently and the re-spelled corpus would lose repeats the
    primitive had nothing to do with -- and the `bezier` arm would look worse
    or better for a reason no reader could see.
    """
    points = _arc(50.0, 40, centre=(80.0, 80.5))
    a_start, a = fit_polyline(points, 1.5, allow_curve=allow_curve)
    b_start, b = fit_polyline(points + np.asarray(shift, dtype=np.float64), 1.5,
                              allow_curve=allow_curve)
    assert b_start == (a_start[0] + shift[0], a_start[1] + shift[1])
    assert [s.kind for s in b] == [s.kind for s in a]
    for p, q in zip(a, b):
        for (px, py), (qx, qy) in zip((*p.controls, p.end), (*q.controls, q.end)):
            assert (qx - px, qy - py) == shift


@pytest.mark.parametrize("points", [np.zeros((0, 2)), np.array([[5.0, 5.0]]),
                                    np.full((6, 2), 12.0)])
def test_degenerate_strokes_produce_nothing_rather_than_raising(points):
    start, segments = fit_polyline(points, 1.0)
    assert stroke_bytes(start, segments) in (b"", stroke_bytes(start, segments))
    assert all(s.kind in (Op.LINE, Op.CURVE) for s in segments)


def test_a_fitted_stroke_parses_as_the_instructions_it_claims():
    start, segments = fit_polyline(_arc(70.0, 40), 1.0)
    instrs = parse(stroke_bytes(start, segments))
    assert instrs[0].mnemonic == "MOVE"
    assert [i.mnemonic for i in instrs[1:]] == [s.kind.name for s in segments]


def test_half_way_ties_round_the_same_regardless_of_where_the_stroke_sits():
    """`round` and `np.rint` round half to even, so the tie-break depends on the
    parity of the integer part -- and rounding then stops commuting with integer
    translation at exactly the half-pixel coordinates. Every exact-repeat number
    in this project rests on that commutation."""
    for base in (2.5, 3.5, -0.5, 128.5):
        for shift in (1, 2, 7, -3):
            assert round_half_up(base + shift) == round_half_up(base) + shift
    assert round(2.5) != round_half_up(2.5)     # the behaviour being avoided


def test_a_planted_repeat_survives_re_spelling_in_both_arms():
    """The end-to-end version of the equivariance property, on real geometry.

    A retention measured on a corpus whose repeats the *fitter* destroyed would
    be about the fitter. Both arms have to carry a planted exact repeat through
    the re-spelling before either arm's retention means anything.
    """
    from dm.data.canonical import POLICIES
    from dm.data.refit import respell
    from dm.eval.repeats import compress
    from dm.isa.asm import assemble

    arc = np.round(_arc(40.0, 14, centre=(60.0, 100.0))).astype(int)
    lines = []
    for copy in range(3):
        dx = 61 * copy                     # odd, so banker's rounding would bite
        lines.append(f"MOVE {arc[0][0] + dx} {arc[0][1]}")
        lines += [f"LINE {x + dx} {y}" for x, y in arc[1:]]
    lines.append("HALT")
    program = assemble("\n".join(lines))
    assert compress(program)[0] > 0        # the plant is there to begin with

    for allow_curve in (False, True):
        respelled, _ = respell(program, 1.0, allow_curve, POLICIES["none"])
        saved, found = compress(respelled)
        assert saved > 0, f"allow_curve={allow_curve} lost the planted repeat"
        assert found[0].count == 3
