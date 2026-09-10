"""Local and global fidelity must not see each other.

The scale-separation test (`docs/scale-separation.md`) decides whether the
hierarchical design is built at all, and it decides it by comparing two numbers.
If either number responds to the other's perturbation, the comparison is
meaningless and the decision would be made on an artifact. That independence is
what is pinned here, on hand-built traces where the right answer is known.
"""

import math

import numpy as np
import pytest

from dm.eval.scale import (
    centroid,
    parts,
    resample,
    scale_fidelity,
    shape_distance,
)
from dm.isa.asm import assemble
from dm.vm.interp import VM, Disc, Stroke, Trace


def _trace(*strokes: list[tuple[float, float]]) -> Trace:
    return Trace(strokes=[Stroke(tuple(s), width=1) for s in strokes])


def _shift(trace: Trace, dx: float, dy: float) -> Trace:
    return Trace(
        strokes=[
            Stroke(tuple((x + dx, y + dy) for x, y in s.points), s.width)
            for s in trace.strokes
        ]
    )


SQUARE = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0), (0.0, 0.0)]
BLADE = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0)]


def test_identical_traces_score_zero_on_both():
    trace = _trace(SQUARE, [(p[0] + 60, p[1] + 60) for p in BLADE])
    report = scale_fidelity(trace, trace)

    assert report["global"] == pytest.approx(0.0)
    assert report["local"] == pytest.approx(0.0)
    assert report["matched"] == 2 and report["unmatched_truth"] == 0


def test_translating_the_whole_drawing_moves_global_and_not_local():
    """Layout is wrong, every stroke is the same shape. This is the case the
    hierarchy predicts for a model that learned local structure and not global,
    and a local metric that responds to it would report exactly the opposite
    conclusion."""
    truth = _trace(SQUARE, [(p[0] + 60, p[1] + 60) for p in BLADE])
    moved = _shift(truth, 25.0, -15.0)
    report = scale_fidelity(truth, moved)

    assert report["local"] == pytest.approx(0.0, abs=1e-9), "shape must be translation-blind"
    assert report["global"] > 20.0


def test_reshaping_in_place_moves_local_and_not_global():
    """The mirror image of the test above: same centroids, different shapes."""
    truth = _trace(SQUARE)
    # A different closed shape with the *same* centroid: a diamond.
    diamond = [(5.0, -1.0), (11.0, 5.0), (5.0, 11.0), (-1.0, 5.0), (5.0, -1.0)]
    report = scale_fidelity(truth, _trace(diamond))

    assert report["global"] == pytest.approx(0.0, abs=1e-9), "layout must be shape-blind"
    assert report["local"] > 1.0


def test_draw_direction_is_not_shape_error():
    """A stroke drawn end-to-start is the same stroke. Counting it would report
    a large local gap for a model that has learned shape perfectly."""
    forward = np.asarray(BLADE)
    assert shape_distance(forward, forward[::-1]) == pytest.approx(0.0, abs=1e-9)


def test_vertex_density_is_not_shape_error():
    """A 2-point line and a 40-point line over the same path are the same
    stroke; resampling is uniform in arclength so that stays true."""
    sparse = np.asarray([(0.0, 0.0), (30.0, 0.0)])
    dense = np.stack([np.linspace(0, 30, 40), np.zeros(40)], axis=1)
    assert shape_distance(sparse, dense) == pytest.approx(0.0, abs=1e-9)
    assert len(resample(sparse)) == len(resample(dense))


def test_the_assignment_is_optimal_not_greedy():
    """Greedy matching on a case built to defeat it.

    Nearest-first pairs truth[0] with prediction[0] and then has to pair the
    remaining two across the canvas; the optimal assignment pays a little on the
    first pair to save a lot on the second. If the matcher is greedy the local
    score is computed between strokes that are not counterparts at all, and the
    number stops meaning anything.
    """
    truth = _trace(
        [(0.0, 0.0), (4.0, 0.0)],
        [(10.0, 0.0), (14.0, 0.0)],
    )
    prediction = _trace(
        [(11.0, 0.0), (15.0, 0.0)],   # nearest to truth[1]
        [(1.0, 0.0), (5.0, 0.0)],     # nearest to truth[0]
    )
    report = scale_fidelity(truth, prediction)

    assert report["matched"] == 2
    # Both pairs are 1.0 apart under the optimal assignment; the greedy pairing
    # would cost (1 + 11)/2 or worse.
    assert report["match_distance"] == pytest.approx(1.0)


def test_unmatched_strokes_are_counted_and_never_folded_into_a_distance():
    """A model that emits three strokes instead of ten must not score well on
    local fidelity for having emitted fewer things to be wrong about."""
    truth = _trace(SQUARE, [(p[0] + 60, p[1] + 60) for p in BLADE], [(p[0], p[1] + 90) for p in BLADE])
    prediction = _trace(SQUARE)
    report = scale_fidelity(truth, prediction)

    assert report["n_truth"] == 3 and report["n_prediction"] == 1
    assert report["matched"] == 1
    assert report["unmatched_truth"] == 2 and report["unmatched_prediction"] == 0
    assert report["local"] == pytest.approx(0.0), "the matched pair is identical"


def test_an_empty_prediction_is_reported_rather_than_crashing():
    """Early in training a model emits nothing at all, and a metric that raises
    there turns a measurement into an exception -- the same mistake the VM's
    fault reporting exists to avoid."""
    report = scale_fidelity(_trace(SQUARE), Trace())

    assert report["n_prediction"] == 0 and report["matched"] == 0
    assert math.isinf(report["global"]) and math.isnan(report["local"])


def test_discs_and_regions_are_geometry_too():
    """Dropping them would make a drawing made of circles score as empty."""
    trace = Trace(discs=[Disc(cx=5.0, cy=5.0, r=3, width=1)])
    assert len(parts(trace)) == 1
    assert scale_fidelity(trace, trace)["local"] == pytest.approx(0.0)


def test_it_runs_on_real_vm_traces():
    """Tier A programs, executed, matched end to end."""
    vm = VM()
    truth = vm.run(assemble("MOVE 10 10\nLINE 40 40\nMOVE 80 80\nCIRCLE 12\nHALT"))
    close = vm.run(assemble("MOVE 12 11\nLINE 41 40\nMOVE 82 80\nCIRCLE 12\nHALT"))
    far = vm.run(assemble("MOVE 200 200\nLINE 230 230\nMOVE 20 20\nCIRCLE 12\nHALT"))

    near_report = scale_fidelity(truth, close)
    far_report = scale_fidelity(truth, far)
    assert near_report["matched"] == 2
    assert near_report["global"] < far_report["global"]
    assert near_report["local"] < 2.0


def test_the_centroid_is_arclength_weighted_not_a_vertex_mean():
    """`global` is the layout number, so anything that moves a centroid without
    moving the drawing corrupts it. Vertex density does exactly that: the same
    path written with more points has a different vertex mean, and the VM emits
    16 points for a curve and 2 for a line."""
    sparse = np.asarray([(0.0, 0.0), (30.0, 0.0)])
    dense = np.stack([np.concatenate([np.linspace(0, 5, 30), np.linspace(5, 30, 5)]),
                      np.zeros(35)], axis=1)  # same path, points bunched at one end

    assert dense.mean(axis=0)[0] < 12.0, "the vertex mean really is pulled by density"
    assert centroid(sparse) == pytest.approx(centroid(dense), abs=0.6)
    assert scale_fidelity(_trace(list(map(tuple, sparse))),
                          _trace(list(map(tuple, dense))))["global"] < 0.6
