"""The sample-quality axis, pinned against sets whose answer is known.

`dm/eval/quality.py` compares two *sets* of drawings, so unlike every other
metric here it has no per-program ground truth to check against. What it has
instead is degenerate generators whose scores are arithmetic: a model that
repeats one drawing must cover exactly one of the reference's `m`, and a model
that returns the reference itself must score zero distance to it. Both are
tested, and they fail in opposite directions on purpose -- that is the whole
argument for reporting three numbers rather than one.

Two of the tests are not about a model at all. The pooled matrix has to agree
with `dm.eval.metrics.chamfer`, which is the distance every other Chamfer number
in this project was computed with, and it has to be invariant to the padding and
the chunking that make it batched -- both of which are silent when wrong, and the
padding one worst on the shortest drawings.

And the size-balance trap gets its own test, because it is the one that would
have sunk the axis: `nna`'s textbook 0.5 and `coverage`'s scale are properties of
the two *set sizes*, not of the drawings, so the same real drawings read as a
separable model the moment the reference is bigger than the sample.
"""

import math

import torch

from dm.data.synthetic import split
from dm.eval.metrics import chamfer
from dm.eval.quality import (
    Clouds,
    chamfer_matrix,
    clouds_of,
    distribution_metrics,
    quality_of,
    subsample,
)
from dm.isa.asm import assemble
from dm.vm.interp import VM

#: Points per drawing. Below `CLOUD_POINTS` because every test here is a
#: property of the estimator rather than a number anyone quotes, and a Chamfer
#: matrix is O(p^2) in it.
POINTS = 64


def corpus(n_train: int = 128, n_val: int = 128, seed: int = 0):
    """Two disjoint sets of real programs. `split` deduplicates val against
    train, which is what makes the second set a legitimate stand-in for a
    perfect generator rather than a copy of the reference."""
    return split(n_train, n_val, seed=seed)


def test_a_collapsed_generator_reads_coverage_at_its_floor_and_nna_at_one():
    """The mode-collapse case, and the reason `coverage` exists.

    A model that draws one thing forever is *good* on fidelity -- its one drawing
    is a real one -- so `mmd` cannot see it. `coverage` is exactly `1/m` because
    every sample has the same nearest reference, and `nna` is ~1 because a
    sample's nearest neighbour is always another copy of itself.
    """
    train, val = corpus()
    gen, _ = clouds_of([train[0]] * 40, POINTS)
    ref, _ = clouds_of(val[:40], POINTS)
    got = distribution_metrics(gen, ref)

    assert got["coverage"] == 1 / 40
    assert got["nna"] > 0.95
    # And the number that stays quiet: fidelity is finite and unremarkable,
    # which is the point -- one real drawing repeated is not a bad drawing.
    assert math.isfinite(got["mmd"])


def test_a_memorising_generator_fails_the_other_way():
    """The opposite degenerate case, and why `nna` is the only one of the three
    with a null value rather than a direction.

    Returning the reference verbatim is perfect on both one-sided numbers --
    `mmd` at zero, `coverage` at one -- and is not a good generator. `nna` goes
    to zero, which is *below* the floor: every drawing's nearest neighbour is its
    own twin in the other set.
    """
    _, val = corpus()
    ref, _ = clouds_of(val[:40], POINTS)
    got = distribution_metrics(ref, ref)

    assert got["coverage"] == 1.0
    assert got["mmd"] < 0.05
    assert got["nna"] < 0.05


def test_the_floor_at_matched_sizes_is_the_estimators_self_test():
    """Real against real, same size: 0.5, or this file has a bug.

    Two equal samples of one distribution are indistinguishable by definition, so
    a floor far from 0.5 is not a property of the corpus -- it is masking,
    normalisation or pooling being wrong. Measured across six seeds at this size:
    mean 0.500, range [0.453, 0.570], so the tolerance is the estimator's own
    sampling noise and not slack.
    """
    train, val = corpus()
    gen, _ = clouds_of(train[:96], POINTS)
    ref, _ = clouds_of(val[:96], POINTS)
    got = distribution_metrics(gen, ref)

    assert abs(got["nna"] - 0.5) < 0.1
    # `coverage` has a floor too, and it is nowhere near 1: with both sets at 96,
    # real drawings reach under half of the reference. Reading a model's 0.40
    # against 1.0 would call a working generator collapsed.
    assert 0.2 < got["coverage"] < 0.8


def test_an_unbalanced_reference_moves_nna_and_caps_coverage_on_unchanged_drawings():
    """The trap that would have sunk the axis, measured on drawings that do not
    change between the two readings.

    The *same* 32 real drawings read near the floor against 32 and as a plainly
    separable model against 200 -- the larger set simply supplies most of
    everybody's neighbours. `coverage` degenerates the same way and is bounded by
    `n/m` before a model does anything. This is why `scripts/resample.py`
    subsamples the reference to the sample count under a seed of its own.
    """
    train, val = corpus(n_val=200)
    gen, _ = clouds_of(train[:32], POINTS)
    balanced, _ = clouds_of(val[:32], POINTS)
    lopsided, _ = clouds_of(val[:200], POINTS)

    even = distribution_metrics(gen, balanced)
    uneven = distribution_metrics(gen, lopsided)

    assert abs(even["nna"] - 0.5) < 0.15
    assert uneven["nna"] > even["nna"] + 0.15
    # The bound is arithmetic: 32 samples have at most 32 distinct nearest
    # neighbours among 200 references.
    assert uneven["coverage"] <= 32 / 200


def test_padding_is_masked_on_both_sides():
    """Masking only the targets leaves a padded row acting as a *source* at the
    canvas origin, which pulls every distance towards the top-left corner --
    silently, and worst on the shortest drawings.

    The test is invariance: widening the padding cannot move a distance, because
    padding is not geometry. An implementation that masked one side would move
    every entry involving a short drawing, since more padding means more phantom
    points at (0, 0).
    """
    _, val = corpus()
    clouds, _ = clouds_of(val[:16], POINTS)
    assert int(clouds.sizes.min()) < int(clouds.sizes.max()), "need ragged drawings"

    wide = Clouds(torch.cat([clouds.points, torch.zeros(len(clouds), 3 * POINTS, 2)],
                            dim=1), clouds.sizes)
    assert torch.allclose(chamfer_matrix(wide), chamfer_matrix(clouds), atol=1e-4)


def test_the_pooled_matrix_is_the_projects_own_chamfer():
    """One batched matrix has to be the same quantity as the pairwise distance
    every other Chamfer number here was computed with, or the sample-quality
    columns are in different units from the rest of the project.

    Compared against `dm.eval.metrics.chamfer` at the same point count. The
    tolerance is float32: `cdist` expands the norm through a matmul where the
    reference subtracts first.
    """
    _, val = corpus()
    programs = val[:12]
    vm = VM()
    got = chamfer_matrix(clouds_of(programs, POINTS)[0])

    want = torch.tensor([[chamfer(vm.run(a), vm.run(b), POINTS) for b in programs]
                         for a in programs])
    assert torch.allclose(got, want, atol=0.02)


def test_the_matrix_is_symmetric_with_a_zero_diagonal_and_survives_chunking():
    """The chunk width is a memory knob and must not be a result.

    `CHAMFER_BUDGET` decides how many rows are materialised at once. A budget of
    one forces a chunk per row, which is where an off-by-one in the row slice or
    the per-row divisor would show up -- and neither would raise.

    The diagonal is zero only to float32: `cdist` expands the norm through a
    matmul, so a drawing's distance to itself comes out at ~1e-2 on a 256-pixel
    canvas rather than at 0. That is the same error that bounds the memorisation
    case above, and it is three decimal orders below the effects being read.
    """
    _, val = corpus()
    clouds, _ = clouds_of(val[:16], POINTS)
    whole = chamfer_matrix(clouds)

    assert torch.allclose(whole, whole.T, atol=1e-5)
    assert torch.allclose(whole.diagonal(), torch.zeros(len(clouds)), atol=0.02)
    assert torch.equal(chamfer_matrix(clouds, budget=1), whole)


def test_a_blank_canvas_is_counted_rather_than_dropped_silently():
    """A drawing with no geometry cannot enter a Chamfer distance at all, so it
    has to leave the set -- but dropping it quietly would let a model that emits
    90% blanks publish the coverage of its remaining 10%."""
    _, val = corpus()
    blank = assemble("HALT")
    assert not len(VM().run(blank)), "the empty program must draw nothing"

    ref, _ = clouds_of(val[:20], POINTS)
    got = quality_of(list(val[:10]) + [blank] * 10, ref, POINTS)

    assert got["empty"] == 0.5
    assert got["n_gen"] == 10, "the blanks must not be scored as drawings"
    assert all(math.isnan(got[k]) for k in ("coverage", "mmd", "nna"))


def test_a_generator_that_draws_nothing_reports_nan_rather_than_a_score():
    """The degenerate end of the above. With no geometry there is no distance,
    and a zero or a one here would be read as a result."""
    _, val = corpus()
    ref, _ = clouds_of(val[:20], POINTS)
    got = quality_of([assemble("HALT")] * 8, ref, POINTS)

    assert got["empty"] == 1.0
    assert all(math.isnan(got[k]) for k in ("coverage", "mmd", "nna"))


def test_subsample_is_reproducible_and_moves_with_its_seed():
    """Both of the file's real-data sets go through it: the reference, which must
    be identical across every arm on one corpus, and the floor's stand-ins, which
    must differ per draw or the floor would have no spread."""
    _, val = corpus()
    assert subsample(val, 16, seed=7) == subsample(val, 16, seed=7)
    assert subsample(val, 16, seed=7) != subsample(val, 16, seed=8)
    assert len(subsample(val, 16, seed=7)) == 16
    # Shorter than the request rather than padded or repeated: a set of 16 that
    # silently contained duplicates would depress every distance in it.
    assert len(subsample(val[:9], 16, seed=7)) == 9
