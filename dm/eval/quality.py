"""Sample quality: does the model draw what the corpus draws?

`bits_per_drawing` cannot see this and never could. It scores the model on *real*
programs, so a model that assigns good probability to the corpus and generates
nothing but scribbles scores exactly as well as one that does not. Every
generation column this project has reported — `validity`, `length_emd`,
`gen_strokes` — is a *marginal*: it compares one summary statistic of the samples
against the same statistic of the corpus, and a distribution can match every
marginal you happen to have thought of while matching nothing else.

So this compares the two **sets** of drawings, in the geometry the VM produces,
with the distance the project already has (`dm.eval.metrics.chamfer`).

Three numbers, and they fail in different directions on purpose:

- **`coverage`** — the fraction of real drawings that are the nearest neighbour
  of at least one sample. It is the mode-collapse detector: a model that draws
  one perfect cat forever scores well on fidelity and near zero here.
- **`mmd`** — mean over real drawings of the distance to the closest sample, in
  canvas pixels. Fidelity, and blind to collapse in the other direction: a model
  that memorises the corpus scores perfectly.
- **`nna`** — leave-one-out 1-nearest-neighbour accuracy over the pooled set.
  Its ideal is *indistinguishability*, so unlike the other two it fails in both
  directions: above the floor the samples are separable from the corpus, below it
  they are sitting inside the corpus's own neighbourhoods more tightly than the
  corpus does, which is memorisation.

> **`nna`'s textbook ideal of 0.5 holds only when the two sets are the same
> size, and this one's are not.** Measured on 200 real drawings against 200,
> real-against-real reads **0.495**; the same drawings at 50 against 200 read
> **0.696**, and at 100 against 200, **0.590**. Nothing about the drawings
> changed — the larger set simply supplies more of everybody's neighbours. Quote
> 0.5 against an unbalanced report and every model on earth is distinguishable.
  Read `nna` against the floor below, never against a constant.

**None of the three means anything without its own floor**, which is the trap
`PLAN.md` records for every other metric in this project and which the crude
pairwise-distance proxy this replaces fell into. `coverage` at n=256 against
m=1000 is bounded above by 0.256 before a model does anything at all, and `mmd`
is in pixels on a canvas whose scale is a corpus property. So `reference_floor`
runs the identical computation with *real* drawings in the samples' place, at the
identical sizes, taken from the training split so they are disjoint from the
reference by construction. Every number here is read as a ratio against that, and
`nna`'s floor is the estimator's own self-test: real against real must land at
0.5, and a floor that does not is a bug in this file rather than a result.

**The trainers deliberately do not record these per eval** (decided 2026-08-11,
`PLAN.md`). Not because of the cost -- one matrix is ~2 s at 256 v 256 -- but
because a per-eval column would be *unreadable*: the trainer's reference is the
whole val split against `gen_samples = 128`, which is the degenerate regime this
file's `subsample` exists to avoid (`coverage` bounded by 0.128, `nna` ~0.9 on
real drawings), and making it readable means subsampling *and* measuring a floor
at every eval, which doubles the cost to print a constant of the corpus. And it
would be a single draw of a column whose draw-to-draw spread is the whole reason
`scripts/resample.py` exists: over five draws of one checkpoint, `coverage`
spans 0.387-0.430 and `nna` 0.600-0.656. Read this axis from a report keyed by
checkpoint and sampler, over several draws, against a floor -- never from a
record's final eval.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor

from ..vm.interp import VM
from .metrics import trace_points

#: Points each drawing is resampled to. `dm.eval.metrics.chamfer` uses 256 for a
#: single pair; the pooled matrix here is O(n^2 p^2), so 128 buys a 4x cut for a
#: distance that changes in the third decimal -- these are 2D sketches of a few
#: strokes, not dense 3D scans. It is recorded in the report because a Chamfer
#: number is only comparable with another at the same resolution.
CLOUD_POINTS = 128

#: Elements of the working distance tensor, which fixes the chunk width the way
#: `dm.train.micro_batches` fixes a micro-batch: the loop below materialises
#: `chunk x points x n x points` floats and nothing else, so one constant bounds
#: the peak. 32M elements is 128 MB in float32.
CHAMFER_BUDGET = 32_000_000


@dataclass(frozen=True)
class Clouds:
    """Drawings as padded point clouds, plus how much of each row is real.

    Padded rather than ragged because the whole point of this file is one
    batched distance computation; `sizes` is what keeps the padding out of the
    arithmetic. Empty drawings are dropped before construction (`clouds_of`
    returns the count), because a Chamfer distance to nothing is infinite and
    one infinity makes every metric here NaN.
    """

    points: Tensor  # (n, p, 2), float32
    sizes: Tensor   # (n,), long

    def __len__(self) -> int:
        return int(self.points.shape[0])

    @property
    def valid(self) -> Tensor:
        """`(n, p)` — which padded slots hold a real point."""
        return (torch.arange(self.points.shape[1])[None, :] < self.sizes[:, None])


def clouds_of(programs: list[bytes], points: int = CLOUD_POINTS,
              vm: VM | None = None) -> tuple[Clouds, int]:
    """Run each program and resample its geometry. Returns the clouds and how
    many programs drew nothing.

    Executed rather than parsed, for the same reason `corpus_stats` measures
    validity in the VM: what a drawing *is* is what the interpreter emits, and a
    generated program that decodes cleanly and draws nothing is a real failure
    that a parse-level metric cannot see. The empty count is returned rather than
    folded in, because "the model drew 12 blank canvases" and "the model drew 12
    bad canvases" are different results and the second is better.
    """
    vm = vm or VM()
    kept = [c for p in programs if len(c := trace_points(vm.run(p), points))]
    if not kept:
        return Clouds(torch.zeros(0, points, 2), torch.zeros(0, dtype=torch.long)), len(programs)
    padded = torch.zeros(len(kept), points, 2)
    sizes = torch.tensor([len(c) for c in kept], dtype=torch.long)
    for i, cloud in enumerate(kept):
        padded[i, : len(cloud)] = torch.from_numpy(cloud)
    return Clouds(padded, sizes), len(programs) - len(kept)


def chamfer_matrix(clouds: Clouds, budget: int = CHAMFER_BUDGET) -> Tensor:
    """`(n, n)` symmetric Chamfer distances in canvas pixels, diagonal at zero.

    The same quantity as `dm.eval.metrics.chamfer` -- symmetric mean
    nearest-neighbour distance, halved -- computed for every pair at once,
    because all three metrics below are functions of this one matrix and
    computing it three times is the whole cost of the file.

    Both directions come out of a single distance tensor. `d[i_point, j,
    j_point]` is read down its last axis for "nearest point of j to this point of
    i" and down its first for "nearest point of i to this point of j", so the
    symmetric average costs one `cdist` rather than two.

    Padding is masked to infinity on *both* sides. Masking only the targets
    leaves padded rows of `i` acting as sources at the origin, which pulls every
    distance towards a drawing's own top-left corner -- silently, and worst on
    the shortest drawings.
    """
    n, points, _ = clouds.points.shape
    if n == 0:
        return torch.zeros(0, 0)
    flat = clouds.points.reshape(n * points, 2)
    valid = clouds.valid
    out = torch.zeros(n, n)
    chunk = max(1, budget // max(1, points * n * points))
    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        block = clouds.points[start:stop].reshape(-1, 2)
        # In place, on `cdist`'s own fresh allocation: this tensor is the peak
        # of the whole computation and a masked copy of it would triple it.
        d = torch.cdist(block, flat).view(stop - start, points, n, points)
        d.masked_fill_(~valid[start:stop, :, None, None], math.inf)
        d.masked_fill_(~valid[None, None, :, :], math.inf)
        # Sums over the real slots only, divided by how many there are: a
        # `nanmean` here would also swallow a genuine infinity.
        rows = valid[start:stop].sum(dim=1, keepdim=True)
        to_other = (d.amin(dim=3).nan_to_num(posinf=0.0).sum(dim=1) / rows)
        from_other = (d.amin(dim=1).nan_to_num(posinf=0.0).sum(dim=2)
                      / clouds.sizes[None, :])
        out[start:stop] = (to_other + from_other) / 2.0
    return out


def distribution_metrics(gen: Clouds, ref: Clouds,
                         budget: int = CHAMFER_BUDGET) -> dict:
    """`coverage`, `mmd` and `nna` for one generated set against one reference.

    One pooled matrix, three readings of it. `nna` needs the generated-generated
    and reference-reference blocks that the other two throw away, so computing
    them separately would run `cdist` over the same points twice.
    """
    n, m = len(gen), len(ref)
    if not n or not m:
        return {"coverage": float("nan"), "mmd": float("nan"), "nna": float("nan"),
                "n_gen": n, "n_ref": m}
    pooled = Clouds(torch.cat([gen.points, ref.points]),
                    torch.cat([gen.sizes, ref.sizes]))
    d = chamfer_matrix(pooled, budget)
    cross = d[:n, n:]

    # Coverage: how much of the corpus is *reached*. Bounded above by n/m, which
    # is why it is only ever read against the floor at the same two sizes.
    covered = cross.argmin(dim=1).unique().numel() / m
    # MMD: the corpus's distance to its nearest sample, not the reverse. The
    # reverse rewards a model that draws one thing well.
    mmd = float(cross.min(dim=0).values.mean())

    # 1-NNA, leave-one-out over the pooled set. The diagonal is a drawing's
    # distance to itself and would win every time.
    labelled = d.clone()
    labelled.fill_diagonal_(math.inf)
    same = torch.cat([torch.zeros(n, dtype=torch.bool), torch.ones(m, dtype=torch.bool)])
    nearest = labelled.argmin(dim=1)
    nna = float((same[nearest] == same).float().mean())
    return {"coverage": covered, "mmd": mmd, "nna": nna, "n_gen": n, "n_ref": m}


def unavailable_metrics(n: int, m: int) -> dict:
    """No set comparison when empty exclusion changed one side's size."""
    return {
        "coverage": float("nan"), "mmd": float("nan"), "nna": float("nan"),
        "n_gen": n, "n_ref": m,
    }


def quality_of(programs: list[bytes], reference: Clouds,
               points: int = CLOUD_POINTS, budget: int = CHAMFER_BUDGET) -> dict:
    """One set of drawings against a prepared reference: the three numbers and
    the fraction that drew nothing.

    `reference` arrives already built because the caller measures several draws
    against one corpus and rebuilding it per draw would run the VM over the whole
    val split five times for an answer that cannot change.

    `empty` is reported beside the three rather than folded into them. A blank
    canvas has no geometry and cannot enter a Chamfer distance at all. If any
    are excluded, all three metrics are NaN: scoring survivors would change the
    set-size ratio and let a model that emits 90% blanks publish the coverage of
    its remaining 10%.
    """
    gen, empty = clouds_of(programs, points)
    # Balanced *requested* sets are the report contract. An empty decode shrinks
    # only the generated side; expose that failure as NaNs instead of silently
    # switching to the unbalanced estimator `distribution_metrics` also supports
    # for its dedicated size-bias diagnostics.
    metrics = (distribution_metrics(gen, reference, budget)
               if len(gen) == len(reference)
               else unavailable_metrics(len(gen), len(reference)))
    return {**metrics, "empty": empty / max(1, len(programs))}


def class_quality(samples_by_class: dict[int, list[bytes]],
                  reference_by_class: dict[int, Clouds],
                  points: int = CLOUD_POINTS,
                  budget: int = CHAMFER_BUDGET) -> dict:
    """Per-class set metrics, plus the full asked × real `mmd` matrix.

    The question this answers is the one `controllability` cannot: does asking
    for a class produce that class's *geometry*, not merely a sample the model
    itself reads back correctly. So every input set is scored against **real
    drawings**, and the off-diagonal cells are what make the diagonal readable —
    a diagonal that only ties the off-diagonal means the conditioning does not
    move geometry, whatever the model's own classifier says. That is recovery
    read against its own ceiling rather than bare, applied to generation.

    Two rules the caller must not undo:

    - **A sample belongs to the class it was asked for.** Nothing here gates or
      relabels samples by any classifier — selecting on the model's own
      read-back would inflate the diagonal with exactly the circularity this
      measurement exists to escape. Empty decodes are excluded by `clouds_of`
      and *counted*, per class.
    - **Every set is the same size**, refused rather than assumed, because two
      of the three metrics are functions of the set-size ratio before they are
      functions of the drawings. Input sizes are refused here; after empty
      exclusion, a class gets an exclusion count and NaN geometry. The report
      driver then refuses that setting — padding or replacing it would score a
      different sampler, while dropping it would change the estimator's ratio.

    Off-diagonal cells carry `mmd` only. `coverage` and `nna` answer "is this
    set a good model of that corpus?", which is a question only the matched
    class is owed; across classes the reading is a *distance*, and `mmd` is the
    one of the three that is a distance.
    """
    if set(samples_by_class) != set(reference_by_class):
        raise ValueError(
            f"sample classes {sorted(samples_by_class)} != reference classes "
            f"{sorted(reference_by_class)}; a class without both sides has no cell"
        )
    sizes = {len(s) for s in samples_by_class.values()}
    ref_sizes = {len(r) for r in reference_by_class.values()}
    if len(sizes) != 1 or len(ref_sizes) != 1 or sizes != ref_sizes:
        raise ValueError(
            f"unbalanced sets (samples {sorted(sizes)}, references "
            f"{sorted(ref_sizes)}): coverage and nna are functions of the size "
            "ratio, so every sample and reference set must have the same size"
        )

    classes = sorted(samples_by_class)
    gen: dict[int, Clouds] = {}
    empty: dict[int, int] = {}
    for c in classes:
        gen[c], empty[c] = clouds_of(samples_by_class[c], points)

    diagonal = {
        c: {
            **(distribution_metrics(gen[c], reference_by_class[c], budget)
               if len(gen[c]) == len(reference_by_class[c])
               else unavailable_metrics(len(gen[c]), len(reference_by_class[c]))),
            "empty": empty[c] / max(1, len(samples_by_class[c])),
        }
        for c in classes
    }
    # The diagonal cells are lifted from the full reading above rather than
    # recomputed, so the matrix cannot disagree with the per-class table. Any
    # row shortened by empty exclusion stays NaN across every class.
    matrix = [[diagonal[asked]["mmd"] if asked == real else
               (distribution_metrics(gen[asked], reference_by_class[real], budget)["mmd"]
                if len(gen[asked]) == len(reference_by_class[real]) else float("nan"))
               for real in classes] for asked in classes]
    return {"classes": classes, "diagonal": diagonal, "matrix": matrix}


def subsample(programs: list[bytes], n: int, seed: int) -> list[bytes]:
    """`n` drawings, picked reproducibly. Both of this file's real-data sets.

    Used twice, for reasons that are worth keeping apart.

    **The reference is subsampled to the sample count**, because two of the
    three metrics are functions of the ratio of the set sizes and not only of
    the drawings. At 48 samples against 1,000 real drawings, `coverage` cannot
    exceed 0.048 no matter how good the model is, and `nna` reads 0.92 on *real
    drawings* because the larger set supplies most of everybody's neighbours.
    Both are then pinned near a constant and carry no signal. Balanced, the
    numbers mean what their definitions say. Pass the same `seed` for every arm
    on one corpus or they are not being compared on the same reference.

    **The floor's stand-ins come from the training split**, never from the
    reference itself. Drawing them from the reference measures a set against a
    superset of itself: `mmd` collapses towards zero and `nna` below 0.5, giving
    a floor that is unbeatable rather than achievable. `dm.train.build_data`
    deduplicates across the splits, so the training half is disjoint by
    construction, which is what makes the floor the value a *perfect* generator
    would score -- the trick `PLAN.md` credits for making claim 2 work, applied
    to a metric instead of a corpus.

    With both sets at `n`, the `nna` floor is also this file's self-test: two
    equal samples of one distribution are indistinguishable and must land near
    0.5 (measured: 0.495 on 200 against 200). A floor far from it is a bug here
    rather than a property of the corpus.
    """
    picked = torch.randperm(len(programs),
                            generator=torch.Generator().manual_seed(seed))
    return [programs[i] for i in picked[:n].tolist()]
