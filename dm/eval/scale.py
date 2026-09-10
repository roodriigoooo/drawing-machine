"""Local and global fidelity, measured separately.

This is the instrument for the one experiment that can falsify the hierarchical
design before it is built (`docs/scale-separation.md`). The claim is that
denoising decomposes by scale and autoregression cannot, and that local
structure (how a line is made) transfers up a complexity ladder while global
structure (how forty strokes organise into a composition) does not. If that is
true, a model trained on `simple` and evaluated on `elaborate` degrades much
more on layout than on stroke shape. If both degrade equally, the levels are not
separable and the hierarchy is the wrong design.

The whole value of the test is in *not* mixing the two numbers:

    global  Chamfer between the two clouds of stroke centroids. Ignores stroke
            shape entirely -- rescaling or reshaping every stroke in place does
            not move it.
    local   optimal assignment of strokes on centroid distance, then each
            matched pair centred on its own centroid, arclength-resampled and
            compared pointwise. Ignores layout entirely -- translating the whole
            drawing does not move it.

Three decisions that the measurement depends on, all of them documented in
`docs/scale-separation.md` before any number existed:

* **Optimal assignment, not greedy.** Greedy matching makes the result depend on
  iteration order, and the comparison being made is precisely whether two
  quantities degrade differently. An order-dependent matcher biases that.
* **Unmatched strokes are a separate count, never folded in.** Rolling a missing
  stroke into either distance contaminates the separation the test exists to
  measure -- a model that emits five strokes instead of twelve would score well
  on local fidelity for the wrong reason.
* **Direction-invariant shape.** A stroke drawn end-to-start is the same stroke.
  Scoring draw order as shape error would report a large local gap for a model
  that has learned shape perfectly.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.optimize import linear_sum_assignment

from ..vm.interp import Trace

#: Points each stroke is resampled to before shapes are compared. Uniform in
#: arclength, so a 2-point line and a 40-point curve are comparable and the
#: comparison is not dominated by how many vertices the VM happened to emit.
RESAMPLE = 32

#: Points a disc is sampled to. It is geometry the drawing contains, so it takes
#: part in matching rather than being silently dropped.
DISC_POINTS = 16


def parts(trace: Trace) -> list[np.ndarray]:
    """Every piece of geometry in the trace, as its own point array.

    Strokes, filled regions and discs all count. `dm.eval.metrics.trace_points`
    flattens them into one cloud, which is what Chamfer needs and what stroke
    matching cannot use: the identity of a stroke is exactly the thing being
    matched.
    """
    out = [np.asarray(s.points, dtype=np.float64) for s in trace.strokes]
    out += [np.asarray(r.points, dtype=np.float64) for r in trace.regions]
    for disc in trace.discs:
        theta = np.linspace(0.0, 2.0 * math.pi, DISC_POINTS, endpoint=False)
        out.append(
            np.stack([disc.cx + disc.r * np.cos(theta), disc.cy + disc.r * np.sin(theta)], axis=1)
        )
    return [p for p in out if len(p)]


def centroid(points: np.ndarray) -> np.ndarray:
    """Arclength-weighted centre, not the mean of the vertices.

    The vertex mean is a density artifact: a closed polygon that repeats its
    first point weights that corner twice, and a curve the VM flattened into 16
    segments outweighs the straight line next to it by 16 to 2. Both would move
    `global` -- the *layout* number -- for reasons that have nothing to do with
    layout. Resampling first makes the centre depend on the path and not on how
    it happened to be written down, which is the same invariance `local`
    already needs.

    Computed exactly, as the segment-length-weighted mean of segment midpoints,
    rather than approximately by resampling: adding a vertex in the middle of a
    straight segment then moves the centre by exactly zero instead of by a
    discretisation error, and the exact form is cheaper.
    """
    if len(points) < 2:
        return points.mean(axis=0)
    midpoints = (points[:-1] + points[1:]) / 2.0
    length = np.linalg.norm(np.diff(points, axis=0), axis=1)
    total = length.sum()
    if total <= 0:  # every vertex in one place
        return points.mean(axis=0)
    return (midpoints * length[:, None]).sum(axis=0) / total


def centroids(pieces: list[np.ndarray]) -> np.ndarray:
    return (
        np.stack([centroid(p) for p in pieces])
        if pieces
        else np.zeros((0, 2), dtype=np.float64)
    )


def resample(points: np.ndarray, n: int = RESAMPLE) -> np.ndarray:
    """Uniform in arclength, so vertex density cannot masquerade as shape."""
    if len(points) == 1:
        return np.repeat(points, n, axis=0)
    step = np.linalg.norm(np.diff(points, axis=0), axis=1)
    walked = np.concatenate([[0.0], np.cumsum(step)])
    if walked[-1] <= 0:  # a degenerate stroke: every vertex in one place
        return np.repeat(points[:1], n, axis=0)
    want = np.linspace(0.0, walked[-1], n)
    return np.stack([np.interp(want, walked, points[:, axis]) for axis in (0, 1)], axis=1)


def shape_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Mean pointwise distance after centring, min over draw direction.

    Centred on the *resampled* mean, i.e. on `centroid`, so that the point
    `local` ignores is exactly the point `global` measures. Centring on the
    vertex mean instead would leave a density-dependent offset inside the shape
    distance, and the two numbers would overlap in a way that no reader could
    see.
    """
    x = resample(a) - centroid(a)
    y = resample(b) - centroid(b)
    forward = float(np.linalg.norm(x - y, axis=1).mean())
    reverse = float(np.linalg.norm(x - y[::-1], axis=1).mean())
    return min(forward, reverse)


def chamfer(a: np.ndarray, b: np.ndarray) -> float:
    """Symmetric mean nearest-neighbour distance between two point sets."""
    if len(a) == 0 or len(b) == 0:
        return float("inf")
    d = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=-1)
    return float(d.min(axis=1).mean() + d.min(axis=0).mean()) / 2.0


def scale_fidelity(truth: Trace, prediction: Trace) -> dict:
    """Global (layout) and local (shape) distance, reported separately.

    `local` averages over matched pairs only, and `unmatched_truth` /
    `unmatched_prediction` carry what was left over. A caller that wants one
    number has to decide how to combine them itself, which is the point: there
    is no combination that preserves the distinction this exists to make.
    """
    a, b = parts(truth), parts(prediction)
    ca, cb = centroids(a), centroids(b)
    result = {
        "global": chamfer(ca, cb),
        "n_truth": len(a),
        "n_prediction": len(b),
        "matched": 0,
        "unmatched_truth": len(a),
        "unmatched_prediction": len(b),
        "local": float("nan"),
    }
    if not a or not b:
        return result

    cost = np.linalg.norm(ca[:, None, :] - cb[None, :, :], axis=-1)
    rows, cols = linear_sum_assignment(cost)
    result["matched"] = len(rows)
    result["unmatched_truth"] = len(a) - len(rows)
    result["unmatched_prediction"] = len(b) - len(rows)
    result["local"] = float(
        np.mean([shape_distance(a[i], b[j]) for i, j in zip(rows, cols)])
    )
    result["match_distance"] = float(cost[rows, cols].mean())
    return result


__all__ = [
    "DISC_POINTS",
    "RESAMPLE",
    "centroid",
    "centroids",
    "chamfer",
    "parts",
    "resample",
    "scale_fidelity",
    "shape_distance",
]
