"""Canonicalising a drawing's frame, and the result that exists before any run.

`docs/direction.md` §7.1 proposes aligning every sketch to a standard frame
before it enters the corpus, so that two people drawing the same object at
different tilts produce more similar programs, and it proposes one test:
canonicalise, then ask `scripts/repeat_oracle.py` whether the corpus's repeat
ceiling went **up**.

**That test cannot fire in the direction it hopes, and the arithmetic says so
before the run** -- the same move `docs/conditioning.md` §1 makes when it puts a
2.32-bit ceiling under a 2.5-bit floor.

Both oracles are *within*-program. A canonicalisation applies **one** map to a
whole drawing, so for any map `g` in D4 ⋉ integer translation:

    a body `B` recurring at `B + t` becomes `g(B)` recurring at `g(B) + L(t)`,

where `L` is `g`'s linear part -- a signed permutation of the two axes, so the
step stays inside `i8` exactly when it started there. The set of foldable
repeats is carried across bijectively and **the ceiling is invariant, exactly,
for every exact policy.** The same argument runs for `REPEATX`: an orbit's step
conjugates to `g s g^-1`, which is another element of the same group.

So the branches split like this, and neither of them is the one §7.1 wanted:

- **an exact canonicalisation** (`d4` below) is a *provable null* on both
  within-program oracles -- not a measurement, a theorem, and
  `tests/test_canonical.py` checks it on real corpora rather than trusting it;
- **an inexact one** (`rot`, `aspect`) is a non-integer map followed by
  rounding, which is exactly the operation measured to destroy 77% of Tabler's
  ceiling at ×1.07. It can only lose, and what it loses is worth measuring
  because §7.1's third branch asks for the rounding loss beside the gain.

**The gain therefore has to be measured somewhere else, and there is a natural
place.** "Two people drawing the same object produce more similar programs" is a
statement about *pairs of drawings*, and no instrument in this project had ever
looked between two drawings. `dm/eval/library.py` is that instrument, and it is
the ceiling `Op.CALL` would need anyway. Canonicalisation is scored there.

Policies act at two levels and the split is not cosmetic:

- `d4` is applied to the finished **program**, because on integer coordinates a
  canvas-centred D4 element is exactly invertible and the null above is exact.
  Applied to floats it would inherit `round`'s half-integer tie-breaking and
  stop being a relabelling.
- `rot` and `aspect` are applied to the **source geometry**, before
  quantisation, because applying them to an already-quantised program rounds
  twice and would charge the idea for an artifact of where it was inserted.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..isa.spec import CANVAS
from .augment import Affine, apply

#: Canvas margin the frame policies fit a drawing into, matching
#: `quickdraw.load`'s default so a canonicalised corpus is framed like the one
#: every Tier B number was measured on.
MARGIN = 8


@dataclass(frozen=True)
class Policy:
    """A canonicalisation, and whether it is inside the exact transform group.

    `exact` is the field that matters: it is what says whether a number measured
    under this policy is a compression number or a description of rounding
    damage, and it is checked rather than declared -- `scripts/canonical.py`
    refuses to report a ceiling *gain* for an inexact policy without the loss
    beside it.
    """

    name: str
    exact: bool
    frame: bool          # acts on source geometry rather than on the program
    doc: str


POLICIES: dict[str, Policy] = {
    p.name: p
    for p in (
        Policy("none", True, False, "the pipeline as it stands"),
        Policy("d4", True, False,
               "the lexicographically least of the eight D4 images"),
        Policy("rot", False, True,
               "principal axis rotated horizontal, sign fixed by skew"),
        Policy("aspect", False, True,
               "bounding box stretched to fill the canvas in both axes"),
        Policy("rot_aspect", False, True, "rot, then aspect"),
    )
}


# ---------------------------------------------------------------------------
# exact: the D4 orbit representative


def d4_representative(program: bytes) -> tuple[bytes, int]:
    """The canonical spelling of `program`'s D4 orbit, and the code that reached it.

    Canonical in the strict sense: two programs related by a D4 element map to
    the *same* representative, because the eight images of a drawing are the
    eight images of any of its images, and "lexicographically least" is a choice
    function on that set. Ties are impossible to get wrong -- a tie means two
    elements produced identical bytes, and then either answer is the same corpus.

    D4 maps the canvas onto itself, so no image can leave it and `apply` never
    refuses for a geometric reason. It can still refuse an L2 program whose
    conjugated `XFORM` shift has no `i8` spelling; that program is returned
    unchanged and its code is `0`, because a canonicalisation that silently
    dropped drawings would change the corpus it was meant to re-frame.
    """
    best, best_code = program, 0
    for code in range(8):
        turns, mirror = (code >> 1) & 3, bool(code & 1)
        image = apply(program, Affine(0, 0, mirror, turns))
        if image is not None and image < best:
            best, best_code = image, code
    return best, best_code


# ---------------------------------------------------------------------------
# inexact: frame policies on source geometry


def _cloud(strokes: list[np.ndarray]) -> np.ndarray:
    return np.concatenate(strokes) if strokes else np.zeros((0, 2))


def principal_frame(strokes: list[np.ndarray]) -> np.ndarray:
    """The 2x2 rotation taking the drawing's principal axis to horizontal.

    The axis comes from the arclength-weighted second moment rather than from
    the vertices, for the reason `dm/eval/scale.py:centroid` gives: vertex
    density is an artifact of how the path was written down, and a curve the VM
    flattened into 16 segments would otherwise outweigh the straight line beside
    it 16 to 2. A drawing's principal axis must be a property of its shape or
    canonicalising by it re-frames the *sampling*.

    The eigenvector is defined up to sign, and a canonicalisation that leaves a
    180-degree ambiguity is not one -- it would send the same object to two
    frames and destroy the similarity it exists to create. The sign is fixed by
    requiring non-negative third central moment along each axis, i.e. the long
    tail points the same way in every canonicalised drawing. A drawing with zero
    skew on an axis is symmetric on it, so either sign gives the same shape and
    the tie is broken by the eigenvector's own first non-zero component.
    """
    if not strokes:
        return np.eye(2)
    midpoints, weights = [], []
    for stroke in strokes:
        if len(stroke) < 2:
            continue
        midpoints.append((stroke[:-1] + stroke[1:]) / 2.0)
        weights.append(np.linalg.norm(np.diff(stroke, axis=0), axis=1))
    if not midpoints:
        return np.eye(2)
    pts = np.concatenate(midpoints)
    w = np.concatenate(weights)
    total = float(w.sum())
    if total <= 0.0:
        return np.eye(2)
    mean = (pts * w[:, None]).sum(axis=0) / total
    rel = pts - mean
    cov = (rel * w[:, None]).T @ rel / total
    values, vectors = np.linalg.eigh(cov)
    major = vectors[:, int(np.argmax(values))]
    minor = np.array([-major[1], major[0]])
    basis = np.stack([major, minor])                 # rows: new x, new y
    projected = rel @ basis.T
    for axis in (0, 1):
        skew = float((w * projected[:, axis] ** 3).sum())
        if skew < 0.0 or (skew == 0.0 and _first_negative(basis[axis])):
            basis[axis] = -basis[axis]
    if float(np.linalg.det(basis)) < 0.0:
        # Keep the frame a rotation. A reflection here would fold mirror pairs
        # together, which is `d4`'s job and is a different (and exact) policy;
        # doing it silently inside a float rotation would hide an exact
        # canonicalisation inside an inexact one and confuse the two results.
        basis[1] = -basis[1]
    return basis


def _first_negative(vector: np.ndarray) -> bool:
    for value in vector:
        if value != 0.0:
            return bool(value < 0.0)
    return False


def _fit_box(strokes: list[np.ndarray], margin: int,
             isotropic: bool) -> list[np.ndarray]:
    """Place the drawing in the canvas, uniformly or one axis at a time."""
    cloud = _cloud(strokes)
    if not len(cloud):
        return strokes
    lo, hi = cloud.min(axis=0), cloud.max(axis=0)
    span = np.maximum(hi - lo, 1e-9)
    usable = float(CANVAS - 1 - 2 * margin)
    scale = np.full(2, usable / float(span.max())) if isotropic else usable / span
    centre = (lo + hi) / 2.0
    shift = np.array([(CANVAS - 1) / 2.0, (CANVAS - 1) / 2.0])
    return [(s - centre) * scale + shift for s in strokes]


def reframe(strokes: list[np.ndarray], policy: Policy,
            margin: int = MARGIN) -> list[np.ndarray]:
    """Apply a frame policy to source geometry. Exact policies are the identity here."""
    if not policy.frame or not strokes:
        return strokes
    out = strokes
    if policy.name in ("rot", "rot_aspect"):
        basis = principal_frame(out)
        out = [s @ basis.T for s in out]
    # Every frame policy re-places the drawing afterwards, because a rotation
    # moves the bounding box and a corpus whose drawings sit at different scales
    # is not canonicalised, it is rotated.
    return _fit_box(out, margin, isotropic=policy.name not in ("aspect", "rot_aspect"))


def canonicalise_program(program: bytes, policy: Policy) -> bytes:
    """Apply the program-level half of a policy. Exact by construction."""
    if policy.name == "d4":
        return d4_representative(program)[0]
    return program


__all__ = [
    "MARGIN",
    "POLICIES",
    "Policy",
    "canonicalise_program",
    "d4_representative",
    "principal_frame",
    "reframe",
]
