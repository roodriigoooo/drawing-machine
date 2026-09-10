"""A compositional corpus whose compression ceiling is arithmetic.

Claim 2 worked because the generator chose the repeat count, so the ceiling was
exact rather than estimated; Tier C's oracle worked because exact translational
repeats are countable. **This does the same thing for transformed reuse**: a
scene is one motif drawn several times under a known element of D4, so the bytes
`REPEATX` could fold are known per program before any model exists
(`docs/direction.md` §6).

**The tension this design exists to resolve is that composition normally means
*scale*, and the transform group has none.** That prohibition is measured, not
stylistic -- ×1.07 scale destroys 77% of Tabler's repeat ceiling and ±1 jitter
65%, because rounding commutes with integer translation and with nothing else.
The resolution is that the prohibition applies to the group *inside the ISA*,
not to corpus construction: a motif is scaled **once**, before it enters the
corpus, and every copy of it is then an exact integer-affine image of the same
rounded bytes. The copies are images of each other rather than of a common
ancestor, so no copy is a near-miss.

Scaling once is `quickdraw.load`'s own `margin`, which fits a drawing into
`255 - 2·margin` pixels. At `margin=72` a motif is ~69 bytes across 111 px and
fits inside a canvas quadrant, which is what makes the geometry below work.

**The one fact this file leans on:** with a translation of zero, containment is
*automatic*, because D4 maps the canvas onto itself. A motif in the top-left
quadrant mirrors into the top-right, half-turns into the bottom-right and
quarter-turns around all four -- and not one of those placements can leave the
canvas or need clamping. Only the translation-only control can, which inverts
the usual difficulty: the transformed arms are the safe ones.

**The generator knows the answer, so the metric does not need a detector.**
Every scene carries the byte span of each copy and the element relating them,
which matters because byte-level repeat detection *cannot* find these: under a
mirror the copies' bytes are not equal (`x -> 255 - x`), so
`dm.eval.repeats.compress` finds nothing at all. A symmetry-aware matcher is
then a thing to **validate against this provenance** rather than a dependency of
the result.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from ..isa.spec import CANVAS, Kind, Op, UnknownOpcode, spec_for
from ..isa.transform import D4, Transform
from ..isa.unroll import unroll
from . import quickdraw
from .augment import Affine, apply

#: How a motif is scaled and simplified on its way in. `margin=72` leaves 111 px,
#: which fits a 128 px quadrant; `rdp_eps=8` keeps a body near 69 bytes so a
#: four-copy scene lands around 277 -- shorter than today's QuickDraw corpus at
#: `rdp_eps=2`, which is 161. Both are recorded in the cache key, so a corpus
#: cannot silently change underneath a result.
MOTIF_MARGIN = 72
MOTIF_EPS = 8.0

#: Cells per side. The quadrant grid is the coarsest one D4 maps onto itself,
#: and a finer grid buys more distractor room at the cost of longer programs --
#: sequence length is what killed Tier D, so the coarse grid is the default.
GRID = 2

#: Translation step for the translation-only control, in pixels. **Not 128**,
#: which is what a quadrant would want: `REPEAT`'s `dx` is `Kind.DELTA`, an i8,
#: so a full-quadrant step is unrepresentable and the ISA itself bounds the
#: control's grid. 120 clears a 111 px motif, so copies stay disjoint.
CONTROL_STEP = 120


@dataclass(frozen=True)
class Scene:
    """One program in both forms, with the provenance the metrics read.

    `flat` is what a model trains on -- the trace, with no `REPEATX` in it, the
    same program-versus-trace distinction claim 2 rests on. `structured` is what
    the ISA could have said instead, and the difference between their lengths is
    the ceiling.
    """

    flat: bytes
    structured: bytes
    #: Byte spans of the orbit's copies in `flat`, copy 0 first. What
    #: `recovery_by_copy` needs, and known rather than detected.
    copies: tuple[slice, ...]
    #: D4 code of the step, so copy `k` is `step^k` applied to copy 0.
    step: int
    #: Bytes `REPEATX` folds away: `(n - 1)·|motif| - 6`.
    saved: int
    distractors: int
    #: Byte span of each distractor in `flat`, in emission order -- the blocks
    #: that are deliberately **not** copies of anything.
    #:
    #: Recorded rather than detected for the reason `copies` is: the boundary
    #: between two adjacent distractors is invisible in the byte stream, because
    #: a motif is a run of `MOVE`/`LINE` and so is the next one. Direction 2's
    #: shape venue needs whole unrelated blocks -- a matched perturbation that
    #: straddled two motifs would be a perturbation of neither. Defaults to the
    #: empty tuple so a hand-built `Scene` in a test stays constructible; the
    #: generator always fills it.
    distractor_spans: tuple[slice, ...] = ()

    @property
    def foldable(self) -> float:
        return self.saved / max(1, len(self.flat))


def bounds(program: bytes) -> tuple[int, int, int, int] | None:
    """`(min_x, min_y, max_x, max_y)` over every `COORD` operand, or None.

    By `Kind`, never positionally: a `CIRCLE`'s radius is a length and a
    `REPEAT`'s `dx` is a displacement, and folding either into a bounding box
    would place the motif by a number that is not a position. The same
    distinction `dm/data/augment.py` makes, for the same reason.
    """
    xs: list[int] = []
    ys: list[int] = []
    pc = 0
    while pc < len(program):
        try:
            spec = spec_for(program[pc])
        except UnknownOpcode:
            return None
        if pc + spec.size > len(program):
            return None
        index = 0
        while index < len(spec.operands):
            if spec.operands[index] is Kind.COORD:
                xs.append(program[pc + 1 + index])
                ys.append(program[pc + 2 + index])
                index += 2
            else:
                index += 1
        pc += spec.size
    if not xs:
        return None
    return min(xs), min(ys), max(xs), max(ys)


def body_of(program: bytes) -> bytes:
    """The motif without its terminator. Scenes carry exactly one `HALT`."""
    return program[:-1] if program and program[-1] == int(Op.HALT) else program


def place(motif: bytes, box: tuple[int, int, int, int],
          rng: random.Random) -> bytes | None:
    """Translate a motif to a random position inside `box`, or None if it will
    not fit. The jitter is the point: a motif pinned to a cell corner would let
    a model read position off the grid instead of off the drawing."""
    extent = bounds(motif)
    if extent is None:
        return None
    x0, y0, x1, y1 = extent
    left, top, right, bottom = box
    slack_x, slack_y = (right - left) - (x1 - x0), (bottom - top) - (y1 - y0)
    if slack_x < 0 or slack_y < 0:
        return None
    return apply(motif, Affine(dx=left + rng.randint(0, slack_x) - x0,
                               dy=top + rng.randint(0, slack_y) - y0))


def cells(grid: int = GRID) -> list[tuple[int, int, int, int]]:
    """The grid as pixel boxes. Square and centred, so D4 permutes them."""
    side = CANVAS // grid
    return [(gx * side, gy * side, (gx + 1) * side - 1, (gy + 1) * side - 1)
            for gy in range(grid) for gx in range(grid)]


def orbit_boxes(motif: bytes, step: Transform, n: int) -> list[tuple[int, ...]] | None:
    """Where the `n` copies land, or None if two of them overlap.

    A transform maps an axis-aligned box to an axis-aligned box because every
    element of D4 is a signed permutation of the axes, so the corners are enough.
    Overlap is checked rather than assumed: a scene whose copies sit on top of
    each other draws one shape and claims the redundancy of several.
    """
    extent = bounds(motif)
    if extent is None:
        return None
    x0, y0, x1, y1 = extent
    out = []
    for k in range(n):
        corners = [step.power(k).point(x, y)
                   for x, y in ((x0, y0), (x1, y0), (x0, y1), (x1, y1))]
        xs = [p[0] for p in corners]
        ys = [p[1] for p in corners]
        out.append((min(xs), min(ys), max(xs), max(ys)))
    for i, a in enumerate(out):
        for b in out[i + 1:]:
            if not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1]):
                return None
    return out


def orbits(control: bool) -> list[tuple[Transform, int]]:
    """`(step, copies)` an orbit may use, as whole transforms.

    Every element's order bounds its own count: a mirror repeated four times
    draws two shapes twice and would claim a ceiling the drawing does not have.
    Only the quarter turns have order 4.

    **The control cannot offer a four-copy orbit and that is a fact about the
    canvas, not a shortcut.** Four disjoint translated copies of a 111 px motif
    need 111 + 3·step ≤ 255, so step ≤ 48 -- less than the motif's own width, so
    the copies overlap. Translation cannot pack four disjoint copies of a
    quarter-canvas motif; the rotations can, because they use the canvas's own
    symmetry. So a matched comparison against the control is a comparison at
    `n = 2`, and `orbit_sizes` is how a caller pins that.
    """
    if control:
        return [(Transform(D4(), dx, dy), 2)
                for dx, dy in ((CONTROL_STEP, 0), (-CONTROL_STEP, 0),
                               (0, CONTROL_STEP), (0, -CONTROL_STEP),
                               (CONTROL_STEP, CONTROL_STEP))]
    return [(Transform(D4.of(code), 0, 0), n) for code, n in (
        (D4(0, True).code, 2), (D4(2, False).code, 2), (D4(1, False).code, 2),
        (D4(1, True).code, 2), (D4(3, True).code, 2), (D4(1, False).code, 4),
    )]


def scene(pool: list[bytes], rng: random.Random, grid: int = GRID,
          distractor_p: float = 0.5, control: bool = False,
          orbit_sizes: tuple[int, ...] | None = None) -> Scene | None:
    """One scene, or None if this draw does not produce a legal one.

    Returns rather than retries so the caller counts rejections. Every rejection
    is a real constraint doing its job -- a motif that will not fit its cell, an
    orbit that would overlap itself, a control translation that leaves the canvas
    -- and a generator that silently retried would hide the rate at which the
    geometry says no.
    """
    boxes = cells(grid)
    choices = [(s, n) for s, n in orbits(control)
               if orbit_sizes is None or n in orbit_sizes]
    if not choices:
        return None
    step, n = rng.choice(choices)

    motif = place(body_of(rng.choice(pool)), rng.choice(boxes), rng)
    if motif is None:
        return None
    placed = orbit_boxes(motif, step, n)
    if placed is None:
        return None

    # Distractors go in cells the orbit does not touch, so the scene has content
    # that is *not* a copy. Without them a model can learn "the rest of the
    # program is a transform of the start" as a rule about position rather than
    # about the drawing, and the whole measurement would be of the wrong thing.
    free = [box for box in boxes
            if all(box[2] < p[0] or p[2] < box[0] or box[3] < p[1] or p[3] < box[1]
                   for p in placed)]
    extras: list[bytes] = []
    for box in free:
        if rng.random() >= distractor_p:
            continue
        other = place(body_of(rng.choice(pool)), box, rng)
        if other is not None:
            extras.append(other)

    # The orbit block sits at a random depth among the distractors. `REPEATX`
    # emits its copies back to back, so they are adjacent by construction -- but
    # *where* the block starts must vary, or its offset is a constant the model
    # can key on instead of the copy relation.
    cut = rng.randint(0, len(extras))
    head, tail = extras[:cut], extras[cut:]
    structured = (b"".join(head)
                  + bytes([int(Op.REPEATX), n, step.d4.code,
                           step.dx & 0xFF, step.dy & 0xFF])
                  + motif + bytes([int(Op.ENDREP)])
                  + b"".join(tail) + bytes([int(Op.HALT)]))
    flat = unroll(structured)
    if flat is None:
        return None

    start = sum(map(len, head))
    copies = tuple(slice(start + k * len(motif), start + (k + 1) * len(motif))
                   for k in range(n))
    # The spans are computed, so they are checked. `unroll` emits copy `k` as
    # `step^k` applied to the body, and if that ever stopped being true every
    # number this corpus produces would be attributed to the wrong bytes.
    for k, span in enumerate(copies):
        want = motif if k == 0 else apply(motif, Affine.of(step.power(k)))
        if want is None or flat[span] != want:
            return None
    # Same treatment for the blocks that are not copies: computed from the
    # emission order, then checked against the bytes rather than trusted.
    spans: list[slice] = []
    cursor = 0
    for block in head:
        spans.append(slice(cursor, cursor + len(block)))
        cursor += len(block)
    cursor += n * len(motif)
    for block in tail:
        spans.append(slice(cursor, cursor + len(block)))
        cursor += len(block)
    for span, block in zip(spans, head + tail):
        if flat[span] != block:
            return None
    saved = (n - 1) * len(motif) - 6
    # The ceiling is the difference between the two forms, so it is read off
    # them rather than trusted to arithmetic that could drift from the emitter.
    if len(flat) - len(structured) != saved:
        return None
    return Scene(flat=flat, structured=structured, copies=copies,
                 step=step.d4.code, saved=saved, distractors=len(extras),
                 distractor_spans=tuple(spans))


def build_with_stats(n: int, split: str = "train", seed: int = 0,
                     grid: int = GRID, distractor_p: float = 0.5,
                     control: bool = False,
                     categories: tuple[str, ...] = ("cat", "dog", "bus",
                                                    "car", "tree"),
                     limit: int | None = None,
                     orbit_sizes: tuple[int, ...] | None = None,
                     ) -> tuple[list[Scene], dict]:
    """`n` deduplicated scenes, and what the geometry refused on the way.

    **Train and val motifs are disjoint because QuickDraw's own splits are**, so
    a val scene cannot be a memorised train motif under a new transform. The
    scenes themselves are deduplicated too: an exact repeat across the split
    would be scored as generalisation while measuring recall, which is the fault
    `dm.data.synthetic.split` documents at 7.0% val/train overlap.

    The rejection counts are returned rather than swallowed. They are the honest
    description of what a composition policy can express: a control that refuses
    two draws in three is telling you its orbits barely fit, and that belongs in
    the corpus's record rather than in a comment.
    """
    pool = quickdraw.load(categories, split, limit=limit,
                          rdp_eps=MOTIF_EPS, margin=MOTIF_MARGIN)
    if not pool:
        raise ValueError(f"no motifs for {categories} / {split}")
    rng = random.Random(seed)
    seen: set[bytes] = set()
    out: list[Scene] = []
    stats = {"drawn": 0, "rejected": 0, "duplicate": 0, "motifs": len(pool)}
    for _ in range(100 * n + 1000):
        if len(out) >= n:
            break
        stats["drawn"] += 1
        candidate = scene(pool, rng, grid, distractor_p, control, orbit_sizes)
        if candidate is None:
            stats["rejected"] += 1
            continue
        if candidate.flat in seen:
            stats["duplicate"] += 1
            continue
        seen.add(candidate.flat)
        out.append(candidate)
    if len(out) < n:
        raise ValueError(f"asked for {n} scenes, the geometry allowed {len(out)}")
    return out, stats


def build(n: int, split: str = "train", **kwargs) -> list[Scene]:
    return build_with_stats(n, split, **kwargs)[0]


def copies_of(scenes: list[Scene]) -> list[list[list[slice]]]:
    """Provenance in `recovery_by_copy`'s shape: one group of copies per scene.

    One group, not several: a scene has exactly one orbit, and the distractors
    are deliberately *not* copies of anything. A second group would have to come
    from a detector, and the whole point of a constructed corpus is that it does
    not.
    """
    return [[list(s.copies)] for s in scenes]


def spans_of(scenes: list[Scene]) -> list[list[tuple[slice, slice]]]:
    """Provenance in `recovery`'s shape: copy 1 against copies 2..n.

    The later copies are one contiguous span because `REPEATX` emits them back
    to back -- the same property `repeat_spans` relies on for `REPEAT`.

    **One entry per scene, including the scenes with nothing to report.**
    `recovery` reads `spans[i]` for program `i`, so a comprehension that
    *filtered* would shift every later scene's spans onto the wrong program and
    attribute the bits to bytes that are not copies. `orbits` never offers an
    orbit of one, so the empty list is unreachable today -- which is precisely
    why the alignment has to be structural rather than true by luck.
    """
    return [[(s.copies[0], slice(s.copies[1].start, s.copies[-1].stop))]
            if len(s.copies) > 1 else []
            for s in scenes]


def policy_label(extra: dict) -> str:
    """One line naming a composition policy, for a report's caption.

    **The orbit sizes are read off `orbits`, never off the config**, because the
    two disagree and the disagreement is not cosmetic: `orbits(control=True)`
    offers no four-copy entry at all -- four disjoint translated copies of a
    quarter-canvas motif overlap -- so a caption built from `orbit_sizes` alone
    printed `n ∈ {2,4}` for a control corpus whose canvas allows only two, on the
    control arm of the whole P5 comparison. It is `dm/data/fingerprint.py`'s rule
    one level up: **a description derived from a config cannot see a default, and
    cannot see a policy the generator overrides.**

    The spelling is named too. Two records differing only in `structured` are two
    different experiments on one corpus, and a caption that omitted it would put
    both under the same heading.
    """
    control = bool(extra.get("control"))
    sizes = extra.get("orbit_sizes")
    offered = sorted({n for _, n in orbits(control)
                      if sizes is None or n in sizes})
    parts = [
        "translation only (control)" if control else "D4 orbits",
        f"n ∈ {{{', '.join(map(str, offered))}}}",
        "REPEATX form" if extra.get("structured") else "flat trace",
    ]
    if extra.get("distractor_p") is not None:
        parts.append(f"distractor p={extra['distractor_p']}")
    return ", ".join(parts)


def ceiling(scenes: list[Scene]) -> dict:
    """What `REPEATX` could fold on these scenes, in `corpus_stats`'s shape.

    The same fields an oracle reports, so a recovery report reads the same
    whichever corpus it came from -- and `source` says which, because the two
    are not the same kind of number. `dm.eval.repeats` *searches* for structure
    and can only ever find a lower bound on what is there; this is read off the
    generator, which chose it. `docs/direction.md` section 6 is the whole reason
    this corpus exists: a ceiling known by construction is what gives `recovery`
    a denominator on day one.
    """
    total = sum(len(s.flat) for s in scenes)
    saved = sum(s.saved for s in scenes)
    with_orbit = sum(len(s.copies) > 1 for s in scenes)
    return {
        "n": len(scenes),
        "source": "construction",
        "bytes": total,
        "saved": saved,
        "saved_frac": saved / total if total else 0.0,
        "programs_with_repeat": with_orbit / len(scenes) if scenes else 0.0,
        "ratio": total / (total - saved) if total > saved else float("nan"),
    }


def val_seed(seed: int) -> int:
    """The val split's seed, given the train split's, and the **only** definition.

    A metric has to rebuild the exact scenes a run was scored on to know which
    of its bytes are copies, and it wants the val half alone -- rebuilding the
    100,000 it does not need to get at the 1,000 it does is a minute of nothing.
    So the offset is a function rather than a literal inside `split_scenes`: a
    second copy of `seed + 10_000` in a script is a corpus free to drift away
    from the one the trainer used with nothing failing, which is the run-4 fault
    one level down -- real bits, attributed to the wrong bytes.
    """
    return seed + 10_000


def split_scenes(n_train: int, n_val: int, seed: int = 0,
                 **kwargs) -> tuple[list[Scene], list[Scene]]:
    """The two splits as `Scene`s, provenance included."""
    train = build(n_train, "train", seed=seed, **kwargs)
    val = build(n_val, "valid", seed=val_seed(seed), **kwargs)
    return train, val


def split(n_train: int, n_val: int, seed: int = 0, structured: bool = False,
          **kwargs) -> tuple[list[bytes], list[bytes]]:
    """Programs for the trainer, in `dm.data.synthetic.split`'s shape.

    **`structured` picks the spelling, and the two spellings are two different
    experiments on one corpus.** Flat is claim 2's setup -- the structure is in
    the geometry and absent from the alphabet, so `recovery` can ask whether a
    model discovers it. Structured is the ISA v2 form, and it is the only one on
    which the fusion axis exists at all: `token` and `token_typed` encode an
    L0-only program *byte-identically* (`docs/direction.md` section 2.5, checked
    on 200/200 scenes), because a typed alphabet splits operands by `Kind` and a
    flat trace has exactly one. `REPEATX` brings `Kind.XF`, `Kind.COUNT` and
    `Kind.DELTA`, and the axis becomes live.

    Which spelling a record used is not inferable from its config alone once it
    is in `extra`, so it is `extra` that carries it and
    `dm/data/fingerprint.py` that makes the two corpora different keys.
    """
    train, val = split_scenes(n_train, n_val, seed=seed, **kwargs)
    text = (lambda s: s.structured) if structured else (lambda s: s.flat)
    return [text(s) for s in train], [text(s) for s in val]
