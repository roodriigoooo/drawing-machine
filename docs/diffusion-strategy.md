# The diffusion approach — why, and the strategy

Moved out of `PLAN.md` on 2026-08-05. This is claim 3's argument in full: why a
hierarchical, non-autoregressive-over-strokes model is the design, and why that
argument — not data volume — decides the dataset strategy. `PLAN.md` §5 keeps
the four-sentence version and the one test that can falsify it today.

# 5. The diffusion approach — why, and the strategy

This is claim 3, and it is the part of the project with the most upside. The
argument below is the reason the hierarchical design exists; it is not a
post-hoc justification for a fashionable method.

## 5.1 The core argument: denoising is intrinsically multi-scale

**The denoising objective decomposes by scale, and autoregression does not.**

High-noise steps train coarse layout statistics. Low-noise steps train local
stroke shape. The noise level *is* a resolution knob, and the objective spends
capacity at each level separately.

AR over a serialised stream has no such decomposition. Every token is predicted
at the same resolution given a prefix — position 400 is predicted exactly the
way position 4 was, conditioned on more context but at identical granularity.
There is no mechanism by which an AR model allocates capacity to "layout" as
distinct from "curvature".

So diffusion genuinely extracts scale-separated structure that AR *structurally*
cannot. That is the strongest argument for diffusion in this project — stronger
than any parameter-efficiency claim — and it is what the factorisation
(non-autoregressive over strokes × autoregressive over bits) is built to exploit.

## 5.2 The asymmetry, and why it decides the dataset strategy

The noise schedule defines "coarse" **relative to the data's own scale.** This
is the crux.

Train on 5-stroke doodles and the top noise band means *"roughly where are these
5 strokes."* On a 50-stroke drawing the equivalent band means *"where are the 12
regions"* — a structural level that does not appear anywhere in the simple
corpus. No signal, no gradient, nothing to transfer.

Transfer therefore runs **one way**:

| Prior | Learnable from QuickDraw? | Transfers up? |
|---|---|---|
| **Local** — how a human makes a line, curvature statistics, stroke termination, joins | yes, abundantly | **yes** |
| **Global** — how 40 strokes organise into a composition | no, absent from the data | **no** |

The "inherent properties of drawing" intuition is the first row. It is real, it
is learnable from simple data, and it does carry upward. The second row is not
a data-quantity problem — it is a data-*level* problem.

## 5.3 The payoff: this dissolves the dataset worry

Because transfer is one-way and level-specific, **the hierarchical latent is not
merely an efficiency device — it lets the two levels be trained on different
datasets.**

- **Stroke-level autoencoder + local prior → QuickDraw.** 50M samples, free,
  and exactly the level QuickDraw is rich in.
- **Composition-level diffusion → Tier D.** Scarce, but the top level has far
  fewer effective degrees of freedom, so **Tier D only has to be small.**

That is the whole point: OpenSketch at a few hundred drawings becomes viable
rather than laughable. The scarce data is only ever asked to supply the level it
uniquely has.

## 5.4 What to hunt for — Tier D is composition-level supervision

Tier D is **not** "more training data." It is composition-level supervision
specifically. Priority:

1. **FS-COCO** — scene sketches, many strokes, per-stroke temporal order.
   Exactly the missing level. Highest priority.
2. **OpenSketch** — tiny, but maximally hierarchical (construction lines, then
   detail). The clearest available signal for coarse-then-fine structure.
3. **Deprioritise further object-sketch sets** (TU-Berlin, Sketchy). They are
   better-drawn QuickDraw: same structural level as what is already on hand,
   so they add samples without adding the level that is missing.

## 5.5 The hypothesis is testable today, with no new data

`dm/data/synthetic.py:LADDER` already spans `trivial → simple → busy → nested →
deep → elaborate`, varying nesting depth independently of stroke count. So:

> **Train on `simple`, evaluate on `elaborate`, and measure local and global
> fidelity separately.**

- **Local** — per-stroke reconstruction: match strokes between traces, centre
  each pair, and measure within-stroke shape distance.
- **Global** — coarse-scale layout: Chamfer on stroke centroids (equivalently,
  heavily downsampled raster IoU), which ignores stroke shape entirely.

**Prediction: local gap small, global gap large.**

**If both degrade equally, the levels are not separable and the hierarchical
design is wrong.** That is worth knowing *before* building it, and it costs one
training run against code that already exists.

What is missing to run it: `dm/eval/metrics.py` has only a single-scale
`chamfer` over a flattened point cloud (`trace_points` discards stroke identity)
and `iou` at size 128. The local/global split needs stroke matching plus those
two derived measures, and the AR model needs a way to produce *paired* traces at
all. §9.6 has the design decisions — prefix-conditioned completion, optimal
assignment, and what must not be folded into which number. It remains the
smallest piece of new code with the largest decision value in the project.
