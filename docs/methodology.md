# Methodology — settled, do not re-litigate

The ablation design, the corpora, what the reference papers are for, and the
argument behind claim 3's factorisation. Moved out of `PLAN.md` on 2026-08-10
when that file was cut to a resume point; nothing here changed in the move.
Section references of the form §N are to the pre-cut `PLAN.md`; the live
equivalents are named in `PLAN.md`'s file table.

## Methodology — settled, do not re-litigate

**Four independent ablation axes, deliberately not confounded.**

| Axis | Levels | Isolates |
|---|---|---|
| Typing | `token` vs `byte` | one free bit per position: *is this an opcode?* Vocab 269 vs 258, lengths identical |
| Granularity | `byte` vs `bit` | byte-boundary and field-width discovery. 8× length, same information |
| Fusion | `token` vs `token_typed` | whether more typing helps or just inflates the table (274 → 1554 at ISA v2). **Closed 2026-08-12** — see below |
| Relativity | `X` vs `X_delta` | canvas position vs offset from the pen. Same alphabet, vocab, length, information |

> **The fusion axis is closed, and it took a corpus built for it to close it.**
> On every L0-only corpus the two codecs encode a program byte-identically — one
> operand `Kind`, so there is nothing to split — which is a degeneracy rather than
> a null. ISA v2's `REPEATX` brings three kinds and the axis became live for the
> first time (0 of 200 val scenes identical). It then read **+2.71** at 12,000
> steps and **−2.42** at 24,000: sign-inverted across the budget ladder, both
> magnitudes inside that corpus's 0.4–4.7-bit seed spread, and the +5.4-bit
> prediction carried over from Tier A's per-byte rate refuted. **It closes
> permanently rather than provisionally** — it was given the conditions it had
> asked for and did not resolve. [`xform.md`](xform.md) §5.

Relativity is orthogonal to the other three by construction, so `CODECS` is a
4 × 2 grid. It is a *codec-level* view of an unchanged absolute bytecode, so the
VM, the ISA and the C port never see it. What it buys is measured: a translation
moves exactly **one coordinate pair** in the relative view against every
coordinate byte in the absolute one (`dm/isa/relative.py`).

- **Coordinates are 8-bit everywhere** and the relative view is an exact mod-256
  bijection, so all eight codecs encode identical information and there is no
  quantization confound. Roundtrip tests over every corpus pin this.
- **bits/drawing is the only cross-codec comparable metric.** *Per-token loss is
  never comparable across arms* — the bit arm's is low because its alphabet is 4.
- **Three budget regimes, not two.** Token-matched and step-matched answer "what
  does a fixed budget buy?"; only **converged** answers "what does this alphabet
  cost?". No budget rule can put two arms at the same place on their own curve
  when their sequence lengths differ 8×. `drift` and `tail` are printed per row
  so the precondition is checked, not assumed — and **only `budget_table`, the
  same arm at two budgets, can establish an asymptote** (§10).
- **Faults are reported, never raised.** Generated programs are malformed early
  in training, so validity is a measurement, and it is measured in one place (the
  VM) for all codecs.
- **Absolute coordinates in the bytecode.** `REPEAT` needs a well-defined
  coordinate transform and absolute keeps drift out of the VM.
- **The architecture is deliberately boring** — pre-norm, RMSNorm, RoPE, SwiGLU,
  tied embeddings — so any difference is attributable to the representation.
  RoPE is not style: the length-generalisation test needs positions never seen in
  training. (`Config.abs_pos` is the exception, for the planner's fixed grid — a
  grid of *fields* never extrapolates, so the argument reverses. §10.)

## Datasets

- **Tier A — synthetic** (`dm/data/synthetic.py`). Programs we generated, so
  recovery is scored against the *program*, not the picture. Hosts the
  length-generalisation test and the `depth` complexity instrument.
  **Build splits with `synthetic.split()`, never two bare `dataset()` calls** (§8).
  **`--flatten` is claim 2's corpus**: the L0 expansion, 100k programs, ceiling
  **21% of bytes at `n ≤ 4` and 68% at `n = 8..16`**, known by construction.
- **Tier B — QuickDraw**, official Google sketch-rnn `.npz` (stroke-3,
  pre-split). This is a deliberate subset of upstream's 50 million drawings /
  345 categories: **five categories, 375,000 source drawings** at 70k train +
  2.5k valid + 2.5k test each. Headline runs consume all 350,000 train drawings
  and a balanced 1,000-drawing prefix of the 12,500 available validation rows;
  test remains untouched. Live conversion is **`rdp_eps=4.0`, `max_len=3072`**,
  112.7 bytes/program, zero truncation on all four codecs. `rdp_eps` decides whether the bit arm fits
  under `max_len` at all — gate any change with `scripts/quickdraw_check.py`.
  [`docs/tier-b.md`](tier-b.md)
- **Tier C — Tabler Icons.** 5,130 MIT outline icons on a **24×24 grid**, so
  `SCALE = 10` onto the 256 canvas is exact. **19.7% of icons carry a compressible
  `REPEAT`, 3.79% of bytes.** 4,613 programs cannot support an asymptote, and
  augmentation was tried and refuted. **Closed as a training venue; sound as an
  oracle.** [`docs/tier-c.md`](tier-c.md)
- **Tier D — FS-COCO, acquired, not ingested.** 10,000 scene sketches, ISA-native
  `(N,3)` int64 on a 0–257 canvas. **2,440 points and 62 strokes per sketch
  against Tier B's ~50 and ~9 — that is the point**, and it also means it cannot
  enter the flat AR pipeline at any usable `rdp_eps`. §9.9 says what to build.

The tiers are not a quality ladder — they are *different structural levels*, and
that distinction decides which dataset trains which part of the model
([`docs/diffusion-strategy.md`](diffusion-strategy.md) §5.3).

## What the papers in `references/` are for

**Scaffolding, not constraints** — technique and failure modes, not permission.
Where this project's reasoning diverges from theirs, this project's wins. Two are
under active test rather than cited: **IconShop** (fused `(op, x, y)`
tokenisation — at `d=256` a 4096-entry table is 1.05M parameters, the entire
budget before a single layer; the `token_typed` arm is that objection made
measurable) and **StrokeNUWA** (learned stroke tokens — same objection plus a
second, that a learned tokenizer is a model whose parameters are rarely counted).

> Honest status: nothing here has been numerically replicated, and nothing should
> be replicated for its own sake. Roster: [`docs/references.md`](references.md).

## The diffusion approach — the argument in four sentences

**The denoising objective decomposes by scale and autoregression structurally
cannot.** High-noise steps train coarse layout, low-noise steps train stroke
shape; AR over a serialised stream predicts position 400 exactly the way it
predicted position 4 — more context, identical granularity. Two consequences the
plan depends on: **transfer runs one way** (local priors are learnable from
QuickDraw and carry upward; global priors are absent from it), and therefore
**the two levels can be trained on different datasets**.

> **§9.6 tested the premise and it holds** — local skill exceeds global at every
> rung and global collapses to chance on the two hardest while local survives
> (`elaborate`: global +0.004, local +0.290). **The premise survives claim 3's
> result**: what failed was this factorisation's ability to *exploit* the
> separation at 825k parameters, not the separation.
> [`docs/scale-separation.md`](scale-separation.md)
