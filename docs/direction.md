# Direction — compositionality as an ISA change

> **Historical direction record.** This file preserves the 2026-08-10 through
> 2026-08-12 proposals and their in-place verdicts. It is not the live roadmap;
> current paper positioning, priorities and implementation gates are in
> [`docs/directions.md`](directions.md).
>
> **Status: argument, not evidence — except §3, which has now been run.**
> `docs/evidence.md` keeps a strict separation between what is defensible, what
> was withdrawn, and what is only an argument; **everything here belongs in the
> third category** until a run says otherwise. It is written down in advance so
> that it can be wrong in public, which is the same discipline that made claim
> 3's falsifier worth having.
>
> Written 2026-08-10, after claims 1–3 closed and claim 4's static half landed.
> **§3 was measured on 2026-08-11 and §2 on 2026-08-12; both verdicts are recorded
> in place**, in the sections themselves, against the branches this file named
> before the runs. §4 and §5 are still arguments, and **§7 — added 2026-08-12 —
> starts by saying which demo any of this is for, because a compression result had
> begun to be read as a generation one.**

Section references of the form §N are to `PLAN.md`.

---

## 1. The premise this starts from, and the correction it needs

The obvious reading of the project so far is "QuickDraw produced the useful
results, so we need richer corpora — scenes, SVGs, drawings with intent." The
record does not support that reading.

**QuickDraw is not the best corpus. It is the only one that converged.**

| corpus | why it stopped paying |
|---|---|
| Tier A — synthetic | capacity-saturated. A 4,812,544-parameter arm converges to 160.71 and *loses* to the 792k bit arm by 0.49 bits. 6× the parameters cannot cross the band its four codecs sit in |
| Tier C — Tabler icons | **no budget exists where both convergence guards pass.** 4,613 programs; val minimum at ~1,000 steps (13.9 epochs) and 1,500 is already worse. Augmentation ×18 was tried and refuted |
| Tier D — FS-COCO | 62 strokes and 2,440 points per sketch → 5,195 bytes flat; **58% over `max_len` even at `rdp_eps=16`.** Cannot enter the flat AR pipeline at any usable setting |

So the constraint is not *richness*. It is that this instrument resolves effects
only on a corpus with ~100k programs and a converged denominator, and only
QuickDraw has one. Reaching for a scene dataset does not fix that; it reproduces
Tier C's failure at larger scale.

**The move is the one that already worked twice.** Tier A flat (claim 2's corpus)
and the Tabler repeat oracle both work because the redundancy is *known by
construction* — the generator chose the count and the step, so the compression
ceiling is exact rather than estimated. Do the same here: **construct a
compositional corpus out of QuickDraw**, where "a cat above a bus, mirrored" is
two known programs under two known transforms, and the ceiling is arithmetic.

---

## 2. Proposal A — `XFORM`, and compositionality as an ISA change

> **MEASURED 2026-08-12, and this section is kept as written.** P1-P5 landed; the
> results, every table and the three things this argument got wrong are in
> [`xform.md`](xform.md). In one line: **the model discovers the orbit
> (`recovery` +0.960), the opcode folds 40.9% of the bytes and 1.2% of the bits,
> and where it actually pays is termination.** §2.6's first falsifier fired and
> could not settle what it was written to settle, which is its own entry in the
> ledger. §2.5's claim that the tier "revives the fusion axis" came true in the
> literal sense — the axis exists on the structured corpus — and then the axis
> resolved as unresolvable and closed for good.

The intuition in the original question was right and slightly undersold: *"the
instructions would have to begin with some instruction that tells the hardware
to, for all executed, do it upside down."* That is not a modelling idea. It is an
opcode, and most of the machinery for it already exists.

### 2.1 The transform group is forced, not chosen

`dm/data/augment.py:Affine` is already exactly the right group — `dx`, `dy`,
`mirror`, `quarter_turns` — and the reason it is that group and not a larger one
is measured, in `scripts/repeat_oracle.py` against Tabler's ceiling:

| transform | effect on exact repeats |
|---|---|
| integer translate / mirror / 90° rotate | **preserved exactly** |
| scale ×1.07 | destroys **77%** |
| jitter ±1 | destroys **65%** |

So the group is **D4 ⋉ integer translation**: 8 symmetries and a translation, and
nothing continuous. Anything else reopens the quantization confound the 8-bit
ISA exists to eliminate (§2: all eight codecs encode identical information, and
roundtrip tests over every corpus pin it). This is a case where a design
constraint that looks arbitrary is a measurement.

`Affine.apply()` already **rejects** a drawing that leaves the canvas rather than
clamping it, for the same reason: clamping is a deformation, and a deformation of
a repeated body is not a repeat.

### 2.2 `CALL` was allocated for this and never built

`dm/isa/spec.py` already carries:

```python
InstrSpec(Op.CALL, "CALL", (Kind.ID,), Tier.L2)   # "learned library"
class Tier(IntEnum):
    L2 = 2        # learned library: CALL
```

and `dm/vm/interp.py` faults it `CALL_UNSUPPORTED`. **"An X on top of a Y" is
literally `CALL X; XFORM …; CALL Y`.** The tier was allocated for compositionality
in the original design and the project never reached it.

### 2.3 Sketch

Two new opcodes, both `Tier.L2`:

```
XFORM  (Kind.COUNT quarter_turns|mirror packed, Kind.DELTA dx, Kind.DELTA dy)
ENDX   ()
```

- `XFORM` pushes an `Affine` onto a transform stack; `ENDX` pops it.
- The VM composes the stack and applies it inside `place()` — **the single funnel
  every coordinate already passes through**, in both `dm/vm/interp.py` and
  `port/src/dm_vm.c`.
- Composing two D4 elements is an 8×8 table lookup. No multiply, no division.

### 2.4 What it costs on the device, predicted from figure 5

Figure 5's fitted cost model is
`instructions ≈ 220 + 49.6·(decoded) + 17.9·(points) + 33.9·(discs)`, so the
prediction is specific and falsifiable: **`XFORM` adds ~nothing to the decode
term and a small constant to the per-point term**, while saving bytes
proportional to the size of the transformed object.

The honest cost is **RAM, and it is not free**. The interpreter's measured peak
stack is 288 bytes with a fixed `DM_MAX_REPEAT_DEPTH = 4` frame array. A
transform stack of the same depth is ~6 ints × 4 = ~96 bytes, which is roughly a
third again on top of the measured figure. That has to be re-measured with
`scripts/footprint.py`, not assumed — and if it pushes the interpreter past a
budget the project cares about, the depth bound is the knob.

> **Measured 2026-08-11: 492 bytes, so this estimate was low by 2×.** The scope
> array is the 48 bytes predicted; the rest is what a sketch cannot see — the
> cached composition the per-point path reads (12), a step and a scope floor in
> every loop frame (16), and compiler spills from a wider dispatch and a
> `place()` that returns two coordinates because a rotation mixes them. Flash
> went 1,365 → 1,862. **The knob was not needed**: peak stack is still a constant
> and 492 B is 3% of the smallest target's SRAM.

### 2.5 Why this is the strongest item on the list

- **Every existing instrument fits it.** `recovery_by_copy` measures bits saved
  on repeated copies; the same code measures bits saved on *transformed* copies.
  `dm/eval/repeats.py` finds exact translational repeats; extending it to eight
  symmetries is mechanical. A claim-2-shaped result is available on day one, with
  a ceiling known by construction.

  **Built and measured 2026-08-11**, and it produced a result this section only
  guessed at: on Tabler's icons the transform tier folds **11.74%** of bytes
  against `REPEAT`'s **3.97%** — 3× — with 71% of the orbits quarter or
  three-quarter turns. §1's "radial symmetry is invisible here however obvious it
  looks" is now a number rather than an observation, **on a natural corpus and
  with no model involved**. [Figure 6](figs/fig6_orbits.svg)
- **It revives the one axis that degenerates everywhere.** `token` vs
  `token_typed` is byte-identical on **every L0-only corpus** (proven by encoding
  identity on 1000/1000 and 512/512 val programs, not inferred), so fusion has
  only ever been measurable on Tier A. `XFORM`'s operands introduce new `Kind`s
  and the axis becomes live on a corpus that converges.
- **It is the only proposal with a claim-4 consequence.** A transform prefix is
  nearly free to execute and saves bytes proportional to object size. That is a
  compression result with a hardware consequence, which is this project's whole
  thesis rather than a side effect of it.

### 2.6 Falsifiers, stated in advance

- **If `recovery` on transformed copies is at or below the translation-only
  number (+0.807), the transform tier buys nothing the model could not already
  do**, and `XFORM` should be recorded as a measured null rather than tuned.
- **If the compositional corpus's fusion axis still reads inside its floor**,
  then fusion is unresolvable at this budget for reasons that have nothing to do
  with operand kinds, and §1.5's closure of that axis stands permanently.
- **If peak stack exceeds the budget** the transform depth has to shrink, and a
  depth-1 transform stack is a different and much weaker proposal.

> **How they read, 2026-08-12.** The second fired and closed its axis: +2.71 at
> 12,000 steps and −2.42 at 24,000, sign-inverted across the ladder and inside a
> 0.4–4.7-bit floor. The third did not fire: 492 B peak stack, 12 B per scope.
>
> **The first fired and licensed nothing, and the fault is in this bullet.** Two
> defects, both visible only in hindsight and both worth keeping on the page.
> *The comparison is against the wrong thing*: +0.807 is Tier A synthetic, which
> `dm/eval/recovery.py` refuses outright, and P5 had to build a matched
> translation-only control to have a legal reference at all. And *the reference is
> the wrong quantity even when it is legal*: the claim is whether a model can
> discover transformed reuse, so what answers it is the distance from paying full
> price — 96–98% of the way, at every arm and seed — and not a 1–2% shortfall
> against a control that is itself the noisier of the two sides. **The clause this
> bullet was reaching for, "the tier buys nothing the model could not already
> do", is not a question about a recovery *difference* at all.** It is settled by
> recovery's absolute level, because the level fixes how much redundancy is left
> for an opcode to remove, and [`xform.md`](xform.md) §3 measures the remainder:
> 11 bits of 479, netting 6 after the opcode pays for its own header.

### 2.7 Traps this will walk into

- **Mirroring reverses stroke order** under some conventions.
  `dm/isa/strokes.py`'s `join(split(p)) == p` invariant is load-bearing for claim
  3's parameter-matching argument and must survive, byte for byte.
- **`XFORM` and `REPEAT` both transform coordinates**, and they must compose in a
  defined order or a nested program means two different drawings. `REPEAT`'s
  per-iteration offset is a translation; the transform stack is the general case;
  the composition order is a specification decision to write down *before* the
  first run, not after.
- **The corpus key must see the transform policy.** A constructed corpus is
  defined by its composition rules, and a config-derived key cannot see a
  default (§10). `dm/data/fingerprint.py` digests the programs themselves, which
  is what makes this safe — but only if the constructed programs go through it.

---

## 3. Proposal B — the redundancy test. **Run 2026-08-11; the verdict is in §3.4.**

**This one requires no training and decides what claim 3's negative result
means.** It should be run before anything else in this file.

### 3.1 The mechanism

The planner factorises as `p(s) · p(x | s)` evaluated at `s = f(x)`, where the
summary is a **deterministic function of the stroke** (`dm/isa/strokes.py` — that
determinism is what makes the bound a bound). So the scheme transmits the
summary's information **twice**: once explicitly as `p(s)`, and once implicitly
inside `p(x | s)`, because a decoder that has already been told the centroid, the
extent and the endpoint should not need to spend bits re-specifying them.

**If the stroke decoder is not exploiting what the summary already pins, that
redundancy is a large part of the 39–49 bits.**

The circumstantial support is in the budget rung: the composition level is
**exact and at its asymptote** (181.68, +0.23 over the second half of its
schedule) while the stroke decoder still buys **−5.73 / −5.81 per doubling**. All
remaining appetite is in the half claim 3 does not contribute.

### 3.2 The measurement

Off existing checkpoints, no training: score the stroke decoder's bits on the
positions the summary already determines — first point, extent, endpoint — and
compare against its bits on the free interior positions.

- **Near zero on the determined positions** → the decoder already exploits the
  conditioning, the redundancy hypothesis is dead, and claim 3's loss is a real
  modelling loss. That is a *stronger* negative result than the one on record.
- **Substantial** → the loss is a coding inefficiency in this implementation of
  the factorisation, not in factorisation as such, and the fix is to code the
  **residual** rather than a better composition model.

Either branch is publishable and the second changes what the project would build
next. Cost: hours.

### 3.3 What would make it uninformative

If the summary's determined positions are a negligible fraction of stroke bytes
on this corpus, the test has no power and should be reported as such rather than
as a null. Check the fraction first; it is arithmetic on `dm/isa/strokes.py`.

**Checked first, and it passes.** On the planner's own val split — 1,000
programs, 6,295 strokes, 25.56 bytes each — **8.6% of all stroke symbols have
exactly one legal value given the summary**: every stroke opens with `MOVE` and
its next two bytes *are* `(x0, y0)` byte for byte (6,295 of 6,295), and every
program's `HALT` is the last byte of a halting stroke (1,000 of 1,000). At
~415 bits/drawing for the stroke half, full price on those positions alone would
be ~32 bits, the same order as the gap under test. So a null here is a null and
not an absence of power.

### 3.4 The result — the hypothesis is refuted where it was sharpest, and a smaller real inefficiency survives

Measured 2026-08-11 with `dm/eval/redundancy.py` and `scripts/redundancy.py`,
one forward pass per arm over the same 1,000 val programs, no training. Every
rule is **exact**: a value is called infeasible only when no program consistent
with the summary could have put it there.

| bits/drawing recoverable | planner, AR comp | planner, diffusion comp | **flat AR** (never told the summary) |
|---|---|---|---|
| **summary, total** | **8.632** | **9.851** | **112.739** |
| `first_point` | 0.030 | 0.023 | 84.502 |
| `coord_box` | 8.598 | 9.823 | 24.611 |
| `halt` | 0.005 | 0.004 | 3.626 |
| `fits` | 0.001 | 0.001 | 2.780 |
| ISA alone (the baseline, subtracted out) | 0.002 | 0.002 | 0.004 |
| bits/symbol where one value is legal | **0.003** | 0.002 | **6.389** |
| bits/symbol everywhere else | 2.862 | 2.871 | 3.230 |

**Where the summary determines the byte, the hypothesis is refuted with an
upper bound.** The decoder spends **0.035 bits/drawing in total** across the
13,786 positions the summary pins — not "a small fraction of the gap" but
0.09% of one bit. Nothing can be recovered there because nothing is being spent
there. The flat arm spends **88.15** on the identical positions, a ratio of
2,500×. §3.2's first branch is the one that fired: **the decoder already
exploits its conditioning, and claim 3's loss is a real modelling loss.**

**Where the summary constrains without determining, a real inefficiency
survives, and it is a lower bound rather than an upper one.** Renormalising onto
the extent box recovers **8.63 / 9.85 bits/drawing** — 2.1–2.4× the planner's
4.16-bit replicate floor, so resolvable, and **18–25% of the 38.75–48.97 bit
gap**, so not an explanation of it. A cleverer coder exploiting more of what the
summary implies could recover more; this is what *these* exact rules recover.

**The control is what makes the first paragraph mean anything.** Paired over the
same 1,000 val programs, the flat arm wastes **+104.11 ±3.71 bits/drawing** more
than the planner's decoder — and the asymmetry runs the conservative way, since
the flat arm sees every previous stroke of the program while the planner's
decoder sees only its own stroke plus six bytes. The conditioning is not
decorative; it is being used, and heavily.

> **And the two composition objectives separate again, in the same direction.**
> The diffusion arm wastes 1.22 ±0.13 bits/drawing more than the AR arm on the
> identical constraints (paired, per program). That is the third axis pointing
> the same way as `mmd` and `nna` — but it is **below every floor this project
> can quote** for a model-to-model difference, so it is a consistency check and
> not a result.

**What this changes downstream:** §3.2's second branch — "code the residual" —
is not the next build. The residual it named is 0.035 bits. The one real target
this leaves is the coordinate extent, worth <10 bits against a 39–49 bit gap,
and that is a small enough prize that **§2's `XFORM` remains the next item**.

---

## 4. The other directions, weighed against what is already measured

### 4.1 SDE / continuous-time reformulation — do not

Reformulating masked diffusion as an SDE is elegant and, here, optimises the
component that was measured not to matter.

**Diffusion and autoregression at the composition level tie**: `gen_length_emd`
4.59 ±0.98 against 3.78 ±0.97 at 24,000 steps, and 5.63 ±3.33 against 4.42 ±0.65
at 12,000 — the AR control is nominally *ahead* at both and inside the spread at
both. What beats the flat arm is **having a composition level**, not denoising.
§1.5's rule applies directly: stop spending runs on sub-floor axes.

Two further reasons: the composition grid is 32×6 **discrete** tokens, where a
continuous-time SDE limit buys nothing structural; and the diffusion arm carries
ELBO slack (6.43 bits) that the AR arm does not, so it starts a comparison
already behind on the only axis that is exact.

### 4.2 Samplers, though, are high-leverage — and there is a test already specified

Sampling is where the largest single effect in claim 3's history was found:
changing the unmasking order moved `gen_length_emd` from **25.08 ±1.80 to 4.59
±0.98**. That was a correctness fault rather than a tuning knob — confidence
ordering is not the reverse process of a masked diffusion, and on a sparse grid
it is a ratchet.

`docs/traps.md` already names the follow-up: **the test for a biased sampler
rather than a coarse one is that more steps make a biased one monotonically
worse.** Cheap, off existing checkpoints, and it either closes the sampler
question or reopens it with evidence.

### 4.3 Reconstruction / denoising autoencoders — admissible only with the count

The project's standing objection to StrokeNUWA is that **a learned tokenizer is a
model whose parameters are rarely counted**, and at 825k a codebook consumes the
budget before a single layer. §9.12 permits a pre-trained encoder only as a
*training-time scaffold that is distilled away*, and requires any such claim to
report both parameter counts.

Beyond the rule, the prior is poor: across four corpora, **a 5.8–6.1× larger
model was better exactly once**. Spending parameters on a latent space is
arguing against nine measurements unless the argument is specifically that the
parameters buy *structure* rather than capacity — which would need to be the
stated hypothesis, with a falsifier.

### 4.4 Contrastive / triplet losses — right target, wrong mechanism, and a free baseline is being skipped

The target is real and is the project's largest genuine blind spot:
**bits/drawing cannot see sample quality**, the project has never reported the
second half, and a preliminary signal exists — over 96 samples the flat AR arm is
**over-dispersed in stroke count** (±5.85 against the corpus's ±3.88) while both
planners sit slightly under (±3.14, ±3.19), which points the same way as the
termination result. A crude pairwise-distance diversity proxy is **inside its own
noise at one seed** and is not quotable.

The mechanism is the problem. Contrastive learning shapes an **embedding space**;
this project's output is a **program**. The only place a metric could live is the
stroke-summary grid, which is a narrow and indirect lever on "pointy ears rather
than round ears".

**And there is a much simpler thing being skipped: QuickDraw has category labels
and this project throws them away.** Class-conditional generation is the standard
way to get "draw a cat", it costs an embedding row, and it is the *prerequisite*
for measuring whether the ears came out pointy at all. Do that before reaching
for a triplet loss.

---

## 5. Order, cost, and what each one settles

| # | work | cost | what it settles |
|---|---|---|---|
| 1 | ~~**Stroke-decoder redundancy test** (§3)~~ **DONE 2026-08-11** | hours, no training | **Settled: a real modelling loss.** ≤0.035 bits/drawing is recoverable where the summary determines the byte; 8.6–9.9 survives on the extent box, 18–25% of the gap |
| 2 | ~~**`XFORM` + constructed compositional corpus** (§2)~~ **DONE 2026-08-12** | the real project | **Settled**: it discovers transformed reuse at +0.960, the fold is worth 40.9% of bytes and 1.2% of bits, and the payoff is generation reliability — 48/48 samples keep an orbit against the flat spelling's 13%. The fusion axis was revived and then closed. [`xform.md`](xform.md) |
| 3 | ~~**Bézier control points** (§7.3)~~ **DONE 2026-08-13** | an afternoon, no training | **Settled: no.** Retention **61.5% ±9.8%** against a matched polyline's **42.3% ±3.8%**, 24/24 paired rounding phases — the endpoint fires and 61.5% is still not lossless, on a ceiling 38% smaller, so scale stays out. What survives is **−13.4%** bytes at matched fidelity on icons and a null on QuickDraw. The 23% on record was one rounding phase of a 42.8% ±2.9% distribution. [`bezier.md`](bezier.md) §5 |
| 4 | ~~**Per-field bit attribution** (§7.2)~~ **DONE 2026-08-13** | hours, no training | **Settled, and it closed §7.2 and decomposed claim 3.** An architectural prior supplying the ISA grid has a ceiling of **0.0008–0.0085 bits/drawing** on four alphabets and three corpora. Claim 3's +39.8 is **−141.9 on the stroke half against +181.7 for the plan**, and two thirds of the −141.9 is the `MOVE`. [`attribution.md`](attribution.md) |
| 5 | ~~**Class conditioning** (§4.4)~~ **DONE 2026-08-12** | ~a day | **Measured**: `I(X;C)` = 2.25/2.28 of the 2.32-bit ceiling, controllability **512/512** — the demo prerequisite holds, and the next step it names is class-conditional `coverage`/`mmd`/`nna`. [`conditioning.md`](conditioning.md) §5 |
| 6 | ~~**Style canonicalisation** (§7.1)~~ **DONE 2026-08-13** | an afternoon | **Died, as designed.** The proposed test is a provable null for exact policies (three ceilings byte-identical) and the inexact ones lose −6.75 points of cross-drawing reuse. By-product: the **`Op.CALL` ceiling**, 25.21% of Tabler's bytes against `REPEATX`'s 11.74%, and 0.00% on QuickDraw. [`bezier.md`](bezier.md) §6 |
| 7 | **Sampler-steps monotonicity** (§4.2) | hours | biased sampler vs coarse sampler, already specified |
| 8 | ~~Convolutional front end (§7.2)~~ **CLOSED 2026-08-13, unbuilt** | days | **Answered without building it.** The mechanism it was proposed for — supplying the ISA's field grid — has a measured ceiling of **0.0008–0.0085 bits/drawing**. What that does *not* refute is a parameter-sharing argument, which would be a different claim needing its own falsifier. [`attribution.md`](attribution.md) §4.1 |
| 9 | Contrastive / autoencoder (§4.3, §4.4) | large | only after class conditioning, only with parameters counted |

---

## 7. Added 2026-08-12 — the demo these are all really about, and three proposals

### 7.0 What the project measures and what a demo needs are not the same thing

Two demos are worth wanting, and they were stated plainly: **(a) I draw one, the
machine draws something similar but new, with its own quirks; (b) I say "a person
on a cat" and it draws something plausible.** Neither is what any run in this
project optimises, and saying so precisely is more useful than any of the
proposals below.

- **Every number here comes from a compression objective.** `bits/drawing` is the
  cost of transmitting a drawing, and claim 3 measured that likelihood and sample
  quality *disagree*: the arm that won on termination lost on fidelity, and the
  arm with the best `nna` was not the best on `mmd`. A roadmap toward either demo
  is therefore **not** the roadmap that has been followed, and the gap is not a
  matter of more steps.
- **The constructed compositional corpus is an instrument, not a step toward
  either demo.** It exists to answer one mechanism question — can a model discover
  that one shape is another one mirrored — and [`xform.md`](xform.md) says it can.
  It cannot say "cat", because it has **no labels**; its compositions are one
  motif copied under a transform rather than **two different bodies** placed in
  relation, so "person on a cat" is not expressible in it at all; and its scenes
  are deliberately *implausible* — a sketch, its mirror, and unrelated sketches
  dropped in the free quadrants — because a plausible scene would have confounded
  the measurement with semantics.
- **What (a) actually needs** is prompt-conditioned continuation, and the record
  has one measurement that bears on it: `elaborate` transfers **local** structure
  at +0.290 and **global** structure at +0.004, which is chance. So a completion
  demo today would continue a stroke plausibly and lay out the drawing
  implausibly. That is a specific, testable failure and it is the thing to fix.
- **What (b) actually needs**, in order: class conditioning (~~labels are free
  and are currently thrown away~~ **done 2026-08-12, both halves** —
  controllability 512/512, and the geometry half against real drawings:
  48 of 50 class-draws land nearest their own class, by 1.08–3.27× the margin a
  perfect generator gets, with `mmd` at floor, [`conditioning.md`](conditioning.md)
  §§5–6), `Op.CALL` with **two distinct bodies** (still allocated at `Tier.L2`
  and still unbuilt — `REPEATX` deliberately did the *one*-body case because its
  ceiling is arithmetic), and a relation vocabulary for "on top of". None of the
  three is expensive; the first is done and two are unstarted.

  > **And the first one arrived with the demo's real obstacle attached.** The
  > label controls *which* class the geometry belongs to, completely. What it
  > does not buy is *variety within* the class: `coverage` reaches 43% of the
  > way to its floor and `nna` 61%, both in the direction of samples clustered
  > on class-typical geometry ([`conditioning.md`](conditioning.md) §6.2). So a
  > "draw me a cat" demo today draws a recognisable cat and draws **the same
  > cat**, which is a different failure from the one this section predicted and
  > a more visible one on screen. **Measured 2026-08-13:** relaxing
  > `top_k=40→80` moves coverage 0.406→0.487 and `nna` 0.639→0.549 toward their
  > floors in 5/5 draws without moving MMD off floor. Full support crosses boundary:
  > first four draws overshoot coverage, worsen MMD/separation, then fifth emits
  > one empty; `T=1.2` fails on another empty. `k80,T1` is best tested conditional
  > operating point. Planner stroke decoders do not inherit diversity gain;
  > full support only worsens their MMD. [`conditioning.md`](conditioning.md) §7.

**So the honest status is that `XFORM` answered a question about mechanism and
moved the demo no closer.** That is what it was for, and it is worth being blunt
about the distinction rather than letting a compression result be read as a
generation one.

### 7.1 Canonicalising drawing style as preprocessing

> **MEASURED 2026-08-13, and the section is kept as written.** The verdict is in
> [`bezier.md`](bezier.md) §6. In one line: **the test proposed below is a
> provable null for every policy that could safely be adopted, and every policy
> that could move it destroys the thing it was measuring.** An exact re-framing
> carries the set of foldable repeats across bijectively, so both within-program
> ceilings are invariant *exactly* — 3.969852% → 3.969852% and
> 11.742708% → 11.742708% on 300 Tabler icons, 200 of which the policy changed.
> An inexact one (principal-axis rotation, aspect fill) is the ×1.07 operation
> under another name and costs **−6.75 points** of cross-drawing reuse and
> −2.1 points of repeat ceiling. The one column a re-framing *can* move needed an
> instrument that did not exist — reuse **between** drawings,
> `dm/eval/library.py` — and there it reads 17.92% → 19.58% at 300 icons and
> falls at 120. The idea dies, as the second branch below says it should, and it
> leaves the `Op.CALL` ceiling behind as the by-product (§6.4: **25.21%**, 2.1×
> what `REPEATX` folds).

**The instinct is right and the project already does a weak version of it**: a
composed motif is scaled **once** on the way in (`quickdraw.load`'s `margin`), and
everything downstream is integer. The generalisation is to canonicalise
*orientation and aspect* as well — align each sketch to a standard frame before it
enters the corpus, so that two people drawing the same object at different tilts
produce more similar programs.

**The hazard is already measured and it decides where this can go.** Non-integer
geometry destroys exact repeats: a ×1.07 scale removes **77%** of Tabler's repeat
ceiling and ±1 jitter **65%**, because rounding commutes with integer translation
and with nothing else. So canonicalisation is admissible **only** as a one-time
normalisation applied before the corpus is fixed, and **never** as augmentation of
a corpus whose structure is being measured. A vertical/horizontal compression is a
non-uniform scale and is exactly the operation that costs 77%.

**And it can be tested for the price of an afternoon, with no training at all**,
which is why it ranks where it does: canonicalise, then ask the *oracle* whether
the corpus's repeat ceiling went **up**. `scripts/repeat_oracle.py` and
`scripts/orbit_oracle.py` already produce that number for five corpora.

- **Ceiling rises materially** → style variation is what hides reuse in natural
  drawings, the preprocessing is worth having, and the ISA's compression argument
  gets a second corpus.
- **Ceiling flat** → style variation is not what hides reuse, and the idea dies
  before any model is trained.
- **Ceiling rises but only under a non-integer rotation** → self-defeating; report
  the rounding loss beside the gain or the number is fiction.

### 7.2 Locality as an architectural prior rather than a learned one

> **MEASURED 2026-08-13, and this section is kept as written.** Verdict in
> [`attribution.md`](attribution.md). The instrument at the end of this section —
> per-field bit attribution — was built and it **answers the proposal at the top
> of it**: the bits an arm spends on symbols that *cannot occur at that position*
> are exactly what an architecture supplying the ISA grid could recover, and they
> total **0.0008–0.0085 bits/drawing** across four alphabets and three corpora,
> or about one part in fifty thousand of the budget. The expectation below was
> "a conv front end buys about what typing buys, which is little"; it buys four
> orders of magnitude less than typing. Even `Kind.XF` — 8 legal values of 256,
> the narrowest field in the ISA, which no alphabet here narrows — wastes 0.0049
> bits. The `bit` arm, the only one never told where a byte begins, spends
> **exactly 0.000 bits on the top six bits of every opcode byte**. The stronger
> reading below was the right one to act on, and the instrument it named
> delivered a decomposition of claim 3 as a by-product: the factorisation wins
> **141.9 bits** on the stroke half and pays **181.7** for the plan, and 65% of
> that win is the `MOVE` alone.

The suggestion comes from a stacked-TCN sketch classifier over fixed-length
coordinate tensors. **The representation part of that does not transfer**: padding
a sketch to a constant-length coordinate vector throws away the property this
whole project rests on — a drawing is a *program*, its length is information, and
a VM executes it. What transfers is the **inductive bias**: dilated causal
convolutions buy hierarchical locality cheaply.

Where that would fit here is a convolutional front end over the byte stream, so
that "an instruction is 1–4 bytes" is architecture rather than something the model
infers from position. **And the project already has a prediction for what that is
worth**, because giving the model exactly that information *for free* is the
typing axis: `token` against `byte` reads −0.67 on Tier A and +11.58 on Tier B
multi — corpus-dependent, and inside the floor on the corpus where the model has
capacity to spare. So the pre-registered expectation is that a conv front end buys
about what typing buys, which is little, and it must be read against the corpus's
own resolution floor or it will look like a result.

**The stronger reading of the same instinct is the one to act on.** What has
actually produced results in this project is *instruments that read a model
against the ISA's own structure*: `recovery` against a constructed ceiling,
`redundancy` against exact feasibility rules, `spelling` against position classes,
the orbit oracle against provenance. Every one of them was hours of work, needed
no training, and changed a conclusion. The next member of that family is **per-field
bit attribution** — how many bits an arm spends on opcodes, on `x`, on `y`, on
`DELTA` operands, and where it is wasteful — which is `dm/eval/redundancy.py`
generalised from the planner's stroke decoder to every arm and codec. That is
where "the system looking within itself" has paid, and it is the cheapest thing on
this list.

### 7.3 Bézier control points as the geometry primitive — the most interesting of the three

> **MEASURED 2026-08-13, and this section is kept exactly as written.** Full
> reading in [`bezier.md`](bezier.md) §5. **The endpoint fires and the ISA still
> gets nothing.** A control-point corpus retains **61.5% ±9.8%** of its repeat
> ceiling under ×1.07 against a matched polyline's **42.3% ±3.8%**, in 24 of 24
> paired rounding phases — but 61.5% is not lossless, so scale stays out of the
> transform group, and the arm retains a larger share of a **38% smaller**
> ceiling so it ends with *fewer* foldable bytes than the control. Three things
> below were wrong and are worth naming: the exponent (`n = 4` is unreachable —
> the oracle folds instruction blocks, and the measured body goes 18 → 15, not
> 40 → 4), the independence assumption (refuted, in the favourable direction:
> control points inside one fitted span round *together*), and the **23% itself**
> — that is one rounding phase of a distribution whose mean is 42.8% ±2.9%.
> What survives is claim-1-shaped: **−13.4% bytes at matched fidelity on icons,
> and a null on QuickDraw at every tolerance.** And the constructive answer this
> section did not anticipate: the only scale that can enter this ISA is an
> **exact integer ratio on a corpus stored with canvas headroom**, which is a
> corpus-design decision and has nothing to do with the primitive.

`Op.CURVE` exists and is almost unused: strokes are polylines produced by RDP
simplification, which is a lossy fit chosen because it is cheap. A cubic Bézier is
a linear combination of four control points against a fixed basis matrix — locality
aware, standardised, and **affine-invariant**, which is the property that matters
here. Three things it would buy, and the third is why this ranks first:

1. **Length.** Fewer bytes per stroke at equal visual fidelity, measurable with no
   training: fit cubics, report bytes/drawing and the reconstruction error against
   the rasterised original. Sequence length is what killed Tier D and what forces
   `rdp_eps`, so this is a real lever on a real constraint.
2. **Exact commutation with the transform group.** An affine image of a Bézier is
   the Bézier of the transformed control points, so `XFORM` and `REPEATX` would act
   on **four numbers per segment instead of forty**.
3. **And that is what could put *scale* inside the group.** Scale is currently
   excluded by measurement, not by taste: ×1.07 destroys 77% of the repeat ceiling
   because every coordinate rounds independently and a repeat needs all of them to
   agree. Model that as an independent per-coordinate divergence probability `p`
   over an `n`-number body: `(1 − p)^n = 0.23` at `n ≈ 40` gives **`p ≈ 0.036`**, and
   the same `p` at `n = 4` gives **`(1 − p)^4 ≈ 0.86`**. So the prediction on record,
   before anything is built, is that **a control-point representation should retain
   ~86% of the ceiling under a ×1.07 scale where a polyline retains 23%.**

**The falsifier is the same measurement**: fit Béziers, scale the control points by
1.07 with integer rounding, re-run the oracle. If the ceiling still collapses, the
independence model is wrong, scale stays out of the ISA, and the Bézier idea
reduces to a compression win — which is claim 1 rather than claim 2, still worth
having and a much smaller prize. **If it holds, "a cat twice the size" becomes
expressible in the ISA**, which is materially larger than anything `XFORM`
established.

> **Two traps this walks into, both load-bearing.** The VM's curve geometry is
> integral at a scale of `CURVE_STEPS³` (claim 4), which is what lets the C port
> compare with `==` and carry no tolerance — a new primitive must preserve that or
> the conformance harness loses its exactness. And changing the geometry primitive
> **changes every corpus**, so every recorded byte count, every ceiling and every
> `bits/drawing` in this project belongs to the old one. That is a fingerprint
> change, not a refinement.

---

## 6. The selection criterion, stated so it can be applied to the next idea too

Every proposal above was ranked by one test, and it is worth naming because it
generalises past this list:

> **Does it create a new kind of redundancy with a ceiling known by
> construction?**

That is what this instrument can measure. Claim 2 worked because the generator
chose the repeat count, so the ceiling was arithmetic. The Tabler oracle worked
because exact translational repeats are countable. `XFORM` qualifies — the
compression available from a known transform group applied to known programs is
exactly computable, so `recovery` has a denominator on day one.

A better noise schedule does not qualify. Neither does a larger corpus of harder
drawings, which is why Tier C and Tier D both stalled. **The constraint has never
been the model's capacity or the data's difficulty — it has been whether the
project could compute what the right answer was.**
