# Class conditioning — the plan, and what it provably cannot settle

**Written 2026-08-12, before the run; §§1–4 are left exactly as written. The run
landed the same day and §5 is the reading** — including the branch that fired
with a conclusion that does not follow, which is why the before-the-run text
stays untouched. **§6 is the generation half, measured the same day**: the
label controls geometry at the corpus's own class separation or better, and the
two columns that fall short do so in the direction that says why.
`docs/direction.md` §7.0 names
this as the first step toward either demo the project might want, and the point of
writing it down first is the same as everywhere else here: so it can be wrong in
public.

QuickDraw ships category labels and this project discarded them for five months.
They are free, they are the only supervision available, and they are the
prerequisite for "draw me a cat" being a sentence the system can be asked.

---

## 1. The result that exists before any run: the ceiling is under the floor

Conditioning can improve `bits/drawing` by at most the mutual information between
drawing and class:

```
I(X; C) = H(C) − H(C | X) ≤ H(C) = log2(5) = 2.3219 bits
```

Tier B multi's run-to-run resolution floor is **~2.5 bits**. So:

> **The likelihood effect of class conditioning is unresolvable at k = 2 on this
> corpus, by construction, and no number of seeds at this budget changes it.**

That is the same move the constructed corpora are built on — know the ceiling
before you measure — applied to an *axis* rather than to a dataset. It is why this
axis gets its own instrument (`dm/eval/conditioning.py`) instead of a row in the
sweep table, and it is worth stating as a finding rather than a caveat: **the
project's headline metric cannot see its most obvious remaining lever.**

### 1.1 And the sharper version, which the smoke run made concrete

The bound is an *identity*, so the same arithmetic says what to measure instead.

- **The paired gap is a difference between two models.** It carries `I(X; C)`
  **plus** whatever the two runs differ by on their own — and that second term *is*
  the resolution floor. On two 30-step smoke arms where the true conditioning
  effect was ~0.01 bits, the paired gap read **+0.34**: 37× the effect, all of it
  the arms differing from each other.
- **`H(C | X)` is a single-model quantity.** Score the *conditional* arm's val
  split under all five classes and Bayes gives `p(c | x) ∝ p(x | c)·p(c)` — five
  forward passes of one checkpoint, no second model, no cross-model term, **no
  floor**.

> **So: the label's value cannot be measured by differencing two runs, and can be
> measured inside one.** Both are reported; the single-model one is the reading.

Two consequences that make this checkable rather than merely stated:

1. **A generative model is a classifier for free.** `p(c | x)` needs no
   discriminative model, so no parameters are spent and the result is about the
   conditioning rather than about a second model — which matters, because this
   project counts parameters and has measured that a 5.8–6.1× larger model was
   better exactly once in nine attempts.
2. **Fano's inequality turns `H(C | X)` into a prediction.**
   `H(C | X) ≤ H(Pe) + Pe·log2(k − 1)` bounds the error rate of *any* reader of
   these drawings, so the measured class-uncertainty predicts an accuracy ceiling
   before the classifier is run. Two instruments, one identity — the arrangement
   that caught three faults elsewhere here.
   *A finite-sample accuracy can sit marginally above the bound; the smoke reading
   was 0.3889 against 0.3870 at n = 90, where the accuracy's own standard error is
   ~0.05.*

---

## 2. What is built

| piece | what it does |
|---|---|
| `quickdraw.load_labelled` | programs **and** their category index, interleaved in one traversal so a label cannot land on the wrong drawing. Backfills labels into an existing cache **only after checking the rebuild is byte-identical** — the cache is the corpus every Tier B number was measured on |
| `Config.n_classes`, `DrawingLM.classes` | an **additive** class embedding, zero-initialised |
| `ProgramDataset(labels=…)` | the conditioning convention on the scoring side |
| `micro_batches(…, classes)` | slices the class tensor with the rows it belongs to |
| `dm/eval/conditioning.py` | the ceiling, the free classifier, the Fano bound, controllability |
| `scripts/conditioning.py` | all four, against an optional matched control |

**Two design decisions carry the rest, and both are about not breaking what
exists.**

**The signal is an additive embedding, not a class token.** Embeddings here are
tied to the output head, so a token in the vocabulary would widen the logits past
`codec.vocab_size` — and `dm/eval/recovery.py`, `dm/eval/spelling.py` and
`dm/eval/redundancy.py` all reshape logits against the codec's own width. That
reshape **does not raise** when the width is wrong; it reinterprets the tensor. The
additive form changes no width, adds no scored position, and costs
`n_classes × d_model` = **640 parameters, +0.08%**. Zero-initialised, so a fresh
conditional model is bit-identically its unconditional control at the same seed:
the arms differ by an addition, not by a second draw — which matters because a
different draw is worth ~2 bits on this project's own measurement, i.e. *the whole
label ceiling*.

**Labels ride on the dataset, not through `collate`.** Every batch already carries
its `index`, because bucketing reorders programs and a paired comparison needs the
score attributed back. So a scorer looks up `dataset.labels[index]` and
`per_program_bits` — which every likelihood number in the project goes through —
gains conditioning with no signature change and no risk to any existing caller.
Sampling has no dataset and takes its classes explicitly.

> **The interaction that would have quietly ruined the experiment.** The control
> arm on record uses `--share-init`, and `share_non_embedding_init` redraws every
> non-embedding *matrix* from one generator — and the class table has the same rank
> as a weight matrix. Filled from that generator it would both destroy the
> matched start **and consume draws the control never consumed**, so the two arms'
> layer weights would have diverged as well. It is now skipped by name, and a test
> asserts every other parameter is identical across the two arms.

---

## 3. The run

One run, because **the control already exists**:
`quickdraw_m5si24000eps4_byte_square_s{0,1}` — Tier B multi, five categories
(`cat bus flower sailboat bicycle`), byte, square, 24,000 steps, `rdp_eps` 4.0,
`max_len` 3072, 350,000 train programs, `--share-init`, two seeds, schema 4,
824,704 parameters. `runs/p7_conditioning.sh` copies every flag from its config, so
the conditional arm lands on the identical val split; `scripts/conditioning.py
--against` verifies that through the corpus digest rather than trusting the flags,
and refuses a control that is itself conditional.

The categories are the control's, and they are a better set for this than the
composed corpus's: cat against **dog** would confound "the label is uninformative"
with "these two classes look alike".

~2 h for both seeds.

---

## 4. What each outcome settles, written before the run

- **`H(C | X)` near `H(C)`, accuracy near chance** → the model learnt nothing
  class-distinctive, conditioning is a measured null, and the demo path needs
  something other than labels — most likely more capacity per class or a corpus
  with less within-class variation. This is the branch that would kill the item.
- **`H(C | X)` well below `H(C)`, but sample class-accuracy at chance** → the model
  can *discriminate* categories and cannot *generate* to one. That is the exact
  shape of the `XFORM` finding — the copy relation was available for prediction at
  3% of full price and for generation in 13% of samples — and finding it twice, on
  two unrelated structures, would make it a property of this training objective
  rather than of either structure. **This is the outcome I expect.**
- **Both good** → conditioning works, "draw me a cat" is a sentence the system can
  be asked, and the next thing to build is the class-conditional sample-quality
  half: `coverage`/`mmd`/`nna` of cat samples against real cats, which is the first
  measurement in this project that could be called a generation result.
- **The paired gap larger than 2.32 bits** → an instrument fault, not a finding.
  Nothing about a class label can buy more than the label's own information, so a
  gap above the ceiling means the arms differ in something else. The most likely
  something is that they are not on the same val split, which is why the digest is
  checked; the next most likely is that the class table absorbed a shared-init
  draw, which is why that is tested.
- **Controllability accuracy above the Fano bound by more than sampling noise** →
  the two halves of the instrument disagree, and since both come from one array
  that cannot happen — so the array is not what it claims and the report is wrong
  before any of it is a result.

**Not owed:** a discriminative classifier for comparison (it would spend
parameters and answer a question about itself), more than two seeds on the paired
gap (the ceiling is under the floor, so more seeds cannot rescue it), and any
prompt-conditioning work until the class half reads.

---

## 5. Measured, 2026-08-12 — the reading

Two seeds, `runs/p7_conditioning.sh` exactly as written above:
`quickdraw_cond24000eps4_byte_square_s{0,1}` at 825,344 parameters (+640, the
class table) against the existing `m5si` controls, the identical val split
verified through the corpus digest rather than the flags. Reports:
`runs/conditioning_quickdraw_cond24000eps4_byte_square_s{0,1}.json` ·
[Figure 13](figs/fig13_conditioning.svg).

| quantity | s0 | s1 | read against |
|---|---|---|---|
| bits/drawing, uncond → cond | 422.73 → 416.34 | 420.24 → 417.99 | — |
| `H(C \| X)` | 0.0708 ±0.0302 | 0.0420 ±0.0180 | `H(C)` = 2.3219 |
| implied `I(X; C)` | **+2.2511** | **+2.2799** | ceiling 2.32; **seeds agree to 0.029** |
| the paired gap | +6.3869 ±0.4098 | +2.2444 ±0.3982 | same identity; **seeds differ by 4.14** |
| classifier accuracy | 0.9910 | 0.9930 | chance 0.200; Fano bound 0.9934 / 0.9964 |
| controllability | **1.0000**, 256/256 non-empty | **1.0000**, 256/256 | chance 0.200 |

**1. The "both good" branch fired — and the branch I expected did not.** §4's
expected outcome was the `XFORM` shape, discriminate-but-not-generate. It is
refuted: asked for each class 512 times across the two seeds, the model reads
back its own class 512 times, with every sample decoding non-empty. Conditioning
penetrates generation completely. Why the two structures part ways is an argument
rather than a measurement, but it is specific: the additive class embedding is
present at *every position of every sample*, imposed from outside the sequence,
while the copy relation is a property of the sampled prefix that generation must
re-elect at each step. Exposure bias can erode the second and has no purchase on
the first.

**2. The single-model instrument replicates and the paired one does not — §1.1
at full scale.** `I(X; C)` = +2.2511 / +2.2799: 0.029 apart across seeds, 97–98%
of the 2.32-bit ceiling. **A drawing carries nearly all of its own class**, and
`H(C | X)` ≈ 0.04–0.07 bits is what a perfect reader would have left to ask. The
paired gap reads +6.39 / +2.24 — 4.14 apart, the smoke run's 37× excess grown to
scale. The s1 pair happened to land 0.035 from its own identity; the s0 pair
drew a +4.14-bit cross-model term. One instrument measures the label; the other
mostly measures which seed you drew.

**3. A pre-registered branch fired and its conclusion does not follow — the
second fault of that kind on the ledger.** §4's fourth branch: *"the paired gap
larger than 2.32 bits → an instrument fault, not a finding."* s0's +6.39 is ten
CI-widths past the ceiling, and **both named faults are excluded** — the digest
check ran (the script refuses a mismatched split) and the shared-init test
passes. What remains is the term §1.1 had already measured on the smoke arms:
the cross-model floor. The branch treated "the arms differ in something else" as
synonymous with "the instrument is broken", when its own document had measured
the something else two sections earlier and given it a name. Same lesson as
[`xform.md`](xform.md) §6, second instance: before writing a branch, name the
sentence of the claim that becomes false when it fires. No sentence does here —
a gap above the ceiling is §1.1 *working*.

**4. Controllability at 1.0000 sits above the val-split Fano bound, and that is
not the fifth branch firing.** The bound (0.9934 / 0.9964) is a property of the
*val distribution*; controllability classifies the model's own top-k samples, a
different `X`. The two halves that must agree — the classifier's accuracy and
the bound derived from the same posterior array — do agree, at both seeds. That
the samples are *more* class-separable than real drawings is itself informative:
the sampler stays inside class-typical modes.

**What follows** is the third branch's consequence as pre-registered: the next
build is the class-conditional sample-quality half — `coverage`/`mmd`/`nna` of
each class's samples against real drawings of that class, `dm/eval/quality.py`
run per class. That would be the first measurement in this project that could be
called a generation result, and the demo path (`docs/direction.md` §7.0 (b)) has
its first prerequisite done.

---

## 6. The generation half, measured 2026-08-12 — geometry, not read-back

The run `PLAN.md` §5 pre-registered, at the pre-registered scale: two seeds,
5 draws × 100 samples per class, every set scored against a fixed 100-drawing
reference of **real val drawings** of that class, against the matched
unconditional `m5si` arms on the identical references. Reports:
`runs/class_quality_quickdraw_cond24000eps4_byte_square_s{0,1}_n100x5.json` ·
[Figure 14](figs/fig14_class_geometry.svg).

**The headline: asking for a class produces that class's geometry — and the
margin it wins by is larger than real drawings of the class manage.** That
second clause is both the result and the warning, and §6.2 is why.

### 6.1 The correction the design needed, found while reading it

The plan's reading was "the diagonal is the row minimum, and the off-diagonal is
the scale that makes it readable". **The off-diagonal is the right idea and the
wrong scale.** How far a class's samples *should* beat the next-nearest class is
a property of the classes' shapes under Chamfer, not of the model: a bus and a
flower separate trivially, a cat and a bus barely separate at all. So a margin
read against zero says only that the model won, never whether it won by enough
— the exact error of reading `recovery` bare instead of against its constructed
ceiling, which this project has already made once.

**So the floor is now a matrix, not a column.** The same `class_quality` call
runs with real **train-split** drawings in the samples' place, giving every cell
the value a perfect class-conditional generator achieves. It costs ~40 s beside
~30 min of sampling, so it is computed unconditionally; the two landed reports
were backfilled with `--floor-only`, which touches no sampled number. That the
two files' floor matrices came out byte-identical is a free check that the floor
is a corpus quantity and cannot depend on the checkpoint.

| class | margin, s0 | margin, s1 | a perfect generator's | ratio |
|---|---|---|---|---|
| cat | +0.47 ±0.57 | +0.75 ±0.14 | +0.43 ±0.14 | 1.08× / 1.73× |
| bus | +2.77 ±0.15 | +2.79 ±0.14 | +2.50 ±0.06 | 1.11× / 1.11× |
| flower | +5.17 ±0.26 | +5.13 ±0.32 | +3.97 ±0.29 | 1.30× / 1.29× |
| sailboat | +1.28 ±0.36 | +1.28 ±0.29 | +0.39 ±0.30 | 3.27× / 3.25× |
| bicycle | +0.83 ±0.24 | +0.89 ±0.15 | +0.76 ±0.12 | 1.10× / 1.17× |

*margin = (nearest other class's `mmd`) − (own class's `mmd`), px, per draw.*

**The diagonal is the row minimum in 48 of 50 class-draws** — 23/25 at s0,
25/25 at s1, the two misses both on `cat`, whose margin the corpus itself puts
at 0.43 px. **And the model confuses exactly the classes the corpus confuses:
the nearest competing class agrees with the floor matrix's at all five classes
and both seeds** (cat→bus, bus→bicycle, flower→cat, sailboat→bicycle,
bicycle→bus). The conditioning is not merely separating classes; it is
reproducing the corpus's own geometric confusion structure.

### 6.2 The price, and why it is one finding rather than three

Averaged over the five classes, against the unconditional control on the
identical references:

| | conditional | no label | floor | of the control's gap, closed |
|---|---|---|---|---|
| `mmd` ↓ | **13.76** | 16.28 | 13.79 | **101%** |
| `coverage` ↑ | 0.422 | 0.353 | 0.512 | 43% |
| `nna` → floor | 0.628 | 0.817 | 0.510 | 61% |

Both seeds separately: `mmd` 101.8% / 100.1%, `coverage` 41.8% / 45.2%, `nna`
63.7% / 59.0% — the ordering and the sizes replicate.

**`mmd` is a null in the good direction: the conditional samples blanket each
class's real drawings as well as a held-out set of real drawings does**, 13.76
against a 13.79 floor, and within 0.22 px at every individual class. On the one
metric that asks "is every real drawing close to something you drew", asking for
a class is as good as having the class.

**`coverage` and `nna` do not get there, and the direction of every miss is the
same as the margin's overshoot.** Samples reach fewer *distinct* real drawings
than real drawings do (0.422 against 0.512), stay more distinguishable from them
(0.628 against 0.510), and separate the classes *better* than real drawings do
(§6.1's ratios, all ≥ 1.08). Those three are not three results. They are one:

> **The sampler sits in each class's typical middle.** A set concentrated on
> class-typical geometry is close to everything in its own class (`mmd` at
> floor), far from the other classes (margin above the corpus's), reaches few
> distinct neighbours (`coverage` under floor) and remains identifiable as a
> tighter-than-real cluster (`nna` over floor). Every column agrees, including
> the one that looks like a win.

This is the same observation §5.4 made from the other side — "the samples are
*more* class-separable than real drawings; the sampler stays inside
class-typical modes" — now measured against real geometry instead of the
model's own read-back, and given a cost.

**And the cause is not yet attributed, which is the honest part.** All of these
samples were drawn at `top_k = 40`, temperature 1.0 — **the identical sampler
`scripts/resample.py` used for figure 8's three arms**. Mode-seeking is exactly
what a truncated sampler does, so "the model's conditional distribution is
over-concentrated" and "the sampler we always use truncates it" are both
consistent with every number above, and nothing here separates them. That is a
confound this axis shares with every sample-quality number on record, and
`PLAN.md` §6 now carries it as the next thing to measure.

### 6.3 The branches split by metric, which none of them allowed for

§5 of `PLAN.md` wrote four outcomes. What fired:

- **"Diagonal beats off-diagonal at every class and sits near its own floor"** —
  fired, on `mmd`, and more strongly than written: the margin beats what a
  perfect generator gets.
- **"Diagonal beats off-diagonal but sits far above floor → conditioning steers
  and sample quality is the binding problem"** — fired, on `coverage` and `nna`,
  though "far above" overstates a 43–61% closure.
- **"Diagonal ties the off-diagonal"** — refuted at every class and both seeds.
- **"Exclusions at scale"** — did not fire. **0 empty decodes in 5,000 samples**,
  validity 0.930–1.000, so the sampler and `controllability` agree about
  non-emptiness and no row above is blocked.

**Two branches fired on different columns of one table.** The set was written as
if the three metrics would answer together, and they answered differently and
consistently — which is the *second* time a pre-registration here has assumed
one number and got a split: direction item 1 split by **position class**
(≈0 bits where the summary determines a byte, 8.6–9.9 where it only constrains)
and this splits by **metric**. The lesson is now general enough to write as a
rule: *when an instrument reports several numbers that fail in different
directions, pre-register a branch per number, or the reading will have to be
invented after the fact.* `dm/eval/quality.py` says in its own docstring that
its three metrics fail in different directions; the branches did not use it.

### 6.4 What this licenses, and what it does not

**Licensed:** "on five geometrically well-separated categories, class
conditioning controls sample geometry, at the corpus's own class separation or
better, with per-class fidelity at the floor." That is the first generation
result on record here, and `docs/direction.md` §7.0 (b)'s first prerequisite is
now external rather than self-reported.

**Not licensed**, each for a named reason:

- **Not "conditional samples are as good as real class sets."** Two of three
  metrics say they are not, at both seeds.
- **Not a statement about hard classes.** `cat bus flower sailboat bicycle` was
  chosen (§3) so that "the label is uninformative" could not be confused with
  "these two classes look alike" — cat/**dog** would test the opposite thing.
  The corpus's own cat margin of 0.43 px is a preview of what that would cost.
- **Not a per-class level.** The reference is one fixed 100-drawing draw, so
  every level carries a reference-draw term nothing here measures. The
  *ordering* across the two seeds is what replicates, and it does.
- **Not a resolution floor.** As everywhere on this axis, the floor is what a
  perfect generator scores, not what a re-run of this config scores. Each cell
  is one checkpoint against fixed reals.
- **Not a semantic claim.** Chamfer sees shape. "Cat-shaped geometry" is what
  was measured; "a human would call it a cat" was not, and is not owed by
  anything in this project's budget.

---

## 7. Sampler attribution — guarded result 2026-08-13

§6 measured one coherent mode-seeking signature and did **not** attribute its
cause. This no-training separating experiment asks whether published quality
readings describe checkpoints or the `top_k = 40`, temperature-1 sampler used to
read them. All 12 expected identities now exist: nine complete reports and
three durable failures. `all,T=1` class quality, `k40,T=1.2`, and flat-AR `all`
are structurally blocked by one empty decode each; planner-AR and planner-
diffusion `all` endpoints complete. Thus `k20→40→80` settles concentration,
full-support partial evidence identifies an overshoot/fidelity trade, and
Figure-8 transfer is arm-dependent rather than universal.

### 7.1 Audit of §6 before moving on

**The result stands.** Both reports are 5 draws × 100 per class, every generated
and reference set is 100 before empty exclusion, both floor matrices are
byte-identical, every control draw retained all 100 geometries, and conditional
sampling had 0 empty decodes in 5,000 outputs. The two checkpoint seeds preserve
the metric ordering. Nothing below retracts §6.

Five instrument debts did matter before a sweep:

1. `scripts/class_quality.py` hardcoded `top_k = 40` and temperature 1.0, so its
   filename could not name either. It also omitted draw count, cloud resolution,
   control and device. A neighbouring setting could overwrite the result it was
   meant to compare. Both report drivers now key every one of those fields and
   refuse replacement without `--overwrite`.
2. `class_quality` refused unequal sizes *within* each side but did not refuse
   100 samples against, say, 80 references. Landed reports are 100 against 100,
   so no number moved; the cross-side refusal and regression test now exist.
3. A `--control` needed only share the val digest. A different codec, budget,
   architecture or train split could pass. Validation now compares both corpus
   digests, schema, reached steps, normalised training config and model config
   apart from the class table — before sampling.
4. Relaxing a sampler can improve geometry by emitting blanks, faults or capped
   prefixes that the geometry metric then excludes or executes partially.
   Conditional and control reports now carry `validity`, `empty`, `truncated`
   and fault counts beside every quality reading. Both report drivers make any
   empty decode fail before publication: dropping it would change 100 v 100 into an
   unbalanced estimator, and replacing it would sample a different setting.
5. PAD and BOS occupy model logits but are not bytecode targets. Full-support
   sampling previously could spend a decode step on a symbol `decode` drops,
   making both the stream and cap guard sampler-artifacts. AR sampling and the
   planner's stroke decoder now mask PAD/BOS before top-k **in schema-2 side
   reports only**; training-time eval keeps raw support so evaluation cannot move
   later training RNG streams. Auditing that guard
   found another live defect: `cap_hit` compared decoded length with the tensor's
   **returned** width, but generation returns when the last row halts, so that
   last valid row was labelled truncated — exact 1/n values recur in old flat-AR
   reports. Cap status now comes from halt-monitor state. Planner cap accounting
   is separate and unaffected. This changes no §6 geometry number; schema-2
   reports carry corrected guards. Old readings remain
   provenance-correct for their old hardcoded sampler and are not rewritten.

### 7.2 Exact sweep and landed subset

One training seed, **s0 by convention rather than by its result**, with the
landed scale unchanged: 5 draw seeds × 100 samples × 5 classes, matched s0
unconditional control, fixed 100-drawing references and the same floor matrix.
Six settings, one factor at a time around the published point:

| axis | settings | purpose |
|---|---|---|
| top-k, temperature 1.0 | **20, 40, 80, all** | concentration dose-response; `all` is the primary endpoint against 40 |
| temperature, top-k 40 | **0.8, 1.0, 1.2** | sharpening/flattening control; 1.0 is shared with the row above |

Every new setting uses **legal output support** (PAD/BOS masked). The landed
schema-1-style reports are the raw-support `k=40, T=1` reading. New legal `k=40`
versus that landed file is therefore a separate **support-correction contrast**,
not a top-k effect; no extra invocation is needed. If it moves, published figures
were partly reading impossible output symbols. Top-k attribution remains legal
`all` versus legal `40`, so two sampler changes are never folded into one number.

No `all × 1.2` interaction cell yet. It cannot distinguish either main effect
and costs another full invocation; add it only if the one-factor curves disagree
in a way that names the interaction as the remaining question.

The earlier plan proposed the same six-setting sweep on one Figure-8 arm and
then spoke about all three. **That transfer does not follow.** The flat arm uses
this AR sampler directly; each planner also samples a composition grid before
its AR stroke decoder, so one arm cannot price the other two. At the same six
invocations, the stronger transfer check is the primary endpoint (`k=40` versus
`all`, temperature 1.0) on **each of Figure 8's three arms**. That is now the
plan. Temperature remains class-quality-only unless it is the axis that moves.

### 7.3 Reading rules, fixed before data

Read each quantity separately; §6.3 already showed why one branch for three
metrics is invalid. First read corrected legal `k=40` against landed raw-support
`k=40`; report any support correction on its own. Primary top-k contrast was preregistered as
legal `all, T=1` minus legal `k=40, T=1`, paired by draw seed and averaged over
classes. Its structural guard fired, so it remains a failed endpoint rather than
being silently replaced by `k80`.

- **Sampler-sensitive for this checkpoint:** endpoint moves toward its floor in
  at least 4 of 5 paired draws and its mean movement exceeds that metric's
  legal-k40 baseline draw-to-draw SD. Top-k means ordered 20 → 40 → 80 → all strengthen
  attribution but are not required to survive one noisy adjacent cell.
- **Sampler-insensitive:** endpoint movement is no larger than legal-k40 baseline SD,
  paired signs split, and no ordered top-k trend exists. Then §6's concentration
  remains a checkpoint property at this resolution.
- **Mixed/non-monotone:** anything else. Report the curve; do not force it into
  “model” or “sampler”. Temperature moving when top-k does not means sampling
  sharpness matters but truncation is exonerated specifically.

Directions are fixed: `coverage` rises toward its floor; `nna` falls toward its
floor while it remains above it; class margin falls toward the corpus margin.
`mmd` should remain in its floor band. If coverage/nna improve and `mmd` moves
upward by the same sensitivity rule, diversity–fidelity is a sampler curve, not
a free correction. If `mmd` moves below floor, it is not called “better” without
coverage and `nna`: a typical-centre or memorising set can do that.

**Failure guard:** any empty decode blocks that setting's geometry reading,
because exclusion makes set sizes unequal. Truncation above 1%, or a validity
drop greater than 2 percentage points from its legal `k=40, T=1` baseline, makes it a
structural-quality trade: geometry remains reportable at matched sizes but not
as diversity gained for free. Loss of the own-class diagonal is a conditioning
failure regardless of aggregate quality.

One checkpoint measures one checkpoint. Figure-8 endpoint agreement across its
three checkpoints can show a shared sampler effect; it still does not become a
run-to-run resolution floor.

### 7.4 Reading — top-k concentration, then full-support boundary

Landed schema-2 reports are complete: five draws each, 100×100 matched class
sets, zero empty/capped outputs, and byte-identical floor matrices. Values below
average classes; `±` is draw-to-draw SD, not checkpoint replication:

| sampler | coverage ↑ | mmd (px) | nna → floor | class margin (px) | validity |
|---|---:|---:|---:|---:|---:|
| `k=20, T=1` | 0.339±0.016 | 13.974±0.077 | 0.709±0.010 | 2.696±0.119 | 0.975 |
| `k=40, T=1` | 0.406±0.023 | 13.805±0.037 | 0.639±0.018 | 2.181±0.085 | 0.993 |
| `k=80, T=1` | **0.487±0.011** | **13.764±0.023** | **0.549±0.011** | **1.840±0.188** | 0.996 |
| real-drawing floor | 0.512±0.010 | 13.788±0.052 | 0.510±0.012 | 1.611±0.072 | — |

`k=20` moves coverage and `nna` away from floor in **5/5 paired draws**;
`k=80` moves both toward floor in **5/5**, by +0.080 and −0.090. Matched
unconditional control independently follows same top-k direction in 5/5 draws
(`k80−k40`: coverage +0.044, `mmd` −1.093, `nna` −0.061), so response is not
created by class aggregation. Those endpoint
movements are 3.5× and 4.9× legal-`k40` draw SD. Every class agrees: coverage
rises +0.052 to +0.100 and `nna` falls −0.070 to −0.123. Class margin also moves
toward corpus margin in 5/5 draws (−0.340, 4.0× its baseline SD). Thus §6's
concentration is **strongly top-k-sensitive** at this checkpoint; wording that
made it a checkpoint property is retracted.

This is not a free monotone win in every column. `mmd` remains in floor band and
moves only −0.041 at `k=80`, 3/5 paired draws toward floor — mixed by the
pre-registered rule. Conditioning remains strong, but cat loses its diagonal in
2/5 `k=80` draws; total own-class wins move 25/25 (`k20`) → 24/25 (`k40`) →
23/25 (`k80`). Relaxation trades some separation margin for diversity, as it
should, and cat is first to expose the trade because corpus cat/bus margin is
smallest.

Temperature sharpening does not explain the result: `k40,T=0.8` against
`k40,T=1` moves coverage −0.004 and `nna` −0.0002 with split signs and changes
smaller than baseline SD. Margin rises +0.250 in 5/5 draws, so temperature affects
class separation, but not the two diversity metrics here.

Guards pass for `k40` and `k80`: no empties or truncation; validity improves
0.993→0.996. `k20` validity falls to 0.975, a 1.84-point drop — just inside the
pre-registered 2-point trade threshold — almost entirely cat (0.886). No quality
claim should hide that boundary.

The legal-support correction alone is not beneficial on conditional geometry:
relative to landed raw-support `k40`, legal `k40` changes coverage −0.011, `mmd`
+0.063 and `nna` +0.019 with split draw signs. Both class reports used MPS, so
these are support-only differences, not top-k effects.

Full support reaches boundary and then crosses it. In first four complete paired
draws, `all` moves coverage **0.402→0.523** and NNA **0.639→0.541**, toward
0.515/0.511 floors in 4/4; every class agrees. But MMD worsens
**13.799→14.036** in 4/4, away from 13.806 floor, and class margin drops
**2.168→1.359** against 1.583 floor. Own-class wins fall to 17/20: sailboat
loses twice and cat once. Fifth draw emits one empty bus geometry among 100, so
report is `status="failed"`; no five-draw aggregate or control geometry is
published. This is not primary-endpoint success. It is strong partial evidence
that `k=80` is near useful elbow: `all` buys little NNA beyond `k80`, overshoots
coverage, loses fidelity/separation, and introduces structural failure.

Temperature flattening reaches same boundary immediately: `k40,T=1.2` emits
one empty sailboat geometry in first draw, so no temperature endpoint exists.
`T=0.8` remains null for coverage/NNA; no interpolation or survivor score is
licensed.

Figure-8 legal `k40` files are valid MPS baselines, but old/new deltas are
**not** support contrasts: published Figure 8 used script-default CPU. Never
difference reports across device.

Within MPS, planner full-support endpoints complete and upstream composition
outputs are unchanged as evidenced by identical per-draw planned/stroke counts;
only stroke-decoder support changes. Planner-AR `all−k40` is coverage **−0.013±0.023**, MMD
**+0.101±0.086**, NNA **+0.000±0.022**. Planner-diffusion is **+0.000±0.011**,
**+0.138±0.093**, **+0.013±0.038**. Coverage/NNA are sampler-insensitive by
pre-registered floor-movement rule; MMD worsens in 4/5 and 5/5 draws respectively, beyond each baseline draw
SD. Length EMD, planned counts, truncation, and validity remain effectively
fixed. Flat AR moves strongly on first four draws (coverage +0.088, NNA −0.068,
MMD +0.260, length EMD −6.69) but emits one empty of 256 on draw 5, blocking its
endpoint.

**Diagnosis:** top-k truncation drives flat/conditional AR concentration, not a
universal Figure-8 effect. Planner composition already supplies diversity, so
relaxing only stroke support cannot improve coverage and instead degrades local
fidelity. Keep `k=40` as planner operating point; use `k=80,T=1` as best tested
conditional operating point. Do not promote `all` or `T=1.2`.

### 7.5 Failure evidence and publication contract

All 12 identities exist. Failures are exact, not inferred:

- class `all,T=1`: one empty bus geometry at draw 4, **1/2,500** conditional
  samples; validity 0.9968, zero caps, eight `unknown_opcode` faults;
- class `k40,T=1.2`: one empty sailboat geometry in first 500 samples; that
  class guard has validity 1.0 and no fault, proving blank output is not merely
  `unknown_opcode` under another name;
- flat AR `all,T=1`: one empty geometry at draw 4, **1/1,280** samples;
  validity 0.9945, zero caps, seven `unknown_opcode` faults.

Tiny rates do not waive guard: one exclusion changes estimator from 100×100 or
256×256. Failure reports preserve guards and partial draws in `draws`; unavailable cells
use JSON `null`, never non-standard `NaN`. Consumers must require
`status="complete"` before aggregating; `scripts/sampler_attribution.py`
enforces that and prints failures without aggregating survivor geometry. No
rerun is owed. More samples under same failed setting would answer frequency,
not recover preregistered endpoint.
