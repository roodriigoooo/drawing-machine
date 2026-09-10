# Bézier control points — can *scale* join the transform group?

> **Status: §§1–4 written 2026-08-13 before the decisive number existed, and left
> exactly as written. §5 is the reading. §6 is the canonicalisation half, whose
> headline result is a theorem rather than a measurement and which is here
> because it was built to serve the same experiment.**
>
> `docs/direction.md` §7.3 put a prediction on record before any code: **a
> control-point representation should retain ~86% of a corpus's repeat ceiling
> under a ×1.07 scale where a polyline retains 23%.** This file is what turns
> that into something that can be wrong in public.

---

## 1. Why this is the largest thing on the queue, and why it is also an afternoon

The transform group in the ISA is **D4 ⋉ integer translation** — eight
symmetries and a shift, nothing continuous. That is not taste. It is a
measurement: against Tabler's repeat ceiling, integer translate, mirror and 90°
rotate preserve it *exactly*, while a ×1.07 scale destroys **77%** of it and a
±1 jitter **65%**. A repeat is lossless or it is nothing, and a repeat needs
*every* coordinate of its body to round the same way.

So the exclusion of scale is not about scale. It is about **body size**. Model
the damage as an independent per-operand divergence probability `p` over an
`n`-operand body:

```
(1 - p)^n = 0.23   at n ≈ 40   ->   p ≈ 0.036
(1 - p)^4  ≈ 0.86  at the same p
```

A polyline body is dozens of numbers. A cubic segment is six. **If the model is
right, the same scale that destroys a polyline corpus leaves a control-point
corpus nearly intact, and "a cat twice the size" becomes expressible in the
ISA** — materially larger than anything `XFORM` established, because `XFORM`
added elements to a finite group and this would add a one-parameter family.

If the model is wrong, the Bézier idea reduces to a compression win: claim 1
rather than claim 2, still worth having and a much smaller prize.

Either way it costs an afternoon and no training, which is the criterion
`docs/direction.md` §6 ranks by: **a new kind of redundancy with a ceiling known
by construction.**

---

## 2. What is built, and the two properties that are load-bearing

| piece | what it does |
|---|---|
| `dm/isa/bezier.py` | fits cubics to a polyline; **integer control points, clamped to the canvas, scored through the VM's own flattener** |
| `dm/data/refit.py` | re-spells a corpus under one primitive; `scale_program`, `jitter_program` |
| `dm/eval/repeats.py:body_numbers` | the operand count of the bodies each oracle actually folds — the model's exponent, measured |
| `dm/eval/library.py` | the between-drawing ceiling (§6, and the `Op.CALL` denominator) |
| `dm/data/canonical.py` | frame policies, and the exactness flag that decides how their numbers may be read |
| `scripts/bezier_oracle.py` | the ablation |
| `scripts/canonical.py` | §6's four ceilings per policy |

**Integer control points, or claim 4 loses its exactness.** `port/include/dm_vm.h`
argues that a `CURVE`'s geometry is exactly integral at a scale of
`CURVE_STEPS ** 3`, which is what lets `scripts/conformance.py` compare the C
port with `==` and carry no tolerance anywhere in the file. That argument holds
for integer corner points and for nothing else. A fitter emitting fractional
controls would have cost the project its one exact device claim silently, so the
fit quantises *before* it scores.

**The error is the drawn one.** A `CURVE`'s geometry is the 16-segment polyline
the VM emits, not the ideal cubic, so `fit_polyline` measures every candidate
through `dm.vm.interp.flatten_cubic` — the same function `VM.run` calls. A fit
that passes tolerance passes it on the device rather than in the fitter's
idealisation. It is also symmetric (both-ways Hausdorff): a curve is free to
bulge between two source points, and a one-sided maximum never sees it, so a
lenient metric would buy the bytes the primitive is supposed to.

**And one boolean separates the arms.** `fit_polyline(..., allow_curve=False)`
runs the *identical* recursion with cubics forbidden, which is
Ramer-Douglas-Peucker under the same quantised, both-ways criterion and the same
integer endpoints. So the two spellings of a corpus differ in the primitive and
in nothing else, and they can be compared at **matched measured fidelity**
rather than at matched knob settings — `rdp_eps` and a fitting tolerance are
different units, and a table indexed by knob says nothing.

### 2.1 The three arms, and why the third is calibration rather than a control

Tabler's programs already contain `CURVE`, because its source is cubics. So
"Tabler as authored" is neither arm — it is the mixture, and it is the corpus
the 23% on record was measured on. It is carried as **`native`**, whose job is
to reproduce that 23%. If it does not, nothing else in the table is comparable
with anything already published.

| arm | primitive | role |
|---|---|---|
| `native` | as authored, mixed `LINE`/`CURVE` | calibration against the 23% on record |
| `polyline` | `LINE` only, refitted | the control |
| `bezier` | `CURVE` where it pays, refitted | the arm under test |

Both refitted arms are scored against the geometry the *native* program draws,
so the source is identical and the fitter is the only thing between them.

---

## 3. The measurement

Tabler, because it is the only corpus with a repeat ceiling to retain:
QuickDraw's is **0.00%** at two simplification levels, and a retention is a
ratio of two zeros there. 300 icons, family-disjoint split, tolerances 1, 2 and
4 canvas pixels with **2.0 as the primary**.

For each arm and tolerance: re-spell, then apply `scale_program(×1.07)` and
`jitter_program(±1)`, then run both within-program oracles over the scaled and
unscaled corpora.

**Pairing is on the source drawing.** The two spellings refuse different
drawings (a scale can push a coordinate off the canvas, and refusing beats
clamping because clamping is a deformation), so every number is computed over
the drawings that survive in **every** arm. Differencing two arms over two
different subsets is the val-set fault this project has already paid for once.

`retention = (bytes the oracle saves after the operation) / (bytes it saves
before)`, per arm, on that paired subset.

---

## 4. What each outcome settles, written before the run

Each quantity gets its own branch. `docs/conditioning.md` §6.3 is the reason:
that instrument reported three numbers, the pre-registration wrote one branch
for all three, and two of them fired in different directions — so the reading
had to be invented afterwards.

**1. `retention_REPEAT`, the primary endpoint.** `bezier − polyline` at the
primary tolerance, paired on the source.

- The sentence under test is: *"a control-point body's operands round
  independently, so shrinking the body restores most of the repeat ceiling under
  a scale."*
- **If `retention_bezier` ≤ `retention_polyline`**, that sentence is false,
  **scale stays out of the transform group permanently**, and the Bézier idea is
  a compression result. This is the falsifier, and it names the sentence it
  kills — `docs/traps.md` carries the rule, having watched two earlier
  falsifiers fire without doing so.

**2. `n̄`, the operand count of the folded bodies.** The model's exponent, and it
is *measured* rather than the "`n ≈ 40`" estimate on record. **If `n̄_bezier` is
not materially below `n̄_polyline`, the mechanism is absent** and branch 1 has no
reason to move; the reading stops there and reports why. This branch can fire on
its own: a fitted cubic covers many vertices, but the oracle folds whole
*instruction blocks*, and nothing guarantees the blocks it finds are shorter.

**3. Model agreement.** Fit `p̂` from the polyline arm at its own measured `n̄`,
then predict the Bézier arm at its `n̄`.

- **within ±0.10 absolute** → the independence model describes the mechanism,
  and it then predicts retention for *any* body size rather than for this one.
- **outside** → retention may still have improved, but the ~86% has no standing
  and no extrapolation to a different primitive is licensed.

**4. `bytes_per_drawing` at matched measured error.** The compression half,
independent of everything above; a win here survives even if 1–3 all fail.

**5. `error_mean` / `error_max`.** `error_max ≤ tol` is an **assertion, not a
result** — the fitter splits until it holds. If it is ever violated the fitter
is broken and no other number in the table is readable.

**6. The unscaled ceiling.** If an arm's unscaled `REPEAT` ceiling falls below
0.5% of bytes, its retention is a ratio of two small numbers and is quoted as a
diagnostic, never as a result.

**7. `rejected` by the operation.** If the paired subset is under 80% of the
corpus, it is not the corpus and the table says so.

**8. `jitter ±1` retention**, the second inexact operation. Same mechanism,
harder test, no exact-in-the-reals story at all. **A gain on the scale but not
on the jitter would say the gain is about ×1.07 being nearly integer at these
magnitudes rather than about body size** — which is the confound the scale alone
cannot exclude.

**Not owed:** a training run of any kind, a new corpus, a second seed (nothing
here is stochastic except which drawings the operation rejects, and that is
deterministic given the corpus), or a `CURVE_STEPS` change.

---

## 5. Measured, 2026-08-13 — the endpoint fires, the model does not, and the ISA gets nothing

Tabler, 300 icons, 24 rounding phases, scale about each drawing's own
bounding-box centre, primary tolerance 2.0 px. Report:
`runs/bezier_oracle_tabler_n300.json`. Nothing was trained.

| arm | B/drawing | measured error | curve fraction | `REPEAT` ceiling | operands/body | retention under ×1.07 |
|---|---:|---:|---:|---:|---:|---:|
| `native` (calibration) | 57.9 | 0.00 | — | 3.99% | 27 | 42.8% ±2.9% |
| `polyline` (control) | 56.7 | 1.08 | 0.00 | 3.15% | 18 | 42.3% ±3.8% |
| `bezier` (under test) | **49.1** | 1.05 | 0.29 | 2.25% | 15 | **61.5% ±9.8%** |

**Primary endpoint: `bezier − polyline` = +19.2% ±10.1%, in 24 of 24 paired
draws.** ± is the phase-to-phase SD, not a replication of anything.

### 5.1 The instrument corrected the number it was calibrated against

The `native` arm is Tabler as authored, under the same operation the 23% on
record was measured with, and it reads **42.8% ±2.9%** — not 23%. The
discrepancy is not a disagreement, it is the **rounding phase**, and finding it
is the most transferable thing here.

A scale is a non-integer map followed by rounding, and *where the rounding
lands* is set by a sub-pixel offset the operation never names. On a corpus
authored on a coarse sublattice — Tabler is a 24 grid at ×10, so most
coordinates are multiples of ten — the phases resonate rather than average. The
identical corpus under the identical ×1.07 retains **18.7%** scaled about the
origin and **34.8%** about each drawing's own centre, which are one operation at
two phases. Marginalising over 24 low-discrepancy phases gives **42.8% ±2.9%**
about the drawing and **42.9% ±4.2%** about the origin
(`runs/bezier_oracle_tabler_n300_origin.json`): **the two conventions agree to a
tenth of a point once the phase is averaged, and disagree by 16 points when it
is not.** The endpoint agrees as well — +19.2% ±10.1% in 24/24 draws about the
drawing, +19.4% ±9.5% in 23/24 about the origin — so nothing below depends on
which convention is called primary.

So **"×1.07 destroys 77% of the ceiling" is one draw of a quantity whose mean
destroys 57% and whose phase spread is ±3 to ±10 points.** The direction of
every conclusion that rested on it survives — scale is still ruinous — but the
number itself was never a number. `docs/traps.md` carries the rule.

### 5.2 Branch by branch, as pre-registered

**1. `retention_REPEAT` — fires.** +19.2% ±10.1%, 24/24 paired draws, and the
sign holds at every factor the ladder can measure (§5.3). The sentence under
test — *a control-point body's operands round independently, so shrinking the
body restores the ceiling* — is not refuted.

**2. `n̄` — fires, and it is the smallest number in the experiment.** The folded
bodies go from **18 operands to 15**, not from 40 to 4. A cubic covers many
vertices, but the oracle folds whole *instruction blocks*, and a block that was
worth folding as six `LINE`s is worth folding as two `CURVE`s: the body shrinks
by the compression ratio, not by the per-instruction ratio. §7.3's `n = 4` was
never reachable by this route, and it was an estimate where the oracle could
have been asked.

**3. Model agreement — fails, in the favourable direction.** `p̂ = 0.0469` fitted
on the control predicts **48.7%** at n = 15; the arm delivers **61.5%**, a miss
of 12.8 points against a pre-registered ±10 band. The independence assumption
*understates* what a control-point body retains, and the reason is visible in
the primitive: a cubic's two control points are computed from one least-squares
solve on one span, so they move together under a scale where two independent
vertices do not. **Operand roundings inside a fitted body are positively
correlated.** That is a better result than agreement would have been — it says
the mechanism is real and the arithmetic on record was the wrong arithmetic —
but it also means **no extrapolation to a smaller `n` is licensed**, including
the 82.5% the fitted `p̂` would give at n = 4.

**4. Bytes at matched error — fires on icons and is a null on QuickDraw.**
Tabler: **56.7 → 49.1 B/drawing, −13.4%**, at mean error 1.08 → 1.05 px, so the
Bézier arm is cheaper *and* very slightly more faithful. QuickDraw, refitted
from the **unsimplified** source so that RDP has not already thrown the geometry
away (`runs/bezier_oracle_quickdraw_raw_n300.json`):

| tol (px) | polyline B | bezier B | polyline err | bezier err |
|---:|---:|---:|---:|---:|
| 0.5 | **210.9** | 215.0 | 0.22 | 0.40 |
| 1.0 | **195.5** | 200.3 | 0.73 | 0.95 |
| 2.0 | **159.5** | 163.4 | 1.83 | 1.94 |
| 4.0 | **120.0** | 120.3 | 3.73 | 3.83 |
| 8.0 | 92.5 | **90.3** | 7.26 | 7.38 |

The Bézier arm is dominated — more bytes *and* more error — at every tolerance
but the coarsest, where it wins 2 bytes while losing 0.12 px. **A cubic buys
nothing on human sketch strokes**, which are not smooth: there is no span long
enough for four control points to beat two vertices. This is the same shape as
the granularity finding — the primitive that buys 13.4% on authored icons buys
nothing on the corpus every trained number in this project lives on — and it is
the reason the compression half is not a general result either.

**5. The error bound — holds, with a correction the test found.** The guarantee
is `error ≤ max(tol, √2/2)`: below half a pixel diagonal no spelling on an
integer grid can do better, whatever primitive it uses. §4 wrote `error_max ≤
tol` as an assertion and that was wrong below 0.71 px. Measured `error_max` is
exactly `tol` at every tolerance used here.

**6. Ceiling readable — yes.** The `bezier` arm's unscaled ceiling is 2.25% of
bytes, above the 0.5% floor. **7. Pairing — 299 of 300** survive ×1.07 in every
arm.

**8. Jitter — the mechanism does not transfer, and this is the sharpest
negative.** Under ±1 jitter the retention is **0.4% / 0.9% / 1.1%** for
native / polyline / bezier: total destruction in every arm, with no advantage to
control points. A ±1 jitter changes ~2/3 of operands, so `(1 − p)^n` is
effectively zero for any body larger than a few numbers, and shrinking 18 to 15
cannot help. **The Bézier advantage is specific to a smooth, near-identity map
where `p` is small — it is not a general robustness to per-coordinate noise.**

### 5.3 The ladder, and the column the pre-registration should have had

Retention of the `REPEAT` ceiling by scale factor, 24 phases, tol 2.0 px:

| factor | native | polyline | bezier | drawings surviving |
|---:|---:|---:|---:|---:|
| 0.75 | 45.5% ±1.6% | 42.9% ±0.5% | **50.8% ±1.5%** | 300/300 |
| 0.93 | 47.5% ±3.9% | 47.9% ±4.5% | **65.3% ±11.7%** | 300/300 |
| 1.07 | 42.8% ±2.9% | 42.3% ±3.8% | **61.5% ±9.8%** | 299/300 |
| 1.25 | 37.4% ±1.7% | 33.1% ±0.3% | **42.0% ±1.5%** | 278/300 |
| 1.50 | 24.5% ±0.0% | 20.3% ±0.0% | **29.5% ±0.0%** | 77/300 |
| 2.00 | — | — | — | 10/300 |

The endpoint's sign holds at every factor with survivors, and the effect is
largest near 1 where `p` is smallest — which is what the divergence model says
should happen even though its magnitude is wrong. The zero SDs at 1.25 and 1.50
are arithmetic rather than luck: `1.5·x` on integers is an integer or a
half-integer, so a phase offset moves every coordinate coherently and the
retention has nowhere to vary. The ×2 row has no retention because only 10
drawings survive it — see below, it is the interesting row.

**And here is the column §4 should have pre-registered and did not.** A
retention is a *ratio*, and the arms do not share a denominator:

| | unscaled foldable | after ×1.07 | of total bytes |
|---|---:|---:|---:|
| polyline | 1.78 B/drawing | **0.75** | 56.7 |
| bezier | 1.10 B/drawing | **0.68** | 49.1 |

**The Bézier arm retains 61.5% of a ceiling that is 38% smaller, and ends with
fewer foldable bytes than the control.** As a fraction of each arm's own bytes
the two land within 0.06 points of each other (1.33% and 1.38%). The cause is
not a fitter fault — `tests/test_bezier.py` carries a planted three-copy repeat
through both arms — it is `docs/traps.md`'s own rule arriving from the other
side: *a compression feature's value is bounded by what is left uncompressed.*
A more compact primitive removes the redundancy `REPEAT` was going to remove, so
the two compressions are substitutes rather than complements.

Missing that column is the third time here that a pre-registration named one
number where the instrument reported several that move differently, after
direction item 1 (split by position class) and conditioning §6.3 (split by
metric). This one splits by **ratio versus level**.

### 5.4 The verdict: scale stays out, and the reason moved

**Scale does not join the transform group.** 61.5% retention is not a small
loss to be tolerated: a repeat is lossless or it is nothing, so a ×1.07 inside
the ISA would silently delete 38% of the structure `REPEAT` and `REPEATX` exist
to fold, and it would delete it from the arm that had less to begin with.

**But the binding constraint is not the one on record, and the ladder shows
it.** An *integer* factor needs no divergence model at all: `round(k·x) = k·x`,
so a repeat at step `t` maps to a repeat at step `k·t` exactly, and
`tests/test_refit.py` pins that a planted repeat survives ×2 byte for byte.
What stops it is the **canvas**: Tabler icons are fitted to a 256 grid with an
8-pixel margin, so ×2 takes 290 of 300 drawings off the canvas and the scale is
refused rather than clamped. So the two ends of the ladder fail for unrelated
reasons — the small factors to rounding, the exact ones to headroom — and
neither is about the geometry primitive.

> **The constructive form of the answer, which §7.3 did not anticipate: the
> only scale that can enter this ISA is an exact integer ratio on a corpus
> stored with room to grow.** That is a corpus-design decision — a wider margin
> in `quickdraw.load` and `tabler.SCALE` — and a control-point representation is
> irrelevant to it. Nothing here licenses a continuous scale under any
> primitive.

**What the Bézier primitive does buy**, and it is claim-1-shaped rather than
claim-2-shaped: **13.4% fewer bytes at matched fidelity on authored icons, and
nothing on QuickDraw.** Sequence length is what killed Tier D and what forces
`rdp_eps`, so 13.4% on the corpus where it applies is worth having. It is not a
reason to change the geometry primitive of a project whose trainable corpus is
the one where it does not apply — and changing it would re-fingerprint every
byte count, ceiling and `bits/drawing` on record.

**Not licensed:** that cubics improve sample quality, likelihood or anything a
model does (nothing was trained); that the −13.4% transfers to another corpus
(it does not transfer to the only other one measured); that `XFORM` or `REPEATX`
become cheaper (their operand counts do not change); or that 86% is reachable at
any body size, since the model that produced it is refuted at the only exponent
where it could be checked.

---

## 6. Canonicalising the frame — a theorem, three exact nulls, and the idea's death

`docs/direction.md` §7.1 proposed aligning every sketch to a standard frame and
proposed one test: canonicalise, then ask whether the corpus's **repeat ceiling**
went up. **That test cannot fire in the direction it hoped, and the arithmetic
says so before the run.**

Both oracles are within-program. A canonicalisation applies one map to a whole
drawing, so for any `g` in D4 ⋉ integer translation a body `B` recurring at
`B + t` becomes `g(B)` recurring at `g(B) + L(t)`, where `L` is a signed axis
permutation — the fold is carried across bijectively and stays inside `i8`.
**The ceiling is invariant, exactly.** The same runs for `REPEATX`, whose step
conjugates to another element of the same group. So the branches are:

- an **exact** policy is a *provable null* on both within-program oracles;
- an **inexact** one is a non-integer map followed by rounding, which is the
  operation measured to destroy the ceiling — it can only lose.

### 6.1 The four ceilings, and where a re-framing could possibly show up

`scripts/canonical.py`, Tabler, 300 icons, library grid 1, frame policies
re-spelled at tol 2.0 px. Report: `runs/canonical_tabler_n300_g1.json`.

| policy | exact | B/drawing | `REPEAT` | `REPEATX` | `CALL`/translation | `CALL`/D4 |
|---|---|---:|---:|---:|---:|---:|
| none | ✓ | 57.9 | 3.97% | 11.74% | 17.92% | 25.21% |
| **d4** | ✓ | 57.9 | **3.97%** | **11.74%** | **19.58%** | **25.21%** |
| none, re-spelled | ✓ | 49.1 | 2.24% | 3.01% | 11.86% | 16.96% |
| rot | ✗ | 53.0 | 0.16% | 0.88% | 5.11% | 8.34% |
| aspect | ✗ | 54.9 | 0.17% | 0.73% | 7.28% | 13.26% |
| rot + aspect | ✗ | 54.7 | 0.12% | 0.82% | 5.26% | 8.40% |

**The three predicted nulls fire exactly** — 3.969852% → 3.969852%,
11.742708% → 11.742708%, and the D4-normalised library saving byte-identical —
on 200 of 300 icons whose bytes the policy actually changed. Not "within noise":
equal. `tests/test_canonical.py` asserts it, so a policy that is not in the
group fails loudly rather than reporting a small movement someone has to
interpret.

**Only one column can move, and it does.** The translation-normalised `CALL`
ceiling goes 17.92% → 19.58%: aligning drawings to a common frame makes them
share more *strokes*. That is the claim §7.1 was actually making —
"two people drawing the same object produce more similar programs" is a
statement about pairs of drawings — and no instrument in this project had ever
compared two drawings before `dm/eval/library.py`.

**Its sign is not stable, which is the honest part.** At 120 icons the same
column *falls*. The representative is chosen from the whole drawing, so two
icons that shared a body can be sent to different frames and stop sharing it: a
canonicalisation aligns drawings that are globally alike and separates ones that
are alike only in a part. The gain is real at 300 and it is not a law.

### 6.2 Every inexact policy loses on every ceiling

Against the same corpus re-spelled with no frame change, so the fitter is not
being charged to the frame:

| policy | Δ `CALL`/t | Δ `REPEAT` | Δ `REPEATX` | mean error |
|---|---:|---:|---:|---:|
| rot | **−6.75%** | −2.08% | −2.13% | 1.29 px |
| aspect | −4.58% | −2.07% | −2.27% | 1.32 px |
| rot + aspect | −6.60% | −2.12% | −2.19% | 1.33 px |

**§7.1's second branch fired, harder than it was written.** The prediction was
"ceiling flat → style variation is not what hides reuse, and the idea dies
before any model is trained". It is not flat: canonicalising orientation
*destroys* two thirds of the cross-drawing reuse and half of the within-drawing
reuse, because it is a rotation in floats followed by rounding — the operation
§5 has just finished pricing. There is no version of this that can win: the
policies that preserve structure are provable nulls, and the policies that could
change anything are the ones that round.

### 6.3 QuickDraw has nothing to canonicalise, at any tolerance

`runs/canonical_quickdraw_n300_g1.json`: every ceiling is 0.00–0.01% under every
policy. And the null is genuine rather than a rounding artifact — relaxing the
library's match onto a coarser grid, which is a *detector* and not a compression
number, still finds almost nothing:

| grid | none | rot |
|---:|---:|---:|
| 1 | 0.00% | 0.00% |
| 4 | 0.00% | 0.00% |
| 8 | 0.01% | 0.00% |
| 16 | 0.20% | 0.08% |

At grid 16 the canvas has been coarsened to 16×16 and human sketches *still*
share 0.2% of their bytes, which canonicalising by rotation halves. This is the
same null the orbit oracle found for `REPEATX` on QuickDraw, from the other
side: **natural sketches repeat neither inside themselves nor between each
other.**

### 6.4 The by-product: the `CALL` ceiling, which is the largest one on record

`dm/eval/library.py` was built to give canonicalisation a place to be scored,
and it answers a queued question on the way. `docs/direction.md` §7.0 (b) names
`Op.CALL` with two distinct bodies as the demo path's next opcode, and
`docs/traps.md` carries the rule that paid for `REPEATX`: *before adding an
opcode to remove redundancy, measure the redundancy it would remove.*

Charging the full call site the ISA as specified requires — `XFORM code dx dy`
to place the body, `CALL id`, `ENDX`, seven bytes, because `CALL` takes a
`Kind.ID` and no placement — and capping the library at the 256 bodies one
`ID` byte can address, on the same 300 Tabler icons:

| oracle | bytes foldable |
|---|---:|
| `REPEAT` (within, translation) | 3.97% |
| `REPEATX` (within, D4 ⋉ translation) | 11.74% |
| **`CALL` (between, translation)** | **17.92%** |
| **`CALL` (between, D4 ⋉ translation)** | **25.21%** |

**Between-drawing reuse is 2.1× what the transform tier folds within drawings,
and the library that captures it is 54 bodies covering 420 call sites.** That is
the largest compression ceiling measured on a natural corpus in this project,
and it is available to an opcode that has been allocated since ISA v1 and never
built. It is also, as usual, corpus-dependent: **0.00% on QuickDraw**, so the
opcode would be worth nothing on the corpus that converges.

**Not licensed:** that a model would find any of it (this is an oracle allowed
to see the whole corpus, exactly like `REPEAT`'s), that 25.21% is achievable
(the library has to be transmitted, which it is, but the greedy admission order
is an underestimate and the `ID` cap a real bound), or that Tabler's ceiling
says anything about a corpus with 100k programs and a converged denominator.
