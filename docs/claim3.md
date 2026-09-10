# Claim 3 — the stroke planner, run by run

The full record of every claim-3 run, newest first, with the tables and the
retractions attached to the runs that produced them. `PLAN.md` §7 carries the
verdict and the two-line summary; this file is what that summary is short for.

**Read in this order:** the budget rung first — it is what makes the headline an
across-schedule statement rather than a snapshot, and it is where three of this
project's own numbers get corrected. Then run 4's re-run, where the claim splits
into a refuted half (likelihood) and a supported half (termination). Then the
AR-composition control, the tightest likelihood number and the one that closes
the bound question. Then run 4, which is what happened when the design met a
corpus. `PLAN.md` §9.10 is the design, and it no longer reads as written.

**The verdict, so it is not buried.** Five runs over two budgets. On likelihood
claim 3 is refuted: **+39.82 to +55.54** bits/drawing against an AR arm at
equal parameters on the same val programs, closing at **5.5 bits per doubling
against a gap of 40**. On termination — the axis the design named as its own
falsifier — it is supported at both budgets, and the win belongs to *having a
composition level* rather than to denoising, which ties with autoregression on
the same grid.

**And the likelihood loss is now known to be a *modelling* loss.** The stroke
decoder spends 0.035 bits/drawing where its summary leaves exactly one legal
value — against the flat arm's 88.15 on the same positions — so the
factorisation was never paying twice for its own conditioning. What survives is
8.6–9.9 bits on the coordinate extent, 18–25% of the gap.

**And on sample quality, added 2026-08-11 and sampler-audited 2026-08-13, the
two composition objectives separate.** Both planners *cover* corpus better than
flat arm and are worse on fidelity. At published CPU/raw support, `nna` is 0.585
for AR composition against 0.646 for diffusion. MPS/legal `k40` reproduces MMD
and NNA ordering in 5/5 draws. Full support leaves planner coverage/NNA fixed or
mixed and worsens MMD by +0.101/+0.138; top-k diversity gain belongs to direct
flat/conditional AR, not planner stroke decoding.

Section references of the form §N are to `PLAN.md`. Where a §7 reference below
points at claim-3 run detail, that detail is now in this file.

Live tables: `python3 scripts/sweep.py --summarise-only --data quickdraw`
→ `runs/summary_quickdraw.md`, which since 2026-08-08 carries `planner_table`,
the planner budget ladder and `planner_vs_ar_table`.

---

## The runs, and how to reproduce each


Kept as commands because each is still the way to reproduce its result, and
because run 4's was wrong in a way that only shows beside the corrected version.
Results follow the block, newest first.

Each is one command; none needs new code. Tags carry their budget, because
records are keyed by tag (§10).

```bash
# 1. Relativity, on the corpus that motivated it. Four cells, ~20 min each.
#    The prediction is directional: the delta view removes the translation
#    nuisance, so byte_delta should beat byte, and if it does not the argument
#    in dm/isa/relative.py is wrong and should be retracted rather than tuned.
python3 scripts/sweep.py --data tabler --codecs byte byte_delta \
    --seeds 2 --converged-steps 1000 --budget-rungs 2000 4000

# 2. Claim 2's first real number, on a corpus that can support an asymptote.
#    Tier A flat: 100k programs, 21% of bytes foldable, ceiling known exactly.
python3 -m dm.train --data synthetic --flatten --codec byte --steps 24000 \
    --tag synthetic_c2flat24000_byte_square_s0
python3 scripts/recovery.py runs/synthetic_c2flat24000_byte_square_s0.pt

# 3. Length generalisation, off the same checkpoint. No training at all.
python3 scripts/recovery.py runs/synthetic_c2flat24000_byte_square_s0.pt \
    --max-repeat 16 --min-repeat 8

# 4. Claim 3, against the AR arm it must beat, on one shared val split.
#    RAN. See below -- and note the baseline is built by the SECOND command,
#    not by `quickdraw_m5b24000eps4_*`, which is a different corpus.
python3 -m dm.train_planner --data quickdraw --categories cat dog bus car tree \
    --codec byte --steps 24000 --tag quickdraw_planner24000_byte_balanced_s0
python3 -m dm.train --data quickdraw --categories cat dog bus car tree \
    --codec byte --steps 24000 \
    --tag quickdraw_planbase24000eps2_byte_square_s0
```

> **The run-4 command in this file was the bug.** It named
> `quickdraw_m5b24000eps4_byte_square_s0` (**422.1 bits, exact**) as the
> baseline while specifying a *different corpus*: `cat dog bus car tree` against
> that arm's `cat bus flower sailboat bicycle`, and no `--rdp-eps`, so the
> planner ran at the default 2.0 against the baseline's 4.0. The val splits share
> no programs and differ 44% in bytes per drawing. Every planner default lines up
> with `dm.train`'s, so the matched AR arm is one command and ~22 minutes; that
> is the second line above, and it is what the comparison rests on.

The planner's number is an *upper bound* (`dm/models/planner.py`), so **lower
means it won with a handicap and higher does not mean it lost** — read
`composition_stderr` first, then the paired difference.

> **Run 4 aborted at its first eval on 2026-08-07 and was fixed (§10).**
> `per_program_bits` cast to float64 *before* leaving the device, which MPS
> refuses, so the run died at step 500 after paying for 500 steps of training.
> Both casts are now `.cpu().double()` and
> `test_the_eval_path_runs_on_the_accelerator` runs the whole eval path —
> bound, stroke NLL and generation — on MPS. **Measured: 0.28 s/step and ~11 s
> per eval, so 24,000 steps at `--eval-every 500` is ~3 h.**


---

### The redundancy test landed 2026-08-11 — the 39–49 bits are a modelling loss, and the decoder was never paying twice for the summary

**This is what the likelihood verdict means, and until now it was open.** The
planner factorises as `p(s) · p(x | s)` at `s = f(x)`, and the summary is a
deterministic function of the stroke — so the scheme transmits the summary's
information *twice*, once explicitly and once implicitly inside the stroke
decoder. If the decoder were re-specifying what it had already been told, a
large part of the 38.75–48.97 bit gap would be a coding inefficiency in this
implementation rather than a verdict on factorisation
([`docs/direction.md`](direction.md) §3).

**It is not.** `dm/eval/redundancy.py`, one forward pass per arm over the same
1,000 val programs, no training:

```bash
python3 scripts/redundancy.py \
    runs/quickdraw_plannerar12000eps2_byte_balanced_s0.pt \
    runs/quickdraw_plannerdiff12000eps2_byte_balanced_s0.pt \
    runs/quickdraw_planbase12000eps2_byte_square_s0.pt
```

| bits/drawing recoverable | planner, AR comp | planner, diff comp | flat AR (no summary) |
|---|---|---|---|
| **summary, total** | **8.632** | **9.851** | **112.739** |
| `first_point` | 0.030 | 0.023 | 84.502 |
| `coord_box` | 8.598 | 9.823 | 24.611 |
| `halt` | 0.005 | 0.004 | 3.626 |
| `fits` | 0.001 | 0.001 | 2.780 |
| ISA alone, subtracted out of every row above | 0.002 | 0.002 | 0.004 |
| bits/symbol where the summary leaves one legal value | **0.003** | 0.002 | **6.389** |
| bits/symbol everywhere else | 2.862 | 2.871 | 3.230 |

Every rule is exact — a value is infeasible only when no program consistent with
the summary could have put it there — and `isa` is subtracted from all four
summary rules so that nothing reclaimable from the opcode table alone is
attributed to the conditioning.

#### The two halves of the answer point in opposite directions, and only one is a bound

**Where the summary determines the byte, the answer is an upper bound and it is
~0.** 8.6% of stroke symbols have exactly one legal value: every stroke opens
with `MOVE` whose next two bytes *are* `(x0, y0)` (6,295 of 6,295 val strokes),
and every program's `HALT` is the final byte of a halting stroke (1,000 of
1,000). The decoder spends **0.035 bits/drawing in total** across all 13,786 of
them. You cannot recover what is not being spent, so this half of the hypothesis
is closed rather than bounded-above-by-a-large-number. The flat AR arm spends
**88.15** on the identical positions.

**Where the summary constrains without determining, the answer is a lower bound
and it is small but real.** Renormalising onto the extent box recovers **8.63 /
9.85 bits/drawing**. Against the planner's 4.16-bit replicate floor that is
1.8–2.1× and therefore resolvable; against the 38.75–48.97 gap it is **18–25%**.
A cleverer exact rule could recover more — this is what *these* rules recover —
so the honest statement is that the extent constraint leaves under ten bits on
the table and the gap is four to five times larger than everything found here.

#### The control, and why the absolute number needed one

Paired per program over the same val split, the flat AR arm wastes **+104.11
±3.71 bits/drawing** more than the planner's decoder on the identical
constraints. That is not a criticism of the flat arm — it was never given a
summary — it is the *value of the conditioning*, measured. And the asymmetry
runs the conservative way: the flat arm sees every previous stroke of the
program while the planner's decoder sees only its own stroke plus six bytes, so
a planner that failed this comparison would have failed it with an advantage.

> **A third axis separates the two composition objectives, in the same
> direction as the second.** The diffusion arm wastes **+1.22 ±0.13**
> bits/drawing more than the AR arm on the identical constraints. `mmd` and
> `nna` said the same thing that day. But 1.22 is far below any floor this
> project can quote for a model-to-model difference, so it is a consistency
> check and is not quotable on its own.

#### What could have made this lie, and the one that nearly did

Every number here is `−log₂ P(feasible)`, so **a rule that excludes the byte
actually present reports an unbounded saving at that position** — and it would
look exactly like a finding. `tests/test_redundancy.py` checks every rule at
every position of a real corpus for precisely this, and it caught a live fault
on the first run: `Summary.length` is u8, so on the two val strokes longer than
255 bytes the `fits` rule was applied to a length the summary cannot state and
excluded the true byte at 48 positions. The length-dependent rules now switch
off when the summary cannot state the length, and those two strokes are counted
in the report.

Reports: `runs/*_redundancy.json`. Twelve tests, including a uniform decoder
whose waste is checked against `log₂(vocab / |feasible|)` computed from the
masks with no model involved, and an alignment test that reads the decoder's
prompt offset straight off the returned log-probabilities — an off-by-one there
would attribute the first point's bits to its neighbour and invert the answer.

---

### The sample-quality axis landed 2026-08-11 — the planners cover better, draw worse, and the prediction that motivated the axis is refuted with its sign inverted

**This is the second half this project has never reported.** `bits/drawing`
scores a model on *real* programs, so it cannot see what the model draws;
`validity`, `length_emd` and `gen_strokes` can, but each is a **marginal** — one
summary statistic of the samples against the corpus's — and a distribution can
match every marginal anyone thought to check while matching nothing else.
`dm/eval/quality.py` compares the two *sets* in the geometry the VM emits.

Three arms, one command, at 12,000 steps and 825k parameters on one corpus
(`val` fingerprint `a54035e474876c1a`), 5 draws × 256 samples each:

```bash
python3 scripts/resample.py \
    runs/quickdraw_planbase12000eps2_byte_square_s0.pt \
    runs/quickdraw_plannerar12000eps2_byte_balanced_s0.pt \
    runs/quickdraw_plannerdiff12000eps2_byte_balanced_s0.pt \
    --seeds 5 --n 256
```

One invocation deliberately: the reference is subsampled to 256 under a seed of
its own and the floor is cached on the val fingerprint, so all three arms are
scored against the *identical* reference and the identical floor. Three
invocations would have produced three floors of the same quantity and invited
someone to difference them.

| | `coverage` ↑ | `mmd` ↓ (px) | `nna` → floor | `length_emd` ↓ |
|---|---|---|---|---|
| flat AR (824,704 params) | 0.402 ±0.017 | **12.561 ±0.107** | 0.632 ±0.026 | 12.759 ±4.554 |
| planner, AR composition | 0.465 ±0.015 | 12.753 ±0.069 | **0.585 ±0.026** | **4.419 ±0.646** |
| planner, diffusion composition | **0.476 ±0.025** | 13.117 ±0.130 | 0.646 ±0.013 | 5.628 ±3.332 |
| **floor** — real drawings, same two set sizes | **0.503 ±0.026** | **12.343 ±0.055** | **0.523 ±0.013** | — |

[Figure 8](figs/fig8_sample_quality.svg) draws the three arms against that
floor. Robust content is inversion between flat fidelity and planner coverage,
plus AR composition beating diffusion on MMD/NNA; exact planner coverage rank
was unresolved. All rows are CPU, `top_k=40`, temperature 1, raw model support — script
defaults at measurement time, now stated because sampler/backend are part of
result identity.

The floor is the same computation with real drawings from the **training** split
in the samples' place, at the same 256 v 256. It is what a *perfect* generator
scores, not what an unbeatable one does, and without it none of the three columns
has a readable value: `coverage`'s ceiling is not 1, `mmd` is in this corpus's
pixels, and `nna`'s textbook 0.5 holds only at matched set sizes.

**The flat arm's row reproduces the 2026-08-10 reading to the digit** — 0.402 /
12.561 / 0.632 and `length_emd` 12.759 ±4.554 — which is the determinism check
for the whole path, from the draw seeds through the reference subsample to the
Chamfer matrix.

#### The contrasts, with what five draws can and cannot carry

`se` is the two-arm standard error over five draws each; **overlap** is whether
the two sets of five draws intersect at all, which is the distribution-free
version of the same statement and the one that does not assume normality at
n = 5.

| column | contrast | Δ | se | t | overlap |
|---|---|---|---|---|---|
| `coverage` | flat AR → planner AR-comp | **+0.063** | 0.010 | +6.2 | none |
| `coverage` | flat AR → planner diff-comp | **+0.074** | 0.014 | +5.5 | none |
| `coverage` | AR-comp → diff-comp | +0.011 | 0.013 | +0.8 | overlaps |
| `mmd` | flat AR → planner AR-comp | +0.192 | 0.057 | +3.4 | overlaps |
| `mmd` | flat AR → planner diff-comp | **+0.556** | 0.075 | +7.4 | none |
| `mmd` | AR-comp → diff-comp | **+0.364** | 0.066 | +5.5 | none |
| `nna` | flat AR → planner AR-comp | −0.047 | 0.016 | −2.9 | overlaps |
| `nna` | flat AR → planner diff-comp | +0.014 | 0.013 | +1.1 | overlaps |
| `nna` | AR-comp → diff-comp | **+0.061** | 0.013 | +4.7 | none |

**The prediction this axis was built to test is refuted, and the sign is
inverted.** §3 predicted that the planners would *cover less* and sit *closer to
the floor on `nna`*, reasoning from the stroke-count marginal: the flat arm is
over-dispersed in stroke count (±5.85 against the corpus's ±3.88) and both
planners are slightly under (±3.14, ±3.19). Under-dispersion in a marginal did
not produce under-coverage in geometry — both planners cover **more** than the
flat arm, by 6.2 and 5.5 standard errors with no overlap in five draws, and the
diffusion arm's 0.476 is inside the floor's own spread of 0.503 ±0.026. **That is
the argument for having built the set comparison rather than adding a fourth
marginal**: the marginal predicted the wrong direction on the axis it was
supposed to point at.

**No arm sweeps, and the three numbers disagree on purpose.** The flat AR arm is
the best of the three on fidelity (`mmd` 12.561, +0.22 over the floor) and the
worst on coverage. Both planners buy coverage and pay for it in `mmd`. This is
exactly the trade the three columns exist to expose, and it is invisible to any
one of them: a report quoting `mmd` alone would call the flat arm the best model
here, and one quoting `coverage` alone would call it the worst.

**`nna` separates the two composition objectives that every other axis ties.**
On likelihood their stroke halves are +1.24 / +1.15 apart, under a third of that
half's own 4.16-bit replicate floor and therefore unresolvable. On termination
they tie (4.42 ±0.65 against 5.63 ±3.33). On the composition bound they tie. On
`nna` they are **0.061 apart with no overlap across five draws each**, and the
autoregressive composition level is the one nearer the floor — 0.585 against
0.646, where the flat arm reads 0.632. On `mmd` they separate the same way, by
0.364 px and again with no overlap. **So the diffusion composition level is not
merely "no better than autoregression on the same grid" — on the geometry the
samples actually contain it is measurably worse**, which is a stronger statement
than the termination tie supports and points the same way as §4.1's advice to
stop spending runs on the denoising axis.

#### What this does not establish, stated before anyone quotes it

- **These are five draws of *one checkpoint per arm*.** The spread is the
  sampler's, and the run-to-run spread of these three columns has never been
  measured. Every arm-to-arm gap above is resolvable against the *sampling*
  floor and against nothing else. A second seed per arm is what would turn any of
  these into a claim about the model rather than about the checkpoint — and it is
  the same k = 1 caveat every other two-arm comparison in this project carries.
- **The real-drawings floor is not a resolution floor**, and the two must not be
  conflated. It says what a perfect generator scores; it says nothing about what
  a re-run of the same config scores. A gap of 0.063 in `coverage` is large
  against the *draw* spread and unmeasured against the *replicate* spread.
- **`mmd` is a distance in canvas pixels at `CLOUD_POINTS = 128`.** It is
  comparable with another number from this report and with nothing else, since a
  Chamfer number is only comparable with another at the same resolution.
- **Nothing here reopens the likelihood verdict.** A 39–49 bit gap is not
  addressed by a metric measured in pixels, and this axis was never offered as a
  defence of it.

Reports: `runs/quickdraw_planbase12000eps2_byte_square_s0_gen_ar.json`,
`runs/quickdraw_plannerar12000eps2_byte_balanced_s0_gen_random.json`,
`runs/quickdraw_plannerdiff12000eps2_byte_balanced_s0_gen_random.json`. These
used `scripts/resample.py`'s default **CPU** device; schema-2 sampler-sweep files
key explicit MPS and must not be differenced against these as support-only
correction. Within MPS/legal reports, full support leaves planner-AR coverage/NNA
at −0.013/+0.000 and planner-diff at +0.000/+0.013 while MMD worsens +0.101 and
+0.138; planned counts are exactly unchanged. Flat-AR full support emits one
empty on draw 5, so its four-draw geometry is diagnostic only. See
[`conditioning.md`](conditioning.md) §7.4. The
estimator's own guards are in `tests/test_quality.py`: a collapsed generator
reads `coverage` at exactly 1/m and `nna` above 0.95, a memorising one reads
`mmd` ≈ 0 and `nna` ≈ 0, the floor at matched sizes lands within 0.1 of 0.5, and
the batched Chamfer is checked against `dm.eval.metrics.chamfer` — the pairwise
distance every other Chamfer number in this project was computed with.

---

### The budget rung landed 2026-08-08 — nothing here is an asymptote, and the AR composition level is the exception

`runs/quickdraw_planner{diff,ar}12000eps2_byte_balanced_s0.json`. Same corpus
fingerprint, same 825,080 parameters, same seed, half the schedule. **This was
the last item this file listed as owed, and it answered its own question against
the second of the two branches §7 named in advance.**

| arm | 12,000 | 24,000 | paired Δ | composition | stroke |
|---|---|---|---|---|---|
| diffusion comp | 612.38 | 604.25 | **−8.13** ±0.53 | 190.43 → 188.11 (−2.32) | 421.95 → 416.14 (−5.81) |
| AR comp | 602.16 | 596.66 | **−5.50** ±0.49 | 181.45 → 181.68 (**+0.23**) | 420.71 → 414.98 (−5.73) |

**Both arms cleared the 2.3-bit threshold this file set in advance by more than
2×, so the absolute planner totals are not quotable and the paired differences
are** — which is the position this file was already in, now on across-schedule
evidence rather than on a tail. (The 2.3 was itself named as a "same-config
floor" and it is not one — see the seventh fault below — but the margin is wide
enough that no candidate floor in the 1–3 bit range changes the reading.)
`budget_table` prints the diffusion rung as
**−9.22**, because it averages run 4 with its re-run as replicates before
differencing; **−8.13 is the re-run alone**, and it is the number to quote,
because run 4 predates four code fixes and the two are not a replicate pair in
any sense the ladder's docstring means.

**The composition levels do not converge together, and that is the finding.**
The AR composition level moved **+0.23 bits** over the second half of its
schedule — it is at its asymptote and marginally past it — while the diffusion
one moved −2.32 and is still descending. The two stroke decoders moved −5.73 and
−5.81. So:

- **181.68 is the first planner-side number in this project entitled to be
  called a cost rather than a rate.** The composition level's exact NLL has an
  asymptote and this is it, on across-schedule evidence.
- **The planner's whole remaining appetite for steps lives in its stroke
  decoder**, which is the half claim 3 does not contribute — the flat AR arm
  models the same bytes and is already flat (`drift +0.012`, `tail −0.320`).
  More budget buys the planner improvement in the one place where its
  factorisation is not the thing under test.

**The gap does not close. It is flat, and on the arm with the tightest number it
*widens*.** Both columns below are now matched-budget measurements: the flat AR
arm exists at 12,000 steps (`quickdraw_planbase12000eps2_byte_square_s0`,
563.42, 10.2 min) and at 24,000 (556.84), same corpus fingerprint, same val
split, and `planner_vs_ar_table` pairs them per program.

| arm | vs flat AR @ 12,000 | vs flat AR @ 24,000 | per doubling |
|---|---|---|---|
| diffusion comp | +48.97 ±2.06 | **+47.41 ±1.93** | −1.56 |
| AR comp | +38.75 ±1.59 | **+39.82 ±1.63** | **+1.07** |

> **RETRACTED — "the gap closes at 5.5 (AR) and 8.1 (diffusion) bits per
> doubling", and the ~7-further-doublings arithmetic built on it.** Those rates
> differenced *both* planner rungs against the *same* 24,000-step baseline, so
> what they measured was the planner arms improving with budget (−5.50 and
> −8.13) while the baseline was assumed still. It is not still: the flat AR arm
> buys **6.58 bits** over the same doubling, 563.42 → 556.84. Subtracting that,
> the diffusion gap moves −1.56 and the AR-composition gap moves **+1.07 in the
> wrong direction**. **The conclusion is unchanged and much stronger: the gap is
> not closing at any rate worth extrapolating, so there is no budget at which
> this factorisation catches the flat arm.**

> **And the correction this replaces was wrong by 6.2 bits, in the way §10
> forbids.** It predicted a true 12,000-step flat AR arm at **≈557.2**; the arm
> lands at **563.42**. The prediction was built by taking the schedule effect
> measured *on the planner arms* — a 12,000-budget run finishing 6.3–7.0 bits
> below the 24,000-run's step-12,000 reading — and applying it to the baseline.
> On the flat AR arm that effect is **0.33 bits**, not 7.0: its 12,000-step
> value 563.42 is essentially its 24,000-run's mid-schedule 563.75. **A schedule
> correction measured on one architecture does not transfer to another**, and the
> supporting argument — "the flat arm is already flat, which its `tail` of −0.32
> says" — read a tail at 24,000 as evidence about where the arm stood at 12,000.
> Its `tail` at 12,000 is **−1.27**. That is "a tail is a guard, never a rate"
> and "only `budget_table` can establish an asymptote" (§10), applied to the
> wrong rung of the very arm the trap was written about.

**What the rung is worth, stated plainly.** It changed no verdict and moved the
headline number by 6.6 bits, and it converted the project's central claim-3
sentence from an extrapolation with a rate in it to a measurement with no rate in
it at all. **Claim 3's likelihood loss is not a budget artifact** — not because
the gap closes slowly, but because across the only doubling on record it does not
close.

> **RETRACTED — "closing 39.83 bits at the *current* rate would take ~120,000
> further steps".** That divided 39.83 by a differential of tails, and a tail
> read as a rate is wrong by 2.7–3.8× (below). The sentence's own hedge — "and
> the rate does not hold" — was right, and the number in front of it was
> optimistic by more than an order of magnitude. The conclusion it supported is
> unchanged and is now better supported.

**`tail` overstates the remaining descent by 2.7–3.8×, measured.** This is the
project's first pair where a tail can be checked against what doubling the
budget actually bought:

| arm | `tail` at 12,000 | it predicts over 12,000 more | it delivered |
|---|---|---|---|
| diffusion comp | −1.83 bits/1k | −22.0 | **−8.13** |
| AR comp | −1.75 bits/1k | −21.0 | **−5.50** |

That is the safe direction for a *guard* — a run that looks unconverged is
unconverged — and the wrong direction for an *extrapolation*. This file did the
second once and the retraction above is what it cost. The calibration is in
`tail`'s own docstring in `scripts/sweep.py`, beside the guarantee it qualifies.

**The ELBO's slack is not a constant, and "≤6.58" was a snapshot.** The
composition-level difference between the arms reads:

| budget | comp, diffusion (bound) | comp, AR (exact) | difference |
|---|---|---|---|
| 12,000 | 190.43 | 181.45 | **8.98** |
| 24,000 | 188.11 | 181.68 | **6.43** |

It falls with budget, which is what a bound getting tighter as its model gets
better looks like. **6.43 is the tightest figure and it supersedes the 6.58**;
the direction means a converged pair would read lower still, so the conclusion —
the bound is not where the 40 bits are — gets stronger with every rung, not
weaker.

> **And the inequality's direction deserves one sentence of precision, because
> this file has stated it as a fact three times.** `comp_diff` is an upper bound
> and `comp_ar` is exact, so their difference bounds the ELBO's slack **only if
> the diffusion composition model's true NLL is no better than the AR model's**.
> The alternative is not absurd on its face and it is not credible here: for it
> to rescue claim 3 the diffusion level's true NLL would have to be ~142 bits —
> 39.8 below the AR level's exact 181.68, and 77 below the *parameter-free*
> per-(field, ordinal) marginal's 218.56. Say "either the slack is ≤6.43 or the
> diffusion level genuinely beats the AR one at the same level", and then say
> why the second is not on the table.

**The falsifier passes at both budgets, and the diffusion/AR tie holds at both.**
Five seeds, n=256, `scripts/resample.py`, off the finished checkpoints:

| arm / budget | strokes planned | `gen_length_emd` | validity |
|---|---|---|---|
| diffusion comp, 12,000 | 6.06 ±0.23 | **5.63 ±3.33** | 0.977 |
| diffusion comp, 24,000 | 6.02 ±0.18 | **4.59 ±0.98** | 0.973 |
| AR comp, 12,000 | 6.37 ±0.19 | **4.42 ±0.65** | 1.000 |
| AR comp, 24,000 | 6.31 ±0.22 | **3.78 ±0.97** | 1.000 |
| flat AR, 12,000 | — | **12.76 ±4.55** | 0.999 |
| flat AR, 24,000 | — | **10.40 ±4.48** | 0.996 |
| *val truth* | *6.295* | — | — |

Three readings, and the first is the one worth keeping:

- **Composition-level termination is done learning at 12,000 steps.** The plan
  lands on 6.06 and 6.37 strokes against a truth of 6.295 at half the schedule
  and does not move by the full one. The level that decides where a drawing ends
  converges long before the decoder that draws it — which is the same split the
  bits show (composition +0.23, stroke −5.73) arriving in a second coordinate.
- **§9.10's falsifier passes at matched budget, which is new.** Until the
  12,000-step flat arm existed, every planner row was read against a baseline
  trained twice as long. Now both budgets pair: at 12,000 the flat arm reads
  **12.76 ±4.55** against 5.63 and 4.42, and at 24,000 **10.40 ±4.48** against
  4.59 and 3.78. **The matched comparison is the larger of the two** — 2.9× on
  the mean at 12,000 against 2.3× at 24,000 — so the cross-budget reading was
  understating the win, in the opposite direction to the likelihood correction
  above. On both axes the honest comparison was the one nobody had run.
- **And denoising still does not separate from autoregression on it.** 5.63
  ±3.33 against 4.42 ±0.65 at 12,000, 4.59 ±0.98 against 3.78 ±0.97 at 24,000 —
  the AR composition level is nominally ahead at both and inside the spread at
  both. §5's specific claim gets no support from a second budget either. What
  budget does buy the diffusion sampler is **consistency**: its spread falls 3.33
  → 0.98 while its mean barely moves.

> **The sixth fault reproduced on a fresh record, which is why this table is not
> the one in the records.** The AR 12,000 run's own final eval reads
> `gen_length_emd` **7.11**; five seeds on that same checkpoint read **4.42
> ±0.65**. Nothing was wrong with the run. A sampling column is a draw, and the
> new `sampler` column in `scripts/sweep.py`'s planner table now says so in the
> caption beside every one of them.

#### The seventh fault: the eval reseeds the global RNG, so the arms train on different data orders

> **FIXED 2026-08-10.** The estimator takes a private `torch.Generator`, threaded
> through `cost_terms`/`nelbo_terms`; nothing reseeds the process.
> `PLANNER_SCHEMA` is 2, and `scripts/sweep.py` flags every schema-1 planner row
> rather than dropping it — the leak moved batch order, not arithmetic, so each
> row remains a valid measurement of its own model while two of them are not a
> controlled pair. Three tests cover it, and the first was checked against the
> old code to confirm it fails there.
>
> **Every number in this file was produced under the leak.** They are not
> retracted: the effect under test is 39–49 bits and every candidate floor is
> 1–3, so no plausible trajectory term reverses a verdict. What they carry is an
> unmeasured trajectory term, and **one re-run at the fixed schema is what turns
> the planner's replicate floor from a quoted number into a measured one.**

`per_program_bits` calls `torch.manual_seed(seed)` to pin the bound's
Monte-Carlo draw (`dm/models/planner.py`). That reset is **global**. So at every
eval both arms restart the stream at the same point, then consume different
amounts of it — the diffusion bound draws 32 noise samples per batch and the AR
level draws none, and the two samplers differ again — and training resumes from
different stream states. The next `torch.randperm` at an epoch boundary
therefore draws a **different epoch**. Measured directly, one eval on a tiny
config:

```
diffusion  next randperm(8) = [2, 7, 1, 0, 5, 4, 3, 6]
ar         next randperm(8) = [4, 0, 7, 3, 2, 5, 1, 6]
```

> **This corrects §10's "sharing no gradient is not sharing no RNG stream" in
> mechanism while confirming it in conclusion.** The entry said the composition
> objective "draws a different number of random numbers, which shifts everything
> downstream of it". True, and the consequence is sharper than a shift: **the
> two arms see different batch orders from the first epoch boundary after step
> 500.** They are not one run with a jittered stream, they are two runs on two
> data orders.
>
> **So the stroke half is not a same-config noise reference and this file has
> quoted it as one twice.** 414.98 against 413.80 was called "a free same-config
> noise reference of ~1.2 bits"; the three-run spread 413.80 / 414.98 / 416.14
> was called a "~2.3 bit" same-config floor and used to license "more seeds are
> not owed". Neither is same-config. **The honest floor for the planner is
> unmeasured**, and the cheapest way to measure it is the fix below, not more
> seeds.

**The fix is small, local and changes training, so it is not applied here.**
Give the bound its own `torch.Generator` instead of reseeding the global one,
and the stroke half becomes genuinely invariant to `comp_objective` — at which
point §9.10's "the stroke half comes out identical" becomes a *prediction* that
can be checked for the price of one run, and the residual after it is the real
floor. It changes every planner trajectory, so it is a `PLANNER_SCHEMA` bump and
a decision, not a cleanup.

**What the rung does not disturb.** The stroke gap between the arms reads
**+1.24** at 12,000 and **+1.15** at 24,000 — same sign, 0.09 apart. Under the
"different batch orders" mechanism that agreement has no reason to hold, so
either it is a coincidence at k=2 or the diffusion arm's stroke decoder is
systematically ~1.2 bits worse. **It is 30× smaller than the effect under test
and it is not worth a run to resolve**; it is recorded so that the next person
does not quote it as noise, which is what it was called before it replicated.

> **Resolved downward 2026-08-10 by the salvage below: it is a quarter of the
> stroke half's own floor and therefore not resolvable at all**, at any seed
> count this project would pay for. "Not worth a run" was the right call for the
> wrong reason.

#### The re-run at schema 2 was killed, and what it printed before it died

**`quickdraw_plannerar12000eps2s2_byte_balanced_s0` reached step 11,500 of
12,000 and left no file**, because the trainer assembled its record after the
loop. Full account, the wall-clock evidence and the verbatim transcript:
[`docs/history/lost-run.md`](history/lost-run.md). The trainer now writes at
every eval (`dm.train.checkpoint`), so this failure mode is closed and the run
is re-runnable at 1.6 h.

The console trajectory is real data with a truncated provenance — the printed
fields are read straight out of `record` — so it is quoted here as an **argument
and not as a measurement**. Differenced against the schema-1 twin at the same
corpus, parameters, budget and seed, over the ten evals from step 7,000:

| | mean Δ (schema 2 − schema 1) | sd | range |
|---|---|---|---|
| **total bits/drawing** | **+4.69** | 0.23 | [+4.30, +5.22] |
| composition (exact, AR level) | **−0.20** | 0.07 | [−0.29, −0.05] |
| stroke decoder | **+4.91** | 0.23 | [+4.55, +5.45] |

**§7's retracted observation replicates in shape and roughly doubles in size.**
The claim that the planner's spread "lives entirely in the stroke decoder" was
retracted above only for its *label*, and a genuinely same-config pair says the
same thing: the composition level agrees to 0.20 bits and the stroke decoder
disagrees by 4.91. What changes is the number — **~4.7 bits, not ~2.3**, itself
superseded on 2026-08-11 by a paired **4.16 [3.67, 4.64]** — and it
is the larger of the two that every claim-3 effect has to be read against.

What it is not: one pair, so no spread; cross-schema by construction, which is
what makes it controlled and also what makes a systematic effect of the fix
indistinguishable from trajectory noise at n = 1; and without `val_bits`, so
there is no paired interval, which is the form every number in this file is
quoted in. **The re-run is still owed and now costs 1.6 h to collect properly.**

**No verdict in this file moves.** The likelihood gap is 38.75–48.97 bits, which
is 9.3–11.8× the measured 4.16-bit floor rather than 17–21× a 2.3-bit one.

---

### Run 4's re-run landed 2026-08-08 — the likelihood loses again, and the falsifier was never measuring the model

`runs/quickdraw_plannerdiff24000eps2_byte_balanced_s0.json`. Same corpus
fingerprint, same 825,080 parameters, same 24,000 steps and same seed as run 4
and as the AR-composition control, differing only in the four fixes.

| | total | composition | stroke |
|---|---|---|---|
| run 4, as first recorded | 602.06 | 188.26 ±0.50 | 413.80 |
| **run 4's re-run** | **604.25** | **188.11 ±0.49** | **416.14** |
| AR-comp control | 596.66 | 181.68, exact | 414.98 |
| flat AR `square`, 824,704 params | 556.84 | — | — |

| pair | Δ bits/drawing | 95% CI | lower on |
|---|---|---|---|
| **re-run − flat AR** | **+47.41** | [+45.47, +49.34] | 13/1000 |
| AR-comp planner − flat AR | +39.82 | [+38.19, +41.45] | 7/1000 |
| re-run − AR-comp planner | +7.59 | [+6.47, +8.70] | 327/1000 |
| **re-run − run 4 (same config)** | **+2.18** | [+1.74, +2.63] | 378/1000 |

**Claim 3 loses on likelihood for the third time**, and the last row says how
much of that number is real: two runs at the same seed, corpus, objective and
budget land 2.18 bits apart, so the diffusion planner's loss is **+46.3 ±1.1**
and the effect is 20× its own replicate spread.

**The two halves replicate very differently, and the surprise is which.** The
composition bound — a Monte-Carlo estimate — came out 188.26 and 188.11, **0.15
bits apart**, while the *exact* stroke NLL came out 413.80 / 414.98 / 416.14
across the three planner runs, **2.34 apart**. The seeded draw (`per_program_bits`
calls `torch.manual_seed`) is why the bound is reproducible; the stroke half
carries the trajectory noise. **So the planner's same-config resolution floor is
~2.3 bits and it lives entirely in the stroke decoder** — an upward revision of
the 1.18 the control estimated from two runs.

> **RETRACTED 2026-08-08 — none of these three runs is same-config, so this is
> not a resolution floor.** `torch.manual_seed` inside `per_program_bits` resets
> the **global** stream at every eval, so the three arms consume different
> amounts of it and resume training on **different batch orders** (§7, the
> seventh fault). The observation stands — the bound replicates to 0.15 bits and
> the stroke half to 2.34 — and the label does not: 2.34 is the spread of three
> trajectories, not of three draws of one. **The planner's replicate floor is
> unmeasured**, and §7's "not owed: more seeds" rested on this number.

**The ELBO's slack replicates too: 6.58 and 6.43 bits.**

> **AMENDED 2026-08-08 — it is not a replicate pair, it is a ladder, and it
> falls.** Read against the 12,000-step rung's 8.98 the sequence is 8.98 → 6.43,
> i.e. the bound tightens as the model improves. **6.43 is the tightest figure
> and supersedes ≤6.58** everywhere it appears below, and the direction means a
> converged pair reads lower still — so every rung strengthens "the bound is not
> where the 40 bits are" rather than weakening it (§7).

#### The fifth fault: the composition sampler's unmasking order

**Every generation column of every diffusion-composition run in this project has
described the sampler rather than the model**, and the re-run is the third time,
after the NaN Gumbel and the half-structural HALT. This one is not a typo.

`CompositionDenoiser.sample` committed the **most confident** masked position
first. That is MaskGIT's heuristic for image tokens. It is **not the reverse
process of the kernel `nelbo_terms` trains against**: the forward process masks
each position independently with probability `t`, so the masked set is uniformly
random and the reverse step must unmask a **uniformly random** subset. Importing
the heuristic turns the sampler into a greedy search over the joint wearing the
reverse process's clothes.

**On this grid the greedy search is a ratchet.** 86% of slots are EMPTY, so the
most confident masked position is almost always one the model wants to fill with
EMPTY; committing it raises the — correct — posterior for EMPTY at its
neighbour, which is then the most confident position. Every step shortens the
plan and none lengthens it.

Four measurements, all off the finished checkpoint and none of them training:

- **The model was never wrong.** Its one-pass marginals from a fully masked grid
  sum to **6.16 expected strokes against the corpus's 6.29**, and track the
  corpus slot by slot to ~0.02.
- **The sampler gets worse the more steps it is given**: 6.84 strokes at 1 step,
  4.09 at 4, 3.19 at 16, 3.13 at 64, 3.17 at 192. An under-resolved sampler
  converges *toward* the model's joint; this converges away from it, which is
  what separates a biased sampler from a coarse one.
- **The reader was not the problem.** `grid_summaries` stops at the first
  all-EMPTY slot, so a hole would silently truncate — measured, **0.00** stranded
  slots per grid. Contiguity is perfect; the level really does plan 3.47.
- **Noise does not rescue it, and neither does slot coherence.** Un-annealed
  Gumbel reads 3.29 strokes against annealed 3.26, because *any*
  probability-weighted order walks the same ratchet. Committing whole slots at a
  time is worse than plain random (EMD 9.14 against 3.61).

**What the correct reverse process does to the same checkpoint.** Five seeds,
n=256, `scripts/resample.py`; the flat AR arm re-measured the same way:

| arm / sampler | strokes planned | `gen_length_emd` | len p50 | validity |
|---|---|---|---|---|
| diffusion comp, confidence order | 3.26 | 25.08 ±1.80 | 136 | 0.919 |
| **diffusion comp, random order** | **6.00 ±0.18** | **4.59 ±0.98** | 151.7 | 0.973 |
| AR comp control | 6.31 ±0.22 | 3.78 ±0.97 | 152.2 | 1.000 |
| flat AR baseline | — | **10.40 ±4.48** | 160.0 | 0.996 |
| *val truth* | *6.295* | — | *154* | — |

#### The sixth fault, and it changes a conclusion in this file

**`gen_length_emd` is a draw and this file has been quoting the final eval of
it.** The flat AR baseline's record reads **2.72**; the same checkpoint over five
seeds reads **10.40 ±4.48**, and that run's own last 25 evals span **2.72 to
42.45**. 2.72 is the minimum of 25 draws.

> **RETRACTED — "§9.5a's termination failure barely exists here, so claim 3's
> termination prediction had almost no room to win on this venue."** It was
> inferred from that 2.72. The venue had 10.40 ±4.48 of room and the prediction
> won on it. Every resolution-floor rule in this project was written for
> `bits_per_drawing` and none was ever applied to the sampling columns, which
> are noisier — the AR arm's EMD has a coefficient of variation of 0.43 against
> `bits_per_drawing`'s ~0.005 on the same corpus.

#### What this settles for claim 3

**§9.10's falsifier passes.** It was stated as *"if `gen_length_emd` does not
improve, the scale argument in §5 is weaker than it claims"*. Against the flat
AR arm it improves **2.3× on the mean and 4.6× on the spread**, and the spread is
the sharper half: the AR arm re-decides termination on every drawing and the
planner decides it once, at the coarse scale, before a stroke byte exists.

**But the win belongs to the factorisation, not to denoising.** The diffusion
composition level (4.59 ±0.98) and the AR one (3.78 ±0.97) do not separate —
0.81 apart against ~1.0 of spread each. What beats the flat arm is *having a
composition level*, and §5's specific claim, that the denoising objective
decomposes by scale where autoregression cannot, gets no support from this
either: on the one axis where it was made falsifiable, the two objectives tie.

**None of this touches the likelihood.** The sampler has no gradient and no path
into the bound — `nelbo_terms` runs on the true grid — so +47.41, +39.82 and the
slack all stand exactly as recorded. Claim 3's headline is unchanged and
negative; what changed is that its *own* named falsifier now has a reading, and
it is the opposite sign to the headline.

---

### The AR-composition control landed 2026-08-08 — the 45 bits are not the bound

`runs/quickdraw_plannerar24000eps2_byte_balanced_s0.json`. Same corpus
fingerprint (`c0ef9464b751f20c` / `a54035e474876c1a`) as both run-4 arms, same
**825,080** parameters, same 424,240 / 400,840 split between the levels.

| arm | total | composition | stroke | slack it carries |
|---|---|---|---|---|
| planner, diffusion comp (run 4) | 602.06 | 188.26 ±0.50 | 413.80 | factorisation + diffusion ELBO |
| **planner, AR comp (the control)** | **596.66** | **181.68, exact** | 414.98 | **factorisation only** |
| flat AR `square`, 824,704 params | **556.84** | — | — | none, exact |

`composition_stderr` is 0.0 and `bound_sources` reads `["factorisation"]`, so the
control did exactly what §9.10 built it to do.

**The composition-level ELBO's slack is at most 6.58 bits at this budget** — 6.43
on the re-run, 8.98 at 12,000, so it falls as the model improves (§7). Of run 4's
+45.22,
no more than that could ever have been the bound. This is §9.10's first predicted
outcome — *"AR near 188 means the masked-diffusion bound is tight and the level
is simply hard"* — and it is the cheap answer the control existed to buy.

Paired on the shared val split, 1,000 programs:

| pair | Δ bits/drawing | 95% CI | lower on |
|---|---|---|---|
| **AR-comp planner − flat AR** | **+39.83** | [+38.20, +41.46] | 7/1000 |
| diffusion planner − flat AR | +45.23 | [+43.32, +47.15] | 12/1000 |
| AR-comp − diffusion planner | −5.40 | [−6.53, −4.27] | 618/1000 |

**+39.83 is now claim 3's tightest number and it is still a loss.** Its only
remaining slack is the factorisation, and with `s = f(x)` deterministic that
slack is the mass the model puts on alternative `(s, strokes)` pairs decoding to
the same `x` — small, and certainly not 40 bits.

> **Do not read −5.40 as "AR models composition better".** The diffusion row is
> an upper bound and the AR row is exact, so that comparison is the slack
> question again, not a modelling result. The control's job was never to rank the
> two objectives on likelihood; it was to produce a planner total carrying
> **one** kind of slack, and the row that decides anything is the +39.83.

**The falsifier flips at the composition level, and that is the finding.**

| | strokes (val truth 6.301) | len EMD | validity | faults / 128 |
|---|---|---|---|---|
| diffusion comp, as recorded | 3.56 | — (void) | 0.195 | trunc 44, no_halt 42, unknown 17 |
| diffusion comp, four fixes | 3.4 | 26.7–37.0 | 0.844–0.914 | no_halt 11–20 |
| **AR comp (the control)** | **6.078** | **4.35** | **1.000** | **none** |
| flat AR arm | 6.55 prims | 2.72 | 1.000 | none |

`gen_faults` is empty, `gen_truncated` is 0.0078, `gen_len_p50` 151 against the
flat arm's 154. **Structural termination works — with an AR composition level.**
The stroke count lands 3.5% under truth where the diffusion arm lands 46% under.

> **Two rows of that table were superseded on 2026-08-08 and the conclusion
> below was half wrong.** The diffusion rows measured the sampler's unmasking
> order, not the level: corrected, the same checkpoint plans **6.00 ±0.18**
> strokes and reads `gen_length_emd` **4.59 ±0.98**. And the flat arm's 2.72 is
> the minimum of 25 draws — five seeds on that checkpoint read **10.40 ±4.48**.
> Read the re-run's section above for the table that supersedes this one. What
> survives here unchanged is the AR control's own row, and the sentence it
> licensed: structural termination works.

**This settles §7's item-3 venue question against both of its options, without
training anything else.** At the *same* 424,240 composition parameters on the
*same* corpus, the level plans 6.078 strokes and reaches 181.68 exact bits. So
the composition level was **not short of capacity** — `PLANNER_SHAPES["layout"]`
is not the sweep — and QuickDraw was **not short of composition to plan** — Tier
D is not what §9.5a's undershoot at this scale was asking for. The undershoot is
a property of the **masked-diffusion composition level and its sampler**.

> **It was the sampler, and the "or the objective" half of that sentence was
> wrong.** Written 2026-08-07 as "the diagnosis points inward, at the objective,
> not outward at a corpus" — the direction was right and the target was not. The
> objective was fine: the model trained under it has marginals within 2% of the
> corpus's stroke count. Only the reverse process was wrong. **Naming two
> suspects and then confirming "inward" is not the same as identifying one**,
> and the cost of not separating them was a section of this file recommending a
> diagnosis of the objective.

> **"The stroke half comes out identical" was §9.10's claim, and it is false.**
> The levels share no gradient, so the design took the stroke term to be
> invariant to `comp_objective`. Measured: **414.98 against 413.80, 1.18 bits
> apart** at the same seed, same corpus, same 24,000 steps. `comp_objective`
> changes how many RNG draws the run makes, which shifts the stroke level's
> stream — the same mechanism as Tier C's 158.51 / 157.18 / 157.39 at fixed seed
> (§Run 1). Two consequences, both worth keeping: the −6.58 composition
> difference is **not** cleanly isolated, and this is a free same-config noise
> reference of **~1.2 bits** for the stroke half on this corpus.

**Neither planner arm is converged, and the control is not either.** Control
`drift +0.000`, `tail −0.565` bits/1k against a 0.5 tolerance; run 4 `−0.665`;
the flat AR baseline `drift +0.012`, `tail −0.238` and passes. The differential
is ~0.33 bits/1k and decaying, so closing 39.83 bits at the *current* rate would
take ~120,000 further steps and the rate does not hold. It does not cover the
gap — but item 4 below stands: a budget rung is owed before any absolute planner
number is quoted.

---

### Run 4 landed 2026-08-07 — claim 3 loses by 45 bits, and four faults were found getting there

**The comparison, once both arms were on one corpus:**

| arm | bits/drawing | kind |
|---|---|---|
| planner, `balanced`, 825,080 params | **602.06** | upper bound |
| AR `square`, 824,704 params | **556.84** | exact |
| **paired difference (planner − AR)** | **+45.22** | 95% CI [+43.31, +47.14] |

Planner lower on **12 of 1,000** programs. The AR arm converged (drift +0.012,
tail −0.320); **the planner did not** (tail −0.907 against a 0.5 tolerance), so
it has room — but −0.907 bits/1k, decaying, does not cover 45 bits.

`composition_stderr` is 0.496, so the bound's Monte-Carlo noise is not the
issue. What remained was the bound's *slack*: to claim the planner did not really
lose, the ELBO had to be ≥45 bits loose. **§9.10's control measured it and it is
≤6.58** — see the control's section above, which supersedes this paragraph.

**The AR arm on this corpus reads validity 1.000 and len EMD 2.72.** §9.5a's
termination failure barely exists here, so claim 3's termination prediction had
almost no room to win on this venue — worth knowing before choosing the next one.

**The four faults, all found in the first day the instrument was pointed at
anything.** Each is a general trap and each is in §10:

1. **The corpus mismatch above** — void comparison, and no key in the project
   could see it.
2. **The composition sampler ordered its unmasking by NaN.** `-torch.log(-torch.log(u).clamp_min(1e-9))`
   parses as `-(log(u).clamp_min(...))`; `log(u)` is non-positive, so every
   element became `1e-9` and the outer log got a negative. Nothing raised, and
   ordering by NaN is backend-defined: **9.17 strokes per grid on CPU against
   3.70 on MPS, one checkpoint and one seed.** So the fix for the mode-seeking
   trap §10 already documents *had never run*.
3. **Termination was half structural.** The composition level set `halts` and
   then left the HALT *byte* to the decoder, which omitted it on 42 of 128
   grids, while the design claimed termination was structural.
4. **The falsifier was never recorded.** `dm/models/planner.py` states the
   prediction as *"if `gen_length_emd` does not improve, the scale argument is
   weaker than §5 claims"*, and `train_planner` called `corpus_stats` alone —
   so `gen_length_emd`, `gen_len_p50` and `gen_truncated` were absent from every
   planner record while `scripts/sweep.py` read them and printed `nan`.

**What fixing 2–4 does to generation**, same checkpoint, MPS, three seeds:

| | validity | faults / 128 |
|---|---|---|
| as recorded | 0.195 | truncated 44, no_halt 42, unknown_opcode 17 |
| + Gumbel fix | 0.50–0.68 | — |
| **+ boundary and structural HALT** | **0.844–0.914** | **no_halt 11–20 only** |

`truncated` and `unknown_opcode` are **gone**, and `no_halt` now means one
thing: the composition level never planned a terminator. Left reported rather
than forced — a plan with no halting stroke is a model failure, and inventing
one would be inventing an output.

**But `len EMD` reads 26.7 / 37.0 / 31.4 against the AR arm's 2.72, so the
falsifier still reads negative.** The failure is not decoding. The plan commits
to **3.4 strokes where the truth is 6.30** — §9.5a's undershoot reappearing at
the coarse scale, which is the one axis claim 3 predicted it would fix.

> **RETRACTED 2026-08-08 — the falsifier does not read negative, and both
> numbers in that sentence were wrong.** The 26.7–37.0 is the fifth fault, a
> sampler whose unmasking order was MaskGIT's rather than this kernel's reverse
> process; corrected, the same checkpoint reads **4.59 ±0.98** and plans 6.00
> ±0.18 strokes. The 2.72 is the sixth, the minimum of that arm's own 25 draws
> against a five-seed mean of **10.40 ±4.48**. **The falsifier passes.** See the
> re-run's section above. Note what this cost: run 4's four faults were found in
> one day and this paragraph, written the same day, spent two of its numbers on
> the two faults that had not been found yet.

**Where the composition level's 188.26 bits go**, measured off the checkpoint,
and what a model with no parameters costs on the same split:

| model | bits/drawing |
|---|---|
| order-0 | 399.34 |
| flat per-slot marginal (192 dists) | 269.30 |
| per-(field, stroke-ordinal) marginal + explicit count model | **218.56** |
| **trained denoiser, 424k params, 3 h** | **188.26** |

**30.3 bits, 16%, over a model with no parameters** — and its tail is already
flat at −0.184 bits/1k, so more steps will not move it. Field split: x0 46.4,
y0 42.3, width 38.8, height 39.2, length 20.9, halts 0.85.

**Three things this ruled out, all without training anything:**

- **The stroke decoder is not the problem, and it uses its conditioning
  exactly as designed.** Per-position probe: the MOVE operands cost **0.001
  bits** — the copy from the summary is learnt — opcodes 0.000–0.003, and only
  real coordinates cost 4.1–4.9.
- **`width` and `height` are load-bearing, not overhead.** They cost 78.06 bits
  at the composition level and the obvious suspicion is that they are redundant
  given the stroke bytes. Measured instead: perturbing them in the prompt costs
  **+1026 bits** (to corpus median) and +535 (to zero), against +796 for the
  x0,y0 control. The decoder leans on extent harder than on position.
- **Run 1's relativity does not transfer to this level.** Delta-coding
  consecutive stroke origins is worth **1.64 bits** at the marginal level
  (218.56 → 216.92): successive QuickDraw strokes do not start near each other.
  Do not build relative summaries.

---

## The ledger — the four runs that were owed, all now closed


1. ~~**The AR-composition control** (§9.10).~~ **DONE 2026-08-08.** The 45 bits
   were the model, not the bound: the ELBO's slack is 6.43 and the planner
   still loses by **+39.82** carrying factorisation slack alone.
2. ~~**Run 4 again**, `--comp-objective diffusion`, with the four fixes.~~
   **DONE 2026-08-08.** +47.41 against the flat AR arm; its generation columns
   are retracted in turn, for the fifth fault (§7), and re-measured off the
   checkpoint by `scripts/resample.py` rather than by a third launch — the
   sampler is a pure function of the checkpoint and the seed, and the likelihood
   it would re-measure is unaffected by any of this.

   ```bash
   # the run itself, kept because it is still how the record is reproduced
   python3 -m dm.train_planner --data quickdraw --categories cat dog bus car tree \
       --codec byte --comp-objective diffusion --steps 24000 \
       --tag quickdraw_plannerdiff24000eps2_byte_balanced_s0

   # the corrected generation columns, seconds and no training
   python3 scripts/resample.py runs/quickdraw_plannerdiff24000eps2_*.pt --seeds 5
   python3 scripts/resample.py runs/quickdraw_plannerar24000eps2_*.pt --seeds 5
   python3 scripts/resample.py runs/quickdraw_planbase24000eps2_*.pt --seeds 5
   ```

   **The tag is new on purpose.** Re-using `quickdraw_planner24000_byte_balanced_s0`
   would overwrite the record that carries the retraction block — §10's
   "records are keyed by tag" applied to a record whose *value* is that it names
   its own bad columns. And `--rdp-eps` is **omitted**, because omitting it is
   what produced corpus `c0ef9464b751f20c` on the control and on
   `quickdraw_planbase24000eps2_byte_square_s0`; typing `--rdp-eps 2.0` gives
   identical programs and a non-empty `extra`.

   > **A third launch is not owed, and saying why matters.** Run 4's original
   > retraction read "cannot be recomputed — sampling is not a function of
   > anything the record stores", which is true of the *record* and not of the
   > checkpoint beside it. The rule that a record's columns come from the run
   > that wrote them is worth keeping, so the corrected numbers live in a side
   > report keyed by checkpoint **and sampler** — `scripts/recovery.py`'s pattern,
   > and for the same reason: the reading is a comparison between two reports, so
   > a key that cannot tell them apart loses the result rather than a file.
3. ~~**Then decide the venue, and do not decide it before.**~~ **The control
   decided it, and against both options.** At the same 424,240 composition
   parameters on the same corpus, an AR composition level plans 6.078 strokes
   against a truth of 6.301 and reaches 181.68 exact bits. So the level is not
   short of *capacity* (`PLANNER_SHAPES["layout"]` is not the sweep) and
   QuickDraw is not short of *composition to plan* (Tier D is not what this
   undershoot was asking for). **What is owed instead is a diagnosis of the
   masked-diffusion composition level itself** — its objective and its sampler
   are the only things left holding the 3.4-against-6.30 undershoot.
   ~~Run 4's re-run is the first evidence.~~ **DONE 2026-08-08, and it was the
   sampler, entirely.** The unmasking order was MaskGIT's rather than the
   reverse process's; corrected, the same checkpoint plans 6.00 ±0.18 strokes
   against a truth of 6.295 (§7). The objective is exonerated for the
   undershoot and the level was never broken.
4. ~~**Both arms need a budget rung. This is now the only thing owed.**~~
   **DONE 2026-08-08.** Run 4's `tail` was −0.907, its re-run's −0.840 and the
   control's −0.782, all against a 0.5 tolerance, so no planner number in this
   file was an asymptote and only the across-schedule test binds.

   ```bash
   # 12,000 is half the schedule, so the rung is ~1.6 h per arm and the pair
   # is what `budget_table` needs. New tags, because records are keyed by tag.
   python3 -m dm.train_planner --data quickdraw --categories cat dog bus car tree \
       --codec byte --comp-objective diffusion --steps 12000 \
       --tag quickdraw_plannerdiff12000eps2_byte_balanced_s0
   python3 -m dm.train_planner --data quickdraw --categories cat dog bus car tree \
       --codec byte --comp-objective ar --steps 12000 \
       --tag quickdraw_plannerar12000eps2_byte_balanced_s0

   # the generation columns, which a record's single final draw cannot supply
   python3 scripts/resample.py runs/quickdraw_planner{diff,ar}12000eps2_*.pt --seeds 5
   ```

   **What it could and could not settle was stated before it ran, and the second
   branch is what happened.** The two branches were: *"if the 12,000 rung reads
   within ~2.3 bits of 24,000 the absolute numbers are quotable and claim 3 is
   closed on likelihood; if it reads much lower, only the paired differences may
   be quoted, exactly as now."* It read **8.13** and **5.50** lower. So the
   absolute totals stay unquotable and the differences stay readable — and the
   prediction's own threshold turned out to rest on a floor that is not a floor
   (§7, the seventh fault), which does not change the reading here because both
   arms cleared 2.3 by more than 2×.

   ~~Not owed: **more seeds.**~~ **The reason given was wrong even though the
   conclusion survives.** It read: *"the planner's same-config spread is ~2.3
   bits, measured three times over on the stroke half"*. Those three runs were
   not same-config — the eval reseeds the global RNG and the arms train on
   different batch orders (§7) — so the project has **no measurement of the
   planner's replicate floor at all**. Seeds are still not what is missing: the
   effect is 39.8–55.5 and every candidate floor discussed here is 1–3 bits. But
   "20× the floor" is now an estimate rather than a measurement, and the run that
   makes it a measurement is the generator fix, not a seed.

   Also not owed: **a third diffusion launch.** The likelihood is unaffected by
   every fault found, and the generation columns are re-measurable off the
   checkpoint in seconds (item 2).

