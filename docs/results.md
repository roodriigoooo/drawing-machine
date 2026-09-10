# Claim 1 — measured results

Moved out of `PLAN.md` on 2026-08-05 when the plan crossed its line budget.
`PLAN.md` §6 keeps the headline numbers, the amendment, and the pointer; this
file keeps the full tables, the retractions, and the per-axis reasoning.

Regenerate the live table with `python3 scripts/sweep.py --summarise-only`
(`runs/summary_synthetic.md`). Nothing here should be trusted over that file;
what is here is the *reading* of it.

Instrument history — why each guard exists — is in
[`history/instrument.md`](history/instrument.md). The write-up spec is in
[`claim1-writeup.md`](claim1-writeup.md).

---

## 6.3 What the schema-4 sweep showed (32/32, 1.9 h, zero failures)

`runs/summary_synthetic.md` is the live table — regenerate it rather than
trusting a copy here. The overfitting fix worked and exposed the same fault
mirrored: **every cell stops short of its optimum instead of past it, and by
wildly unequal amounts.**

| regime | codec | shape | params | bits/drawing | tail (bits/1k) | gen valid | len EMD |
|---|---|---|---|---|---|---|---|
| tokmatch | token | deep / square / wide | 912k / 826k / 937k | **161.4 / 161.4 / 164.2** | −0.24 … −0.53 | 0.99–1.00 | 3.3–7.1 |
| tokmatch | byte | deep / square / wide | 911k / 825k / 935k | **161.4 / 161.9 / 162.8** | −0.21 … −0.41 | 0.98–1.00 | 4.4–8.1 |
| tokmatch | bit | deep / square / wide | 887k / 792k / 886k | **177.9 / 204.4 / 234.9** | **−20 … −55** | 0.14–0.68 | 21.2–32.7 |
| tokmatch | token_typed | deep / square / wide | 1010k\* / 957k / 1134k\* | **161.3 / 162.7 / 169.1** | −0.22 … −0.48 | 1.00 | 5.2–6.4 |
| stepmatch | all four | square | 826k / 825k / 792k / 957k | **162.9 / 164.8 / 168.0 / 165.6** | −1.2 … −3.6 | 0.91–1.00 | 5.7–17.0 |

`\*` over the sub-1M framing. `halted` = 1.000 in all 32 cells again. Tails are
the §6.5 window definition, so they are 1.5–4× the values this table carried
before that fix — the flags do not change, but the numbers here did.

### The surviving results, and they are shape interactions

These are the rows where both arms are flat, so the difference is a cost:

| axis | shape | Δ bits/drawing | 95% CI | per-seed |
|---|---|---|---|---|
| typing (byte − token) | wide | **−1.45** | ±0.92 | −1.92, −0.98 |
| fusion (token_typed − token) | wide | **+4.87** | ±2.18 | +3.76, +5.97 |

> **Amended 2026-08-05 (evening): there were three, and the third was an
> artifact of how `tail` was measured.** `typing/square/tokmatch` read
> **+0.47 ±0.10** with both seeds agreeing to two decimal places, and it is now
> marked `(unconverged)`. Nothing about the run changed — the *guard* was
> measuring the final eval **interval**, which is 500 steps on a converged row
> and 1,474 on a token-matched one, and a single interval is dominated by the
> ±0.11–0.17 bits of eval-to-eval noise in a val mean. Measured over a fixed
> **fraction** of the schedule instead (§6.5), `token/square`'s worst seed sheds
> −0.53 rather than −0.20 bits/1k and the row fails the same guard it used to
> pass. The converged regime then measured the same axis at +0.93 ±1.50 with
> per-seed deltas +1.70 and +0.17, i.e. genuinely unresolved. **Typing at square
> is not a result.** Keep the retraction visible: an effect that survives a
> guard only because the guard was noisy is the failure mode this project keeps
> rediscovering one level up.

**This overturns the schema-3 reading that fusion is unresolved**
(§6.1, §6.2). It was unresolved then because the drift was the same size as
the effect.
At schema 4 the per-seed deltas agree in sign *within* a shape and the sign flips
*across* shapes — that is an interaction, not noise, and it is the first thing in
this project that the sub-1M framing directly predicts:

- **Fusion tracks the embedding table's share of the budget.** The 1293-entry
  table is 22% of parameters at wide (2 layers, d=192), 17% at square, 12% at
  deep (8 layers, d=96) — and the penalty is +4.87 / +1.27 (unres.) / −0.07 in
  that order. This *is* the IconShop objection (§4), measured: fused
  tokenisation is affordable only where depth carries the parameters instead.
- **Typing is not a parameter-count story** — byte's 258 vs token's 269 is ~2k
  parameters — and it is unresolved everywhere except wide, where byte *saves*
  1.45 bits. Mechanism not established. Report the measurement and say so; do
  not narrate a cause for a 1.5-bit effect at k = 2.

The step-matched typing (+1.93) and fusion (+2.77) rows are now correctly marked
`(unconverged)`: their arms differ 2–3× in tail, so part of each is "which arm
got going faster", which is exactly the confound the flag exists to name.

### The result that is independent of all of this — superseded by §6.4

Schema 4 read the bit arm as "+3.23 bits behind byte on NLL next to a 3× sampling
gap". The converged control removes the NLL gap entirely and keeps the sampling
gap, which is a sharper version of the same finding; §6.4 has it.

### bits/drawing still has no denominator

Recomputed on the schema-4 split (the val set is not identical to schema 3's —
`split()` filters val against train, so a 5× larger train set excludes more):

| reference | bits/drawing |
|---|---|
| raw bytes, no model | 338.5 |
| order-0 byte model | 281.4 |
| order-1 byte model (best n-gram; order-2 is 272.8, order-3 285.7 — overfit) | 261.3 |
| AR transformer, best cell (token_typed/deep, 1010k) | **161.3** |
| AR transformer, best in-budget cell (token/deep, 912k) | **161.4** |
| AR transformer, best converged cell (token/square, 826k) | **161.0** |
| achievable floor | in flight — §9.4 |

The transformer beats the best n-gram by ~100 bits, so the corpus is not
trivial. Without the floor a 1.22-bit fusion effect cannot be read as 0.8% of the
total or as 20% of the headroom. Unchanged priority, and now the *only* thing
between the surviving results and a claim-1 write-up that means something.
## 6.4 What the converged control showed (8/8, 2.4 h) — the granularity answer

`runs/summary_synthetic.md` is the live table. All four codecs, `square`,
12,000 steps, 2 seeds. **Every cell is flat**: `drift` +0.00, `tail` −0.18 to
−0.29 against a 0.5 tolerance, `best@` the final eval, `halted` 1.000 for the
third consecutive sweep.

| codec | params | bits/drawing | per-seed | tail | Mtok | gen valid | len EMD | gen p50 | wall |
|---|---|---|---|---|---|---|---|---|---|
| token | 826k | **161.0** | 160.9, 161.1 | −0.21 | 31.8 | 0.996 | 5.44 | 32 | 3.7 m |
| byte | 825k | **161.9** | 162.6, 161.3 | −0.29 | 31.8 | 0.996 | 7.02 | 29 | 4.2 m |
| bit | **792k** | **161.2** | 160.8, 161.7 | −0.27 | **254.3** | **1.000** | **13.56** | **25** | 48 m |
| token_typed | 957k | **162.2** | 162.1, 162.3 | −0.18 | 31.8 | 0.992 | 6.04 | 32 | 3.9 m |
### The granularity axis, which is the headline

| regime | steps bit/byte | Δ bits/drawing (bit − byte) | 95% CI | per-seed | verdict |
|---|---|---|---|---|---|
| token-matched | 1,128 / 8,844 | +42.52 | ±37.99 | +23.1, +61.9 | `(unconverged)` |
| step-matched | 3,000 / 3,000 | +3.23 | ±1.77 | +4.1, +2.3 | `(unconverged)` |
| converged | 12,000 / 12,000 | −0.72 | ±2.18 | −1.83, +0.40 | signs disagree — unresolved |
| **24,000-step rung** | **24,000 / 24,000** | **−0.67** | **±0.77** | **−1.06, −0.27** | **both negative — the answer** |

**A two-symbol alphabet costs one to two orders of magnitude less than a fixed
budget makes it look.** The penalty that read +42.5 bits at a fixed token budget
— statistically impeccable, both seeds agreeing, the interval clearing it — is
**−0.72 ±2.18** once both arms have stopped improving (and see the amendment
below: at 12,000 steps "stopped improving" was itself too generous), on an arm
with **4% fewer parameters** (792k vs 825k) and the *best* generation validity
in the table. The three budget-regime numbers are a separation rate
sampled at three points, exactly as §6.3 predicted, and the prediction is now
measured rather than argued.

Two things this is **not**:

- It is not a compute-matched claim, and the table says so: the bit arm spent
  254.3 Mtok against 31.8 and 48 minutes against 4. That is not a confound, it
  is what an 8× longer sequence *is*, and it is the whole reason the converged
  regime exists (§9.3). The practical answer to "what does a fixed budget buy?"
  is still the token-matched one, and for claim 4 the 8× is an inference cost
  that has to be paid on the device.
- It is not unequal *data* exposure. 12,000 steps × 64 is 7.68 epochs of the
  same 100k programs for every arm, so both arms see identical programs an
  identical number of times, and the byte arm cannot be given the bit arm's
  token count without 61 epochs on a split sized for 5.7. **Equal steps here is
  equal data and equal optimiser updates**, and the extra is FLOPs alone.

> **Amended 2026-08-05 17:00, and this is the honest state of it: −0.72 is
> provisional, because its reference arm was not converged either.** The
> 24,000-step rung (§9.3a) reports byte/square/s0 at **160.91** against the
> **162.62** the same seed gave at 12,000 — 1.7 bits, and the seed-0 ladder is
> 161.80 → 162.62 → 160.91, not even monotone. `tail` on that 12,000-step run
> was −0.29, comfortably inside tolerance. **A within-schedule tail passed on an
> arm with 1.7 bits left in it**, which is the strongest evidence the project
> has that a tail cannot establish an asymptote and the budget ladder can. It
> also sets the resolution floor for everything else: run-to-run spread at fixed
> config is ~1 bit, so no axis smaller than that is resolvable at k = 2.
>
> **Resolved 2026-08-05 22:30 by the 24,000-step rung at two seeds.**
> bit reads 159.84 and 160.60 against byte's 160.91 and 160.88, so granularity
> is **−0.67 ±0.77 with both per-seed deltas negative** — the sign agreement
> −0.72 never had. Quote the 24,000-step row, not −0.72. Typing resolves at
> **+0.11 ±0.07** (per-seed +0.10, +0.11); fusion does *not* resolve at 24,000
> (+0.93 ±0.98, per-seed +0.43, +1.43) having read +1.22 ±0.09 at 12,000.
>
> **And the headroom run says the corpus, not the representation, is now the
> constraint.** The byte codec at the `reference` shape — 4,812,544 parameters,
> 6.1× the arms — converges to **160.71** on its own ladder (161.64 at 12,000 →
> 160.71 at 24,000) and *loses to the 792k bit arm by 0.49 bits*. Six times the
> capacity cannot cross the band the four codecs sit in. That is the strongest
> available statement that these representations are equivalent, and it means no
> further seed or rung on Tier A can resolve fusion or anything else.

What was missing is the across-schedule half (§6.5), and running it is what
produced the amendment above. `budget_table` now pairs rungs on the seed, which
is what turned an apparent byte asymptote (161.9 at 8,844 and 161.9 at 12,000,
two-seed means) into a 1.7-bit shortfall (161.80 → 162.62 → 160.91, seed 0). A
tail is measured inside a schedule that anneals its LR to zero; the ladder is
not.
### The sampling gap is now clean, and it is the result to lead with second

At the converged point the bit arm is **first on likelihood and last on
sampling, with perfect validity**:

| | bit | byte | token |
|---|---|---|---|
| bits/drawing | 161.2 | 161.9 | 161.0 |
| gen validity / halted / wellformed | 1.000 / 1.000 / 1.000 | 0.996 | 0.996 |
| length EMD (bytes) | **13.6** | 7.0 | 5.4 |
| generated p50 bytes (real: 41) | **24–26** | 27–31 | 32 |

Schema 4 framed this as "a 2% NLL gap next to a 3× sampling gap". That framing
is now **too weak**: there is no NLL gap left, and the sampling gap is 1.9× byte
and 2.5× token with both seeds agreeing (12.6, 14.6). Nor is it a validity
failure — the bit arm is the only arm at 1.000 on all three validity columns.
The model that transmits a drawing in the fewest bits generates the worst length
distribution. **Teacher-forced likelihood cannot see compounding sampling
error**, and 8× more sequential draws is where it compounds. This is the
strongest argument in the project for reporting generation metrics at all, and
it no longer needs the convergence guard as a caveat.
### Fusion resolves at square, and the axes are per-byte rates

`fusion` at converged/square is **+1.22 ±0.09**, per-seed +1.19 and +1.25 — the
tightest resolved row in the table, and it agrees with the token-matched
+1.27 that the guard (rightly) refuses. Typing at converged/square is
+0.93 ±1.50 (per-seed +1.70, +0.17): unresolved, and the reason §6.3's +0.47 is
retracted.

Regressing the *per-program* paired difference on program length (all four arms
scored identical programs, so this is free) says something the per-drawing means
hide:

| axis | seed 0 slope | seed 1 slope | milli-bits per bytecode byte |
|---|---|---|---|
| fusion (token_typed − token) | +42.1 | +39.2 | **+0.041 bits/byte, consistent** |
| typing (byte − token) | +62.8 | +6.6 | inconsistent |
| granularity (bit − byte) | −44.9 | +12.2 | inconsistent |

**A representation cost is a per-symbol rate, not a per-drawing constant.** The
fusion penalty is +0.2 bits on the shortest val quintile and +2.7 on the
longest; "+1.22 bits/drawing" is that rate times *this corpus's* mean length and
will not carry to Tier B, whose programs are an order of magnitude longer. The
two unresolved axes are unresolved in the slope as well as in the mean, which is
a second, independent way of saying the same thing. `dm/train.py` now records
`val_bytes` — per-program length in bytecode bytes, the one unit all four codecs
share — so this becomes a reported column rather than an ad-hoc analysis when
Tier B lands.
## 6.5 The instrument fix the converged control forced

Reading the converged rows exposed the third generation of the same fault: a
guard that passes for the wrong reason. All of it is derived from `history`, so
it applies retroactively to every record on disk — no schema bump, no re-run.
Detail and evidence: [`docs/history/instrument.md`](history/instrument.md).

1. **`tail` measured the final eval *interval*, and interval width is a regime
   property.** `eval_every` is `steps // 6` under a token budget and a flat 500
   otherwise, so a converged row's tail covered the last 4% of a
   cosine-to-zero schedule and everything it is differenced against the last
   17%. Worse, one interval is noise: the val mean wanders ±0.11–0.17 bits
   between consecutive evals, a ±0.32 bits/1k floor against a 0.5 tolerance. It
   showed — converged/token reported a **positive** tail on a curve that fell
   monotonically across its whole last third. `tail` is now measured over a
   fixed **fraction** of the schedule (`TAIL_WINDOW = 1/3`, which divides both
   cadences exactly), dropping the noise floor to ±0.04. One verdict changes on
   existing data: `typing/square/tokmatch`, retracted in §6.3.
2. **A tail is a within-schedule test and cannot prove an asymptote**, because
   every run anneals its LR to zero over its own `steps`. `budget_table` prints
   the across-schedule half — the same arm at every budget on record, and what
   the largest increase bought. It is free, since the regimes differ in `steps`
   and nothing else, and it separates byte (+0.06 bits over a 1.36× increase:
   an asymptote) from bit (−6.78 over 4×: not yet).
3. **The paired table prints `steps arm/ref` and sorts by budget**, so one axis
   at one shape reads as a ladder and the reader sees whether the *difference*
   is stable across schedules — the cost-versus-rate test applied to the number
   actually reported rather than to each arm's level.
4. **`--converged-steps` did nothing.** The resume check skipped any cell whose
   tag was on disk and the tag does not carry the budget, so the documented
   procedure for "raise it if `tail` is still flagged" silently reported the
   12,000-step number in a row headed 20,000. `resume_verdict` now compares
   budgets and refuses with a message naming both, because overwriting would
   destroy the ladder in point 2. `--budget-rungs` adds extra budgets as
   first-class cells whose tags carry the step count.

At the time this section was written 60 tests passed; five pinned the tail window
against eval cadence and against val-mean noise, two pinned the budget-mismatch
verdict. The current count is in `PLAN.md` §6 — this file records what each
change was worth, not the running total.
