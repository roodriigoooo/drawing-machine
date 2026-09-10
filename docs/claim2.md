# Claim 2 — structure discovery, run by run

"A model trained on flat traces should recover `REPEAT`." The corpus, the metric
and the ceiling are in `PLAN.md` §9.8; this file is the record of the runs that
produced the numbers, newest first. `PLAN.md` §7 carries the verdict.

**The verdict.** It recovers, inside the counts it was trained on, and by a lot:
**22.75 of 28.20 bits/drawing**, `recovery = +0.807` at 24,000 steps on Tier A
flat with both convergence guards passing. It has **not** learned `REPEAT` as a
rule — the benefit vanishes at exactly the trained count bound, and the failure
is a hard prior over the count rather than an inability to copy. **That last
clause is now measured rather than inferred**: on a corpus one count past the
bound, with the length confound removed, the break is still at copy 5 (§Eval 2).

**Replicated at two seeds, and the replication cut the claim in half.**
`recovery` reads +0.8068 and +0.7836, and the in-distribution per-copy curve
agrees to 0.047 bits/symbol. But **the out-of-count copies do not replicate at
all** — 1.4775 against 0.9466, 1.4980 against 0.5587 — so the *location* of the
failure is a property of the model and its *shape and depth* are properties of
the seed. Everything this file says about copies past the trained bound is
therefore about where the break is, and nothing about how deep it goes.

Section references of the form §N are to `PLAN.md`.

---

### Seed 1 landed 2026-08-10 — the replication, and it splits the curve in two

`synthetic_c2flat24000_byte_square_s1`, 9.0 minutes, identical config and corpus
fingerprint to seed 0, `--seed 1` and nothing else. Both converged: tails −0.225
and −0.163 bits/1k against a 0.5 tolerance.

**The replicate floor on this corpus is now measured rather than assumed:
paired s1 − s0 = −0.627 ±0.200 bits/drawing**, on 164.62 against 164.00.
Consistent with the ~1 bit §10 quotes for Tier A, and the first time claim 2's
own corpus has had one.

**In-distribution, claim 2's headline replicates.**

| | seed 0 | seed 1 | Δ |
|---|---|---|---|
| `recovery` | +0.8068 | +0.7836 | 0.023 |
| copy 1 (reference) | 1.8165 | 1.7948 | 0.022 |
| copy 2 | 0.4695 | 0.4983 | 0.029 |
| copy 3 | 0.2098 | 0.2572 | 0.047 |
| copy 4 | 0.2707 | 0.3147 | 0.044 |

Same number, same curve, same shape — the 74% collapse at copy 2, the minimum at
copy 3, the slight uptick at copy 4. **`recovery = +0.807` is no longer one
seed**, and the reading built on the curve's shape in §Run 2 — that copy 2 is
where the translation vector gets inferred — survives replication.

**Out of the trained counts, nothing replicates, and that is the finding.**

| copy | regime | seed 0 | seed 1 | Δ |
|---|---|---|---|---|
| 2 | in-count | 0.5112 | 0.5113 | **0.000** |
| 3 | in-count | 0.2692 | 0.3019 | 0.033 |
| 4 | in-count, at the bound | 0.3527 | 0.3554 | 0.003 |
| **5** | **out-of-count** | **1.4775** | **0.9466** | **0.531** |
| **6** | **out-of-count** | **1.4980** | **0.5587** | **0.939** |
| | `recovery` | +0.5862 | +0.7009 | 0.115 |

**The in-count copies agree to 0.033 and the out-of-count copies disagree by 0.53
and 0.94 — 16× and 28× the largest in-count difference, on the same two runs, the
same corpus and the same evaluation.** The copy mechanism is a property of the
model. The count prior is a property of the seed.

> **RETRACTED, one seed old — "the pure count effect is a plateau: the copy
> discount collapses from ~80% to ~18% at the bound and stops."** That was seed
> 0's shape (1.4775, 1.4980, flat). Seed 1 peaks at copy 5 and then *falls* by
> 41% at copy 6 (0.9466, 0.5587), which is not a plateau and not a monotone
> decline either. **No shape for the out-of-count region is supported at k = 2.**
> Written from one seed on the day a second was already owed, which is the
> mistake §10 exists to prevent, made in the file that quotes §10.

**What survives, and it is the part that mattered.**

- **The break at copy 5 replicates.** Both seeds jump at exactly the trained
  bound: 0.3527 → 1.4775 (4.2×) and 0.3554 → 0.9466 (2.7×). The *location* of the
  failure is a property of the model even though its depth is not.
- **Neither seed inverts.** Both out-of-count copies stay below their corpus's
  own fresh-body rate (1.7951 / 1.7785), so "anti-recovery needs the length
  confound" (§Eval 2) holds at k = 2 and is not what is being retracted.
- **The scalar's instability tracks the regime, not the metric.** `recovery` moves
  0.023 in-distribution and 0.115 on the extrapolation corpus — 5× — because the
  extrapolation corpus is 42% out-of-count bytes and those are the unstable ones.

**The rule this earns.** A curve's out-of-distribution tail has its own noise
floor, and it is not the floor of its in-distribution body. Measuring the floor
once per *metric* and reusing it across regimes is what made a one-seed shape look
quotable here. Measure it per regime.

---

### Eval 2 landed 2026-08-10 — the break is the count, with position removed

No training. One forward pass on a corpus built at `--min-repeat 5
--max-repeat 6`, against a model trained at `max_repeat=4`, which is the
separation run 3 asked for and could not make: **every repeat has 5 or 6 copies,
so copies 2–4 are in-count and 5–6 are out, and almost nothing is out of
distribution in length.**

| corpus | past 192 symbols | past the train split's own max (242) |
|---|---|---|
| `n = 5..6` | **3.0%** | **0.9%** |
| `n = 8..16` (run 3) | 30.2% | 14.9% |

*(Run 3 called 192 "the trained maximum length"; 192 is the **val** split's max
and the train split's is 242. Both thresholds are given because the reading does
not depend on which one is meant — `n = 5..6` is ~10× cleaner either way.)*

`recovery = +0.5862`, ceiling 42.26% of bytes foldable, 410/1000 programs.

| copy | regime | bits/symbol | vs the fresh-body rate 1.7951 |
|---|---|---|---|
| 1 | reference | 1.7951 | — |
| 2 | in-count | 0.5112 | 28% |
| 3 | in-count | 0.2692 | 15% |
| 4 | in-count, at the bound | 0.3527 | 20% |
| **5** | **out-of-count** | **1.4775** | **82%** |
| 6 | out-of-count | 1.4980 | 83% |

**The break is the count. It is not position, and it is not the corpus.** Copy 5
costs **4.2×** copy 4 on a corpus where 0.9% of programs are longer than anything
the model trained on. Run 3 reported the same break at the same ordinal and could
not exclude length; this excludes it.

**The copy mechanism survives being embedded in a longer repeat.** Copies 2–4
read 0.5112 / 0.2692 / 0.3527 here against 0.4695 / 0.2098 / 0.2707
in-distribution — the same shape, 15–30% more expensive. The two corpora's
first-copy rates are 1.7951 and 1.8165, within 1.2%, which is what makes that
cross-corpus comparison readable at all. So appending a 5th and 6th copy does not
damage the copies before it: the model is not thrown by a long repeat, it stops
at a number.

**And the "actively surprised" region of run 3 was at least partly length.**
There, copies 6–12 ran 2.26 → 3.35 against that corpus's own fresh-body rate of
2.2436 — *above* it, i.e. a repeated body cost more than a new one. Here copies 5
and 6 sit at 82% and 83%, **below** the fresh-body rate, and seed 1 puts them at
53% and 31% — lower still. **Anti-recovery needs the length confound**, and that
holds at both seeds.

> **The shape of this region does not.** This entry originally read the two
> numbers above as a *plateau* — "the copy discount collapses to ~18% at the
> bound and stops". Seed 1 peaks and then falls (§Seed 1), so the plateau is
> retracted and only the two-seed statements survive: the break is at copy 5, and
> neither seed inverts.

**Three scalars, three different denominators, which is the case for never
quoting `recovery` alone.** `+0.8068` at `n ≤ 4` (ceiling 23.38% foldable),
`+0.5862` at `n = 5..6` (42.26%), `−0.0502` at `n = 8..16` (66.72%). The middle
one is not "half as good as in-distribution" — it is a corpus where 42% of the
bytes are redundant and the model captured 59% of that while paying full price on
every copy past the fourth.

> **The in-distribution report is back on disk** (`runs/recovery_..._indist.json`),
> re-run in seconds, and it reproduces run 2's pasted-stdout numbers exactly:
> 1.8165 / 0.4695 / 0.2098 / 0.2707, `recovery = +0.8068`, 435/1000. The path
> collision lost the artifact and did not corrupt the record.

---

### Run 2 landed 2026-08-07 — claim 2 has a number, and the run converged

**`recovery = +0.807` at 24,000 steps on the flat Tier A corpus, and both
convergence guards pass**: `drift +0.019`, `tail −0.137` bits/1k against a 0.5
tolerance, validity 1.000, `halted` 1.000, zero truncation. That is the first
claim-2 measurement in the project, and the first Tier-C-adjacent run of any kind
that is entitled to be called a cost rather than a rate.

| copy | bits/symbol | bodies |
|---|---|---|
| 1 (the reference) | 1.8165 | 524 |
| 2 | **0.4695** | 524 |
| 3 | 0.2098 | 346 |
| 4 | 0.2707 | 165 |

**In the corpus's own units**: the redundant copies would cost **28.20
bits/drawing** at the first-copy rate and actually cost **5.45**, so the model
has captured **22.75 bits/drawing** of a corpus that totals 164.62 — repeat
structure was **17.1%** of the whole corpus and 81% of it is gone.

**The residual is nearly all real.** The repeat count is uniform on {2,3,4}
(178/181/165 in val), so the count alone carries 1.584 bits × 524 repeats =
**0.83 bits/drawing that no model can avoid**. Against 5.45 spent, the remaining
headroom is ~4.6 bits/drawing, not 5.45.

**The shape of the curve is the evidence that this is discovery and not
memorisation.** Cost collapses 74% at copy 2 and a further 55% at copy 3. Copy 2
is where the model pays to *infer the translation vector* — after one copy the
offset is unknown, after two it is determined — and a model that had merely
learned "squares are cheap" has no reason to price copy 2 at twice copy 3. The
metric already controls for content by scoring each body against **its own** first
copy, so the curve is varying position and nothing else.

Two caveats that stay attached: **one seed**, so this is a first number and not a
replicated one; and Tier A's repeats are axis-aligned squares with a fixed body
(§8), so this measures the copy relation under the easiest available conditions.

> **Bug found and fixed: `scripts/recovery.py` overwrote its own reports.** The
> output path was keyed on the run name alone, so run 3's extrapolation report
> destroyed run 2's in-distribution one — `PLAN.md` §10's "records are keyed by
> tag and the tag does not carry the budget", reproduced on the first day the
> script existed. The path now carries the evaluation corpus (`_indist`,
> `_n8-16`). **The length-generalisation reading is a comparison between those
> two reports, so the collision did not merely lose a file, it lost the result.**
> Run 2's numbers above were recovered from the pasted stdout; re-run the
> in-distribution report (seconds, no training) to put the artifact back on disk.

### Run 3 landed 2026-08-07 — it learned a table, and the break is at the count

Train `n ≤ 4`, evaluate `n = 8..16`. **The scalar reads `recovery = −0.0502` and
the scalar is actively misleading**: it says the model recovered nothing, when it
in fact recovers 47% on copies 2–4 and then *anti*-recovers past that. The curve
is the result and a single number could not have shown it — which is the case
`recovery_by_copy` was built for, decided on its first run.

| copies | regime | bits/symbol | reading |
|---|---|---|---|
| 1 | reference | 2.2436 | — |
| **2–4** | in-count, in-position | **1.00–1.18** | the copy mechanism works, ~50% off |
| **5** | **out-of-count, in-position** | **2.1491** | **the benefit vanishes at exactly the trained bound** |
| 6–12 | out-of-count, mostly in-position | 2.26 → 3.35 | actively surprised; the count prior is hard |
| 13–16 | out-of-count, **out-of-position** | 3.64 → **5.95** | position compounds it, 2.7× a fresh body |

**The break is not positional, and that is the finding.** Copy 5 costs 2.15
against copy 4's 1.18 — an 82% jump — while only **16.7%** of copy-5 spans sit past
the trained maximum length of 192 symbols. The model is at positions it saw
constantly during training and has still lost the copy relation. What it learned
is *"repeats have 2 to 4 copies"*, as a prior over counts, and each further copy
is more improbable under it. Position only becomes a factor at copy 13, where
100% of spans are past 192 and the curve steepens from +0.25 to +1.15 per copy.

> **This amends why RoPE is in the design.** §2 and `dm/models/transformer.py`
> justify RoPE by "the length-generalisation test needs positions never seen in
> training, and learned position tables cannot extrapolate there at all". True,
> and it was necessary to *run* this test — but the test's decisive region is at
> **seen** positions, so its answer is not about positional encoding at all. RoPE
> is exonerated for the primary failure and implicated only in the tail. Keep
> RoPE; correct the reason it is being credited.

> **The corpus confounds the two failures, and a cheaper one does not.**
> `n = 8..16` has mean 125 and max 753 symbols against a trained max of 192, so
> 30.2% of its programs are out-of-distribution in length as well as in count.
> **`--min-repeat 5 --max-repeat 6` puts the count one past the bound with 3.0%
> of programs past the trained length**, isolating count from position for the
> price of one eval and no training. Run it before the write-up quotes any of
> the copies past 12.
>
> **DONE 2026-08-10 (§Eval 2), and the prediction held to the decimal — 3.0%
> against 30.2%.** The break is still at copy 5 at both seeds, so it is the
> count. The one thing it amends is *this* run's tail: copies past 6 here are
> above their own fresh-body rate and the isolated count effect is not at either
> seed, so the "actively surprised" reading below belongs at least partly to
> length. **Do not quote the copies past 12 as a count effect** — and note that
> this run is one seed, on the region §Seed 1 shows is the unstable one.

**What this does to claim 2.** The claim is "a model trained on flat traces
should recover `REPEAT`". It does, inside the counts it was trained on, and by a
lot — 22.75 of 28.20 bits/drawing (§Run 2). It has **not** learned `REPEAT` as a
rule, and the failure is a hard prior over the count rather than an inability to
copy. That is a sharper and more useful negative than "it did not generalise",
because it names the thing to fix: vary `max_repeat` in training, or supervise
the count separately, and the copy mechanism is already there to build on.

- **Run 1 is falsifiable and cheap.** If `byte_delta` does not beat `byte` on
  Tier C, the relative view does not buy what §2 assumed, and the honest move is
  to say so — the axis stays in the codebase as a measured null, exactly like
  fusion on every L0-only corpus.
- **Run 2 is claim 2's first number and it needs the ceiling beside it.**
  `recovery` alone is a fraction with no denominator; `scripts/recovery.py`
  prints both and refuses to be read without the ceiling.
- **Run 3 costs nothing and decides the most.** Train `n ≤ 4`, evaluate
  `n = 8..16`: a model that learnt the *rule* is flat in the copy ordinal and one
  that learnt a table of short repeats turns up near the trained bound. This is
  what RoPE was chosen for, and the per-copy curve is what can see it.
- **Run 4 was the one with real upside and the one most likely to be negative,
  and it was negative.** Its premise is supported (§9.6: local +0.290 against
  global +0.004) and the premise was never the claim. **The premise survives
  the result**: nothing in run 4 contradicts local/global scale separation, and
  what failed was this particular factorisation's ability to exploit it at
  825k parameters. Do not read +45.22 as evidence against §5.

