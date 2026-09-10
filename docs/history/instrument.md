# Instrument history — how the measurement was rebuilt

Moved out of `PLAN.md` on 2026-08-05 during a compaction pass. This is the
evidence behind rules the plan still states; the plan keeps the rules.

Read it when you want to know *why* a guard exists before changing it. Four
sweeps were run (schemas 1–4 plus the converged control) and each one found the
instrument wrong one level above where the previous one had fixed it:

| generation | fault | fixed by |
|---|---|---|
| schema 1 | 45 GiB MPS OOM at cell 11/32 | `LayerCache`, bucketing, `micro_batches` |
| schema 2 | halting decided by symbol match; every seed on a different val set | `HaltMonitor`, independent `data_seed` |
| schema 3 | paired CI omitted seed variance; every arm past its val optimum | between-seed term, `n_train` 20k → 100k |
| schema 4 | `drift ≈ 0` is a one-sided gate; every arm short of its optimum | `tail` + the convergence guard |
| converged | `tail` measured one eval interval, which is noise and regime-dependent | `TAIL_WINDOW`, `budget_table` |

---

## Sweep infrastructure rebuild (2026-08-04)

The first full sweep **crashed at cell 11/32** with MPS OOM at 45 GiB. Causes,
all measured, all fixed — do not reintroduce these:

1. `Attention.forward` grew the KV cache with `torch.cat` per decode step. An
   800-step decode over 8 layers requests ~6400 *distinct* allocation sizes;
   caching allocators keep a block per size and never reuse across sizes.
   `generate()` alone reserved **40.02 GiB for a 3.0 MiB model**. Fixed by
   `LayerCache` (preallocated buffers, in-place writes, returns views):
   **40.02 GiB → 1.31 GiB.**
2. Uniformly random batches padded to **2.31× the mean length** on every codec.
   Attention is quadratic, so the bit arm trained at T≈780 instead of T≈340.
   Fixed by `BucketedBatchSampler`: **2.31× → 1.08×.**
3. A fixed row count is the wrong memory knob when attention is quadratic.
   Fixed by `micro_batches`, splitting against a `rows × T²` budget
   (`TrainConfig.attn_budget`, default 24M ≈ 7.5 GiB). The gradient is
   unchanged — `test_accumulation_does_not_change_the_gradient` pins that.
4. One failing cell killed the whole sweep. The driver now records and skips.
5. HALT is now an aligned *sequence*, not a token, so the bit arm stops on a
   byte-boundary zero byte instead of always decoding to the length cap.

Net: the cell that OOM'd at 45 GiB now runs at **2.08 GiB**; the token arm is
1.6× faster. Full sweep ~2.3h, was 15h+ and did not finish.

**This held.** The schema-2 sweep ran 32/32 cells in 8,835 s with zero failures.
The memory and throughput work is done and is not the bottleneck any more.

---

## Measurement rebuild (2026-08-04, schema 3)

The schema-2 sweep completed, and reading it showed that the *instrument* was
wrong in three ways. Each is fixed and pinned by a test; §6.1 has the evidence.

1. **Halting was decided by matching the HALT symbol.** In any untyped alphabet
   `0x00` is both the HALT opcode and the operand value zero, and only the
   instruction boundary separates them, so the sampler cut programs
   mid-instruction and the cut was scored as the representation's fault.
   Instructions are variable-length, so the bit arm's fixed `stop_stride` could
   not fix this either — locating boundaries requires a parse.
   `dm.isa.codec.HaltMonitor` now carries per-row parse state and stops exactly
   where `VM.run` stops. `test_halt_monitor_stops_where_the_vm_stops` pins the
   invariant: truncating a stream at the monitor's verdict cannot change its
   trace.
2. **Every model seed was scored on a different val set.** `build_data` derived
   the val seed from the model seed. Val sets differ from each other by ~3.5
   bits/drawing; the codec effects under test are ~1 bit. `TrainConfig.data_seed`
   is now independent, so every arm sees identical programs and the comparison
   is **paired** — `runs/*.json` carries `val_bits`, the per-program cost, and
   `summarise` reports per-axis paired differences with a 95% interval.
3. **The Tier A defect in §8 is fixed**, which changes the distribution.

Also landed: `gen_truncated` and `gen_length_emd` (earth-mover distance in bytes
between generated and real length distributions) describe termination without
reference to a length cap; the generation cap is now `2 × p99` because at 1× a
real program's own length distribution puts ~1% past the cap; `summarise`
reports the seed count **per row**, so a partial sweep no longer claims the best
cell's count for every cell.

**This held too.** The schema-3 sweep ran 32/32 cells in 2.3 h with zero
failures, `halted` = 1.000 in every cell, and within-run intervals on a paired
difference of ±0.13–0.36 bits. Fixes 1 and 2 above are confirmed and closed.
What schema 3 then exposed is one level up — the *statistics* and the *training
budget*, not the sampler or the split. See §6.2, and §6.3 for what fixing that
in turn exposed.

Run records carry `schema` (`dm/train.py:SCHEMA`, currently 4). Bump it when a
change makes new runs incomparable with old ones; `summarise` then excludes
stale runs rather than averaging across two regimes. Superseded runs are
archived in `runs/schema1/`, `runs/schema2/` and `runs/schema3/` and are **not**
comparable to current ones. Note that `tail` and the convergence guard needed no
bump: they are derived from `history`, which every record already carried, which
is the test for whether a reporting change is a regime change.

---

# 6.1 What the schema-2 sweep showed — history, superseded by §6.3

Schema-2 data carried the §8 leak and the schema-2 sampler, so **none of its
numbers are reportable** and the table has been dropped. Two of its conclusions
were about the *instrument*, not the data, and both have since been confirmed by
schemas 3 and 4 — they are kept here because they are why the instrument was
rebuilt.

**Retracted: "typing buys termination."** The byte arm's `halted` of 0.72
against token's 1.00 was the sharpest-looking result in the sweep and it was an
artifact. Diagnostics, all against the schema-2 checkpoints:

- Teacher-forced probability of HALT at the true terminal position is the same
  for both arms — token 0.478, byte 0.466. The models are equally good at
  knowing when to stop.
- The byte model puts 0.9997 of its mass on valid opcodes at instruction
  boundaries. It knows the parse state; that is why its `wellformed` is 1.000.
- Same byte checkpoint, same prompt, cap raised 4× and `top_k` removed:
  `halted` stays 0.66–0.71. Not a cap problem, not a truncation-sampling problem.
- Same byte checkpoint with early stopping **off**, letting the VM find HALT:
  `halted` = **0.980**. With the parse-state monitor: **0.984**, and the token
  arm is unchanged at 0.992, because its HALT token was never ambiguous.

So the gap was the sampler, and it was codec-specific in exactly the way that
made it look like a finding about typing.

**Retracted: the seed spread.** Per-cell s0/s1 spread on bits/drawing is 1.6–3.7
bits and **s1 > s0 in 16 of 16 cells** — 2⁻¹⁶ under any model-noise account.
Scoring a single fixed model against four different val seeds gives 232.4 /
235.9 / 230.3 / 237.5 bits, a ±3.5 spread. The reported error bars were val-set
variance. The typing axis (0.5 bits) and the fusion axis (0.1 bits) are
therefore **unresolved**, not null, and cannot be called either way until the
paired re-run lands.

**Survives in direction, not in magnitude: granularity costs bits, depth buys
them back.** Schemas 3 and 4 both reproduce this and make it much larger;
see §6.3.

**Superseded, three times over: "most of the token-matched bit gap is
optimisation."** Schema 2 put the representation cost at ~4 bits, schema 3 at
~1.8, schema 4 at +3.2 step-matched. Every one of those was measured on arms
that had not converged (§6.3), so no number belongs here until §9.3 runs — which
is the whole reason that regime exists.

---

# 6.2 What the schema-3 sweep showed — history, superseded by §6.3

Schema 3's numbers are superseded on every axis and the tables have been
dropped; `runs/schema3/` holds the records. Three things it established are why
the current code looks the way it does, and they are kept:

1. **Both schema-2 instrument fixes work.** `halted` was 1.000 in all 32 cells,
   so the parse-state `HaltMonitor` closed the schema-2 artifact exactly as §6.1
   predicted. Pairing on a fixed `data_seed` cut the within-run interval on a
   difference from ±3.5 bits to ±0.13–0.36. Neither needs revisiting; schema 4
   reproduces both.
2. **Fault — the paired CI omitted seed variance and was 5–15× too narrow.**
   `paired_table` combined the two runs' *within-run* intervals in quadrature,
   which is the interval of the mean assuming both seeds estimate the same
   quantity and program sampling is the only noise. Measured per-arm seed sd was
   ~0.57 bits on the non-bit arms and 7–22 on the token-matched bit arm. Fixed
   (§9.1) and pinned by `tests/test_sweep.py`.
3. **Fault — every token-matched arm trained past its own val optimum.** Best
   val landed at step 4404–5872 of 8812 and final−best drift was 0.2–1.2 bits,
   the size of the effects under test. The direct proof needed no history:
   token/square was 163.6 at 3000 steps and 164.7 at 8812, so 3× the compute was
   1.1 bits *worse*. Cause was 27.7 epochs on a 20k split. Fixed (§9.2) by
   `n_train = 100_000`, and schema 4 confirms it: **drift is +0.00 in all 32
   cells.**

---

## The diagnosis: `drift ≈ 0` is a one-sided gate and it passed vacuously

§7 said to read `drift` first and that a nonzero value meant the split was still
too small. Drift is **+0.00 in all 32 cells** and `best@` is the final eval in
all 32, so on its own terms the gate passed and the schema-3 overfit is closed.

It is the wrong test on its own. `drift` only detects a run reported *past* its
minimum. Every schema-4 run is reported *short* of it, and the shortfall is not
remotely comparable between the arms being differenced:

| arm class | bits/drawing still being shed per 1,000 steps at the final eval |
|---|---|
| token-matched, 43-token arms (token, byte, token_typed) | −0.02 … −0.29 |
| step-matched, all four codecs at 3000 steps | −0.55 … −1.57 |
| **token-matched bit arm** | **−6.4 … −14.8** |

So the token-matched granularity comparison differences an arm that has stopped
moving against one falling **50–500× faster**. `+42.52 ±37.99` is the rate the
two were separating at when the sweep stopped. This is schema 3's fault 2 one
level up: then, one arm post-optimum and one pre-; now, both pre-optimum by two
orders of magnitude of difference.

**Fixed in the instrument.** `scripts/sweep.py` now computes `tail` from each
record's `history` (no schema bump — the data was always there), prints it beside
`drift`, and `paired_table` marks any row whose arms are not both settled
`(unconverged)` *before* looking at the interval and overriding it. That marking
is a precondition, not a statistic: more seeds cannot remove it, only more steps.
Under the guard, exactly three rows in the whole table survive as results.

---

# 8. Fixed — the Tier A defect (2026-08-04)

Tier A leaked train into val and `random_grid` was near-deterministic. Recorded
here because the fix changed the data distribution, which is what SCHEMA 3 marks.

| | before | after |
|---|---|---|
| train unique | 18,878 / 20,000 | 20,000 / 20,000 |
| val also in train | 70 / 1,000 (7.0%) | 0 / 1,000 |
| `random_grid` distinct, fixed box | 3 / 5,000 draws | 5,000 / 5,000 |

`random_grid` derived every field from `count ∈ {2,3,4}`, so for a fixed box it
had exactly three possible outputs; single-motif samples that picked the grid
collapsed onto those three, which is where both the duplicates and the leak came
from. Step, cell size and origin are now drawn independently of `count`.

`synthetic.split()` is the entry point that gives a deduplicated train set and a
val set filtered against it — a different seed alone does not make splits
disjoint, it only makes the collisions harder to notice. `dataset(unique=True)`
raises `Exhausted` rather than silently returning a short list, because
shrinking a split changes the denominator of every metric computed from it.

**One design decision was deliberately not taken.** The repeat body is still a
square. Varying the stroke shape inside `REPEAT` would add entropy but would
also change what "recover `REPEAT`" means for claim 2, so it is left alone; the
three fields that were *derived from `count`* were a defect, the body shape is a
design choice. Revisit it when claim 2 starts, not before.

`dx`/`dy` are now capped at `DELTA_MAX = 127` in the generator rather than
staying in range as a side effect of the canvas size, and
`test_repeat_delta_stays_inside_the_i8_operand` pins it over 2,000 random boxes.
That closes the trap §10 used to list as unpinned.


---

# The three instrument steps these produced (was PLAN.md §9.1–9.2a)

1. **DONE — the paired CI now contains seed variance** (`scripts/sweep.py:paired_table`).
   The old line was
   `ci = (sum(d["ci95"] ** 2 for d in deltas) ** 0.5) / k`, whose comment —
   "seeds are independent runs, so their intervals combine in quadrature" — was
   the error: quadrature of within-run intervals estimates the SEM of the mean
   *assuming the runs differ only by program sampling*. §6.2 shows they differ by
   5–15× that. The table now computes both terms, reports `1.96 × max(within,
   between)`, and prints `within-run`, `between-seed` and the per-seed deltas as
   separate columns — with `k = 2` the between-seed term has one degree of
   freedom and 1.96 understates it, so the deltas themselves are the primary
   read and a sign flip across seeds means unresolved however narrow the
   interval. `tests/test_sweep.py` pins that a sign-flipping axis is marked
   `(indistinguishable)` while a consistent effect still resolves.


2. **DONE — the overfitting fix: `n_train` 20,000 → 100,000, `SCHEMA` 3 → 4.**
   Token-matched training was 27.7 epochs; at 100k it is 5.5 and the val minimum
   moves outside the budget. `synthetic.split(100_000, 1_000)` returns 100k
   unique, val-disjoint programs in ~3.6 s, so this costs nothing but the
   re-run. `SCHEMA` is bumped because the fit regime changes, not the
   distribution: `summarise` excludes schema-3 rather than averaging across two
   regimes. `dm/train.py` also keeps the argmin eval's per-program bits
   (`best`, `best_val_bits`) next to the final one, and `summarise` prints
   `drift` and `best@` from them, so the overfit regime is visible in the table
   instead of needing a per-record post-mortem.

   Measured on the 100k split before launch: it builds in 1.6 s, peak RSS is
   0.6 GB on the bit arm, and the token budget still derives the same step
   counts (8,844 token-matched, 1,128 bit). Epochs go 27.7 → **5.7** on the
   43-token arms and 3.6 → **0.72** on the bit arm.

   **Confirmed by the sweep: `drift` is +0.00 in all 32 cells.** The fix worked
   exactly as specified, and specifying only that side of it is what step 2a
   below is about.


2a. **DONE — the convergence guard, which is what fixing the overfit exposed.**
   `drift ≈ 0` is a one-sided gate: it detects a run reported past its minimum
   and is silent about one reported short of it. Every schema-4 cell is the
   second kind, and the token-matched bit arm is short by 50–500× more than the
   byte arm it is differenced against (§6.3). `scripts/sweep.py:tail` measures
   bits/1k steps at the final eval — from `history`, so no schema bump and no
   re-run — the summary prints it beside `drift`, and `paired_table` refuses to
   resolve any row whose arms are not both settled. `TAIL_TOLERANCE = 0.5` is a
   reporting guard sized against the axes under test, not a constant; the
   docstring says so. Params over 1M are marked `*` (schema-3 fault 3, closed).
   `tests/test_sweep.py` pins that a consistent, large, tight-interval
   difference against a still-descending arm is refused. 52 tests pass.


## Generation 5 — the platform faults, and Tier B's two instrument bugs

Not measurement faults like schemas 1–4; these are ways the *rig* lies.

**The 78-minute Metal wedge (2026-08-05).** The 24,000-step bit cell — the 4th
model built in one process — blocked for 78 minutes inside `model.to(device)`,
in `[_MTLCommandBuffer waitUntilCompleted]`, having written nothing. Diagnosis,
in the order that worked:

1. The run *header* never printed, where a fresh process reaches it in 7.9 s.
   That is the tell: not slow training, no training at all.
2. `lsof -a -p PID -d 1` showed the stdout offset equal to the file size, so the
   silence was not block buffering.
3. `sample PID 4`: 97% of samples in `__psynch_cvwait` under `mps_copy_`.
4. It coincided with 7.5 GB of 8 GB swap in use; killing the process returned
   3.9 GB.

Hence the rule the plan keeps: **one process per training run in a chain**, and
if a run is silent for longer than its first eval interval, `sample` it rather
than waiting. `scripts/status.py` exists to make that check one command.

**Tier B, fault 1: `max_len` truncated the bit arm alone (2026-08-06).**
`ProgramDataset` cut at `max_len` silently. The bit codec emits 8× the symbols
for the same drawing, so at the shipped `rdp_eps=2.0` a `max_len=2048` cuts
3.61% of the train split on the bit arm and 0.00% on every other arm. A cut
program is scored on fewer bits than it costs, so the bias lands on exactly the
granularity axis that is claim 1's headline, in the direction that flatters it.
Found by `scripts/quickdraw_check.py` before any Tier B model was trained.
Closed by counting truncation in `ProgramDataset`, reporting it from
`length_stats()` for train *and* val, and warning from `train()`. Tier A never
reached `max_len`, which is why six sweeps never saw it.

**Tier B, fault 2: the fusion axis degenerated (2026-08-06).** `token_typed`
offsets operand ids by `256 × int(Kind)`. QuickDraw L0 emits only MOVE/LINE/HALT,
whose one operand kind is `COORD` — and `Kind.COORD == 0`. So on Tier B the
typed codec is *byte-identical* to `token`: verified equal on 1000/1000 val
programs, with 1,024 of its 1,293 rows unreachable. The two runs still differed
by −1.48 ±0.49 bits/drawing, which is noise dressed as a result. Two uses:
the axis is untestable on any L0-only corpus (it needs `REPEAT`'s `COUNT`/`DELTA`
or `CIRCLE`/`WIDTH`'s `SCALAR`), and the null pair is a free measurement of the
Tier B resolution floor. Test before reading any new corpus's table:
`codec_a.encode(p) == codec_b.encode(p)` over the val split.

---

# The traps, in full

`PLAN.md` §10 is the one-line-per-rule version, kept short so it is actually
read before a change. This is the same list with the evidence attached: what was
measured, what it cost, and why the guard is shaped the way it is. **Read the
long form here before changing a guard**; the short form is for remembering that
the guard exists.

Preserved verbatim from `PLAN.md` §10 as it stood on 2026-08-08, when that file
was compacted from 2,197 lines to a resume point.


- **Never compare per-token loss across codecs.** Use bits/drawing.
- **Every geometric transform in the pipeline must be integer-affine, or exact
  repeats stop existing** — quantisation, augmentation, normalisation alike.
  Measured on Tabler's 3.88% ceiling: integer translate / mirror / 90° rotate
  preserve it *exactly*; a ×1.07 scale destroys 77% of it and ±1 jitter 65%.
  That is the DeepSVG failure reproduced on demand (0.62% against 3.17%).
  Rounding commutes with integer translation and not with anything else.
  **Validate any new policy with `scripts/repeat_oracle.py` before adopting it**,
  which takes seconds.
- **Quantise from the source grid by an *integer* scale factor.** A pipeline that
  normalises to a float bounding box and *then* rounds turns repeats into
  near-repeats — on SVG-Icons8, 0.62% compressible against 3.17%
  ([`docs/tier-c.md`](../tier-c.md)). Icon sets are authored on integer grids
  and 24 × 10 = 240 ≤ 255, so ×10 preserves every repeat by construction
  (Bootstrap's 16 grid takes ×15).
- **A `limit=` that slices a concatenated corpus makes it single-category without
  saying so.** `quickdraw.load` appended category after category, so `n_val=1000`
  over five categories drew all 1,000 from `cat` and train/val would have been
  different distributions. `_interleave` round-robins before the cache is
  written, so any prefix is balanced. It survived because the loader had no test
  file at all; it has one now.
- **Run-to-run noise is the *trajectory*, not the initialisation — common random
  numbers do not fix it.** RETRACTED as advice 2026-08-06, having been written
  here before it was tested. Sharing the non-embedding init (`--share-init`, kept
  and tested) moved typing's per-seed deltas from +0.62 / −1.16 to −1.00 / +2.01:
  signs still disagreeing, spread no smaller. SGD is chaotic, so a shared start
  does not correlate endpoints. At k=2 the honest claim is "not detectably
  better". An axis below the floor is unresolvable at this budget, and saying so
  is the result.
- **Validate a variance-reduction idea before writing it down as advice.** The
  entry above was a trap for one session because it read as settled guidance
  while being an untested hypothesis. A plan that records proposals as rules
  teaches the reader to distrust the rules.
- **Sharing no gradient is not sharing no RNG stream, so "this half is
  unaffected" is a claim about the code and not about the number.** §9.10
  designed the planner's two levels to share no gradient and concluded the
  stroke half would come out *identical* across `comp_objective`. It came out
  **414.98 against 413.80, 1.18 bits apart** at the same seed, corpus and step
  count. **An arm you believe is held fixed must be *differenced*, not
  assumed.**
  > **AMENDED 2026-08-08 — the mechanism is worse than a shifted stream, and the
  > last sentence of this entry was wrong.** `per_program_bits` calls
  > `torch.manual_seed(seed)` to pin the bound's Monte-Carlo draw, and that reset
  > is **global**: both arms restart the stream at the same point at every eval,
  > consume different amounts of it, and resume training from different states,
  > so the next `torch.randperm` draws a **different epoch**. Measured on a tiny
  > config, one eval apart: `[2, 7, 1, 0, 5, 4, 3, 6]` against
  > `[4, 0, 7, 3, 2, 5, 1, 6]`. The two arms are not one run with a jittered
  > stream — **they are two runs on two different batch orders.**
  >
  > So the difference is *not* "a same-config noise reference measured on the run
  > you are already paying for", which is what this entry said and what §7 then
  > quoted twice: once as ~1.2 bits and once as a ~2.3-bit floor used to argue
  > that more seeds were not owed. **The planner's replicate floor is
  > unmeasured.** The fix is a local `torch.Generator` for the bound's draw; it
  > changes every planner trajectory, so it is a `PLANNER_SCHEMA` bump.
  >
  > **The general form: a seeded reset inside an eval is a global side effect on
  > training.** Pinning an estimator's draw is right; pinning it by reseeding the
  > process is how the pin reaches the optimiser.
- **A tail is a guard, never a rate, and multiplying one by a step count is a
  category error worth 3×.** The planner's 12,000-step rung is the project's
  first pair where a tail can be checked against what doubling the budget
  actually bought: `tail` −1.83 predicted −22.0 bits over the next 12,000 steps
  and delivered **−8.13**; −1.75 predicted −21.0 and delivered **−5.50**. It
  overstates by 2.7–3.8× because the slope is still falling inside the window it
  is measured over — the safe direction for "is this run converged?" and the
  wrong direction for "how many steps to close X?". §7 did the second once and
  concluded a 40-bit gap was ~120,000 steps away; the across-schedule answer is
  ~7 doublings. **`budget_table` is the only thing in this project entitled to
  answer a question about rates.**

- **A relative codec gives claim 2's answer away, and `recovery` cannot tell.**
  Under `dm.isa.relative` a translational repeat is a *literally repeated symbol
  sequence* — later copies differ from the first only in the delta that steps
  between them — so a delta arm's recovery number scores copy detection, which
  is a far easier problem than the structure discovery the metric was built for.
  Claim 2's headline stays on the absolute view; a delta arm is a contrast and
  belongs beside it or nowhere. `scripts/recovery.py` prints the warning, and
  the warning is not a substitute for not doing it.
- **Confidence-ordered argmax decoding collapses to the majority class, and the
  result is well-formed.** The planner's composition sampler committed the
  arg-max of each slot; on a grid that is 86% EMPTY it drew **zero strokes on
  32/32 grids**, while the same checkpoint produced 93.75% valid programs from
  *true* summaries. Validity reported it as a model that cannot draw. Sample the
  token rather than taking its arg-max. **The general form: when a metric can be
  satisfied by producing nothing, check what the model produced before believing
  what the metric says about it.**
  > **AMENDED 2026-08-08 — this entry named half the fault and the fix it
  > prescribed was the wrong half.** It said to "order by its log-probability
  > plus annealed noise", and that ordering is itself the collapse in slower
  > motion; see the entry below. Sampling the value was necessary and it was
  > never sufficient.
- **Confidence-ordered unmasking is not the reverse process of a masked
  diffusion, and on a sparse grid it is a ratchet.** The forward kernel masks
  each position independently with probability `t`, so the masked set is
  uniformly random and the reverse step must unmask a **uniformly random**
  subset. Committing the most confident position first is MaskGIT's heuristic
  for image tokens; imported here it made the sampler a greedy search over the
  joint. On a grid that is 86% EMPTY the most confident masked position is
  almost always one the model wants to fill with EMPTY, and committing it raises
  the *correct* posterior for EMPTY at its neighbour — so every step shortens the
  plan and none lengthens it. Measured on one checkpoint, five seeds: **3.26
  strokes planned and `gen_length_emd` 25.08 against 6.00 and 4.59 under uniform
  random ordering**, on a val truth of 6.295. Three things make the diagnosis
  rather than the symptom:
  - **the model's own one-pass marginals were right all along** — 6.16 expected
    strokes against the corpus's 6.29, so nothing was wrong with training, the
    corpus, the capacity or the bound;
  - **more steps make it monotonically worse** — 6.84 strokes at 1 step, 3.19 at
    16, 3.17 at 192. An under-resolved sampler converges toward the model's
    joint. **That monotonicity is the test for whether a sampler is biased or
    merely coarse, and it costs one sweep of the step count.**
  - **noise does not fix an ordering** — un-annealed Gumbel reads 3.29 against
    annealed 3.26, because any probability-weighted order walks the same ratchet.
  **The general form: a sampler is part of the model's specification, and one
  borrowed from a paper with a different data distribution is an untested
  hypothesis.** Sparse-majority-class grids punish it hardest.
- **A generation column is a draw, and a record's final eval is one sample of
  it.** `gen_length_emd` on the flat AR baseline reads **2.72** in the record,
  **10.40 ±4.48** over five seeds on that same checkpoint, and spans **2.72 to
  42.45** across the run's own last 25 evals — a coefficient of variation of
  0.43 against `bits_per_drawing`'s ~0.005 on the same corpus. §7 drew a
  conclusion from the 2.72 and the conclusion was wrong. **Every
  resolution-floor rule in this file was written for `bits_per_drawing` and none
  of them was ever applied to the sampling columns, which need them more.**
  `scripts/resample.py` reports mean, sd, min and max over `k` seeds; quote a
  sampling column from it or not at all.
- **RoPE is relative, so anything on a fixed grid of *fields* needs absolute
  positions too.** The composition denoiser predicted the same distribution at
  all 192 of its positions — the corpus-wide EMPTY rate — because nothing told
  it that position 5 is a `halts` flag and position 0 is an x coordinate. RoPE
  was chosen for the AR arms because the length-generalisation test needs unseen
  positions (§2), and that argument does **not** transfer to a grid that never
  extrapolates. `Config.abs_pos`, off for every AR arm.
- **`-x.clamp_min(eps)` clamps `x`, not `-x`, and a NaN that reaches a `topk`
  makes the result backend-defined.** The composition sampler's Gumbel noise was
  written `-torch.log(-torch.log(u).clamp_min(1e-9))`. Unary minus binds *after*
  the method call, so the clamp landed on `log(u)` — non-positive everywhere —
  drove every element to `1e-9`, and handed the outer log a negative number.
  **NaN in every element, on every device, and nothing raised.** The NaNs
  ordered the unmasking, and ordering by NaN differs by backend: one checkpoint
  at one seed planned **9.17 strokes per grid on CPU and 3.70 on MPS** against a
  truth of 6.30, so every generation column in run 4's record described the
  backend. `gumbel_like` now clamps the *input*, and the test asserts the
  moments (0.5772, 1.2825) rather than the shape. **The general form: assert
  what a quantity is, not that the code ran** — the existing test checked the
  grid's shape and that no mask survived, and both hold under NaN.
- **A corpus key derived from the config cannot see a default.** `--rdp-eps 2.0`
  typed and omitted produce identical programs and two different keys, because
  `extra` is empty when the flag defaults; `n_train` was not in the key at all,
  so 100k and 350k programs of five categories were one corpus. Run 4 was
  compared against a baseline differing in categories, `rdp_eps` **and**
  `n_train`, and every key in the project said they matched. **The corpus is now
  digested from the programs themselves** (`dm/data/fingerprint.py`), it is in
  every record, and it is part of the key in the summary's seed aggregation, in
  `budget_table` and in `paired_table`'s cell lookup. Before this the corpus
  travelled in the *regime string* by convention (`m5b24000eps4`), and a
  convention cannot be checked.
- **A record can be half wrong, so retract columns rather than runs.** Run 4's
  likelihood was sound — 598.147 on CPU against 598.506 on MPS over 200
  programs, inside the Monte-Carlo standard error, stroke term identical to
  three decimals — and its generation columns were void. Deleting it would lose
  a good number; averaging it whole would publish a bad one. The record names
  its own bad columns and every reader honours that. **Retract the whole
  generation half, never single keys**: `halted` and `wellformed` are both read
  off `gen_faults`, so blanking that one key makes the `or {}` default report
  `halted` **1.000** — a fabricated number where the honest answer is that
  nobody knows. **And a record that retracts its own columns must never be
  re-run under its own tag.** Records are keyed by tag, so fixing the fault would
  delete the only worked example of it; run 4's re-run is
  `quickdraw_plannerdiff24000eps2_…`, deliberately not `quickdraw_planner24000_…`.
- **A column the design names as its own falsifier must actually be recorded.**
  `dm/models/planner.py` stated the claim-3 prediction as *"if `gen_length_emd`
  does not improve, the scale argument is weaker than §5 claims"*, and
  `train_planner` called `corpus_stats` alone — so that column, `gen_len_p50`
  and `gen_truncated` were absent from every planner record, while
  `scripts/sweep.py` read all three and printed `nan`. A prediction nothing
  measures is not falsifiable, however precisely it is worded.
- **A structural guarantee the sampler does not enforce is a claim, not a
  property.** The planner's design says termination is structural: the
  composition level decides how many strokes, how long each is, and which ends
  the drawing. `length` was enforced and `halts` was not — the HALT *byte* was
  left to the decoder, which omitted it on 42 of 128 grids. Note what parity
  actually required before adding a constraint: the AR arm's `HaltMonitor`
  already stops each row **at an instruction boundary**, so giving the planner
  `to_boundary` restores a rule the other arm has had for months rather than
  granting it an advantage. Validity 0.195 → 0.844–0.914, with `truncated` and
  `unknown_opcode` eliminated.
- **Cast to float64 only once off-device — MPS has no float64, and the fault
  waits for the first eval.** `dm/eval/metrics.py` had the rule in a comment;
  `per_program_bits` in `dm/models/planner.py` broke it twice (`.double().cpu()`
  for both the diffusion bound and the stroke NLL), so run 4 trained 500 steps
  and then died with `Cannot convert a MPS Tensor to float64 dtype`. **Every
  test in the suite runs on CPU, where the same code is correct**, which is why
  205 passing tests said nothing about it. `test_the_eval_path_runs_on_the_accelerator`
  now exercises the eval path on the accelerator when one exists, and skips
  rather than passes when one does not.
- **A run tag whose regime contains an underscore was silently unreadable.**
  `summarise` split the name from the left, so every Tier C record — `c2_800`,
  `ladder1000`, `aug6000` — parsed as codec `800_bit` and the whole corpus was
  invisible to `--summarise-only`; its table had only ever been read by hand.
  The codec now comes from a closed set matched against the tail, and an
  unparseable tag is **named** in the summary rather than dropped.
- **An ablation axis can silently degenerate on a new corpus, and it will not
  announce it.** `token` vs `token_typed` is real on Tier A and *byte-identical*
  on **every L0-only corpus** — QuickDraw and Tabler alike — because the typed
  codec keys on operand `Kind` and `MOVE`/`LINE`/`CURVE` are all `COORD`. Both
  corpora still produced confident-looking rows (−1.48, −2.45) that were pure
  noise. The axis needs `CIRCLE`/`WIDTH`'s `SCALAR` or `REPEAT`'s `COUNT`/`DELTA`.
  **Run `codec_a.encode(p) == codec_b.encode(p)` over the val split before
  reading any new corpus's table** — this trap was already written here and got
  read *after* the Tier C table, which is why it now names the corpora.
- **`max_len` truncates, and it truncates the bit arm first.** The bit codec
  emits 8× the symbols for the same drawing, so a cap that is generous for every
  other arm can cut only that one — 3.6% of Tier B's train split at the default
  `rdp_eps=2.0`, 0% everywhere else — and a cut program is scored on fewer bits
  than it costs, biasing the granularity axis specifically.
  `length_stats()["truncated"]` is in every record and `train()` warns: **a row
  with non-zero `truncated` is not comparable across codecs.** Choose `rdp_eps`
  and `max_len` together from the *full* corpus, never a sample.

- **A long-lived sweep process can wedge in Metal, and it looks like slow
  training.** One process per training run in a chain; if a run is silent for
  longer than its first eval interval, `sample PID 4` it rather than waiting —
  the tell is that the run *header* never printed, where a fresh process reaches
  it in 7.9 s. Cost 78 minutes once; full forensics in
  [`docs/history/instrument.md`](instrument.md) §5.

- **Never compare bits/drawing across two different val sets.** Two draws from
  one generator differ by ~3.5 bits/drawing, several times any effect under test.
  `data_seed` is independent of `seed` for this reason; prefer the paired
  difference whenever both arms scored the same split.
- **Pairing removes val-set variance, not seed variance.** Quadrature of
  within-run intervals answers "how precisely did these two runs measure their
  own difference" — 5–15× narrower than "does this generalise to another seed".
  Measured seed sd ~0.57 bits converged, 7–22 on the token-matched bit arm.
  `paired_table` reports the wider of the two and `tests/test_sweep.py` pins it;
  do not "simplify" it back to quadrature. Evidence: §6.1–6.2.
- **`drift ≈ 0` is half a convergence test; passing it proves nothing alone.**
  Drift (final − best) catches a run reported *past* its val minimum — schema 3
  failed it by 0.2–1.2 bits, the size of the axes under test. Fixing that moved
  every arm to the *other* side, where drift is structurally blind. **Read
  `drift` and `tail` together, always**; a row is an asymptote only when both are
  near zero. Tier B's floor is the live example: drift +12.36 at 24,000 steps.
- **Neither `drift` nor `tail` can establish an asymptote, because both are
  measured inside a schedule that anneals its LR to zero.** byte/square/s0 passed
  with `tail` −0.29 at 12,000 and then reported **1.7 bits lower** at 24,000.
  Only the across-schedule test binds: the same arm at two budgets, paired on the
  seed (`budget_table`). Run a rung before quoting any converged number.

- **Run-to-run spread at fixed config is the resolution floor, and it scales with
  bits/drawing.** Tier A ~1 bit on a ~160-bit corpus; Tier B `cat` **1.97** for a
  *provably identical* encoding; Tier B multi **~2.5** across three draws of one
  byte config (422.09 / 422.73 / 420.24). **No axis smaller than the floor is
  resolvable at k = 2, whatever the paired interval says** — pairing cancels the
  val set, not this. Scale the floor to the corpus before quoting any delta.

- **"It scales with program length" is not evidence that a difference is
  representational.** Regressed on length, Tier B's *null* pair — two provably
  identical encodings — gives −27 m-bits per bytecode byte, bootstrap CI
  [−47, −6], excluding zero. A better model is better at every token and long
  programs have more tokens, so any global quality difference is
  length-proportional by construction. The per-byte rate is as noise-prone as
  bits/drawing; only between-seed replication separates the two.

- **A tail measured over one eval interval is noise.** The val mean wanders
  ±0.11–0.17 bits between evals, so a 500-step interval carries a ±0.32 bits/1k
  floor against a 0.5 tolerance, and cadence differs by regime. `tail` takes a
  fixed *fraction* of the schedule. Do not "simplify" it back to one interval.
- **A difference between two arms stopped at different points on their own curves
  is a rate, not a cost, and no interval can tell you.** Tier A's token-matched
  granularity rows read +16/+42/+72 bits with both seeds agreeing, while the bit
  arm shed 6–15 bits/1k against a byte reference flat at 0.03–0.27.
  `paired_table` marks these `(unconverged)` *before* it looks at the interval,
  so **more seeds cannot clear it, only more steps can.**

- **A budget regime cannot equalise fit, only compute.** Arms whose sequence
  lengths differ 8× descend at different rates by construction, which is why
  granularity reads +42 bits in one regime and +3.2 in another and neither is
  the cost. Budget regimes answer what a fixed budget buys; only the `converged`
  regime answers what the alphabet costs (§9.3).
- **bits/drawing cannot see sample quality.** At the converged point the bit arm
  had the best likelihood, the worst length distribution and perfect validity;
  on Tier B multi it has the *worst* likelihood and still the best validity.
  Teacher-forced NLL cannot see compounding sampling error. Report both.
- **Do not detect HALT by matching a symbol.** In every untyped alphabet the
  HALT byte also occurs as an operand value; only the parse state distinguishes
  them. `HaltMonitor` walks instruction boundaries — a symbol match reported
  `halted` at 0.33 where the truth was 1.000.
- **A validity number is not a description of an arm.** `halted` and
  `wellformed` fail independently, and `no_halt` also absorbs whatever the
  length cap cuts off — read `gen_truncated` and `gen_length_emd` alongside it.
  **Correction 2026-08-13:** old flat-AR `gen_truncated` used returned tensor
  width; early-stop generation returns when its last row halts, so exactly that
  valid row was marked capped (recurring 1/n). Monitor state now supplies cap
  status. Planner cap accounting is separate and unaffected. Old flat-AR cap
  values are withdrawn; geometry, validity and length EMD are unchanged.
- **Never treat same seed across devices as a paired sampler contrast.** Published
  Figure 8 ran on `scripts/resample.py`'s CPU default; schema-2 sweep explicitly
  uses MPS. Planner composition plans differ before downstream PAD/BOS masking,
  so old/new deltas include backend. Device is now in report identity.
- **A refusal that writes nothing destroys evidence; failed survivor geometry
  must not masquerade as successful draws either.** First relaxed commands left
  no reports, so root cause was unknowable. Reruns proved exact failures: one
  empty in class `all` draw 4, one in class `T=1.2` draw 0, and one in flat-AR
  `all` draw 4; both planner `all` endpoints completed. Drivers now atomically
  persist status, guards, seed and partial draws, continue across checkpoints,
  and use strict JSON. Partial draws remain diagnostic, unavailable geometry is
  `null`, and consumers require `status="complete"` before aggregation. Geometry remains blocked regardless of how good
  earlier partial draws look.
- **Do not read a halting hazard off one teacher-forced pass unless the codec is
  stride-1.** On the bit arm `P(next byte = HALT)` is a product of eight
  conditional bit probabilities and only the first is on the teacher-forced path,
  so one pass measures the wrong quantity with the right shape (§9.5a).

- **A table that cannot see a record cannot check it, and "listed rather than
  averaged in" is not the same as reported.** `summarise` dropped every planner
  record before building `rows`, so `budget_table` and `paired_table` never saw
  one — and §7 spent a 3.2-hour pair of runs on a rung "because `budget_table`
  needs it". The planner's numbers had therefore always been differenced by
  hand and pasted into this file, which is precisely how run 4 got compared
  against a corpus differing in categories, `rdp_eps` and `n_train` at once.
  **A comparison that lives in prose has no key, and a key is the only thing
  that ever catches this class of fault.** `planner_table`,
  `planner_vs_ar_table` and an `arm_fields`-parameterised `budget_table` now
  carry it, joined on the corpus fingerprint.
- **What names an arm is part of the ladder's key, and the planner's arm is not
  its codec.** Keyed on `(codec, shape)` — the AR sweep's key — both planner
  arms fall in one cell and the ladder differences a diffusion rung against an
  AR one as if `comp_objective` were a budget. That is Tier C's augmentation
  mix-up in a new coordinate, and it is why `budget_table` takes the arm's
  fields rather than assuming them.
- **A sampler is part of a model's specification, so the record has to name
  it.** The composition unmasking order was a *default argument*, so changing it
  changed what every existing planner record's generation columns meant while
  leaving every record byte-identical — and three separate faults rode that same
  default. `PlannerTrainConfig.gen_order` is stamped into new records; the five
  older ones read `unknown` and the summary names them rather than guessing.
  **It is a config field and not a schema bump**, because the sampler has no
  gradient and no path into the bound: it makes the *generation* columns
  incomparable and leaves `bits_per_drawing` comparable, and those are different
  statements about one record.
- **Never mix run records across `SCHEMA` values.** The summary guards it; do not defeat the guard.
- **Do not launch a sweep from a wrapper shell that exits.** It gets orphaned and
  a second launch races it into the same log, overwriting `runs/*.json`. This
  happened. Verify exactly one trainer is alive with `scripts/status.py`.
- **`TrainConfig.steps` is mutated by `train()`** when `token_budget` is set, so
  never reuse a config object across runs. The saved record is correct.
- **Run a cell with `python3 -m dm.train`, never `python3 dm/train.py`** — the
  latter breaks the package-relative imports and dies at the first `from .data`.
- **Records are keyed by tag and the tag does not carry the budget.** Re-running
  an arm at a new step count overwrites the old record unless the tag says the
  budget — hence `budget24000`, `m5b24000eps4`, `c2_1000`.
- **`REFERENCE_SHAPES` is over budget on purpose** and deliberately absent from
  `SHAPES`, which the sweep iterates. Merging them puts a 5M-parameter model into
  a sub-1M comparison.
- **Do not train the composition level on QuickDraw.** It is large and free and
  contains no composition: 9 strokes per drawing against FS-COCO's 62. Level, not
  volume (§3, `docs/diffusion-strategy.md` §5.2).
  **Scoped 2026-08-08, and the scope matters.** This is about what a composition
  level can *learn* — scene-level structure QuickDraw does not contain. It is
  **not** a licence to blame QuickDraw for a composition level that fails on
  QuickDraw's own statistics: the AR-composition control plans **6.078 strokes
  against a val truth of 6.301** on exactly this corpus (§7). A level that
  undershoots 3.4 against 6.30 is not starved of composition, it is broken, and
  reaching for Tier D at that point buys a harder corpus for an untested model.
