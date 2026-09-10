# The run that produced no record

`quickdraw_plannerar12000eps2s2_byte_balanced_s0`, 2026-08-10. Launched to
measure the planner's replicate floor at `PLANNER_SCHEMA` 2 — the one item the
seventh instrument fault left owed. It reached step **11,500 of 12,000** across
2.6 hours, took a `SIGKILL`, and **wrote no file at all**.

This is the eighth entry in the project's failure ledger and the first that is
not an instrument fault. The instrument was correct; there was nowhere for its
reading to go.

```bash
python3 -m dm.train_planner --data quickdraw --categories cat dog bus car tree \
    --codec byte --comp-objective ar --steps 12000 \
    --tag quickdraw_plannerar12000eps2s2_byte_balanced_s0
```

---

## Why nothing survived

`dm/train_planner.py:train` assembled its record after the loop and wrote it
once:

```python
    result = { ... }
    (RUNS / f"{name}.json").write_text(json.dumps(result, indent=2))
    torch.save({...}, RUNS / f"{name}.pt")
```

So the record's existence was a *proof of completion* — which is why
`scripts/sweep.py` could safely read `config["steps"]` as the achieved budget,
and why `PLAN.md` said "the record is written at exit, so nothing reads it until
then". Every one of those properties was load-bearing and every one of them was
a consequence of the same decision. A kill at 95.8% of the run was worth exactly
as much as a kill at 0%.

**Fixed.** `dm.train.checkpoint` publishes the record and the weights at every
eval, atomically (staged in `runs/`, then `os.replace`), weights first so the
record is the commit marker. Both trainers carry `complete` and `steps_done`,
and `scripts/sweep.py` keys its ladders on the steps a run *reached*. Four tests
cover it, two per trainer.

## What killed it is not established

Nothing in `~/Library/Logs/DiagnosticReports` names the process — no jetsam
report, no crash report — so `SIGKILL` from memory pressure is a hypothesis and
not a finding. What *is* measured is that the run was competing for the machine
throughout, which the wall clock shows plainly. Per 500-step block, against its
schema-1 twin at the identical config:

| step | schema-1 | schema-2 | ratio |
|---|---|---|---|
| 500 | 251 s | 319 s | 1.27 |
| 3,000 | 319 s | 366 s | 1.15 |
| 6,000 | 229 s | 405 s | 1.77 |
| 9,000 | 233 s | 454 s | 1.95 |
| 11,500 | 230 s | 531 s | 2.31 |

The schema-1 run settles at ~229 s/block after step 3,000 and holds it for the
remaining 9,000 steps. The schema-2 run never reaches that floor and degrades
monotonically to 2.3× it. QEMU conformance, a sweep summary and figure
regeneration all ran on the same machine inside that window, which is
`docs/traps.md`'s "a long-lived sweep can wedge in Metal" seen from the other
side: it was not the sweep that wedged, it was everything else.

> **Wall clock is not arithmetic.** Contention changed how long the run took and
> could not change a single number it computed. The salvage below is unaffected
> by any of this.

---

## The salvage

The console transcript is a faithful copy of `record` at each eval — the printed
fields are read straight out of it — so the trajectory is real data with a
truncated provenance. It is **not** a record: there is no `val_bits`, so no
paired interval, and no `.pt`, so nothing to resample. It is transcribed verbatim
at the bottom of this file and cited from `docs/claim3.md` as an argument.

Differenced against `runs/quickdraw_plannerar12000eps2_byte_balanced_s0.json`,
its schema-1 twin — same corpus, same 825,080 parameters, same budget, same
seed, differing only in the RNG fix and therefore only in trajectory:

| steps 7,000–11,500 (n=10) | mean Δ | sd | range |
|---|---|---|---|
| **total bits/drawing** | **+4.69** | 0.23 | [+4.30, +5.22] |
| composition (layout) | **−0.20** | 0.07 | [−0.29, −0.05] |
| stroke (shape) | **+4.91** | 0.23 | [+4.55, +5.45] |

**The shape of the retracted result replicates and its magnitude roughly
doubles.** `docs/claim3.md` §7 observed that the planner's spread "lives entirely
in the stroke decoder" and put it at ~2.3 bits, then retracted the label because
none of the three runs behind it was same-config. A genuinely same-config pair
says the same thing about *where* the noise is — the composition level agrees to
0.20 bits while the stroke decoder disagrees by 4.91 — and puts the number at
**~4.7 bits, not ~2.3**.

Both levels are trained by one optimiser over disjoint parameters on the *same*
batches, so they absorb an identical trajectory perturbation. In relative terms
the composition level moves 0.11% of its 181 bits and the stroke decoder 1.16%
of its 421: the coarse level is close to saturated on a 32×6 grid over a small
alphabet, and the fine one is not.

### What this is worth, stated exactly

- It is **one pair**, so it is a point estimate of a floor with no spread of its
  own.
- It is **cross-schema by design** — that is what makes it controlled, since the
  schema-1 run's batch order was perturbed by its own evals and this one's was
  not. The cost is that a systematic effect of the fix and pure trajectory noise
  cannot be separated at n = 1. Both are trajectory terms; only one of them
  shrinks with seeds.
- It has **no per-program interval**, which is the form every claim-3 difference
  is quoted in.
- The schema-2 run was still descending (−0.3 bits over its last 500 steps)
  where the schema-1 run had flattened (0.0), so at a matched 12,000 the gap
  would read slightly under +4.7.

**No claim-3 verdict moves.** The likelihood gap is 38.75–48.97 bits, so a
4.7-bit floor leaves it 8–10× resolvable rather than 17–21×. What does move is
the small stuff: the +1.24 / +1.15 stroke gap between the two composition
objectives is a quarter of the stroke half's own floor and is now **unresolvable**
rather than merely not worth a run.

---

## The transcript, verbatim

```
[quickdraw_plannerar12000eps2s2_byte_balanced_s0] params=825,080 (composition 424,240 + stroke 400,840) vocab=258 strokes(mean=6.3 p99=17 over_cap=0.001) steps=12000 device=mps
  step    500  loss 3.0237  bits/drawing    748.3 (layout 193.5 + shape 554.8)  gen_valid 1.000  strokes 6.4 planned/6.4 drawn (val 6.3)  len EMD 15.1  319s
  step   1000  loss 2.7985  bits/drawing    685.4 (layout 189.8 + shape 495.6)  gen_valid 0.992  strokes 6.7 planned/6.7 drawn (val 6.3)  len EMD 15.6  591s
  step   1500  loss 2.7229  bits/drawing    662.0 (layout 187.8 + shape 474.2)  gen_valid 1.000  strokes 6.3 planned/6.3 drawn (val 6.3)  len EMD 6.5  908s
  step   2000  loss 2.6812  bits/drawing    652.8 (layout 186.9 + shape 466.0)  gen_valid 1.000  strokes 6.1 planned/6.1 drawn (val 6.3)  len EMD 8.7  1236s
  step   2500  loss 2.6206  bits/drawing    646.7 (layout 186.2 + shape 460.5)  gen_valid 1.000  strokes 6.1 planned/6.1 drawn (val 6.3)  len EMD 8.4  1571s
  step   3000  loss 2.6141  bits/drawing    641.9 (layout 185.4 + shape 456.5)  gen_valid 0.992  strokes 5.9 planned/5.9 drawn (val 6.3)  len EMD 5.2  1937s
  step   3500  loss 2.5864  bits/drawing    635.4 (layout 184.7 + shape 450.7)  gen_valid 1.000  strokes 6.2 planned/6.2 drawn (val 6.3)  len EMD 4.8  2324s
  step   4000  loss 2.6073  bits/drawing    633.8 (layout 184.2 + shape 449.6)  gen_valid 1.000  strokes 6.4 planned/6.4 drawn (val 6.3)  len EMD 7.3  2716s
  step   4500  loss 2.6565  bits/drawing    629.0 (layout 183.9 + shape 445.1)  gen_valid 1.000  strokes 6.5 planned/6.5 drawn (val 6.3)  len EMD 4.2  3098s
  step   5000  loss 2.5584  bits/drawing    627.5 (layout 183.4 + shape 444.1)  gen_valid 1.000  strokes 6.5 planned/6.5 drawn (val 6.3)  len EMD 7.6  3485s
  step   5500  loss 2.5535  bits/drawing    624.8 (layout 183.3 + shape 441.5)  gen_valid 1.000  strokes 7.2 planned/7.2 drawn (val 6.3)  len EMD 11.0  3865s
  step   6000  loss 2.5462  bits/drawing    621.4 (layout 182.9 + shape 438.5)  gen_valid 1.000  strokes 6.6 planned/6.6 drawn (val 6.3)  len EMD 11.2  4270s
  step   6500  loss 2.5353  bits/drawing    620.3 (layout 182.7 + shape 437.6)  gen_valid 1.000  strokes 7.1 planned/7.1 drawn (val 6.3)  len EMD 14.7  4674s
  step   7000  loss 2.4935  bits/drawing    617.9 (layout 182.3 + shape 435.6)  gen_valid 1.000  strokes 6.1 planned/6.1 drawn (val 6.3)  len EMD 7.1  5104s
  step   7500  loss 2.5035  bits/drawing    615.7 (layout 182.2 + shape 433.5)  gen_valid 1.000  strokes 6.2 planned/6.3 drawn (val 6.3)  len EMD 6.2  5514s
  step   8000  loss 2.5564  bits/drawing    613.9 (layout 182.1 + shape 431.9)  gen_valid 1.000  strokes 7.0 planned/7.0 drawn (val 6.3)  len EMD 10.3  5904s
  step   8500  loss 2.5083  bits/drawing    612.4 (layout 181.7 + shape 430.7)  gen_valid 1.000  strokes 6.6 planned/6.6 drawn (val 6.3)  len EMD 8.0  6340s
  step   9000  loss 2.5122  bits/drawing    610.8 (layout 181.5 + shape 429.4)  gen_valid 1.000  strokes 6.1 planned/6.2 drawn (val 6.3)  len EMD 5.0  6794s
  step   9500  loss 2.5066  bits/drawing    609.6 (layout 181.5 + shape 428.1)  gen_valid 1.000  strokes 7.2 planned/7.2 drawn (val 6.3)  len EMD 9.2  7287s
  step  10000  loss 2.4996  bits/drawing    608.5 (layout 181.4 + shape 427.1)  gen_valid 1.000  strokes 6.6 planned/6.6 drawn (val 6.3)  len EMD 4.7  7765s
  step  10500  loss 2.4798  bits/drawing    607.7 (layout 181.3 + shape 426.5)  gen_valid 1.000  strokes 6.4 planned/6.4 drawn (val 6.3)  len EMD 7.3  8294s
  step  11000  loss 2.5303  bits/drawing    607.2 (layout 181.2 + shape 425.9)  gen_valid 1.000  strokes 6.8 planned/6.8 drawn (val 6.3)  len EMD 4.5  8765s
  step  11500  loss 2.4432  bits/drawing    606.9 (layout 181.2 + shape 425.7)  gen_valid 1.000  strokes 6.6 planned/6.6 drawn (val 6.3)  len EMD 4.0  9296s
zsh: killed     python3 -m dm.train_planner --data quickdraw --categories cat dog bus car tre
```

The generation columns are draws and are not read here at all — `gen_length_emd`
averages 6.63 ±2.13 over the last ten evals against the schema-1 run's 6.44
±1.44, which is what two samples of the same distribution look like and is
**not** a replication of anything. `scripts/resample.py` is the only thing
entitled to rank those columns, and it needs a checkpoint this run did not
leave.

## Re-running it

Unchanged except that it now survives:

```bash
python3 -m dm.train_planner --data quickdraw --categories cat dog bus car tree \
    --codec byte --comp-objective ar --steps 12000 \
    --tag quickdraw_plannerar12000eps2s2_byte_balanced_s0
```

Run it with nothing else on the machine — no sweep, no `scripts/plots.py`, no
QEMU conformance pass. At ~229 s per 500 steps that is **1.6 hours**; at the
contended rate it was 2.6 and climbing. `python3 scripts/status.py` reads live
progress now, because the record on disk is current as of the last eval.

---

## Sequel, 2026-08-11 — the same kill, and this time it cost nothing

**The re-run was killed at step 11,500 of 12,000, exactly where the first one
died.** That is the sequel this file wanted: a defect closed by writing code was
then closed by the same failure happening again, which is the only test that
counts.

What came back, against what the first kill destroyed:

| | 2026-08-10 | 2026-08-11 |
|---|---|---|
| record on disk | **none** | written at every eval, `steps_done = 11500`, `complete = false` |
| weights | **none** | 3.2 MB, resampleable |
| per-program `val_bits` | **none** | 1,000 of them, so the difference is *paired* |
| what the floor was worth | an unpaired transcript reading | **4.16 bits, 95% CI [3.67, 4.64]** |

**And the transcript's number was wrong in the direction an unpaired reading of
correlated evals is wrong.** It said +4.69 ±0.23; the paired measurement says
+4.16 [3.67, 4.64], and **4.69 sits outside that interval**. The *shape* it
established held exactly — Δ composition −0.04 against Δ stroke +4.20 — so the
salvage was right about where the floor lives and biased high about its size.

**The 500 steps it never ran are worth 0.07 bits.** At the matched step 11,500
the two runs differ by +4.09 against the +4.16 read at final-versus-final, so no
further run is owed: 12,000 steps would refine a 4.16 ±0.49 number by less than
a tenth of a bit.

### Why it died, which is a different failure from the first one

The first kill had no diagnosis beyond "the box was shared". This one has a
trace, because `elapsed_s` is written at every eval:

| | first eval | last eval | trend | total |
|---|---|---|---|---|
| schema 1, completed 12,000 | 251 s | 232 s | **−2.4 s/eval** | 5,864 s |
| schema 2, killed at 11,500 | 283 s | **497 s** | **+9.3 s/eval** | 9,047 s for fewer steps |

**A completed run holds a flat wall clock; this one degraded monotonically to
2.15× the idle rate and was then `SIGKILL`ed by the OS.** A test suite, a
resample and a corpus build were running beside it throughout. That is
`docs/traps.md`'s "train on an otherwise idle machine", now with a second
instance and a worse outcome than slowness — **contention does not merely cost
wall clock, it ends runs** — and a rising elapsed trace is the evidence that a
run did not have the machine.
