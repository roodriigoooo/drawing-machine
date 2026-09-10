# Tier C (SVG-Icons8) — acquired, tested, and blocked on the wrong artifact

Written 2026-08-06. `PLAN.md` §3 keeps the conclusion; this keeps the evidence.

## What is actually on disk

`data/deepsvg/` is the DeepSVG repo. `dataset/download.sh` needs `wget` (absent
on macOS) and its Google Drive confirm-token flow is stale; `uvx gdown <id> -O
<file>` works. It yields two artifacts and **neither is raw SVG**:

- `icons_meta.csv` — 99,508 rows: `id, platform, total_len, nb_groups,
  len_groups, max_len_group, category, subcategory`.
- `icons_tensor.zip` — 3.57 GB, **25.4 GB unpacked**, 99,509 pickles. Each is
  `{"tensors": [20 augmentations][nb_groups paths], "fillings": [...]}` where a
  path is `(n_commands, 14) float32`: col 0 is the command (0 MOVE, 1 LINE,
  2 CURVE), cols 6–13 are start / control1 / control2 / end, cols 1–5 are arc
  parameters and always −1. Coordinates are **integers stored as floats**,
  max 253. Command mix over 400 icons: 47% LINE, 47% CURVE, 6% MOVE — all three
  map directly onto the ISA.

The other two Drive ids in the README are the **fonts** dataset, not icons.

## The acceptance test

Claim 2 needs *exact translational* repeats: `REPEAT n dx dy` is translation
only, i8 deltas, and a repeat that is off by one unit cannot be emitted
losslessly. Whole-path repeats, 400 icons:

| corpus | icons with a compressible repeat | bytes saved | L0→L1 ratio |
|---|---|---|---|
| SVG-Icons8 (DeepSVG tensors) | 3.2% | 0.62% | 1.0063 |
| QuickDraw `cat` | 1.0% | 0.01% | 1.0001 |

60× QuickDraw and still far too little to quantify structure discovery against.

## Why it fails, which is not the icons' fault

Loosening the match tolerance separates "no repeats" from "repeats destroyed":

| tolerance | icons with a repeat | bytes saved |
|---|---|---|
| exact | 3.2% | 0.62% |
| **±1 px** | **6.0%** | **3.17%** |
| ±2 px | 9.5% | 4.47% |
| ±4 px | 16.8% | 6.15% |

**One pixel of tolerance quintuples the recoverable structure.** That is
quantisation error, not absent repetition. DeepSVG preprocesses with
`normalize() → zoom(0.9) → canonicalize() → simplify_heuristic()` — arbitrary
float scaling, *then* rounding — so two originally identical elements land one
unit apart and stop being expressible as a lossless `REPEAT`.

## The rule this yields

**Quantise from the source grid by an integer scale factor, and exactness is
preserved by construction.** Icon sets are authored on integer grids (commonly
24×24). The ISA canvas is 256, and 24 × 10 = 240 ≤ 255, so a ×10 scale keeps
every authored coordinate an integer and every authored repeat exact. Any
pipeline that normalises to a float bounding box first has already lost the
property, whatever it does afterwards.

## The blocker

DeepSVG's README, on the raw icons:

> For full flexibility and more research freedom, we however recommend
> downloading the original SVG icons from icons8, for which you will need a
> **paid plan**. Instructions to download the dataset from source are coming
> soon.

Those instructions never shipped. So the raw SVG-Icons8 corpus is behind a
commercial licence and the released artifact is the one that destroys the
property claim 2 needs.

## Resolved: Tabler passes the same test (2026-08-06)

`dm/data/tabler.py` ingests the 24×24 grid at `SCALE = 10`. Exactness survives
because **rounding commutes with integer translation** — `round(x + k) ==
round(x) + k` — so points may round while the offset between two repeated
elements does not. Verified visually (`runs/figs/tabler_roundtrip.png`: gears,
rounded squares and wifi arcs all correct) and by `scripts/repeat_oracle.py`:

| corpus | icons/drawings with a repeat | bytes saved | ratio | tol 0 → 1 |
|---|---|---|---|---|
| **Tabler outline**, 5,125 icons, 77 B each | **19.7%** | **3.79%** | **1.0393** | 3.79 → 3.85 |
| SVG-Icons8 (DeepSVG tensors) | 3.2% | 0.62% | 1.0063 | 0.62 → **3.17** |
| QuickDraw `cat` | 1.4% | 0.01% | 1.0001 | 0.01 → 0.06 |

**380× QuickDraw's exact repeat structure and 6× SVG-Icons8's — and the flat
tolerance curve is the proof that the pipeline kept what the icons had.** The
5× jump in the SVG-Icons8 row is the same measurement finding the same fault it
found before. Claim 2 has a corpus.

Programs are also *shorter* than Tier B's (77 bytes against 112), so the bit arm
costs ~617 symbols and Tier C is cheap to train on.

## What made it work

Source icons whose SVGs are hand-authored, permissively licensed, and on an
integer grid, and do the quantisation ourselves.

**First candidate, verified 2026-08-06: Tabler Icons** —
`github.com/tabler/tabler-icons`, **6,184 icons, MIT, every one designed on a
24×24 grid with a 2px stroke**. The grid is the whole point: 24 × 10 = 240 ≤ 255,
so a ×10 scale onto the ISA canvas is exact. It is also *stroke* art rather than
filled shapes, which matches an ISA built from `MOVE`/`LINE`/`CURVE` far better
than solid icons do. Material Design Icons (Google, Apache-2.0, 24dp grid) is a
reasonable second; Bootstrap Icons, Lucide, Feather and Iconoir are further
candidates whose counts and licences are **not yet verified**.

Volume is not the requirement — claim 2 needs repeated structure, not samples —
and 6k icons on an exact integer grid beats 100k whose repeats have been
rounded apart. Line icons are also where translational repeats actually live:
list rows, menu bars, signal bars, grids, dashes, tick marks.

Re-run this test on any candidate before ingesting it. That is the whole point
of having it.

## Caveats on the numbers above

- **Oracle ceiling, not a result.** These are repeats a perfect compressor would
  find; a model still has to discover them.
- **Whole paths only**, so it is a *lower* bound: repeats *inside* one path
  (dashed borders, tick rows) are uncounted.
- **Translation only.** Much icon symmetry is rotational or mirrored, which
  `REPEAT n dx dy` cannot express at all — extending the ISA is a separate
  decision from sourcing the corpus.

---

## Run 1 landed 2026-08-07 — relativity wins here, and it is mostly a head start

_Moved out of `PLAN.md` when that file was compacted on 2026-08-08. Section
references of the form §N are to `PLAN.md`._


**The delta arm is ahead in 11 of 11 rows**, across three shapes, five budgets
and both seeds. But only one rung has both arms at their own val minimum, and
that is the only one that is a *cost*:

| rung | both arms at minimum? | Δ bits (byte_delta − byte) | per-seed |
|---|---|---|---|
| **converged 1,000** | **yes, drift +0.00 both** | **−5.58 ±3.08** | −4.12, −7.03 |
| budget2000 | no, drift +34/+42 | −11.95 | −11.95 |
| stepmatch 3,000 | no, drift +99/+106 | −12.45 | −10.82, −14.09 |
| budget4000 | no, drift +137/+155 | −20.95 | −20.95 |
| tokmatch 4,797 | no, drift +150/+168 | −24.34 | −20.61, −28.07 |

**The gap grows monotonically with how badly both arms overfit**, so everything
below the first row measures which arm memorises 4,613 icons more slowly. That is
consistent with the mechanism §7 proposed — capacity not spent on translation
invariance is capacity not spent memorising — but it is a sample-efficiency
statement and must never be quoted as what the view costs.

**And the mechanism is mostly low-order statistics.** N-gram models on the same
split, no training at all:

| model | absolute | relative | Δ |
|---|---|---|---|
| order-0 | 487.4 | 423.8 | −63.6 |
| order-1 | 442.9 | 367.4 | −75.5 |
| order-2 | 400.2 | 323.7 | **−76.5** |
| **trained, at each arm's minimum** | **158.8** | **153.2** | **−5.6** |

**The relative view starts 76 bits ahead and ends 5.6 ahead: the transformer
learns away 93% of the representational advantage by itself.** That is claim 1's
Tier A finding in a new coordinate — representations carrying identical
information converge to nearly the same cost, and the alphabet's gift is
mostly a head start. It also names the mechanism: delta operands cluster near
zero and absolute coordinates are near-uniform on 0–255, so a *context-free byte
histogram* already collects 64 of the 76 bits. No invariance argument is needed
to explain most of the effect.

Sampling did not regress — validity 0.938/0.961 against 0.930/0.961, length EMD
22.5/24.5 against 23.2/27.0 — so unlike granularity this is not a likelihood
gain paid for at generation time.

**Tier C is now closed as a venue for this axis.** Its val minimum is at ~1,000
steps (13.9 epochs) on every schedule and 1,500 is already worse, so no budget
gives a converged pair and no number from it is a cost. **The axis should be
re-run where a converged rung exists** — Tier A at 24,000, where all four codecs
converge and the ladder is flat — and that is the only place it can produce a
representational cost rather than a rate.

> **The floor was wrong, and this corrected it.** Three draws of byte/square/1,000
> at **seed 0** — differing only in RNG stream, since eval cadence shifts it —
> read 158.51 / 157.18 / 157.39, spread **1.33 bits**; seed 1 spans 1.51. Tier C's
> resolution floor is **~1.5 bits, not the 0.4 in `docs/evidence.md`**, which was
> a two-seed estimate within a single launch and understated it 4×. So −5.58 is
> **3.7× the floor**, not 14×. Every replicate spread the records support is now
> a column in `budget_table`.

> **Bug found and fixed: `budget_table` mixed corpora.** `(codec, shape, steps,
> seed)` is not unique on Tier C — `aug6000` (the ×18 **augmented** corpus) and
> `ladder6000` are both byte/square/6,000/s0 — and the augmented rung silently
> overwrote the plain one, then got differenced against a 4,797-step run as if
> augmentation were a budget. It printed **−75.61, per-seed +17.39 / −168.62**.
> The ladder is now keyed by corpus, replicates are averaged rather than
> overwritten, and two tests pin both halves. The same latent fault was about to
> hit Tier B, where `cat` and the five-category corpus share step counts.

---

## Training on Tier C — corpus solved, training not (2026-08-06)

Tabler is ingested, split (4,613 / 512, grouped by icon family *and* program
identity) and measured: oracle ceiling **3.79% of bytes, ratio 1.0393**, 380×
QuickDraw. [`docs/tier-c.md`](tier-c.md). Claim 2's metric now exists too —
`dm/eval/recovery.py`, bits per symbol on redundant copies against their own
first copy, validated against oracle models with known answers.

**But no budget makes both guards pass.** At 800 steps every arm is short of its
minimum by tails of 14–103 bits/1k; by 1,000 some are +3.96 past it; the 4.8M
floor's own best (159.41) is worse than the 825k arm's (156.75). 4,613 programs
cannot support an asymptote.

**Augmentation was tried and refuted.** ×18 (83,010 programs, ceiling unchanged
at 3.88%, val untouched) gave best 166.04 / 156.11 against the un-augmented
156.75, still drifting +9.6. **The ISA uses absolute coordinates, so a translated
icon shares no token with its original — augmentation multiplies the task as
much as the corpus.**

> **The delta codec is built (2026-08-07) and the claim it rests on is now a
> test, not a hope.** `tests/test_relative.py` pins it: under the relative view a
> translation changes **≤ 2 bytes** of any program, against a mean of >20 in the
> absolute view. Run 1 in the resume point above is the experiment.
>
> **Two things it does not fix, and both must stay attached to any result.**
> Mirrors and quarter turns act on the deltas themselves, so they remain a full
> task multiplier — a policy leaning on those has learnt nothing from ×18. And a
> translation-augmented delta corpus is *near-duplicate* data rather than 18×
> the corpus, so the mechanism under test is "the model no longer spends
> capacity learning translation invariance", not "there is more data". If the
> arm wins, that is the reason; if the two are conflated in the write-up the
> number will not replicate anywhere else.

> **Fusion is degenerate here too**, checked only after reading the table:
> `CURVE`'s six operands are all `Kind.COORD`, so `token` ≡ `token_typed` on
> 512/512 val programs. **Every L0-only corpus does this.** Claim 2 makes the
> axis live again, because `REPEAT` introduces `COUNT` and `DELTA`.

