# Tier B (QuickDraw `cat`) — the full record

Spun out of `PLAN.md` on 2026-08-06 when the corpus moved to five categories.
The plan keeps the conclusions; this keeps the evidence. Records are
`runs/quickdraw_*eps4_*.json` and `runs/qdB_gate6000_*.json`.

## The corpus

`rdp_eps=4.0`, `max_len=2560`, `cat` only: 70,000 train / 2,500 valid, 129.2
bytes per program against Tier A's ~42, bit sequences to 2,457 symbols, zero
truncation on all four codecs, chamfer 2.26 px on a 256 px canvas, validity
1.000, nothing dropped. Chosen by `scripts/quickdraw_check.py` over the *full*
train split; see `docs/history/instrument.md` §5 for the truncation fault that
search was built to avoid.

## The arms, 24,000 steps

| arm | params | seed 0 | seed 1 | drift s0/s1 |
|---|---|---|---|---|
| token_typed | 957,184 | 488.31 | *(null, see below)* | +0.04 |
| byte | 824,704 | 489.18 | 491.15 | +0.02 / +0.02 |
| token | 826,112 | 489.80 | 489.99 | +0.00 / +0.13 |

All cells flat on both guards, 1,000 paired val programs, no truncation.

## Why no axis is claimable here

**The fusion axis degenerates.** `token` and `token_typed` encode *identically*
on any L0-only corpus: QuickDraw emits MOVE/LINE/HALT, whose one operand kind is
`COORD`, and `Kind.COORD == 0`, so the typed codec's per-kind offset is zero
everywhere. Verified equal on 1000/1000 val programs; 1,024 of its 1,293 rows
are unreachable. The two runs still differ by **−1.48 ±0.49**.

**The seed null is larger still.** `byte` seed 1 minus `byte` seed 0 — same
encoding, same vocabulary, same parameter count, differing only in seed — is
**+1.97 ±0.48 bits/drawing**. So the ~1.5–2.0 bit gap is initialisation noise,
not a systematic cost of dead vocabulary rows.

**And typing flips sign across seeds:** +0.62 at seed 0, −1.16 at seed 1, mean
−0.27 with a between-seed sd of 1.26. By this project's own rule an axis whose
per-seed deltas disagree in sign is unresolved however narrow the interval. To
resolve a 0.3-bit effect against that spread would take ~68 seeds.

**Length scaling does not rescue it.** Regressed on program length the *null*
pair gives −27 m-bits per bytecode byte, bootstrap CI [−47, −6], excluding zero.
A better model is better at every token and long programs have more tokens, so
any global quality difference is length-proportional by construction. The
per-byte rate is exactly as noise-prone as bits/drawing.

## The floor, and why the corpus moved

| shape | params | best | final | drift |
|---|---|---|---|---|
| byte @ square | 824,704 | 489.16 @23,000 | 489.18 | +0.02 |
| byte @ reference | 4,812,544 | **484.17 @12,000** | 496.53 | **+12.36** |

The reference model's ladder is a clean U (508.9 → 484.2 → 496.5) while train
loss falls monotonically 2.73 → 2.44. **Headroom at its own optimum is +5.01
bits**, so Tier B genuinely has capacity room — but 24,000 steps over 70,000
programs is 21.9 epochs and the denominator overfits. The arms do not.

The 6,000-step gate that authorised the sweep read +10.25 bits of headroom;
that was a rate and it overstated the converged number by 2×. Sign right,
magnitude wrong.

**Conclusion: capacity is not the binding constraint on Tier B (`cat`) — seed
variance is, and the floor is data-limited.** Both point the same way: more
corpus. Hence the five-category corpus in `PLAN.md` §7.
