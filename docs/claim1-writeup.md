# Claim 1 — the write-up spec, and the numbers it landed with

Moved out of `PLAN.md` §9.5 on 2026-08-05 as a specification. **Landed in
`README.md` on 2026-08-07**, which is now the write-up; this file keeps the
ordering rule and the record of what changed between the spec and the text,
because three of the six bullets were specified against numbers that were later
retracted.

`README.md` is the write-up. `docs/results.md` is the full evidence.
`runs/summary_*.md` is the live table and beats both.

## The ordering rule, which did not change

Order the findings by **what survived the guard**, not by what is largest. A
result that passed a noisy guard ranks below one that passed a strict one, and
the retractions are part of the write-up rather than an appendix to it.

## What each finding had to carry, and what it landed as

**1. Headline: granularity, and the pair that makes it a result.**

- *Specified as:* "a two-symbol alphabet is free", −0.72 ±2.18 at the converged
  point against +42.52 ±37.99 at a fixed token budget.
- *Landed as:* **−0.67 ±0.77 at the 24,000-step rung** (per-seed −1.06, −0.27),
  and the claim is **equivalence, not victory** — the interval crosses zero.
  −0.72 is retracted: its *reference* arm was not converged either, and the
  budget ladder that proved it is now part of the write-up rather than a
  footnote to it.
- *Added after the spec was written:* the pair is no longer +42.5 against −0.67.
  It is **Tier A −0.67 against Tier B multi +11.58 ±0.60**, and the corpus is the
  variable. Lead with that; the budget-regime pair is the second-order point and
  goes beside it, explicitly labelled as separation rates.
- State the 8× inference cost in the same breath — 254 Mtok against 32, 48
  minutes against 4. It is why the practical answer differs from the
  representational one, and it is a claim-4 constraint.

**2. Best likelihood, worst sampling — independent of every guard.**

At Tier A's converged point the bit arm is first on bits/drawing (161.2 vs byte
161.9), perfect on all three validity columns, and 1.9–2.5× worse on length EMD
(13.6 vs 7.0 and 5.4), generating median-25-byte programs against a real 41 with
both seeds agreeing. Retract the schema-4 framing of "a 2% NLL gap next to a 3×
sampling gap": at convergence there is no NLL gap at all, which makes the
disproportion sharper.

*Added:* on Tier B multi the bit arm has the **worst** likelihood and still the
best validity, so the two failures are not one failure. And §9.5a has since
diagnosed the undershoot as a halting-hazard bias — 1.5–2.6× over-assignment at
instruction boundaries, 22% undershoot **with no sampling involved** — so the
write-up may now state a mechanism, which the spec forbade it from doing.

**3. Fusion is the IconShop objection measured, and it is a per-byte rate.**

- *Specified as:* +1.22 ±0.09 at converged/square.
- *Landed as:* **+0.93 ±0.98 at 24,000, unresolved**, and the resolved row is
  **+4.87 ±2.18 at tokmatch/wide** against ≈0 at deep. Quote the wide row as the
  measurement and the square row as unresolved; +1.22 was a 12,000-step number
  whose arms were not at their asymptote.
- Report **+0.041 bits per bytecode byte** as well as per drawing — the slope is
  stable across seeds (+42.1, +39.2 milli-bits/byte) and the per-drawing number
  is that slope times *this corpus's* mean length.
- *Added:* fusion is measurable **only on Tier A**. The typed codec keys on
  operand `Kind`, and every L0-only corpus has one kind, so `token` ≡
  `token_typed` byte-for-byte on QuickDraw and Tabler alike. The write-up must
  say this, or the missing Tier B row reads as an omission rather than as a null.

**4. Typing.**

- *Specified as:* unresolved except at wide (−1.45 ±0.92), with the +0.47
  retraction included.
- *Landed as:* both — **+0.11 ±0.07 at 24,000/square** (per-seed +0.10, +0.11) is
  the tightest resolved row in the project, **and −1.45 ±0.92 at wide** has the
  opposite sign. Report both and do not narrate a mechanism. Everywhere but
  Tier A the per-seed deltas disagree in sign and the axis is below the floor.

**5. Say in words that `token_typed` deep and wide are over 1M parameters.**
The summary marks them `*`. Unchanged, and landed.

**6. Every arm undershoots generated length**, not only the bit arm (p50 24–33
against a real 41). The schema-3 explanation — reporting the overfit endpoint —
is retracted now that drift is zero. Unchanged, and landed with the §9.5a
diagnosis attached.

## One bullet the spec did not have, and the write-up needs

**7. The resolution floor is a first-class result.** Measured per corpus from
nulls: ~1.0 bits Tier A, ~2.0 Tier B `cat`, ~2.5 Tier B multi, ~0.4 Tier C. No
axis smaller than its corpus's floor is resolvable at k = 2, whatever the paired
interval says. Granularity clears it by 5×; typing and fusion do not, anywhere.
Stating the floor turns "a table of unresolved rows" into a measurement of what
this instrument can and cannot see, which is the more useful claim.
