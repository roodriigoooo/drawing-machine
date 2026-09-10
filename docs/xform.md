# `XFORM` — the transform tier, run by run

`docs/direction.md` §2 argued for it, `PLAN.md`'s "Direction item 2" carries the
plan and the falsifiers as they were written **before** the runs, and this is the
record of what the runs said. P1–P4 (the ISA, the constructed corpus, the oracle,
the device port) landed 2026-08-11 with no result to report beyond conformance;
**P5 is the measurement, and it landed 2026-08-12.**

**The verdict, in three parts, and the first one is the shape of the other two.**

1. **A model discovers transformed reuse essentially as well as it discovers
   translated reuse.** `recovery` **+0.9603 / +0.9601** on D4 orbits against
   **+0.9700 / +0.9801** on a matched translation-only control. So the claim the
   tier was built to test is *supported* — and its pre-registered falsifier,
   which read "at or below the control's", **fired**. The inference that
   falsifier licensed does not follow from it, and §5 is about why.
2. **Because the model already found the structure, the opcode's own term is
   small — and the fold's indirect term turned out to be real.** `REPEATX`
   folds **40.9%** of this corpus's bytes. Its direct term — copies removed
   minus header paid — is **+5.84 ±0.18** of 479.2 bits/drawing (**1.2%**),
   staying within **5.67–6.14** across three structured codecs. The rest of the
   arms' difference was confounded with the codec until the structured `byte`
   arm landed (§3.1): with the codec held fixed, `context` survives at
   **+12.30/+15.62**, so a 41% shorter program is genuinely easier to model and
   the tier's like-for-like value is **+19.80 ±1.84 bits/drawing (4.1%)** —
   understated, not overstated, by the first reading.
3. **Where the tier pays is termination, and it pays there by a factor of 4–10.**
   On the identical scenes, the flat spelling's sampled length distribution sits
   **41–112%** of a program's own length away from the corpus's; the `REPEATX`
   spelling sits **9–11%** away at every seed and codec. A copy count that is an
   operand is decoded. A copy count implied by the geometry has to be guessed at.

And one axis closes: **fusion is unresolvable and now permanently closed.**
`token_typed` − `token` reads **+2.71** at 12,000 steps and **−2.42** at 24,000 —
the sign inverts across the ladder and both magnitudes sit inside this corpus's
own same-config seed spread of 0.4–4.7 bits.

[Figure 9](figs/fig9_transform_recovery.svg) ·
[Figure 10](figs/fig10_fold_price.svg) ·
[Figure 11](figs/fig11_termination.svg)

---

## 1. What ran

Fourteen runs, two seeds each, all complete at `schema = 4` with `drift ≤ +0.07`.
Launched from `runs/p5_recovery.sh` and `runs/p5_fusion.sh`; 200,000 train scenes
throughout, so 24,000 steps is 7.7 epochs. An eighth pair — the structured
`byte` arm, `runs/p6_spelling_byte.sh` — landed 2026-08-12 as §7's first
follow-up, and it is the pair §3.1 reads.

| tag | corpus | spelling | codec | params | steps | bits/drawing (s0, s1) | seed spread |
|---|---|---|---|---|---|---|---|
| `composed_c24000n2` | translation only, n = 2 | flat | byte | 824,704 | 24,000 | 521.12, 521.50 | 0.38 |
| `composed_x24000n2` | D4 orbits, n = 2 | flat | byte | 824,704 | 24,000 | 524.89, 520.16 | **4.73** |
| `composed_x24000n24` | D4 orbits, n ∈ {2,4} | flat | byte | 824,704 | 24,000 | 477.93, 480.54 | 2.61 |
| `composed_sconv12000_token` | D4 orbits, n ∈ {2,4} | `REPEATX` | token | 826,752 | 12,000 | 463.81, 464.32 | 0.51 |
| `composed_sconv12000_token_typed` | ″ | `REPEATX` | token_typed | 990,592 | 12,000 | 468.15, 465.41 | 2.74 |
| `composed_sconv24000_token` | ″ | `REPEATX` | token | 826,752 | 24,000 | 458.58, 461.87 | 3.29 |
| `composed_sconv24000_token_typed` | ″ | `REPEATX` | token_typed | 990,592 | 24,000 | 457.37, 458.23 | 0.86 |
| `composed_sconv24000_byte` — §7's follow-up | ″ | `REPEATX` | byte | 824,704 | 24,000 | 459.97, 458.90 | 1.07 |

**The three flat corpora are three different val sets and their bits/drawing are
not comparable to each other** — 521 against 522 against 479 says nothing, it is
three corpora. What *is* comparable is `recovery` (a fraction of each corpus's
own ceiling) and the flat/structured pair (the same 1,000 scenes in two
spellings, §3).

> **This corpus's resolution floor, measured for the first time: 0.4–4.7 bits,
> mean 2.0.** Eight same-config seed pairs (the structured `byte` pair added
> 1.07), which is more replicate evidence
> than any previous corpus in this project got at once. It is not one number,
> and the arm with the widest spread is the *transformed n = 2* arm at 4.73 —
> the same arm whose generation columns fail to replicate in §4. **Every
> difference below is read against this.**

### 1.1 Why these runs converge lower than every earlier one, and what that is not

Every composed arm's val curve settles below every QuickDraw arm's — 458–522
against 489–557 — and the first thing to say is that **the comparison is illegal
as stated**: different val sets, so the totals are of different drawings.
Normalised per bytecode byte it is fully explained, and the explanation is a
property of the corpus:

| corpus / arm | bytes/drawing | bits/byte |
|---|---|---|
| Tier A — synthetic | 55 | 2.98 |
| QuickDraw — `rdp_eps` 2 | 161 | 3.46 |
| **QuickDraw — `rdp_eps` 4** | 129 | **3.81** |
| composed — control, n = 2 | 208 | 2.51 |
| composed — D4, n = 2 | 208 | 2.52 |
| composed — D4, n ∈ {2,4} | 221 | **2.16** |
| composed — `REPEATX` | 131 | 3.52 |

**The flat constructed corpus costs 2.16 bits per byte because 44% of its bytes
are copies the model gets for 0.11.** Fold them away and the same scenes cost
3.52. That is the same arithmetic as §3 seen from the other side: the low total
*is* the planted redundancy, and no part of it is a model or a schedule doing
better.

> **And the row worth looking for is the one nobody asked for.** On the bytes
> that are **not** copies the constructed corpus costs **3.74–3.76 bits/byte**,
> against QuickDraw's 3.81 — within 2%, on 124.8 bytes per drawing. **Outside its
> planted structure this corpus is as hard as the natural corpus its motifs come
> from**, which is the evidence that it is an instrument and not a toy. A
> constructed corpus that had accidentally become predictable would show it
> exactly there. (Descriptive, not paired: two different val sets and two
> different `rdp_eps`. It explains a level and no claim rests on it.)
>
> The same point in the other direction: the composed arms reach within 1% of
> their final `bits/drawing` at **67–69%** of their schedule where QuickDraw's
> reach it at **50–54%**. Cheaper per byte and *slower* to converge — not an
> easier corpus, a more redundant one.

[Figure 12](figs/fig12_compression_rate.svg) draws both halves, and its right
panel is the argument as a shape: area is bits/drawing, so a drawing's price is
its length times its rate, and the copies are 44% of the width at 3% of the
height.

---

## 2. Recovery: the claim is supported

`dm/eval/recovery.py`, spans read from the generator rather than detected —
a mirrored copy is `x → 255 − x` and shares no coordinate byte with its
original, so the translational matcher finds **nothing at all** on the corpus
built to contain the structure (`tests/test_composed.py` pins that at 0 of 60).

| arm | ceiling (construction) | copy 1 | copies 2..n | `recovery` |
|---|---|---|---|---|
| D4 orbits, n = 2, s0 | 30.39% of bytes, ratio 1.4365 | 3.7416 | **0.14844** | **+0.9603** |
| D4 orbits, n = 2, s1 | ″ | 3.7029 | **0.14788** | **+0.9601** |
| control, n = 2, s0 | 30.16%, ratio 1.4317 | 3.7180 | 0.11168 | +0.9700 |
| control, n = 2, s1 | ″ | 3.7415 | 0.07439 | +0.9801 |
| D4 orbits, n ∈ {2,4}, s0 | 40.94%, ratio 1.6931 | 3.7599 | 0.11334 | +0.9699 |
| D4 orbits, n ∈ {2,4}, s1 | ″ | 3.7777 | 0.11732 | +0.9689 |

**The matched pair is matched to two parts in a thousand**: 207,631 val bytes
against 207,595, 63,090 foldable against 62,601, 69,090 copy symbols against
68,601. The two corpora differ in the group and in nothing else, which is what
makes the contrast legal at all — `dm/eval/recovery.py` refuses cross-corpus
recovery comparisons, so claim 2's +0.807 is quoted as a landmark and never as
the reference.

**The per-copy curve falls monotonically and does not break.** On the four-copy
arm: 0.1511 → 0.0277 → 0.0113 (s0) and 0.1499 → 0.0594 → 0.0131 (s1). Copy 4
costs a twelfth of copy 2. Claim 2's Tier A curve turned *up* one copy past the
trained bound; here every count is inside the training distribution, so this is
the in-count regime that replicated there rather than the out-of-count regime
that did not — 13,860 symbols at each of ordinals 3 and 4, from the ~20% of
scenes drawn with a quarter-turn orbit.

**A copy byte costs 3.0% of an original byte** — 0.113 bits against 3.76 on the
same programs (§3's `motif` row). That single ratio is the whole reason §3 comes
out the way it does.

---

## 3. The fold's price in bits: `dm/eval/spelling.py`

**The instrument, and why the obvious reading needed one.** Both arms are the
same 1,000 scenes at 24,000 steps, one trained on the flat trace and one on the
`REPEATX` form, and both spellings decode to the identical geometry — so their
`bits/drawing` is the cost of transmitting one drawing under two encodings of it,
which is the ISA-as-compression-format question stated exactly. But the two arms
also differ in codec (`byte` against `token`) and in sequence length (221.4 bytes
against 130.8), so the difference of totals is not the fold's value.

The two spellings share most of their bytes *exactly*:

```
flat        prefix │ motif │ copies 2..n │                   suffix │ HALT
structured  prefix │ REPEATX │ motif │ ENDREP │              suffix │ HALT
```

`prefix`, `motif`, `suffix` and `HALT` hold byte-identical content in both, so
differencing the model's cost on them holds content fixed and varies only
context. What is left is the fold, with a sign each way.

`composed_x24000n24_byte_square_s0` against `composed_sconv24000_token_square_s0`
(seed 1 in brackets), bits/drawing:

| class | flat | structured | Δ | flat B | struct B | what it is |
|---|---|---|---|---|---|---|
| `prefix` | 100.66 | 98.95 | +1.70 | 26.9 | 26.9 | shared bytes |
| `header` | — | 5.12 | **−5.12** | 0.0 | 6.0 | `REPEATX` + `ENDREP`: what the fold costs |
| `motif` | 259.13 | 250.59 | +8.54 | 68.9 | 68.9 | shared bytes |
| `copies` | 10.95 | — | **+10.95** | 96.6 | 0.0 | what the fold removes |
| `suffix` | 105.97 | 103.00 | +2.97 | 28.0 | 28.0 | shared bytes |
| `halt` | 1.22 | 0.92 | +0.31 | 1.0 | 1.0 | shared bytes |
| **total** | **477.93** | **458.58** | **+19.36** | 221.4 | 130.8 | |

| term | s0 | s1 | typed side, s0 / s1 |
|---|---|---|---|
| **fold** = `copies` − `header` | **+5.83** | **+6.13** | +5.77 / +6.14 |
| `context` = the four shared classes | +13.53 | +12.54 | +14.79 / +16.17 |
| total | +19.36 | +18.67 | +20.56 / +22.31 |

**The fold's term is +5.84 ±0.18 and the confounded term is not.** Across two
seeds *and* three structured codecs (§3.1 added `byte`) the fold stays within
**5.67–6.14** while `context` moves by 3.9 — which is the signature of one term
being a property of the ISA and the other a property of the arm. Every side
reconciles to **±0.0000** against its own record's `bits_per_drawing`, which is
the check that the classes partition the program.

**The arithmetic behind the smallness, since it makes the result predictable
rather than surprising.** The 96.6 copy bytes would cost 96.6 × 3.76 ≈ **363
bits/drawing** at the price the model pays for the motif they copy. It pays
**11**. The model has already banked ~352 of those bits by itself (§2), so the
opcode's whole remaining budget is 11, and after its own 5.1-bit header it nets
6.

> **What was still confounded here is now measured — §3.1.** `context` contained
> a real effect of the fold — a shorter sequence — mixed with a codec change,
> and the structured `byte` arm at 24,000 steps was named as the run that would
> split them. It ran on 2026-08-12 and the split is below.

### 3.1 The confound, removed — the structured `byte` arm (2026-08-12)

`composed_sconv24000_byte`, k = 2: the same 1,000 scenes, budget and parameter
count as the flat arm — 824,704, the identical architecture — differing from it
in the spelling alone. Both sides of the difference are now one codec, so
`context` can no longer hide an alphabet effect:

| term | byte pair s0 | byte pair s1 | token pair (§3's) | typed pair |
|---|---|---|---|---|
| **fold** = `copies` − `header` | **+5.67** | **+6.02** | +5.83 / +6.13 | +5.77 / +6.14 |
| `context` | **+12.30** | **+15.62** | +13.53 / +12.54 | +14.79 / +16.17 |
| total | **+17.96** | **+21.64** | +19.36 / +18.67 | +20.56 / +22.31 |

Both branches §7 pre-registered are answered, and the second one fired:

- **The fold's term is codec-independent.** Six readings — three structured
  codecs × two seeds — span **5.67–6.14 bits**, mean 5.93. The ISA's term does
  not care what alphabet the stream is read in, which is what "a property of the
  ISA" has to mean.
- **`context` survives the codec being held fixed**, at +12.30/+15.62 against
  the cross-codec pairs' +12.54–16.17. It was never the alphabet. "A shorter
  program is easier to model" is a real, separate consequence of the fold: the
  structured model predicts the *byte-identical* prefix, motif and suffix
  content 12–16 bits/drawing more cheaply, with content, codec, parameters and
  budget all held fixed.

**So the tier's likelihood value was understated, exactly as the fired branch
said it would be.** The honest like-for-like number is the whole spelling
effect, **+19.80 ±1.84 bits/drawing (4.1%)** — the opcode's direct term of
**+5.84 ±0.18 (1.2%)** plus a context term of **+13.96 ±1.66** that exists
because the fold made the program 41% shorter. The 1.2% stays quoted wherever
the *opcode alone* is being priced; the verdict's item 2 now carries both
numbers, and [Figure 10](figs/fig10_fold_price.svg) draws the codec-fixed pair.

---

## 4. Termination: where the tier actually pays

Every generation column is a draw, so all of these are `scripts/resample.py`,
**5 draws × 128 samples**, never a final eval. `length EMD` is also given as a
fraction of each corpus's own mean program length, because the structured
spelling is 41% shorter and an absolute byte distance would credit it for that.

| arm | spelling | length EMD | % of own mean length | validity |
|---|---|---|---|---|
| control, n = 2, s0 / s1 | flat | 26.7 ±12.2 / 25.9 ±6.6 | 13% / 12% | 0.998 / 0.986 |
| D4, n = 2, s0 / s1 | flat | **191.0 ±16.2** / 52.3 ±11.2 | **92%** / 25% | **0.666** / 0.956 |
| D4, n ∈ {2,4}, s0 / s1 | flat | **248.7 ±22.2** / 90.8 ±8.1 | **112%** / 41% | 0.977 / 0.959 |
| D4, n ∈ {2,4}, s0 / s1 | `REPEATX` | 12.3 ±3.6 / 14.0 ±3.0 | 9% / 11% | 0.997 / 0.988 |
| ″, typed alphabet | `REPEATX` | 13.6 ±4.8 / 13.8 ±3.0 | 10% / 11% | 0.989 / 0.991 |

**Three readings, in order of how well they hold.**

- **The structured spelling terminates well and the flat transformed spelling
  does not.** Four structured arm-seeds land at 9–11% with a spread of 3–5
  bytes; four flat transformed arm-seeds land at 25–112%. Same scenes, same
  budget, same parameter budget to within 0.3%.
- **The control terminates well too, at 12–13%.** So this is not "the corpus is
  hard" — it is the group. The control's copies are translations by 120 px, and a
  third copy leaves the canvas; **D4 is closed on the canvas**, so a mirrored
  prefix always has a legal continuation. The generator's own convenience — "with
  a translation of zero, containment is automatic", which is what made the
  transformed arms the *safe* ones to construct — is the sampling hazard.
- **The flat transformed arms do not replicate and the others do.** 191 against
  52, 249 against 91, where the control replicates to 0.8 and the structured arms
  to 1.7. Claim 2's shape again: the direction is a property of the corpus and
  the depth is a property of the seed. **Nothing here should be quoted as a
  level; the ordering is what replicates.**

The failures are almost entirely `unknown_opcode` (213 of 640 samples on the
worst arm-seed, 1 of 640 on the control) rather than `no_halt`: the sampler emits
more content — median 405 bytes against the corpus's 205 — and eventually walks
off the instruction grid.

### 4.1 The mechanism, measured — and it is the opposite of the hypothesis

The reading above was written with a mechanism attached: D4 is closed on the
canvas, so a mirrored prefix always has a legal continuation, and the over-long
samples are the model *continuing the orbit*. **That was wrong, and pointing the
orbit oracle at the samples is what says so** (`scripts/orbit_oracle.py
--samples`, 48 samples per checkpoint, both oracles over generated programs
instead of over a corpus).

| arm | samples carrying an orbit | copy counts found | bytes gen / corpus |
|---|---|---|---|
| the corpus itself, D4 n = 2 | **48/48** | {2: 48} | 208 |
| flat, D4 n = 2 | **6/48** | {0: 24, 2: 6}, 18 unparseable | 412 (2.0×) |
| flat, D4 n ∈ {2,4} | **11/48** | {0: 36, 2: 9, **3: 2**} | 446 (2.0×) |
| flat, control n = 2 | 19/48, 32/48 (two seeds) | {0: 29, 2: 19} / {0: 16, 2: 32} | 240, 196 |
| **structured, unrolled** | **48/48** | **{2: 30, 4: 18}** | 234 (1.06×) |

**The over-generation is lost structure, not continued structure.** The
transformed arm's samples are twice the corpus's length and carry an orbit in
**13%** of draws against the corpus's 100%; the foldable fraction of what it
generates is **5.4%** against the corpus's 30.4%. Free-running, the model does not
over-apply the copy relation — it stops applying it and fills the space with new
content instead. Only 2 samples in 48 carry a count the corpus does not contain,
which is the whole of the evidence for continuation.

> **And the instrument's own control passed.** A generated "mirror" that is a few
> pixels off would be invisible to an exact matcher, which would produce this same
> zero — so the oracle was re-run at `--tol 8`, and the counts are **identical**.
> Where a copy exists in a sample it is exact; there are no near-miss orbits being
> missed. That is the same discipline as figure 6's Tabler row: a null is only
> readable once the detector is shown not to be blind.

**Three things follow, and the third is the strongest case for the tier on
record.**

1. **The copy relation is available for prediction and not for generation.**
   Teacher-forced, a mirrored copy costs 3% of its own original (§2). Free-running,
   the same checkpoint emits one in 13% of samples. This is exposure bias, made
   exact on a structure whose presence can be checked byte for byte rather than
   inferred from a distance.
2. **Transformed reuse is harder to *generate* than translated reuse, while being
   equally easy to *predict*.** The control retains its orbit in 40% and 67% of
   samples against the D4 arm's 13%. **The pre-registered falsifier was looking in
   the wrong column, not merely at the wrong contrast** (§6): the difference
   between the group and plain translation is real and it lives in generation,
   where the falsifier read likelihood and found 1–2%.
3. **`REPEATX` turns a 13% behaviour into a certainty.** Every one of 48
   structured samples carries an orbit once unrolled, in a count mixture close to
   the corpus's own, at 1.06× its length. The copies are not something the model
   must decide to produce — they are the semantics of one instruction it already
   emitted. **That is what the transform tier buys, and it is not a bit count.**

> **What this is not.** One draw of 48 samples per checkpoint, and one seed on
> most rows: the control's two seeds read 40% and 67%, so **quote the ordering and
> never the level**. The structured arm's 48/48 has nowhere to move, and the flat
> arms' 13–23% would have to move by 3× to reach the control's worse seed.

---

## 5. The fusion axis, closed

`token_typed` splits operand values by `Kind`, 274 → 1554 symbols. On any L0-only
corpus the two codecs encode a program byte-identically and the axis does not
exist; the structured spelling is the first corpus in this project on which it
does (0 of 200 val scenes encode identically).

| rung | Δ bits/drawing (typed − token) | per-seed | tail arm/ref |
|---|---|---|---|
| 12,000 steps | **+2.71** | +4.34, +1.09 | −1.04 / −1.13 |
| 24,000 steps | **−2.42** | −1.21, −3.64 | −0.40 / −0.42 |

**The sign inverts across the budget ladder.** Both rungs are sign-consistent
within themselves and they disagree with each other, which is the definition of a
rate rather than a cost — and the ladder says why: over the doubling, `token`
bought 3.84 bits and `token_typed` bought **8.98**. The typed arm carries 19.8%
more parameters (990,592 against 826,752, all in the embedding), converges later,
and crosses over. Both magnitudes are inside this corpus's 0.4–4.7-bit seed
spread.

**The pre-registered prediction is refuted.** Tier A measured fusion as a
per-byte rate, +0.041 bits/byte, which over these 130.8-byte programs predicted
**+5.4 bits/drawing**. At 12,000 steps the reading is half that and inside the
floor; at 24,000 the sign is wrong. A per-byte rate measured on one corpus does
not transfer to another, which is the same lesson as the schedule correction that
was wrong by 6.2 bits.

**So the axis closes permanently rather than provisionally**, exactly as
pre-registered: it was given a corpus where operand kinds are real, a converged
regime, a budget ladder and two seeds, and it did not resolve.

The §3.1 arm also gives this corpus a **typing** reading for free (`byte` −
`token`, structured spelling, 24,000 steps): **−0.79 ±4.27**, per-seed **+1.39 /
−2.96**. Signs disagree inside the floor — typing is unmeasurable here too, on
its fifth corpus, consistent with the claim-1 table everywhere else.

---

## 6. The falsifier that fired on the wrong comparison

Written before the run: *"`recovery` on transformed copies at or below the
control's → the model discovers a D4 orbit no better than the plain translation
it already handles, the transform tier buys nothing the model could not already
do, and `XFORM` is recorded as a measured null."*

It fired. Both seeds: −0.0096 and −0.0201 in `recovery`, +0.037 and +0.073
bits/symbol on the copies, same sign twice. **And the conclusion it licensed is
wrong**, for a reason worth writing down because it is not a statistical one:

- The claim under test was **"a model can discover transformed reuse"**. The
  quantity that answers it is how far the arm is from paying *full price*, and
  both arms are 96–98% of the way there. Nothing about a 1–2% shortfall against
  the control speaks to it.
- The clause the falsifier really wanted — **"does the ISA feature buy
  anything?"** — is not a question about recovery *differences* at all. It is
  answered by recovery's absolute *level*, because the level is what determines
  how much redundancy is left for an opcode to remove. §3 measured it: 40.9% of
  bytes, 1.2% of bits.
- Worse, the comparison is not equally precise on its two sides. The transformed
  arm's two seeds differ by 0.0003; the control's by 0.0101 — 34×. The smaller of
  the two gaps *is* the control's own spread, so at k = 2 the difference is at the
  edge of resolvable and the sign is the only part worth quoting.

**This is a ninth entry in the instrument ledger and the first of its kind.**
The eight on record split three ways — the gap between design and corpus, the
design, the instrument reading it. This one is in the **pre-registration**: a
falsifier can be specific,
measurable, honestly reported, and *still* not bear on its own claim. The
protection is not more seeds — it is asking, of each pre-registered branch, "if
this fires, which sentence of the claim becomes false?" and refusing to write the
branch until that sentence exists.

The three other branches read as written: peak stack did not fire (P4, 492 B),
the convergence guard did not fire (all fourteen runs converged, `drift ≤ +0.07`),
and the fusion branch fired and closed its axis.

---

## 7. What is owed

Nothing on the claim itself. Three measurements would each sharpen a number that
was reported with its confound named — two are now done:

1. ~~**A structured `byte` arm at 24,000 steps, k = 2**~~ **DONE 2026-08-12 —
   §3.1.** Both pre-registered branches read as written, and the second fired:
   the fold term is codec-independent (5.67–6.14 across three structured
   codecs) *and* `context` survives the codec being held fixed (+12.30/+15.62),
   so "shorter programs are easier" is a real and separate effect of the fold
   and the tier's likelihood value was **understated** — like-for-like it is
   +19.80 ±1.84 (4.1%), of which the opcode's own term is +5.84 ±0.18 (1.2%).
2. ~~**The orbit oracle over generated samples**~~ **DONE 2026-08-12 — §4.1.**
   The mechanism inverted under measurement: over-generation is lost structure,
   not continued structure, and `REPEATX` turns a 13% behaviour into 48 of 48.
3. **The sample-quality half on these arms** (`scripts/resample.py` without
   `--no-quality`). `coverage`/`mmd`/`nna` are unmeasured here, and the flat
   transformed arms' samples are the most likely place in this project for a
   coverage/fidelity trade to be visible in geometry rather than in length.

**Not owed:** more seeds on the fusion axis, any run that differences two flat
corpora against each other, and a scale-bearing transform group — ×1.07 scale
destroys 77% of a repeat ceiling by rounding, which is the measurement that made
the group D4 in the first place.

---

## 8. What the whole approach establishes

**Why the corpus had to be built at all.** The question — *can a model discover
that one shape is another shape mirrored?* — was **unaskable** on any corpus this
project had. The D4 orbit oracle finds transformed reuse in **0 of 300** QuickDraw
programs at two simplification levels, so on QuickDraw the question has no
denominator: a `recovery` of anything divided by nothing. Claim 2 worked because
the generator chose the repeat count and the ceiling was arithmetic; `dm/data/composed.py`
does the same thing for transformed reuse, and [`corpus_composed.svg`](figs/corpus_composed.svg)
is what one scene is — a motif, its D4 images, and distractors that are
deliberately not copies of anything, each byte class in its own colour.

### The positive results, in the order they matter

1. **A byte-level model infers a group action between distant spans of a program,
   with nothing in the alphabet to mark it.** `recovery` **+0.96**: a mirrored or
   quarter-turned copy costs 3% of what its own original cost. The flat trace
   contains no `REPEATX`, no `REPEAT`, no marker of any kind — the relation is
   only in the geometry, and the exact translational matcher finds **nothing** on
   this corpus. This is a strictly new capability result for the project, and it
   is within 1–2% of what the same model achieves on plain translation.
2. **The copy-ordinal curve falls monotonically** — 0.151 → 0.028 → 0.011. Claim
   2's Tier A curve turned *up* one copy past the trained bound; every count here
   is in-distribution, and inside the trained counts both corpora agree. So the
   two results are consistent and the composed one adds that a *four*-element
   orbit is not harder than a two-element one.
3. **The corpus is a valid instrument.** Outside the planted structure it costs
   3.74 bits/byte against the natural corpus's 3.81 (§1.1), the oracle that reads
   it is validated in both directions (11.74% on Tabler icons, 0.00% on
   QuickDraw, provenance recovered on 40 of 40 scenes), and its ceiling is
   arithmetic rather than estimated.
4. **The ISA extension is real on the device**: `XFORM`, `ENDX`, `REPEATX` in
   1,862 B of flash and 492 B of peak stack, **12,670/12,670 traces bit-identical**
   natively, under UBSan and as Thumb on an emulated Cortex-M0, at 12 B of stack
   per scope. And the fuzzer that came with it **found a crash in the reference
   interpreter** that had been there since the tier existed.
5. **`REPEATX` makes the model generate composition.** This is the result that is
   easiest to see and hardest to quote: at 24,000 steps on the same scenes, the
   flat arm's samples run 2.4× too long and degenerate into over-drawn scribble,
   while the structured arm's samples are mirrored and rotated pairs of clean
   motifs at roughly the corpus's own length.
   [the two-spelling sheet](figs/spellings_drawn.svg) is the two arms side by side, and
   [Figure 11](figs/fig11_termination.svg) is the same thing as a measurement:
   9–11% length error against 25–112%.
6. **Two instruments exist that did not.** `dm/eval/spelling.py` prices a
   compression feature in bits by position class — the general tool for "what did
   this ISA change actually buy" — and the fusion axis, which had degenerated on
   every previous corpus, was finally given a corpus where it exists and **closed
   on evidence** rather than left open.

### The negative results, which are the more useful half

1. **A compression feature's value in bits is bounded by the model's own failure
   to compress, and that bound was never checked before the opcode was built.**
   `REPEATX` folds 40.9% of the bytes and **1.2%** of the bits *directly*. The
   363 bits of redundancy per drawing were real; the model had already taken 352
   of them. (§3.1 adds the indirect term the bound cannot see — the shorter
   sequence — which lifts the like-for-like value to 4.1%; the point about the
   bound stands for the opcode's own term, which is the one the bound bounds.)
   [Figure 10](figs/fig10_fold_price.svg)
2. **A pre-registered falsifier fired and could not settle its own claim** (§6).
   The lesson is procedural and it generalises past this project.
3. **The flat transformed arms are the worst-behaved arms on record when
   sampled** — validity 0.666 at the worst seed, and a length error that varies
   4× between two seeds of one config. A corpus-construction convenience (D4 is
   closed on the canvas, so no placement needs clamping) turned out to be a
   sampling hazard (a mirrored prefix always has a legal continuation).
4. **Typing is closed.** Given the corpus it asked for, a converged regime, a
   budget ladder and two seeds, the axis inverted sign across the ladder inside
   its own floor.

### What it does *not* establish

- **That `CALL` is worth building.** `REPEATX` folds one motif drawn several
  times. "An X on top of a Y, mirrored" — two *different* bodies — is the `CALL`
  case, and the bit accounting here is the argument against expecting much from
  it: whatever redundancy a scoped call would remove, measure what the model
  already spends on it first.
- **Anything about natural corpora.** The corpus was constructed precisely
  because natural drawings do not contain this structure. The one natural
  datapoint is Tabler's icons at 11.74%, and no model has been trained on those
  folds.
- **Anything about scale.** The group is D4 ⋉ integer translation because ×1.07
  scale destroys 77% of a repeat ceiling through rounding. "A cat twice as big"
  is outside the ISA by construction and stays there.

---

## 9. Regenerating every number above

`runs/` is not under version control, so each of these is what stands between a
figure and an empty panel. No training; minutes on CPU apart from the resamples.

```bash
# §1's table, the budget ladder and the fusion rungs
python3 scripts/sweep.py --data composed --summarise-only

# §2, and figure 9. Writes runs/recovery_composed_*_indist.json
python3 scripts/recovery.py runs/composed_*24000n*_byte_square_s?.pt

# §3, and the cross-codec replication figure 10 quotes as a range. Refuses any
# pair that is not two spellings of one scene set
for s in 0 1; do
  python3 scripts/spelling.py \
      --flat runs/composed_x24000n24_byte_square_s$s.pt \
      --structured runs/composed_sconv24000_token_square_s$s.pt
done

# §3.1, and figure 10's own pair: the codec held fixed across the comparison
for s in 0 1; do
  python3 scripts/spelling.py \
      --flat runs/composed_x24000n24_byte_square_s$s.pt \
      --structured runs/composed_sconv24000_byte_square_s$s.pt
done

# §4, and figure 11. ~45 min on mps; the slow one, and the only one that samples
python3 scripts/resample.py runs/composed_{c24000n2,x24000n2,x24000n24}_byte_square_s?.pt \
    runs/composed_sconv24000_token{,_typed}_square_s?.pt \
    --seeds 5 --n 128 --no-quality --device mps

python3 scripts/plots.py --only 9 10 11
```

> **`scripts/resample.py` keys its report on the checkpoint and the sampler, not
> on `--seeds` or `--n`.** A smoke test at `--seeds 1` silently replaces a
> five-draw reading, which is how a report was lost on 2026-08-10. If the numbers
> in §4 ever disagree with `runs/*_gen_ar.json`, check that file's own `seeds`
> field before believing either.
