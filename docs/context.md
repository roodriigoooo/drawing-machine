# Direction 2 — causal context use, measured

**Status 2026-08-16.** Direction 2's evidence record, in the shape
[`state.md`](state.md) holds Direction 1's. The independent
[`artifact/implementation audit`](context-audit.md) traces every headline to
content hashes and records residual risks. [`directions.md`](directions.md) §1.1
summarizes closure and records Direction 3's later bounded stop;
[`evidence.md`](evidence.md) carries the one-line claims. The protocols that
authorised each run are [`context-protocol-v1.json`](context-protocol-v1.json),
[v2](context-protocol-v2.json), [v3](context-protocol-v3.json),
[v4](context-protocol-v4.json) and [v5](context-protocol-v5.json), and none of
them is ever edited after a scoring run.

**Both stages have run.** C3 measured the conditional distribution and C4 the
model's own outputs. The one-sentence result is that the effect is very large
under teacher forcing, survives free-running decoding in sign and interval, and
is far too small in generation to produce the compatible continuation —
[Figure 15](figs/fig15_context_generation.svg) is that sentence as a picture.

## 1. What the run establishes

Both co-primary venues return **`relational_context_use`** under
[`context-protocol-v2.json`](context-protocol-v2.json). `Delta`, bits per target
byte, seed 0 / seed 1, with component intervals and case counts:

| venue | n | `Delta` | CI s0 | CI s1 | `Delta > 0` | components |
|---|---:|---|---|---|---|---:|
| `composed_shape_copy2` | 28 | **+4.2796 / +4.7193** | [4.126, 4.428] | [4.392, 5.116] | 28/28 both | 21 |
| `synthetic_flat_step` | 64 | **+1.3337 / +1.2839** | [1.163, 1.501] | [1.114, 1.462] | 63/64 both | 64 |
| `synthetic_flat_step_yaxis` | 48 | −0.0520 / +0.0443 | [−0.263, 0.174] | [−0.155, 0.216] | 20/48, 29/48 | 47 |
| `synthetic_flat_step_d4r1` | 64 | +0.0505 / −0.0052 | [−0.055, 0.161] | [−0.128, 0.106] | 35/64, 32/64 | 64 |

Holm p = 0.001 on both co-primary venues at α = 0.05. Both matched controls are
**measured ≈0 rather than forced to 0**, repairing the schema-1
[fault](state.md): donor-target −0.039/−0.109 (shape) and −0.001/−0.005 (step); unrelated-block
−0.0006/−0.0055 and −0.0004/+0.0040. The numbers reproduce bit-identically
across a manifest rebuild between v1 and v2.

## 2. What rules out the alternatives

Four checks, all computed from the frozen manifests and the persisted raw NLLs
with no extra model run:

1. *Not distance.* The unrelated block sits ≥30 bytes before the target and the
   relevant block abuts it, so an exact position match could not be made and the
   two are confounded by construction. The prefix-NLL column settles it: on the
   step venue the irrelevant edit costs **+21.25 / +18.05** bits of prefix
   surprise and moves the target **−0.0004 / +0.0040** bits/byte, while the
   relevant edit costs **+2.19 / +2.66** and moves it **+1.332 / +1.279**. On
   the composed venue the two edits are surprise-matched (+2.80 vs +2.44, +3.30
   vs +3.83) and report the same null.
2. *Not proximity.* 38 of 64 step cases put the crossed target numerically
   nearer the preceding copy than the compatible one — where a proximity
   heuristic predicts `D < 0`. `D` is **+1.306 / +1.272** there, 38/38 positive.
3. *Not frequency.* Per-case `D` correlates −0.156/−0.229 (step) and
   −0.416/−0.050 (shape) with the frozen marginals: every sign is against the
   hypothesis direction.
4. *Not coordinate-bag matching.* The shape effect holds at all five orbit
   elements and is weakest on the mirror (+3.82/+4.03), the element that
   preserves y verbatim and would favour a bag model. A 45-byte rotated motif is
   reproduced at 0.16–0.39 bits/byte against a measured generic non-copy rate of
   4.19–4.85.

## 3. The axis question is open, and both diagnostics failed instructively

The exact quarter turn preserves the relation byte for byte and costs the model
**×2.18 / ×2.38** its prefix rate and ×27.6 / ×24.3 its matched-target rate — a
null from a model with no usable distribution. The in-place y venue, built in
response, moves only the two copies the relation is about (15 of ~76 prefix
bytes), costs **×1.34 / ×1.32**, and reports `D` spanning zero with 23/48 and
22/48 cases positive; the compatible *vertical* copy costs 3.45/3.74 bits/byte
against 0.199/0.247 horizontal and a generic non-copy rate of 2.7–2.9.

That last reading is **exploratory**: v2's precondition gated the diagnostic on
the cost of the compatible target — the outcome, not the instrument — and
declared it unreadable in exactly the case it exists to detect.
[v3](context-protocol-v3.json) keeps the prefix bound and drops the target one,
after the fact. v2's `incomplete` is the confirmatory record.

**This changes the step venue's label, exploratorily.** Once the y venue is
readable and does not transfer, `scripts/context_gate.py` returns
`anisotropic_relation_use` for `synthetic_flat_step` rather than
`relational_context_use` — that is what `runs/context_gate_v4.json` says, and it
is the *corrected* reading, not the confirmatory one. The confirmatory record
remains v2's `relational_context_use` on both venues, because the rule that
produces the new label was fixed after seeing the data it reads. The composed
venue is `relational_context_use` under every version of the rule; it has no
axis diagnostic attached to it.

## 4. What generation does with it — C4

C4 prompts at the same instruction boundary, decodes both worlds under **one
shared block of pre-generated uniforms**, and asks whether each world's
completions lean toward its own compatible continuation:

```text
f(w)  = d(c_w, y_other) - d(c_w, y_own)
D_gen = 1/2 * [f(A) + f(B)]
```

Run 2026-08-16 under [v5](context-protocol-v5.json), cap 512, `top_k` 40,
temperature 1.0, structural mask **off**, 64 draws per world on the co-primary
venues and 8 on the diagnostics, both seeds. `Delta_gen` subtracts the matched
irrelevant prefix edit:

| venue | n | seed | `D_gen` | donor ctl | block ctl | **`Delta_gen`** | CI | **hit_own** |
|---|---:|---|---|---:|---:|---|---|---:|
| `synthetic_flat_step` | 64 | 0 | +0.0826 | −0.0006 | +0.0001 | **+0.0824** | [.0766, .0881] | **.135** |
| | | 1 | +0.0587 | +0.0001 | +0.0003 | **+0.0583** | [.0529, .0637] | **.074** |
| `composed_shape_copy2` | 28 | 0 | +0.0303 | +0.0001 | −0.0034 | **+0.0337** | [.0197, .0462] | **.013** |
| | | 1 | +0.0329 | +0.0000 | +0.0014 | **+0.0315** | [.0159, .0497] | **.011** |
| `synthetic_flat_step_yaxis` | 48 | 0/1 | −0.0035 | +0.0014 | +0.0000 | −0.0035 | [−.0106, 0] | 0 |
| `synthetic_flat_step_d4r1` | 64 | 0/1 | −0.0004 / +0.0001 | −0.0007 | −0.0004 | ≈0 | spans 0 | 0 |

Both co-primary venues return **`generation_robust_context_use`** on `D_gen` and
**`controlled_generation_context_use`** on `Delta_gen`: both intervals exclude
zero with the same sign in both seeds, at the bootstrap's p floor of 0.0005.

**Both generation controls are measured at zero, not assumed.** The matched
irrelevant edit moves the completions by −0.0034 to +0.0014 while the relevant
edit moves them by +0.030 to +0.083 — a factor of 25 to 800. So the completions
are following the *relation*, not reacting to the fact that the prefix changed.
That was the one thing v4's uncontrolled `D_gen` could not say, and it is the
generation-side echo of C3's specificity result.

The earlier 8-draw run under [v4](context-protocol-v4.json) stands as the
diagnostic that authorised this one; its numbers (`D_gen` +0.0839/+0.0572 and
+0.0351/+0.0177) are superseded as estimates, not retracted.

**Read the `hit_own` column before either label.** Neither carries an
effect-size floor — none was declared for `D_gen`, and `Delta_gen`'s label
inherits that — so both mean the preference is detectable, not that the model
produces the compatible continuation. On the composed venue it produces it in
**1.3% / 1.1%** of draws. The preference is real, and controlled, and the
capability is not there. `relation_consistent` equals `hit_own` to three
decimals on every co-primary cell, so the model is not even applying the
relation to the wrong content — when it misses, it misses structurally.

The step venue is the interesting cell: **13.5% / 7.4%** exact hits, an order of
magnitude above the composed venue, on a relation that is one translation rather
than a D4 image of a substituted motif.

**The two nulls fail differently, and only one is about the relation.** On the
y venue 47 of 48 cases have `D_gen` *exactly* zero: the model reaches
(0.935/1.00), emits a skeleton-matching block (0.928/1.00) and lands equidistant
from both targets. It is structurally right and relationally uninformed. On the
exact quarter turn it does not even reach the target width in 44%/56% of draws —
consistent with C3's finding that it has no usable distribution there.

Seed 0's composed cell also faults on 34% of completions (`canonical` 0.656)
where seed 1 is at 0.997. Two checkpoints disagree that much about basic
validity, which is one more reason two of them are not a population.

### 4.1 How the two generation controls are defined

v4's estimand had **no control**, and C3's headline is `Delta = D − D_control`
precisely because a raw `D` is confounded. A raw `D_gen` says the completions
moved when the prefix moved, not that the *relation* moved them — a model that
lurches at any prefix edit scores the same, which is the schema-1
[retraction](state.md) one stage later.
[v5](context-protocol-v5.json) declared both controls, in C3's own pairing,
**before** the run that carries them:

```text
D_gen_target_control = 1/2 * [(d(c_A,ct_B) - d(c_A,ct_A)) + (d(c_B,ct_A) - d(c_B,ct_B))]
D_gen_block_control  = 1/2 * [(d(cb_A,y_B) - d(cb_A,y_A)) + (d(cb_B,y_A) - d(cb_B,y_B))]
Delta_gen            = D_gen - D_gen_block_control
```

The donor-target control costs no extra decoding — it scores the same two
completions against the control targets. The block control costs one extra
decode rather than two, because `block_prefix_a == prefix_a` in both venues, so
`cb_A` is `c_A` byte for byte and `Delta_gen` reduces to
`1/2 * [f(B) - f_block(B)]`. C3's block control has the identical property,
which is why `D_minus_block_control` there sits so close to raw `D`.

`D_gen` stays the primary endpoint: promoting `Delta_gen` to primary *after* v4
returned a positive `D_gen` would be choosing an endpoint from a result. The
controlled verdict is its own label, and `tests/test_context_c4.py` pins it with
two scripted models — a relation-follower and an edit-reactor — that produce the
identical `D_gen` and opposite `Delta_gen`.

### 4.2 What the 64 draws bought, and what they could not buy

The expansion was worth making and it is worth saying exactly what it did. At 64
draws per world, resampling the draws while holding the case set fixed gives the
sampler's own contribution to the venue mean:

| venue | seed | components | reported interval sd | sampler sd | sampler share of variance |
|---|---|---:|---:|---:|---:|
| `synthetic_flat_step` | 0 | 64 | 0.00287 | 0.00133 | 21% |
| `synthetic_flat_step` | 1 | 64 | 0.00270 | 0.00112 | 17% |
| `composed_shape_copy2` | 0 | 21 | 0.00624 | 0.00198 | 10% |
| `composed_shape_copy2` | 1 | 21 | 0.00920 | 0.00191 | 4% |

So four fifths of the remaining uncertainty is **between cases**, and on the
composed venue it is 90–96%. The step venue's half-width duly fell from 0.0101
to 0.0056; the composed venue's did not fall at all, and for seed 1 it *rose*
from 0.0102 to 0.0180 — a 21-component bootstrap's width is itself unstable, and
the honest reading is that its interval was never draw-limited.

**The composed venue is component-limited, and that ceiling is the corpus.** 668
of 1,000 val scenes have no same-skeleton donor motif, which is why there are 28
cases and 21 components at all ([audit](context-audit.md)). No
amount of decoding narrows an interval whose width is 96% between-case. Only a
larger corpus would, and enlarging it after seeing a result is the one thing the
freeze forbids.

Composed seed 1's point estimate moved from +0.0177 to +0.0329 between the two
runs. Those are independent samples on different variate streams, so the right
comparison uses both standard errors: the gap is about 1.4 of them, which is
unremarkable. It is not evidence of an instrument problem, and it is a good
reminder that a point estimate from 8 draws over 21 components was never precise
enough to quote on its own.

## 5. What this does not establish

- **Two checkpoints are not a population.** Every number here is conditional on
  `synthetic_c2flat24000_byte_square_s{0,1}` and
  `composed_x24000n2_byte_square_s{0,1}`. A learned-architecture headline needs
  independently trained paired seeds declared before the run.
- **A generation preference is not a generation capability.** C4 is built and
  run, and both co-primary venues clear its interval — but on the composed
  venue that verdict sits on 1.3% / 1.1% of draws emitting the compatible
  continuation (§4). *Exposure-limited context use* is not the label here;
  neither is "the model can do it".
- **The composed venue's precision is capped by the corpus, not the budget.**
  90–96% of its interval width is between-case over 21 components, and 668 of
  1,000 val scenes admit no case at all (§4.2). More decoding cannot fix that,
  and enlarging the corpus after a result is what the freeze forbids.
- **Transform awareness is not shown.** The shape effect is not coordinate-bag
  matching, but nothing here separates "computes `A delta`" from "copies content
  into a learned slot". A new frozen D4 source-shift venue would be required.
- **Memorisation is untestable here.** Sources/donors come from validation
  programs/scenes, but no in-identity arm or persisted motif-equivalence group
  exists ([audit](context-audit.md)).
- **The axis reading is exploratory**, for the reason in §3.

## 6. A broken freeze, and how it was closed

`dm/eval/context_c4.py` was edited **after** the C4 run and after
[v4](context-protocol-v4.json) froze its digest, so no file matching v4's
recorded `7ac8ab3f…` existed any more. On its face the run was unreproducible:
the protocol named a source that was gone.

It was re-run under the edited file (`632bec1cfcbe…`) against v4 unchanged, and
**every case row, every draw row and every venue summary reproduced
bit-identically** on both manifests and both seeds. The edit was inert. v4's
reports therefore stand as evidence, and [v5](context-protocol-v5.json) records
the current digests so the tree and the freeze agree again.

Two lessons, both in [`traps.md`](traps.md): a freeze is only as good as the
last edit after it, and "the digest moved" is a question to answer by rerunning,
not a verdict either way. Had the numbers moved, the reports would have been
retracted rather than explained.

C4 also shipped with **no tests**, which inverts §4.7's own rule that tests
precede the run — the one place Direction 2 broke its own discipline.
`tests/test_context_c4.py` is that debt paid, and it precedes the v5 run.

## 7. Artifacts

| what | where |
|---|---|
| step / rotation / y-axis manifest | `runs/context_c3_synthetic_flat_v4.json`, sha `dcc93caf0003…` |
| shape manifest | `runs/context_c3_composed_shape_v4.json`, sha `2679fc2356da…` |
| C3 reports | `runs/context_c3_synthetic_byte_dcc93caf0003.json`, `runs/context_c3_composed_byte_2679fc2356da.json` |
| C4 diagnostic reports, 8 draws | `runs/context_c4_synthetic_byte_dcc93caf0003.json`, `runs/context_c4_composed_byte_2679fc2356da.json` |
| **C4 controlled reports, 64 draws** | `runs/context_c4_synthetic_byte_dcc93caf0003_v5.json`, `runs/context_c4_composed_byte_2679fc2356da_v5.json` |
| frozen gate output | `runs/context_gate_v2.json` |
| exploratory gate output | `runs/context_gate_v3_exploratory.json` |
| gate with the uncontrolled C4 verdict | `runs/context_gate_v4.json` |
| **gate with the controlled C4 verdict** | `runs/context_gate_v5.json` |
| figure | [`docs/figs/fig15_context_generation.svg`](figs/fig15_context_generation.svg) |
| independent audit and exact hashes | [`docs/context-audit.md`](context-audit.md) |

`runs/` is ignored, so every artifact above is identified by content hash and
rebuilt from the protocol's `builder.reproduce` commands, never copied.
