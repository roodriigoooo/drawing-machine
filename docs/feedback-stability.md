# The recurrent stability probe — what it measures, and what it does not

**Written 2026-08-17; audited through 2026-08-21.** Sections 1–8 are a methods
and qualification-history note about `dm.eval.feedback.stability` and the clauses
that read it; their legacy engineering cells have zero scientific outcome value.
Sections 9–10 contain the terminal F5c evidence and the independent closure
audit. The file remains whole because triangularity is a property of the
operator, and because historical failure interpretation must stay visible after
retraction.

Decision and reason live in [`PLAN.md`](../PLAN.md); this file is the working.

## 1. The iteration is triangular

`stability` iterates the fused prefill: pass `k` fuses pass `k-1`'s normalized
states, shifted right by one, with position 0 held plain.

Position 0 is always plain, so `state_k[0] = state_0[0]` for every `k`. Position
`t` fuses `previous[t-1]` and attends causally over `0..t`, so `state_k[t]` is a
function of `state_{k-1}[0..t-1]`. By induction **`state_k[t]` is final for every
`t < k`** — for any weights, trained or not.

Verified bit-exactly on the smoke checkpoint: at each `k` in `1,2,4,8,16,32` the
first `k` positions are unchanged from pass `k-1` and position `k` onward still
moves. Pinned by `test_the_fused_prefill_converges_one_position_per_pass`.

Three consequences.

**There is no fixed-point question to probe.** The recurrence converges in
exactly `T` passes, one position per pass, never asymptotically. A channel cannot
"fail to have a fixed point" here; it can only blow up on the way, which is what
finiteness, the RMS bands and the logit bound catch.

**`update_q95` at the deepest pass reads sequence length.** It averages exact
zeros from the converged prefix against a live tail. On the smoke's subset — the
16 shortest validation programs, lengths `6,6,25,31,37,46,46,60,66,75,78,79,82,
82,102,175` — twelve rows are fully converged at pass 32 and the clause read
`0.0008`, comfortably inside `<=0.25`. Restricted to positions that could still
move it was `0.5919`. On the real split, 70% of the 1,000 validation programs
exceed 32 bytes and p99 is 148, so the same checkpoint would fail the same clause
there. `update_q95_wavefront` is now reported beside it.

**`update_q95_not_settling` is unreachable, and is now withdrawn.** "Still
receding at the deepest pass" cannot distinguish a bad channel from a long
sequence, by the same argument. The clause was removed from `stability_verdict`
on 2026-08-19 and its reason travels in every verdict as
`contract.WITHDRAWN_CLAUSE_REASONS`, so a reader can see why the failure list is
shorter than v0's rather than wondering. Its replacement,
`update_q95_wavefront_not_settling`, applies the identical arithmetic to the
positions that can still move, which is where a contraction has to be
contracting. The hand-edited test that pinned the old clause was rewritten
against the new one; it still pins arithmetic and still says nothing about
whether the operator can produce that trajectory.

**A deepest-pass average blends two regimes.** The report now carries
`converged_bits_per_byte` and `standard_converged_bits_per_byte`, the fused and
standard costs over the identical converged positions, so the converged part can
be priced without the tail.

## 2. Every RMS quantile was taken over padding — and then so were two more clauses

`stability` masked `update_q95` to non-PAD targets and left the three RMS
quantiles unmasked. In the smoke batch 1,804 of 2,800 positions are padded, and a
padded input position carries the PAD embedding row.

That row is not a typical embedding. The head is tied to the embedding table, so
row 0 is simultaneously the PAD input vector and the PAD output vector, and it is
trained hard as a negative class at every scored position while receiving almost
nothing as an input. After 400 steps its RMS is `0.12291` — 13th largest of 258
rows, against a mean content row of `0.04038`.

`standard_input_rms_p99` was therefore `0.12291`: the PAD row, exactly. The band
`[0.25, 4]x` gave `[0.0307, 0.4916]`, the fused input at `0.0242` fell below it,
and `fused_input_rms_band` failed at every depth. Masked to scored positions the
reference is `0.07328`, the band is `[0.0183, 0.2931]`, and the fused input is
inside it at every depth.

The test covering these quantiles already carried the comment "PAD, so the
quantiles skip padded positions". They did not. Now pinned by
`test_every_rms_quantile_skips_padded_positions`, which holds the scored region
fixed, changes the padded region, and requires every reported quantile not to
move.

**`finite` and `max_abs_logit` were still unmasked, and are not any more
(2026-08-19).** Both are gated clauses. A padded logit row is the tied head
reading its own PAD row as a negative class, so `max_abs_logit` over it measured
how confidently the model rejects PAD; `finite` over it could fail on a state no
loss ever touched. `test_finiteness_and_the_logit_bound_skip_padded_positions`
puts a 500x-scaled embedding row at the padded positions only and requires the
unmasked maximum to move while the reported one does not.

One thing that repair *cannot* do, and the test says so: a non-finite value
cannot be confined to the padded region at all. The head is tied to the embedding
table, so poisoning any row poisons that logit column at every position. Masking
*positions* is the operation that is available and the one that is correct.

## 3. The clause that is actually failing

After the masking fix, one clause fails: pass-32 validation increase `+147.4`
bits/drawing against `+1.0`.

| pass | bits/drawing | converged positions | fused b/byte | standard b/byte, same positions |
|---:|---:|---:|---:|---:|
| 0 | 329.48 | 0 | — | — |
| 1 | 353.05 | 16 | 0.9162 | 0.9162 |
| 2 | 373.33 | 32 | 4.8636 | 3.4526 |
| 4 | 409.24 | 64 | 6.0966 | 3.9188 |
| 8 | 438.73 | 124 | 6.8508 | 4.9439 |
| 16 | 461.30 | 236 | 7.3312 | 5.0523 |
| 32 | 476.87 | 452 | 7.6409 | 5.2113 |

Pass 1's converged region is position 0, which is plain, so the two costs agree
exactly — a consistency check on the split rather than a result.

The cost is not a transient of the tail: on the converged prefix alone the fused
prefill costs `7.64` bits/byte against standard's `5.21` on the same positions,
with `8.01` the byte vocabulary's entropy. The fused fixed point is genuinely
expensive for this checkpoint.

The schedule makes under-training a plausible hypothesis, not an established
diagnosis. It gives `K>=2` to 12.75% of batches and draws the plain prefix
uniform over each row, so fused positions receive far fewer updates than plain
positions. The short-run trajectory therefore cannot identify whether the
likelihood failure is caused by budget, schedule, scale or mechanism.

### The ladder, run 2026-08-17

Four historical engineering cells, data / seed / codec fixed, steps varied. The
old arm's gap remains large at sampled rungs, but this one-seed ladder does not
identify cause and says nothing about source-aligned `glu_source_v1`.

| steps | train bits | conv_fused | conv_std | **gap** | band ratio | gain | embed RMS | g/e |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 400 | 296.7 | 7.105 | 5.228 | **+1.877** | 0.339 | 0.0193 | 0.0404 | 0.48 |
| 2,000 | 229.3 | 7.164 | 4.101 | **+3.062** | 0.284 | 0.0194 | 0.0918 | 0.21 |
| 6,000 | 176.3 | 5.250 | 3.276 | **+1.975** | 0.158 | 0.0214 | 0.1610 | 0.13 |
| 12,000 | 169.5 | 6.430 | 3.128 | **+3.302** | 0.153 | 0.0189 | 0.1923 | 0.10 |

The band ratio crosses its `0.25` floor between 2,000 and 12,000 steps and keeps
falling, so `fused_input_rms_band` fails at F6 scale on the *repaired*
measurement. `g/e` falls monotonically because the embedding grows `9.7x` while
the gain does not move.

**The earlier weight-decay diagnosis is withdrawn.** `zero_grad(set_to_none=True)`
leaves untouched fusion parameters with `grad is None`, and AdamW skips those
parameters rather than decaying them. Under the realized schedule the fusion was
touched on 1,496 of 12,000 steps; applying the learning rates on those steps
predicts approximately `0.9919x`, not `0.835x`. The observed `0.945x` therefore
does not establish gradient cancellation or a stationary point. Per-step
gradient and update accounting is required before diagnosing the gain.

**The claimed `128x` scale sweep is withdrawn.** No persisted report, raw table,
reproducible command or checkpoint hash survives, so it is not an accepted
artifact. Even a recreated post-training global scalar sweep would test only an
inference-time scalar perturbation: it would not test removing RMSNorm during
training, a learned per-dimension gain, normalized gate input, or the
optimization trajectory. The 12,000-step gain is strongly anisotropic
(`mean=.0189`, `std=.0231`, `min=-.0319`, `max=.0825`), which further rules out
using one global scalar to claim that learned normalization does not matter. No
scale conclusion or F7 threshold is supported by this note.

### 3b. What the old arm measured

On the estimand's own region -- a `~15`-byte target after a `47`-byte plain
prompt -- the legacy arm's soft cost exceeded standard at every rung. This is a
reproducible engineering observation for that arm, not a claim about the source
mechanism:

| steps | primary bits/byte, standard | soft | soft − standard | `Delta_std` | `Delta_soft` | `G` |
|---:|---:|---:|---:|---:|---:|---:|
| 400 | 5.2769 | 5.6792 | +0.40 | +0.0015 | +0.0020 | +0.0006 |
| 2,000 | 2.7208 | 3.6517 | +0.93 | +0.1023 | +0.7828 | +0.6805 |
| 6,000 | 0.6383 | 1.2613 | +0.62 | +0.4177 | +0.7284 | +0.3107 |
| 12,000 | 0.5999 | 2.5899 | **+1.99** | +0.2919 | +0.8695 | +0.5776 |

`G` is positive at every rung and 9–12 of 16 cases, which is what the design
predicts: `Delta` subtracts a control measured in the same mode, so a uniform
soft penalty cancels out of `G`. **It means nothing about H3-structure** -- the
relation-destroyed arm is not in this table, and a generic recurrence effect
produces exactly this signature. That is the null F3's scripted "generic soft
improver" exists to catch.

The exact generic guard is different from this target-case diagnostic.
`GENERIC_GUARDS["max_validation_cost_bits_per_drawing"]` applies sequential
teacher-forced soft minus standard over the complete validation split. On the
old 12,000-step engineering checkpoint it is `169.543` versus `395.271`, a
`+225.728` bits/drawing difference against `+1.0`. The earlier `172.5 -> 425.3`
number was a repeated-prefill/target-case cost. The repaired guard fails
decisively, while valid-halt and truncation deltas are now computed with shared
uniforms as separate clauses.

## 3a. The band reference is not yet a valid gate

The masking fix cleared `fused_input_rms_band` at 400 steps, but later audit
found a deeper issue. A learned vector RMSNorm gain is not itself fused-input
RMS. Output RMS depends on gain components **and the normalized product's
direction**; once gain becomes anisotropic, its mean is only a weight summary.
At 12k gain mean is `.0189`, std `.0231`, min `-.0319`, max `.0825`.
`channel_scales` labelled mean-gain/mean-embedding as if it were the band. As of
2026-08-19 it names three things it does not claim — that the gain is the fused
input's RMS, that the ratio is the frozen band, and (under `glu_source_v2`) that
it describes an output scale alone, since the shared norm also scales what `W_G`
reads. Its keys are `shared_input_norm_gain_*` and it carries a `gain_role` that
says which reading applies to the arm in front of it: the same tensor is a
fused-output gain in the two earlier arms and the shared input norm in the
faithful one. It stays diagnostic and nothing gates on it. The actual
fused/standard p99 ratio measured on scored rows is the relevant observed column.

Historical nominal gain against mean content-row embedding RMS—1.0 at
initialization by construction:

| steps | gain | embed row RMS | gain/embed | fused/standard input RMS |
|---:|---:|---:|---:|---:|
| 20 | 0.0199 | 0.0227 | 0.88 | 0.566 |
| 80 | 0.0198 | 0.0270 | 0.74 | 0.475 |
| 400 | 0.0196 | 0.0404 | 0.49 | 0.331 |

Mean gain moved less than embedding RMS, while actual fused/input p99 crossed the
frozen `0.25` floor. Neither establishes channel collapse: denominator growth,
vector-gain anisotropy and input direction all move. F5b therefore records
fused/input scale as diagnostic and does not let it authorize or reject
qualification until a source- or operator-derived reference exists. The 20/80
rows are throwaway code checks.

## 3c. Against the source recipe

`references/Full-bandwidth transformer.pdf` (Wang et al., arXiv 2608.08888).
The legacy `glu_v1` matches the broad value/gate shape and the multi-pass
objective, but it is not a source-replication result. The complete paper's §3.3
stabilization recipe says to apply RMSNorm to the fused input. Listing 3
normalizes the embedding before `W_G` and normalizes the fused output before the
model. Eq. (4)'s compact equation omits those stabilization operations; it is
not evidence that the Listing 3 operations are absent.

First smoke used raw `e_t` in the gate, so its low gate variation and soft
likelihood failure describe legacy implementation only. Proposed arm removing
fused-output RMSNorm is dropped.

**The normalization question is settled, 2026-08-19.** The source PDF's appendix
listing (Fig. 9) is the stabilized recipe; Fig. 2's Listing 1 is the unstabilized
one. It reads:

```text
 8   h = h + uniform(-delta, delta)
 9   x = glu_cross(shift_right(h), input_rmsnorm_1(e))
10   x = prefix_mixin(x, e)
11   h = model(input_rmsnorm_1(x))
```

Lines 9 and 11 name the **same** module. One shared norm, applied to the gate
input and to the post-mixin stack input, with the mixin between them — so
plain-prefix rows are normalized rather than bypassed. Line 5's pass 1
(`h = model(e)`) and Listing 2's standard prefill stay raw, so the asymmetry is
deliberate rather than an omission.

`glu_source_v1` differs on all three counts and stays as it is, loadable and not
qualified: its gate normalization is functional and affine-free rather than the
learned module; its fused product goes through a separately-read learned
`fuse.norm`; and its `torch.where` happens *after* that norm, so plain-prefix rows
bypass the gain entirely. (Two distinct normalization operations, not two
modules — the gate's is inline.) `glu_source_v2` implements the listing and is
the arm F5b qualifies.

The `0.02` gain is coherent under the faithful reading, and more so than before:
with the norm applied to plain positions too, post-mixin plain rows get an
expected embedding-scale RMS, which is exactly what makes a fused pass's plain
prefix match the raw-embedding prefill it exists to imitate. Not an exact
raw-embedding equality — RMS normalization rescales each row — but the right
scale and the right direction. The same `2D^2 + D` tensors suffice, so v0's frozen
counts are untouched.

The paper describing only two extra `D x D` projections is consistent with this:
`input_rmsnorm_1` is the model's existing input norm, not a new parameter.

**The paper chooses the schedule *by running this diagnostic*, and Direction 3
froze an adaptation without running the source diagnostic.** From §3.3: "We
choose the number of passes by checking whether the iterated feedback map reaches
a stable fixed point." Fig. 3 is that check. A 75/25 one-/two-pass mix "performs
well at the trained depth but fails to extrapolate: beyond that depth, validation
loss rises sharply and the hidden-state change oscillates rather than decays";
adding 3% three-pass batches makes the loss flat through 30 feedback steps and
the change decay to a plateau. Direction 3 took the terminal `75/22/3` mix from
the paper at F0, but chose project `50%/25%/25%` phase boundaries; the source does
not specify those boundaries. The project schedule is not a copied source
condition, and other optimizer, LR/cooldown/z-loss, scale/context, batching and
shared-input-norm calibration differences remain material.

Old 12,000-step engineering cell shows a non-contractive wavefront under legacy
arm. It cannot replicate the paper curve: the gate and the normalization
placement both differ from the listing, and the faithful arm has not been
qualified.
Wavefront-restricted update q95, which is analogous to their
`||h^(k) - h^(k-1)||`, is:

| steps | 3-pass batches | 1 | 2 | 4 | 8 | 16 | 32 | verdict |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 400 | 3 | 0.995 | 0.807 | 0.227 | 0.098 | 0.022 | 0.000 | decays |
| 2,000 | 7 | 1.048 | 1.118 | 0.877 | 0.455 | 0.136 | 0.000 | decays |
| 6,000 | 47 | 0.956 | 0.776 | 0.718 | 0.464 | 0.244 | 0.001 | decays |
| 12,000 | 93 | 1.026 | 0.970 | 1.159 | 1.247 | 1.348 | **2.469** | **grows** |

More three-pass batches did not help on this one legacy engineering trajectory.
Treat that as a diagnostic direction and not a result: `n=1` per rung, and two
independent 400-step runs of the identical configuration gave wavefront q95 of
`0.59` and `0.00` at pass 32, so `mps` run-to-run variance at these budgets is
comparable to the effect. It does not authorize a schedule change or a closure
label.

**Scale.** Paper is 1B parameters on 200B tokens; 3% is an enormous absolute
number of three-pass batches. Pilot is 857,600 parameters on roughly 87M tokens;
legacy 12k run saw 93 three-pass batches. Reachability at this budget remains
open until two full-budget source-aligned qualification cells run.

### 3d. Two mechanism observations behind the F5b result (2026-08-19)

Read off the qualification's own development cells. They carry no scientific
outcome value (§7 invariant 16); what they can say is why the mechanism did not
reach viability, which is the question the qualification exists to answer. The
result itself is §6.

**The same cell ran twice and disagreed about stability.**
`terminal_mix_v1 / seed 100`, identical training and measurement code -- the edits
between the two runs touched only the qualification Module, its Adapter, the
freeze Adapter and `provenance.py`, none of which is in the training path or in
`stability`:

| run | baseline | pass 32 | wavefront q95 @32 | exact soft-standard | valid-halt delta |
|---|---:|---:|---:|---:|---:|
| first | 187.3 | 373.4 | **0.0000** | +152.1 | -0.088 |
| second | 186.2 | 494.3 | **1.6061** | +256.6 | -0.465 |

Standard-mode likelihood reproduced to ~0.4% (`161.8` against `161.2`
bits/drawing at 24k). The feedback channel's *qualitative* verdict did not: one
run contracted to numerical zero and the other did not contract at all. §3c
already recorded that `mps` run-to-run variance at 400 steps was comparable to
the effect; this says it is still true at 24,000.

**This is a two-point observation and the first run's artifact no longer exists**
-- it was deleted before the re-run, so these numbers survive only here. It is
not a variance estimate. It is recorded because it bears directly on the
two-seed rule: two seeds are meant to show that a schedule's behaviour is not
seed-specific, and if one seed's behaviour is not run-stable then the rule is
partly measuring nondeterminism. Establishing it costs one repeated cell.

**The fused input arrives well below the scale the trunk reads.** Off the three
finished checkpoints:

| cell | norm gain mean | embed row RMS | fused input p99 | standard input p99 | ratio |
|---|---:|---:|---:|---:|---:|
| `terminal_mix_v1` s100 | 0.0687 | 0.2488 | 0.0736 | 0.3841 | 0.192 |
| `terminal_mix_v1` s101 | 0.0871 | 0.2307 | 0.0887 | 0.2688 | 0.330 |
| `project_progressive_v1` s100 | 0.0968 | 0.2371 | 0.0972 | 0.3340 | 0.291 |

`FUSED_NORM_GAIN = 0.02` matches the embedding's RMS *at initialisation*. The
fusion is switched on at the midpoint, by which time the embedding has grown
about 12x, so the channel starts an order of magnitude below the input scale the
trunk spent 12,000 steps learning to read. The gain is climbing to close that --
`0.02` to `0.07-0.10` -- and runs out of budget: it takes gradient only on the
`K>=2` batches, about 3,000 of 24,000 steps. Fused input RMS tracks the gain
vector's own RMS (`0.0715` predicted against `0.0736` measured on s100), and it
would need roughly `0.33`.

`W_G` grew about 9x from initialisation while `W_U` grew about 1.8x. Weight
tying is meant to spare `W_U` a large corrective rotation between the embedding
and hidden-state spaces, and for *basis* it does; the two distributions still
differ about 7x in **scale** (embed row RMS `0.24`, carried state RMS `1.7`), and
that part `W_U` did not absorb.

The same measurement contradicts a rationale inside v0's frozen payload.
`JITTER_HALF_WIDTH = 0.02` is documented as matching "the size of the signal's
own initial scale", but the signal it perturbs is the carried state at RMS `~1.7`,
so the perturbation is about **1.1% relative** -- far short of the source's stated
purpose of exposing the map to a local neighbourhood around each training state.
The number is frozen and stays frozen; the rationale is wrong and belongs in the
source-condition ledger as a material difference the next package records.

**Neither observation authorises anything.** No threshold moves on a run that
failed it, and the levers these point at -- the gain's calibration reference and
the jitter's scale -- are both inside v0's payload, so they belong to a later
arm layered over v0 in the way `glu_source_v2` was, not to a mid-package patch.

## 4. Repair verification and the residual defects, now closed

F5a did not mutate v0: `scripts/feedback_contract.py --verify` still reports
`e67d8626887b` and five baseline cells replay bit-for-bit. Exact generic
likelihood, valid-halt/truncation guards and `max_valid_halt_loss` now execute.
Sampler/eval RNG streams are isolated. Smoke record/report paths and hashes are
separate; failed reports cannot freeze.

Five defects were listed as blocking on 2026-08-18. All five are closed as of
2026-08-19, and none of them moved a threshold:

- `finite` and `max_abs_logit` are scored-position quantities, pinned by a batch
  whose padded region carries a 500x-scaled embedding row;
- `stability_subset` draws a frozen model-blind subset from
  `seed_for("stability")` over programs longer than the deepest pass, persists
  indices, lengths and a digest, and fails closed rather than shrinking. Both the
  smoke and the qualification Adapter use it;
- `converged_bits_per_byte` divides by `positions / symbols_per_byte`, so the bit
  arm aggregates eight symbols to one bytecode byte, and the report names its
  `cost_unit`;
- `update_q95_not_settling` is withdrawn, with its reason carried in every
  verdict; the gate reads `update_q95_wavefront` at the frozen `0.25` and against
  the pass-8 value;
- the fused/input RMS band remains diagnostic and is listed in
  `QUALIFICATION_GATE["diagnostic_only_clauses"]`, so the qualification reports it
  and does not gate on it.

These were pre-outcome implementation and criterion repairs, and they live in a
qualification layer over v0 rather than inside it. `REPORT_SCHEMA` is part of v0's
frozen payload and so cannot move to describe a report v0 did not define; the
repaired shapes carry `STABILITY_REPORT_SCHEMA = 2` and
`QUALIFICATION_SCHEMA = 1` instead. v0 still verifies at `e67d8626887b`, and no
threshold was inferred from new checkpoint output.

## 5. F5b qualification package, as it was run

The Module and its thin Adapter: `dm/eval/feedback_qualification.py` and
`scripts/feedback_qualify.py`. Fixed model-blind long-row selection,
scored-position masks, semantic-byte accounting and the wavefront verdict were
implemented and tested before any cell ran. Qualification uses only generic
validation/termination, convergence, resources and stability; no Direction 2
cases, `Delta`, `G` or relation interaction.

**The six cells ran on 2026-08-19 and the package stopped** (§6). §7 records what
a later audit found wrong with the *instrument* that produced that stop, and §8
the single correction package the stop is allowed to buy.

Run three frozen byte/relational schedules × seeds `100/101` (six 24k cells),
`data_seed=100`, full `100,000/1,000` scale, byte-disjoint from final data and
Direction 2 cases: eligible `terminal_mix_v1` (first half one-pass, second half
`75/22/3`), eligible current `project_progressive_v1`, and diagnostic-only
`two_pass_control_v1` (first half one-pass, second half `75/25/0`). Each eligible
schedule qualifies only if both seeds pass:

- exact soft-standard validation `<=+1.0` bits/drawing;
- valid-halt delta `>=-0.05`; truncation increase `<=0`;
- scored finite and max logit `<100`;
- hidden RMS p99 in `[0.25,4]x` pass-0;
- wavefront q95 at pass 32 `<=0.25` and `<=` pass 8;
- pass-32 validation increase `<=+1.0` bits/drawing;
- final-third `|tail| <=0.5` bits/drawing/1k and final-best drift `<=1.0`;
- complete provenance, strict reload, hash and resource reconciliation.

Fused/input p99 and gain anisotropy stay diagnostic. Freeze first listed eligible
both-seed pass; control cannot authorize v1. No eligible pass yields
`unstable_feedback` or bounded `no_viable_feedback_implementation_at_scale`,
never `no_feedback_gain`; mixed seed fails that schedule.

## 6. The F5b result — the package ran, and it stops

**2026-08-19.** Six full-budget development cells: three frozen schedules ×
seeds `100/101`, `data_seed=100`, `100,000/1,000` programs, `24,000` steps,
`857,600` parameters, arm `glu_source_v2`, byte codec, relational arm only, all
on `mps`. **No eligible schedule passed both seeds. Label: `unstable_feedback`.
Protocol v1 is not frozen and F6 is not authorized.**

| cell | k=0 | k=1 | k=8 | k=32 | wavefront q95 @32 | @8 | hidden RMS ratio | max logit | exact soft−std | valid-halt Δ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `project_progressive_v1` s100 | 191.2 | 197.3 | 218.5 | 217.9 | **0.0000** | 0.108 | 0.984 | 32.4 | +48.6 | −0.273 |
| `project_progressive_v1` s101 | 187.3 | 194.5 | 225.1 | 234.3 | **0.1411** | 0.140 | 0.981 | 29.5 | +62.8 | −0.270 |
| `terminal_mix_v1` s100 | 186.2 | 196.4 | 401.6 | 494.3 | 1.6061 | 1.292 | 0.872 | 35.4 | +256.6 | −0.465 |
| `terminal_mix_v1` s101 | 187.9 | 200.8 | 280.0 | 342.1 | **0.0831** | 0.425 | 0.928 | 29.4 | +136.7 | −0.098 |
| `two_pass_control_v1` s100 | 185.8 | 202.2 | 454.5 | 578.1 | 1.8814 | 1.284 | 0.893 | 33.7 | +256.3 | −0.228 |
| `two_pass_control_v1` s101 | 188.2 | 209.9 | 519.1 | 626.7 | 1.4799 | 1.929 | 0.849 | 56.9 | +351.0 | −0.661 |

Bold marks a wavefront inside the frozen `0.25` bound. Bits/drawing columns are
the fused prefill at depth `k`; `k=0` is the standard forward on the same rows.

Every cell failed on **both** `generic_guards` and `stability`. Which stability
sub-clauses fired:

| cell | validation increase | valid-halt loss | wavefront bound | wavefront not settling |
|---|:--:|:--:|:--:|:--:|
| `project_progressive_v1` s100 | ✗ | ✗ | | |
| `project_progressive_v1` s101 | ✗ | ✗ | | ✗ |
| `terminal_mix_v1` s100 | ✗ | ✗ | ✗ | ✗ |
| `terminal_mix_v1` s101 | ✗ | ✗ | | |
| `two_pass_control_v1` s100 | ✗ | ✗ | ✗ | ✗ |
| `two_pass_control_v1` s101 | ✗ | ✗ | ✗ | |

Every other clause passed in every cell: completeness, development provenance,
record fields, architecture, schedule, seeds, budget, parameters, truncation,
pass accounting, hashes, strict reload, validation tail and final-minus-best
drift. The instrument found nothing wrong with the runs; it found the mechanism.

### 6.1 What replicated, and it is a direction rather than a result

The paper chooses its schedule by running this diagnostic, and reports that a
75/25 one-/two-pass mixture "performs well at the trained depth but fails to
extrapolate", while adding 3% three-pass batches makes validation loss flat
through 30 feedback steps and the hidden-state change decay to a plateau.

**Something ordering-shaped appears here, at roughly a thousandth of the
parameter count and a two-thousandth of the token budget, and it is narrower than
this section first claimed.** The defensible statement is one pair of columns:
**`project_progressive_v1` has the lowest repeated-prefill loss at `k=8` and at
`k=32` in both seeds, and `two_pass_control_v1` has the highest in both.**
`218.5 / 225.1` against `454.5 / 519.1` at `k=8`, and `217.9 / 234.3` against
`578.1 / 626.7` at `k=32`. `terminal_mix_v1`, which introduces both pass counts
at once at the midpoint, sits between them on both, and is the cell that is not
reproducible (§3d).

**The stronger claim this section carried until 2026-08-20 is withdrawn.** It
said the control was worst and the progressive schedule best "on every column",
which the table directly above it contradicts on four counts:

- seed 100 valid-halt delta is better for the control (`−0.228`) than for
  `project_progressive_v1` (`−0.273`);
- seed 100 `k=0` and `k=1` are not best for `project_progressive_v1` — it is the
  worst of the three at `k=0` (`191.2` against `185.8` and `186.2`);
- seed 101 wavefront q95 is better for `terminal_mix_v1` (`0.0831`) than for
  `project_progressive_v1` (`0.1411`);
- seed 100's exact soft−standard penalty is marginally worse for
  `terminal_mix_v1` (`+256.6`) than for the control (`+256.3`).

The correction matters beyond tidiness: "every column" is what would make this
look like a replication of the source's Fig. 3, and the two columns that survive
are a *diagnostic direction* on one axis, which is a much weaker thing.

**Treat even the surviving statement as a direction.** There is `n=1` per cell,
no replicate within a cell, and §3d shows one cell's wavefront verdict flipping
between two runs of the identical configuration. It also cannot select anything:
the qualification's rule is first-predeclared-eligible passing both seeds, and
neither eligible schedule passed.

### 6.2 What did not replicate

The absolute level. Even the best cell costs `+48.6` bits/drawing on the exact
sequential soft-minus-standard guard against a budget of `+1.0`, and loses 27
points of valid-halt rate against a budget of 5. The best-behaved map converges,
holds its scale and stays bounded — and reads the corpus far worse than the same
weights do with the channel off.

This is **not the paper's failure mode.** Their green curve is fine at the
trained depth and diverges beyond it; every cell here is already `+6` to `+22`
bits at `k=1`, the trained depth, before any extrapolation. More three-pass
batches is the remedy for an extrapolation failure, and this is not one. §3d
gives the mechanism the measurements support: the fused input arrives 3–5× below
the input scale the trunk spent 24,000 steps learning to read, because the norm
gain is calibrated to the embedding's *initial* RMS and the channel is switched
on at the midpoint, by which time the embedding has grown about 12×.

### 6.3 What the label means, and what it does not

`unstable_feedback` is the frozen rule's output, applied mechanically: the label
rule was written and tested before any cell ran, and it selects
`unstable_feedback` whenever a stability clause fails anywhere. It is also
substantively right here — three of six cells miss the wavefront bound outright.

It says: **this implementation, at this scale, under these three predeclared
schedules, did not produce a feedback map that is both recurrently stable and
generically harmless.** It does not say latent feedback fails, does not evaluate
the source mechanism at the source's scale, and carries no relation claim
whatever — `G`, `Delta`, `I_structure` and Direction 2's cases were never
computed and are structurally unreachable from this stage. `no_feedback_gain`
requires complete stable pilot scores and remains unreachable.

### 6.4 Artifact chain

`runs/feedback_qualification.json`, SHA-256
`f0f68681fb980828d4c61a74b0021703bafc9a0f662758da86bbec3140f40f21`. All six cell
reports verified against their own canonical payload digests in a later process;
all six recorded the same corpus payload digest `8da5b8fb…`, the same protocol v0
digest `e67d8626…` and the same combined source digest `dfd9b9ac…`. Checkpoints
and records are in `runs/` under `feedback_qual_<schedule>_s<seed>`.

## 7. What the audit found wrong with the instrument (2026-08-20)

The F5b result stands: six full-budget cells ran, every eligible schedule failed
`generic_guards` and `stability`, and the label the frozen rule produced is
`unstable_feedback`. What a later audit found is that **the instrument that
produced it was weaker than §5 and §6 claimed**, on five counts. None of them
changes the six cells' numbers. Two of them change what "every other clause
passed" is worth, and all five are repaired before anything else runs.

### 7.1 The corpus clause passed against the wrong corpus

All six cells trained on `data_seed=100`, corpus fingerprint
`c4d24e9dc75fecd3 / 6ebe4c3b5c1bdfae`. Their recorded `hashes["corpus"]` instead
names the audited **`data_seed=0`** manifest, fingerprint
`e092df3b0eab292f / 4189b240a89d2405`.

The mechanism is concrete and was not subtle in hindsight. `train` rebuilds its
corpus from the config *before* `--corpus` is looked at, and the argument is then
only hashed: the Adapter wrote the manifest's canonical digest into `recorded`
and a recomputation of the *same manifest* into `recomputed`, and the clause
found them equal — as it always would. Nothing ever compared either to
`record["corpus"]`, which is the only statement in the artifact about which
programs a checkpoint actually saw.

This is very unlikely to explain failures of `+48.6` to `+351.0` bits/drawing:
both corpora are the same generator at the same scale under different seeds. What
it does invalidate is the sentence "every other clause passed in every cell". The
corpus clause did not pass; it was not a clause.

**Repaired.** `dm.eval.feedback_evidence.corpus_identity` reconciles the
manifest's per-arm fingerprint against the record's, exactly, and reports the
`data_seed` and split sizes beside it; `qualify_cell` gates on it. Run against the
existing artifacts it reproduces the defect in five lines, which is how the
repair was checked.

### 7.2 The source digest omitted training-critical files

The Adapter hashed eight files. The contract's authoritative
`QUALIFICATION_SOURCES` additionally names `dm/data/dataset.py`,
`dm/data/synthetic.py`, `dm/isa/codec.py` and `dm/isa/unroll.py` — batching, the
corpus bytes and the encoding. Two lists is not two provenances; it is one, and
the shorter one wins silently.

**Repaired.** The Adapter has no list. `source_identity()` is the only reader of
the contract's set, and a test asserts `scripts.feedback_qualify` has no
`QUALIFICATION_SOURCES` attribute at all.

### 7.3 "Structurally blind" was true of the artifacts, not guaranteed

`qualify_cell` rejected an unexpected *top-level* key and accepted arbitrary
nesting inside `record`, `generic_guards` or `stability`. The artifacts contain
no outcome, so the factual claim survives; the structural guarantee did not.

**Repaired.** `dm.eval.feedback_evidence.CELL_SHAPE` declares the closed key set
at every level a cell is assembled from, `shape_faults` walks it, and a `Delta`
two levels down costs an error. `record["config"]`'s allowed keys are *derived*
from `TrainConfig`'s dataclass fields, and `RECORD_FIELDS` is pinned against a
real one-step feedback record, so the declaration cannot drift from the thing it
describes.

### 7.4 The schedule-ordering sentence was false as written

Corrected in §6.1 above, against the table it contradicted.

### 7.5 Archival status, and the disjointness claim

The F5b implementation was uncommitted while all its artifacts sat under ignored
`runs/`, so no repository revision contained the exact qualification code, and
the deleted first `terminal_mix_v1 / seed 100` artifact (§3d) cannot be
independently audited at all. Committing the snapshot fixes the first; nothing
fixes the second, and §3d says so.

`docs/feedback-artifacts.json` is the missing half: one committed file naming
every Direction 3 artifact under `runs/` by size and SHA-256. It is not a copy
and not a re-verification — a cell report still verifies against its own
canonical payload digest — but an auditor holding only the repository can now see
what was claimed to exist.

**One consequence of the uncommitted state is permanent, and the archive says so
rather than hiding it.** The F5b cell reports name the exact bytes that trained
and judged them; those bytes were never committed, and repairing the instrument
overwrote five of the eight files in place. The archive's
`source_reconciliation` block records the comparison file by file, computed
rather than asserted: **3 of 8 still reproduce their recorded digest**
(`dm/data/feedback.py`, `dm/data/feedback_audit.py`, `dm/eval/feedback.py`) and
five do not (`feedback_contract.py`, `feedback_qualification.py`,
`transformer.py`, `train.py`, `train_feedback.py`). The F5b instrument is
therefore identified by hash and is **not recoverable as a revision**. That is
the cost of the defect, it is not repairable after the fact, and it is the reason
every commit from here carries the code before the cells rather than after.

**One further claim is corrected here.** §5 and the plan said the development
corpus was "byte-disjoint from final data and Direction 2 cases". Measured, it is
not disjoint and never was: the `data_seed=200` corpus shares **4 programs of
101,000** with the pilot's `data_seed=0` corpus and **4** with F5b's
`data_seed=100`, and all eight are 6-byte minimum-length programs — one `MOVE`,
one `LINE`, `HALT` — where the coordinate space is small enough that a collision
at 100,000 draws is expected rather than surprising. The honest statement is
*0.004% overlap concentrated entirely in the shortest programs the grammar can
produce*, not disjointness.

### 7.6 The scale diagnosis remains plausible, not established

§3d's measurement stands and §6.2 reads it correctly as the mechanism the
evidence supports. It is not isolated. Across all six checkpoints fused/standard
input p99 is `0.19–0.33`; the learned shared gain ends at `0.069–0.097` against an
embedding-row RMS of `0.230–0.249`. But the cell with the *largest* ratio still
carries a `+136.7` soft penalty, and optimizer and parameter groups, cooldown,
z-loss, batching, depth scaling and the tiny tied vocabulary all differ from the
source as well. §8 changes the one lever the measurement points at and predeclares
what will count as an answer; it does not assert the diagnosis in advance.

## 8. The F5c correction package

**One package, predeclared, terminal.** F5b's decision rule said stop and report,
and offered exactly two continuations: close Direction 3 on the bounded negative,
or run one further bounded package addressing the measured mechanism. This is
that package, and its criteria are frozen in
`dm.eval.feedback_contract`'s F5c layer before any cell runs.

### 8.1 One lever

**The arm does not change.** `glu_source_v2` stays the qualified seam and there is
no `glu_source_v3`. Gain calibration is *training behaviour*, not a forward
equation, and a fourth arm with the same equation and the same three tensor names
would add a schema branch that distinguishes nothing a reader could check —
while joining a set of arms whose checkpoints already load `strict=True` into one
another and compute different functions in silence. What changes is a named
**training-protocol condition**, `TrainConfig.gain_calibration`, recorded in the
config, in the accounting and in the freeze.

**The rule.** At the predeclared feedback phase transition — the first step whose
phase can draw more than one pass, derived from the schedule rather than written
down, `12,000` of `24,000` for all three project schedules — the shared input
norm's gain is set once, in every component, to

```text
g = sqrt(sum_v p_v * mean_d E[v, d]^2)
```

where `E` is the token embedding table at that step and `p` is the token
frequency of the training split's **input** positions, BOS included and PAD
excluded.

**Why this statistic.** `Fusion.stack_input` returns `N(m)`, so a gain vector
filled with `g` hands the trunk rows at RMS exactly `g` (to a few parts in ten
thousand — `RMSNorm` adds `eps` under its reciprocal square root). Setting `g` to
the frequency-weighted RMS of the raw embeddings therefore makes a fused pass's
*plain prefix* the size of the raw-embedding prefill it exists to imitate. That
is precisely the rationale the contract already gives for the `0.02`
initialisation, evaluated at the moment the channel actually switches on rather
than at step 0. Frequency-weighted rather than a plain row mean because the trunk
meets tokens at their corpus frequency, not uniformly over the vocabulary; on a
200-step engineering smoke the two differ by 23% (`0.0398` against `0.0323`).
PAD is excluded because it is the tied head's negative class and never a content
input; BOS is included because it is a real input position.

**Model-blind in the sense the contract means.** A deterministic function of the
embedding table and the corpus histogram. It reads no loss, no validation score,
no sample, no Direction 2 case, no `Delta` and no `G`, and `calibrate_gain` has
no parameter through which one could arrive.

**The jitter does not move.** The source states a fixed `0.02` and specifies no
state-relative rule, so changing it would be a second post-hoc intervention *and*
a source departure rather than a correction. §3d's finding stands — the
perturbation is ~1.1% of the carried state it perturbs, not "the size of the
signal's own initial scale" as v0's payload says — and it is recorded as a
`carried_state_jitter_scale` row in the source-condition ledger, status
`source_value_project_rationale_retracted`. The number is the source's and stays.

**No threshold moves.** `CORRECTION_GATE["thresholds_moved"]` is empty and a test
pins it, along with `FUSED_NORM_GAIN == JITTER_HALF_WIDTH == 0.02`.

### 8.2 One schedule, fresh data, three cells

| what | value | why |
|---|---|---|
| schedule | `project_progressive_v1` | one, predeclared, no comparison |
| data seed | `200` | fresh; F5b's cells may motivate this package and may not qualify it |
| model seeds | `200`, `201` | disjoint from the pilot's `0/1` and F5b's `100/101` |
| replicates | seed 200 twice, seed 201 once | the repeat is a reproducibility control, not a second sample |
| cells | 3, no retry | more than one such package is the architecture search the pilot exists to prevent |

`project_progressive_v1` is chosen as the project's standing default, not as
F5b's best: §6.1's surviving observation is a diagnostic direction on two columns
and explicitly cannot select anything.

The corpus is built and audited before any cell runs, at
`runs/feedback_corpus_f5c_audited.json`, canonical payload digest
`81c44ed6fa20295d…`, file SHA-256 `1fa1320426498406…`. It carries
`29,147/100,000` treated programs, all three VM arms valid, halted and
zero-fault, and its relational fingerprint `6c065df5bef1fcd2 / 25bb986123ca052e`
is the fingerprint `training_config_correction` actually builds — verified, not
assumed, which is §7.1's whole point.

### 8.3 Reproducibility, and the terminal rule

Every cell asks torch for deterministic kernels and records the resulting state,
along with Python, Torch, OS, device, thread counts and the environment variables
that change numerics without changing code. F5b recorded none of this, which is
why §3d's two disagreeing runs of one configuration cannot be attributed to
anything.

**The repeated seed's two cells must carry one checkpoint SHA-256.** If they do
not, the label is `incomplete_nondeterministic`: two runs of one configuration
that disagree have measured the machine, not the mechanism, and the honest next
step is a deterministic device or a stop.

**Post-closure audit:** this frozen comparator is invalid for cross-run state
identity because it hashes filename-dependent `torch.save` containers carrying
different `record_name` values. Section 10 preserves the historical rule while
retracting its evidentiary reading.

**Then the terminal rule, and it is the package.**

- Any reproducibility failure, or any gate failure: **Direction 3 closes
  permanently at this scale.**
- All three cells pass every existing generic and stability clause: freeze a new
  training protocol, and only then authorize F6.
- No further gain choices, jitter variants, schedules or retries.

### 8.4 The instrument, deepened

`dm/eval/feedback_evidence.py` is new and owns what F5b had spread across the
Adapter and the contract: corpus identity, source identity, execution
environment, checkpoint reconstruction, the declared cell shape and report
validation. That consolidation is the audit's central recommendation, and the
reason for it is §7.1 — an invariant that lives in two places is an invariant
nobody owns, which is exactly how a fully green test suite missed a corpus
mismatch. The Adapter now reads arguments, trains, measures and writes files; the
gate applies clauses to what it is handed; and *what a cell is* has one home.

`qualify_cell` is one function judging both packages, parameterised by a frozen
`Package`. An F5c cell is judged by exactly the instrument that judged F5b, plus
the two clauses F5b was missing.

**The six F5b cell reports are not rewritten and not re-decided.** Under the
repaired instrument they would be refused — for want of corpus identity and of an
environment block — and that refusal is §7.1 restated, not a new result. The
recorded stop is what the frozen rule produced on the day, and it stands.

## 9. The F5c result — the lever moved, and Direction 3 closes

**2026-08-20.** Three full-budget cells on `project_progressive_v1`, `data_seed=200`,
model seeds `200/201` with seed 200 run twice, `857,600` parameters, `24,000`
steps, arm `glu_source_v2` under `gain_calibration="embedding_rms_at_switch_on_v1"`,
all on `mps` with deterministic algorithms requested and enabled.

**No cell passed.** The historical decision artifact emitted
`incomplete_nondeterministic`; §10 audits and rejects its checkpoint-file
comparator while independently confirming tensor-state divergence.
`runs/feedback_correction.json`, SHA-256
`757764a47c355b8c489b8fc604cd193d7d8d56a58fa7fbad19f54ba11fcc727b`.
The valid gate-failure half of the terminal rule fired in all three cells, so
**Direction 3 closes permanently at this scale.**

| cell | guard | k32−k0 | wavefront @32 | valid-halt Δ | fused/std p99 | g/e |
|---|---:|---:|---:|---:|---:|---:|
| F5b s100 (uncalibrated) | +48.6 | 26.8 | **0.0000** | −0.273 | 0.291 | 0.408 |
| F5b s101 (uncalibrated) | +62.8 | 47.1 | **0.1411** | −0.270 | 0.271 | 0.390 |
| F5c s200 r1 | +68.1 | 100.1 | 0.6305 | −0.488 | 0.735 | 0.968 |
| F5c s200 r2 | +59.4 | 152.7 | 0.4116 | −0.244 | 0.770 | 1.003 |
| F5c s201 r1 | +24.3 | 107.1 | 1.2218 | −0.123 | 0.804 | 1.009 |

`guard` is the exact sequential soft−standard validation cost in bits/drawing
against a budget of `+1.0`; `k32−k0` is the repeated-prefill penalty at pass 32
over the same rows; `g/e` is the shared gain over the mean content-row embedding
RMS. Bold marks a wavefront inside the frozen `0.25` bound. Drawn as
[`figs/fig16_feedback_calibration.svg`](figs/fig16_feedback_calibration.svg).

### 9.1 The lever moved, by design and by 2.7×

The calibration fired at step `12,000` in all three cells, taking the shared gain
from `0.0200` to `0.1810–0.1834`, and it **held**: the gain ends at `0.229–0.243`
against an embedding-row RMS of `0.237–0.241`, so `g/e` finishes at `0.97–1.01`
where F5b's finished at `0.39–0.41`. The fused input now arrives at the scale the
trunk reads — `0.735–0.804` of the standard input p99, against `0.271–0.291`
before. §3d's measurement named this quantity, the package targeted it, and it
moved by a factor of 2.7 with nothing else changed.

### 9.2 And the outcome did not follow it

**The exact guard did not come down.** `+24.3 / +59.4 / +68.1` against F5b's
`+48.6 / +62.8`, on a budget of `+1.0`. Overlapping ranges, and the best cell is
still 24× over budget.

**The recurrence got worse.** `k32−k0` is a within-checkpoint difference, so the
two packages' different validation splits cancel and it is the one column on
which they may be compared directly. It goes from `26.8 / 47.1` to
`100.1 / 107.1 / 152.7`, with no overlap. The wavefront leaves the frozen `0.25`
bound in all three cells having been inside it in both of F5b's.

**The most economical reading, offered as a reading.** The small gain may have
been keeping an expensive channel quiet rather than suppressing a useful one.
At `g/e ≈ 0.4` the fused input was small enough that the trunk could largely
ignore it; after calibration the channel was audible and still failed every
absolute gate.

The result rejects **this gain-calibration remedy as a route to qualification**.
That is narrower than falsifying input scale as a causal explanation. F5b and
F5c deliberately use fresh seeds and corpora, so they are not a paired ablation;
the shared norm also controls both the gate input and the post-mixin stack input.
The larger repeated-prefill penalty is therefore a consistent diagnostic
direction, not an isolated calibration effect. What remains unexplained is why
the fused map is expensive at all, and §7.6's material differences from the
source — optimizer and parameter groups, cooldown, z-loss, batching, depth
scaling, model depth/width/context and the tiny tied vocabulary — remain open.

### 9.3 What the repeat showed

`docs/feedback-stability.md` §3d recorded one cell's stability verdict flipping
between two runs of the identical configuration and could not establish it: the
first artifact had been deleted. The repeat establishes it.

**Two runs of one configuration, with deterministic algorithms requested and
enabled (`deterministic_algorithms: true` in both recorded environments), on
`mps`, diverge from the first evaluation.** “Enabled” records a PyTorch policy;
it is not an end-to-end certificate that every backend operation reproduces.

| step | r1 | r2 | Δ |
|---:|---:|---:|---:|
| 2,000 | 186.70 | 186.20 | +0.51 |
| 4,000 | 174.20 | 178.26 | −4.06 |
| 12,000 | 166.55 | 169.24 | −2.69 |
| 24,000 | 162.55 | 164.65 | −2.09 |

The divergence begins at the first eval, **before** the calibration fires at
12,000, so it is not the lever. Downstream: `8.7` bits/drawing between the two
cells' exact guards, `52.6` bits/drawing between their `k32−k0`, and `1.53×`
between their wavefronts.

Three consequences follow, qualified by the comparator audit in §10.

**The gate-based closure is not marginal.** The guard misses its budget by 23 to
67 bits/drawing. The two repeated cells differ by `8.656`, which is one observed
pair difference rather than a variance estimate or bound. More directly, the
smallest guard is `+24.317`; over its persisted 1,000 paired per-program values,
the descriptive standard error is `2.326` and a normal-approximation 95% interval
is `[19.76, 28.88]`, still far above `+1.0`. This post-hoc interval does not move
the frozen rule; it shows that ordinary validation-sample error cannot explain
the miss. The wavefront clause alone is less decisive and is not needed for
closure.

**§9.2's degradation is suggestive rather than settled.** The `k32−k0`
separation between the packages (F5b's worst `47.1` against F5c's best `100.1`,
a gap of `53.0`) is about one replicate-spread (`52.6`). The direction is
consistent across all five cells; the margin is one measurement wide. It is
reported as a direction, which is the same discipline §6.1 was corrected into.

**This pair is numerically non-repeatable on the recorded MPS stack despite the
deterministic-algorithms policy being enabled.** The `8.656`-bit guard difference
is not a general MPS noise floor: two replicates cannot estimate a distribution,
and another workload or backend version may behave differently. It is enough to
disqualify single-cell precision claims smaller than that observed difference
for this configuration until the measurement platform is qualified.

### 9.4 What the label means, and what closes

`incomplete_nondeterministic` is the historical rule's emitted output. Section
10 shows that its whole-checkpoint-file comparator is invalid by construction,
so the label cannot be carried forward as a valid frozen reproducibility verdict.
The immutable result artifact is not rewritten; its reviewed interpretation is
“closed by independent gate failure; reproducibility comparator invalid; model
states observed to diverge in a post-closure content audit.”

The terminal rule was written before any cell ran and it is applied mechanically:
any reproducibility failure or any gate failure closes Direction 3 permanently at
this scale. The valid gate-failure branch fired in all three cells. **No further
gain choices, jitter variants, schedules or retries.** There is no protocol v1,
F6 is not authorized, and `G`, `Delta`,
`I_structure` and `I_representation` were never computed and now never will be
under this pilot.

What Direction 3 carries forward is a **bounded negative about one implementation
and one correction package at one scale**. The calibrated-gain remedy failed to
qualify; the broader scale explanation is not isolated. This is not a claim
about latent feedback, not an evaluation of the source mechanism at the source's
scale, and not a relation result of any kind.

### 9.5 Artifact chain

`runs/feedback_correction.json`, SHA-256
`757764a47c355b8c489b8fc604cd193d7d8d56a58fa7fbad19f54ba11fcc727b`. Three cell
reports, each verified against its own canonical payload digest in a later
process, each reconciled against the audited `data_seed=200` corpus manifest
(canonical `81c44ed6…`, file `1fa13204…`) by exact fingerprint equality — the
check F5b did not have. Checkpoints under
`runs/feedback_f5c_project_progressive_v1_s<seed>_r<replicate>`; every artifact's
SHA-256 is in [`feedback-artifacts.json`](feedback-artifacts.json).

## 10. Independent closure audit — decision retained, rationale corrected

**Reviewed 2026-08-21. Verdict: close Direction 3 at this scale.** The terminal
rule was committed in `f1c21ff` before the cells and says that *any* gate failure
is terminal. All three F5c cells failed `generic_guards` and `stability`, so that
branch is sufficient. Declining closure would reopen a bounded package after the
predeclared condition fired.

The review does **not** endorse every sentence in the original closure.

### 10.1 The checkpoint-byte reproducibility clause is invalid

F5c compared SHA-256 of the entire `.pt` file. That cannot answer whether two
learned model states are identical:

1. `torch.save` writes a ZIP container whose member names depend on the output
   filename. Saving the same object to `a.pt` and `b.pt` produces different
   bytes.
2. `dm.train.checkpoint` embeds `record_name` in development checkpoints, and the
   two replicates deliberately use different names ending in `_r1` and `_r2`.
3. The test that appeared to cover repeatability wrote arbitrary byte files whose
   contents depended only on seed; it did not exercise two real `torch.save`
   checkpoints.

Therefore a whole-file mismatch was guaranteed even if every tensor reproduced.
Artifact file SHA-256 remains the right identity/tamper check; it is the wrong
cross-run state comparator. The historical decision report stays immutable, but
its `incomplete_nondeterministic` label is not accepted without this caveat.

`dm.eval.feedback_evidence.state_dict_sha256` now hashes sorted tensor name,
dtype, shape and raw contiguous CPU content while excluding the container and run
metadata. A regression test saves the same state under different filenames and
`record_name` values, requires equal state digests, then mutates one tensor and
requires inequality.

### 10.2 The saved model states nevertheless did diverge

The invalid comparator produced the right qualitative diagnosis for the wrong
reason. A content audit of the two seed-200 checkpoints finds equal model configs,
34 of 34 tensors unequal, maximum absolute element difference `3.907`, and these
state digests:

| cell | canonical model-state SHA-256 |
|---|---|
| s200 r1 | `4b93b425296ead1a78d9cf7f12fd19a710874ea2283660de0eb8264e57dff1bb` |
| s200 r2 | `5b1c8e2d3bcfb65f0756c6245a95620b508c3817c57ed1443932175943df2dc` |

Their training configs differ only in the output tag, their deterministic pass
histograms and corpus identity agree, and their validation trajectories diverge
at the first evaluation. This establishes numerical non-repeatability for this
pair. It does not identify which MPS operator caused it and does not estimate a
general noise distribution.

### 10.3 Why closure still follows

Remove the entire reproducibility branch and the result is unchanged:

- every F5c cell fails exact soft-minus-standard likelihood by `+24.317` to
  `+68.103` against `+1.0`;
- every cell loses valid-halt rate by `0.123` to `0.488` against an allowed
  `0.05`;
- every cell fails pass-32 likelihood and wavefront stability bounds.

These are three independently evaluated cells under the repaired corpus/source/
environment instrument, not one checkpoint-byte comparison. The narrow reviewed
claim is:

> `glu_source_v2`, at 857,600 parameters and 24,000 steps, under the three F5b
> schedules and the one allowed gain-calibration correction, did not produce a
> feedback path that was both generically harmless and recurrently stable.

“Four schedules” is inaccurate: F5b used three distinct schedules; F5c repeated
`project_progressive_v1` under a different gain-calibration condition.

### 10.4 What is rejected and what remains open

The calibration intervention worked mechanically and failed its absolute
qualification target. That rejects `embedding_rms_at_switch_on_v1` as the
permitted remedy. It does not isolate “input scale” as a universal cause: F5b and
F5c use different seeds/corpora, the shared norm also changes the gate operating
point, and the source-condition ledger remains incomplete.

The source stability result used a 1B-parameter, 24-layer, 1,536-wide model at
200B tokens; outcome runs span 100–400B tokens. This project used four layers,
width 128, 83.10M non-PAD symbols, a 258-token tied vocabulary, AdamW for every
parameter, program batching, no source-equivalent WSD cooldown/z-loss/depth
scaling package, and 182 three-pass batches per F5c cell. The source's split
NorMuon/Adam optimizer, 300k-token global batch, 8,192-token context and model
scale are untested here.

Consequently `I_structure` remains uncomputed and the scientific relation claim
remains untested. A future feedback attempt must be a new direction with a new
contract—preferably a source-condition or explicit scaling study—not F5d.

### 10.5 Next instrument package

Before the next expensive direction, qualify the measurement platform:

- compare canonical model/optimizer/RNG state content, never serialized file
  bytes;
- record state digests at first evaluation and final step;
- either use a backend that reproduces exactly or predeclare repeat count,
  metric-level tolerance and resolvable effect floor;
- never infer a variance or “noise floor” from one repeated pair.

This package is cross-direction infrastructure. It does not authorize another
Direction 3 cell.
