# Direction 4 — explicit copy and relation actions

**Updated 2026-09-10: R4 engineering closed; R5 route acceptance is the next package.**
R0/R1 are frozen and R1.1/R2/R3 are complete. The
[independent F-4 review](copy-relation-r4-f-independent-review.md) closes all
seven remaining R4 points on its recorded CPU/source snapshot, retaining the
training and decoder acceptances. The prior pending CR-1–5/points 5–7 statements
are superseded by that review; their reasons and repairs remain in the cited
historical records. No backend other than CPU is qualified on this machine.

The existing `dm/relation/wiring.py` and `scripts/relation_wiring.py` implement
an R5 engineering card, execution route, checkpoint envelope and restart child.
This supersedes the old “no R5 route” handover. The
[live R5 whiteboard](copy-relation-r4-readiness-package.md) assesses Point 5,
records open R5 findings W1–W6 and specifies implementation slices A–E with
acceptance witnesses. Physical instrumentation is accepted; production resource
qualification is still absent. No accepted R5 smoke, development qualification
or scientific Direction 4 result is established. The current assessment is
planning and engineering verification only.
Direction 3 is closed. Direction 4 is a new claim with a new mechanism, protocol
and artifact namespace; it may reuse audited primitives, but no Direction 3
threshold, cell or result selects an option here.

The proposed mechanism is:

```text
EMIT
COPY(source instruction span, affine step, total count)
```

The model still produces ordinary flat ISA bytecode. `COPY` is an internal
decoder action: it selects an earlier whole-instruction span, applies an exact
element of D4 semidirect integer translation, appends the resulting bytes, and
then advances the transformer's cache through those bytes. It never emits
`REPEAT`, `REPEATX`, `XFORM` or private control tokens into the program.

This file is the live claim and implementation contract. R1's model-blind subset
is frozen in `copy-relation-protocol-v0.json`, and a test proves its JSON and
Python forms identical; later-stage design remains editable only before the
stage it governs and with its reason recorded here.

## 1. Question and scope

Claim 2 established two facts:

- within the observed counts, the byte model recovers `22.75` of `28.20`
  redundant bits/drawing (`recovery = +0.8068/+0.7836` across two seeds);
- both seeds lose the copy discount exactly at copy 5 after training on counts
  2–4, while the depth of that loss does not replicate.

Direction 2 then established that ordinary checkpoints condition strongly on a
relation-bearing prefix under teacher forcing, but exact compatible generation
remains weak. Direction 4 asks the next causal question:

> Does an explicit, supervised relation action turn already-detectable copy
> structure into exact flat-bytecode generation on combinations not observed
> during training?

What this direction does not ask:

- whether a larger transformer would solve the task;
- whether `REPEATX` should replace flat bytecode in the ISA;
- whether a model can know a randomly sampled final repeat count that is absent
  from its prefix;
- whether copying is general program synthesis, recursion or a learned library;
- whether a result transfers beyond the two frozen construction venues.

The architecture must remain below one million trainable parameters and retain
the standard byte decoder as a first-class runtime mode.

### 1.1 The count-boundary question is partly unidentifiable

In the flat Claim 2 corpus, nothing before copy 5 says whether the generator
sampled a total count of 4, 5 or 6. If training assigns zero support to 5 and 6,
an unconditional decoder has no observation from which to infer either value.
Its failure is a learned prior failure; its success could only come from an
unintended corpus cue or an extrapolating prior.

Direction 4 therefore separates:

1. **plan inference** — predict a copy action from a prefix when every atomic
   value has training support but the combination does not;
2. **plan execution** — given an action, produce exactly the transformed flat
   bytes, including counts beyond the training range;
3. **unconditional count preference** — the original copy-5/6 curve, retained
   as a diagnostic and never relabelled as rule execution.

Held-out counts in a confirmatory combination stratum are values seen elsewhere
in training under different motifs/transforms. Numerically unseen counts are
oracle-execution tests only. This is not a concession to the model; it is what
makes the two questions identifiable.

## 2. Claim contract

### 2.1 Arms and runtime modes

Two training arms, paired on corpus, schedule and non-relation initialization:

| arm | training objective | parameters | purpose |
|---|---|---:|---|
| `none` | ordinary byte NLL | current byte baseline | controls corpus and additional training |
| `span_affine_v1` | ordinary byte NLL plus relation-action supervision | baseline plus bounded head | learns the proposed internal action |

One `span_affine_v1` checkpoint is evaluated in three runtime modes:

| mode | action source | use |
|---|---|---|
| `standard` | none | auxiliary-supervision effect versus `none` |
| `predicted_copy` | model head | primary explicit-mechanism contrast versus the same checkpoint in `standard` |
| `oracle_copy` | frozen annotation | software ceiling and diagnosis only; never a scientific headline |

The primary comparison is `predicted_copy − standard` within the same weights.
It has exact parameter, optimizer, corpus and training equality. `standard −
none` separately measures what auxiliary supervision changed in the trunk.
`oracle_copy − predicted_copy` locates plan inference as the remaining error.

**Accepted pre-outcome comparator amendment (Point 4 execution):** all paired
cells, including the `none` arm, explicitly request
`literal_policy=content_bytes_v1`, with fixed `stop=flat_l0_stop_v1`.
Scientific standard is `(mode=standard, literal_policy=content_bytes_v1,
stop=flat_l0_stop_v1)`; predicted and oracle use those same policies with their
distinct action sources. The omitted/default standard call remains the named
`legacy_codec_v1` compatibility path, not the causal comparator: its support
includes PAD/BOS and its stopping is caller-defined. Frozen protocol v0 is
unchanged. [Point 4 progress](copy-relation-r4-decoder-plan.md) tracks acceptance;
this policy decision is not a learned result or a comparability acceptance claim.
The [2026-09-08 independent acceptance](copy-relation-r4-decoder-acceptance.md)
closes Point 4's engineering witnesses and report-integrity findings. It supplies
no learned-accuracy or speedup claim. The subsequent
[F-4 acceptance](copy-relation-r4-f-independent-review.md) closes points 5–7.

### 2.2 Primary estimands

For case `i`, seed `s` and paired draw `r`, let `H(mode)` be exact equality of
the relation-bearing target block to the frozen compatible bytes. Rows that do
not reach the target remain failures and stay in the denominator.

```text
D_copy = H(predicted_copy) - H(standard)
R_copy = D_copy / (1 - H(standard))
```

`R_copy` is recovery of the same-checkpoint oracle gap. The denominator is
known: accepted oracle actions must reproduce the target exactly. If standard
is already perfect for a component, that component is reported and omitted
from `R_copy`, not assigned a convenient value.

Co-primary strata:

1. `heldout_affine_tuple` — D4, `dx`, `dy`, count and motif atoms all occur in
   training; the complete tuple does not;
2. `heldout_nested_composition` — each inner/outer action occurs in training,
   but their ordered composition and motif pairing do not.

Secondary estimands:

- `heldout_motif_pair` — the affine tuple is trained, but the motif pairing is
  withheld; these are the pure pair-novelty cases;
- `affine_pair_interaction` — both the affine tuple and motif pairing are
  withheld; these are secondary because a failure mixes the two interventions;

- exact valid-action mass and exact predicted action;
- valid-equivalent source-span mass;
- exact D4, signed `dx`, signed `dy` and total-count components;
- copy-decision precision/recall and false-copy rate;
- standard byte NLL, per-program `val_bits` and generic generation validity;
- copy-action count, deterministically appended bytes, sequential samples
  avoided, cache-fill time, wall time and peak memory;
- Claim 2 `recovery_by_copy`, including the count-5/6 diagnostic curve.

### 2.3 Decision labels

Labels are applied in this order:

1. `incomplete` — missing cell/artifact/hash, truncation, failed reload,
   unqualified device, corpus defect or malformed case;
2. `unsafe_copy_channel` — a generic, destroyed-relation, resource or validity
   guard fails;
3. `no_explicit_relation_gain` — the flat affine effect misses its floor in
   either seed;
4. `explicit_relation_gain` — the flat affine stratum passes both seeds but the
   nested stratum does not;
5. `compositional_relation_gain` — both co-primary strata pass in both seeds and
   every guard passes.

There is no “promising” label and no pooling seeds to rescue a failed seed.
Failure on nesting does not erase a valid flat result; it limits the claim by
name.

### 2.4 Proposed practical floors

Before the first scientific checkpoint, protocol v1 must freeze:

- paired exact-block improvement `D_copy >= +0.10` in the flat affine stratum
  and both seeds for any positive label, and also in the nested stratum for
  `compositional_relation_gain`;
- connected-component bootstrap 95% lower bound for `D_copy > 0`, plus a
  component-level randomization test with Holm-adjusted `p < 0.05` across the
  two co-primary strata;
- oracle-gap recovery `R_copy >= 0.50` under the same flat/nested label rule;
- at least 128 accepted connected source components per co-primary stratum;
- `standard` likelihood cost versus paired `none <= +1.0` bit/drawing;
- standard valid-halt loss versus paired `none <= 0.05`;
- predicted-copy generic valid-halt loss versus the same checkpoint's standard
  mode `<= 0.01` and false-copy invocation on non-relational programs `<= 0.01`;
- no more than `1.50x` baseline peak training memory or wall time at the frozen
  smoke shape;
- no more than 32,768 new trainable parameters: at the current 824,704-parameter
  byte model, total `<= 857,472`.

The effect floors are deliberately much larger than the one-to-five-point
changes that this project has repeatedly found difficult to separate. R0 must
show the selected device can resolve them. If it cannot, no full run launches;
the floors do not move down.

## 3. Architecture: bounded span pointer plus exact transducer

The design lineage is explicit. [Pointer Networks](https://arxiv.org/abs/1506.03134)
show how attention can define a distribution over a variable-size set of source
positions. [CopyNet](https://aclanthology.org/P16-1154/) and the
[pointer-generator](https://aclanthology.org/P17-1099/) combine generating from a
fixed vocabulary with copying from a source. Direction 4 keeps their central
generate-versus-point separation but changes the copied object: a bounded
whole-instruction span is transformed by an exact typed affine action and may
emit several bytes. That segment action is why v1 does not inherit their
token-level normalized likelihood automatically (§3.5).

### 3.1 Action semantics

At a canonical instruction boundary after a reference block, an action is one
of:

```text
EMIT
COPY(start_instruction, stop_instruction, step, total_count)
```

`[start, stop)` is an earlier, non-empty, whole-instruction span that excludes
`HALT` and lies wholly in the already generated prefix. `step` is
`Transform(D4, dx, dy)`. `total_count` includes the reference span, so a count
of 4 appends `step^1`, `step^2` and `step^3` images.

“Nested” in v1 means compositional provenance without a hidden recursive output
language: an earlier action may create bytes that become part of a later source
span, and the later affine action copies that accumulated block. The flat action
schedule is left-to-right and non-overlapping. A structured nested scope may
admit several equivalent flat schedules; corpus tracing retains that equivalence
set and the decoder may choose any member that produces the target.

The transducer uses `dm.isa.transform.Transform.power` and the by-`Kind`
rewriter in `dm.data.augment.apply`. It never treats alternating operands as
coordinates, never transforms radius/width/count as positions, never clamps an
off-canvas coordinate, and never invents a partial instruction.

An invalid action is masked before selection. There is no catch-and-fall-back to
`EMIT`: selecting an action that later fails validation is an implementation
fault and makes the run incomplete.

### 3.2 Candidate set

Candidate construction is deterministic and model-blind:

- source endpoints are canonical instruction boundaries in the prefix;
- source and target never overlap;
- maximum source instructions/bytes are frozen from a model-blind coverage
  audit before training;
- only spans whose requested images remain canonical, on-canvas and within
  `max_len` are eligible;
- actions that save no stochastic decisions are excluded by a frozen minimum
  body size;
- equivalent actions are retained as an equivalence set, not collapsed to the
  generator's arbitrary provenance label.

The candidate cap must cover at least one valid derivation for 100% of accepted
positive cases. A case outside the cap fails corpus acceptance; it is not
dropped after model scoring.

The implementation must use a bounded source-span set and bounded, explicitly
frozen value supports. It may not materialize an unrestricted `T x T` byte-span
tensor or the full ISA-wide Cartesian product of all signed bytes and counts.

### 3.3 Relation head

Proposed checkpoint field: `relation_schema="none"|"span_affine_v1"`, defaulting
to `none` and appended to `Config`. Direction 4 v1 refuses simultaneous latent
feedback: `feedback_schema` must be `none` when relation parameters exist.

At a queried boundary, the head reads the normalized top-layer states already
available from `DrawingLM.forward(..., return_state=True)`:

- a low-rank query from the current boundary;
- one low-rank key per `(source start, source stop)` candidate, constructed from
  its two boundary states and a length embedding;
- an `EMIT/COPY` logit;
- a categorical D4 head over exactly eight values;
- separate categorical heads over the model-blind `dx`, `dy` and count supports
  frozen in R1.

Factorizing D4, `dx`, `dy` and count makes an unseen *combination* of supported
atoms representable without claiming an unseen atomic value. Numerically unseen
counts bypass the head in oracle-execution tests, exactly as §1.1 requires.
Every component logit is persisted for audit.

Generation retains normalized top-layer states at canonical instruction
boundaries in a separate bounded boundary-state cache. Attention KV tensors are
not source-span representations: they are layer-local projected keys/values, not
the top-layer semantic state the head trained on. The cache costs
`O(B * d_model)` and its bytes enter the inference-memory report.

For predicted decoding, the factor scores define a joint action score over the
small frozen supports. A bounded k-best Cartesian enumerator considers actions
in descending score order and masks geometrically invalid tuples before the
`EMIT`/best-valid-`COPY` decision. Its candidate and expansion caps are frozen
in R1 and recorded. It may not select independent argmax factors, discover that
their tuple is invalid, and silently fall back.

The relation module is registered last and initialized from a forked RNG, as the
feedback module is today. Common parameters in `none` and `span_affine_v1` must
be bit-identical under shared initialization. A test compares every shared
tensor by name and content.

### 3.4 Decoder integration

Default `standard` calls the existing generation path byte for byte. Explicit
`content_bytes_v1` standard delegates with PAD/BOS forbidden and the shared
flat-L0 stopping adapter, without modifying `generate()`.

`predicted_copy` queries the action head only at canonical instruction
boundaries. `EMIT` delegates one ordinary symbol to the existing sampler.
`COPY` deterministically appends the selected transformed span. The appended
bytes are then passed through the ordinary causal cached path to materialize all
KV states and new instruction-boundary states before decoding resumes. In a
mixed batch, copied bytes are queued per row and consumed one synchronized
output position at a time: rows with queued bytes emit them deterministically
while other rows may sample or begin their own actions. This removes sampling
decisions without pretending the transformer can skip conditioning work.

The synchronized queue is a correctness constraint of the current cache, not an
optimization preference. `LayerCache.n` is one sequence depth for the whole
batch, while copy actions can append different byte counts per row; padding a
shorter row's cache would make padding part of its causal context. A future
homogeneous multi-byte block fill is allowed only with an explicit prefix-offset
causal mask and cached-versus-uncached state/logit parity. The present attention
path disables causality when a query block is shorter than the populated cache;
using that path for a multi-byte copied block would let earlier positions in the
block attend to later ones and is forbidden.

`oracle_copy` exercises the identical transducer and cache-fill path with the
frozen action. Any output difference between oracle bytes and the target is a
software failure.

Action selection has no tuned confidence threshold. `EMIT` and the best valid
copy action participate in one declared decision rule; greedy evaluation uses
argmax, and stochastic promotion uses frozen uniforms. A threshold selected
after seeing false copies is forbidden.

Stochastic pairing uses counter-based uniforms keyed by case, draw, absolute
output position and stream (`token` or `action`). It never advances one shared
RNG sequentially: a copy action consumes fewer token samples than standard mode,
and sequential consumption would shift every later draw exactly where the modes
first differ.

### 3.5 Probability and accounting boundary

The comparator samples all 256 content bytes at temperature 1 with no top-k.
This is codec support, not a grammar or VM-validity mask. Shared stopping retains
HALT, unknown and non-flat opcode bytes; failed rows remain in denominators.
Operand zeros do not stop; horizon-boundary and partial-tail outcomes are distinct.
Position-keyed uniforms pair coordinates, not contexts or total sequential RNG use.

The ordinary byte head remains trained and scored at every non-PAD byte,
including bytes that an action could copy. Its `bits_per_drawing` stays directly
comparable to every existing byte checkpoint.

The action head has a valid supervised action NLL, reported as `relation_bits`,
but v1 does **not** add it to or subtract it from byte NLL. A deterministic
multi-byte action is a semi-Markov decoding policy, not a normalized per-byte
distribution. Calling its action cost “copy bits/drawing” would require a
separate marginal likelihood that sums over literal and all equivalent action
derivations. That is explicitly outside v1.

Accordingly the primary claim is exact generation, not an artificial likelihood
win purchased by letting one action stand for many bytes.

## 4. Corpus and supervision

### 4.1 Construction venues

Two frozen venues share the same action schema:

1. `synthetic_nested_repeat` — structured `REPEAT`/`REPEATX` programs with
   counts 2–4, depth 1 in training and selected depth-2 compositions held out;
2. `composed_motif_relation` — QuickDraw-derived motif spans under exact D4 and
   integer translations, with held-out motif-pair and affine tuples.

Training targets are the flat result of unrolling the structured source.
Scientific outputs contain only L0 drawing instructions and one `HALT`.

The corpus builder must add `unroll_with_trace`: one return value is byte-for-
byte equal to `dm.isa.unroll.unroll`; the other maps every structured scope to
its flat reference/target spans and composed `Transform`. The existing unroller
and VM remain independent checks on the trace builder.

### 4.2 Split discipline

- Split source motif/group identities before applying transforms or pairing
  motifs. No D4/translation image of a validation motif enters training.
- Build a graph whose vertices are motif/source groups and whose edges are
  combinations. Bootstrap connected components, not individual cases.
- In a confirmatory held-out tuple, every atomic D4 value, `dx`, `dy`, count,
  span-length bin and motif category has positive training support; their joint
  tuple has zero support.
- “Motifs never seen together” means an edge-disjoint pair split with both
  endpoint motifs observed elsewhere in training. Completely novel motifs are
  a separate transfer stratum.
- The pure affine co-primary excludes pair-novelty cases. Cases carrying both
  interventions are labelled `affine_pair_interaction`, never silently assigned
  to the affine gate.
- Numerically unseen count values are tagged `oracle_only_count` and cannot enter
  a predicted-plan promotion gate.
- Development and scientific seeds, manifests, source identities and case IDs
  are disjoint. For composed motifs, a stable provenance hash partitions source
  identities before limits and islands; development artifacts never enter a
  scientific threshold.

### 4.3 Annotation acceptance

Before training, every program and action must pass:

- structured and flat programs execute without VM fault and render identical
  strokes; this equality is persisted in `vm_census.structured_flat` and is a
  corpus acceptance clause, not just a prose claim;
- `unroll_with_trace(...).bytes == unroll(...)`;
- each annotated target equals the independent affine application of its source;
- source/target boundaries are complete instructions and target ranges do not
  overlap;
- relation powers and nested composition use `Transform.then/power`, never a
  hand-written second group law;
- no coordinate was clamped and no sequence is truncated;
- candidate enumeration recovers at least one equivalent valid action;
- equivalent-action sets reproduce exactly the same target bytes;
- train/development/scientific fingerprints and group splits reconcile;
- the relation-destroyed control preserves length and model-blind byte/field
  marginals within its frozen tolerance while containing no annotated positive;
  its donor case, donor groups/source identities and destruction namespace are
  persisted and the donor map is a closed, fixed-point-free permutation.

If exact negative construction is impossible for a stratum, name the residual
valid actions and exclude that stratum before the protocol freeze. Never call a
row negative because the generator did not annotate a relation the bytes still
contain.

### 4.4 Training targets

At every canonical instruction boundary the gate target is `EMIT` or the set of
valid `COPY` derivations. Positive actions additionally supervise source-span
equivalence mass and categorical D4, signed `dx`, signed `dy` and total-count
components over the frozen supports.

Losses:

```text
L_action_joint = -log sum_{a in valid(b)} p_span(a) p_d4(a) p_dx(a) p_dy(a) p_count(a)
L_action       = L_gate + L_action_joint
L_total        = L_byte + 0.1 * L_action
```

**The action term marginalises the joint valid-action set, not each factor
separately.** A per-factor sum -- one `L_span`, one `L_d4`, one `L_dx` and so on,
each over its own marginal mass -- is minimised by putting mass on any span, any
D4 element and any translation that appear *somewhere* in the valid set,
including combinations that appear nowhere in it. On a corpus whose whole point
is that only some tuples are valid, that lets probability collect on invalid
cross-products and still score well. The per-factor numbers are reported as
diagnostics and are not what is optimised.

Each term is averaged over the positions where it is defined and PAD contributes
to none. **`L_gate` is averaged over *reachable* boundaries only**: after a
`COPY` the decoder appends the whole target block and resumes at its end, so it
never stands at a boundary inside it, and supervising `EMIT` there would train
the head on positions no decode visits while making a nested corpus's boundary
count depend on how deeply its copies nest. Those bytes keep ordinary byte
supervision; only the gate skips them.

`0.1` is one fixed coefficient, not the winner of a sweep. R4 records component
losses and shared-trunk gradient norms. If the smoke shows a non-finite or
zero-gradient component, the implementation fails qualification; the coefficient
is not tuned on an outcome.

Marginalising rather than picking one derivation is what keeps annotation choice
out of the loss: several distinct actions append byte-identical continuations,
and training against one arbitrary member would turn a correct decoder into a
model error.

## 5. Experimental matrix and statistics

### 5.1 Fixed matrix

Proposed scientific matrix after qualification:

- byte codec, four layers, width 128, four heads, ordinary causal RoPE model;
- flat relation corpus, same train/validation bytes for both arms;
- `none` and `span_affine_v1`, model seeds derived from two frozen seed names;
- 24,000 steps, same batch construction, AdamW and schedule in both arms;
- common non-relation initialization within a seed;
- final checkpoint only, never validation-selected best;
- one run per declared cell; no retry or replacement seed.

The corpus size, `max_len`, source-span cap and action-case counts freeze at R1
after model-blind acceptance. They may not be selected from a model curve.

The full matrix is four training cells. Runtime modes do not create checkpoints.
No codec, width, depth, loss-weight or candidate-cap sweep is part of Direction
4 v1.

### 5.2 Statistical unit

- Teacher-forced action metrics are paired by frozen case.
- Free-running modes use common random numbers within checkpoint and draw.
- Token/action uniforms are position-keyed, so a copied block or early stop
  cannot shift the other mode's later random numbers.
- A row that never reaches its target remains a zero exact hit.
- Bootstrap 2,000 shared draws over connected source/motif components.
- Seeds are reported separately and both must pass.
- A connected-component bootstrap supplies intervals; a component-level paired
  randomization test supplies p-values; Holm correction covers the two
  co-primary strata at alpha `0.05`.
- Raw per-case/per-draw rows and all component NLLs are persisted before tables.

Population-of-model claims remain unavailable at two seeds. The valid statement
is checkpoint-conditional replication under two named initializations.

### 5.3 Generic and destroyed-relation controls

The copy-enabled checkpoint is also decoded on:

- the ordinary Tier A flat validation split;
- a relation-destroyed length/marginal-matched split;
- prompts ending at non-copy instruction boundaries;
- source spans for which one action component is deliberately wrong.

The destroyed arm deranges only within a namespace keyed by venue, source
stratum, connected co-occurrence component, target length and opcode skeleton
(with the donor case, groups and source identities persisted on every row).
Residual or newly introduced relations are re-enumerated and rejected; dropping
a row triggers a fresh closed derangement so donor and recipient multisets stay
balanced.

These measure false copy invocation, validity, target corruption and whether the
head responds to a complete relation rather than mere span similarity. A model
that improves the primary by copying indiscriminately is `unsafe_copy_channel`,
not a positive result.

## 6. R0 measurement-platform qualification

Direction 4 inherits the lesson, not the malformed comparator, from Direction
3. Before a full cell:

1. Persist canonical model, optimizer and RNG-state digests at the first eval
   and final step. Serialized checkpoint-file SHA remains artifact identity only.
2. Run one fixed short training sentinel repeatedly on each candidate backend,
   with the frozen configuration `{vocab_size: 258, d_model: 64, n_layers: 2,
   n_heads: 4, max_len: 128, steps: 12, batch: 8, length: 48, lr: 1e-3}` and
   the frozen sentinel seed.
3. Prefer a backend with exact state reproduction. **The metric-level fallback
   this clause once allowed is withdrawn in v1** -- see §12 item 14 -- so a
   backend that cannot reproduce state exactly fails R0 until a paired
   relation-task measurement exists, which cannot be before R3.
4. Exercise interrupted/resumed equivalence separately at step 6; serialise to
   disk, rebuild fresh model/optimizer objects and restore all RNG streams.
   Final model equality alone does not verify optimizer or RNG continuation.
5. Fail R0 before any scientific cell if the platform cannot resolve the
   `+0.10` exact-hit floor.

The raw report is authoritative: the validator independently derives digest
sets, distinct counts, exact reproduction, resume differences and the
`equivalent` verdict. It binds repeatability and resume to the same device,
frozen seed/configuration and recorded environment; a self-reported
`is_frozen_shape` flag is not evidence.

`torch.use_deterministic_algorithms(True)` is recorded policy, never a device
certificate.

## 7. Non-negotiable invariants

1. Freeze protocol, fixtures, thresholds, seed namespaces and source hashes
   before an outcome-bearing checkpoint.
2. Keep Direction 4 provenance and artifact names disjoint from Direction 3.
3. Default `relation_schema="none"` reproduces existing fixtures and checkpoints
   exactly; unknown schemas fail closed.
4. Register optional parameters last and preserve common initialization exactly.
5. Standard mode never reads relation logits. Legacy standard sampling is unchanged;
   paired cells explicitly use the common content-byte and flat-L0 stop policies.
6. Predicted mode reads prefix and model state only; target bytes and oracle
   annotations are structurally unreachable.
7. Oracle mode never supplies scientific evidence; it diagnoses the transducer
   and the inference gap.
8. Output is ordinary canonical flat bytecode. No hidden token enters the ISA,
   codec vocabulary, LM head or VM.
9. Source and target are whole-instruction spans; transforms operate by `Kind`.
10. Invalid/off-canvas/overlength actions are masked before selection, never
    clamped or silently changed to `EMIT`.
11. Multiple valid derivations are marginalized and accepted as equivalent.
12. Every byte retains ordinary LM supervision; relation loss never replaces it.
13. Never add action NLL to byte NLL and call the result comparable
    bits/drawing without a normalized marginal sequence model.
14. No padding enters token, action, source-span or efficiency denominators.
15. Split motif identities and relation tuples before augmentation; partition
    composed source identities before limits/islands; freeze the co-occurrence
    dependence graph and resample its connected components.
16. Every held-out confirmatory tuple has atomic support in training. Unseen
    count values are oracle-only.
17. Candidate caps and corpus rejection rules are model-blind and frozen before
    training; no post-score case removal.
18. The relation checkpoint and baseline see identical program bytes, steps,
    batches and non-relation initialization within seed.
19. Final checkpoints only; no best-on-validation selection, retries or
    replacement seeds.
20. Report parameters, content tokens, padded positions, wall time, peak memory,
    action queries, copied bytes and cache-fill work separately.
21. Preserve raw action logits, equivalence sets and per-draw outputs so every
    aggregate can be recomputed.
22. A generic or destroyed-relation guard failure precedes every positive label.
23. Keep a separate top-layer boundary-state cache; never reinterpret attention
    KV entries as the span representation the relation head was trained on.
24. Key stochastic token/action uniforms by absolute position and stream; never
    pair variable-consumption decoders by advancing one sequential RNG.

## 8. Implementation stages

### R0 — qualify the instrument

Implement state/optimizer/RNG canonical digests and the repeated sentinel.
Freeze the backend/repeatability rule. No relation code or corpus outcome is
needed to complete R0.

### R1 — executable contract and traced corpus, tests first

Create the Python contract plus `docs/copy-relation-protocol-v0.json`. Add
`unroll_with_trace`, action/equivalence schemas, split builder, acceptance report
and tiny hand-auditable fixtures. Tests pin transform composition, nested spans,
candidate coverage, group disjointness and every invariant that can be made
structural.

R1 may revise this draft only from model-blind corpus facts. The diff and reason
are recorded before protocol v0 is written.

### R2 — deterministic transducer

Implement `COPY` independently of the neural head, behind a narrow
`dm.relation` runtime package rather than making inference import the 4,000-line
corpus builder. The corpus Module may re-export shared action values for
compatibility, but it remains the sole owner of construction, acceptance and
manifest validation. Any extraction must leave `--rebuild-verify` byte-identical.

The transducer receives a flat prefix plus an action key and derives its target;
it never trusts recorded `target_start/target_stop`. It validates canonical
whole-instruction boundaries, a non-empty earlier source wholly inside the
prefix, no `HALT`, affine/count support policy, canvas safety and the full
`max_len` append before mutating output. It either appends the complete block or
raises a typed transducer fault — never clamp, truncate or fall back to `EMIT`.
Predicted-support validation and oracle-only counts are separate policies over
the same executor, not separate implementations.

Tests first, in this order:

1. exhaustive pure fixtures over all eight D4 elements, signed translations,
   counts/powers, `CURVE`, scalar operands that must remain unchanged, nesting,
   boundary/source/`HALT` rejection, canvas rejection and max-length rejection;
2. replay every scheduled action in a small traced corpus and require exact
   target bytes plus VM/render equivalence;
3. feed copied bytes one at a time through the existing cache and require state
   and next-token-logit parity with an uncached full forward;
4. exercise heterogeneous copy lengths through per-row deterministic queues and
   require no copied row consumes token randomness or padding as context.

Oracle output must be byte-identical and VM-equivalent. R2 adds no relation
parameter, loss, predicted head, threshold or scientific scorer.

### R3 — relation head and compatibility

After separate authorization, add `relation_schema`, the bounded span scorer and
factor heads. Use a packed candidate layout linear in admitted spans, never an
unrestricted byte-span matrix or a padded span denominator. Pin parameter
arithmetic, default checkpoint compatibility, shared initialization, gradients,
joint-equivalence-mass loss, deterministic bounded search and refusal of
feedback/relation coexistence. R3 does not modify `generate()` or a training
loop; the complete pre-implementation package and stop conditions are §12.7.

### R4 — training, accounting and non-outcome decoder integration

Implement the authorized package in
[`copy-relation-r4-plan.md`](copy-relation-r4-plan.md): add the relation dataset
adapter and action loss without modifying ordinary byte loss; pin padding masks,
reachable-boundary counts, separate loss denominators, gradients, paired
optimization, checkpoint/provenance fields, content digests and exact resume;
and integrate non-outcome `standard`, greedy `predicted_copy` and `oracle_copy`
decoding through R2's executor and synchronized byte queues. R4 writes no
persistent model artifact and runs no smoke, qualification or scientific case.

### R5 — end-to-end smoke and qualification

One tiny corpus exercises `none`, relation `standard`, `predicted_copy` and
`oracle_copy`. Then two development seeds at the frozen reduced budget check
training health, false copies, resource overhead and action learning. Development
sees no scientific cases and cannot move the effect floors.

The only allowed repair after a failed smoke is a named implementation defect
whose fix is testable without inspecting scientific outcomes. Architecture or
loss redesign opens `span_affine_v2`; it is not silently folded into v1.

### R6 — freeze scientific protocol and train four cells

Protocol v1 adds exact corpus, source, environment, config and seed identities.
Train baseline/relation × two seeds. Incomplete or guard-failing cells stop the
direction before relation scoring.

### R7 — freeze checkpoints, score once and decide

Protocol v2 hashes records/checkpoints and freezes cases before any result.
Evaluate action metrics, paired runtime modes, generic/destroyed controls,
co-primary strata and count-boundary diagnostics. Apply the label once and
archive raw rows, reports, hashes and source provenance.

### R8 — optional promotion, only if pre-authorized

If `compositional_relation_gain` passes, a predeclared promotion may evaluate
more distant motif categories or longer nested depth. It cannot change the v1
claim, rescue a failure or introduce a new mechanism.

## 9. Expected outcomes and diagnostic readings

These are expectations, not results:

- Oracle execution should be exactly 100%; anything else is a software defect.
- Auxiliary supervision may improve relation probes while leaving standard
  generation nearly unchanged. That is `standard − none`, not an executor win.
- Predicted copy should help most when the source span and affine tuple are
  unambiguous. A large oracle/predicted gap points to plan inference, not byte
  emission.
- Source-span ambiguity and false `COPY` decisions are more likely bottlenecks
  than affine arithmetic; the arithmetic is exact and already tested in the VM,
  augmenter and unroller.
- Nested performance should compound action errors. Passing flat and failing
  nested earns the narrower `explicit_relation_gain` label.
- An unconditional improvement on numerically unseen counts is not expected
  without a prefix cue. If it appears, audit corpus leakage and count-correlated
  length/position before calling it extrapolation.
- Copy actions can reduce sequential sampling decisions but still require a
  cache-fill forward over appended bytes. Report both; do not claim that copied
  bytes are free inference.
- The strongest negative is a well-qualified head that predicts relations but
  does not improve exact generation: explicit recognition would then be
  insufficient even when made directly available to the decoder.

## 10. Implementation map

Rows marked **built** exist and are covered by the suite; the rest are proposed
and may still be renamed. Once a name is serialized in a protocol or checkpoint
it becomes compatibility surface.

| path | role | state |
|---|---|---|
| `docs/copy-relation.md` | live Direction 4 design and decision contract | built |
| `docs/copy-relation-r4-plan.md` | full R4 implementation, acceptance and review package | acceptance-closed by F-4 |
| `docs/copy-relation-r4-correction.md` | 2026-09-06 implementation audit and bounded correction | retained correction record |
| `docs/copy-relation-r4-readiness-package.md` | R5 findings, implementation slices and review board | current package |
| `docs/copy-relation-protocol-v0.json` | R1 model-blind executable freeze | built |
| `dm/data/relation.py` | traced construction, splits, equivalence sets and acceptance | built |
| `dm/eval/relation_contract.py` | constants, seed derivation, gates and labels | built |
| `dm/eval/relation_evidence.py` | R0 canonical digests, sentinel and platform rule | built |
| `scripts/relation_contract.py` | thin write/verify adapter | built |
| `scripts/relation_corpus.py` | thin corpus/audit adapter | built |
| `scripts/relation_qualify.py` | R0 runner and fail-closed decision | built |
| `tests/test_relation_corpus.py` | trace, actions, equivalence, splits, controls | built |
| `tests/test_relation_contract.py` | contract, artifact and namespace disjointness | built |
| `tests/test_relation_evidence.py` | digests, sentinel and the R0 rule | built |
| `dm/relation/transducer.py` | pure action validation and exact byte executor | built in R2 |
| `dm/relation/queue.py` | deterministic heterogeneous per-row byte queues | built in R2 |
| `dm/relation/candidates.py` | target-free prefix candidate spans shared by corpus and decoder | built in R4 |
| `tests/test_relation_transducer.py` | exhaustive executor, oracle and cache-parity fixtures | built in R2 |
| `dm/eval/relation_cases.py` | frozen cases and dependence graph | R7 |
| `dm/eval/relation.py` | action metrics, paired inference and report logic | R7 |
| `dm/models/relation.py` | bounded span head, packed distributions, NLL and bounded search | built in R3 |
| `dm/models/transformer.py` | default-compatible registration/state seam; model-owned relation cache loop | built in R3/R4 |
| `tests/test_relation_model.py` | architecture, compatibility, loss and bounded-search acceptance | built in R3 |
| `dm/train_relation.py` | corpus adapter, charged planner, auxiliary loss, paired schedule, diagnostics and continuation | R4 acceptance-closed |
| `dm/relation/resources.py` | opt-in CPU scopes, peak bounds, ledger and allocation inventory | R4 Point 5 acceptance-closed; no production qualification |
| `dm/eval/relation_training_evidence.py` | training source/corpus/environment identities and resume validation | built in R4; correction applied |
| `scripts/relation_train.py` | thin training Adapter; validation only | built in R4 |
| `dm/relation/wiring.py` | fixed R5 card, execution, checkpoint and restart orchestration | implemented in part; acceptance open |
| `scripts/relation_wiring.py` | R5 prepare/validate/execute Adapter | built; route review pending |
| `tests/test_relation_wiring.py` | R5 refusal, temporary checkpoint and short continuation witnesses | built; full acceptance matrix pending |
| `tests/test_train_relation.py` | R4 adapter/loss/accounting/checkpoint/resume acceptance | built in R4 |
| `tests/test_relation_decode.py` | R4 standard/predicted/oracle queue/cache acceptance | built in R4 |
| `tests/test_relation_r4_correction.py` | adversarial training/resume boundaries and independent update parity | built in R4 correction |
| `scripts/relation_eval.py` | frozen R7 evaluator | R7 |

The case index and the dependence graph are written by `dm/data/relation.py`
into the corpus manifest rather than by a separate `relation_cases` Module: the
graph is a property of the corpus and splitting it across two files would give
the resampling unit two owners.

## 11. Current authorization and how to run what exists

R0/R1 maintenance, R1.1, R2 and R3 are complete. R4 was authorized at §12.10
under [its engineering contract](copy-relation-r4-plan.md) and is now closed by
[independent F-4 acceptance](copy-relation-r4-f-independent-review.md).
The older statements here requiring CR-1–5 repairs and Point 7 closure are
superseded; preserve their accepted contracts and evidence.

The owner has directed the next work to proceed. The current task assesses and
plans the R5 package in the [live whiteboard](copy-relation-r4-readiness-package.md).
It completes the existing bounded route's evidence, lifecycle and acceptance
boundaries before the fixed smoke handoff. `scripts/relation_wiring.py` has
prepare/validate/execute routes; `scripts/relation_train.py` remains validation-only.
This assessment runs no smoke or development/scientific cells and does not
freeze protocol v1/v2 or change thresholds. Development budgets and resource
qualification still require concrete predeclared specifications; R6/R7 remain
subsequent stages, not outputs of this planning task.

The R0/R1 creation/refreeze commands below preserve historical setup context;
they are not an instruction to overwrite frozen artifacts while implementing R4.

Establish the shared primitives' baseline before touching them:

```bash
PYTHONPATH=. .venv/bin/python -m pytest -q \
  tests/test_repeats.py tests/test_context.py tests/test_unroll.py \
  tests/test_transform.py
```

R0, which needs no corpus and no relation code. There is one route -- exact
reproduction of model, optimizer and RNG state across five repeats, plus a
serialised interrupted-restart -- and a backend that fails it fails R0:

```bash
PYTHONPATH=. .venv/bin/python scripts/relation_qualify.py \
  --device cpu --out runs/relation_r0_cpu.json
PYTHONPATH=. .venv/bin/python scripts/relation_qualify.py \
  --device mps --out runs/relation_r0_mps.json
```

R1, corpus then contract. The smoke build is meant to fail its component floor;
a corpus that cannot be frozen must not be freezable:

```bash
PYTHONPATH=. .venv/bin/python scripts/relation_corpus.py --smoke
PYTHONPATH=. .venv/bin/python scripts/relation_corpus.py \
  --out runs/relation_corpus_v1.json
PYTHONPATH=. .venv/bin/python scripts/relation_contract.py --write --refreeze \
  --corpus runs/relation_corpus_v1.json
PYTHONPATH=. .venv/bin/python scripts/relation_contract.py --verify
```

R1.1's two audit modes over an existing manifest. The first is under a second on
the frozen file; the second rebuilds the corpus and takes about eight minutes,
so it is explicit rather than implied. The exhaustive route reproduces the
frozen artifact. The fast route is now the accepted file-boundary audit; the
exhaustive route remains the only check that regenerates evidence absent from
the case index:

```bash
PYTHONPATH=. .venv/bin/python scripts/relation_corpus.py \
  --audit runs/relation_corpus_v1.json
PYTHONPATH=. .venv/bin/python scripts/relation_corpus.py \
  --rebuild-verify runs/relation_corpus_v1.json
```

## 12. Revision, authorization and implementation record

§8 requires that R1 revise this draft only from model-blind corpus facts, and
that the diff and the reason be recorded before protocol v0 is written. The
record below is the state after the **R0/R1 blocking-finding correction package** of
2026-08-22; items marked *(corrected)* replace an earlier revision that a review
found unsound, and the earlier form is described so the record is a history
rather than a tidy-up.

None of these was chosen by looking at a model, because no model exists.

**Forced by the parameter budget.**

1. **`TRANSLATION_SUPPORT` is `(-32, 0, 32)`, not every signed byte.** A factor
   head over 256 translations costs `2 * 128 * 256 = 65,536` parameters at
   `D = 128` -- twice the whole extension allowance, before a single span key
   exists. The venues emit exactly these three values, so the support the corpus
   has and the support the head can represent are the same set. The complete
   head costs 15,104, leaving 17,664 of headroom.
2. **The audit uses a wider translation range than supervision does.** The
   relation-destroyed control has to prove that *no* copy relation survives in
   the bytes, not merely none the head could have named; a donor block related to
   its prefix by a translation of seven is still a relation. `AUDIT_TRANSLATIONS`
   spans every `i8` and supervision does not.
3. **`RELATION_RANK = 32` and the span-length bin *edges*** are frozen at R1.
   *(corrected)* An earlier revision froze only a bin **count**, which fixes a
   parameter total and nothing else: with no boundaries there is no way to check
   that a bin is reachable, that every accepted span falls in one, or that two
   builds agree on which bin a span lands in. The edges are now
   `(12, 21, 33, 48, 69, 99, 141, 192)`, the last of them *is*
   `MAX_SOURCE_BYTES`, and `length_bin_report` audits reachability and coverage
   on every build.
4. **Candidate caps are 64 source instructions and 192 source bytes**, set from a
   model-blind coverage audit with headroom over the observed maxima and then
   enforced. `coverage_report` re-checks 100% coverage on every build rather than
   trusting the audit that chose them.
5. **The R3 enumerator's caps are frozen here too.** *(corrected)* §3.3 declares
   them frozen at R1 and an earlier revision left them unwritten.
   `RELATION_MAX_CANDIDATE_SPANS = 768` and `RELATION_KBEST_ACTIONS = 32`. The
   candidate cap is checked against `CANDIDATE_SPAN_BOUND`, which is
   `(gap + 1) * MAX_SOURCE_INSTRUCTIONS = 320` -- arithmetic about the rule, so
   it holds for every program rather than for the ones one build happened to
   produce.

**Forced by what the venues contain.**

6. **A candidate source span may end up to `MAX_SOURCE_GAP_INSTRUCTIONS = 4`
   instructions before the queried boundary.** *(corrected)* An earlier revision
   froze `source_stop == boundary`, which covers every canonical action and is
   cheap -- but it turns the mechanism §3.1 asks for, a pointer at an arbitrary
   earlier whole-instruction span, into a suffix copier, and it makes the head's
   `key_stop` projection constant across every candidate at a boundary, so 4,096
   parameters would buy no discrimination at all. The endpoint is free again,
   bounded so the candidate set stays `O(gap * cap)` per boundary. A span that
   skips intervening bytes is admitted on its merits: the estimand is exact
   equality of the target block, so an action that renders it *is* correct
   however unusual the span it points at, and refusing it would score a decoder
   wrong for being right. Four rather than more is a cost bound, not a
   principle -- every canonical action has gap zero, so the cap governs only how
   wide the equivalence set may be.
7. **Supervision is over *derivations*, not only whole-block actions.** A count-4
   scope is produced by one action of count 4, by count 2 followed by `step^2` at
   count 2, or by count 3 followed by count 2 -- all append byte-identical bytes.
   §3.1 named this equivalence; R1 makes it computable. `derivations` returns
   every valid first step whose remainder is itself derivable;
   `equivalent_actions` is the single-action subset.
8. **Motif identity is placement-invariant.** `dm.data.composed.place` jitters a
   motif inside its cell, so one QuickDraw drawing has different bytes in every
   scene. Hashing those bytes gave one vertex per *placement*, every co-occurrence
   edge a bridge between two degree-one vertices, and a pair split that could
   withhold almost nothing. Normalising by translation collapses the placements
   and nothing else.
9. **The affine-tuple split is global across venues.** Every composed orbit is a
   `(d4, 0, 0, count)` the synthetic venue can also express, and a checkpoint
   trains on both venues at once. A per-venue split left tuples withheld in one
   venue and trained in the other, so the "combination" under test had been seen.
10. **`composed_motif_relation` contributes exactly one withheld tuple**, and
    which one is derived rather than chosen. Its six orbits have counts bounded
    by each element's own order in D4, so `(quarter turn, 0, 0, 2)` is the only
    tuple whose atoms both survive elsewhere. A plain 15% fraction rounds six
    tuples to zero, so `held_out_tuples` withholds at least one where one is
    holdable.
11. **A generic negative is audited by the enumerator, not by provenance.**
    *(corrected)* An earlier revision rejected a "generic" program when
    `case.actions` was non-empty -- but `actions` comes from *scope* provenance
    and a flat program has no scopes, so the test was a tautology that accepted
    whatever the generator produced. A rebuild of the earlier frozen split found
    11 of 1,000 accepted cases containing audit-wide copy actions, in the split
    whose whole job is to be the false-copy denominator. `derivable_boundaries`
    now walks every boundary and asks the enumerator over the audit translation
    range; the earlier build rejected 0 draws on this rule, the corrected one
    rejects about 40%.

**Forced by the estimator.**

12. **The motif pool is partitioned into islands of four, and a case draws every
    motif it contains from one island.** *(corrected)* An earlier revision found
    that the §4.2 co-occurrence graph collapses to a *single* component -- a chain
    of scenes links every motif to every other -- and worked around it by
    resampling *source* motifs alone, which leaves the dependence through shared
    distractors unmodelled. Islands remove the dependence instead of renaming the
    unit: the graph is disconnected by construction, the literal §4.2 reading is
    back in force, and `island_faults` refuses a corpus in which any case spans
    two islands, since one leaked case merges two clusters and narrows every
    interval drawn afterwards. `dm.data.composed.build_with_stats` gained an
    optional `pool` argument for this; its default is the pre-Direction-4 path.
13. **`heldout_nested_composition` uses motifs training actually contains.**
    *(corrected)* An earlier revision built it from the *evaluation* motif pool,
    which confounded compositional generalisation with motif transfer: a failure
    could not have said which caused it. The co-primary now holds motifs fixed --
    and fixed to motifs that reached an accepted training case, not merely to the
    training pool, because the venue refuses most draws on geometry -- and
    withholds only the ordered composition. The harder version survives as a
    separate secondary stratum, `nested_motif_transfer`.
14. **Motif-identity disjointness is required per stratum, not everywhere.** Only
    `novel_motif_transfer` and `nested_motif_transfer` require it, and both are
    checked. `heldout_affine_tuple` and `heldout_nested_composition` hold motifs
    fixed on purpose: their novelty is the tuple and the composition, and seeing
    a motif under a trained action cannot tell a model what a withheld one does
    to it. A held-out program byte-identical to a training program is refused
    outright.
15. **The relation-destroyed derangement re-closes after dropping a row, and a
    splice may not introduce a relation.** A donor block can be an accidental
    image of its recipient's prefix, and such a row is not a negative; dropping
    it from a permutation leaves donors and recipients as different multisets and
    the arm's byte census stops matching its sources' for a reason unrelated to
    the intervention. The stratum is re-deranged instead, up to eight attempts,
    so the census equality is exact. *(corrected)* A second clause now also
    refuses a splice that makes some *other* boundary derivable that its source
    was clean at. Coincidental relations the source already carried are reported
    and never used to reject a row, because filtering on those would select the
    arm on its content.

**R0's own corrections.** All four are from the same review and none is a corpus
fact; they are recorded here because R0 has no other record.

16. **Deterministic kernels are requested before anything is measured.** The
    policy was being enabled while the report was assembled -- after every repeat
    had finished -- so the readings were taken with it off and the artifact said
    it had been requested. `run_sentinel` now requests it first, and the granted
    flags travel with the reading.
17. **The resume check serialises to a real file and rebuilds.** It previously
    handed an in-memory dictionary back to the *same* model and optimizer
    objects, which exercises nothing: no `torch.save`, no `torch.load`, no fresh
    module, no fresh optimizer, and neither the accelerator stream nor the private
    data generator was restored. It now writes the whole carried state to disk,
    constructs a new `DrawingLM` and a new `AdamW`, restores all five generators,
    and generates its batches lazily so the data generator's state is genuinely
    part of the resume. A process boundary is still missing and the report says so
    (`process_boundary: false`) rather than implying otherwise.
18. **The metric-level fallback route is withdrawn.** As implemented it was
    degenerate: the sentinel's exact-block rate moved in steps of
    `1/batch = 0.125`, coarser than the `0.10` floor it was meant to resolve, and
    it read exactly zero in every repeat because the continuations were scored
    against random reference bytes. A statistic that is always zero has an
    observed range of zero, so an arbitrarily nondeterministic backend would have
    "resolved" the floor by missing every continuation. The task metric is gone
    with it; the loss is reported and gates nothing. Restoring the route needs a
    paired measurement on the relation task, which cannot exist before R3.
19. **The digests were completed and `qualify` was made fail-closed.** The RNG
    digest omitted the private `torch.Generator` that produces every sentinel
    batch while labelling the CPU stream twice; the optimizer digest framed group
    *settings* but not group *membership*, so two optimizers with parameters
    swapped between different-learning-rate groups digested identically; and
    `qualify` accepted a schema-1 report that claimed exact reproduction while
    carrying no repeat count, no readings and no digests. All three are fixed and
    each has a test that fails without the fix.

**One clause the contract now enforces that nothing enforced before.**

20. **The corpus seed the contract derives is the seed a build must use.**
    `seed_for("corpus")` is `412653006`; the builder took a `data_seed` argument
    that defaulted to `0`, and no clause compared them -- so the frozen manifest
    recorded one seed while the protocol recorded another. `corpus_seed_faults`
    compares them, a second `development_corpus` seed exists so a development
    manifest cannot be mistaken for the scientific one, and
    `load_protocol(require_corpus=True)` now resolves the manifest **file**,
    loads it, and compares its payload digest to the frozen one. Previously it
    checked only that some digest string was present, so a fresh clone with no
    `runs/` directory passed the purported training precondition while possessing
    no corpus at all.

21. **R0 verdicts are derived from raw readings.** *(corrected)* The earlier
    validator accepted summary digest sets, distinct counts, exactness and
    resume booleans without recomputing them. It now requires exact raw reading
    fields, derives every digest set/count, compares uninterrupted and resumed
    maps, checks the claimed differing list and requires the resume verdict to
    be true.
22. **The R0 sentinel and resume point are protocol data.** *(corrected)* A
    self-reported `is_frozen_shape` boolean no longer qualifies a different
    experiment. The complete shape, seed, environment schema and step-6 resume
    specification are frozen in `relation_contract.py` and the protocol JSON,
    and both reports must bind to the same values.
23. **R1 acceptance and manifest loading fail closed on missing or edited
    safety evidence.** *(corrected)* Required zero/false fields are structural
    requirements rather than defaults, and `load_manifest` reruns `accepts()`
    and compares its verdict and problem list to the recorded ones after the
    manifest digest passes. A self-consistent coverage-zero edit is refused.
24. **Destroyed donors carry the dependence intervention.** *(corrected)*
    Global `(length, skeleton)` derangement erased donor provenance and allowed
    cross-island swaps. Donors are now deranged within venue, source stratum,
    co-occurrence component, length and skeleton namespaces; each row persists
    donor case/groups/source identities and acceptance checks a closed map.
25. **Affine and pair novelty are separate strata.** *(corrected)* A case with
    both a withheld tuple and withheld pair is now `affine_pair_interaction`, a
    secondary diagnostic. `heldout_affine_tuple` is pure tuple novelty and
    `heldout_motif_pair` is pure pair novelty, so a failed affine gate has one
    interpretation.
26. **Development/scientific composed sources are identity-disjoint.**
    *(corrected)* The complete ordered QuickDraw pool is partitioned by a stable
    provenance hash before limits and islands; different scene seeds no longer
    serve as a false source split. The acceptance report records source overlap
    separately from motif overlap.
27. **Pair and structured/flat equality are acceptance evidence.** *(corrected)*
    The exact held-out-edge set, endpoint support, secondary representation and
    pure/interaction labels are audited, and every accepted structured program
    is compared with its flat arm in the VM and renderer. Both audits are
    persisted in the manifest and protocol-linked corpus.

### 12.1 The frozen build

`scripts/relation_corpus.py --out runs/relation_corpus_v1.json` at the contract's
scientific corpus seed passes every acceptance clause. The numbers, the manifest
digest and the protocol digest are recorded in [`PLAN.md`](../PLAN.md) §3 rather
than duplicated here, so one edit moves one place.

The manifest lives in `runs/`, which is untracked like every other run artifact;
protocol v0 records its canonical payload digest, and
`load_protocol(require_corpus=True)` refuses to authorize training unless the
file is present and hashes to it.

### 12.2 What R0 found, and what it did not

On this machine's CPU, at the frozen sentinel shape, five repeats reproduce the
model-state, optimizer-state and RNG-state digests exactly, and a run interrupted
halfway, written to disk and restarted into a fresh model and optimizer reaches
all three. The persisted artifact is `runs/relation_r0_cpu.json` (schema 2,
resume point 6); its route is `exact_state_reproduction`, and there is no other
route in v1.

Three limits, stated rather than implied:

- **`mps` has not been qualified**, and Direction 3's cells ran on `mps`.
  PyTorch is MPS-built on this machine but reports `mps_available: false`, so
  the qualification command cannot produce an artifact here; run it before R6
  in an environment with an available MPS device.
- **The resume check has no process boundary.** It serialises, reloads and
  rebuilds inside one interpreter, which exercises the serialisation and the
  fresh objects but not a genuine restart. The report says `process_boundary:
  false`.
- **`torch.use_deterministic_algorithms(True)` is requested policy, not a device
  certificate**, and R0 records the flags it was granted rather than asserting
  them.

### 12.3 R1.1 evidence boundary — closed after three reviews

R1's acceptance clauses all read the recorded acceptance report, and nothing
asked whether that report described the corpus the same manifest carries. The
digest defended the file and `accepts` defended the corpus; between them sat a
body that had been rehashed after an edit, and it passed. This is Direction 3's
F5b corpus finding in a new place — a clause that compares an artifact against
itself proves nothing — so the repair has the same shape: recompute, then
compare.

**`validate_manifest` is the canonical validator, and it answers a different
question from `accepts`.** `accepts` decides whether a *corpus* may be frozen;
`validate_manifest` decides whether the *manifest* is internally honest. Keeping
them apart is what lets a development build have a perfectly valid manifest that
is still refused for training, and it is why a failure in one is legible without
reading the other. `load_manifest` now runs both, in the order a forgery has to
defeat them: payload digest, structure and index agreement, acceptance verdict,
acceptance itself.

**Exact schemas, in both directions.** Every envelope field, every case-index
record, every acceptance section and the pair,
donor, fingerprint, census and structured/flat evidence inside them are declared
with an exact key set. A missing field was already caught by whichever reader
needed it; an *extra* one was not, and invariant 17 says a second copy of an
invariant is the one that wins
silently. `bool` is refused wherever an `int` is expected, because JSON has one
number type, Python makes `True` an `int`, and `True` is not a count.

The correction extends that rule through the nested evidence the agreement pass
dereferences: per-venue four-integer held-out tuples, canonical pair edges,
donor/structured-flat name lists, case motif/source lists, and VM-arm value types,
counter maps and internal arithmetic. Shape correctness is now real for those
sections; parser-reachable numeric magnitude and coordinated arithmetic are
closed by the second and third reviews below.
The VM rule intentionally remains in Direction 4's namespace: the census builder
is shared, but `dm.data.feedback_audit` is hashed historical Direction 3
qualification source, so moving acceptance arithmetic there would rewrite the
F5b/F5c source-digest trail rather than improve reuse.

**Most facts the case index can produce are produced from it.** Per-stratum and
total case counts, canonical graph-component counts, positives, venues, target
bytes, novel pairs and motif/source overlap with training; the whole pair split
— the held-out edge set,
endpoint support, training leakage and every pure/interaction label; the donor
map's references, metadata, identity exclusion, namespace and closure; the VM
census's arm names and program counts; `structured_flat.cases`; every
fingerprint's program and byte totals; and the caps and bin edges against the
executable contract. Where the report and the index disagree, both numbers are
named.

The headline attack is closed in the reviewed implementation.
`strata.<co-primary>.components` is what the frozen 128-component floor reads,
and stored `component` values are claims rather than measurements. The validator
now rebuilds the exact graph `components()` builds from `groups`, compares every
stored case label with its canonical label, derives every stratum count from that
reconstruction, and uses it for both donor-namespace clauses. Moving stored
labels and matching counts or namespaces together therefore no longer moves the
dependence graph.

**Two clauses are deliberately *containment* rather than equality.** The donor
namespace is `(venue, stratum, component, target length, opcode skeleton)` and
the index carries the first four but not the fifth, so the recomputed fault list
is required to be a subset of the reported one. A report may know more than the
index; it may never know less.

**What the index cannot reconstruct is stated rather than skipped.** Candidate
coverage, residual and introduced relations, the island map, the leak count,
every fingerprint digest and every VM figure need the program bytes, which the
index deliberately does not carry. `--rebuild-verify` is the exhaustive route
for exactly those: rebuild the corpus from the manifest's own configuration,
regenerate the manifest, and require canonical payload equality. A build is a
pure function of its `BuildConfig` — pinned separately — so the comparison is
exact. It is also the only route that catches an edited *configuration*, because
a build size is a request and the index can only say what came back.

**Two corpus facts the validator surfaced by being wrong about them first.**
`source_ids` is parallel to `groups` *or absent*: a `synthetic_nested_repeat`
motif is its own provenance, and the destroyed splice keeps its recipient's
motifs without their source identities. And a destroyed row keeps a **non-empty**
target block while carrying no action, unlike `generic`, which has neither — the
block is what its length, byte-census and namespace comparisons are made over, so
a destroyed row with an empty target is a row whose intervention left nothing to
audit. Both are now required rather than assumed.

The manifest shape did not change, so the frozen corpus was not refrozen. Its
payload digest is still `41c7821b…` and protocol v0 is still `edc8b753…`; an
exhaustive rebuild reproduced that payload exactly.

**First review, 2026-08-22: corrected.** Four self-consistent mutations were
applied to the frozen manifest after rehashing:

1. rewritten `component` labels plus the matching strata count were accepted;
2. a string in numeric `vm_census` evidence was accepted because arm keys, but
   not value types or internal arithmetic, are checked;
3. a duplicate held-out pair edge was accepted because the list is converted to
   a set without requiring canonical, duplicate-free evidence;
4. a `None` held-out-tuple value raised `TypeError` instead of producing the
   promised complete fault list.

The correction added regressions before implementation, closes all four at both
the Python and file boundaries, and goes beyond them: donor namespaces read the
rebuilt graph, VM arms reconcile seven internal identities and bounds, malformed
self-loop/one-ended/unsorted edges fail, and a path-deduplicated mutation fuzz
erases, retypes and deletes `956` evidence slots. The fuzz promises only a list
of string faults, not rejection of every mutation; some program-byte facts are
deliberately outside the fast index.

Reported verification is `106` focused corpus tests, `185` Direction 4 outcomes
and `1,264` full-suite outcomes, with changed-file Ruff clean. Independent review
reproduced `106` focused and all `1,264` full-suite outcomes (`1,262` pass plus
two data-dependent skips), the unchanged contract digest `edc8b753…`, and fast
acceptance of the frozen corpus at payload `41c7821b…` / file `fb229b31…`.
The implementation report records two exact exhaustive rebuilds at
`data_seed=412653006`; independent review reproduced the same payload once more.
No manifest field or accepted corpus fact changed, so schema 1 and both frozen
artifacts correctly remain untouched.

**Second review, 2026-08-24: corrected.** The fuzz varied shape at one
existing path; it does not vary numeric magnitude. Two valid Python values that
are representable in a parsed JSON body still escape the promised fault-list
API:

1. `vm_census.arms.generic.mean_strokes = 10**400` reaches `float(...)` and
   raises `OverflowError`;
2. a digit-only `stroke_count_histogram` key of 5,000 characters passes the key
   predicate, reaches `int(key)`, and raises Python's digit-limit `ValueError`.

The second case exposed a separate Adapter defect: catching builtin
`ValueError` does not mean “catch an expected refusal,” because validator code
can raise that class too. It was printed as an ordinary refusal while the
`OverflowError` was loud, so exception classification depended on the failed
operation rather than on the contract boundary.

The correction now rejects canonical-domain failures before conversion: ASCII
decimal histogram keys have one spelling, a bounded width and a bounded value;
non-finite, negative and single-field unbounded means are named as domain faults;
and inconsistent integer identities return before their quotient is computed.
The implementation report records all `17` new parameterized cases failing
before the patch moved; the final tree cannot independently prove that red
state, so it is provenance from the work session rather than a rerun result.
Fault messages use bounded representations. `ManifestRefused(ValueError)` is
now the sole expected file-boundary refusal; `load_manifest` raises it for JSON,
digest, structure, verdict and acceptance failures, and the Adapter catches only
it. A monkeypatched builtin validator `ValueError` propagates, as a defect must.
The tolerant summary path separately catches `OverflowError` because its job is
to print what it can from malformed evidence, not decide validity.

The fuzz now applies shape, extreme-number and extreme-key mutations over `965`
deduplicated paths: `8,194` mutations rather than roughly `2,900`. The
implementation report records six named forgeries replayed against the frozen
manifest, all returning faults rather than raising. Reported verification is
`122` focused corpus tests, `201` Direction 4
outcomes and `1,280` full-suite outcomes. Independent review reproduced `122`
focused and all `1,280` full-suite outcomes (`1,278` pass plus two data-dependent
skips), changed-file Ruff clean, the unchanged `61` repository findings, both
contracts, fast acceptance at payload `41c7821b…` / file `fb229b31…`, and one
more exact exhaustive rebuild at `data_seed=412653006`.

**Third review, 2026-08-24: corrected and independently closed.** Single-slot magnitude fuzz is
still not a composition test. A count-consistent arm was constructed with
`programs`, valid/halted counts, histogram mass, `total_strokes` and `vm_steps`
at `10**400`, a one-stroke histogram, and `mean_strokes=10**400`. The seven
integer identities and `0 <= recorded <= total` pass; `float(recorded)` then
raises `OverflowError` before the later case-index mismatch can be reported. The
mean is wrong — its expected value is one — but malformed evidence must produce
that arithmetic fault, not escape the validator.

The present interpreter bound also needs correction while this site moves.
`CURVE_STEPS` bounds points added by `CURVE`; it does not bound strokes emitted
per step. `Trace.strokes` grows only through `flush()`. There is at most one
flush per executed instruction plus finalization, and a successful final flush
requires the last instruction not to have flushed, so successful flushes cannot
exceed executed steps. The default census's per-program bound is therefore
`DEFAULT_FUEL`, not `DEFAULT_FUEL * CURVE_STEPS`.

The correction uses
`0 <= mean <= min(total_strokes, MAX_STROKES_PER_PROGRAM)` before conversion,
sets the bound to `DEFAULT_FUEL`, and keeps the early return when integer
identities fail. A small table-driven arm factory derives every dependent counter
from its independent inputs, so the coordinated `10**400` case preserves the
algebra rather than falling back to single-slot mutation. The validator returns
a bounded mean-domain fault, `load_manifest` raises `ManifestRefused`, the
Adapter exits 1, and the injected validator exception still propagates.

Independent closure review reproduced `125/125` focused corpus tests, `204`
Direction 4 outcomes, and the full suite at `1,281` passed plus two data-dependent
skips in `208.05s`, with only the pre-existing planner warning. Changed-file
Ruff, both contracts and fast audit pass; the coordinated attack was rehashed and
replayed against the frozen 44,033-case manifest, returning the named fault and
`ManifestRefused`. The exhaustive rebuild again gives payload `41c7821b…`, file
`fb229b31…` and protocol `edc8b753…`. The repository-wide Ruff baseline remains
the same `61` unrelated findings. R1.1 is closed without a manifest, schema or
protocol refreeze.

The totality claim is intentionally the file format's domain: bodies reachable
through this runtime's JSON parser. CPython refuses an integer literal beyond its
configured digit limit while parsing, and `load_manifest` classifies that as
unreadable JSON. A manually constructed Python dictionary can contain a larger
integer and still make a raw diagnostic `repr` raise; if `validate_manifest` is
ever promoted to a hostile in-memory-object API, every diagnostic formatter must
be hardened before making that broader claim. This does not affect the file
boundary used by audits or training authorization.

### 12.4 R2 pre-implementation review — queued before authorization

R1.1 is closed, so R2 is the next bounded package, but it still requires explicit
authorization. Its dependency boundary is important: inference must not import
the corpus builder merely to execute an action. Introduce a small `dm.relation`
runtime package for the action value,
prefix validation, exact transducer and typed faults; keep construction,
acceptance and manifest validation in `dm.data.relation`. Re-export only where
compatibility needs it, and require the frozen corpus to rebuild identically
after any extraction.

The executor derives output from the prefix and action key, validates before
mutation, and returns the entire copied block plus accounting. Oracle and
predicted modes share it; only their support policy differs. Exhaustive fixtures
and whole-small-corpus replay precede cache integration. R2 stops after
byte/VM/render equality and byte-at-a-time cached-versus-uncached parity; it adds
no head, parameter, loss, training loop or result.

Two cache facts are now explicit acceptance risks. First, `LayerCache.n` is one
depth for a whole batch, so rows with different copy lengths cannot be block-
filled by padding the shorter context. Use a per-row deterministic byte queue at
synchronized output positions unless a true per-row cache is implemented and
tested. Second, the existing attention path disables causality when a query
block is shorter than its populated cache; a multi-byte block fill would let an
early copied position attend to a later copied byte. Byte-at-a-time fill is the
correctness baseline. A block optimization requires an explicit prefix-offset
causal mask and exact state/logit parity before it is admissible.

### 12.5 R2 authorization — 2026-08-25

The project owner explicitly authorized **only** the R2 package described in
§8 and reviewed in §12.4: a pure `dm.relation` runtime package containing the
target-free action key, frozen support policies, prefix validation, exact
transactional transducer, typed faults, and deterministic heterogeneous
per-row byte queues. `dm.data.relation` may re-export the runtime values for
compatibility, provided the frozen corpus construction, case index, manifest,
protocol-v0 JSON and rebuild output remain byte-identical.

The authorization is append-only and does **not** alter
`copy-relation-protocol-v0.json`, whose `authorized_stages` remains `["R0",
"R1"]`. Executable authorization in a protocol would require a new protocol
version. R2 remains forbidden from adding a relation head or parameters, a
loss/training path, checkpoint schema, model decode integration, scientific
scoring or an outcome-bearing run. Oracle support is explicitly finite at total
counts `2, 3, 4, 5, 6`; it shares the executor with predicted support and never
weakens structural, canvas, or `max_len` validation.

### 12.6 R2 correction review — 2026-08-25

The first R2 implementation was sound for accepted, well-typed R1 actions but
did not yet prove every runtime clause. Before R2 closure, the correction:

- restricts every prefix instruction to ISA `Tier.L0`, refusing `CALL`, `COLOR`
  and control scopes while retaining the distinct `HALT`-in-source refusal;
- validates every public `CopyActionKey` field before use, including rejecting
  booleans as integer coordinates/counts and malformed `Transform`/`D4` values
  with `TransducerFault` rather than leaking `TypeError` or `AttributeError`;
- calculates `(total_count - 1) * source_bytes` and refuses an overlength action
  before an affine rewrite; and
- replaces the synthetic queue fixture with real `execute_copy(...).appended`
  output encoded through `ByteCodec`. It fills the existing cache one symbol at
  a time and compares both normalized top-layer state and next-token logits to
  uncached full forwards. Its sampler spy is invoked only for rows where the
  synchronized queue returns `None`, proving copied rows do not consume token
  randomness and no padding token is supplied as context.

The focused R2 suite is `243` tests and the combined R0/R1/R2 suite is `447`.
This correction still adds no relation model integration, parameter, loss,
training route, checkpoint, scorer or scientific result.

### 12.7 R3 pre-implementation package — historical specification

R3 was the next bounded implementation package when this section was written.
It fixed its engineering scope before any model output existed and was **not
authorization** by itself; the subsequent explicit authorization and result are
recorded in §§12.8–12.9. It must not edit protocol
v0, whose `authorized_stages` remains `["R0", "R1"]`, and it cannot authorize
R4 training, decoder integration, a checkpoint-producing run or scientific
scoring by implication.

#### Objective and boundary

R3 makes the R2 action *scoreable* without yet making it trainable or executable
inside `DrawingLM.generate()`. It adds:

1. `relation_schema="none"|"span_affine_v1"`, appended as the final defaulted
   `Config` field;
2. an opt-in bounded neural head over normalized top-layer boundary states;
3. raw EMIT/COPY, span, D4, `dx`, `dy` and count logits;
4. a numerically stable per-boundary joint-valid-equivalence NLL primitive; and
5. a deterministic k-best decision primitive capped at 32 expansions and
   validated through the R2 executor.

R3 explicitly excludes the relation dataset adapter, reachable-boundary batch
construction, action-loss reduction/weighting, optimizer integration, checkpoint
or record production, boundary-state decode-cache ownership, queue integration,
`predicted_copy`/`oracle_copy` generation, evaluation cases, thresholds and any
outcome. `dm/train_relation.py`, `scripts/relation_eval.py` and model-bearing run
artifacts remain absent.

#### Head algebra and exact parameter surface

Let `N` be the head's learned RMSNorm, applied to normalized top-layer states;
let the frozen rank be `R=32`. For query state `h_b` and candidate endpoint states
`h_s`, `h_e`:

```text
q_b   = W_query N(h_b)
k_i   = W_start N(h_s) + W_stop N(h_e) + E_length[length_bin_i]
z_i   = <q_b, k_i> / sqrt(R)
```

The gate and D4/translation/count factor heads read `N(h_b)`. Every matrix is
bias-free. The module owns exactly:

```text
norm.weight
query.weight
key_start.weight
key_stop.weight
length.weight
gate.weight
d4.weight
dx.weight
dy.weight
count.weight
```

At `D=128` this is exactly `15,104` parameters: the relation-capable model is
`839,808`, below the frozen `857,472` cap with `17,664` headroom. An extra bias,
calibration scalar, hidden layer, threshold or temperature is an architecture
change, not an implementation detail, and is refused in R3.

#### Packed candidate contract and state indexing

The scorer accepts a bounded packed representation, not a dense span square:

```text
query_states       [Q, D]
start_states       [S, D]
stop_states        [S, D]
length_bins        [S]
source_start/stop  [S]
row_splits         [Q + 1]
```

`row_splits[q]:row_splits[q+1]` owns query `q`'s candidates. Per-query counts are
bounded by `RELATION_MAX_CANDIDATE_SPANS=768`; the current model-blind structural
bound is 320. Packed segment normalization makes the span denominator contain
only real candidates. It also keeps work `O(S)` and forbids both an unrestricted
`T x T` byte-span tensor and silent padding in the action distribution.

Candidate construction remains caller-owned. Production code in `dm.models`
must not import `dm.data.relation`; tests may use the corpus builder to prove the
compact interface represents its exact candidate set. The byte/BOS indexing is:

- program boundary byte offset `0` uses the BOS state at model position `0`;
- boundary offset `b > 0` uses model position `b`, the state after byte `b-1`;
- source endpoints and the queried boundary use this same convention.

Hand-assembled off-by-one fixtures must pin all three mappings. A candidate with
bad shapes, dtypes, row splits, source ordering, length-bin index, query ownership
or more than 768 spans fails before projection or allocation of a larger result.
A query with no candidate has a valid EMIT gate distribution and no span
denominator; it can never carry positive action supervision.

#### Compatibility and initialization

`relation_schema="none"` creates no module and no parameter. An old config that
omits the field reconstructs as `none`; its state dictionary loads with
`strict=True`; and the existing frozen baseline states and logits remain
bit-identical. Unknown schemas fail closed. A non-`none` relation schema with a
non-`none` `feedback_schema` is refused during configuration, before a model can
be constructed or trained under both mechanisms.

The optional relation module is registered last and constructed inside
`torch.random.fork_rng(devices=[])`. Shared parameters are initialized before
relation parameters. `share_non_embedding_init` skips `relation.*` during its
shared loop and initializes the relation block at the tail. For one seed, every
shared tensor and ordinary forward logit in `none` and `span_affine_v1` must be
exactly equal; relation tensors remain reproducible and vary across seeds.

`DrawingLM.forward` and `DrawingLM.generate` retain their current arithmetic and
signatures. A narrow explicit method delegates already-produced top-layer states
to the optional relation module and refuses a model with no head. Merely owning a
relation head cannot change ordinary token logits or make standard decoding read
relation weights.

#### Joint-equivalence probability and loss

For one valid action tuple `a`:

```text
log p(a) = log p_span(span_a)
         + log p_d4(d4_a)
         + log p_dx(dx_a)
         + log p_dy(dy_a)
         + log p_count(count_a)

L_action_joint = -logsumexp({log p(a) : a in valid(boundary)})
```

The implementation works in log space and accumulates normalization/logsumexp in
float32 under reduced-precision execution. It returns one unreduced value per
positive boundary; R4 owns denominators, gate reduction, the `0.1` coefficient
and combination with byte NLL. Equivalent-action order is irrelevant. Duplicate
action keys are refused or canonicalized before `logsumexp`, because counting one
derivation twice manufactures probability mass. Empty, cross-query, out-of-
support or non-candidate equivalence entries are typed failures.

Tests must distinguish this loss from the forbidden per-factor marginal loss:
two valid correlated tuples plus their two invalid cross-products are the minimal
fixture. Increasing probability on a valid action cannot increase NLL; permuting
the equivalence set cannot move it; and the packed result must match brute-force
enumeration at tight tolerance with finite gradients into every head component
and selected trunk state.

#### Bounded decision primitive

R3 implements search independently of decoding. Each conditional factor is
ordered by descending log probability with canonical-index tie-breaking. A
best-first heap explores the Cartesian index lattice without materializing it,
visits at most `RELATION_KBEST_ACTIONS=32` tuples, converts each to a target-free
`CopyActionKey`, and validates it through the R2 predicted-support executor.
Counts 5/6 remain oracle-only and cannot be selected.

The declared greedy comparison is hierarchical:

```text
score(EMIT)   = log p_gate(EMIT)
score(COPY a) = log p_gate(COPY) + log p(a)
```

Invalid-copy mass is not renormalized onto valid actions: allocating it is a
model error later measured as missing valid-action mass. EMIT wins when its score
is larger or when no valid tuple is found within the frozen expansion cap. There
is no confidence threshold and no silent independent-argmax/fallback path. The
primitive may return R2's validated `CopyExecution`, but R3 never enqueues or
emits its bytes.

Small exhaustive and randomized fixtures must prove exact agreement between the
bounded algorithm and brute-force ordering whenever the winning valid action is
inside the expansion cap. Adversarial fixtures pin ties, all-invalid prefixes,
EMIT wins, COPY wins, cap exhaustion and the exact number/order of executor
calls. Complexity is recorded analytically and with a non-gating CPU benchmark at
candidate widths 320 and 768; wall-clock timing is engineering evidence, not a
scientific threshold.

#### Files and implementation order

After authorization, work proceeds tests first:

1. add `tests/test_relation_model.py` with strict expected interfaces and
   failures before production code;
2. add `dm/models/relation.py` with packed types, the head, stable segmented
   probabilities, unreduced joint-valid NLL and bounded search;
3. modify `dm/models/transformer.py` only for the final defaulted schema field,
   fail-closed coexistence rule, exact analytic count, tail registration,
   initialization and explicit state-to-head delegation;
4. extend compatibility/contract tests so the model and protocol constants,
   tensor names and parameter arithmetic cannot drift; and
5. update the live handover after independent review, without creating a
   checkpoint or advancing authorization to R4.

No production change to `dm/data/relation.py` is expected. If implementation
reveals that a runtime candidate helper must move, stop and review that extraction
as a named boundary change; require fast audit plus exhaustive rebuild and frozen
payload/file/protocol digests before accepting it.

#### Acceptance and stop conditions

R3 closes only if:

- all old config/state fixtures load strict and reproduce exact baseline logits;
- `none` owns zero relation tensors and `span_affine_v1` owns exactly the ten
  named tensors and 15,104 parameters at `D=128`;
- same-seed shared tensors and standard logits are bit-identical across schemas;
- feedback/relation coexistence and unknown schemas fail before construction;
- packed scores, segment probabilities, joint-equivalence NLL and bounded search
  match independent brute-force fixtures;
- malformed layouts, duplicate equivalences, out-of-support values, invalid
  actions and candidate/search caps fail closed;
- every intended head/trunk path has finite nonzero gradient and no padded or
  unrelated candidate affects a denominator or gradient;
- model code imports neither corpus construction nor training/evaluation code;
- the focused R0/R1/R2 suites, shared model/feedback/cache tests and full suite
  pass; changed-file Ruff and `git diff --check` pass; and
- protocol `edc8b753…`, corpus payload `41c7821b…`, corpus file `fb229b31…` and
  `44,033/44,033` fast acceptance remain unchanged.

Any default-path drift, parameter mismatch, corpus dependency, dense span square,
padding denominator, equivalence overcount, non-finite/zero intended gradient,
protocol/corpus movement or need to modify training/generation reopens the R3
design instead of being patched around. R3 completion is an engineering result:
it supports no statement about whether a model learns, predicts or benefits from
the relation action. R4 remains a separate decision.

### 12.8 R3 implementation and verification — 2026-08-25

The project owner authorized the bounded R3 package in §12.7 only. The completed
implementation adds `dm/models/relation.py`, `tests/test_relation_model.py`, the
final defaulted `Config.relation_schema`, exact analytic parameter accounting,
tail registration/initialization and `DrawingLM.score_relation()`. It adds no
dataset adapter, training objective, optimizer path, checkpoint, record,
generation/decode/queue integration, evaluator, threshold or outcome.

`span_affine_v1` owns exactly the ten declared bias-free tensors and 15,104
parameters at `D=128` (`839,808` model parameters). Packed candidates validate
shape, dtype, ownership splits, cap, source ordering/byte length and length-bin
agreement before projection; no padded denominator or dense span square exists.
The NLL uses float32 log normalization/logsumexp over de-duplicated joint action
sets. The deterministic heap explores at most 32 candidates, applies canonical
tie ordering, and validates each COPY only through R2 predicted support.

Initial verification: the focused R3 suite passes `10`; the full suite passes `1,567`
with `2` data-dependent skips in `208.4s`; changed-file Ruff and `git diff
--check` pass. `scripts/relation_contract.py --verify` preserves protocol
`edc8b753437a67876d5f5c92451e8a7b735cba0f95e20b517041ba92cbcadde5`; the fast
corpus audit retains `44,033/44,033` acceptance, payload
`41c7821b39a32f1c1c6a9568fba5d01a30b9374bb9978c0b83e9714fd9463c66` and file
`fb229b314e8682c72b77fbf8c332d79a631df52526fcd910642d7b99a8ffe69d`.
R4 remains a new, separately authorized decision.

### 12.9 R3 correction review — 2026-08-25

Review found fail-closed and closure-evidence gaps. The correction preflights the
prefix before every EMIT/no-candidate exit and permits only an explicit whitelist
of candidate-local source/canvas/length faults to count as invalid COPY mass;
`ACTION_TYPE`, unsupported-factor and new fault codes propagate. Packed state
validation requires one floating dtype equal to the head dtype before projection.
Equivalence fields validate before canonical-key deduplication, so malformed or
unhashable values raise `RelationLayoutError` rather than builtin `TypeError`.

The focused suite now has 22 tests. It binds all model constants, including the
32-expansion cap, to R1; independently checks the span algebra; exercises a real
769-span cap refusal; pins BOS/byte boundary indexing; executes non-gating CPU
packed-score checks at widths 320 and 768; and verifies complete executor-call
sequences/full selected actions for 100 randomized brute-force landscapes. A
compensating-tie fixture now pins canonical action, rather than factor-rank,
ordering; EMIT, COPY, all-invalid, cap exhaustion and typed structural failures
are each direct fixtures.

Independent closure verification reproduces `22 passed` for R3, `382 passed`
for R3/R2/feedback compatibility, and `1,581 passed, 2 skipped` repository-wide
in `211.75s`. The previously written `384`/`1,583 passed` totals counted the two
skips as passes and are retracted here. Changed-runtime Ruff, `git diff --check`
and documentation links pass. Protocol `edc8b753...`, corpus payload/file
`41c7821b...` / `fb229b31...` and `44,033/44,033` fast acceptance are unchanged.
R3 is closed as an engineering package; at the time of this correction review,
R4 remained separately unauthorized. The later authorization is §12.10.

### 12.10 R4 authorization and pre-implementation package — 2026-08-28

The project owner explicitly authorized R4. The full implementation, acceptance
and correction-review contract is
[`copy-relation-r4-plan.md`](copy-relation-r4-plan.md). It resolves R4's scope as
candidate-batch construction, exact loss reduction/weighting, paired
optimization, accounting, provenance-bound checkpoint/resume and non-outcome
decoder integration. The latter is included because the current `PLAN.md`
resume point assigns it to R4 and R5 cannot exercise `predicted_copy` or
`oracle_copy` before a decoder seam exists.

The package begins with one named boundary change: extract prefix-only candidate
span enumeration from `dm.data.relation` into the target-free `dm.relation`
runtime so inference does not import the corpus builder. `dm.data.relation`
retains derivations, future-target inspection, construction and acceptance. The
extraction is accepted only if all-case enumeration, exhaustive rebuild,
protocol `edc8b753…`, corpus payload/file `41c7821b…` / `fb229b31…` and fast
acceptance `44,033/44,033` remain exact.

R4 freezes no new architecture or scientific rule. It retains R3's ten tensors,
15,104-parameter overhead, packed span cap 768, structural bound 320, k-best cap
32 and greedy EMIT/COPY decision. Training uses separately normalized byte,
reachable-gate and positive-joint numerators; every non-PAD copied byte keeps
ordinary byte supervision. Decoder COPY bytes advance the causal cache one byte
at a time through R2's synchronized queues, never consume token randomness and
never pad an active row's causal context.

This authorization permits implementation, temporary test checkpoints and the
verification ladder only. It does not alter protocol v0 or its
`authorized_stages=["R0", "R1"]`, and it does not authorize R5, a persistent
model-bearing artifact, development/scientific case output, protocol v1/v2,
stochastic action promotion or any threshold/architecture/loss-weight change.
R4 completion will be an engineering result and cannot be quoted as evidence
that the head learns or improves generation.

### 12.11 R4 implementation audit and correction — 2026-09-06

The project owner requested assessment, planning and execution of the next
implementation package, with the MSc portfolio mission explicit. The worktree
already contained R4 candidate extraction, training adaptation, loss/optimizer,
checkpoint/resume and decoder code. The handover's claim that these were absent
is retracted; their existence did not satisfy the full R4 closure contract.

The next bounded package is [the R4 correction](copy-relation-r4-correction.md).
It replaces per-span autograd indexing with batched gathers, caches immutable
case metadata within a trainer, validates query/target ownership and separately
reduced objectives, reconciles sampler/RNG/history/accounting evidence, and
stores decoder boundary values in owned preallocated memory. Regression tests
exercise historical byte-update parity, fresh-process resume, predicted COPY
and unequal oracle queue lengths with cached/full-forward state and logit parity.

Raw engineering measurements and verification results are linked from that
record and [the live plan](../PLAN.md). The living plan was compacted with its
previous decisions and detailed negative-result history preserved in
[the August archive](history/plan-2026-08-28.md).

At this September 6 snapshot R4 remained in flight until its schedule, factor/component diagnostics,
phase resource/work accounting, complete acceptance matrix and closure reviews
are delivered. This correction creates no scientific model checkpoint, changes
no head equation or protocol threshold, and supports no learned-generation claim.

On 2026-09-07 the owner requested a detailed, file-backed plan for remaining
points 1–3 only. [That specification](copy-relation-r4-training-plan.md) records
the observed implementation gaps, invariants, algorithms, passing criteria and
test matrices. Later the same day the owner requested its execution. The
implementation (its §8) charges every retained metadata object under a
published method with a 16 MiB default beside the case/span limits; reuses the
historical `lr_at` cosine schedule under required `steps`/`warmup` and binds
seeds to the frozen table or an explicit `engineering` namespace; extracts the
historical bucket-epoch algorithm into one `dm.data.dataset.bucket_epoch_plan`
shared by the old sampler and the R4 cursor, seeded per epoch by the model
seed; refuses a further update at the horizon before any state moves; reports
detached per-factor valid-value NLL sums every step and aggregate shared-trunk
component-gradient norms at the precomputed `{1, warmup, warmup+1, steps}`
cadence, measured by `autograd.grad(retain_graph=True)` and vector-accumulated
across microbatches; and moves temporary bundles to schema 2 with full
history/sampler/LR/RNG/diagnostic reconciliation, refusing schema 1 outright.
Diagnostics on/off and every cache policy are numerically invisible to the
update; the historical byte-step witness still holds exactly. The worker
reported 1,762 full-suite passes with no skips. Initial independent review
reproduced 1,760 passes / 2 accelerator-dependent skips / 4 warnings; continued
review on unchanged production source now independently reproduces 1,762 passes /
0 skips / 4 warnings with MPS available to its test process. R4 probes stay CPU. Protocol, corpus payload/file digests and
44,033/44,033 acceptance are unchanged. Head equations, parameter surface,
supports, caps, coefficient and decoder are untouched. Decoder-policy,
resource-accounting, hardening and closure work (points 4–7) remains outside
this slice; no learned-behavior claim follows from it.

The subsequent [independent review](copy-relation-r4-training-review.md)
supersedes the points 1–3 completion claim, not their implemented behavior.
It reproduces 214 focused passes but finds non-transactional cache admission,
unbound seed provenance, malformed-domain builtin exceptions, permissive trunk
identity and impossible diagnostic work evidence. Component-gradient overhead
has not been measured by the metadata/gather benchmark. Those named corrections
and the missing measurement precede acceptance of this training slice; the
review applied no production fix and supports no scientific outcome claim.
Continued review adds TREV-7 (failed update skips a batch on retry and can publish
invalid continuation) and strengthens TREV-5 (coordinated denominator copies can
agree yet contradict verified rows). The [immediate correction plan](copy-relation-r4-training-correction-plan.md)
owned C1–C7: regressions/boundary repairs, direct parity witnesses and matched diagnostic
overhead. All six boundary findings (TREV-1–5, TREV-7) and the measurement gap
(TREV-6) are resolved and acceptance-closed on 2026-09-07: row facts recomputed within
schema 2, fail-stop lifecycle enforced, single provenance rule bound, exact trunk verified,
and diagnostic overhead measured. Verification: **299 focused passes**, **1,847 full-suite
passes, zero skips, four warnings**. That September 7 snapshot closed points 1–3;
subsequent Point 4 and F-4 reviews closed the remaining engineering work.
The current R5 implementation/review handoff is in the
[live whiteboard](copy-relation-r4-readiness-package.md).
