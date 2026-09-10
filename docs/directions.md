# Directions — Direction 3 closed, next focus pending

**Status 2026-08-21.** Directions 1/2 are closed. Direction 3's bounded pilot is
**closed at the tested scale**. F5b ran six full-budget development cells and no
eligible schedule passed both seeds. F5c, the one allowed gain-calibration
correction, ran three more cells; every cell failed `generic_guards` and
`stability`, so the predeclared terminal rule closed the line. Protocol v1 was
never written and F6/F7/F8 are not authorized.

The historical decision artifact emitted `incomplete_nondeterministic`, but its
whole-checkpoint-file reproducibility comparator is invalid: `torch.save`
container names and replicate-specific `record_name` differ even when model
states do not. A post-closure tensor-content audit nevertheless finds all 34
state tensors different. Closure stands on the independent gate failures, not
on the malformed comparator. No pilot or destroyed checkpoint, `G`, `Delta`,
`I_structure`, `I_representation` or relation result exists. Reviewed decision:
[`feedback-stability.md`](feedback-stability.md) §10 and
[`PLAN.md`](../PLAN.md).

Evidence: [`state.md`](state.md), [`context.md`](context.md),
[`context-audit.md`](context-audit.md), [`evidence.md`](evidence.md).

## 1. Project question and settled evidence

Paper-level question:

> How do sub-million-parameter models learn, use and fail to use structure in a
> compact executable drawing language?

Keep four evidence layers separate: **representation**, **observation**,
**causal use**, and **generation**. Direction 3 adds a fifth question,
**computational accessibility**: does returning a fully processed latent state
to shallow layers make an already available relation easier to use?

| direction | closed conclusion | consequence |
|---|---|---|
| 1 — execution-state inputs | no material state-bearing relevance under tested gate | do not revive G/C/E state arms |
| 2 — causal context | large controlled teacher-forced relation use; smaller controlled generation preference; weak exact capability | reuse instrument and controls |
| 3 — latent feedback | package did not qualify at this scale; no relation evidence | closed; any future feedback work is a new direction, not another correction |

### 1.1 Direction 2 closure, summarized

Both named checkpoint pairs return confirmatory `relational_context_use`:

| venue | `Delta` seed 0 / 1 | component CIs | controlled generation `Delta_gen` | exact `hit_own` |
|---|---|---|---|---|
| composed shape | `+4.2796 / +4.7193` | `[4.126,4.428] / [4.392,5.116]` | `+0.0337 / +0.0315` | `1.3% / 1.1%` |
| flat step | `+1.3337 / +1.2839` | `[1.163,1.501] / [1.114,1.462]` | `+0.0824 / +0.0583` | `13.5% / 7.4%` |

Holm-adjusted C3 `p=0.001` on both venues. Donor-target and irrelevant-prefix
controls measure near zero. Distance, proximity, coordinate frequency and
coordinate-bag matching do not explain the effect. C4 uses shared uniforms,
raw decoding and complete denominators; its labels mean detectable controlled
preference, not achieved capability.

Independent audit rebuilt both manifests, replayed both C4 report payloads
bit-identically from commit `ab9e980`, reconciled report/checkpoint/record hashes
and repaired future fail-closed checks. Residual limits carry forward: two seeds
are checkpoint-conditional, motif-equivalence novelty was not persisted, axis
transfer is exploratory, C4 has no SESOI, and composed precision is capped by 21
components. Full detail and exact hashes stay in [`context-audit.md`](context-audit.md),
not here.

## 2. Direction 3 claim contract

### 2.1 Scientific question

Does gated latent feedback improve use of separated relational context beyond
what the sampled-symbol stream supports, while holding checkpoint weights fixed
at evaluation and keeping generic likelihood, validity and compute visible?

The mechanism adds accessibility, not information. Previous top-layer state is
a deterministic function of visible history. Never call a gain new knowledge,
reasoning, a mutable register or greater asymptotic depth.

### 2.2 Hypotheses and estimands

- **H3-feedback:** soft decoding increases controlled compatibility relative to
  standard decoding in the same feedback-trained checkpoint.
- **H3-structure:** that gain is larger after relational training than after a
  model-blind relation-destroyed training control.
- **H3-representation:** structure-specific gain differs between bit and byte
  representation packages. This is exploratory; it is not a vocabulary-size
  claim.
- **H3-generation:** gain survives raw free-running decoding and improves exact
  compatible output, not only teacher-forced preference.

For case `i`, representation `r`, training structure `s`, model seed `m`:

```text
Delta(mode) = D_primary(mode) - D_target_control(mode)
g_i(r,s,m) = Delta_i(soft) - Delta_i(standard)
i_i(r,m)   = g_i(r,relational,m) - g_i(r,destroyed,m)
b_i(m)     = i_i(bit,m) - i_i(byte,m)
```

`G`, `I_structure` and `I_representation` are component-bootstrap means of
`g_i`, `i_i` and `b_i`; contrasts are formed per case before averaging.
`D_block_control` remains a required specificity guard. First-target-token
logits are identical in standard and soft mode after standard prefill, so report
gains by byte/field and report the tail separately; never hide the structural
zero in an aggregate.

### 2.3 Claim labels

- `incomplete`: missing artifact, failed freeze, failed convergence or malformed
  report;
- `unstable_feedback`: any recurrent stability criterion fails;
- `no_viable_feedback_implementation_at_scale`: pre-estimation qualification
  cannot produce one robust generic/stable implementation; no gain claim;
- `no_feedback_gain`: complete/stable pilot but primary interaction does not pass;
- `teacher_forced_feedback_only`: teacher-forced gate passes, generation gate
  does not;
- `structure_specific_feedback`: H3-feedback and H3-structure pass in both named
  seeds with generic guards intact;
- `exploratory_representation_interaction`: preceding label plus
  `I_representation` clearing its exploratory interval/SESOI rule in both seeds.

No pilot label supports a model-population, universal reasoning, vocabulary or
parameter-efficiency claim. A positive pilot authorizes a new confirmatory
protocol; it is not that protocol.

## 3. Mechanism and model architecture

### 3.1 Fusion

Let `e_t` be the tied token embedding and `h_(t-1)` the normalized top-layer
state used by the LM head. `N` is the fusion's own learned RMSNorm,
`fuse.norm`; `rms` is the affine-free operation; `mixin(r, e)` restores the raw
embedding at the positions the caller holds plain. Three explicitly named arms:

```text
glu_v1        : z = mixin(N(W_U h_(t-1) * sigmoid(W_G e_t)),      e_t)
glu_source_v1 : z = mixin(N(W_U h_(t-1) * sigmoid(W_G rms(e_t))), e_t)
glu_source_v2 : r = W_U h_(t-1) * sigmoid(W_G N(e_t))
                z = N(mixin(r, e_t))
```

`glu_v1` is the legacy compatibility arm used by the first engineering smoke; it
is not a source comparison. `glu_source_v1` implemented the unambiguous half of
§3.3/Listing 3 — normalize the embedding before `W_G` — with two distinct
normalization operations and a plain prefix that bypassed the second.

**`glu_source_v2` is the Listing-3-faithful seam and the one that is qualified.**
The appendix listing (Fig. 9) settles what Eq. (4) and the Fig. 2 listing left
open:

```text
 8   h = h + uniform(-delta, delta)
 9   x = glu_cross(shift_right(h), input_rmsnorm_1(e))
10   x = prefix_mixin(x, e)
11   h = model(input_rmsnorm_1(x))
```

Lines 9 and 11 name the *same* module, so one learned norm reads the gate input
and the post-mixin stack input, and line 10's mixin sits between them: a
plain-prefix position reaches the stack as `N(e)`, not as `e`. Line 5's pass 1
(`h = model(e)`) and Listing 2's standard prefill stay raw, and that asymmetry is
deliberate — a pass with no carried state has no mixin to normalize. In this
implementation `feedback is None` is structurally the only condition returning the
raw embedding.

Keeping the gain at `0.02` is *more* coherent under the faithful reading, not
less: with the norm applied to plain positions too, a gain at the embedding's own
RMS is what makes a fused pass's plain prefix the size of the raw-embedding
prefill it exists to imitate. RMS normalization rescales each row, so this is the
right scale and direction rather than an equality. The parameter set is
unchanged — `2D^2 + D`, the same three names — so v0's frozen counts still hold.

The two earlier arms stay loadable and are **not** qualified. They are retained
rather than corrected in place because all three own byte-identical tensor sets:
an artifact written under one loads `strict=True` under any of them and computes
a different function in silence. Proposed fused-output-RMSNorm removal remains
dropped.

One consequence to state rather than discover. Under `glu_source_v2` a soft
decode's prompt carries the standard prefill's *raw* inputs while its generated
positions carry `N(m)`, so the decode has no single-forward equivalent — one
`plain` count cannot express two input normalizations. That is the source's
Listing 2 and Listing 3 side by side, not an implementation artifact, and both
input distributions are trained: pass 1 sees raw embeddings everywhere and later
passes see `N(·)` everywhere.

Both arms use bias-free `W_U,W_G in R^(D x D)`. Hidden state is value; token
embedding is gate. Do not add an embedding shortcut: it lets recurrence-trained
weights ignore feedback and recover ordinary pretraining loss without using the
wide channel.

Use `self.norm(x)` as carried state—the exact tensor consumed by tied head.
Initialize the shared input-norm gain to `0.02`, matching current embedding
initialization; a default gain of 1 would inflate stack input RMS by roughly 50×
before training. Pin this scale in arithmetic/stability tests and preserve it in
shared-init code. Optional class and absolute-position additions occur after
fusion; `W_G` sees token embedding only. Pilot is unconditional, causal,
RoPE-only, so these paths remain compatibility obligations rather than
experimental factors.

### 3.2 Config and checkpoint compatibility

Add `Config.feedback_schema`, default `"none"`; retain `"glu_v1"` and
`"glu_source_v1"` for existing checkpoints and use `"glu_source_v2"` for
qualification and estimation. `contract.QUALIFICATION_FEEDBACK_SCHEMA` is the one
pointer every training, smoke and freeze check reads, so no call site decides for
itself which source arm is current. Inference mode is a runtime argument, not
checkpoint architecture. Requirements:

1. Old config dictionaries omit field and load as `none`.
2. `none` creates no feedback parameters; old state dictionaries load strict.
3. `none` forward/generation are bit-for-bit equal to F0 CPU fixtures.
4. Every feedback arm owns `W_U`, `W_G` and one dedicated trainable RMSNorm gain
   under the same three names. `glu_source_v1` normalizes the gate input without
   an affine parameter; `glu_source_v2` reads the gate through the *same learned*
   norm it applies after the prefix mixin.
5. `Config.n_params()` equals actual trainable count in every schema.
6. Shared non-embedding initialization includes feedback matrices across codecs,
   keeps the norm gain at `0.02`, and leaves tied embedding
   representation-specific.
7. `standard` mode never reads fusion weights or latent state.

At `D=128`, every feedback arm has overhead `2D² + D = 32,896`, including the
RMSNorm gain:

| representation | standard | feedback-capable | overhead | under 1M |
|---|---:|---:|---:|:---:|
| bit | 792,192 | 825,088 | 4.15% | yes |
| byte | 824,704 | 857,600 | 3.99% | yes |
| token | 826,752 | 859,648 | 3.98% | yes |
| token-typed | 990,592 | 1,023,488 | 3.32% | no |

Pilot uses bit/byte only. Do not shrink trunks or add dead parameters to equalize
cross-representation totals; primary standard/soft contrast already has exact
parameter equality within checkpoint.

### 3.3 Runtime modes

- **standard:** one standard prefill; generated token embeddings only;
- **soft:** one standard prefill; each generated token uses fusion with state
  that produced it;
- **fused:** standard prefill followed by one full fused prefill; soft generation
  thereafter. Diagnostic because it doubles prompt compute.

For soft decoding, prompt KV cache and final normalized state come from same
standard prefill. For fused decoding, discard standard-prefill cache, rebuild a
fresh cache from fused prompt inputs, and keep BOS plain. Reusing standard cache
with fused states is invalid.

Latent is request-local and row-aligned. Stopped rows cannot affect active rows;
state never survives request completion. Prompt, monitor, legality, `forbid`,
`top_k`, temperature, classes, pre-generated variates and `on_logits` retain
current ordering and meaning.

### 3.4 Module package and locality

| Module | interface responsibility | implementation locality |
|---|---|---|
| `dm/eval/feedback_contract.py` | frozen schemas, seeds, thresholds, model seam | protocol payload and its digest |
| `dm/eval/feedback_fixture.py` | baseline capture and bit-exact replay | tensor serialisation, cell recipes |
| `dm/models/transformer.py` | standard/soft/fused model behavior | fusion, normalized states, KV lifecycle |
| `dm/train_feedback.py` | deterministic pass plan and multi-pass objective | schedule, prefix mixin, jitter, accounting |
| `dm/data/feedback.py` | paired relational/destroyed corpora | donor derangement, balance, fingerprints |
| `dm/eval/feedback.py` | feedback scoring and inference | sequential NLL, stability subset and gate, paired bootstrap |
| `dm/eval/feedback_qualification.py` | F5b pass/stop over development cells | clause table, schedule verdicts, predeclared choice |
| `scripts/feedback_train.py` | thin training Adapter | protocol validation and invocation only |
| `scripts/feedback_eval.py` | thin evaluation Adapter | hash checks, report path, invocation only |
| `scripts/feedback_gate.py` | thin gate Adapter | load protocol/reports; decision Module owns rules |
| `scripts/feedback_contract.py` | thin freeze Adapter | write or verify the F0 fixture and protocol |
| `scripts/feedback_corpus.py` | thin corpus Adapter | build, publish census, print donors for review |
| `scripts/feedback_smoke.py` | thin smoke Adapter | F5's end-to-end engineering run |
| `scripts/feedback_qualify.py` | thin qualification Adapter | run one development cell; decide among the schedules |

Do not split cache execution from transformer implementation: doing so would
expose private state and make a shallow seam. Training and evaluation are two
real callers of feedback behavior, so model interface is a real seam. CLI
Adapters contain no scientific arithmetic.

## 4. Parallel feedback training

### 4.1 Objective

Pass 1 uses plain inputs and ordinary next-token loss. Pass `k>1` shifts pass
`k-1` normalized states right by one, fuses with original token embeddings,
keeps BOS and sampled prefix plain, then reruns full stack in parallel.

```text
L_K = L_1 + (1/(K-1)) * sum(L_k, k=2..K), K>1
L_1 when K=1
```

Every loss uses same non-PAD targets and token-count denominator. Set feedback
weight `lambda=1`. Later losses backpropagate through earlier states; no detach.
This follows paper objective more closely than summing three unnormalized losses.

For each row/pass, draw plain-prefix length uniformly over valid input positions.
Draw carried-state jitter independently from `Uniform[-0.02,0.02]`. Pre-generate
pass count, prefix masks and jitter on CPU for whole batch, then slice them with
micro-batches. Memory splitting must satisfy `rows * T² * K <= attn_budget`.
Micro-batch count may change memory, never RNG draws, objective or gradient.

### 4.2 Named pass schedules

Three schedules are frozen in `contract.PASS_SCHEDULES` before any cell runs.
`project_progressive_v1` **is** v0's `PASS_SCHEDULE` under a name rather than a
copy of it — two objects that had to stay equal by inspection would not.

| schedule | phases | role | expected passes/batch |
|---|---|---|---:|
| `terminal_mix_v1` | `[0,50%)` one pass; `[50%,100%]` `75/22/3` | eligible, first | `1.1400` |
| `project_progressive_v1` | `[0,50%)` one; `[50%,75%)` `75/25`; `[75%,100%]` `75/22/3` | eligible, second | `1.1325` |
| `two_pass_control_v1` | `[0,50%)` one pass; `[50%,100%]` `75/25/0` | diagnostic only | `1.1250` |

The pass-count uniform is seeded from `(pass_plan, step)` and not from the
schedule, so all three read the same draw against different weights: two cells
differing only in schedule differ only where the schedule differs, and a step
whose draw falls outside every candidate's boundary is one-pass in all three.

Record actual counts *and the schedule name*. The realized histogram alone cannot
separate a schedule from an unlucky draw, so `Accounting` carries
`pass_schedule` and the qualification refuses a cell whose config and accounting
disagree about it.

The terminal `75/22/3` mixture and the progressive introduction are the paper's;
the exact `50%/25%` boundaries are not. Both eligible schedules remain frozen
**project adaptations**, not literal source-condition copies, and sweeping only
the three-pass fraction cannot turn either into a source replication. No schedule
tuning from relational outcomes: F5b never sees one.

Source-condition ledger is explicit. The seam is Listing-3-faithful; the protocol
around it is not, and both statements travel in the same artifact:

| source dimension | Direction 3 condition | interpretation |
|---|---|---|
| pass schedule | project `50%/25%/25%` phases, terminal `75/22/3` mix | adaptation; source phase boundaries unspecified |
| optimizer / parameter groups | project AdamW and its named groups | not a source replication claim |
| LR, cooldown and z-loss | project `TrainConfig` defaults; no source-equivalent cooldown/z-loss package | material difference |
| scale and context | `D=128`, four layers, `max_len=2048` | tiny pilot scale/context, not paper scale |
| batching | program-batched rows with a program-level pass plan | not token-batched source training |
| gate input | `glu_source_v2` applies the same learned `fuse.norm` at the gate | **Listing-3-faithful** |
| normalization placement/affinity | one shared learned norm at the gate and after the prefix mixin, mixin first | **Listing-3-faithful** |
| raw embedding paths | `feedback is None` only: training pass 1 and the standard prefill | **Listing-3-faithful** |
| superseded source arm | `glu_source_v1` stays loadable and is not qualified | retained, not qualified |

Protocol v1 records every row, marks `source_conditions_complete=false`, and
names the qualified schedule alongside them. Three rows now read
`listing_3_faithful` and the rest still read `project_adaptation` or
`material_difference`: the architecture seam is the source's, the training
protocol around it is the project's.

Record separately: optimizer steps, programs/semantic bytes seen, content
symbols, padded positions, one/two/three-pass batches, content-symbol forward
passes, padded forward positions, peak memory, wall time and device. “Same
training tokens” is not “same compute.”

## 5. Experimental design

### 5.1 Continuation-relation-destroyed training control

Use flat synthetic translation-repeat corpus because it supplies Direction 2's
64-case in-support step venue. Both arms share original 100,000-program train
source, unchanged 1,000-program relational validation split and program order.
Only training targets change.

Build destroyed arm without weights:

1. recover exact repeat/copy spans from generator provenance before flattening;
2. group continuation blocks by byte length, opcode/operand skeleton and target
   ordinal;
3. deterministically derange whole continuation blocks among groups;
4. reject identity assignments and any donor preserving source step relation;
5. preserve each program length/skeleton and corpus-wide continuation-byte
   multiset exactly; fail if a stratum lacks a complete derangement—never drop it;
6. persist donor map, rejection census, train/val fingerprints and source digest.

Internal target n-grams remain because blocks move whole; seam bigrams and local
continuity change by design. Report seam byte/bigram imbalance, coordinate
marginals, target ordinal, prefix/target length and donor reuse. Also report
program-level treatment prevalence (`28,977/100,000` on the audited deterministic
build; `71,023` programs are unchanged) and the full-corpus VM fault/stroke
census. The manifest must carry those artifacts before training is freeze-ready.
Freeze tolerances before training. Never retry a seed because balance or later
model score looks inconvenient.

### 5.2 Bounded matrix

| factor | levels |
|---|---|
| representation package | bit, byte |
| training structure | relational, relation-destroyed |
| model seed | 0, 1 |
| evaluation mode | standard, soft; fused diagnostic |

Eight feedback-capable checkpoints. Square trunk, unconditional causal model,
same semantic programs, batch-order seed and 24,000 optimizer steps. Use shared
non-embedding initialization within each seed across representation/structure;
structure arms also share embedding initialization. Bit costs more symbols and
compute—part of representation package and explicitly reported. No
standard-trained controls in estimation pilot; within-checkpoint mode contrast
is cleaner and bounded.

One short smoke is integration-only and cannot authorize v1, enter reports,
move thresholds or select cells. F6 requires an eligible both-seed F5b pass.

### 5.3 Sequential teacher-forced scorer

Direction 2 manifests are immutable byte-level cases and can be encoded under
bit or byte. Reuse synthetic manifest hash `dcc93caf0003…`, but new protocol
owns checkpoints, source hashes and claim rules. Score only
`synthetic_flat_step`; axis diagnostics and composed venue are outside pilot.

`Delta_soft` cannot come from current batched scorer. New scorer must:

1. prefill each case prefix in standard mode;
2. score first gold target symbol from prefill logits;
3. feed each subsequent gold symbol through standard or fused input while using
   same gold history;
4. score all 12 primary/control requests and persist raw symbol NLLs nested as
   three named four-way blocks (`primary`, `target_control`, `block_control`);
5. prove sequential-standard output equals current full-forward scorer;
6. aggregate symbols to semantic bytes before Direction 2 contrasts.

Each target remains teacher-forced under its own history. Report donor-target and
irrelevant-block controls, prefix support, first byte, later bytes and ISA fields.
Do not alter Direction 2 reports or labels.

### 5.4 Free-running promotion

Run only after teacher-forced gate passes under frozen rule. Reuse C4 semantics:
raw structural-mask-off decoding, complete denominators, matched irrelevant edit
and one shared variate block across worlds, controls and standard/soft modes.
Start at 8 draws/world/case; expand all promoted cells to 64 if frozen precision
rule fires. Persist reach, termination, `Delta_gen`, `hit_own`, compatible-prefix
length and relation consistency.

Generation capability guard is paired `hit_own_soft - hit_own_standard >= 0.02`
with paired component/draw interval excluding zero in both seeds. Preference guard is paired
`Delta_gen_soft - Delta_gen_standard >= 0.01` with paired component/draw interval
excluding zero. Both are required for promotion beyond pilot; teacher-forced-only gain is
reported and stopped.

### 5.5 Statistical inference

- Form mode, structure and representation contrasts per case before summaries.
- Use identical connected source/donor component resamples across all modes,
  structures and representations; 2,000 frozen bootstrap replicates.
- Keep model seeds separate. Both named seeds must pass; never pool them into a
  model-population interval.
- Primary family is two `I_structure` tests, bit and byte. Apply Holm at
  `alpha=0.05`; practical floor is `0.02` bits/target-byte, inherited from same
  Direction 2 estimand/unit.
- Require `G_relational > 0` with interval excluding zero in both seeds and
  `I_structure >=0.02` in both seeds for at least one representation.
- `I_representation` is exploratory even if its interval excludes zero and
  magnitude clears `0.02`; bit/byte changes sequence length, entropy, compute
  and feedback-event count together.
- Missing cells/controls or failed hashes yield `incomplete`, never zero.

Generic guards are executable requirements, not constants waiting for a caller:
the exact sequential teacher-forced soft-minus-standard cost over the complete
validation split must be no worse than `+1.0` bits/drawing; paired free-running
valid-halt delta must be no worse than `-0.05`; and truncation-rate increase must
be `<=0`. The valid-halt loss is also passed into the stability verdict, which
applies `STABILITY_GATE["max_valid_halt_loss"] = 0.05`. Shared pre-generated
uniforms keep the two generation modes paired even when stopping differs.

## 6. Implementation stages

| stage | deliverable | passing criterion |
|---|---|---|
| F0 contract | schemas, fixtures, equations, resources, gates | **met**: baseline reproduces bit-for-bit; protocol v0 frozen |
| F1 architecture | fusion and three runtime modes | **met**: Listing-3-faithful `glu_source_v2` hand-checked; both earlier arms bit-identical |
| F2 training | pass planner, loss, mixin, jitter, counters | **met**: objective, shift and split gradients pinned |
| F3 evaluator | sequential scorer, stability, paired inference | **met**: scorer pinned; scored-position masks, frozen long subset, semantic bytes and wavefront gate in place |
| F4 corpus | paired relational/destroyed training data | **met**: audited artifact plus zero-fault acceptance, reconciled prevalence and both digest names |
| F5a repairs | provenance, RNG, generic guards, raw NLLs | **verified 2026-08-18**, no freeze |
| F5b qualification | repaired protocol; three schedules × two seeds | **ran and stopped**: no eligible both-seed pass, label `unstable_feedback` |
| F5c correction | repaired instrument; one lever, one schedule, three cells | **ran and closed 2026-08-20; audited 2026-08-21**: all cells failed generic/stability gates; historical file-byte comparator invalid |
| F6 estimation training | eight complete checkpoints | **not authorized**: no training protocol exists and Direction 3 is closed |
| F7 eval freeze/stability | protocol v2 adds checkpoint/record hashes | **not authorized** |
| F8 evaluation/decision | C3-style table, promoted C4, gate, audit | **not authorized** |

### F0 — contract before code (closed 2026-08-17)

CPU baseline fixture captured from `main`: five cells covering byte and bit
codecs, the conditional/absolute-position tables and both shared-init arms, each
storing config, weights or weight digest, inputs, standard logits, prompted
monitored decode with pre-generated variates and a greedy decode. Tensors are
raw little-endian buffers in base64 inside a hashed canonical JSON body, so the
freeze does not inherit a pickle format. Protocol v0 is generated from
`dm/eval/feedback_contract.py` and compared against the committed file by test,
so prose, code and artifact cannot drift apart.

Frozen: fixture bytes, `feedback_schema` names and default, fusion equation,
carried state, fused-norm gain, model seam, the three feedback parameter names,
seed namespaces and values, objective, pass schedule, jitter, matrix, venue
manifest hash, estimands, SESOI, alpha, multiplicity, bootstrap, generic guards,
stability band, promotion floors, claim labels, decision rule, record/report
fields and expected parameter counts. Explicitly not frozen in v0: source
digests, corpus fingerprints and pilot checkpoint hashes. Engineering smoke
hashes may exist, but they do not identify pilot cells; v2 freezes pilot
checkpoint/record bytes before scoring.

### F1 — architecture, tests first (closed 2026-08-19)

`tests/test_feedback.py` was committed at F0 as 24 strict xfails and F1 removed
the marker; it now holds 31 passing tests. Covered: old config and checkpoint
strict load, refusal to promote an old checkpoint into `glu_v1`, none-path bit
equality against the F0 fixture, `2D^2 + D` in the live module at both tiny and
pilot width, gain at `0.02`, hand-computed fusion, position-wise fusion, no
additive embedding shortcut, carried state equal to the head's input, standard
no-read, plain-position semantics including per-row `plain`, misaligned feedback
refused, soft cached versus uncached, fused cache replacement, first-generated-
token identity between standard and soft under a shared variate block, legality
mask ordering in both modes, stopped-row isolation, request-local latent,
class/absolute-position ordering after fusion, and shared init across
vocabularies.

Three implementation decisions the tests pinned. **The caller aligns the carried
state**: `feedback[:, t]` is the state fused into position `t`, because only the
caller knows where the state came from and a shift hidden inside the model cannot
serve both F2's parallel pass and a one-token decode. **`plain` is a count, not a
mask over `feedback`**: fusing a zero state is `RMSNorm(0) = 0`, which deletes the
position's input, and it accepts a per-row tensor because F2 draws the prefix
length per row. **The fusion is constructed off the global RNG stream** and
registered last, so a feedback-capable model and its no-feedback twin are
bitwise identical in every parameter they share — the same rule the
zero-initialised `pos` and `classes` tables already follow.

**The Listing-3-faithful seam, 2026-08-19.** Nine further tests were written
before `glu_source_v2` existed and are hand-computed against the listing. They
pin: one learned `fuse.norm` computing `N(e)` at the gate and `N(m)` after the
mixin, and differing measurably from the affine-free gate `glu_source_v1` uses;
the mixin landing before that norm, so a plain position is `N(e)` and an
after-norm selection is visibly a different tensor; a fully plain later pass
differing from the standard forward, with the integer and per-row `plain` paths
agreeing bit-for-bit; pass 1 and the standard prefill ignoring all three fusion
tensors even after they are perturbed; the fused prefill normalizing BOS while a
soft decode's first generated logits still match the standard prefill's; the same
three parameter names, the same `2D^2 + D` and the same `0.02` gain; both earlier
arms still computing their frozen expression *including* the raw plain prefix;
and a fused decode agreeing cached with uncached in all three arms.

Two decisions the tests forced. **The plain-prefix shortcut is a property of the
arm, not of the caller**: `_stack_input` may return the raw embedding for a fully
plain block only where `Fusion.plain_prefix_is_raw`, because under
`glu_source_v2` that block is `N(e)` and still a fused pass. **The soft decode has
no single-forward twin** under that arm — its prompt is raw and its tail is
`N(m)` — so its equivalence is pinned against `teacher_forced`, a second
independent cached loop, rather than against a whole-sequence recomputation that
cannot express two input normalizations.

### F2 — training objective (closed 2026-08-17)

`dm/train_feedback.py` owns the pass plan, the objective, the memory guard and
the accounting; `dm/train.py` keeps the one loop and calls in for the step body
when feedback is on. `tests/test_train_feedback.py` is 30 tests.

The shift is pinned against an explicit reconstruction *and* against the wrong
direction failing: a left shift hands position `t` the state of `t+1`, so the
model reads the future, trains well and invalidates every later number. `K=1/2/3`
objectives are exact against hand-built logits, later-pass loss carries gradient
into pass 1's states, padding is excluded from every pass and from the one shared
denominator, and whole against split micro-batches agree to `1e-6` at every `K`.

Three decisions. **Each stream is seeded from `(namespace, step)` rather than
advanced**, so step 17,000's plan is computable without replaying the 16,999
before it — which makes it testable, resumable and identical on any device.
**The plan is drawn from CPU row lengths read before the batch moves**, so an
accelerator's dispatch cannot enter a random draw. **The standard step body is
left untouched** rather than routed through the feedback one: "K=1 is the same
arithmetic" is an argument, not a guarantee, and a schema-4 record has to stay
byte-reproducible. `dm.train.SCHEMA` is deliberately not bumped — incomparability
is carried per record by `feedback_schema`, and a global bump would retire a
closed sweep for nothing.

Measured schedule over 24,000 steps: `0.874 / 0.118 / 0.008` against the frozen
`0.875 / 0.1175 / 0.0075`, mean `1.1334` passes/batch against `1.1325` expected.

### F3 — scorer and repaired stability (closed 2026-08-19)

`dm/eval/feedback.py` owns the sequential scorer, the paired inference and the
stability gate; `DrawingLM.teacher_forced` owns the cached execution underneath
it. The split is the one §3.4 requires: the recurrence needs the KV cache, cache
behaviour stays local to the transformer, and an evaluator reaching into
`LayerCache` would be a seam only in appearance.

Sequential standard matches `score_spans` to `<1e-5` bits per byte -- four orders
under the `0.02` floor -- and reproduces `score_cases` contrast for contrast, so
`Delta_standard` is Direction 2's published estimand computed a different way.
`teacher_forced` and `generate` return bit-identical logits in all three modes,
which binds the scoring and sampling loops to one latent lifetime and one
plain-BOS rule. The first-symbol structural zero is exact and published beside
the tail rather than inside it.

Direction 2's contrast helpers are imported rather than re-derived, including two
underscore-prefixed ones: `dm/eval/context.py` is digest-frozen inside
[`context-protocol-v5.json`](context-protocol-v5.json), so it cannot be widened,
and a re-derivation would risk arithmetic that merely looks like the published
estimand.

Three scripted models pin specificity. A **relation follower** shows a positive
`G` with a null unrelated-block contrast. A **generic soft improver** shows `G`
exactly zero, because the donor-target control subtracts a uniform improvement.
An **edit reactor** shows a positive `G` *and* a positive `Delta` -- the naive
reading passes -- and is caught only by the unrelated-block contrast, which is
as large as its primary. That column is the whole difference between the first
model and the third.

Resamples are built once and shared by every cell, byte-identically to Direction
2's procedure, so paired-cell differences stay aligned. Holm steps down over the
two `I_structure` tests. Malformed, unfinished or equivalence-unchecked reports
raise `incomplete`.

**The four stability repairs, made without reading any new model output.**

*Scored positions everywhere.* `finite` and `max_abs_logit` joined the RMS
quantiles behind the scored mask. A padded logit row is the tied head reading its
own PAD row as a negative class, and `max_abs_logit` over it measured how
confidently the model rejects PAD. The test holds the scored region fixed and
puts a 500x-scaled embedding row in the padded region: the unmasked maximum moves
and the reported one does not. A non-finite value cannot be confined to the
padded region at all — the head is tied, so poisoning any embedding row poisons
that logit column at every position, which is precisely why *positions* are what
get masked.

*A frozen, model-blind, long subset.* `stability_subset` draws
`STABILITY_SUBSET_SIZE = 32` rows from `seed_for("stability")` over the programs
whose scored length exceeds the deepest pass, and persists their indices, lengths
and a digest. It takes a corpus and a codec and has no parameter through which a
model output could arrive. It fails closed rather than shrinking: a shorter
subset converges before the deepest pass and reports sequence length as
convergence, and it would do so most readily on exactly the corpora too short to
test the clause. The F5 smoke's first-sixteen subset had twelve rows fully
converged at pass 32.

*Semantic bytes.* `_bits_per_byte` divides by `positions / symbols_per_byte`, so
the bit arm's eight symbols aggregate to one bytecode byte and the report names
its `cost_unit`. `bits_per_drawing` is a program-level unit and is unchanged.

*The wavefront, and one withdrawal.* The gate reads `update_q95_wavefront` at the
frozen `0.25` and against the pass-8 value. `update_q95_not_settling` is
withdrawn, not merely unmet: the iteration is triangular, so a deepest-pass
quantile over all scored positions falls as rows get shorter and rises as they
get longer whatever the channel does, and no trajectory distinguishes a bad
channel from a long sequence under it. `WITHDRAWN_CLAUSE_REASONS` travels in
every verdict so the shorter failure list is legible from the artifact.

The report carries `STABILITY_REPORT_SCHEMA = 2`, separate from v0's frozen
`REPORT_SCHEMA = 1`, because those field names now mean different things.

`scripts/feedback_eval.py` is deferred to F7: its job is checkpoint and record
hash verification, and those hashes do not exist until protocol v2.

### F4 — audited corpus and its acceptance rules (closed 2026-08-19)

`dm/data/feedback.py` builds both arms from one source. The intervention is
**continuation-relation destruction with exact listed marginals**: whole-block
permutation preserves program lengths, per-program skeletons, byte/coordinate
marginals and block multiset, while intentionally changing seam bigrams and
local continuity. This is not a claim that relation and nothing else changed at
every token-level scale. Generator-side L1 provenance and span verification make
the splice auditable without weights.

Within each `(byte length, skeleton, ordinal)` stratum, shuffle-then-cycle gives
no fixed point and uniform donor reuse; fixed-order repair rejects a donor that
preserves its source relation, and an impossible stratum fails rather than drops.
Every ordinal from 3 on is deranged. The exact invariants are program lengths,
skeletons, byte/coordinate marginals, block multiset and train/val disjointness;
validation is shared and the seam bigram is the intended changed quantity.

Audited deterministic `100,000`/`1,000` manifest is
`runs/feedback_corpus_v1_audited.json`. Canonical payload digest is
`8da5b8fb66311ab6121b5ea95b9912e06459ed451e41c21a2ed2d364cece0006`; actual
file SHA-256 is
`08a53e3b1519698048030bc22547318338a4d9b55128234be21c6204328b5fc9`.
It treats `28,977/100,000` programs, leaves `71,023` unchanged, and records equal
`471,453`-stroke, zero-fault training-arm censuses plus shared validation
(`4,839` strokes, zero faults). Small build is only regression.

**Three acceptance holes closed 2026-08-19.** `audit_accepts` now requires every
arm to be all-valid, all-halted and zero-fault. Paired equality answers whether
the intervention preserved the VM marginals and says nothing about whether those
marginals were acceptable: a control in which every program faulted identically
satisfied it, and the two arms would then differ in the relation *and* in being
broken. `accepts` requires the reported treatment fraction to equal
`treated / n_train`; `treated + untreated == n_train` was checked and the fraction
was not, so three numbers could agree pairwise and not jointly.
`manifest_digests` returns the canonical payload digest and the file SHA-256
under names that cannot be swapped — reformat the file and the first is unchanged
while the second moves — and the corpus Adapter and the freeze both print both.
The audited manifest still loads under all three rules.

The builder imports no model, evaluator or torch code; the Adapter prints the
census and donor sample for human review. Balance changes bump `CORPUS_SCHEMA`;
manifest-interface changes bump `MANIFEST_SCHEMA`, before weights exist.

### F5a — repair verification (met 2026-08-18; no freeze)

Implemented and verified: normalized gate arm, separate training/report paths,
checkpoint hashes, engineering provenance, failed-smoke refusal, sampler/eval
RNG isolation, exact generic likelihood, valid-halt/truncation guards,
`max_valid_halt_loss`, and raw symbol NLL arrays for all 12 requests. Full suite
is `1,030 passed` at the time of the F5b close; v0 and five fixture cells replay bit-for-bit; changed files are
Ruff-clean. A real one-step CPU smoke wrote distinct artifacts, failed generic
likelihood at `+3.6652` bits/drawing, and `--freeze-training` refused it without
writing v1.

Legacy `glu_v1` 12k exact guard is `169.543` standard versus `395.271` soft
(`+225.728`), diagnostic only: it measures neither the destroyed arm nor the
Listing-3-faithful one.
Weight-decay/scale claims remain withdrawn; see
[`feedback-stability.md`](feedback-stability.md).

Verification exposed four pre-freeze defects not covered by passing tests. All
four are closed as of 2026-08-19:

- source norm identity/affinity/placement — resolved by the appendix listing and
  implemented as `glu_source_v2` (§3.1);
- stability padding, seed, bit-unit and unreachable-settling defects — repaired
  (F3 above);
- corpus accepting equal faults, unreconciled prevalence, ambiguous hash names —
  all three are acceptance rules now (F4 above);
- v1's missing final provenance, report versioning, source closure and
  freeze-side strict reconstruction — added (F5b below).

A short smoke proves integration only. It no longer authorizes v1.

### F5b — qualification protocol (ran and stopped 2026-08-19)

`dm/eval/feedback_qualification.py` is the Module and `scripts/feedback_qualify.py`
the Adapter. The Module computes only completeness, provenance, resources,
convergence, generic guards and repaired stability; never Direction 2 cases,
`D`, `Delta`, `G` or interaction outcomes.

**The prohibition is structural rather than conventional.** `qualify_cell`
accepts a closed set of fields and *refuses* a cell carrying anything else — an
unexpected key is how a relation outcome would arrive, and refusing costs an
error while ignoring costs the experiment. Provenance fails closed in both
directions: an engineering smoke is one short cell and cannot qualify anything,
and a **scientific** cell is estimation data whose use here would let the pilot
select its own configuration.

Three further decisions the tests pinned. **Hashes are recorded-against-recomputed
pairs**, and `decide` recomputes them in a later process from the files
themselves: a digest a program computed and immediately compared against itself
proves nothing. **`validation_tail` is pinned against `scripts/sweep.py`'s
existing guard** rather than asserted to match it — the threshold is inherited, so
the arithmetic behind it has to be, and a re-derivation that merely looked right
would make it a different guard under the same number. **The choice among
schedules is an order, not a comparison**: one test gives the second eligible
schedule strictly better numbers on everything the stage is allowed to see and
requires the first to win anyway.

Six clause groups were frozen before any cell ran:

1. the Listing-3-faithful arm, `glu_source_v2`, resolved rather than ledgered;
2. model-blind stability subset: eligible programs longer than pass 32,
   deterministically sampled with `seed_for("stability")`, identities persisted;
3. scored-position finiteness/logit masks and semantic-byte cost units;
4. wavefront q95 gate; `update_q95_not_settling` withdrawn with its reason in the
   artifact; fused/input scale diagnostic until a justified reference exists;
5. zero-fault/all-valid/all-halted corpus acceptance, exact prevalence
   reconciliation, canonical and file hashes;
6. versioned stability and qualification report shapes, `QUALIFICATION_SOURCES`
   hashing every file that trains *or* judges a cell, development provenance on
   every artifact, and freeze-side strict checkpoint/config reconstruction — the
   freeze rebuilds each qualifying checkpoint from its own config and loads it
   `strict=True` in-process rather than reading a boolean the report wrote about
   itself.

Run three frozen byte/relational schedules × seeds `100/101` (six 24k cells),
`data_seed=100`, full `100,000/1,000` scale, byte-disjoint from final seed-0 data
and Direction 2 cases:

- eligible `terminal_mix_v1`: first 50% one-pass, then `75/22/3`;
- eligible `project_progressive_v1`: current `50%/25%/25%` adaptation;
- diagnostic-only `two_pass_control_v1`: first 50% one-pass, then `75/25/0`.

No retry or magnitude ranking. First eligible schedule in listed order qualifies
only when both seeds pass:

- complete final record; strict reload; matching source/config/checkpoint/record/
  report hashes and development provenance;
- `<1M` parameters, no input truncation, reconciled pass/resource accounting;
- final-third validation `|tail| <=0.5` bits/drawing/1k (existing project guard)
  and final-minus-best drift `<=1.0` bits/drawing (generic degradation budget);
- exact validation soft-standard `<=+1.0` bits/drawing; valid-halt delta
  `>=-0.05`; truncation increase `<=0`;
- scored states/logits finite; scored max logit `<100`; hidden RMS p99 within
  `[0.25,4]x` pass-0; wavefront q95 at pass 32 `<=0.25` and `<=` pass 8;
  pass-32 validation increase `<=+1.0` bits/drawing.

Freeze first eligible both-seed pass; negative control cannot authorize v1. No
eligible pass: `unstable_feedback` or bounded
`no_viable_feedback_implementation_at_scale`, never `no_feedback_gain` — that
label requires complete, stable pilot *scores* and this stage has none. Mixed
seed fails that schedule; no retry.

Protocol v1 would have frozen the qualified schedule into all eight estimation
cells, so F6 would train under the schedule that was actually qualified rather
than whichever one happened to be the module default. No schedule qualified, so
no v1 exists. The commands that produced the stop:

```bash
PYTHONPATH=. .venv/bin/python scripts/feedback_qualify.py run \
  --schedule <schedule> --seed <100|101> --device mps \
  --corpus runs/feedback_corpus_v1_audited.json
PYTHONPATH=. .venv/bin/python scripts/feedback_qualify.py decide \
  runs/feedback_qual_*_cell.json --out runs/feedback_qualification.json
```

`--freeze-training` was not run and would have refused: it requires a
qualification naming an eligible schedule that passed both seeds.

**Those exact commands no longer run**, and the reason is F5c below: `--corpus`
is now required and reconciled against the record rather than hashed, and `run`
takes a `--package`. The six reports they produced are preserved unchanged and
are not re-decided.

### F5c — the one correction package (ran and closed 2026-08-20)

F5b's decision rule offered two continuations and no third: close Direction 3 on
the bounded negative, or run **one** further predeclared package addressing the
measured mechanism. This is that package. Its criteria are frozen in
`dm.eval.feedback_contract`'s F5c layer, and it is post-hoc by construction, so
the rules that keep it from becoming a search are part of the freeze rather than
part of the intent.

**Before anything trains, the instrument is repaired.** A 2026-08-20 audit found
five defects in the thing that produced the stop, two of them substantive:
the corpus clause compared a manifest against itself and so never noticed that
all six cells trained on `data_seed=100` while filing the audited `data_seed=0`
manifest; and the Adapter kept its own eight-file source list against the
contract's twelve, so batching, corpus bytes and encoding determined every cell
without entering any cell's digest. Full working, including the three lesser
findings and the corrected schedule-ordering sentence, is in
[`feedback-stability.md`](feedback-stability.md) §7.

The repair is a Module rather than a patch. `dm/eval/feedback_evidence.py` owns
corpus identity, source identity, execution environment, checkpoint
reconstruction, the declared cell shape and report validation — the five
invariants that were half-owned by the Adapter and half by the contract, which is
how a fully green suite missed a corpus mismatch. `qualify_cell` gains a
`corpus_identity` clause, an `environment` clause and a `gain_calibration`
clause, refuses unexpected keys at *every* declared level rather than only the
top, and is parameterised by a frozen `Package` so one function judges both
packages.

**One lever moves: gain calibration.** The arm stays `glu_source_v2` and there is
no `glu_source_v3` — the forward equation is unchanged, so a fourth arm sharing
the same three tensor names and the same arithmetic would be a schema branch
distinguishing nothing a reader could check. What changes is a named
training-protocol condition, `TrainConfig.gain_calibration`. At the predeclared
feedback phase transition — derived from the schedule, `12,000` of `24,000` — the
shared input norm's gain is set once, in every component, to
`g = sqrt(sum_v p_v * mean_d E[v,d]^2)`: the token-frequency-weighted RMS of the
raw embeddings the channel is about to meet, over the training split's input
positions with BOS included and PAD excluded. The formula, the step, the value
and the overwritten gain's four-number summary all land in the record.

The jitter stays at the source's `0.02`. The source specifies no state-relative
rule, so changing it would be a second post-hoc intervention *and* a source
departure; §3d's finding that the perturbation is ~1.1% of the state it perturbs
is recorded as a source-condition ledger row instead. **No threshold moves**, and
`CORRECTION_GATE["thresholds_moved"]` is empty with a test on it.

**One schedule, fresh data, three cells.** `project_progressive_v1`, data seed
`200`, model seeds `200/201`, no schedule comparison. Seed 200 runs twice and seed
201 once; the repeat is a reproducibility control rather than a second sample.
The six F5b cells may motivate this design and may not qualify it, so nothing
here shares a seed or a corpus with them — the measured overlap between the two
corpora is 4 programs of 101,000, all of them 6-byte minimum-length programs
(§7.5 corrects the "byte-disjoint" claim this document used to carry).

Every cell asks torch for deterministic kernels and records the resulting policy
state, along with Python, Torch, OS, device, threads and environment variables.
The historical rule also required the repeated seed's two cells to carry one
checkpoint-file SHA-256. Section 10 of
[`feedback-stability.md`](feedback-stability.md) shows why that comparator is
invalid: PyTorch's serialized container and the embedded `record_name` differ by
replicate even for identical tensor content. Preserve the rule and result
artifact as history; never reuse the comparator.

**The valid terminal branch fired.** All three cells failed `generic_guards` and
`stability`. That is sufficient under the predeclared “any gate failure” rule,
so **Direction 3 is closed at this scale**. No training protocol exists, F6 is
not authorized, and no further gain choices, jitter variants, schedules or
retries are permitted. `incomplete_nondeterministic` remains the immutable
artifact's emitted label, not the reviewed scientific label.

The lever did move — fused/standard input p99 went from `0.27`–`0.29` in F5b to
`0.74`–`0.80` in F5c — and no F5c cell qualified: the exact guard is `+24` to
`+68` against `+1.0`, with pass-32 increase `+100` to `+153`. This rejects the
named calibration remedy. It does not isolate input scale as a universal cause:
F5b/F5c use fresh seeds and corpora, and the shared norm changes both gate input
and stack input. A content-based audit finds all 34 seed-200 state tensors
different, confirming numerical non-repeatability for that pair; its `8.656`-bit
guard difference is one observation, not an MPS noise floor. Full working:
[`feedback-stability.md`](feedback-stability.md) §§9–10.

```bash
PYTHONPATH=. .venv/bin/python scripts/feedback_corpus.py --n-train 100000   --n-val 1000 --data-seed 200 --out runs/feedback_corpus_f5c_audited.json
PYTHONPATH=. .venv/bin/python scripts/feedback_qualify.py run --package f5c   --seed <200|201> --replicate <1|2> --device mps   --corpus runs/feedback_corpus_f5c_audited.json
PYTHONPATH=. .venv/bin/python scripts/feedback_qualify.py decide --package f5c   runs/feedback_f5c_*_cell.json --out runs/feedback_correction.json
```

### F6 — full training (not authorized)

Historical design only. A verified v1 was never written and Direction 3 is
closed. Do not launch these eight cells.

### F7 — content freeze and stability (not authorized)

Protocol v2 hashes final weights/records before any relational score and applies
F5b's repaired fixed-long-subset, scored-position, semantic-byte wavefront gate.
Any generic/stability failure stops outcome scoring.

### F8 — estimation, promotion, audit (not authorized)

Score all frozen cells, run gate once, then apply frozen C4 promotion. Audit raw
rows, tables, source, protocols, records and checkpoints independently. Stop on
null/instability. Positive result creates confirmatory design; do not expand
codecs, venues, passes or capacity inside v2.

## 7. Non-negotiable invariants

1. No model output chooses corpus row, donor, case, threshold or checkpoint.
2. Old `feedback_schema="none"` behavior and strict loading remain exact.
3. Standard and soft compare same weights, prompts, cases and variates.
4. First position is plain; carried state is previous normalized top state.
5. Later-pass gradients are attached; prefix/jitter RNG is micro-batch invariant.
6. Semantic program, symbol, padded-position, pass and wall-clock costs stay
   separate.
7. Raw NLLs and raw completions persist; masks/fusion are named cells.
8. Teacher forcing, free generation and generic validity remain separate claims.
9. Case components, model seeds and sampler draws are never pooled.
10. A training protocol freezes only after a predeclared package passes every
    clause on every one of its cells; immutable v2 freezes checkpoint/record
    bytes before evaluation. Records and reports use distinct paths and never
    overwrite prior artifacts.
11. Runtime source/corpus/config/provenance checks fail closed. Missing means
    `incomplete`; recorded booleans never replace freeze-side recomputation.
12. Canonical payload digest and actual file SHA-256 stay separate and both
    verify. **Neither identifies a corpus.** A manifest's digests identify the
    manifest *file*; only the record's program fingerprint says which programs a
    checkpoint saw, and the two are reconciled exactly. A checkpoint file digest
    likewise identifies one serialized artifact; cross-run reproducibility uses
    a canonical state-content digest, never `.pt` byte equality.
13. Qualification uses separate seeds/data and no relation outcomes; schedule
    choice is first predeclared eligible pass, never magnitude ranking.
14. Bit versus byte is representation-package evidence only. Vocabulary claim
    needs later 1/2/4/8-bit chunks and byte-boundary feedback control.
15. CLI scripts stay thin Adapters; arithmetic belongs in tested analysis
    Modules and cache behavior stays local to transformer implementation.
16. Engineering and development cells have zero scientific outcome value.
17. One authoritative list per invariant. A second copy of the source set, the
    corpus identity or the cell shape is not a second provenance; it is one, and
    the weaker copy wins silently.
18. A post-hoc correction package moves exactly one lever, on fresh seeds and
    fresh data, with its criteria frozen before any cell runs and a terminal rule
    at the end. The cells that motivated it may never qualify it, and a second
    such package is the architecture search the bounded pilot exists to prevent.

## 8. Immediate handoff

**Direction 3 is closed at this scale and there is nothing left to run.** F5b
stopped at `unstable_feedback`; F5c, the one correction package that stop
allowed, ran its three cells and every cell failed the independent generic and
stability gates. There is no training protocol, F6 is not authorized, and `G`,
`Delta`, `I_structure` and `I_representation` were never computed.

What the direction carries forward is a bounded negative about one implementation
and correction package at one scale. The named gain-calibration remedy moved its
target and did not qualify; the broader scale explanation is not isolated. The
repeated seed's model-state content also differs despite deterministic algorithms
being enabled. Its `8.656`-bit guard difference is one observed pair difference,
not a variance estimate or universal MPS noise floor.

The historical result artifact's checkpoint-file comparison is invalid by
construction and must not be reused. Closure stands because “any gate failure”
was terminal and all three cells failed valid gates. The post-closure content
audit and next instrument package are in
[`feedback-stability.md`](feedback-stability.md) §10.

Nothing in this repository should be re-run to change either result. The six F5b
cell reports are never rewritten or re-decided — under the repaired instrument
they would be refused, and that refusal is the audit finding restated rather than
a new result. After any code change:

```bash
PYTHONPATH=. .venv/bin/python -m pytest -q
PYTHONPATH=. .venv/bin/python scripts/feedback_contract.py --verify
git diff --check
python3 /Users/rosastre/.agents/skills/living-plan/scripts/outline.py PLAN.md --links
python3 /Users/rosastre/.agents/skills/living-plan/scripts/outline.py docs/directions.md --links
```

Never rewrite F0 (`--write`), freeze a training protocol, run F6, edit Direction
2 protocols, overwrite reports, re-decide F5b's or F5c's cells, or start a
codec/corpus/schedule sweep. **There is no next Direction 3 authorization.** A
future feedback attempt is a new direction with a new contract; the next
cross-cutting package should qualify device repeatability with state-content
digests before any expensive cell.
