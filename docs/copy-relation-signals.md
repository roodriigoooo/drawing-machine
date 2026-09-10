# Direction 4 — evidence synthesis and figure specification

**2026-09-07 — DOCUMENTATION SYNTHESIS; NO NEW EXPERIMENT OR FIGURE RENDER.**
This note connects existing evidence to the explicit-action hypothesis and
specifies a useful future figure. The scientific authority remains the
[claim inventory](evidence.md), [Direction 2 result](context.md), its
[independent audit](context-audit.md), and the [Direction 4 design](copy-relation.md).
The next implementation package is [R4 Point 4](copy-relation-r4-decoder-plan.md).

## 1. The supported motivation and the untested hypothesis

The strongest motivation is an observed separation: the named small models use
relation-bearing context under controlled interventions, yet exact compatible
continuations remain uncommon when those models generate their own bytes.
This makes an explicit, learned COPY action a well-motivated experiment.
It does not establish that plan inference is already solved, that serial byte
sampling is the cause of failure, or that adding an executor will produce a gain.

Direction 4 tests a specific intervention: infer a whole-instruction source
span, an exact affine step and a total count, then realize that action through
the deterministic executor. The public output remains flat bytecode. Its primary
contrast changes runtime mode within the same relation-capable checkpoint.
The `none` training arm separately tests the effect of auxiliary supervision;
oracle COPY tests execution with supplied actions.

Three evidence classes must remain visibly separate:

| class | what exists | what it licenses |
|---|---|---|
| measured scientific motivation | Direction 2's controlled context-use and generation results | a concrete capability gap in named checkpoints and venues |
| verified engineering conditions | frozen traced corpus, target-free executor, bounded head, corrected training/continuation and cache witnesses | an increasingly inspectable instrument for testing the hypothesis |
| unmeasured Direction 4 outcomes | learned action accuracy, exact-generation gain, generalization and qualified resource cost | future questions only; no numerical Direction 4 effect bar yet |

R4 points 1–3 being acceptance-closed is progress in the second row. It is not a
new observation of the first or third row.

## 2. Quantitative motivation, with units and uncertainty

The following are existing results, transcribed from [Direction 2 §§1 and 4](context.md).
Intervals are the recorded component-bootstrap intervals, conditional on each
named checkpoint; the two seeds are reported separately. C3 uses 28 cases/21
components for composed shape and 64 cases/64 components for flat step. The
controlled C4 reports use 64 draws per world per case; draws are not additional
independent components or model seeds.

| venue | seed | teacher-forced controlled Δ, bits/target byte (95% CI) | controlled generation Δ_gen, normalized distance (95% CI) | exact `hit_own` |
|---|---:|---|---|---:|
| composed shape | 0 | +4.2796 [4.126, 4.428] | +0.0337 [0.0197, 0.0462] | 1.3% |
| composed shape | 1 | +4.7193 [4.392, 5.116] | +0.0315 [0.0159, 0.0497] | 1.1% |
| flat step | 0 | +1.3337 [1.163, 1.501] | +0.0824 [0.0766, 0.0881] | 13.5% |
| flat step | 1 | +1.2839 [1.114, 1.462] | +0.0583 [0.0529, 0.0637] | 7.4% |

Exact-hit percentages here are descriptive point estimates. This table does not
supply new intervals for them or treat the seed range as an interval. A future
chart must recover any uncertainty from the existing report fields and their
recorded resampling unit, or display these as explicitly descriptive values.
Do not invent binomial intervals over all draws while ignoring shared cases.

C3 reports Holm-adjusted p=0.001 on both co-primary venues. Direction 2's C4
primary endpoint remains raw `D_gen`; the controlled `Delta_gen` has its own
predeclared v5 verdict. This note does not retrospectively promote it to primary.
There is no meaningful ratio between the first two metric columns: one is
likelihood in bits and one is a normalized-distance contrast.

Raw record paths and checkpoint/manifest hashes are indexed in
[context.md §7](context.md#7-artifacts) and [the audit](context-audit.md):
`runs/context_c3_{synthetic_byte_dcc93caf0003,composed_byte_2679fc2356da}.json`
and the corresponding `runs/context_c4_*_v5.json` reports. The `runs/` tree is
ignored; this transcription is not a claim to have replayed those model runs.
Use the audited source/record identities if implementing a new chart.

## 3. Relationships that guide the next experiment

### 3.1 Relation-specific influence is useful; universal rule inference is open

Direction 2's donor-target and irrelevant-prefix controls were measured near
zero, and its proximity, frequency and coordinate-bag analyses challenge those
specific alternative explanations. The controlled intervention therefore
provides stronger motivation than an attractive generated drawing alone.
However, teacher forcing supplies the true preceding target bytes, and a
source-motif memory can still contribute. These controls do not establish an
abstract reusable transformation rule or unseen-motif generalization.

Direction 4 responds with explicit source-component separation and novelty
strata: held-out affine tuples and ordered nested compositions are co-primary;
pure motif-pair novelty, affine/pair interaction and motif transfer remain
separate secondary interventions. A gain confined to familiar motifs would
support a narrower claim than a gain on held-out combinations and motif transfer.

### 3.2 Better preference and exact execution are different endpoints

The composed venue has the larger teacher-forced contrast and much lower exact
hits than the step venue. That ordering is a useful descriptive relationship;
it is not a controlled cross-venue causal test because motifs, lengths, supports
and dependence structures differ. It motivates measuring exact target blocks
alongside action decisions, rather than promoting preference alone to capability.

The next package supports this by separating target-free runtime observations
from a later evaluator join. A source-span choice or gate score is not action
accuracy until compared with the complete frozen equivalence set outside the
decoder. Marginal factor scores can identify a weak span/transform/count factor;
they cannot replace joint-valid action mass because valid marginal values may
form invalid joint tuples.

### 3.3 Component diversity determines useful precision

Direction 2's composed estimate has only 21 dependence components. Its audited
sampler-variance decomposition attributes 90–96% of the remaining variance to
between-case uncertainty; those percentages are conditional decompositions,
not estimates of model-population variance ([context.md §4.2](context.md#42-what-the-64-draws-bought-and-what-they-could-not-buy)).
More draws cannot supply absent source diversity.

Direction 4's frozen corpus contains 633 components in `heldout_affine_tuple`
and 377 in `heldout_nested_composition`, both above its proposed 128-component
floor. These are exhaustive manifest counts, not uncertain model measurements;
source: `runs/relation_corpus_v1.json`, identities in [PLAN.md](../PLAN.md).
They make the intended analysis possible; they do not guarantee narrow intervals
or license treating model seeds, components and draws as interchangeable.

### 3.4 Transformation and count claims need identifiable tests

Direction 2's exact rotation diagnostic is badly out of distribution. Its
in-place y-axis null is exploratory because the support rule was revised after
the outcome. Neither licenses a universal claim of affine understanding or
failure. Direction 4 should keep familiar atoms/new combinations distinct from
out-of-support transformations and genuinely new motifs.

Likewise, an unconditional prefix cannot reveal a randomly assigned unseen final
count. Direction 4's predicted count support is 2/3/4; counts 5/6 belong to the
finite oracle diagnostic. The first tests learned choice among supported atoms;
the latter tests execution with a supplied choice. A figure must not label an
oracle count-6 drawing “learned extrapolation.” See
[the identifiability boundary](copy-relation.md#11-the-count-boundary-question-is-partly-unidentifiable).

### 3.5 Execution cost and statistical success must both be observed

The executor can produce exact bytes for a supplied valid key. The accepted
candidate-extraction proof covers 47,550 cases, 2,229,707 boundaries and
327,372,306 candidate entries with zero mismatches; these are exhaustive
engineering counts under the recorded source, not learned predictions
([proof record](../artifacts/relation/r4-candidate-verification.json)).

COPY can replace literal sampling decisions while each delivered byte still
enters the causal transformer before a later decision uses its state. This
relationship motivates a separate byte/action/cache timeline, and is why
Point 4 counts operations before Point 5 measures physical time and memory.
Neither a long COPY nor reduced sampler calls alone demonstrates acceleration.

### 3.6 The negative direction narrows a mechanism, not the research question

Direction 3's one permitted calibration correction did not qualify under the
frozen generic/stability gates. Its relation estimands were never computed.
This is evidence against that tested implementation and remedy at that scale;
it does not show that explicit COPY will work or establish that latent feedback
cannot use relations. Keep the reviewed negative and its measurement lessons
as context, not as a favorable Direction 4 result
([closure audit](feedback-stability.md)).

## 4. What future patterns would mean

These are diagnostic interpretations of the existing design, not additional
selection criteria or permission to inspect R7 outcomes during R4/R5.

| future pattern | supported reading | remaining limit |
|---|---|---|
| relation-trained standard improves over paired `none`, COPY adds little | auxiliary supervision changed useful trunk behavior | runtime explicit-action benefit is not demonstrated |
| predicted COPY improves exact blocks within the same weights and passes guards | learned explicit-action channel contributes under the declared policy | effect magnitude, seed consistency and novelty scope still determine the label |
| oracle works, predicted COPY fails | valid supplied actions execute; learned selection remains inadequate | requires later joint-action/error analysis; oracle alone does not isolate one neural factor |
| affine tuples pass, nested compositions fail | bounded flat relation gain if all corresponding criteria pass | no compositional gain label |
| copied bytes increase while false COPY or generic invalidity worsens | action channel has a specificity cost | aggregate exact successes cannot waive frozen guards |
| samples decrease, cache positions stay similar, time rises | sample savings coexist with head/orchestration cost | no speedup claim; use Point 5 measured costs |

The proposed floors and decision labels remain in
[Direction 4 §§2.3–2.4](copy-relation.md#23-decision-labels). No thresholds,
endpoints or retries are added by this synthesis.

## 5. Figure specification: evidence, mechanism and computational truth

### 5.1 Editorial purpose

Working title: **“From relation-sensitive context to an executable action.”**
Make a standalone schematic with three labeled panels. Its purpose is to show
why Direction 4 is the next experiment and exactly what COPY changes. It must
remain publishable with the current evidence: the learned Direction 4 result
is visibly marked **not yet measured**.

| panel | content | evidence boundary |
|---|---|---|
| A — measured motivation | compact two-venue display of Direction 2 context effect, controlled generation preference and descriptive exact hits, with seeds separate | existing measurements and component intervals; independent axes/units |
| B — proposed intervention | prefix → normalized boundary states → bounded span/affine/count head → EMIT or exact executor → flat bytes → VM drawing | implemented mechanism illustrated with a scripted engineering action, not a model success |
| C — work under COPY | shared byte-position axis with action, output and transformer-feed lanes; copied block bracketed; final un-fed symbol explicit | schematic now; later populate directly from accepted Point 4 engineering trace |

Panel A may instead be a small cited evidence callout if repeating three numeric
axes would make the mechanism unreadable. Existing
[Figure 15](figs/fig15_context_generation.svg) already visualizes the preference/
capability gap. Its middle panel is raw `D_gen`; a new controlled `Delta_gen`
panel must read the v5 field explicitly and label it correctly. Never relabel
Figure 15's existing numbers or overwrite the historical figure.

### 5.2 Concrete mechanism and trace example

Use the six-byte L0 motif `MOVE 112 120; LINE 128 136` as the source and an
identity affine action with total count 2. Show source interval `[0,6)`, query
boundary 6, six deterministic appended bytes and a following literal HALT.
This choice illustrates byte accounting without introducing rounding or an
unseen transform. State **scripted engineering example** beside the drawing.
A later optional second sketch can illustrate a supported nonidentity transform
only after the executor supplies its exact output; do not draw approximate
geometry and present it as a transducer trace.

Panel C aligns the six copied bytes and final HALT with seven output positions.
The COPY action occurs once, the literal sampler is used for HALT, and six
incremental transformer positions feed the copied bytes. BOS+prompt prefill is
separate. HALT is emitted but not fed in this one-row example. A second row can
show all-literal emission of the same bytes: seven literal decisions, identical
conditioning-position count. No wall-time axis or fictional speedup is shown.

In panel B, targets/equivalence annotations sit in a separate **later evaluation**
box reached only from completed output/logs. There is no arrow from target data
to predicted selection. Oracle supplied actions, if shown, use a dashed,
explicitly diagnostic route separate from the learned head. Avoid making the
unrolled flat output look like a new on-device COPY opcode: COPY is internal to
the host decoder; the existing VM executes ordinary flat instructions.

### 5.3 Style and implementation ownership

The repository's research graphics already use SVG with pgfplots conventions in
[dm/eval/plot.py](../dm/eval/plot.py); [scripts/plots.py](../scripts/plots.py)
owns data-to-figure composition. The presentation figures in
[dm/demo/figures.py](../dm/demo/figures.py) add a compatible paper treatment.
No local TikZ/LaTeX source system was found. Prefer extending this established
vector path to achieve the requested TikZ-like appearance without a new build
dependency.

Use a white background, serif/Latin Modern fallbacks, thin black rules, outward
axis ticks where applicable, direct labels, generous spacing and modest teal/clay
accents. Encode evidence status and action type with line style/text as well as
color. Use monospace for byte offsets and instruction text; preserve a fixed
canvas scale for geometry. Do not reuse the demo's black ink as an implicit
claim of device provenance: every mechanism drawing is labeled host/scripted.

Proposed future artifacts: `docs/figs/fig17_copy_relation_mechanism.svg` and a
small adjacent source-data JSON with record/policy/fixture/source identities.
Confirm the next unused figure number when implemented. Add a thin figure
composition function following `scripts/plots.py`; only add reusable SVG
primitives to `dm/eval/plot.py` if genuinely shared. Native TikZ export can be a
later publishing requirement, not part of the decoder implementation package.

### 5.4 Data and visual acceptance

Before rendering, fix the panel content and source-data manifest. The figure
builder must refuse absent/mismatched required reports rather than silently
substitute doc literals, zero effects or illustrative learned results. Source
manifest fields: panel role (`measured`, `engineering`, `proposed`), source path
and digest, metric/unit, seed, component/draw counts, interval method or explicit
absence, fixture/policy identities and generation-source identity.

Render deterministically, inspect the standalone SVG and an exported raster at
publication size, and check clipping, text overlap, offset alignment and
readability in grayscale. Reconcile all plotted values with source records and
all geometry/byte lanes with the exact engineering trace. A conceptual version
can precede Point 4, but its timeline must be labeled schematic; measured runtime
observations require the accepted Point 4 artifact. No figure is generated by
this planning session.

Suggested caption:

> Controlled prefix interventions reveal relation-sensitive predictions in the
> Direction 2 checkpoints, while compatible exact continuations remain uncommon.
> Direction 4 tests a learned source-span and affine action realized as ordinary
> flat bytes by an exact executor. The scripted COPY example replaces literal
> sampling decisions; its bytes still require causal transformer conditioning.
> Direction 4 learned-generation effects have not yet been measured.
