# Traps — one line per rule

Short enough to actually read before a change. **The long form — what was
measured, what it cost, why the guard is shaped this way — is in
[`history/instrument.md`](history/instrument.md). Read that before changing a
guard.** `PLAN.md` carries only the handful that bite hardest and points here.

Every line was paid for. Nothing is here because it sounded prudent.

**Measurement**

- **Never compare per-token loss across codecs.** Use bits/drawing.
- **Never compare bits/drawing across two different val sets.** Two draws differ
  by ~3.5 bits, several times any effect under test. Prefer the paired difference.
- **Two *spellings* of one corpus are the one exception, and only per position
  class.** The same drawings encoded flat and with `REPEATX` are a legitimate
  compression comparison — both decode to identical geometry — but the arms also
  differ in codec and in sequence length, so the difference of totals is not the
  fold's. Split it: `dm/eval/spelling.py` classes the bytes the fold removes, the
  bytes it adds, and the bytes that are byte-identical in both. Taken bare the
  difference was ~19.8 bits and the fold's direct term 5.84 — and the codec-fixed
  pair showed the rest is the fold's shorter sequence, a real indirect term, not
  noise (`docs/xform.md` §3.1).
- **The same rule applies to an architecture, and it is one forward pass.**
  Before adding an inductive bias that *supplies* structure, measure what the
  model spends failing to have it. The bits an arm puts on symbols that cannot
  occur at a position are exactly what supplying the grid could recover, and
  they total **0.0008–0.0085 bits/drawing** over four alphabets and three
  corpora — four orders of magnitude under what typing buys. `dm/eval/
  attribution.py`; it closed §7.2's conv front end without building it.
- **A codec is reconstructed from the *record's* width, never from today's
  table.** `TokenCodec` lays out `[specials][opcode slots][values]`, so ISA v2
  moved `token` from 269 symbols to 274 and `token_typed` from 1,293 to 1,554. A
  pre-v2 checkpoint scored through today's codec reads its own symbols as
  different opcodes, and every instrument in `dm/eval/` reshapes logits against
  `codec.vocab_size` — a reshape that does not raise when the width is wrong.
  Use `dm.eval.records.codec_for`. `byte` and `bit` are width-stable at any
  opcode-table size, which is why that migration moved no claim-1/2/4 number.
- **Before building an opcode, measure the redundancy it would remove — and
  charge the whole instruction the ISA specifies.** `CALL` takes a `Kind.ID` and
  no placement, so a library body has to be positioned by `XFORM`/`ENDX` and a
  call site costs 7 bytes rather than 2; and the library is capped at the 256
  bodies one `ID` addresses. Priced that way, between-drawing reuse is **25.21%**
  of Tabler's bytes against `REPEATX`'s 11.74% — and **0.00%** of QuickDraw's
  (`dm/eval/library.py`).
- **Before building an opcode to remove redundancy, measure what the model already
  spends on the bytes it would remove.** `REPEATX` folds 40.9% of a corpus's bytes
  and 1.2% of its bits, because a copy byte already costs 3.0% of the byte it
  copies. A compression feature's bit value is bounded by the model's own failure
  to compress, and that bound is one forward pass.
- **An axis whose ceiling is under the corpus's floor cannot be read in
  bits/drawing, and the arithmetic is available before the run.** A class label
  carries at most `H(class)` bits, so conditioning can buy at most
  `I(X;C) <= log2(5) = 2.32` against Tier B multi's ~2.5-bit floor. Check the
  information a signal *contains* against the floor before designing a likelihood
  comparison around it.
- **A paired gap between two models carries the floor; a single-model quantity does
  not.** The conditional-minus-unconditional gap is `I(X;C)` plus however much the
  two runs differ on their own -- measured at +0.34 bits on two smoke arms whose
  true effect was 0.01. `H(C|X)` from one checkpoint scored under every class has no
  cross-model term at all. **Prefer the reading that needs one run.**
- **A falsifier can fire without bearing on its claim — twice now.** P5's fired
  at both seeds and settled nothing: it differenced two arms that were both
  96–98% of the way to the ceiling, where the claim was about the distance *to*
  the ceiling. Conditioning's "gap above the ceiling = instrument fault" branch
  fired with both named faults excluded — the excess was the cross-model floor
  its own document had measured two sections earlier (`docs/conditioning.md`
  §5). Before writing a pre-registered branch, name the sentence of the claim
  that becomes false when it fires; if there is none, the branch is decoration.
- **A difference between two unequally precise arms is quotable in sign only.**
  P5's transformed arm replicates to 0.0003 in `recovery` and its control to
  0.0101 — 34x — so the smaller of the two gaps *is* the control's own spread.
- **Pairing removes val-set variance, not seed variance.** The interval must carry
  the between-seed term; do not "simplify" it back to quadrature.
- **Run-to-run spread at fixed config is the resolution floor, and it scales with
  bits/drawing.** Tier A ~1 bit, Tier A flat 0.63, Tier B `cat` 1.97, Tier B
  multi ~2.5, Tier C ~1.5, **the planner 4.16 with a 95% CI of [3.67, 4.64],
  nearly all of it the stroke decoder's**. **No axis smaller than the floor is
  resolvable at k = 2**, and a model's floor is its own: the planner's is ~2x the
  flat arm's on the same corpus. **Composed: 0.4-4.7 bits, mean 2.0 over eight
  same-config pairs — and a corpus's floor is not one number.** Its widest pair is
  12x its tightest, and the widest belongs to the arm that is also the worst
  behaved when sampled, so an axis read on that arm needs its own floor.
- **A resampled sampling column describes the checkpoint, not the arm.** Five
  draws on one checkpoint gave `len EMD` 191.0 ±16.2, and the same config at
  another training seed gave 52.3 ±11.2 — the between-seed spread is 4x the
  between-draw one. `scripts/resample.py` fixes the "a column is a draw" fault and
  not this one: **quote the ordering across seeds, never the level.**
- **A curve's out-of-distribution tail has its own floor, and it is not its
  body's.** On claim 2's per-copy curve the in-count copies replicate to 0.033
  bits/symbol while the out-of-count copies differ by 0.53 and 0.94 — 16–28×, on
  the same two runs and the same evaluation. **Measure the floor per regime, not
  per metric**, or a one-seed shape in the extrapolation tail looks as quotable
  as a two-seed number in the body.
- **"It scales with program length" is not evidence that a difference is
  representational.** A *null* pair of provably identical encodings gives a
  per-byte rate whose bootstrap CI excludes zero.
- **`drift ≈ 0` is half a convergence test.** Read `drift` and `tail` together,
  always.
- **Neither `drift` nor `tail` can establish an asymptote**, because both are
  measured inside a schedule that anneals its LR to zero. **Only `budget_table` —
  the same arm at two budgets — binds.**
- **A tail is a guard, never a rate.** Multiplying one by a step count overstates
  by 2.7–3.8×, measured on the planner's rung (§7).
- **A schedule correction measured on one architecture does not transfer to
  another**, and a `tail` read at one rung says nothing about an earlier one. The
  planner arms finish 6.3–7.0 bits below their own step-12,000 reading; the flat
  AR arm finishes 0.33 below its own. Predicting the missing rung instead of
  running it was wrong by 6.2 bits and inverted a per-doubling sign (§7).
- **Two arms differenced against a *third* arm at one budget are not a budget
  ladder.** Both claim-3 rungs were read against the same 24,000-step baseline,
  so the "per doubling" column measured the planners moving and silently assumed
  the baseline still. **Difference each rung against its own budget.**
- **A difference between two arms stopped at different points on their own curves
  is a rate, not a cost**, and no interval can tell you. More seeds cannot clear
  it; only more steps can.
- **A budget regime cannot equalise fit, only compute.**
- **bits/drawing cannot see sample quality.** Report both. The set comparison is
  `dm/eval/quality.py` (`coverage`, `mmd`, `nna`); every column above it is a
  *marginal*, and a distribution can match every marginal anyone thought to check
  while matching nothing else.
- **A set metric is a function of the two set sizes and not only of the
  drawings.** `nna`'s textbook ideal of 0.5 holds only when the sets are the same
  size: unchanged real drawings read **0.495** at 200 v 200, **0.590** at 100 v
  200 and **0.919** at 64 v 1,000, because the larger set supplies most of
  everybody's neighbours. `coverage` is bounded by n/m before a model does
  anything. **Subsample the reference to the sample count**, under a seed of its
  own so no draw of the model can move it and every arm on one corpus is scored
  against the identical reference.
- **None of the three has a value without a floor measured at those same two set
  sizes**, on real drawings held out of the reference — `mmd` is in the corpus's
  own pixels and `coverage`'s ceiling is nowhere near 1. Take the stand-ins from
  the **training** split: drawn from the reference itself they measure a set
  against a superset of itself, and the floor becomes unbeatable rather than
  achievable. The `nna` floor is also the estimator's self-test — real against
  real must land near 0.5, and a floor that does not is a bug in the metric.
- **A feasibility mask that excludes the byte actually present reports an
  unbounded saving**, and it looks exactly like a finding. Check every rule at
  every position of a real corpus before believing any number computed from one.
  It caught a live fault: `Summary.length` is u8, so on strokes past 255 bytes
  the length rules were applied to a length the summary cannot state and
  excluded the truth at 48 positions.
- **"Nothing is recoverable here" and "nothing more is recoverable here" are
  different claims.** Bits *spent* at a position upper-bound what any coder could
  save there; bits saved by one exact rule only lower-bound what a better rule
  could save. The redundancy result is an upper bound on the determined
  positions and a lower bound everywhere else, and the two must not be quoted in
  one sentence without saying which is which.
- **A "floor" measured on real drawings is not a resolution floor**, and the two
  words must not be traded. The real-drawings floor says what a *perfect*
  generator scores; the resolution floor says what a *re-run of the same config*
  scores, and only the second bounds a comparison between two arms. The
  sample-quality axis has the first at every set size and has never measured the
  second, so every gap on it is one checkpoint against one checkpoint.
- **Padding is not geometry.** In a batched Chamfer, mask it on *both* sides of
  the distance: masking only the targets leaves padded rows acting as sources at
  the canvas origin, which pulls every distance towards the top-left corner and
  does it worst on the shortest drawings. Measured on the correct code against a
  one-sided mask: 1,028 px per entry, and 3,525 px once the padding widens.
- **A generation column is a draw, and a record's final eval is one sample of it.**
  The flat AR baseline reads 2.72 in its record and **10.40 ±4.48** over five
  seeds. **Quote a sampling column from `scripts/resample.py` or not at all.**
- **A generation report key must name checkpoint, sampler, draw count, set size,
  quality resolution, control and device.** A two-seed smoke run once replaced a
  five-seed result; hardcoded `top_k` then made a sampler sweep capable of
  replacing itself one setting at a time. Both report drivers now refuse an
  existing exact key unless `--overwrite` is explicit.
- **Never score quality after empty exclusion at a changed set-size ratio.** Count
  empties, then block geometry. Dropping blanks turns 100 v 100 into another
  estimator; replacing them samples another distribution. Persist failed draw,
  guards, seed and reason before exit — missing output is not failure evidence.
  Keep partials for diagnosis, encode unavailable values as JSON `null`, and
  make every aggregating consumer require `status=complete` first.
- **A sampler endpoint that fails its structural guard stays failed even when
  earlier paired draws look decisive.** Class `all` improved diversity in 4/4
  complete draws, then emitted one empty in draw 5; flat AR did same at 1/1,280.
  Quote partials as diagnosis, never as endpoint, and do not rerun until lucky.
- **Same RNG seed is paired only when backend and every upstream draw match.**
  Published Figure 8 used CPU default; sampler sweep uses MPS. Backend changes
  planner composition plans before downstream support masking, so old/new is
  not a support contrast. Device belongs in report key and comparison contract.
- **A validity number is not a description of an arm.** `halted` and `wellformed`
  fail independently; read `gen_truncated` and `gen_length_emd` alongside.
- **Report `-log2(1-q)`, never `-log2(q)`.** The recoverable cost of illegal mass
  `q` is `-log2(1-q)`; `-log2(q)` *grows as illegal mass shrinks* and so reads
  backwards. Compute it from the mass on the **legal** set rather than from
  `1 - q`: under the bit alphabet a raw prefix drives `q` to within float epsilon
  of 1, where `1 - q` has no significant digits left (`dm/eval/state_support.py`).
- **Illegal mass concentrates, so a mean is not the finding.** On the structured
  venue `q` averages 5e-5 and reaches **0.95**, and one position in one program
  carries ~35% of the whole corpus's recoverable bits. Report the maximum, the
  quantiles and the worst positions beside the mean, or a real failure mode looks
  like rounding.
- **A grammar mask is prefix-local, and a loop replays its body.** A body a
  non-canonical mask let through executes *again* on the next iteration: a
  dangling `XFORM` in a `REPEAT` body faults `DEPTH_OVERFLOW` several iterations
  later, and a crossing `ENDX` faults on the second. Measured under the VM-safe
  cell. Only the canonical mask is safe from this, because proper nesting makes
  every iteration's scope bookkeeping identical.
- **A canonical program can still exhaust fuel**, and 3/8 rows of an untrained
  smoke model did. `OUT_OF_FUEL` is a resource bound and belongs in neither
  structural predicate; keep it in its own column and out of the validity
  conjunction.
- **A mask does not buy termination.** Forbidding `HALT` inside an open scope
  converts an invalid early halt into a longer program and sometimes into a cap
  hit — measured at `cap_rate` 0.00 → 0.17 for one VM-safe cell. A capped row is
  incomplete, never valid; report how many bytes it was from closing.
- **Seeding is not pairing once a mask is involved.** A mask changes the
  distribution, so it changes how the sampler walks its RNG stream, and two
  identically seeded cells diverge after the first masked position for a reason
  that is not the mask. Pre-generate the variates, one per row per step, and
  consume one per row per step whether or not the row has stopped
  (`DrawingLM.generate(variates=...)`).
- **`Tier` is not a language key.** `Tier.L2` contains `CALL`, which `VM.run`
  refuses, and `TrainConfig.tier` is ignored by the QuickDraw and composed
  loaders. Reconstruct the opcode allowlist from the corpus the checkpoint was
  scored on (`LanguagePolicy.from_programs`), never from the config.

- **`scripts/footprint.py` measures whatever `.o` files are in `port/build`**, so
  a prior `conformance.py --qemu` leaves the runtime and startup objects there
  and the footprint comes back as the whole image. It fails loudly rather than
  reporting a wrong number — non-zero static RAM and unplaceable sections — but
  **`make clean` before measuring** or read a failure as a stale build rather
  than a regression.
- **A port is a second implementation, and the second implementation audits the
  first.** With `REPEATX` reachable the conformance fuzzer generated a loop whose
  body closes the loop's own transform scope, and the *reference* raised
  `IndexError` where it is required to report a fault. `dm/isa/unroll.py` had
  refused that program as a crossing all along, so two of the three agreed and
  the third crashed. **Keep the fuzz corpus drawing from the whole opcode table**:
  a structured corpus reaches almost none of the fault handling.
- **A cost sketch counts the array and misses the cache.** `direction.md` §2.4
  budgeted ~96 bytes for the transform tier's stack and it cost 204: the scope
  array was right, and the cached composition, the per-frame step and floor, and
  the compiler's spills were not in the estimate. Predict device cost if you
  like, but **the number that goes in a table comes from `size` and `-fstack-usage`.**

**Causal case construction** (Direction 2, 2026-08-16)

- **A control that is zero by construction is not a control.** The retracted
  schema-1 C3 had 64 identical control target pairs, which forces `D_control` to
  zero without testing anything. Schema 3 asserts both control targets differ,
  both block-control prefixes differ, and both worlds' prefixes differ — and
  `load_manifest` refuses a manifest that violates any of them.
- **A per-case total-variation distance between coordinate histograms ranks
  nothing.** Two targets carry ~10 coordinate pairs from a 256-value alphabet,
  so any two of them are nearly disjoint and TV sits near 1 for every candidate
  donor. Match on **frozen-corpus frequency** per case; keep the profile TV for
  the pooled balance table, where the histograms are dense.
- **Do not infer a transform from the layout policy.** Today's composed corpus
  uses one quarter turn about the origin with zero translation, and a builder
  that assumed it would mis-derive every target the day the policy gains a
  shift. Search the eight D4 elements, read the translation off the bounding
  boxes, then check `apply(source, T) == image` byte for byte.
- **Two adjacent distractors in a composed scene have no boundary in the byte
  stream** — both are runs of `MOVE`/`LINE`. `Scene.distractor_spans` records
  them, because a "whole unrelated block" that straddled two motifs would be a
  perturbation of neither.
- **Swapping the two worlds reverses the control contrast** unless the control
  pair is relabelled with them: which control target counts as "matched" is
  decided by the prefix it is paired with. Assert both directions, or a sign
  error looks like an invariance.
- **An exact transformation of the input is not automatically a controlled
  comparison.** Rotating a whole program by a quarter turn preserves the
  relation byte for byte and still costs the model 2.2–2.4× its own prefix rate
  and 27× its matched-target rate. A null there says the model has no
  distribution, not that the relation failed to transfer. Move the smallest
  thing that asks the question — the in-place y venue moves 15 of ~76 prefix
  bytes and lands at 1.34×.
- **A support precondition must measure the instrument, never the outcome.**
  `context-protocol-v2` gated the axis diagnostic on how expensive the
  *compatible target* was, which made the diagnostic unreadable in exactly the
  case it exists to detect. Condition on whether the model can read the input
  (prefix cost); never on the quantity the estimand is about.
- **Check the confound the estimand does not cancel.** The symmetric four-way
  contrast cancels unconditional target preference and nothing else. In the step
  venue the crossed target is sometimes *nearer* the preceding copy than the
  compatible one — 38 of 64 cases — and that stratum is the whole test of
  whether the effect is a proximity heuristic. It is computable from the frozen
  manifest with no model run.
- **Read the prefix-NLL column before believing a specificity result.** An
  unrelated-block control that abuts nothing is confounded with distance. What
  settles it is that the irrelevant edit bought 8–10× more prefix surprise than
  the relevant one and moved the target by ~0.001 bits/byte.
- **Yield is a corpus property, and a low one is a finding.** The composed shape
  venue gives 28 cases from 1,000 scenes because 668 of them have no
  same-skeleton donor motif at all. Record the rejection census and freeze the
  number; never re-roll the seed after seeing a model result.

**Generation measurement** (Direction 2 C4, 2026-08-16)

- **A label with no effect-size floor says "detectable", never "achieved".**
  C4's `generation_robust_context_use` was assigned on an interval alone,
  because no SESOI was declared for `D_gen`. On the controlled v5 run's composed
  venue it sits on 1.3% / 1.1% exact-hit rates. Publish the behavioural column beside the label or the
  label will be read as a capability claim — including by whoever wrote it.
- **A contrast between two sampled cells is not controlled just because it is
  paired.** `D_gen`'s symmetric form cancels an unconditional preference for
  either target string and nothing else. A model that lurches at *any* prefix
  edit scores exactly what a relation-follower scores; only the matched
  unrelated-block arm separates them, and it is one extra decode because
  `block_prefix_a == prefix_a`.
- **A freeze is only as good as the last edit after it.** `dm/eval/context_c4.py`
  was edited nine seconds after the run that its protocol had frozen the digest
  of, so the protocol named a file that existed nowhere. Nothing failed; it was
  found by hand. When digests disagree, **rerun** — "the digest moved" is a
  question, not a verdict. Here every row reproduced bit-identically and the
  reports stood; had they not, they would have been retracted.
  `test_the_current_protocol_names_the_code_that_is_actually_here` now fails on
  the drift instead, for the newest protocol only — superseded ones name old
  code on purpose, which is why they are kept and never edited.
- **More draws cannot narrow an interval that is not draw-limited.** After
  expanding C4 to 64 draws per world, resampling draws with the case set held
  fixed puts the sampler's share of the venue-mean variance at 17–21% on a
  64-component venue and **4–10% on a 21-component one**. The composed venue's
  half-width did not fall, and for one seed it rose — a bootstrap over 21
  components has an unstable width of its own. Decide what a run is limited by
  before spending eight times the compute on it.
- **Derive an output path from everything that distinguishes the run, not just
  the manifest.** C4's report name was keyed on the manifest hash, and v5 scores
  v4's manifests unchanged — so the second freeze's run landed on the first's
  path and the only thing standing between a corrected rerun and the deletion of
  the record it was correcting was an `--overwrite` flag the operator was being
  told to pass. The protocol tag is now in the filename.
- **Tests precede the run, and a stage that skips them has to pay it back.**
  C4 shipped and was scored with no test file at all, in a direction whose own
  §4.7 requires the opposite. Nothing was wrong with it, which is the point:
  the discipline is not there to catch the runs you already trust.
- **A scripted-completion fixture is the only way to test a generation metric
  against arithmetic.** Two scripted models — one that follows the prompt's own
  relation, one that emits the counterfactual target after any edit — produce
  the *identical* `D_gen` and opposite `Delta_gen`. That pair is the test that
  the control works; a random-weight model can only test that nothing crashes.
- **`HaltMonitor.done` fires on HALT and on an unknown opcode**, because `VM.run`
  breaks at both. Reporting them as one category scores a dead byte as a clean
  termination; C4 separates `halt` from `fault` by re-checking canonical
  membership and the final byte.
- **A figure's shapes can all sit inside the frame while its labels hang off the
  page.** A centred axis label longer than its panel passes every geometry
  check, because a `<text>` element carries only its anchor point. `test_plot.py`
  now estimates text extents too — it was written after fig15 shipped a label
  18px past its canvas.

**Corpora and keys**

- **Every geometric transform must be integer-affine, or exact repeats stop
  existing.** Integer translate/mirror/90° rotate preserve Tabler's ceiling
  exactly; a ×1.07 scale destroys **57% ±3%** of it and ±1 jitter **99.6%**.
  Validate any new policy with `scripts/repeat_oracle.py`.
- **The damage a non-integer map does depends on its *rounding phase*, and one
  phase is a draw.** The identical corpus under the identical ×1.07 retains
  18.7% of its repeat ceiling scaled about the origin and 34.8% about each
  drawing's own centre — one operation, two phases — because Tabler's
  coordinates are mostly multiples of ten and the phases resonate rather than
  average. Marginalise over sub-pixel offsets (a low-discrepancy set, never a
  grid: a grid at k/10 samples the resonance) and quote the spread. The 77% on
  record was a single draw; the direction it supported survives and the number
  does not.
- **A retention is a ratio and two arms do not share a denominator.** The
  control-point arm keeps 61.5% of its ceiling against a polyline's 42.3% and
  ends with **fewer** foldable bytes, because its ceiling was 38% smaller to
  begin with. Report the level beside the ratio, or a more compact primitive
  looks like a structural win when it has merely removed the redundancy the
  fold was going to remove.
- **`round` and `numpy.rint` round half to *even*, so rounding stops commuting
  with integer translation at half-pixel coordinates** — `round(x + k) !=
  round(x) + k` when `frac(x) == 0.5` and `k` is odd. Every exact-repeat number
  here rests on that commutation. Use `dm.isa.bezier.round_half_up`, and in any
  fitter work relative to an **integer anchor**: multiplying an absolute
  coordinate by a weight rounds differently at 200 than at 20, so an absolute
  least-squares solve makes a fitted shape depend on where it sits. Both faults
  were live and both silently deleted planted repeats from one arm only.
- **A tolerance below half a pixel diagonal is a request no primitive can
  honour.** `sqrt(2)/2` is the quantisation floor on an integer grid, so a fit
  guarantees `error <= max(tol, floor)` — and below the floor two arms are
  compared at whichever error each happened to reach rather than at a matched
  one.
- **A canonicalisation inside the exact group is a *provable null* on any
  within-program oracle**, because it carries the foldable set across
  bijectively. Measured to byte equality on 300 icons. If a re-framing appears
  to move one of those ceilings, the policy is not in the group; nothing else
  measured under it is readable. Reuse **between** drawings
  (`dm/eval/library.py`) is the only place a re-framing can show up, and its
  sign is not monotone in corpus size.
- **Quantise from the source grid by an *integer* scale factor.**
- **A corpus key derived from the config cannot see a default.** The corpus is now
  digested from the programs themselves (`dm/data/fingerprint.py`) and is part of
  every ladder and pairing key. A convention cannot be checked.
- **Neither can a caption.** P5's control reports were captioned `n ∈ {2,4}`
  because `orbit_sizes` was absent from the config, on a corpus whose canvas
  allows exactly two — `orbits(control=True)` has no four-copy entry at all. A
  policy description asks the generator what it offers
  (`dm.data.composed.policy_label`), never the config what was typed.
- **A transform group closed on the canvas gives a sampler no reason to stop.**
  D4 maps the canvas onto itself, which is what makes a transformed scene safe to
  *construct* — no placement can need clamping — and is why its flat arms
  over-generate by up to 2x while the translation-only control, whose third copy
  would leave the canvas, does not. One geometric fact, a construction
  convenience at build time and a sampling hazard at generation time.
- **A `limit=` that slices a concatenated corpus makes it single-category without
  saying so.** `_interleave` round-robins before the cache is written.
- **`max_len` truncates, and it truncates the bit arm first.** A row with non-zero
  `truncated` is not comparable across codecs. Choose `rdp_eps` and `max_len`
  together from the *full* corpus.
- **An ablation axis can silently degenerate on a new corpus.** `token` vs
  `token_typed` is byte-identical on **every L0-only corpus**. Run
  `codec_a.encode(p) == codec_b.encode(p)` over the val split before reading any
  new corpus's table.
- **A relative codec gives claim 2's answer away, and `recovery` cannot tell.**
  Claim 2's headline stays on the absolute view.
- **Records are keyed by tag and the tag does not carry the budget.** Re-running an
  arm at a new step count overwrites the old record unless the tag says so.
- **Never mix run records across `SCHEMA` values.**
- **A replicate key covers the experiment and cannot see the code.** Two runs at
  one arm, codec, shape, budget and corpus are a replicate pair only if the
  interpreter between them did not change, and no field in a record says that —
  `schema` is the only proxy, and it is exactly as good as the discipline of
  bumping it. Run 4 and its re-run share every key and are *four code fixes*
  apart at one schema. `planner_replicate_table` therefore prints both sides'
  schema and refuses to call any pair a floor on its own authority.

**Models, samplers and records**

- **`share_non_embedding_init` redraws every non-embedding *matrix*, and a
  conditioning table has the same rank as one.** Left in, it would have destroyed
  the zero-init that makes a conditional arm start where its control starts *and*
  consumed generator draws the control never consumed, so the two arms' layer
  weights would have diverged too. Any new parameter that must keep its initial
  value has to be skipped there **by name** -- shape cannot distinguish it.
- **Conditioning a tied-embedding model with a class *token* widens the logits.**
  `dm/eval/recovery.py`, `spelling.py` and `redundancy.py` all reshape logits
  against `codec.vocab_size`, and that reshape does not raise when the width is
  wrong -- it reinterprets the tensor. An additive class embedding changes no
  width, adds no scored position, and costs `n_classes x d_model`.
- **A sampler is part of a model's specification, so the record has to name it.**
  The unmasking order was a *default argument*, so changing it changed what every
  existing record's generation columns meant while leaving the records
  byte-identical — and three separate faults rode that one default.
  `PlannerTrainConfig.gen_order` is stamped into new records.
- **PAD and BOS are model-control symbols, not bytecode output.** Mask them before
  top-k in generation side reports. Full support otherwise spends steps on
  symbols `decode` drops. Keep training-time support unchanged unless bumping its
  regime: changing eval RNG consumption can move later training batches. Same
  rule applies to planner's AR stroke decoder.
- **Flat-AR cap status comes from halt-monitor state, never returned tensor
  width.** Generation returns when its last row halts; comparing decoded length
  with that width marked the last *valid* row truncated on every draw — exact
  1/n values in old flat-AR reports. A live monitor row is the cap hit. Planner
  cap accounting is separate and unaffected.
- **Confidence-ordered unmasking is not the reverse process of a masked diffusion,
  and on a sparse grid it is a ratchet.** The reverse step must unmask a
  **uniformly random** subset. **The test for a biased sampler rather than a
  coarse one: more steps make a biased one monotonically worse.**
- **When a metric can be satisfied by producing nothing, check what the model
  produced before believing what the metric says about it.** An all-empty grid is
  well-formed, and validity reported it as a model that cannot draw.
- **A structural guarantee the sampler does not enforce is a claim, not a
  property.**
- **A column the design names as its own falsifier must actually be recorded.**
- **A record can be half wrong, so retract columns rather than runs — and retract
  the whole generation half, never single keys.** `halted` and `wellformed` are
  both read off `gen_faults`, so blanking one key makes the `or {}` default report
  a fabricated 1.000. **A record that retracts its own columns must never be
  re-run under its own tag.**
- **A table that cannot see a record cannot check it.** `summarise` dropped every
  planner record, so a 3.2-hour pair of runs was spent on a rung for a table that
  could not print it — and the planner's numbers were differenced by hand instead.
  **A comparison that lives in prose has no key.**
- **What names an arm is part of the ladder's key, and the planner's arm is not
  its codec.** Keyed on `(codec, shape)`, the two planner arms fall in one cell.
- **A seeded reset inside an eval is a global side effect on training.** FIXED
  2026-08-10: the estimator holds a private `torch.Generator`. The rule stands —
  `torch.manual_seed` in `per_program_bits` makes the two planner arms resume on
  **different batch orders**. Pinning an estimator's draw is right; pinning it by
  reseeding the process is how the pin reaches the optimiser.
- **Sharing no gradient is not sharing no RNG stream**, so "this half is
  unaffected" is a claim about the code and not about the number. **An arm you
  believe is held fixed must be *differenced*, not assumed.**
- **Run-to-run noise is the *trajectory*, not the initialisation.** Shared init
  (`--share-init`) did not correlate endpoints. RETRACTED as advice, having been
  written down before it was tested — and **validate a variance-reduction idea
  before writing it down as advice**, because a plan that records proposals as
  rules teaches the reader to distrust the rules.
- **RoPE is relative, so anything on a fixed grid of *fields* needs absolute
  positions too.** `Config.abs_pos`, off for every AR arm.
- **Do not detect HALT by matching a symbol.** In every untyped alphabet the HALT
  byte also occurs as an operand value; only the parse state distinguishes them.
- **Do not read a halting hazard off one teacher-forced pass unless the codec is
  stride-1.**
- **Transfer logits once, then normalise in float64 off-device.** MPS has no
  float64, but softmaxing in float32 and casting the probabilities afterward
  cannot recover an illegal tail already rounded out of `legal_mass ≈ 1`.
- **A large mask win in a language with no dynamic scopes is not evidence for
  execution state.** It can be excellent static/policy decoder evidence and
  still be categorically ineligible for an execution-state gate.
- **Two identical control targets make a four-way control contrast zero by
  algebra, not by measurement.** Assert target inequality in the manifest.
  Donor reuse also connects rows: bootstrap connected source/donor components,
  not source labels alone.
- **A deterministic empty cannot be rerun away.** With fixed weights, uniforms
  and seeds, retrying the same geometry draw is result selection. Treat emptiness
  as an observed floor failure or freeze a different estimator before the run.
- **An `IntEnum` member can be zero, so truthiness is not membership.**
  `Kind.COORD` is `0`, so `f"operand_{kind.name}" if kind else "none"` filed
  *every coordinate operand* — most of the corpus — under a stratum meaning "no
  kind expected". Use `is not None`. The same shape of bug is available for
  `Phase.OPCODE`, `Status.LIVE` and any future zero-valued member.
- **`-x.clamp_min(eps)` clamps `x`, not `-x`, and a NaN that reaches a `topk`
  makes the result backend-defined.** **Assert what a quantity is, not that the
  code ran** — the existing test checked the grid's shape, and that holds on NaN.
- **`REFERENCE_SHAPES` is over budget on purpose** and deliberately absent from
  `SHAPES`, which the sweep iterates.
- **Do not train the composition level on QuickDraw** — it is large and free and
  contains no scene-level composition. **Scoped: this is about what a composition
  level can *learn*, not a licence to blame QuickDraw for a level that fails on
  QuickDraw's own statistics.**

**Operations**

- **A trainer that writes at exit writes nothing when it is killed.** Cost 2.6
  hours and the one number the run existed to produce
  (`docs/history/lost-run.md`). Both trainers now publish through
  `dm.train.checkpoint` at every eval — atomically, weights before record — and
  a record carries `complete`. **A partial record is a real measurement of a
  shorter run and is never a rung of the budget it asked for**: read `steps`,
  which is what the loop reached, not `config["steps"]`.
- **Train on an otherwise idle machine. A contended box does not merely cost
  wall clock; it ends runs.** Twice now. The same config held a flat 231 s per
  500 steps alone, ran at 531 s while a QEMU pass and a figure regeneration
  shared the box, and on 2026-08-11 degraded *monotonically* from 283 s to 497 s
  (+9.3 s per eval, 2.15× the idle rate) while a test suite, a resample and a
  corpus build ran beside it — then took a `SIGKILL` from the OS at step 11,500
  of 12,000. **A flat wall-clock trace is the evidence a run had the machine; a
  rising one is the evidence it did not.**
- **Run a cell with `python3 -m dm.train`, never `python3 dm/train.py`.**
- **`TrainConfig.steps` is mutated by `train()`** when `token_budget` is set, so
  never reuse a config object across runs.
- **Do not launch a sweep from a wrapper shell that exits.** It gets orphaned and a
  second launch races it into the same log. Verify with `scripts/status.py`.
- **A long-lived sweep process can wedge in Metal, and it looks like slow
  training.** If a run is silent for longer than its first eval interval,
  `sample PID 4` it rather than waiting — the tell is that the run *header* never
  printed, where a fresh process reaches it in 7.9 s. Cost 78 minutes once.
- **A run tag whose regime contains an underscore was silently unreadable.** The
  codec now comes from a closed set matched against the tail, and an unparseable
  tag is **named** in the summary rather than dropped.
