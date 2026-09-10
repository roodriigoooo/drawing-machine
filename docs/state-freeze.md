# P0/S0 freeze — Direction 1, explicit execution state

> **Retrospective protocol — superseded 2026-08-15.** The claimed
> pre-implementation chronology of this document is not independently
> auditable from git: the freeze, implementation, tests and reports were
> untracked, while `runs/` is ignored. Treat this file as the historical v1
> protocol and its result narrative as provisional. The content-addressed v2
> repair record is [`docs/state-protocol-v2.json`](state-protocol-v2.json), but
> the corrected audit in [`state.md`](state.md) finds its gate category-invalid
> and closes S4. Both protocols are historical and neither can authorize work.

**Originally frozen 2026-08-14.** This is the dated preregistration copy the
original S0 design required before any code in the state branch was written and
before any state arm was trained. It fixes four
artifacts: (a) the canonical-versus-VM legality truth table, (b) the
per-checkpoint opcode allowlists, (c) the primary estimands, SESOIs, seeds,
sample sizes, sampler and multiplicity policy, and (d) the state coverage table.

Scope. This freezes **Direction 1 only**. Direction 2's C0 census
(natural-twin cases, compatibility `D`, its SESOIs and sample sizes) is
deliberately *not* frozen here and no case in it may be selected from anything
this document licenses. Nothing below may be amended to fit a result: a later
implementation bug may amend the *instrument* and must be recorded as such, and
the hypothesis, the primary endpoint and the exclusion accounting stay as
written.

Every number in §1, §2 and §4 was measured on 2026-08-14 against the code at
commit `11ae727`, not asserted. The measurement is reproducible with
`python3 -m pytest -q tests/test_state.py` (the truth table and the coverage
allowlists are pinned there as tests, so a drift in the VM or in a corpus
generator fails rather than silently re-defining the language).

---

## 1. The legality truth table

Three predicates disagree, and the paper needs all three named separately.

- **Canonical language membership** — `dm.isa.state.LanguagePolicy`. Program
  membership for the paper. Nothing else defines it.
- **VM acceptance** — `dm.vm.interp.VM.run` reports no `Fault`. Deliberately
  more permissive than canonical in places, because a generated program must
  execute totally.
- **Analysis acceptance** — `dm.isa.unroll.unroll` returns bytes. Used by the
  corpora and the oracles, and by nothing in this study's mask.

`vm_safe` in the code below is a *fourth*, weaker thing and must not be confused
with VM acceptance: it is the per-symbol predicate "emitting this symbol here
does not make a VM fault **inevitable**". A crossing whose fault depends on a
repeat count is therefore VM-safe and non-canonical at once.

| program | canonical | VM | `unroll` | note |
|---|---|---|---|---|
| `XFORM 3 10 10 … ENDX HALT` | ✔ | ✔ (no fault) | ✔ | the standalone transform branch; **absent from every corpus**, see §4 |
| `REPEATX 2 2 0 0 … ENDREP HALT` | ✔ | ✔ | ✔ | the structured corpus's only control flow |
| raw `XF = 8` on `XFORM`/`REPEATX` | ✘ operand domain | ✔ — masked to `code & 7`, no fault | **raises `ISAError`** | audited disagreement 1, plus a new defect (§1.1) |
| reserved `COLOR 3` | ✘ reserved opcode | ✔ — no-op, no fault | ✔ | audited disagreement 2 |
| `REPEAT 1 0 0  XFORM …  ENDREP  ENDX  HALT` | ✘ crossed scopes | ✔ — no fault at count 1 | ✘ (`None`) | audited disagreement 3 |
| the same at `REPEAT 2 4 0` | ✘ crossed scopes | ✘ `unterminated_repeat` | ✘ | the count is what decides, so no mask can call the count-1 case a VM fault |
| `XFORM …  REPEAT 1 …  ENDX  ENDREP  HALT` | ✘ crossed scopes | ✔ — `ENDX` closes below the frame's floor | ✘ | disagreement 3's mirror image, found here |
| `CALL 3` | ✘ under `call_policy="unsupported"` | ✘ `call_unsupported` | ✔ — copies the bytes | new disagreement (§1.1) |
| `ENDREP` closing an `XFORM` scope | ✘ | ✘ `unmatched_endrep` | ✘ | |
| `ENDX` closing a `REPEAT` frame | ✘ | ✘ `unmatched_endrep` | ✘ | |
| `ENDX` closing `REPEATX`'s own transform | ✘ | ✘ `unmatched_endrep` | ✘ | the loop owns the slot (`_Frame.xform_floor`) |
| `HALT` inside any open scope | ✘ halt not legal | ✘ `unterminated_repeat`, **and `halted` is true** | ✘ | an invalid halt is a halt; §3 counts it apart from a valid one |
| `REPEAT 0 …` / `REPEATX 0 …` | ✘ `COUNT` domain | ✘ `zero_repeat`, then executes as count 1 | ✘ | |
| a 5th nested `REPEAT` or transform scope | ✘ depth bound | ✘ `depth_overflow` | ✘ | `MAX_REPEAT_DEPTH = 4` bounds each stack, and `REPEATX` consumes one of each |
| `WIDTH 0` | ✔ | ✔ — clamped to 1 | ✔ | **not** masked: a clamp is a semantic choice, not a structural fault |
| a byte outside `0x00..0x0D` at an opcode boundary | ✘ | ✘ `unknown_opcode` | ✘ | the static grid, and the dominant free-running fault today (§3) |
| nested canonical loops whose counts multiply past `DEFAULT_FUEL` | ✔ | ✘ `out_of_fuel` | ✔ | a **resource** bound, in neither structural predicate — see §1.2 |

**The policy this fixes.** The paper reports canonical membership as program
membership and VM faults as a separate executor outcome. The mask may not
relabel a VM-accepted but non-canonical program as a VM fault, and may not
relabel a non-canonical program the VM accepts as valid. `unroll` is used by
neither predicate.

### 1.1 Two live defects, recorded and not repaired here

Both are in `dm.isa.unroll`, are outside the mask's path, and are left visible
rather than fixed inside a freeze:

1. **`unroll` raises where it contracts to return `None`.** Its docstring says
   it returns `None` for a program with no exact flat trace, and it catches
   `ISAError` only around `_chunks`; `Transform.of` then raises `ISAError` from
   `_expand` for any `XF > 7`. So `unroll(program_with_XF8)` raises instead of
   refusing. Unreachable from authoring (`encode_operand` refuses `XF > 7`) and
   reachable from **generated** bytes, which is the path any future
   oracle-on-samples would take.
2. **`unroll` accepts `CALL`.** It copies unknown-to-it opcodes verbatim, so a
   program the VM refuses outright expands cleanly. Also unreachable from every
   current corpus and reachable from generated bytes.

Neither changes a landed number: no corpus contains `XF > 7` or `CALL`. They are
named here so that a later repair is a repair and not a silent redefinition of
analysis acceptance.

### 1.2 Amendment, 2026-08-14, before any report existed

Two facts turned up while `dm/isa/state.py` was being built against the VM, both
before a single state report was written. They are recorded here as dated
amendments rather than folded in silently, because the rule this document opens
with cuts both ways: the instrument may be corrected, the hypothesis and the
primary metric may not.

1. **A canonical program can still exhaust fuel**, and three of eight rows from
   an untrained smoke model did: nested canonical loops whose counts multiply
   past `DEFAULT_FUEL = 100,000` executed steps. Fuel is a *resource* bound and
   belongs to neither structural predicate, so §3.1's primary endpoint reads
   "no **structural** fault", where structural means every `FaultKind` except
   `OUT_OF_FUEL`, and `fuel_rate` is reported as its own column. Nothing else
   about the endpoint moves: it is still the conjunction of a boundary halt,
   canonical membership and a clean structural execution.
2. **The VM-safe mask is prefix-local, and cannot be otherwise.** A loop replays
   its body, so a body that a *non-canonical* mask let through can fault on a
   later iteration: a dangling `XFORM` inside a `REPEAT` body pushes one more
   transform scope per iteration and eventually faults `DEPTH_OVERFLOW`, and a
   crossing `ENDX` faults `UNMATCHED_ENDREP` on the second. Both were observed
   under the VM-safe cell. Seeing either from the prefix needs the suspendable
   executor of S4, which is precisely the layer the gate decides on. So the
   VM-safe cell's guarantee is stated as *prefix-local* and its residual faults
   are a reported result, not an instrument failure. The canonical cell has no
   such residue by construction: proper nesting makes every iteration's scope
   bookkeeping identical.

---

## 2. Opcode allowlists, per checkpoint

`Tier.L2` is not a policy key — it contains `CALL`, which `VM.run` refuses — and
`TrainConfig.tier` is ignored by the QuickDraw and composed loaders entirely.
Every report in this branch therefore carries an **explicit allowlist
reconstructed from the record's own corpus**, by walking the programs the
checkpoint was scored on with `spec_for`. Measured over both splits of each
corpus (train and val agree in every case, which is a property of the
generators and is checked rather than assumed):

| venue | checkpoints | opcode allowlist | role in the audit |
|---|---|---|---|
| Tier A flat (`synthetic`, `flatten`) | `synthetic_c2flat24000_byte_square_s{0,1}` | `MOVE LINE CURVE CIRCLE WIDTH HALT` | **L0 static-only negative control**: no dynamic state exists in this language at all |
| Tier A L1 (`synthetic`) | `synthetic_converged_byte_square_s{0,1}`, `synthetic_tokmatch_byte_*` | `MOVE LINE CURVE CIRCLE WIDTH REPEAT ENDREP HALT` | L1 repeat checkpoint |
| composed, structured spelling | `composed_sconv24000_byte_square_s{0,1}` | `MOVE LINE REPEATX ENDREP HALT` | structured `REPEATX` checkpoint |
| composed, flat spelling | `composed_x24000n24_byte_square_s{0,1}`, `composed_x24000n2_*`, `composed_c24000n2_*` | `MOVE LINE HALT` | flat control on the structured corpus's geometry |

`FILL`, `XFORM`, `ENDX`, `CALL` and `COLOR` are outside every allowlist above.
`CALL` and its `Kind.ID` operand are masked under the frozen
`call_policy="unsupported"`; `COLOR` is masked as reserved. Both remain
separately attributed in the exclusion accounting of §5, because "the model put
mass on an opcode this corpus never contains" and "the model put mass on an
opcode the ISA reserves" are different observations.

---

## 3. Estimands, effect sizes, sample sizes, sampler, multiplicity

### 3.1 The primary endpoint

**`valid_halt_rate`** — the fraction of generated rows that

1. terminated at a `HALT` the parse reached at an instruction boundary (the
   halt-monitor verdict, never a symbol match), **and**
2. decode to a program that is canonical under this checkpoint's policy (§2),
   **and**
3. execute with no **structural** fault — every `FaultKind` except
   `OUT_OF_FUEL`, which is a resource bound (§1.2) and gets its own column.

All three, conjoined, on the *requested* sample, with no row dropped. Its
companions are reported beside it and never folded into it:

- `invalid_halt_rate` — halted, but non-canonical or structurally faulted;
- `fuel_rate` — `OUT_OF_FUEL`, whatever else the row did;
- `cap_rate` — the halt monitor was still live when the decode budget ended;
- `close_cost` — at the cap, the minimum number of symbols that would have
  closed the open scopes and halted (`0` where nothing was open);
- `faults` by `FaultKind`, and canonical violations by the §5 rule;
- `empty`, `len_p50`, `length_emd`, and `coverage`/`mmd`/`nna` against a
  real-drawing floor at matched set sizes.

### 3.2 The two interventions, which are not one number

- **Decoder intervention (S3, no training).** `canonical − raw` and
  `vm_safe − raw` on `valid_halt_rate`, per checkpoint, paired at the level of
  the draw under common random numbers. This is the *only* Direction 1 estimand
  that existing checkpoints can answer.
- **Model-input intervention (S5, gated).** `C − G` and `E − C` on the same
  endpoint under **unmasked** generation first, aggregated over five paired
  model seeds, then the mask interaction from the 3×2 table.

### 3.3 SESOI and the equivalence band

- **`valid_halt_rate`: SESOI = 0.05 absolute.** Derived from the historical
  between-draw spread of the closest existing column, `gen_validity`, over
  every absolute-byte AR generation report in `runs/`: the largest is
  **sd 0.0411**, on `composed_x24000n2_byte_square_s0` (mean 0.666 over five
  draws of n = 128), and that spread is *binomial* —
  `sqrt(p(1-p)/n)` at that mean and n is 0.0417. So 0.05 is one increment above
  the worst single-draw noise this project has ever recorded on the venue with
  the worst free-running behaviour. At the frozen sample size (§3.4) the
  unpaired binomial standard error at `p = 0.7` is 0.013, so an effect at the
  SESOI is ~4 standard errors before pairing buys anything.
- **Raw teacher-forced bits/drawing: equivalence band = the venue's own
  measured resolution floor.** Tier A `±1.0`, Tier A flat `±0.63`, composed
  `±2.0` (mean of eight same-config pairs; its widest pair is 4.7, and an arm
  read on that pair needs its own floor — `docs/traps.md`). Recorded now
  because it is S5's band and must predate S5's result. **The S3 decoder cells
  cannot move raw likelihood at all** — a decode-time mask changes no weight —
  so a non-zero raw difference between S3 cells is an instrument fault, not a
  finding, and the report asserts it.
- **Constrained (legal-renormalised) NLL is a diagnostic.** It is never
  substituted for raw likelihood and never enters the equivalence test.

### 3.4 Sample sizes, seeds, sampler, device

| knob | frozen value | why it is in the key |
|---|---|---|
| primary sampler | `top_k = 40`, `temperature = 1.0`, PAD/BOS forbidden | the project's published operating point; `k80` and `all` are secondary replications |
| samples per draw | **256** | matches the existing resample reports, so a raw cell is comparable with them |
| draw seeds | **5** (`0..4`) | a generation column is a draw |
| decode cap | `min(max_len, gen_cap × val p99)` from the record, `gen_cap = 2.0` | the trainer's own rule; a different cap is a different experiment |
| random variates | pre-generated `(rows, steps)` uniforms, one per row per step, consumed unconditionally | the three cells must differ in the mask and in nothing else |
| device | recorded in the key; `cpu` for the primary | backend changes the draw (`docs/traps.md`) |
| geometry | `cloud_points = 128`, reference subsampled to 256 at `REFERENCE_SEED = 1000`, floor from the training split | a set metric is a function of the two set sizes |
| model seeds | existing `s0`/`s1`, **reported separately** | two seeds bound a claim to those checkpoints |

**Unit hierarchy.** Draw seeds measure Monte Carlo noise; a scene/program
bootstrap measures uncertainty conditional on a checkpoint; only independently
trained model seeds support a claim across trained models. The S3 result is
therefore **checkpoint-conditional by construction** and is reported that way.
A five-paired-model-seed design belongs to S5 and is not licensed by anything
here.

### 3.5 Multiplicity

The S3 primary family is one test per checkpoint family — `canonical − raw` on
`valid_halt_rate` — over the **four** families of §2, Holm-corrected across those
four, and applied to each model seed's column separately because seeds are
reported seed by seed and never pooled. `vm_safe − raw`, every fault column, every length column and the
geometry columns are mechanism diagnostics and are not corrected; they are also
not promoted to a headline if the primary is null. A null primary closes the
decoder result. It is not permission to go looking for a significant column.

### 3.6 The gate this document arms

S4's gate stays **closed** until (i) the S2 dynamic-support audit, (ii) the S3
three-cell sampler report and (iii) Direction 2's C3 causal-logit report all
exist. It opens only if the baseline shows a material free-running reliability
gap, or dynamic exclusions carry non-negligible mass, or the canonical mask
moves `valid_halt_rate` by at least the §3.3 SESOI without breaking the length
and geometry floors. If all three readings are null, the branch closes and no
state arm is trained.

---

## 4. State coverage

Measured over the val split each checkpoint was scored on (1,000 programs
each), by walking the source bytes:

| venue | bytes/program | opcode boundaries at scope depth 0 | at depth ≥ 1 | max scope depth | scope openers present | closers present |
|---|---:|---:|---:|---:|---|---|
| Tier A flat | 55.2 | 18,629 | 0 | **0** | — | — |
| Tier A L1 | 42.3 | 11,376 | 3,132 | **1** | `REPEAT` (522) | `ENDREP` (522) |
| composed structured | 130.8 | 20,285 | 23,973 | **1** | `REPEATX` (1,000) | `ENDREP` (1,000) |
| composed flat | 221.4 | 74,471 | 0 | **0** | — | — |

"Scope depth" counts open **source** scopes. A `REPEATX` is one scope even though
it occupies an entry in each of the VM's two stacks — a repeat frame *and* the
transform slot the loop owns — so the composed corpus's single loop is depth 1,
not 2. The two stacks are reported separately (`loop_depth`, `xform_depth`,
`xform_floor`) wherever a feature or a mask rule reads them, because the closer
rules depend on which stack a scope sits in.

Read this table as a set of prohibitions:

- **No natural venue contains scope depth ≥ 2.** A depth feature or a
  depth-conditioned result above 1 is an unseen-embedding result until a
  supplement supplies depth, and the supplement must then be in *every* arm
  identically.
- **No natural venue contains a standalone `XFORM … ENDX`.** `endx_legal` is
  therefore a state the natural corpora never enter. It may be exercised by the
  conformance fixture and by the held-out compositional set, and an `ENDX`
  number may not be quoted as learned state (`docs/traps.md`).
- **`FILL`, `CALL`, `COLOR`, `CURVE`-in-composed and depth ≥ 2 are all
  zero-support.** Any per-state column with zero train support is printed with
  its support and read as an accounting result.
- The structured venue puts **54%** of its opcode boundaries inside a scope, so
  it is the one venue where `halt_legal = false` is common. The flat venues put
  **0%** there, which is exactly what makes them the negative control.

### 4.1 The three probe objects, kept apart

1. **Conformance fixture** — exhaustive short programs over the whole opcode
   table, in `tests/test_state.py`. Pins the parser against the VM and against
   the §1 truth table. **Never used as data**, in any arm, ever.
2. **Balanced training supplement** — not built. If S5 needs a state the
   natural corpus lacks, the supplement is generated once, fingerprinted, and
   included *identically* in G, C and E. A supplement in one arm only is not a
   control.
3. **Held-out compositional state evaluation set** — not built; it belongs to
   S6. Frozen design: vary scope type (`REPEAT`/`REPEATX`/`XFORM`), depth
   (1, 2, 3), position of the block in the program, and opener/closer
   composition *independently*, with every individual level seen in training and
   the nested ordering unseen. It is fingerprinted before use and **may not be
   used to tune bins, losses or masks** — which is why it is not built by the
   same commit that builds the mask.

---

## 5. Frozen exclusion accounting

One symbol removed from support is attributed to exactly one rule, by the first
rule below that excludes it. The order is frozen; the union is reported
separately; overlapping marginal masses are never summed.

| # | rule | what it removes | kind |
|---|---|---|---|
| 1 | `control` | `PAD`, `BOS`, and token-alphabet slots no opcode occupies | never bytecode in any alphabet |
| 2 | `static_grid` | at an opcode boundary, a byte outside the ISA opcode table | position-class only — what `dm/eval/attribution.py` already priced at 0.0008–0.0085 bits/drawing |
| 3 | `representation` | a symbol that spells a legal byte non-canonically for its alphabet: a value token at an opcode boundary, an opcode token at an operand position, a wrong-`Kind` value token under `token_typed` | canonical-only; **not** a VM fault, and reported as such |
| 4 | `opcode_policy` | `CALL` (unsupported), `COLOR` (reserved), and any ISA opcode outside this checkpoint's allowlist | policy |
| 5 | `operand_domain` | `XF ∉ 0..7`, `COUNT = 0` | `XF` is canonical-only; `COUNT = 0` is also an inevitable VM fault |
| 6 | `scope` | a closer with no matching top scope: `ENDREP` with no open repeat frame or with a transform scope above it, `ENDX` whose top closable scope is not a standalone `XFORM` | partly canonical-only — the crossing cases the VM accepts |
| 7 | `depth` | an opener at its depth bound | inevitable `depth_overflow` |
| 8 | `halt` | `HALT` at an opcode boundary with any scope open | inevitable `unterminated_repeat` |

Reported at every teacher-forced and every generated prefix, from
**temperature-1 logits before `forbid` and before `top_k`**:

- `q = P(symbol ∉ support)`, the raw illegal mass; and
- `-log2(1 - q)`, the renormalisation cost a mask recovers.

`-log2(q)` is **not** reported: it grows as illegal mass shrinks and reads
backwards. `q` is decomposed by the table above and by state stratum
(`opcode` with halt legal, `opcode` inside a scope, and one stratum per operand
`Kind`), at both mask levels:

- **`vm_safe`** removes rules 1, 2, 5 (`COUNT = 0` only), 7, 8, the
  VM-terminal part of 6, and 4's `CALL`.
- **`canonical`** removes all eight, and is a subset of `vm_safe`.

An **empty** legal set is an instrument error and is reported as one. Falling
back to the raw distribution would turn a structural claim into a best-effort
sampler. A row whose prompt or whose earlier unmasked symbols already left the
canonical language is **absorbed**: it maps to the declared absorbing state
(`Status.CANONICAL_FAULT`, which is monotone) and the count of absorbed rows is
published beside every masked cell. Its mask is **not** loosened — the rules are
local, so the declared level stays well defined and non-empty on a non-canonical
prefix, and loosening it would be the best-effort sampler this paragraph
forbids one sentence earlier. (The first draft of this section said the mask
falls back to `vm_safe`; the instrument does not, and this is the corrected
statement rather than a change of policy.)

---

## 6. What a mask does not buy

Recorded before the run, because it is the most likely misreading of a good
number: **a legality mask does not guarantee termination.** Forbidding `HALT`
inside an open scope converts an invalid early halt into a longer program, and
possibly into a cap hit. A row that hits the cap is *incomplete*, never
upgraded to valid. A budget-aware force-close decoder is a different policy,
is not part of the primary mask result, and if it is ever added it is a
**fourth** cell with its own identity — it may not appear under the canonical
cell's key.
