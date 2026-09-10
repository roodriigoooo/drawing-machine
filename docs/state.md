# Direction 1 state audit — corrected interpretation

**Status: S4 closed, 2026-08-15.** The 2026-08-15 v2 run repaired several
real implementation faults and produced useful measurements, but its original
gate interpretation is withdrawn. Geometry was not the sole blocker. The only
large decoder effect came from a flat language with no dynamic scope state; the
state-bearing families show neither the frozen baseline gap nor a material
dynamic-support or decoder effect. The C3 control was also degenerate.

[`state-protocol-v2.json`](state-protocol-v2.json) and its ignored `runs/`
artifacts are retained as historical, content-addressed inputs. They are not an
independently preregistered authorization record: v2 was written after the first
implementation, and the current source fixes intentionally bump the S2/S3/C3
report schemas. No checkpoint was trained and no weight changed.

## What Direction 1 has actually found

| signal | reading | interpretation |
|---|---:|---|
| static field-grid cost | 0.0008–0.0085 bits/drawing in the earlier attribution study | models already learn opcode/operand position structure; this is a useful negative control for a state input |
| canonical dynamic cost, synthetic L1 | 0.000070–0.000131 bits/drawing | measurable but tiny; below the 0.01 gate threshold |
| canonical dynamic cost, structured `REPEATX` | 0.002215 / 0.007722 bits/drawing | the largest genuine dynamic signal, still below threshold in both model seeds |
| canonical minus raw validity, state-bearing families | +0.0023 to +0.0109 | no material structured-family decoder effect; none survives Holm at the frozen SESOI |
| canonical minus raw validity, flat `composed_x24000n2` s0 | +0.3391 | a strong static/policy decode-time safety result, not evidence for execution state |
| VM-safe minus raw, flat s0 | validity +0.0758, cap +0.1703, length EMD +42.6 bytes | prefix-local legality can move failure mass into long/capped programs; a mask is not an execution proof |
| raw C3 step contrast | about +1.23 bits/target-byte in both checkpoints | a promising compatibility signal only; the recorded control and uncertainty calculation do not validate a relational claim |

The strongest trustworthy Direction 1 result is therefore about decoding:
canonical constraints can repair exposure-bias failures in a weak flat model.
That result is operationally useful, but it does not answer whether providing
grammar or execution state to the model improves learning.

## Signals not found

- No state-bearing raw baseline crosses the historical `valid_halt <= 0.95`
  relevance threshold. The four structured readings are 0.9859–0.9969.
- No state-bearing teacher-forced canonical dynamic cost reaches 0.01
  bits/drawing.
- No state-bearing canonical decoder effect reaches the 0.05 SESOI; observed
  effects are +0.0023 to +0.0109 and none is Holm-rejected.
- No experiment has yet shown that cursor, transform, path, repeat-replay or
  other execution features add predictive value. Those features have not been
  implemented or trained.
- No valid C3 relational-context result exists yet. The raw step contrast is not
  enough without a non-degenerate control, donor-aware uncertainty and the
  frozen support/frequency/held-out checks.
- Natural training coverage does not include depth >=2 or a standalone
  `XFORM ... ENDX` venue. The conformance fixture tests those states but is not
  evidence that a model learned them.

## Historical v2 measurements

These numbers remain useful as provisional, checkpoint-conditional
measurements. They must not be represented as current-schema outputs. In
particular, v2 normalised support probabilities in float32 and reported
population rather than sample SD; its SD/tiny-tail columns are withdrawn.

### Teacher-forced support

The quantity is `-log2(legal mass)` from temperature-1 logits before `forbid`
and `top_k`. Dynamic columns are exact stagewise scope/depth/halt costs.

| family / seed | canonical total | canonical dynamic | VM-safe total | VM-safe dynamic |
|---|---:|---:|---:|---:|
| synthetic c2flat / s0 | 0.004613 | 0.000000 | 0.004519 | 0.000039 |
| synthetic c2flat / s1 | 0.004233 | 0.000000 | 0.004096 | 0.000046 |
| synthetic converged / s0 | 0.003585 | 0.000131 | 0.003453 | 0.000170 |
| synthetic converged / s1 | 0.003751 | 0.000070 | 0.003624 | 0.000095 |
| composed structured / s0 | 0.012082 | 0.002215 | 0.005971 | 0.002247 |
| composed structured / s1 | 0.018966 | 0.007722 | 0.016957 | 0.007815 |
| composed flat n=2 / s0 | 0.002921 | 0.000000 | 0.002709 | 0.000042 |
| composed flat n=2 / s1 | 0.004409 | 0.000000 | 0.004200 | 0.000022 |

Dynamic canonical cost is exactly zero in both flat controls. This is also why
the large flat-mask gain cannot satisfy an execution-state relevance gate.

### Decoder cells

| family / seed | state-bearing | raw | VM-safe | canonical | canonical − raw | historical status |
|---|:---:|---:|---:|---:|---:|---|
| synthetic c2flat / s0 | no | 0.9984 | 1.0000 | 1.0000 | +0.0016 | complete |
| synthetic c2flat / s1 | no | 1.0000 | 1.0000 | 1.0000 | +0.0000 | complete |
| synthetic converged / s0 | yes | 0.9969 | 1.0000 | 1.0000 | +0.0031 | incomplete geometry |
| synthetic converged / s1 | yes | 0.9977 | 0.9992 | 1.0000 | +0.0023 | incomplete geometry |
| composed structured / s0 | yes | 0.9922 | 0.9938 | 0.9977 | +0.0055 | complete |
| composed structured / s1 | yes | 0.9859 | 0.9938 | 0.9969 | +0.0109 | complete |
| composed flat n=2 / s0 | no | 0.6609 | 0.7367 | 1.0000 | +0.3391 | complete |
| composed flat n=2 / s1 | no | 0.9555 | 0.9648 | 0.9992 | +0.0438 | complete |

Holm rejects only the flat n=2 family in each model seed. Seed 0 clears the
0.05 SESOI; seed 1 does not. This supports a decoder intervention on one weak
flat checkpoint, not S4.

The two empty geometry draws are real incomplete endpoints. Because cells use
the same fixed weights, seeds and variates, rerunning the same draw will
reproduce the empty. A future protocol must define emptiness as an observed
floor failure or predeclare another estimator; it cannot “repair” the draw by
retrying or dropping it.

### C3 retraction

The historical manifest has 64 cases from 53 primary sources, but all 64
control pairs use identical targets. Consequently `D_control` is zero by
construction (approximately `4e-9` and exactly `0` in the two reports), so the
reported `D-control` is just `D`. Donor source 439 is reused in 62 cases and
source 845 in two, while the bootstrap clusters only by primary source. The
report also lacks the support/frequency/held-out diagnostics required by its
own decision rule.

The +1.2363 and +1.2272 raw `D` estimates may motivate a corrected experiment,
but the claim “C3 positive/complete” is withdrawn.

## Instrument repairs now in source

- `LegacyStateMonitor` leaves raw termination to the real `HaltMonitor` while a
  separate observer records grammar faults. Generated-prefix mass stops when
  that observer becomes terminal without stopping raw sampling.
- Prompt trailing bytes, absorbing faults, call-policy closure, VM-versus-policy
  depth and symbol-level representation faults have direct tests.
- Inverse-CDF variates must be finite values in `[0,1)`, and a float32 CDF tail
  cannot select a masked zero-probability symbol.
- Support softmax now runs in float64 on the host rather than normalising in
  float32 and casting afterward. Reports use sample SD and retain unrounded
  per-program values.
- Empty rate is structural and is reported even with `--no-quality`; geometry
  floor caching keys both the training and validation fingerprints.
- S2 and S3 are schema 3; Holm is schema 2. Old reports cannot be mixed with a
  repaired rerun.
- C3 manifest schema 2 rejects identical control targets, balances donor sources
  and bootstraps connected source/donor components. Manifest construction is a
  separate pre-model step.
- The gate excludes flat languages from execution-state evidence, verifies Holm
  identities, requires complete C3 controls, current artifact schemas and
  machine-readable numeric criteria. It uses canonical dynamic bits directly;
  the historical canonical-minus-VM-safe “dynamic residual” is withdrawn because
  stagewise attribution changes when earlier policy rules remove support.
  Historical prose is explanatory only and cannot authorize S4.

The architecture now has a useful separation: grammar, provenance, support
analysis and context scoring are distinct Modules; codec and monitor Interfaces
are explicit; scripts are mostly Adapters. Before another gate run, move the
remaining decision logic into a tested analysis Module rather than deepening the
script Adapter.

## What blocks S4

There is a scientific blocker and an instrumentation blocker.

1. **Scientific:** the corrected state-bearing evidence fails all three relevance
   routes: baseline gap, dynamic mass and material decoder effect. Completing
   geometry alone cannot change that conclusion.
2. **Instrumentation:** no current protocol can authorize a rerun. A replacement
   must freeze machine-readable thresholds, state-bearing family eligibility,
   report schemas/source identity, an explicit empty-geometry rule and a new C3
   manifest before model scoring.
3. **C3:** implement and freeze support/frequency/position and held-out-donor
   controls. For the full Direction 2 claim, add the exact y-axis rotation and
   composed shape-twin venue as well.
4. **Fresh evidence:** only after that freeze, rerun S2/S3 under schema 3, C3
   under schema 2, then Holm and the gate. Historical v2 artifacts are not
   current inputs.

Until those steps are complete and the mechanically checked state-bearing gate
passes, do not implement `ExecutionSnapshot` or train G/C/E. If the project
chooses an explicitly exploratory S4 engineering branch despite the negative
gate, record that as a scope decision rather than calling it evidence-driven.
