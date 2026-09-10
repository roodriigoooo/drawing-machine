# Direction 2 independent artifact and implementation audit

**Audited 2026-08-16 on branch `direction-2-c4`.** This closes paper handoff for
Direction 2. Evidence-producing source is commit `ab9e980`; every source digest
in [`context-protocol-v5.json`](context-protocol-v5.json) matches that commit.
Later audit hardening changes no frozen manifest, report, gate result or claim.

## Verdict

C3 and C4 numbers in [`context.md`](context.md) trace to complete, content-hashed
reports over identical manifest and checkpoint bytes. Both manifests rebuilt to
the frozen hashes; both C4 `reports` payloads replayed bit-identically from
commit `ab9e980` under v5. Recomputed table values,
component intervals, controls, hit rates and labels match artifacts after four
presentation corrections: diagnostic rows had mixed raw `D` with `Delta`, step
`Delta` is positive in 63/64 rather than 64/64 cases per seed, y-axis raw-`D`
positive counts were seed-reversed, and two paragraphs retained v4's superseded
0.7%/0.2% composed hit rates.

Result survives audit. Scope narrows where implementation proves less than
prose once implied: checkpoint content hashes were recorded after scoring but
not frozen before it, and held-out status persists split/program identity rather
than a separately defined motif-equivalence identity.

## Artifact chain

| artifact | SHA-256 |
|---|---|
| synthetic manifest | `dcc93caf0003f42ecb5b0ebcb23f29caf5a213942c754f1e7b4480ab0cd79b50` |
| composed manifest | `2679fc2356da178421b31de01c52758a4a0262392d2bc0a56c3a504b340344b5` |
| synthetic C3 report | `dda610378e3c6faa56cf972286a77543d6efc8e070e515cca9ad1dff156adcc8` |
| composed C3 report | `05e6d1656485c5c59ecdd08ea5327689b07933ea5628ebe26e5219993b179f49` |
| synthetic controlled C4 report | `165ae3cbe54f3144e97b99db77d4c1ae71887233141d59edd80e94f180095b20` |
| composed controlled C4 report | `6781bc2323343bd1addece466b9da92fd4187b236735c9fd9d25fabb9cf262a7` |
| confirmatory v2 gate | `ceb43d1491f6548dd561d00a80da898019ad5f84987e3c543882d0a35a5e8f13` |
| controlled v5 gate | `b30651c8a684590620adb652fc6c05e4350f68321b83a202e697f4cdc65d130c` |
| Figure 15 after correction | `d1ccb0f2389fcce9c1ca38ad7fc415db938557ef65e37dcbfc7e6234988955db` |

Both gate outputs name report hashes above. Both C3 and C4 reports name identical
checkpoint and record hashes:

| checkpoint | weights SHA-256 | record SHA-256 |
|---|---|---|
| synthetic seed 0 | `537968278c47115d118bb534cbd4e7a6da51c83e6ef41aea093db8432326fc9a` | `4da06c7687a09f7bf811ea8954e557a161f99e82cd2022a7b38c7b5945342742` |
| synthetic seed 1 | `c43ca6ced115a9c9e34e2bfbc6e6393dba7e1b9133df08324673dda7ce8368d8` | `d980b4e224ea4ba367ad18943a5af4156642d26c512b9a6764708269942e6aa0` |
| composed seed 0 | `7c991933756ca5e6a55e65ff3bbe95f32fb777e075a92debdde86439ac768172` | `ce5e353b597fc049f45525a6ede04203324b464ebea1199556f97193ed807438` |
| composed seed 1 | `b9fbde5338d54780c16c69bf8d13c8f229cb5c2071c0f58957d46d8e222f75e6` | `230de9a089b33a5ab50d5355a7edd931f6e54fb3c55aac415d6adfb192d5b138` |

## Reconciled result

- C3 co-primary `Delta`: composed `+4.2796/+4.7193`, step
  `+1.3337/+1.2839` bits/target-byte. Both component intervals exclude zero;
  Holm-adjusted p is 0.001. Donor-target and unrelated-block controls measure
  near zero.
- C4 controlled `Delta_gen`: composed `+0.0337/+0.0315`, step
  `+0.0824/+0.0583`. All four component intervals exclude zero. Irrelevant-edit
  controls range from -0.0034 to +0.0014.
- Compatible output remains rare: composed 1.3%/1.1%; step 13.5%/7.4%.
- Confirmatory v2 labels both co-primary venues `relational_context_use`.
  Corrected post-hoc axis rule changes step to `anisotropic_relation_use`
  exploratorily; v5 preserves that distinction.

## Implementation findings

Audit found and repaired three fail-closed gaps for future use:

1. Gate accepted absent required-instrument keys and one-checkpoint reports.
2. Gate did not prove C3 and C4 used identical checkpoint/record bytes.
3. Drivers relied on a test for source drift; selected protocol did not enforce
   source hashes at runtime. C4 also omitted the per-checkpoint policy check.

New checks reject all four cases. Current verification: 73 context tests and
760 full-suite tests pass; scoped Ruff and both link/budget checks pass. Full
Ruff still reports 68 unrelated baseline findings. Frozen v2/v5 outputs remain
historical artifacts; exact reruns use commit `ab9e980`, not post-audit code
under an old protocol.

## Residual limits carried forward

- v1-v5 froze checkpoint paths, not checkpoint/record hashes. Cross-stage hashes
  agree now, but preregistration did not content-address weights. Direction 3
  later closed before its planned content freeze; any future scientific
  direction must name checkpoint and record hashes before evaluation.
- `held_out_from_train` proves train/validation program disjointness; composed
  source pools additionally come from QuickDraw train/validation splits. Schema
  does not persist a separately defined motif-equivalence group. Quote
  “held-out validation programs/scenes,” not universal motif novelty.
- Two trained seeds support checkpoint-conditional claims only.
- Axis transfer remains exploratory because readable precondition changed after
  result.
- C4 labels have no SESOI. They establish controlled detectable preference, not
  useful generation capability.
- Composed precision is capped by 21 dependence components and corpus yield;
  more draws do not fix it.
