#!/usr/bin/env python3
"""Apply Direction 2's frozen decision rule across the co-primary venues.

`scripts/context_c3.py` scores one manifest over one corpus.  The two
co-primary venues live on two corpora, so the multiplicity correction and the
claim label cannot be decided inside either run -- this Adapter is where the
frozen procedure lives, and it reads the protocol rather than restating it.

```bash
PYTHONPATH=. .venv/bin/python scripts/context_gate.py \
  --protocol docs/context-protocol-v1.json \
  runs/context_c3_synthetic_byte_<hash>.json \
  runs/context_c3_composed_byte_<hash>.json
```

Fails closed.  A missing venue, a missing instrument, an unfrozen protocol or
a report from another manifest yields `incomplete` -- never a negative and
never a positive (§4.8).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dm.eval.context import CO_PRIMARY_VENUES
from dm.eval.provenance import sha256_file, verify_frozen_sources
from dm.eval.reports import json_safe, staged_path
from dm.eval.state_analysis import holm

ENDPOINT = "D_minus_control"
POINT_FIELD = "D_minus_control_bits_per_byte_mean"
SOURCE_PATHS = (
    Path("scripts/context_gate.py"), Path("dm/eval/context.py"),
    Path("dm/eval/state_analysis.py"), Path("dm/eval/provenance.py"),
)


def _report_problems(path: Path, report: dict, manifests: dict,
                     *, stage: str, required_instruments: tuple[str, ...] = (),
                     protocol: dict | None = None,
                     protocol_path: Path | None = None) -> list[str]:
    """Validate report identity before reading any result column."""
    problems: list[str] = []
    if report.get("status") != "complete":
        problems.append(f"{path}: {stage} status={report.get('status')!r}")
    expected_schema = 3 if stage == "C3" else None
    if expected_schema is not None and report.get("report_schema") != expected_schema:
        problems.append(
            f"{path}: {stage} report_schema={report.get('report_schema')!r}, "
            f"expected {expected_schema}"
        )
    entry = manifests.get(report.get("manifest", ""))
    if not entry or entry.get("manifest_sha256") != report.get("manifest_sha256"):
        problems.append(f"{path}: {stage} manifest is not the frozen one")
        return problems

    rows = report.get("reports")
    if not isinstance(rows, list):
        problems.append(f"{path}: {stage} reports are missing")
        return problems
    expected_checkpoints = sorted(entry.get("checkpoints", ()), key=str)
    actual_checkpoints = sorted((row.get("checkpoint") for row in rows), key=str)
    if not expected_checkpoints or actual_checkpoints != expected_checkpoints:
        problems.append(
            f"{path}: {stage} checkpoints {actual_checkpoints} != frozen "
            f"{expected_checkpoints}"
        )
    if any(not isinstance(row, dict) for row in rows):
        problems.append(f"{path}: {stage} contains a malformed checkpoint row")
        return problems
    seeds = [row.get("model_seed") for row in rows]
    if any(seed is None for seed in seeds) or len(set(seeds)) != len(seeds):
        problems.append(f"{path}: {stage} model seeds are missing or duplicated")
    weight_hashes = [row.get("checkpoint_sha256") for row in rows]
    if len(set(weight_hashes)) != len(weight_hashes):
        problems.append(f"{path}: {stage} checkpoint hashes are duplicated")

    expected_venues = set(entry.get("venues", ()))
    for row in rows:
        actual_venues = set((row.get("venues") or {}).keys())
        if not expected_venues or not expected_venues.issubset(actual_venues):
            problems.append(
                f"{path}: {stage} checkpoint {row.get('checkpoint')!r} misses "
                "a frozen venue"
            )
        for field in ("checkpoint_sha256", "record_sha256"):
            value = row.get(field)
            if not isinstance(value, str) or len(value) != 64:
                problems.append(
                    f"{path}: {stage} checkpoint {row.get('checkpoint')!r} "
                    f"lacks {field}"
                )

    if stage == "C3":
        checks = (report.get("gate_input") or {}).get("instrument_checks")
        if not isinstance(checks, dict):
            checks = {}
        for name in required_instruments:
            if checks.get(name) is not True:
                problems.append(f"{path}: instrument check {name} missing or failed")
    elif protocol is not None:
        c4 = (protocol.get("context_c3") or {}).get("c4")
        if c4:
            if c4.get("estimand", {}).get("controls") and report.get(
                    "report_schema") != 5:
                problems.append(
                    f"{path}: controlled C4 report_schema="
                    f"{report.get('report_schema')!r}, expected 5"
                )
            expected_config = {
                "draws_per_world": c4["diagnostic"]["draws_per_world"],
                "cap_symbols": c4["diagnostic"]["cap_symbols"],
                "top_k": c4["sampler"]["top_k"],
                "temperature": c4["sampler"]["temperature"],
                "variate_seed": c4["sampler"]["variate_seed"],
                "structural_mask": "off",
                "shared_uniform_variates": True,
            }
            if report.get("completion_config") != expected_config:
                problems.append(f"{path}: C4 completion config differs from protocol")
            if (protocol_path is None or report.get("protocol_sha256")
                    != sha256_file(protocol_path)):
                problems.append(f"{path}: C4 report used another protocol")
    return problems


def _venue_rows(reports: list[dict]) -> dict[str, dict]:
    """Per venue, the per-seed point estimates, intervals and p-values."""
    out: dict[str, dict] = {}
    for report in reports:
        for entry in report["reports"]:
            for venue, summary in entry["venues"].items():
                bootstrap = summary[f"bootstrap_{ENDPOINT}"]
                out.setdefault(venue, {"seeds": {}})["seeds"][
                    str(entry.get("model_seed", entry["name"]))
                ] = {
                    "checkpoint": entry["checkpoint"],
                    "n_cases": summary["n_cases"],
                    "delta_bits_per_byte": summary[POINT_FIELD],
                    "raw_D_bits_per_byte": summary["D_bits_per_byte_mean"],
                    "delta_block_bits_per_byte": summary[
                        "D_minus_block_control_bits_per_byte_mean"],
                    "ci95": bootstrap["ci95"],
                    "p_value": bootstrap["p_value"],
                    "clusters": bootstrap["clusters"],
                    "block_ci95": summary[
                        "bootstrap_D_minus_block_control"]["ci95"],
                    "prefix_bits_per_byte": summary.get(
                        "prefix_bits_per_byte_mean"),
                    "matched_target_bits_per_byte": summary.get(
                        "matched_target_bits_per_byte_mean"),
                }
    return out


def _support(venues: dict, diagnostic: str, primary: str, limits: dict) -> dict:
    """Is an axis diagnostic close enough to its primary to mean anything?

    A diagnostic venue that the model cannot predict has a null for a reason
    that has nothing to do with the relation, and reading it as "the relation
    does not transfer" is the same category error as reading a missing control
    as a pass.  The ratios are read off the frozen `prefix_bits_per_byte` and
    matched-target columns, and both bounds are declared in the protocol before
    the diagnostic is scored.
    """
    if diagnostic not in venues or primary not in venues:
        return {"in_support": False, "reason": "venue missing"}
    ratios = []
    for seed, row in venues[diagnostic]["seeds"].items():
        base = venues[primary]["seeds"].get(seed)
        if not base or not base.get("prefix_bits_per_byte"):
            return {"in_support": False, "reason": f"no primary for seed {seed}"}
        ratios.append({
            "seed": seed,
            "prefix_cost_ratio": row["prefix_bits_per_byte"]
            / base["prefix_bits_per_byte"],
            "matched_target_cost_ratio": row["matched_target_bits_per_byte"]
            / base["matched_target_bits_per_byte"],
        })
    prefix_max = float(limits.get("max_prefix_cost_ratio", float("inf")))
    target_max = float(limits.get("max_matched_target_cost_ratio", float("inf")))
    ok = all(r["prefix_cost_ratio"] <= prefix_max
             and r["matched_target_cost_ratio"] <= target_max for r in ratios)
    return {
        "in_support": ok,
        "primary": primary,
        "max_prefix_cost_ratio": prefix_max,
        "max_matched_target_cost_ratio": target_max,
        "seeds": ratios,
        "reason": None if ok else "diagnostic prefixes are outside the "
                                  "model's support; its null is uninformative",
    }


def _generation(reports: list[dict]) -> dict[str, dict]:
    """Per venue, C4's paired preference and what it did behaviourally.

    Kept apart from `_venue_rows` because a C4 report is a different stage with
    a different estimand: `D_gen` is a normalised distance, not a rate in bits,
    and the two must never be compared as though they were one number.
    """
    out: dict[str, dict] = {}
    for report in reports:
        for entry in report["reports"]:
            for venue, summary in entry["venues"].items():
                out.setdefault(venue, {"seeds": {}})["seeds"][
                    str(entry.get("model_seed", entry["name"]))
                ] = {
                    "checkpoint": entry["checkpoint"],
                    "n_cases": summary["n_cases"],
                    "d_gen": summary["D_gen_mean"],
                    "ci95": summary["bootstrap_D_gen"]["ci95"],
                    "half_width": summary["half_width"],
                    "reach_rate": summary["reach_rate"],
                    "hit_own_rate": summary["hit_own_rate"],
                    "hit_other_rate": summary["hit_other_rate"],
                    "relation_consistent_rate": summary["relation_consistent_rate"],
                    "canonical_rate": summary["canonical_rate"],
                    # Present only from report schema 5. A v4 report has no
                    # generation control, and `None` says so rather than
                    # letting a missing control read as a passed one.
                    "d_gen_block_control": summary.get(
                        "D_gen_block_control_mean"),
                    "d_gen_target_control": summary.get(
                        "D_gen_target_control_mean"),
                    "delta_gen": summary.get("D_gen_minus_block_control_mean"),
                    "delta_gen_ci95": (
                        summary.get("bootstrap_D_gen_minus_block_control")
                        or {}).get("ci95"),
                }
    return out


def _generation_label(rows: dict | None) -> str:
    """`exposure_limited` or `generation_robust`, on sign and interval only.

    **This rule carries no effect-size floor and the report says so.** C3's
    label clears a frozen SESOI; C4's does not, because none was frozen for
    `D_gen`. A venue can therefore be `generation_robust` on a preference far
    too small to produce the compatible continuation, and the behavioural
    columns beside it are what a reader must use to see that.
    """
    if not rows or not rows["seeds"]:
        return "not_run"
    seeds = list(rows["seeds"].values())
    if all(seed["ci95"][0] > 0 for seed in seeds):
        return "generation_robust_context_use"
    if all(seed["ci95"][1] < 0 for seed in seeds):
        return "generation_reversed"
    return "exposure_limited_context_use"


def _controlled_generation_label(rows: dict | None) -> str:
    """The same verdict on `Delta_gen`, once the irrelevant edit is subtracted.

    `D_gen` answers "did the completions move when the prefix moved". It does
    not answer "was it the *relation* that moved them", and a model that lurches
    at any prefix edit scores the same on it -- which is the schema-1 fault one
    stage later (`docs/directions.md` §2.2). This reads the matched
    unrelated-block contrast instead.

    Declared here **before** the run that carries the control, so the rule is
    not chosen from the number. A v4 report has no control column and returns
    `not_measured`: a missing control is never a passed one (§4.8).
    """
    if not rows or not rows["seeds"]:
        return "not_run"
    seeds = list(rows["seeds"].values())
    if any(seed.get("delta_gen_ci95") is None for seed in seeds):
        return "not_measured"
    if all(seed["delta_gen_ci95"][0] > 0 for seed in seeds):
        return "controlled_generation_context_use"
    if all(seed["delta_gen_ci95"][1] < 0 for seed in seeds):
        return "controlled_generation_reversed"
    return "no_controlled_generation_evidence"


def _label(venue: dict, sesoi: float, axis: list[tuple[dict, dict]]) -> str:
    """The frozen claim label for one co-primary venue.

    `axis` carries `(diagnostic venue, its support verdict)` pairs. A diagnostic
    that is out of support is skipped rather than read as a failure to transfer:
    §4.8's `incomplete` rule applies to a diagnostic that cannot see, exactly as
    it applies to a missing control.
    """
    seeds = list(venue["seeds"].values())
    if not seeds:
        return "incomplete"
    excludes = all(seed["ci95"][0] > 0 or seed["ci95"][1] < 0 for seed in seeds)
    positive = all(seed["delta_bits_per_byte"] > 0 for seed in seeds)
    clears = all(seed["delta_bits_per_byte"] >= sesoi for seed in seeds)
    block = all(seed["block_ci95"][0] > 0 for seed in seeds)
    raw = all(seed["raw_D_bits_per_byte"] >= sesoi for seed in seeds)
    if not venue["holm"]["reject"] or not excludes:
        # A raw compatibility preference that does not survive the matched
        # control is the historical schema-1 finding, and it has its own name.
        return "lower_order_context_use" if raw else "no_controlled_evidence"
    if not (positive and clears and block):
        return "causal_prefix_sensitivity"
    readable = [rows for rows, verdict in axis if verdict["in_support"]]
    if readable and not all(
        seed["ci95"][0] > 0
        for rows in readable for seed in rows["seeds"].values()
    ):
        return "anisotropic_relation_use"
    return "relational_context_use"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("reports", type=Path, nargs="+")
    ap.add_argument("--c4", type=Path, nargs="*", default=(),
                    help="C4 completion reports; adds the generation verdict")
    ap.add_argument("--protocol", type=Path,
                    default=Path("docs/context-protocol-v5.json"))
    ap.add_argument("--out", type=Path,
                    default=Path("runs/context_gate_v5.json"))
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    if args.out.exists() and not args.overwrite:
        raise SystemExit(f"{args.out} already exists; pass --overwrite")
    protocol = json.loads(args.protocol.read_text())
    if protocol.get("status") != "frozen":
        raise ValueError("selected context protocol is not frozen")
    if "source_sha256" in protocol:
        verify_frozen_sources(protocol, SOURCE_PATHS)
    # A protocol missing a frozen field raises: that is a broken freeze, not a
    # measurement outcome, and it must not be reportable as `incomplete`
    # alongside honest data problems.
    frozen = protocol["context_c3"]
    manifests = frozen["manifests"]
    sesoi = float(frozen["sesoi_bits_per_target_byte"])
    alpha = float(frozen["alpha"])
    family = list(frozen["co_primary_venues"])

    reports = [json.loads(path.read_text()) for path in args.reports]
    problems: list[str] = []
    # A protocol may not promote a diagnostic venue into the confirmatory
    # family after the fact; the code owns which venues are eligible at all.
    unknown = [venue for venue in family if venue not in CO_PRIMARY_VENUES]
    if unknown:
        problems.append(f"protocol names non-co-primary venues: {unknown}")
    required = tuple(frozen.get("required_instruments", ()))
    for path, report in zip(args.reports, reports):
        problems.extend(_report_problems(
            path, report, manifests, stage="C3", required_instruments=required,
        ))

    c4_reports = [json.loads(path.read_text()) for path in args.c4]
    for path, report in zip(args.c4, c4_reports):
        problems.extend(_report_problems(
            path, report, manifests, stage="C4", protocol=protocol,
            protocol_path=args.protocol,
        ))

    # C3 and C4 must use identical weight and record bytes, not merely identical
    # checkpoint filenames. The protocol historically froze paths only; this
    # cross-stage check closes substitution between the two measured stages.
    c3_identity = {
        (report["manifest_sha256"], row.get("checkpoint")):
            (row.get("checkpoint_sha256"), row.get("record_sha256"))
        for report in reports for row in report.get("reports", ())
    }
    for path, report in zip(args.c4, c4_reports):
        for row in report.get("reports", ()):
            key = (report.get("manifest_sha256"), row.get("checkpoint"))
            identity = (row.get("checkpoint_sha256"), row.get("record_sha256"))
            if key not in c3_identity or c3_identity[key] != identity:
                problems.append(
                    f"{path}: C4 checkpoint bytes differ from C3 for "
                    f"{row.get('checkpoint')!r}"
                )
    try:
        generation = _generation(c4_reports)
    except (KeyError, TypeError, ValueError) as exc:
        problems.append(f"malformed C4 result columns: {exc}")
        generation = {}

    try:
        venues = _venue_rows(reports)
    except (KeyError, TypeError, ValueError) as exc:
        problems.append(f"malformed C3 result columns: {exc}")
        venues = {}
    missing = [venue for venue in family if venue not in venues]
    if missing:
        problems.append(f"missing co-primary venues: {missing}")

    # Holm over the co-primary family only.  The per-venue p-value is the worst
    # of the model seeds, because the claim requires both of them (§4.8) and a
    # correction applied to the better seed would be a correction applied to the
    # wrong hypothesis.
    pvalues = {
        venue: max(seed["p_value"] for seed in venues[venue]["seeds"].values())
        for venue in family if venue in venues
        and venues[venue]["seeds"]
    }
    decisions = holm(pvalues, alpha=alpha) if pvalues else {}
    for venue, decision in decisions.items():
        venues[venue]["holm"] = decision

    # Support verdicts first, so a label never depends on a diagnostic that
    # could not see. `axis_diagnostic` maps primary -> one or many diagnostics.
    declared = frozen.get("axis_diagnostic", {})
    limits = frozen.get("axis_support", {})
    support: dict[str, dict] = {}
    for primary, names in declared.items():
        for name in ([names] if isinstance(names, str) else names):
            support[name] = _support(venues, name, primary, limits)

    labels = {}
    for venue in family:
        if venue not in venues or "holm" not in venues[venue]:
            labels[venue] = "incomplete"
            continue
        names = declared.get(venue, [])
        axis = [(venues[name], support[name])
                for name in ([names] if isinstance(names, str) else names)
                if name in venues]
        labels[venue] = ("incomplete" if problems
                         else _label(venues[venue], sesoi, axis))

    generation_labels = {
        venue: ("incomplete" if problems else _generation_label(generation.get(venue)))
        for venue in family
    }
    controlled_labels = {
        venue: ("incomplete" if problems
                else _controlled_generation_label(generation.get(venue)))
        for venue in family
    }
    result = {
        "gate_schema": 2,
        "protocol": str(args.protocol),
        "protocol_sha256": sha256_file(args.protocol),
        "reports": [{"path": str(path), "stage": "C3",
                     "sha256": sha256_file(path),
                     "manifest_sha256": report.get("manifest_sha256")}
                    for path, report in zip(args.reports, reports)]
        + [{"path": str(path), "stage": "C4",
            "sha256": sha256_file(path),
            "manifest_sha256": report.get("manifest_sha256")}
           for path, report in zip(args.c4, c4_reports)],
        "sesoi_bits_per_target_byte": sesoi,
        "alpha": alpha,
        "family": family,
        "venues": venues,
        "holm": decisions,
        "axis_support": support,
        "generation": generation,
        "generation_labels": generation_labels,
        "controlled_generation_labels": controlled_labels,
        "labels": labels,
        "problems": problems,
        "status": "incomplete" if problems else "complete",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    staged = staged_path(args.out)
    staged.write_text(json.dumps(json_safe(result), indent=2, sort_keys=True,
                                 allow_nan=False))
    staged.replace(args.out)
    for venue, label in labels.items():
        generated = generation_labels.get(venue, "not_run")
        controlled = controlled_labels.get(venue, "not_run")
        print(f"{venue}: {label}  |  generation: {generated}"
              f"  |  controlled: {controlled}")
        rows = generation.get(venue)
        if rows:
            for seed, row in sorted(rows["seeds"].items()):
                print(f"    seed {seed}: D_gen={row['d_gen']:+.4f} "
                      f"CI={[round(v, 4) for v in row['ci95']]} "
                      f"reach={row['reach_rate']:.3f} "
                      f"hit_own={row['hit_own_rate']:.4f} "
                      f"relation={row['relation_consistent_rate']:.4f}")
                if row.get("delta_gen_ci95") is not None:
                    print(f"             Delta_gen={row['delta_gen']:+.4f} "
                          f"CI={[round(v, 4) for v in row['delta_gen_ci95']]} "
                          f"block={row['d_gen_block_control']:+.4f} "
                          f"target={row['d_gen_target_control']:+.4f}")
    for name, verdict in sorted(support.items()):
        ratios = ", ".join(f"seed {r['seed']} prefix x{r['prefix_cost_ratio']:.2f}"
                           f" target x{r['matched_target_cost_ratio']:.1f}"
                           for r in verdict.get("seeds", []))
        state = "readable" if verdict["in_support"] else "OUT OF SUPPORT"
        print(f"  axis {name}: {state} ({ratios or verdict['reason']})")
    for problem in problems:
        print(f"  ! {problem}")
    print(f"  -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
