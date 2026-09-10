#!/usr/bin/env python3
"""Mechanically evaluate a content-addressed Direction 1 S4 gate."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dm.eval.context import load_manifest
from dm.eval.provenance import environment, sha256_file
from dm.eval.reports import json_safe, staged_path

SCHEMA = 2
ARTIFACT_SCHEMAS = {
    "support": 3,
    "sampler": 3,
    "holm": 2,
    "context_report": 2,
    "context_manifest": 2,
}
DYNAMIC_LANGUAGE_OPCODES = frozenset(
    {"REPEAT", "ENDREP", "XFORM", "ENDX", "REPEATX"}
)
# Only used to explain a historical prose-only protocol. A protocol without a
# validated machine-readable ``gate.numeric`` block cannot authorize S4.
HISTORICAL_THRESHOLDS = {
    "baseline_valid_halt_max": 0.95,
    "dynamic_bits_min": 0.01,
    "decoder_delta_min": 0.05,
    "coverage_delta_min": -0.05,
    "nna_delta_min": -0.05,
    "length_emd_delta_max": 10.0,
}


def write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    staged = staged_path(path)
    staged.write_text(json.dumps(json_safe(value), indent=2, sort_keys=True,
                                 allow_nan=False))
    staged.replace(path)


def _resolved(path: str | Path) -> Path:
    return Path(path).resolve()


def _state_bearing(value: dict) -> bool:
    opcodes = set((value.get("policy") or {}).get("opcodes", []))
    return bool(opcodes & DYNAMIC_LANGUAGE_OPCODES)


def _holm_entries(holm: dict) -> dict[str, dict]:
    return {
        entry["name"]: entry
        for seed in holm.get("by_model_seed", {}).values()
        for entry in seed.get("families", {}).values()
    }


def _c3_contract_complete(gate_input: dict) -> bool:
    checks = gate_input.get("instrument_checks", {})
    return (
        gate_input.get("instrument_complete") is True
        and bool(checks)
        and all(value is True for value in checks.values())
    )


def _valid_numeric_gate(value: object) -> bool:
    if not isinstance(value, dict) or set(HISTORICAL_THRESHOLDS) - set(value):
        return False
    if any(
        isinstance(value[key], bool)
        or not isinstance(value[key], (int, float))
        or not math.isfinite(float(value[key]))
        for key in HISTORICAL_THRESHOLDS
    ):
        return False
    return (
        0.0 <= value["baseline_valid_halt_max"] <= 1.0
        and value["dynamic_bits_min"] >= 0.0
        and 0.0 <= value["decoder_delta_min"] <= 1.0
        and -1.0 <= value["coverage_delta_min"] <= 0.0
        and -1.0 <= value["nna_delta_min"] <= 0.0
        and value["length_emd_delta_max"] >= 0.0
    )


def _family_role_contract(protocol: dict, values: list[dict]) -> bool:
    entries = protocol.get("checkpoint_manifest", [])
    roles = protocol.get("family_roles")
    families = {entry.get("family") for entry in entries}
    if (
        not isinstance(roles, dict)
        or set(roles) != families
        or any(role not in {"state_bearing", "flat_control"}
               for role in roles.values())
    ):
        return False
    by_checkpoint = {
        _resolved(entry["checkpoint"]): entry["family"] for entry in entries
    }
    return bool(values) and all(
        (family := by_checkpoint.get(_resolved(value.get("checkpoint", ""))))
        is not None
        and _state_bearing(value) == (roles[family] == "state_bearing")
        for value in values
    )


def _sampler_contract(protocol: dict, reports: list[dict]) -> bool:
    sampler = protocol.get("sampler") or {}
    seed_roles = protocol.get("seed_roles") or {}
    levels = sampler.get("mask_levels")
    draw_seeds = seed_roles.get("draw_seeds")
    required = ("n", "draws", "top_k", "temperature", "forbid")
    if (
        any(key not in sampler for key in required)
        or not isinstance(levels, list)
        or set(levels) != {"raw", "vm_safe", "canonical"}
        or not isinstance(draw_seeds, list)
        or len(draw_seeds) != sampler.get("draws")
    ):
        return False
    expected_sampler = {
        "top_k": sampler["top_k"],
        "temperature": sampler["temperature"],
        "forbid_specials": sampler["forbid"] == [0, 1],
        "mask_mode": None,
        "policy_digest": None,
    }
    for report in reports:
        if (
            report.get("n") != sampler["n"]
            or report.get("seeds") != sampler["draws"]
            or report.get("draw_seeds") != draw_seeds
            or report.get("sampler") != expected_sampler
            or set(report.get("cells", {})) != set(levels)
            or (report.get("identity") or {}).get("draw_seeds") != draw_seeds
        ):
            return False
        policy_digest = report.get("policy_digest")
        for level in levels:
            cell_sampler = (report.get("cell_samplers") or {}).get(level, {})
            if cell_sampler != {
                **expected_sampler,
                "mask_mode": level,
                "policy_digest": policy_digest,
            }:
                return False
            draws = (report["cells"].get(level) or {}).get("draws", [])
            if len(draws) != sampler["draws"] or any(
                draw.get("n") != sampler["n"] for draw in draws
            ):
                return False
    return bool(reports)


def _support_contract(support: dict) -> bool:
    readings = support.get("readings", [])
    return bool(readings) and all(
        reading.get("limit") is None
        and reading.get("n") == (reading.get("corpus") or {}).get("n_val")
        and set(reading.get("levels", {})) == {"vm_safe", "canonical"}
        for reading in readings
    )


def manifest_check(protocol: dict, reports: list[dict], support: dict,
                   c3: dict) -> dict:
    """Verify that every gate input names the frozen content-addressed files."""
    entries = protocol.get("checkpoint_manifest", [])
    expected = {
        _resolved(entry["checkpoint"]): entry for entry in entries
    }

    def artifact_ok(checkpoint: str, record: str | None,
                    checkpoint_sha: str | None, record_sha: str | None) -> bool:
        entry = expected.get(_resolved(checkpoint))
        if entry is None:
            return False
        actual_record = record or str(Path(checkpoint).with_suffix(".json"))
        return (
            _resolved(actual_record) == _resolved(entry["record"])
            and Path(checkpoint).exists()
            and Path(actual_record).exists()
            and sha256_file(Path(checkpoint)) == entry["checkpoint_sha256"]
            and sha256_file(Path(actual_record)) == entry["record_sha256"]
            and checkpoint_sha == entry["checkpoint_sha256"]
            and record_sha == entry["record_sha256"]
        )

    s3_ok = all(
        artifact_ok(
            report.get("checkpoint", ""),
            (report.get("identity") or {}).get("record_path"),
            (report.get("identity") or {}).get("checkpoint_sha256"),
            (report.get("identity") or {}).get("record_sha256"),
        )
        for report in reports
    ) and {_resolved(report.get("checkpoint", "")) for report in reports} == set(expected)

    support_readings = support.get("readings", [])
    support_ok = all(
        artifact_ok(
            reading.get("checkpoint", ""),
            None,
            (reading.get("identity") or {}).get("checkpoint_sha256"),
            (reading.get("identity") or {}).get("record_sha256"),
        )
        for reading in support_readings
    ) and {_resolved(reading.get("checkpoint", "")) for reading in support_readings} == set(expected)

    context_manifest = protocol.get("context_c3", {})
    manifest_path = Path(c3.get("manifest", ""))
    c3_reports = c3.get("reports", [])
    try:
        loaded_manifest = load_manifest(manifest_path)
    except (OSError, ValueError, json.JSONDecodeError):
        loaded_manifest = None
    context_families = context_manifest.get("checkpoint_families")
    if context_families is None:
        available_families = {entry.get("family") for entry in entries}
        context_families = (
            ["synthetic_c2flat24000"]
            if "synthetic_c2flat24000" in available_families
            else list(available_families - {None})
        )
    expected_context = {
        _resolved(entry["checkpoint"])
        for entry in entries
        if entry.get("family") in context_families
    }
    if not expected_context and all(entry.get("family") is None for entry in entries):
        expected_context = set(expected)
    c3_checkpoints = {
        _resolved(report.get("checkpoint", "")) for report in c3_reports
    }
    c3_ok = (
        _resolved(manifest_path) == _resolved(context_manifest.get("manifest", ""))
        and loaded_manifest is not None
        and loaded_manifest.get("manifest_sha256") ==
            context_manifest.get("manifest_sha256")
        and c3_checkpoints == expected_context
        and all(
            artifact_ok(
                report.get("checkpoint", ""),
                None,
                report.get("checkpoint_sha256"),
                report.get("record_sha256"),
            )
            for report in c3_reports
        )
    )
    return {
        "s3": s3_ok,
        "support": support_ok,
        "c3": c3_ok,
        "all": s3_ok and support_ok and c3_ok,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--protocol", type=Path, default=Path("docs/state-protocol-v2.json"))
    ap.add_argument("--support", type=Path, required=True)
    ap.add_argument("--holm", type=Path, required=True)
    ap.add_argument("--c3", type=Path, required=True)
    ap.add_argument("reports", type=Path, nargs="+")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    if args.out.exists() and not args.overwrite:
        raise SystemExit(f"{args.out} already exists; pass --overwrite")
    protocol = json.loads(args.protocol.read_text())
    support = json.loads(args.support.read_text())
    holm = json.loads(args.holm.read_text())
    c3 = json.loads(args.c3.read_text())
    protocol_hash = sha256_file(args.protocol)

    reports = [json.loads(path.read_text()) for path in args.reports]
    manifest = manifest_check(protocol, reports, support, c3)
    numeric = (protocol.get("gate") or {}).get("numeric")
    protocol_criteria = (
        _valid_numeric_gate(numeric)
        and (protocol.get("gate") or {}).get("dynamic_metric")
        == "canonical_dynamic_bits_per_drawing"
    )
    thresholds = numeric if protocol_criteria else HISTORICAL_THRESHOLDS
    holm_entries = _holm_entries(holm)
    report_names = {report.get("name") for report in reports}
    holm_identity = (
        set(holm_entries) == report_names
        and all(
            holm_entries[report["name"]].get("identity")
            == (report.get("identity") or {}).get("identity")
            for report in reports
        )
    )
    c3_input = c3.get("gate_input", {})
    c3_contract_complete = _c3_contract_complete(c3_input)
    schema_contract = protocol.get("artifact_schemas") == ARTIFACT_SCHEMAS
    geometry_contract = (protocol.get("sampler") or {}).get(
        "empty_geometry_policy"
    ) in {"incomplete", "floor_failure"}
    family_role_contract = _family_role_contract(
        protocol, reports + list(support.get("readings", []))
    )
    sampler_contract = _sampler_contract(protocol, reports)
    support_contract = _support_contract(support)
    required_statuses = {
        "support": (
            support.get("status") == "complete"
            and support.get("report_schema") == ARTIFACT_SCHEMAS["support"]
            and bool(support.get("readings"))
            and all(
                reading.get("report_schema") == ARTIFACT_SCHEMAS["support"]
                for reading in support.get("readings", [])
            )
        ),
        "holm": (
            holm.get("status") == "complete"
            and holm.get("report_schema") == ARTIFACT_SCHEMAS["holm"]
            and holm_identity
        ),
        "c3": (
            c3.get("status") == "complete"
            and c3.get("report_schema") == ARTIFACT_SCHEMAS["context_report"]
            and c3_contract_complete
        ),
        "s3": bool(reports) and all(
            report.get("status") == "complete"
            and report.get("report_schema") == ARTIFACT_SCHEMAS["sampler"]
            for report in reports
        ),
        "checkpoint_manifest": manifest["all"],
        "artifact_schema_contract": schema_contract,
        "geometry_policy_contract": geometry_contract,
        "family_role_contract": family_role_contract,
        "sampler_contract": sampler_contract,
        "support_contract": support_contract,
        "machine_readable_gate": protocol_criteria,
        "protocol_identity": all(
            report.get("protocol_sha256") == protocol_hash for report in reports
        ) and all(
            reading.get("protocol_sha256") == protocol_hash
            for reading in support.get("readings", [])
        ) and bool(support.get("readings"))
        and holm.get("protocol_sha256") == protocol_hash
        and c3.get("protocol_sha256") == protocol_hash,
    }

    baseline_rows = [
        {"name": report["name"],
         "state_bearing": _state_bearing(report),
         "raw_valid_halt_mean": report["cells"]["raw"]["valid_halt_rate_mean"]}
        for report in reports
    ]
    baseline_gap = any(
        row["state_bearing"]
        and row["raw_valid_halt_mean"] <= thresholds["baseline_valid_halt_max"]
        for row in baseline_rows
    )

    dynamic_rows = []
    dynamic_mass = False
    for reading in support.get("readings", []):
        canonical = reading["levels"]["canonical"]
        vm_safe = reading["levels"]["vm_safe"]
        row = {
            "name": reading["name"],
            "state_bearing": _state_bearing(reading),
            "canonical_dynamic_bits": canonical["dynamic_bits_per_drawing"],
            "vm_safe_dynamic_bits": vm_safe["dynamic_bits_per_drawing"],
            "canonical_minus_vm_safe_dynamic_bits": (
                canonical["dynamic_bits_per_drawing"]
                - vm_safe["dynamic_bits_per_drawing"]
            ),
        }
        dynamic_rows.append(row)
        dynamic_mass |= (
            row["state_bearing"]
            and row["canonical_dynamic_bits"] >= thresholds["dynamic_bits_min"]
        )

    floor_failures = []
    decoder_candidates = []
    for report in reports:
        endpoint = report["paired_vs_raw"]["canonical"]["valid_halt_rate"]
        paired = report["paired_vs_raw"]["canonical"]
        floors_ok = True
        if report.get("completion", {}).get("geometry_missing"):
            floor_failures.append({"name": report["name"], "reason": "geometry_incomplete"})
            floors_ok = False
        coverage = paired.get("coverage")
        if coverage and coverage["delta"] < thresholds["coverage_delta_min"]:
            floor_failures.append({"name": report["name"], "reason": "coverage_floor"})
            floors_ok = False
        nna = paired.get("nna")
        if nna and nna["delta"] < thresholds["nna_delta_min"]:
            floor_failures.append({"name": report["name"], "reason": "nna_floor"})
            floors_ok = False
        length = paired.get("length_emd")
        if length and length["delta"] > thresholds["length_emd_delta_max"]:
            floor_failures.append({"name": report["name"], "reason": "length_emd_floor"})
            floors_ok = False
        row = {
            "name": report["name"],
            "state_bearing": _state_bearing(report),
            "delta": endpoint["delta"],
            "holm_reject": holm_entries.get(report["name"], {}).get(
                "holm", {}
            ).get("reject", False),
            "floors_ok": floors_ok,
        }
        decoder_candidates.append(row)
    decoder_effect = any(
        row["state_bearing"]
        and row["delta"] >= thresholds["decoder_delta_min"]
        and row["holm_reject"]
        and row["floors_ok"]
        for row in decoder_candidates
    )
    c3_positive = (
        required_statuses["c3"] and c3_input.get("c3_positive") is True
    )
    artifacts_complete = all(required_statuses.values())
    gate = {
        "baseline_gap": baseline_gap,
        "dynamic_mass": dynamic_mass,
        "decoder_effect": decoder_effect,
        "c3_positive": c3_positive,
        "artifacts_complete": artifacts_complete,
        "pass": artifacts_complete and baseline_gap and (dynamic_mass or decoder_effect)
               and c3_positive,
    }
    result = {
        "report_schema": SCHEMA,
        "status": "complete",
        "protocol": str(args.protocol),
        "protocol_sha256": protocol_hash,
        "provenance": environment(
            argv=sys.argv, device="analysis",
            source_paths=[Path(__file__), Path("dm/eval/provenance.py"),
                          Path("dm/eval/context.py"), args.protocol],
        ),
        "thresholds": thresholds,
        "threshold_source": (
            "protocol.gate.numeric"
            if protocol_criteria else
            "historical prose interpretation; not authorizing"
        ),
        "required_statuses": required_statuses,
        "checkpoint_manifest": manifest,
        "baseline": baseline_rows,
        "dynamic_mass": dynamic_rows,
        "decoder_candidates": decoder_candidates,
        "floor_failures": floor_failures,
        "c3": c3_input,
        "gate": gate,
        "s4_authorized": gate["pass"],
        "interpretation": (
            "S4 authorized by the content-addressed gate"
            if gate["pass"] else
            "S4 remains closed; at least one evidence term or instrument-contract check is false"
        ),
    }
    write(args.out, result)
    print(json.dumps(gate, sort_keys=True))
    print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
