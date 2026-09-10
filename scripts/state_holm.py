#!/usr/bin/env python3
"""Apply a protocol's Holm family to completed S3 reports."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dm.eval.provenance import environment, sha256_file
from dm.eval.reports import json_safe, staged_path
from dm.eval.state_analysis import holm

SCHEMA = 2


def write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    staged = staged_path(path)
    staged.write_text(json.dumps(json_safe(value), indent=2, sort_keys=True,
                                 allow_nan=False))
    staged.replace(path)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("reports", type=Path, nargs="+")
    ap.add_argument("--protocol", type=Path,
                    default=Path("docs/state-protocol-v2.json"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--alpha", type=float, default=None,
                    help="must equal inference.alpha in the protocol; v2 "
                         "historically implied 0.05")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    if args.out.exists() and not args.overwrite:
        raise SystemExit(f"{args.out} already exists; pass --overwrite")

    protocol = json.loads(args.protocol.read_text())
    protocol_digest = sha256_file(args.protocol)
    entries = protocol.get("checkpoint_manifest", [])
    if not entries:
        raise SystemExit("protocol has no checkpoint_manifest")
    by_checkpoint = {Path(entry["checkpoint"]).resolve(): entry for entry in entries}
    families = list(dict.fromkeys(entry["family"] for entry in entries))
    model_seeds = sorted({int(entry["model_seed"]) for entry in entries})
    protocol_alpha = (protocol.get("inference") or {}).get("alpha")
    if protocol_alpha is None and protocol.get("protocol_id") == "direction1-state-v2":
        protocol_alpha = 0.05
    if (isinstance(protocol_alpha, bool)
            or not isinstance(protocol_alpha, (int, float))
            or not math.isfinite(float(protocol_alpha))):
        raise SystemExit("protocol inference.alpha must be finite and numeric")
    alpha = float(protocol_alpha)
    if not 0.0 < alpha < 1.0:
        raise SystemExit("protocol inference.alpha must lie strictly between 0 and 1")
    if args.alpha is not None and args.alpha != alpha:
        raise SystemExit(
            f"--alpha={args.alpha} differs from protocol inference.alpha={alpha}"
        )
    source_paths = [Path(__file__), Path("dm/eval/state_analysis.py"),
                    Path("dm/eval/provenance.py"), args.protocol]
    rows = []
    for path in args.reports:
        report = json.loads(path.read_text())
        if report.get("status") not in ("complete", "incomplete"):
            raise SystemExit(f"S3 report is not finished: {path}")
        if report.get("report_schema") != 3:
            raise SystemExit(f"S3 report schema is not 3: {path}")
        completion = report.get("completion", {})
        if completion.get("completed_draws") != completion.get("requested_draws"):
            raise SystemExit(f"S3 report has partial draws: {path}")
        checkpoint = Path(report.get("checkpoint", "")).resolve()
        entry = by_checkpoint.get(checkpoint)
        if entry is None:
            raise SystemExit(f"S3 checkpoint is absent from protocol: {path}")
        identity = report.get("identity") or {}
        if (
            identity.get("checkpoint_sha256") != entry["checkpoint_sha256"]
            or identity.get("record_sha256") != entry["record_sha256"]
            or report.get("protocol_sha256") != protocol_digest
        ):
            raise SystemExit(f"S3 identity differs from protocol: {path}")
        name = report["name"]
        family = entry["family"]
        model_seed = int(entry["model_seed"])
        endpoint = report["paired_vs_raw"]["canonical"]["valid_halt_rate"]
        rows.append({
            "family": family,
            "model_seed": model_seed,
            "name": name,
            "report": str(path),
            "mean_delta": endpoint["delta"],
            "ci95": endpoint["ci95"],
            "p_value": endpoint["p_value"],
            "p_exact_sign_flip": endpoint["p_exact_sign_flip"],
            "identity": report["identity"]["identity"],
            "status": report["status"],
            "geometry_complete": not bool(report["completion"]["geometry_missing"]),
        })
    expected = {
        (entry["family"], int(entry["model_seed"])) for entry in entries
    }
    if len(expected) != len(entries):
        raise SystemExit("protocol repeats a family/model-seed checkpoint cell")
    if {(row["family"], row["model_seed"]) for row in rows} != expected:
        raise SystemExit("S3 reports do not exactly match the protocol family/seed set")

    by_seed = {}
    for seed in model_seeds:
        selected = {row["family"]: row for row in rows
                    if row["model_seed"] == seed}
        if set(selected) != set(families):
            raise SystemExit(
                f"model seed {seed} does not contain every protocol family"
            )
        corrections = holm({family: row["p_value"]
                            for family, row in selected.items()}, alpha)
        by_seed[str(seed)] = {
            "alpha": alpha,
            "families": {
                family: {**selected[family], "holm": corrections[family]}
                for family in families
            },
        }
    write(args.out, {
        "report_schema": SCHEMA,
        "status": "complete",
        "protocol": str(args.protocol),
        "protocol_sha256": protocol_digest,
        "provenance": environment(argv=sys.argv, device="analysis",
                                    source_paths=source_paths),
        "method": "paired Student-t p-values with t95 intervals, followed by Holm step-down; exact sign-flip p-values retained as a robustness diagnostic",
        "families": families,
        "model_seeds": model_seeds,
        "by_model_seed": by_seed,
    })
    print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
