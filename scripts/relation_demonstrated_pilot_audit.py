"""Persist or verify the fixed I0 pilot candidate, including task-invalid witnesses."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from dm.relation.demonstrated_pilot import PILOT_SEED, audit_pilot, build_pilot_fixture


def report():
    fixture = build_pilot_fixture()
    result = audit_pilot(fixture)
    result.update({
        "schema": "demonstrated-affine-pilot-i0-v1",
        "build_seed": PILOT_SEED,
        "construction": "preallocated orbit-distinct sources; adjacent donor swaps; no redraw",
        "sources_hex": [s.hex() for s in fixture.sources],
        "panels": {name: [{
            "case_id": c.case_id, "class": c.stratum, "flat_hex": c.flat.hex(),
            "target_start": c.target_start, "target_stop": c.target_stop,
            "source_ids": c.source_ids, "donor_source_ids": c.donor_source_ids,
            "source_case_id": c.source_case_id, "donor_case_id": c.donor_case_id,
            "plan": c.plan, "actions": [a.key() for a in c.actions],
        } for c in cases] for name, cases in fixture.panels},
        "training_authorized_by_this_artifact": False,
        "not_completed": ["scientific source exclusion", "frozen run card", "training route"],
    })
    root = Path(__file__).resolve().parents[1]
    paths = {Path(__file__).resolve()}
    for module in tuple(sys.modules.values()):
        source = getattr(module, "__file__", None)
        if source:
            path = Path(source).resolve()
            if path.suffix == ".py" and path.is_relative_to(root / "dm"):
                paths.add(path)
    result["source_sha256"] = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in sorted(paths)}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("build", "verify"))
    parser.add_argument("path", type=Path)
    args = parser.parse_args()
    result = report()
    data = json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n"
    if args.mode == "build":
        args.path.parent.mkdir(parents=True, exist_ok=True)
        with args.path.open("x") as stream:
            stream.write(data)
    elif args.path.read_text() != data:
        parser.error("stored pilot audit differs from source-bound deterministic rebuild")
    print(json.dumps({"status": result["status"], "mode": args.mode,
                      "sha256": hashlib.sha256(data.encode()).hexdigest(),
                      "passed_rows": result["passed_rows"],
                      "witnesses": len(result["rejection_ledger"]),
                      "optimizer_updates": 0}))


if __name__ == "__main__":
    main()
