"""Build the scientific source-exclusion evidence without model calls."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from dm.relation.demonstrated_exclusion import audit_scientific_sources


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists; preserve previous evidence")
    result = audit_scientific_sources(args.manifest)
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
    data = json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        stream.write(data)
    print(json.dumps({"status": result["status"], "sha256": hashlib.sha256(data.encode()).hexdigest(),
                      "overlaps": len(result["overlaps"]), "model_calls": 0}))
    if result["status"] != "scientific_source_exclusion_pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
