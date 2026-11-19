"""Build or verify the model-blind candidate fit audit, never launch training."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from dm.relation.demonstrated_fixture import audit_fit_fixture, build_fit_fixture


def encoded(value):
    return json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("build", "verify"))
    parser.add_argument("path", type=Path)
    args = parser.parse_args()
    report = audit_fit_fixture(build_fit_fixture())
    root = Path(__file__).resolve().parents[1]
    # Bind every local Python source imported by this audit, including shared
    # parsers, VM and training planner. Import torch does not construct a model.
    import sys
    paths = {Path(__file__).resolve()}
    for module in tuple(sys.modules.values()):
        source = getattr(module, "__file__", None)
        if source:
            path = Path(source).resolve()
            if path.suffix == ".py" and path.is_relative_to(root / "dm"):
                paths.add(path)
    report["source_sha256"] = {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(paths)
    }
    data = encoded(report)
    if args.mode == "build":
        args.path.parent.mkdir(parents=True, exist_ok=True)
        with args.path.open("x") as stream:
            stream.write(data)
    elif args.path.read_text() != data:
        parser.error("stored candidate audit differs from source-bound deterministic rebuild")
    print(json.dumps({"status": report["status"], "mode": args.mode,
                      "sha256": hashlib.sha256(data.encode()).hexdigest(),
                      "model_calls": 0, "optimizer_updates": 0}))


if __name__ == "__main__":
    main()
