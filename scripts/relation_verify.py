"""Model-blind R4 corpus rebuild and exhaustive candidate extraction proof."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from dm.data.relation import ManifestRefused
from dm.eval.relation_r4_verification import verify_frozen_corpus


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("runs/relation_corpus_v1.json"))
    parser.add_argument("--output", type=Path, help="optional engineering JSON; refuses overwrite")
    args = parser.parse_args()
    if args.output is not None and args.output.exists():
        parser.error(f"output already exists: {args.output}")
    try:
        report = verify_frozen_corpus(args.manifest, progress=lambda text: print(text, flush=True))
    except ManifestRefused as exc:
        parser.error(str(exc))
    encoded = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as output:
            output.write(encoded)
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
