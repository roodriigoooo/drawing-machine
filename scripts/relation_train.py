"""R4 relation-training Adapter.

R4 intentionally exposes validation only.  Launching a model-bearing training
run remains an R5/R6 authorization boundary, so this script cannot create a
checkpoint or enter an optimizer loop.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from dm.train_relation import RelationTrainingRefused, load_training_corpus


def main() -> int:
    parser = argparse.ArgumentParser(description="validate the frozen R4 training corpus")
    parser.add_argument("--manifest", type=Path, default=Path("runs/relation_corpus_v1.json"))
    parser.add_argument("--provenance", choices=("scientific", "development"),
                        default="scientific")
    parser.add_argument("--validate", action="store_true",
                        help="rebuild and validate only; no model is constructed")
    args = parser.parse_args()
    if not args.validate:
        parser.error("R4 permits validation only; optimizer runs require later authorization")
    try:
        training = load_training_corpus(args.manifest, provenance=args.provenance)
    except RelationTrainingRefused as exc:
        parser.error(str(exc))
    print(f"validated {len(training.cases)} frozen train cases")
    print(f"payload {training.canonical_payload_sha256}")
    print(f"file    {training.file_sha256}")
    print(f"program {training.program_fingerprint}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
