"""R0 Adapter: qualify a measurement platform before any Direction 4 cell runs.

Thin by contract. `dm.eval.relation_evidence` owns the digests, the sentinel and
the rule; this file chooses a device, runs the two readings and writes the report.

    # qualify this machine's CPU
    PYTHONPATH=. .venv/bin/python scripts/relation_qualify.py \\
        --device cpu --out runs/relation_r0_cpu.json

    # the backend Direction 3's cells actually ran on
    PYTHONPATH=. .venv/bin/python scripts/relation_qualify.py \\
        --device mps --out runs/relation_r0_mps.json

**Exit code is the decision, and there is one route.** A platform that cannot
reproduce model, optimizer and RNG state exactly across `R0_MIN_REPEATS` repeats
fails, and no scientific cell may launch on it. The metric-level fallback
`docs/copy-relation.md` §6 clause 3 allowed is withdrawn in v1: the version that
existed was degenerate, and a replacement needs a paired measurement on the
relation task that cannot exist before R3.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from dm.eval import relation_contract as contract
from dm.eval import relation_evidence as evidence
from dm.eval.reports import staged_path


def _write(path: Path, body: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    staged = staged_path(path)
    staged.write_text(json.dumps(body, indent=1, sort_keys=True) + "\n")
    staged.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--repeats", type=int, default=contract.R0_MIN_REPEATS)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    contract.check_consistency()
    repeatability = evidence.repeatability(device=args.device, repeats=args.repeats)
    resume = evidence.resume_equivalence(device=args.device)
    decision = evidence.qualify(repeatability, resume)

    body = {
        "schema": evidence.EVIDENCE_SCHEMA,
        "direction": contract.DIRECTION,
        "stage": "R0",
        "protocol": contract.PROTOCOL,
        "provenance": "engineering",
        "decision_value": "platform qualification only",
        "repeatability": repeatability,
        "resume": resume,
        "decision": decision,
    }
    if args.out is not None:
        _write(args.out, body)
        print(f"report     {args.out}")

    print(f"device     {args.device}  ({repeatability['mean_seconds']:.2f}s/repeat "
          f"x {args.repeats})")
    print(f"digests    {repeatability['distinct_digests']}")
    print(f"exact      {repeatability['exact_state_reproduction']}")
    print(f"loss range {repeatability['observed_loss_range']:.6g}  (diagnostic; "
          f"it gates nothing)")
    print(f"resume     equivalent={resume['equivalent']} "
          f"differing={resume['differing_digests']}")
    print(f"route      {decision['route']}  (fallback: withdrawn in v1)")
    if not decision["qualified"]:
        for problem in decision["problems"]:
            print(f"R0 FAILED: {problem}", file=sys.stderr)
        return 1
    print("qualified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
