"""Prepare, validate or execute the frozen bounded demonstrated-affine package."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from dm.relation.demonstrated_run import execute, make_card, validate_card, write
from dm.relation.wiring import digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("directory", type=Path)
    prepare.add_argument("--audits", type=Path, required=True)
    prepare.add_argument("--recover-accounting-failure", action="store_true")
    validate = sub.add_parser("validate")
    validate.add_argument("directory", type=Path)
    run = sub.add_parser("execute")
    run.add_argument("directory", type=Path)
    run.add_argument("--card-sha256", required=True)
    run.add_argument("--authorize-recovery-replay", action="store_true",
                     help="Allow the 768 explicitly recorded repair replays in addition to primary work")
    args = parser.parse_args()
    if args.command == "execute":
        execute(args.directory, args.card_sha256,
                authorize_recovery_replay=args.authorize_recovery_replay)
        return
    if args.command == "prepare":
        card = make_card(args.audits, recover=args.recover_accounting_failure)
        args.directory.mkdir(parents=True, exist_ok=False)
        write(args.directory / "run-card.json", card)
    else:
        card = json.loads((args.directory / "run-card.json").read_text())
        validate_card(card)
    print(json.dumps({"status": "frozen_pretraining" if args.command == "prepare" else "validated",
                      "card_sha256": digest(card)}))


if __name__ == "__main__":
    main()
