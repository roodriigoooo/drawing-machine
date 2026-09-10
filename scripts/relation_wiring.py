"""Prepare/validate fixed R5 engineering scope; execute only after separate approval."""
from __future__ import annotations

import argparse
from pathlib import Path

from dm.relation.wiring import digest, execute, prepare, restart_worker, validate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("prepare", help="reserve exclusive directory and freeze run card; no training")
    p.add_argument("run_id")
    p = commands.add_parser("validate", help="validate prepared recipe, fixture, sources and runtime; no training")
    p.add_argument("directory", type=Path)
    p = commands.add_parser("execute", help="launch only after independent run-card approval")
    p.add_argument("directory", type=Path)
    p.add_argument("--approve-card-sha256", required=True)
    p = commands.add_parser("restart-worker", help=argparse.SUPPRESS)
    p.add_argument("directory", type=Path)
    p.add_argument("--arm", required=True, choices=("none", "span_affine_v1"))
    args = parser.parse_args()
    if args.command == "prepare":
        path, identity = prepare(args.run_id)
        print(f"prepared_not_launched {path}\ncard_sha256 {identity}")
    elif args.command == "validate":
        print(f"valid_not_launched card_sha256 {digest(validate(args.directory))}")
    elif args.command == "execute":
        execute(args.directory, args.approve_card_sha256)
    else:
        restart_worker(args.directory, args.arm)


if __name__ == "__main__":
    main()
