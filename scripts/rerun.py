#!/usr/bin/env python3
"""Re-run one cell with the exact config of a record that already exists.

A budget ladder is only a ladder if every rung differs in the one variable it
is measuring. When a cell has to be re-run -- because a process wedged, or a
seed is missing -- retyping its flags is how a rung silently acquires a
different `lr` or `attn_budget` and stops being comparable. So the config is
copied from a sibling record on disk and only the named fields are overridden:

    # the bit rung, matching the byte rung it will be differenced against
    python3 scripts/rerun.py --like synthetic_budget24000_byte_square_s0 \
        --set codec=bit --tag synthetic_budget24000_bit_square_s0

    python3 scripts/rerun.py --like ... --set seed=1 --tag ... --dry-run

Every override is echoed, and the run refuses to overwrite an existing record
unless `--force` is given, for the same reason `resume_verdict` refuses.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dm.train import RUNS, TrainConfig, train

#: Fields worth coercing from the string an override arrives as.
_CASTS = {int: int, float: float, bool: lambda s: s.lower() in {"1", "true", "yes"}}


def coerce(field_type: type, raw: str):
    if field_type in _CASTS:
        return _CASTS[field_type](raw)
    return raw


def apply_overrides(config: dict, overrides: list[str]) -> dict:
    """`key=value` pairs, typed against `TrainConfig`'s annotations."""
    hints = {f.name: f.type for f in TrainConfig.__dataclass_fields__.values()}
    out = dict(config)
    for item in overrides:
        if "=" not in item:
            raise SystemExit(f"--set expects key=value, got {item!r}")
        key, raw = item.split("=", 1)
        if key not in out:
            raise SystemExit(f"{key!r} is not a field of the recorded config")
        hint = hints.get(key)
        target = type(out[key]) if out[key] is not None else (hint if isinstance(hint, type) else str)
        out[key] = coerce(target, raw)
        print(f"  override {key}: {config[key]!r} -> {out[key]!r}", flush=True)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--like", required=True, help="tag of the record to copy the config from")
    ap.add_argument("--set", dest="overrides", nargs="*", default=[], metavar="KEY=VALUE")
    ap.add_argument("--tag", help="tag for the new record (defaults to the override-implied one)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true", help="overwrite an existing record")
    args = ap.parse_args()

    source = RUNS / f"{args.like}.json"
    if not source.exists():
        raise SystemExit(f"no record at {source}")
    config = json.loads(source.read_text())["config"]
    print(f"config copied from {args.like}", flush=True)

    config = apply_overrides(config, args.overrides)
    if args.tag:
        config["tag"] = args.tag
    if config["tag"] == args.like and not args.force:
        raise SystemExit("that would overwrite the source record; pass --tag or --force")

    target = RUNS / f"{config['tag']}.json"
    if target.exists() and not args.force:
        raise SystemExit(f"{target.name} already exists; move it aside or pass --force")

    cfg = TrainConfig(**config)
    print(f"-> {cfg.tag}: codec={cfg.codec} shape={cfg.shape} steps={cfg.steps} "
          f"seed={cfg.seed} lr={cfg.lr} attn_budget={cfg.attn_budget}", flush=True)
    if args.dry_run:
        return 0
    train(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
