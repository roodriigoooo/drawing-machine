#!/usr/bin/env python3
"""Halting-hazard report for trained checkpoints.

Answers one question: does an arm generate short because it over-assigns HALT
at instruction boundaries, or because of how it is decoded? The first is
teacher-forced and this measures it; whatever is left over is the sampler.

    python3 -m scripts.termination runs/synthetic_converged_*_square_s0.pt
    python3 scripts/termination.py runs/synthetic_converged_byte_square_s0.pt

Runs on CPU by default: it is one forward pass over 1,000 short programs, and
keeping it off the accelerator means it can be run while a sweep is training.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dm.data import synthetic
from dm.eval.termination import StrideUnsupported, halt_hazard, summarise
from dm.isa.codec import CODECS
from dm.isa.spec import Tier
from dm.models.transformer import Config, DrawingLM


def load(path: Path) -> tuple[DrawingLM, dict]:
    """Checkpoint plus the run record beside it, which carries the split."""
    record = json.loads(path.with_suffix(".json").read_text())
    blob = torch.load(path, map_location="cpu", weights_only=True)
    model = DrawingLM(Config(**blob["cfg"]))
    model.load_state_dict(blob["state"])
    return model, record


def val_split(config: dict) -> list[bytes]:
    """The *same* val programs the run was scored on.

    Rebuilt from the recorded config rather than regenerated with defaults:
    `split()` filters val against train, so a different `n_train` gives a
    different val set, and two different val sets differ by ~3.5 bits/drawing.
    """
    _, val = synthetic.split(
        config["n_train"], config["n_val"], seed=config["data_seed"],
        tier=Tier(config["tier"]),
    )
    return val


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoints", type=Path, nargs="+")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--rows", type=int, default=12, help="boundaries to print")
    args = ap.parse_args()

    for path in args.checkpoints:
        model, record = load(path)
        codec = CODECS[record["config"]["codec"]]
        print(f"\n## {record['name']}  ({codec.name}, {record['model']['params']:,} params)")
        try:
            report = halt_hazard(
                model.to(args.device), codec, val_split(record["config"]),
                device=args.device, max_len=record["config"]["max_len"],
            )
        except StrideUnsupported as exc:
            print(f"skipped: {exc}")
            continue
        final = record.get("final", {})
        print(
            f"reported: bits/drawing {final.get('bits_per_drawing', float('nan')):.1f}  "
            f"len EMD {final.get('gen_length_emd', float('nan')):.2f}  "
            f"generated p50 {final.get('gen_len_p50', float('nan')):.0f} bytes"
        )
        print(summarise(report, rows=args.rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
