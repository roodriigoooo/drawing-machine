#!/usr/bin/env python3
"""How many bits the stroke decoder spends on what its summary already pinned.

`docs/direction.md` §3, and the first item on that file's list because **it
decides what claim 3's negative result means and needs no training**. The
planner transmits the summary twice -- once explicitly as `p(s)`, once
implicitly inside `p(x | s)`, since `s = f(x)` is deterministic -- and if the
stroke decoder is not exploiting what it was told, a large part of the 39-49 bit
loss is a coding inefficiency in this implementation rather than a verdict on
factorisation.

Two readings of one forward pass, both from `dm/eval/redundancy.py`:

- **the wasted-mass reading**, `-log2 P(feasible)` summed over positions: the
  bits recoverable by renormalising the decoder onto the values the summary
  leaves possible. No retraining, so it is an *achievable* saving with a ceiling
  known by construction.
- **the partition reading**, §3.2 as specified: bits/symbol where the summary
  leaves exactly one legal value against bits/symbol everywhere else.

**Pass the flat AR arm too.** It was never told the summary, so its number is
what the planner's has to beat; without it "the decoder wastes 12 bits" has no
denominator. The asymmetry runs the conservative way -- the flat arm sees every
previous stroke and the planner's decoder sees only its own -- so a planner that
does not waste less has failed the test with an advantage.

    python3 scripts/redundancy.py \
        runs/quickdraw_plannerar12000eps2_byte_balanced_s0.pt \
        runs/quickdraw_plannerdiff12000eps2_byte_balanced_s0.pt \
        runs/quickdraw_planbase12000eps2_byte_square_s0.pt

CPU by default, no training, minutes.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dm.eval.records import corpus_config, is_planner, load
from dm.eval.redundancy import (
    SUMMARY_RULES,
    flat_redundancy,
    planner_redundancy,
)
from dm.isa.codec import CODECS
from dm.train import RUNS, build_data

#: Printed in this order. `summary` is the headline and the per-rule rows are
#: what it decomposes into -- they do not sum to it, because the rules overlap
#: on the positions they constrain, and a report that implied they did would be
#: inviting a subtraction that is not defined.
ROWS = (
    ("summary_bits_per_drawing", "summary, total", "bits/drawing recoverable"),
    *((f"{rule}_bits_per_drawing", f"  {rule}", "its own marginal, over ISA")
      for rule in SUMMARY_RULES),
    ("isa_bits_per_drawing", "ISA alone", "the baseline, not the summary's"),
    ("determined_bits_per_symbol", "bits/symbol, pinned", "one legal value"),
    ("free_bits_per_symbol", "bits/symbol, free", "everything else"),
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoints", type=Path, nargs="+")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--limit", type=int, default=None,
                    help="score only the first N val programs; for a smoke test, "
                         "never for a reading -- it changes the denominator")
    args = ap.parse_args()

    for path in args.checkpoints:
        model, record = load(path)
        codec = CODECS[record["config"]["codec"]]
        _, programs = build_data(corpus_config(record))
        if args.limit:
            programs = programs[: args.limit]
        model.to(args.device)

        if is_planner(record):
            report = planner_redundancy(model, programs, codec,
                                        args.device, args.batch_size)
        else:
            # The flat arm's own `max_strokes` does not exist, so the split is
            # unbounded: it is being scored on the whole program either way, and
            # the strokes are only a way of indexing the summary's constraints.
            report = flat_redundancy(model, programs, codec, None,
                                     record["config"]["max_len"],
                                     args.device, args.batch_size)
        report |= {"name": record["name"], "kind": record.get("kind", "ar"),
                   "codec": codec.name, "steps": record.get("steps"),
                   "params": record["model"]["params"]}

        print(f"\n## {record['name']}  ({codec.name}, "
              f"{record['model']['params']:,} params, "
              f"{'planner' if is_planner(record) else 'flat AR'})")
        print(f"{report['n']} val programs, {report['strokes']} strokes\n")
        for key, label, note in ROWS:
            print(f"  {label:<22}{report[key]:9.3f}   {note}")
        print(f"\n  {report['determined_symbols']:,} of "
              f"{report['determined_symbols'] + report['free_symbols']:,} scored "
              f"symbols have exactly one legal value "
              f"({report['determined_symbols'] / max(1, report['determined_symbols'] + report['free_symbols']):.1%}).")
        if report["length_mismatched_strokes"]:
            print(f"  {report['length_mismatched_strokes']} stroke(s) longer than the "
                  "summary's u8 length field: the length rules are off for those.")
        out = RUNS / f"{record['name']}_redundancy.json"
        out.write_text(json.dumps(report, indent=2))
        print(f"\n  -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
