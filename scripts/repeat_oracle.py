#!/usr/bin/env python3
"""How much `REPEAT` structure does a corpus actually contain?

Claim 2 scores a model by the L0 -> L1 compression ratio it recovers. That is
uninterpretable without the ceiling: an oracle allowed to see the whole program.
Run this on any candidate corpus **before** ingesting it, not after -- it is a
few seconds and it is what disqualified SVG-Icons8 (`docs/tier-c.md`).

Read `tol` as a diagnosis, not a result. Only `tol=0` is a compression number,
because `REPEAT` is lossless or it is nothing. A large jump from 0 to 1 means a
preprocessing stage rounded authored repeats apart, which is a bug in the
pipeline rather than a property of the data.

    python3 scripts/repeat_oracle.py --corpus tabler
    python3 scripts/repeat_oracle.py --corpus quickdraw --limit 2000
    python3 scripts/repeat_oracle.py --corpus synthetic --tier 1
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dm.data import quickdraw, synthetic, tabler
from dm.eval.repeats import corpus_stats
from dm.isa.spec import Tier

RUNS = Path("runs")


def build(args) -> list[bytes]:
    if args.corpus == "tabler":
        return tabler.load(args.icons, args.style, limit=args.limit)
    if args.corpus == "quickdraw":
        return quickdraw.load(tuple(args.categories), "valid", limit=args.limit,
                              rdp_eps=args.rdp_eps)
    if args.corpus == "synthetic":
        return synthetic.dataset(args.limit or 512, tier=Tier(args.tier))
    raise ValueError(args.corpus)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", required=True, choices=("tabler", "quickdraw", "synthetic"))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--tol", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--max-body", type=int, default=16)
    ap.add_argument("--icons", default="data/tabler/icons", help="tabler icons directory")
    ap.add_argument("--style", default="outline")
    ap.add_argument("--categories", nargs="+", default=["cat"])
    ap.add_argument("--rdp-eps", type=float, default=4.0)
    ap.add_argument("--tier", type=int, default=1, help="synthetic tier")
    args = ap.parse_args()

    programs = build(args)
    total = sum(len(p) for p in programs)
    print(f"{args.corpus}: {len(programs)} programs, {total:,} bytecode bytes, "
          f"{total / max(1, len(programs)):.0f} bytes each\n")
    print(f"{'tol':>4} {'programs w/ repeat':>19} {'bytes saved':>12} {'L0->L1 ratio':>13}")
    rows = []
    for tol in args.tol:
        stats = corpus_stats(programs, max_body=args.max_body, tol=tol)
        rows.append(stats)
        print(f"{tol:>4} {100 * stats['programs_with_repeat']:>18.1f}% "
              f"{100 * stats['saved_frac']:>11.2f}% {stats['ratio']:>13.4f}")

    if rows[0]["saved_frac"] < 0.01 <= max(r["saved_frac"] for r in rows):
        print("\n  WARNING: structure appears only above tol=0 -- a preprocessing stage")
        print("  is rounding authored repeats apart. Fix the pipeline, not the corpus.")

    RUNS.mkdir(exist_ok=True)
    out = RUNS / f"repeat_oracle_{args.corpus}.json"
    out.write_text(json.dumps({"args": vars(args), "rows": rows}, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
