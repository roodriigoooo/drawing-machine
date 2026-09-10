#!/usr/bin/env python3
"""Does canonicalising a drawing's frame buy any reuse? Four ceilings per policy.

`docs/direction.md` §7.1 proposes aligning every sketch to a standard frame and
proposes one test for it: canonicalise, then ask whether the corpus's **repeat
ceiling** went up. `dm/data/canonical.py` shows that test cannot fire in the
direction it hopes -- an exact re-framing carries the set of foldable repeats
across bijectively, so both within-program ceilings are invariant *exactly*, and
an inexact one is a non-integer map followed by rounding, which is the operation
measured to destroy 77% of Tabler's ceiling.

So this script reports **four** ceilings per policy, not one:

    REPEAT    within-program, translation only        (dm/eval/repeats.py)
    REPEATX   within-program, D4 ⋉ translation        (dm/eval/repeats.py)
    CALL/t    between-program, translation-normalised (dm/eval/library.py)
    CALL/D4   between-program, D4-normalised          (dm/eval/library.py)

Two of the four are *predicted nulls* for the exact policy and firing them is
the instrument check: `d4` must leave `REPEAT`, `REPEATX` and `CALL/D4`
byte-identical (the library oracle has already quotiented by the same group) and
may move `CALL/t` alone. If any of the three moves, the canonicalisation is not
in the group it claims to be in and no other number here is worth reading.

The inexact policies are run through `dm/data/refit.py`, so they are compared
against a `none` row that went through the identical re-spelling: otherwise the
frame would be charged for the fitter. **Their gain is never quoted without
their loss** -- a rotated corpus that shares more strokes and folds fewer
repeats has been paid for, and §7.1's third branch says so in advance.

    python3 scripts/canonical.py --corpus tabler
    python3 scripts/canonical.py --corpus quickdraw --limit 500
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dm.data import quickdraw, tabler
from dm.data.canonical import POLICIES, d4_representative
from dm.data.fingerprint import digest
from dm.data.refit import respell_corpus
from dm.eval.library import library_stats
from dm.eval.repeats import corpus_stats, symmetry_stats

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"

CATEGORIES = ("cat", "dog", "bus", "car", "tree")

#: Nulls that must fire exactly for `d4`. Named here rather than checked by eye,
#: because a "provable null" nobody re-derives is a comment.
EXACT_NULLS = ("repeat", "repeatx", "call_d4")


def build(args) -> list[bytes]:
    if args.corpus == "tabler":
        return tabler.split(args.icons, args.style)[0][: args.limit]
    return quickdraw.load(tuple(args.categories), "valid", limit=args.limit,
                          rdp_eps=args.rdp_eps)


def ceilings(programs: list[bytes], max_body: int, grid: int) -> dict:
    """The four ceilings, plus the corpus's own identity."""
    flat = corpus_stats(programs, max_body=max_body)
    orbit = symmetry_stats(programs, max_body=max_body)
    call_t = library_stats(programs, grid=grid)
    call_d4 = library_stats(programs, grid=grid, d4=True)
    total = max(1, orbit["bytes"])
    return {
        "digest": digest(programs),
        "n": len(programs),
        "bytes_per_drawing": orbit["bytes"] / max(1, len(programs)),
        "repeat": flat["saved"] / total,
        "repeatx": orbit["fraction"],
        "call_t": call_t["fraction"],
        "call_d4": call_d4["fraction"],
        "call_t_library": call_t["library"],
        "call_d4_library": call_d4["library"],
        "distinct_strokes": call_t["distinct"],
        "strokes": call_t["strokes"],
        "grid": grid,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", default="tabler", choices=("tabler", "quickdraw"))
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--max-body", type=int, default=64)
    ap.add_argument("--tol", type=float, default=2.0,
                    help="re-spelling tolerance, canvas px, for the frame policies")
    ap.add_argument("--grid", type=int, default=1,
                    help="library match grid; only 1 is a compression number")
    ap.add_argument("--icons", default="data/tabler/icons")
    ap.add_argument("--style", default="outline")
    ap.add_argument("--categories", nargs="+", default=list(CATEGORIES))
    ap.add_argument("--rdp-eps", type=float, default=4.0)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    programs = build(args)
    rows: list[dict] = []

    # --- exact half: the program-level policies, no re-spelling anywhere.
    base = ceilings(programs, args.max_body, args.grid)
    rows.append({"policy": "none", "level": "program", "exact": True, **base})
    canon = [d4_representative(p)[0] for p in programs]
    row = ceilings(canon, args.max_body, args.grid)
    rows.append({"policy": "d4", "level": "program", "exact": True,
                 "changed": sum(a != b for a, b in zip(programs, canon)), **row})

    broken = [k for k in EXACT_NULLS if abs(row[k] - base[k]) > 1e-12]

    # --- inexact half: every frame policy through the identical re-spelling, so
    #     the comparison is the frame and not the fitter.
    for name in ("none", "rot", "aspect", "rot_aspect"):
        policy = POLICIES[name]
        respelled, stats = respell_corpus(programs, args.tol, True, policy)
        rows.append({
            "policy": name, "level": "frame", "exact": policy.exact,
            "tol": args.tol, "error_mean": stats["error_mean"],
            "error_max": stats["error_max"], "refused": stats["refused"],
            **ceilings(respelled, args.max_body, args.grid),
        })

    print(f"{args.corpus}: {len(programs)} programs, tol {args.tol} px, "
          f"library grid {args.grid}\n")
    print(f"{'policy':<12}{'level':<9}{'exact':>6}{'B/draw':>9}"
          f"{'REPEAT':>9}{'REPEATX':>9}{'CALL/t':>9}{'CALL/D4':>9}")
    for r in rows:
        print(f"{r['policy']:<12}{r['level']:<9}{r['exact']!s:>6}"
              f"{r['bytes_per_drawing']:>9.1f}{r['repeat']:>8.2%}{r['repeatx']:>9.2%}"
              f"{r['call_t']:>9.2%}{r['call_d4']:>9.2%}")

    print("\nPredicted nulls for `d4` (exact policy, within-program oracles and the"
          "\nD4-normalised library): " + ("all fired exactly" if not broken
                                          else f"BROKEN on {broken}"))
    if broken:
        print("  A policy that moves one of these is not in D4 ⋉ integer translation."
              "\n  Nothing else in this table is readable until that is explained.")

    frame = [r for r in rows if r["level"] == "frame"]
    if frame:
        ref = frame[0]
        print("\nFrame policies, against the same corpus re-spelled with no frame "
              "change.\nGain and loss are printed together because an inexact policy "
              "buys the first\nwith the second (`docs/direction.md` §7.1, third "
              "branch):\n")
        print(f"{'policy':<12}{'Δ CALL/t':>10}{'Δ REPEAT':>10}{'Δ REPEATX':>11}"
              f"{'err mean':>10}{'B/draw':>9}")
        for r in frame[1:]:
            print(f"{r['policy']:<12}{r['call_t'] - ref['call_t']:>+10.2%}"
                  f"{r['repeat'] - ref['repeat']:>+10.2%}"
                  f"{r['repeatx'] - ref['repeatx']:>+11.2%}"
                  f"{r['error_mean']:>10.2f}{r['bytes_per_drawing']:>9.1f}")

    out = args.out or RUNS / f"canonical_{args.corpus}_n{len(programs)}_g{args.grid}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"args": {k: str(v) for k, v in vars(args).items()},
                               "exact_nulls_fired": not broken,
                               "broken_nulls": broken, "rows": rows}, indent=2))
    print(f"\n  -> {out}")
    return 1 if broken else 0


if __name__ == "__main__":
    raise SystemExit(main())
