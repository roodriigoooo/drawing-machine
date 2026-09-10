#!/usr/bin/env python3
"""What each corpus offers `REPEAT`, and what it offers `REPEATX`.

`scripts/repeat_oracle.py` answers "how much *translational* structure is
there?" for one corpus. This answers the ISA-v2 question across corpora at once:
**how much of what a corpus contains can only be folded once the transform tier
exists**, and — the part that decides whether a constructed ceiling can be
believed — **how much transformed structure a natural corpus already had.**

The two oracles are run over the identical programs, so the difference between
their columns is the tier and nothing else. The output is a record rather than a
printout, because `scripts/plots.py` draws figure 6 from it and a figure that
disagrees with a table has to mean the table is stale.

**And it runs over what a model *generates*, not only over what a corpus
contains.** `--samples` points both oracles at a checkpoint's own samples and
reports the orbit count beside the corpus's. That is what decides whether the
flat transformed arms over-generate because they are *continuing the orbit* --
D4 is closed on the canvas, so a mirrored prefix always has a legal
continuation -- or for some other reason. A length distribution cannot tell
those apart; an orbit count can.

    python3 scripts/orbit_oracle.py                 # writes runs/orbit_oracle.json
    python3 scripts/orbit_oracle.py --limit 200     # quicker, for a smoke test

    # the mechanism behind figure 11, over 64 samples per checkpoint
    python3 scripts/orbit_oracle.py --samples runs/composed_x24000n24_byte_square_s0.pt \
        --n 64 --device mps
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from dm.data import composed, quickdraw, tabler
from dm.eval.metrics import sample_programs
from dm.eval.records import corpus_config, load
from dm.eval.repeats import best_symmetry_repeat, corpus_stats, symmetry_stats
from dm.isa.codec import CODECS
from dm.isa.asm import AsmError
from dm.isa.spec import UnknownOpcode
from dm.isa.unroll import unroll
from dm.train import build_data

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"

CATEGORIES = ("cat", "dog", "bus", "car", "tree")


def corpora(limit: int) -> list[tuple[str, str, list[bytes]]]:
    """`(key, label, programs)`, natural first.

    Tabler is here because it is the corpus that makes the QuickDraw zero
    *readable*: a detector that found nothing anywhere would be broken, and this
    one finds a great deal in icons. Without that row the null below is an
    absence of evidence rather than evidence of absence.
    """
    return [
        ("tabler", "Tabler icons — natural",
         tabler.split()[0][:limit]),
        ("quickdraw", "QuickDraw — natural",
         quickdraw.load(CATEGORIES, "valid", limit=limit)),
        ("motifs", "QuickDraw motifs — the parts scenes are built from",
         quickdraw.load(CATEGORIES, "valid", limit=limit,
                        rdp_eps=composed.MOTIF_EPS, margin=composed.MOTIF_MARGIN)),
        ("control", "composed, translation only — the control",
         [s.flat for s in composed.build(limit, "valid", seed=0, control=True)]),
        ("composed", "composed, transformed — the corpus under test",
         [s.flat for s in composed.build(limit, "valid", seed=0)]),
    ]


def orbit_counts(programs: list[bytes], max_body: int, tol: int = 0) -> dict:
    """How many copies the best orbit in each program has, as a histogram.

    **The count is the measurement and the fraction is not.** A model that
    over-generates by drawing more distractors and a model that over-generates by
    continuing the orbit both produce longer programs and a higher foldable
    fraction; only the *count* separates them, because the corpus's own count is
    bounded by its policy and a continuation exceeds it.

    A generated program need not parse -- the flat arms lose up to a third of
    their samples to an unknown opcode -- so the unparseable ones are counted and
    excluded rather than silently dropped, and the denominator says which.

    **`tol` changes what is being asked, and both readings are needed here.** At
    `tol = 0` the matcher is exact, so it answers "did the model reproduce the
    group action byte for byte" -- a compression question, and the only setting at
    which a saving is real. Above zero it is a *detector*: it answers "is there
    something in the orbit's place", which is the question about over-generation,
    because a model drawing an approximate mirror is continuing the orbit and is
    invisible to an exact match.
    """
    counts: dict[int, int] = {}
    unparseable = saved = total = 0
    for program in programs:
        try:
            best = best_symmetry_repeat(program, max_body=max_body, tol=tol)
        except (AsmError, UnknownOpcode, ValueError, IndexError):
            # `symmetry_stats` would raise here. A generated program is exactly
            # where an unparseable one comes from, so the guard is this function's
            # reason to exist rather than a duplicate of that one -- and it needs
            # `AsmError` as well as `UnknownOpcode`, because a sample that hit the
            # length cap mid-instruction is *truncated* rather than illegal, which
            # is a different exception and cost a run to find out.
            unparseable += 1
            continue
        counts[best.count if best else 0] = counts.get(best.count if best else 0, 0) + 1
        total += len(program)
        saved += best.saved if best else 0
    return {"counts": dict(sorted(counts.items())), "unparseable": unparseable,
            "n": len(programs), "tol": tol, "bytes": total, "saved": saved,
            "fraction": saved / total if total else 0.0}


def sampled(checkpoint: Path, n: int, device: str, seed: int) -> tuple[str, list[bytes], dict]:
    """`n` samples from a checkpoint, through the sampler the metrics use.

    The corpus is rebuilt only to get the length cap the run's own generation used
    (`gen_cap x p99`), so a sample here is the same kind of draw the record's
    generation columns are -- and a *draw*, which is why the seed is reported.
    """
    from dm.data.dataset import ProgramDataset

    model, record = load(checkpoint)
    cfg = corpus_config(record)
    codec = CODECS[record["config"]["codec"]]
    _, val = build_data(cfg)
    p99 = ProgramDataset(val, codec, cfg.max_len).length_stats()["p99"]
    torch.manual_seed(seed)
    programs, _ = sample_programs(model.to(device), codec, n=n,
                                  max_new=min(cfg.max_len, int(cfg.gen_cap * p99)),
                                  device=device, top_k=40)
    extra = record["config"].get("extra") or {}
    policy = composed.policy_label(extra) \
        if record["config"]["data"] == "composed" else record["config"]["data"]

    # **A structured sample has to be unrolled before either oracle can see it.**
    # `REPEATX` *is* the fold, so a program containing one has no repeated
    # instruction block to find and the matcher would report the arm that emits
    # orbits by construction as the arm with none. Unrolling asks the question the
    # comparison is about -- does the drawing have an orbit -- of both spellings in
    # the same form. A sample that will not unroll is dropped like an unparseable
    # one and counted the same way.
    folded = bool(extra.get("structured"))
    if folded:
        programs = [flat for flat in (unroll(p) for p in programs) if flat]
        val = [flat for flat in (unroll(p) for p in val) if flat]
    return record["name"], programs, {"policy": policy, "codec": codec.name,
                                      "val": val, "unrolled": folded}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, default=300,
                    help="programs per corpus; the oracle is O(n^2) in body length")
    ap.add_argument("--max-body", type=int, default=64,
                    help="longest body to consider, in instructions. A composed "
                         "motif is ~26, so the default clears it with room")
    ap.add_argument("--samples", type=Path, nargs="+", default=None,
                    help="checkpoints whose *samples* get the same two oracles, "
                         "reported beside the corpus they were trained on")
    ap.add_argument("--n", type=int, default=64, help="samples per checkpoint")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tol", type=int, default=2,
                    help="second, tolerant reading over the samples. Exact matching "
                         "answers a compression question; over-generation needs a "
                         "detector, since an approximate mirror is still a continuation")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    default = "orbit_oracle_generated.json" if args.samples else "orbit_oracle.json"
    # A separate file by default: figure 6 reads the corpus record, and a mode
    # that appended rows to it would put generated programs in a panel whose
    # whole argument is about what natural corpora contain.
    out = args.out or RUNS / default

    if args.samples:
        return generated(args, out)

    rows = []
    print(f"{'corpus':<46}{'REPEAT':>9}{'REPEATX':>10}{'with orbit':>13}")
    for key, label, programs in corpora(args.limit):
        flat = corpus_stats(programs, max_body=args.max_body)
        sym = symmetry_stats(programs, max_body=args.max_body)
        row = {
            "key": key, "label": label, "n": sym["n"], "bytes": sym["bytes"],
            "repeat_saved": flat["saved"],
            "repeat_fraction": flat["saved"] / max(1, sym["bytes"]),
            "repeatx_saved": sym["saved"],
            "repeatx_fraction": sym["fraction"],
            "with_orbit": sym["with_orbit"],
            "elements": sym["elements"],
        }
        rows.append(row)
        print(f"{label:<46}{row['repeat_fraction']:>8.2%}"
              f"{row['repeatx_fraction']:>10.2%}"
              f"{row['with_orbit']:>8}/{row['n']}")

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        {"limit": args.limit, "max_body": args.max_body,
         "categories": list(CATEGORIES), "rows": rows}, indent=2))
    print(f"\n  -> {out}")
    return 0


def generated(args, out: Path) -> int:
    """Both oracles over each checkpoint's samples, against its own corpus."""
    rows = []
    print(f"{'checkpoint':<42}{'orbit':>8}{'copies in the best orbit':>34}")
    for checkpoint in args.samples:
        name, programs, meta = sampled(checkpoint, args.n, args.device, args.seed)
        sample_hist = orbit_counts(programs, args.max_body)
        loose = orbit_counts(programs, args.max_body, tol=args.tol)
        # The corpus the run was scored on, at the same count, so the comparison
        # is against this arm's own policy and not against a shared reference.
        corpus_hist = orbit_counts(meta["val"][: args.n], args.max_body)
        row = {
            "name": name, "policy": meta["policy"], "codec": meta["codec"],
            "n": args.n, "seed": args.seed,
            "mean_bytes": sum(map(len, programs)) / max(1, len(programs)),
            "corpus_mean_bytes": sum(map(len, meta["val"])) / len(meta["val"]),
            "unrolled": meta["unrolled"],
            "generated": sample_hist, "generated_tol": loose,
            "corpus": corpus_hist,
        }
        rows.append(row)
        found = sum(v for k, v in sample_hist["counts"].items() if k > 1)
        print(f"{name + (' [unrolled]' if meta['unrolled'] else ''):<42}"
              f"{found:>4}/{args.n:<3}"
              f"{'gen, exact ' + str(sample_hist['counts']):>34}")
        print(f"{'':<42}{'':>8}{f'gen, tol {args.tol} ' + str(loose['counts']):>34}")
        print(f"{'':<42}{'':>8}{'corpus ' + str(corpus_hist['counts']):>34}"
              f"   {row['mean_bytes']:.0f} B gen / {row['corpus_mean_bytes']:.0f} B corpus"
              f", {sample_hist['unparseable']} unparseable")

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"max_body": args.max_body, "n": args.n,
                               "seed": args.seed, "rows": rows}, indent=2))
    print(f"\n  -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
