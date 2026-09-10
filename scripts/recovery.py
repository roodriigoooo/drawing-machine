#!/usr/bin/env python3
"""Claim 2, read off a trained checkpoint: how much `REPEAT` did it find?

`dm/eval/repeats.py` gives the ceiling -- what an oracle allowed to see the
whole program could fold. `dm/eval/recovery.py` gives the numerator -- how much
of that redundancy the model has stopped paying for. This puts the two beside
each other on the corpus the run was actually scored on, which is the only
combination that means anything: recovery is a fraction of a corpus property,
and quoting it without its own denominator is how a number that is 0.4 on one
corpus and 0.4 on another gets read as the same finding.

    # the number, on the corpus the run was trained for
    python3 scripts/recovery.py runs/synthetic_c2_*_square_s0.pt

    # the length-generalisation test: train n <= 4, evaluate n = 8..16
    python3 scripts/recovery.py runs/synthetic_c2_byte_square_s0.pt \
        --max-repeat 16 --min-repeat 8

    # the transform tier (direction item 2, P5): the copies are mirrors and
    # rotations, so they are read from the generator instead of detected
    python3 scripts/recovery.py runs/composed_x24000n2_byte_square_s0.pt

CPU by default. It is one forward pass over the val split, so it can be run
while a sweep is training rather than after it.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dm.data import composed
from dm.eval.records import composed_val_scenes, load
from dm.eval.recovery import recovery, recovery_by_copy
from dm.eval.repeats import corpus_stats
from dm.isa.codec import CODECS, RelativeCodec
from dm.train import RUNS, build_data, config_from_record


def corpus_slug(args) -> str:
    """Names the *evaluation* corpus, for the output filename.

    Without it the in-distribution and extrapolation reports of one checkpoint
    write to the same path and the second silently destroys the first -- which
    is `PLAN.md` section 10's "records are keyed by tag and the tag does not
    carry the budget", reproduced here on the first day this script existed.
    The length-generalisation reading is a *comparison* between those two
    reports, so a collision does not just lose a file, it loses the result.
    """
    if args.max_repeat is None and args.min_repeat is None:
        return "indist"
    return f"n{args.min_repeat or 2}-{args.max_repeat or 4}"


def eval_programs(record: dict, args) -> tuple[list[bytes], str]:
    """The val split, or an out-of-distribution one at a different repeat count.

    Rebuilt through `build_data` so the corpus comes from the same code path
    training used -- including `split()`'s filtering of val against train, and
    including Tier C's grouped split, which a hand-rolled rebuild here would
    quietly get wrong.
    """
    overrides: dict = {}
    if args.max_repeat is not None or args.min_repeat is not None:
        extra = dict(record["config"].get("extra") or {})
        if args.max_repeat is not None:
            extra["max_repeat"] = args.max_repeat
        if args.min_repeat is not None:
            extra["min_repeat"] = args.min_repeat
        overrides["extra"] = extra
    cfg = config_from_record(record, **overrides)
    _, val = build_data(cfg)
    trained = (record["config"].get("extra") or {}).get("max_repeat", 4)
    label = (
        f"extrapolation (trained max_repeat={trained}, "
        f"evaluated {cfg.extra.get('min_repeat', 2)}..{cfg.extra.get('max_repeat')})"
        if overrides
        else "in-distribution (the split this run was scored on)"
    )
    if cfg.data == "composed":
        # The composition policy *is* the corpus here -- a control scene and a
        # transformed one differ in nothing else -- so a report that did not
        # name it would put the two arms of the whole comparison under the same
        # caption. `composed.policy_label` owns the wording because it asks the
        # generator what it offers rather than reading the config back, which is
        # the distinction that had the control arm captioned `n ∈ {2,4}` on a
        # corpus whose canvas allows it exactly two.
        label += f" — {composed.policy_label(cfg.extra)}"
    return val, label


def provenance(record: dict, programs: list[bytes], args) -> tuple[dict, dict]:
    """The ceiling, and the copy spans to score -- detected, or read off the
    generator.

    Two sources for one pair of numbers, and which one applies is a property of
    the corpus rather than a preference. `dm.eval.repeats` matches bytes, so on
    the constructed corpus it finds **nothing at all**: a mirrored copy is
    `x -> 255 - x` and shares no coordinate byte with its original, so a
    detector-driven reading would report `NaN` on the one corpus built to
    contain the structure and it would look like a null result rather than like
    a broken instrument (`tests/test_composed.py`). The generator chose the
    orbit, so it is asked.

    `None` spans mean "detect", which is what `dm.eval.recovery` does by
    default; they are returned explicitly so the call sites below have one
    shape.
    """
    if record["config"].get("data") != "composed":
        return corpus_stats(programs, tol=args.tol), {"spans": None, "copies": None}
    scenes = composed_val_scenes(record, programs)
    return composed.ceiling(scenes), {"spans": composed.spans_of(scenes),
                                      "copies": composed.copies_of(scenes)}


def copy_curve(report: dict, width: int = 40) -> str:
    """Bits per symbol against copy ordinal, as a bar chart.

    A curve, not a number, on purpose: a model that learnt the *rule* is flat in
    the ordinal and one that learnt a table of short repeats turns up near the
    trained bound. `recovery` averages the two together and cannot separate
    them.
    """
    rates, ordinals = report["bits_per_symbol"], report["copies"]
    if not rates:
        return "  no repeats in this corpus"
    top = max(rates) or 1.0
    lines = ["  copy   bits/symbol   symbols", "  ----   -----------   -------"]
    for ordinal, rate, count in zip(ordinals, rates, report["symbols"]):
        bar = "#" * max(1, round(width * rate / top))
        lines.append(f"  {ordinal:>4}   {rate:11.4f}   {count:>7}  {bar}")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoints", type=Path, nargs="+")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--tol", type=int, default=0,
                    help="oracle match tolerance; only 0 is a compression number")
    ap.add_argument("--max-repeat", type=int, default=None,
                    help="rebuild the eval corpus at this repeat count (Tier A)")
    ap.add_argument("--min-repeat", type=int, default=None)
    args = ap.parse_args()

    for path in args.checkpoints:
        model, record = load(path)
        codec = CODECS[record["config"]["codec"]]
        if record["config"].get("data") == "composed" and (
            args.max_repeat is not None or args.min_repeat is not None
        ):
            # The overrides rebuild the *corpus* at a different repeat count,
            # and the constructed corpus's copies are read from a generator run
            # at the record's own settings. Scoring one against the other's
            # spans is the exact misattribution `composed_val_scenes` refuses,
            # so it is refused here too rather than caught downstream.
            raise SystemExit(
                f"{record['name']}: --max-repeat/--min-repeat are Tier A knobs "
                "and do not apply to the constructed corpus, whose orbit sizes "
                "are a property of the run (--orbit-sizes)"
            )
        programs, label = eval_programs(record, args)
        ceiling, spans = provenance(record, programs, args)
        found = recovery(model.to(args.device), programs, codec, device=args.device,
                         max_len=record["config"]["max_len"],
                         batch_size=args.batch_size, tol=args.tol,
                         spans=spans["spans"])
        curve = recovery_by_copy(model, programs, codec, device=args.device,
                                 max_len=record["config"]["max_len"],
                                 batch_size=args.batch_size, tol=args.tol,
                                 copies=spans["copies"])

        print(f"\n## {record['name']}  ({codec.name}, {record['model']['params']:,} params)")
        print(f"corpus: {label}")
        print(
            f"ceiling: {ceiling['saved_frac']:.2%} of bytes foldable, "
            f"ratio {ceiling['ratio']:.4f}, "
            f"{ceiling['programs_with_repeat']:.1%} of programs carry one"
            + (f"  [{ceiling['source']}]" if "source" in ceiling else "")
        )
        print(
            f"recovery: {found['recovery']:+.4f}  "
            f"({found['bits_per_symbol_later']:.4f} bits/symbol on later copies "
            f"against {found['bits_per_symbol_first']:.4f} on their own first, "
            f"{found['n_with_repeat']}/{found['n']} programs)"
        )
        if isinstance(codec, RelativeCodec):
            print(
                "  NOTE: a relative codec makes a translational repeat a literally "
                "repeated symbol sequence, so this measures copy detection rather "
                "than structure discovery. Read it beside the absolute arm, never "
                "instead of it (dm/isa/relative.py)."
            )
        print(copy_curve(curve))

        out = RUNS / f"recovery_{record['name']}_{corpus_slug(args)}.json"
        out.write_text(json.dumps(
            {"name": record["name"], "codec": codec.name, "corpus": label,
             "ceiling": ceiling, "recovery": found, "by_copy": curve},
            indent=2,
        ))
        print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
