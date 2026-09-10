#!/usr/bin/env python3
"""What the `REPEATX` fold is worth in bits, separated from what changed around it.

P5 trained one constructed corpus in both of its spellings: the flat trace, and
the ISA v2 form where one `REPEATX` replaces the copies. The structured form is
**40.9% shorter** and decodes to the identical geometry, so the difference in
`bits/drawing` between the two arms looks like the fold's value -- and it is not,
because those arms also differ in codec and in sequence length.

`dm/eval/spelling.py` splits that difference into the part the ISA can claim and
the part it cannot, by using the fact that the two spellings share most of their
bytes exactly. **Read the `fold` row against the `context` row**: the first is
what removing the copies bought, the second is everything else about being a
shorter program in a different alphabet.

    python3 scripts/spelling.py \
        --flat runs/composed_x24000n24_byte_square_s0.pt \
        --structured runs/composed_sconv24000_token_square_s0.pt

Two forward passes over 1,000 val programs and no training, so it runs on CPU
while a sweep does. The scenes are rebuilt once and verified against each
record's own corpus digest, which is also the check that the two arms are two
spellings of the *same* drawings -- pass an arm trained at a different
`--orbit-sizes` and it is refused rather than reported.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dm.data.fingerprint import digest
from dm.eval.records import composed_val_scenes, is_planner, load
from dm.eval.spelling import CLASSES, check, class_bits, delta, flat_spans, structured_spans
from dm.isa.codec import CODECS
from dm.train import RUNS


def paired_scenes(flat_record: dict, structured_record: dict):
    """The scenes both arms were scored on, or a refusal naming which side broke.

    Rebuilt from the **flat** record, because that is the one whose config still
    describes the scenes as the generator drew them, and then held against the
    structured record's own digest. The structured arm's val split is
    `[s.structured for s in scenes]` exactly when the two runs drew the same
    scenes, so one digest comparison decides the whole pairing -- composition
    policy, seed, split size and category list at once. A config comparison would
    have to enumerate those, and a term missing from such a list is the run-4
    fault and the `augment` fault both (`dm/data/fingerprint.py`).
    """
    for side, record in (("flat", flat_record), ("structured", structured_record)):
        if is_planner(record):
            raise SystemExit(f"the {side} side is a planner record; this reads AR arms")
    if not (structured_record["config"].get("extra") or {}).get("structured"):
        raise SystemExit(
            f"{structured_record['name']} was not trained with --structured, so "
            "both sides are the same spelling and there is nothing to difference"
        )
    if flat_record["config"]["steps"] != structured_record["config"]["steps"]:
        # Not a nicety. A difference between two arms stopped at different points
        # on their own curves is the rate they were separating at, not a cost
        # (`PLAN.md`, the rules that bite). The ladder is the fix, not a caveat.
        raise SystemExit(
            f"{flat_record['name']} ran {flat_record['config']['steps']} steps and "
            f"{structured_record['name']} ran {structured_record['config']['steps']}; "
            "difference these at a matched budget or not at all"
        )

    scenes = composed_val_scenes(flat_record)
    if digest([s.structured for s in scenes]) != structured_record["corpus"]["val"]:
        raise SystemExit(
            f"{structured_record['name']} was not scored on the structured "
            f"spelling of {flat_record['name']}'s scenes -- the two arms differ in "
            "their composition policy, so their bits are not a difference"
        )
    for scene in scenes:
        check(scene)
    return scenes


def side_report(path: Path, scenes, spans_of, args) -> tuple[dict, dict]:
    """One arm's per-class bits, with the reconciliation that validates it."""
    model, record = load(path)
    codec = CODECS[record["config"]["codec"]]
    structured = bool((record["config"].get("extra") or {}).get("structured"))
    programs = [s.structured if structured else s.flat for s in scenes]
    report = class_bits(model.to(args.device), programs, codec,
                        [spans_of(s) for s in scenes], device=args.device,
                        max_len=record["config"]["max_len"],
                        batch_size=args.batch_size)
    # The record's own number, from the run's final eval on the same programs.
    # It is not an independent measurement -- it is the same forward pass a day
    # earlier -- which is exactly why it is the right check: the classes must
    # add up to it, and a decomposition that does not close is a decomposition
    # of something else.
    report.update(
        name=record["name"], codec=codec.name, params=record["model"]["params"],
        steps=record["steps_done"], spelling="structured" if structured else "flat",
        recorded=record["history"][-1]["bits_per_drawing"],
    )
    report["reconciles"] = report["bits_per_drawing"] - report["recorded"]
    return report, record


def table(flat: dict, structured: dict, split: dict) -> str:
    """The two arms and their difference, one class per row."""
    lines = [
        f"{'class':<10}{'flat bits':>11}{'struct bits':>13}{'Δ':>9}"
        f"{'flat B':>9}{'struct B':>10}  what it is",
        "-" * 82,
    ]
    kind = {"both": "shared bytes — the context changed, not the content",
            "flat": "the copies. What the fold removes",
            "structured": "REPEATX + ENDREP. What the fold costs"}
    for name, side in CLASSES.items():
        f = flat["classes"][name]
        s = structured["classes"][name]
        lines.append(
            f"{name:<10}{f['per_drawing']:>11.2f}{s['per_drawing']:>13.2f}"
            f"{f['per_drawing'] - s['per_drawing']:>+9.2f}"
            f"{f['bytes_per_drawing']:>9.1f}{s['bytes_per_drawing']:>10.1f}"
            f"  {kind[side]}"
        )
    lines += [
        "-" * 82,
        f"{'total':<10}{flat['bits_per_drawing']:>11.2f}"
        f"{structured['bits_per_drawing']:>13.2f}{split['total']:>+9.2f}"
        f"{flat['bytes_per_drawing']:>9.1f}{structured['bytes_per_drawing']:>10.1f}",
        "",
        f"  fold      {split['fold_total']:>+8.2f} bits/drawing  "
        f"= copies {split['fold']['copies']:+.2f} − header "
        f"{-split['fold']['header']:.2f}   ← the ISA's term",
        f"  context   {split['context_total']:>+8.2f} bits/drawing  "
        "= the same bytes, a different alphabet and a shorter sequence",
        f"  residual  {split['residual']:>+8.2f} bits/drawing  "
        "(truncation; 0 when neither spelling hit max_len)",
    ]
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--flat", type=Path, required=True,
                    help="checkpoint trained on the flat trace")
    ap.add_argument("--structured", type=Path, required=True,
                    help="checkpoint trained on the REPEATX form of the same scenes")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    flat_record = json.loads(args.flat.with_suffix(".json").read_text())
    structured_record = json.loads(args.structured.with_suffix(".json").read_text())
    scenes = paired_scenes(flat_record, structured_record)

    flat, _ = side_report(args.flat, scenes, flat_spans, args)
    structured, _ = side_report(args.structured, scenes, structured_spans, args)
    split = delta(flat, structured)

    print(f"\n## {flat['name']} (flat, {flat['codec']}, {flat['params']:,} params)")
    print(f"vs {structured['name']} (structured, {structured['codec']}, "
          f"{structured['params']:,} params) — {flat['steps']:,} steps each, "
          f"{flat['n']} val scenes")
    for side in (flat, structured):
        print(f"  {side['spelling']:<10} classes sum to {side['bits_per_drawing']:.2f} "
              f"against the record's {side['recorded']:.2f} "
              f"(reconciles to {side['reconciles']:+.4f})")
    print()
    print(table(flat, structured, split))

    out = args.out or RUNS / f"spelling_{flat['name']}_vs_{structured['name']}.json"
    out.write_text(json.dumps({"flat": flat, "structured": structured,
                               "delta": split}, indent=2))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
