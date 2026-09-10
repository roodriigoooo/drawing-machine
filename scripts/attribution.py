#!/usr/bin/env python3
"""Where an arm's bits go, by ISA field — every arm, every codec, no training.

`docs/direction.md` §7.2 asks for `dm/eval/redundancy.py` generalised from the
planner's stroke decoder to every arm and codec, reporting bits by operand
`Kind`. This is that, plus the reading §7.2 actually wants:

**it prices the architectural prior before anyone builds it.** §7.2 proposes a
convolutional front end so that "an instruction is 1-4 bytes" is architecture
rather than something a model infers from position. The bits an arm spends on
symbols that **cannot occur at that position** are exactly what supplying the
grid could recover, and they are one forward pass away. That is
`docs/traps.md`'s rule for opcodes -- measure the redundancy before building the
thing that removes it -- applied to an architecture rather than to an opcode.

Three cuts, each partitioning every scored byte, plus a fourth on the `bit`
alphabet where a byte is eight symbols:

    field     opcode, coord_x, coord_y, delta_x, delta_y, count, scalar, id, xf
    slot      the operand's ordinal inside its own instruction
    opcode    every byte charged to the instruction it belongs to
    bitplane  `field[k]`, MSB first -- the only cut that can see whether a bit
              model has found the byte grid, since no symbol is ever illegal to it

Pass several checkpoints and they are printed side by side. Arms are differenced
only when their corpus digests agree, which the script checks rather than
assuming: two arms on two val splits differ by ~3.5 bits/drawing before anything
under test moves.

    python3 scripts/attribution.py runs/quickdraw_m5b24000eps4_{byte,token,bit}_square_s0.pt
    python3 scripts/attribution.py runs/quickdraw_plannerar24000eps2_byte_balanced_s0.pt \
        runs/quickdraw_planbase24000eps2_byte_square_s0.pt

CPU by default, minutes.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dm.eval.attribution import (
    BITPLANE,
    CUTS,
    attribute,
    attribute_planner,
    reconcile,
)
from dm.eval.records import codec_for, corpus_config, is_planner, load
from dm.train import RUNS, build_data

#: Fields printed even when empty, so two corpora with different opcode mixes
#: line up column for column and a missing row reads as "this corpus has none"
#: rather than as a shifted table.
SHOWN = ("opcode", "coord_x", "coord_y", "delta_x", "delta_y", "count", "xf",
         "scalar", "unparsed")


def read(path: Path, limit: int | None, device: str, batch_size: int) -> dict:
    model, record = load(path)
    # The record's width, never today's table: a pre-v2 `token` checkpoint is
    # 269 symbols wide and `CODECS["token"]` is 274, and every instrument here
    # reshapes logits against `codec.vocab_size`.
    codec = codec_for(record)
    config = corpus_config(record)
    _, val = build_data(config)
    programs = val[:limit] if limit else val
    if is_planner(record):
        reading = attribute_planner(model.to(device), programs, codec, device,
                                    batch_size)
    else:
        reading = attribute(model.to(device), programs, codec, device,
                            config.max_len, batch_size)
    reading |= {
        "name": record.get("name", path.stem),
        "checkpoint": str(path),
        "corpus": record.get("corpus", {}),
        "device": device,
        "limit": limit,
        "steps": record.get("steps"),
        "schema": record.get("schema"),
    }
    reading["reconciliation"] = reconcile(reading)
    return reading


def table(readings: list[dict], cut: str, key: str, rows: tuple[str, ...] | None,
          scale: str = "{:>10.2f}") -> None:
    names = rows or tuple(
        dict.fromkeys(name for r in readings for name in r["cuts"][cut])
    )
    print(f"{'':<14}" + "".join(f"{r['name'][-24:]:>26}" for r in readings))
    for name in names:
        line = f"{name:<14}"
        for r in readings:
            row = r["cuts"].get(cut, {}).get(name)
            line += (f"{'-':>26}" if row is None
                     else f"{scale.format(row[key]):>26}")
        print(line)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoints", type=Path, nargs="+")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None,
                    help="score only the first N val programs; for a smoke test, "
                         "never for a reading -- it changes the denominator")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    readings = [read(p, args.limit, args.device, args.batch_size)
                for p in args.checkpoints]

    digests = {r["corpus"].get("val") for r in readings}
    shared = len(digests) == 1
    print(f"\n{len(readings)} arm(s), val digest"
          f"{' shared' if shared else 's DIFFER -- do not difference these columns'}"
          f": {sorted(d or '?' for d in digests)}\n")

    print("## bits/drawing by field\n")
    table(readings, "field", "bits_per_drawing", SHOWN)
    print(f"\n{'total':<14}"
          + "".join(f"{r['bits_per_drawing']:>26.2f}" for r in readings))
    print(f"{'bytes/draw':<14}"
          + "".join(f"{r['bytes_per_drawing']:>26.1f}" for r in readings))

    print("\n## bits per *byte* of that field — comparable across alphabets\n")
    table(readings, "field", "bits_per_byte", SHOWN, "{:>10.3f}")

    print("\n## bits/drawing spent on symbols that cannot occur there\n")
    print("   The upper bound on what an architectural prior supplying the ISA")
    print("   grid could recover, per `docs/direction.md` §7.2.\n")
    table(readings, "field", "waste_per_drawing", SHOWN, "{:>10.4f}")
    print(f"\n{'total waste':<14}"
          + "".join(f"{r['field_total_waste_per_drawing']:>26.4f}" for r in readings))

    print("\n## bits per byte by operand slot within its instruction\n")
    print("   Operand k is predicted after operands 0..k-1 of the same")
    print("   instruction. A falling curve is within-instruction locality the")
    print("   model already exploits; a flat one is locality left on the table.\n")
    table(readings, "slot", "bits_per_byte", None, "{:>10.3f}")

    print("\n## bits/drawing by instruction\n")
    table(readings, "opcode", "bits_per_drawing", None)

    if any(BITPLANE in r["cuts"] for r in readings):
        print("\n## bits per symbol by bitplane, MSB first (bit alphabet only)\n")
        planes = tuple(
            name for r in readings for name in r["cuts"].get(BITPLANE, {})
            if name.startswith(("opcode[", "coord_x[", "coord_y["))
        )
        table(readings, BITPLANE, "bits_per_symbol",
              tuple(dict.fromkeys(planes)), "{:>10.3f}")

    print("\n## reconciliation — every cut must sum to the same total\n")
    for r in readings:
        rec = r["reconciliation"]
        print(f"  {r['name'][-40:]:<42} spread {rec['cut_spread']:.2e}"
              f"  truncated {r['truncated']}/{r['n']}")

    for cut in CUTS:
        for r in readings:
            r["cuts"][cut] = {k: {kk: (round(vv, 6) if isinstance(vv, float) else vv)
                                  for kk, vv in v.items()}
                              for k, v in r["cuts"][cut].items()}

    stem = "_".join(sorted(Path(c).stem for c in map(str, args.checkpoints)))[:120]
    out = args.out or RUNS / f"attribution_{stem}_n{readings[0]['n']}.json"
    if out.exists() and not args.overwrite:
        print(f"\n  refusing to replace {out} without --overwrite")
        return 1
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"shared_val_digest": shared,
                               "readings": readings}, indent=2))
    print(f"\n  -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
