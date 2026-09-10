#!/usr/bin/env python3
"""Tier B gate: does QuickDraw survive the L0 pipeline, and at what length?

`dm/data/quickdraw.py` has never been run end to end (PLAN.md 9.7). Two things
have to be true before any Tier B training number means anything:

1. **The roundtrip is faithful.** stroke-3 -> L0 bytecode -> VM -> geometry has
   to still be the drawing. RDP, the fit to canvas, 8-bit quantisation and the
   zero-length-LINE dedup all discard something; this measures how much and
   renders it so it can be looked at rather than assumed.
2. **The sequences fit.** `ProgramDataset` truncates at `max_len` (2048). The
   bit codec is 8x the byte count, so a corpus that is comfortable for byte can
   be silently *cut* for bit -- and a truncated program is scored on fewer bits
   than it costs, which would corrupt the one cross-codec comparable metric this
   project has. `rdp_eps` is the dial; this finds the setting where every arm
   fits.

Reference geometry is the sketch's own polyline fit to the canvas with *no* RDP
and *no* quantisation, so the reported error is exactly what the pipeline costs
and not what the source data lacks.

    python3 scripts/quickdraw_check.py --n 300
    python3 scripts/quickdraw_check.py --eps 0 1 2 3 4 6 8 --n 500
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dm.data.quickdraw import _fit_to_canvas, _strokes_from_stroke3, download, stroke3_to_program
from dm.eval.metrics import chamfer, iou
from dm.isa.codec import CODECS
from dm.vm.interp import Stroke, Trace, VM
from dm.vm.render import to_image

RUNS = Path("runs")
FIGS = RUNS / "figs"
MAX_LEN = 2048  # dm.train.TrainConfig.max_len -- the truncation this checks for


def reference_trace(s3: np.ndarray, margin: int) -> Trace:
    """The sketch itself, in canvas coordinates. No RDP, no quantisation."""
    strokes = _strokes_from_stroke3(np.asarray(s3))
    if not strokes:
        return Trace()
    fitted = _fit_to_canvas(strokes, margin)
    return Trace(strokes=[Stroke(tuple(map(tuple, s)), 1) for s in fitted])


def montage(pairs: list[tuple[Trace, Trace]], path: Path, cell: int = 128) -> None:
    """Reference on the top row, roundtrip beneath it, aligned by column."""
    n = len(pairs)
    sheet = Image.new("L", (cell * n, cell * 2), 255)
    for i, (ref, got) in enumerate(pairs):
        sheet.paste(to_image(ref, cell), (i * cell, 0))
        sheet.paste(to_image(got, cell), (i * cell, cell))
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)


def measure(sketches: list[np.ndarray], eps: float, margin: int, vm: VM) -> dict:
    kept, chamfers, ious, valid, pairs = [], [], [], 0, []
    for s3 in sketches:
        program = stroke3_to_program(s3, margin=margin, rdp_eps=eps)
        if not program:
            continue
        ref = reference_trace(s3, margin)
        got = vm.run(program)
        if ref.is_empty or got.is_empty:
            continue
        kept.append(program)
        chamfers.append(chamfer(ref, got))
        ious.append(iou(ref, got))
        valid += got.valid
        if len(pairs) < 12:
            pairs.append((ref, got))

    lengths = np.array([len(p) for p in kept])
    row = {
        "rdp_eps": eps,
        "n_in": len(sketches),
        "n_kept": len(kept),
        "dropped": 1.0 - len(kept) / max(1, len(sketches)),
        "validity": valid / max(1, len(kept)),
        "chamfer_mean": float(np.mean(chamfers)),
        "chamfer_p90": float(np.percentile(chamfers, 90)),
        "iou_mean": float(np.mean(ious)),
        "bytes_mean": float(lengths.mean()),
        "bytes_p99": float(np.percentile(lengths, 99)),
        "bytes_max": int(lengths.max()),
        "codecs": {},
    }
    # +1 for BOS: `ProgramDataset` stores `codec.with_bos(p)[:max_len]`.
    for name, codec in CODECS.items():
        seq = lengths * getattr(codec, "stride", 1) + 1
        row["codecs"][name] = {
            "mean": float(seq.mean()),
            "p99": float(np.percentile(seq, 99)),
            "max": int(seq.max()),
            "over_max_len": float((seq > MAX_LEN).mean()),
        }
    return row, pairs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--category", default="cat")
    ap.add_argument("--split", default="valid")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--margin", type=int, default=8)
    ap.add_argument("--eps", type=float, nargs="+", default=[0.0, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0])
    args = ap.parse_args()

    with np.load(download(args.category), encoding="latin1", allow_pickle=True) as z:
        sketches = list(z[args.split][: args.n])
    print(f"{args.category}/{args.split}: {len(sketches)} sketches, margin={args.margin}\n")

    vm = VM()
    rows = []
    for eps in args.eps:
        row, pairs = measure(sketches, eps, args.margin, vm)
        rows.append(row)
        montage(pairs, FIGS / f"quickdraw_{args.category}_eps{eps}.png")
        bit, byte = row["codecs"]["bit"], row["codecs"]["byte"]
        print(
            f"eps={eps:>4.1f}  bytes {row['bytes_mean']:6.1f} mean "
            f"{row['bytes_max']:4d} max | bit seq {bit['mean']:7.1f} mean "
            f"{bit['max']:5d} max, {100 * bit['over_max_len']:5.1f}% over 2048 "
            f"| byte {100 * byte['over_max_len']:4.1f}% over "
            f"| chamfer {row['chamfer_mean']:5.2f}px  IoU {row['iou_mean']:.3f} "
            f"| drop {100 * row['dropped']:.1f}%  valid {row['validity']:.3f}"
        )

    RUNS.mkdir(exist_ok=True)
    out = RUNS / f"quickdraw_check_{args.category}_{args.split}.json"
    out.write_text(json.dumps({"args": vars(args), "rows": rows}, indent=2))
    print(f"\nwrote {out} and {len(args.eps)} montages to {FIGS}/")


if __name__ == "__main__":
    main()
