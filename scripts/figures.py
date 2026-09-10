#!/usr/bin/env python3
"""Sample sheets: what a checkpoint actually draws, next to what the corpus is.

Every number in this project is a likelihood or a distance, and neither can see
sample quality -- `PLAN.md` §10 says so and then the project spent five months
without a figure. This produces the figure: a grid of drawings from a
checkpoint, and the same grid of real programs beside it, rendered through the
*same* VM and renderer the metrics use so nothing is prettied up on the way out.

    python3 scripts/figures.py runs/quickdraw_planbase24000eps2_*.pt
    python3 scripts/figures.py --corpus quickdraw --rows 3 --cols 8
    python3 scripts/figures.py runs/*.pt --out docs/figs

Writes one SVG per sheet. SVG because the strokes are polylines already: a
raster is a second lossy step between the trace and the eye, and the whole point
of looking is to see what the model did rather than what the rasteriser did.

Sampling is a draw (§10), so the seed is in the filename and in the caption. A
sheet is an illustration and never evidence: `scripts/resample.py` is what is
entitled to rank two arms.

**A sheet is never named `figN_...`.** `tests/test_plot.py` checks every
`docs/figs/fig*.svg` against its own canvas by reading panel groups in figure
coordinates, and a sheet's cells are nested SVGs with their own 256-unit viewBox
-- so a numbered sheet fails that check on every stroke past x = 160. The
measurement figures in `scripts/plots.py` are the numbered ones.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dm.isa.codec import CODECS
from dm.isa.spec import CANVAS, Op, Tier
from dm.models.planner import generate as planner_generate
from dm.train import build_data
from dm.vm.interp import VM
from dm.vm.render import to_svg
from scripts.resample import corpus_config, is_planner, load

ROOT = Path(__file__).resolve().parent.parent

#: Rendered at this many pixels per cell in the sheet. The canvas is 256 and the
#: strokes are 1-4 units wide, so anything below ~120 loses the thin ones.
CELL = 160
PAD = 10


def named(path: Path) -> str:
    """Repo-relative when it is inside the repo, absolute when `--out` is not."""
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def caption_lines(text: str, width: float, size: float = 11.5) -> list[str]:
    """A caption broken to the sheet's width.

    An SVG `<text>` element does not wrap, so a caption longer than the sheet is
    silently cut off at the edge -- which is how a sheet whose caption existed to
    say "no model was involved in this one" shipped without saying it. Measured
    against renders at 11.5px in the sans stack: ~6.2 px a character.
    """
    budget = max(20, int(width / (size / 1.9)))
    lines, line = [], ""
    for word in text.split():
        candidate = f"{line} {word}".strip()
        if len(candidate) > budget and line:
            lines.append(line)
            line = word
        else:
            line = candidate
    if line:
        lines.append(line)
    return lines


def sheet(programs: list[bytes], cols: int, title: str, caption: str) -> str:
    """A grid of drawings as one standalone SVG.

    Each cell is its own `<svg>` with the ISA's own viewBox, so a cell is
    exactly `dm/vm/render.py`'s output placed on a grid rather than a
    reimplementation of it that could drift.
    """
    rows = (len(programs) + cols - 1) // cols
    width = cols * (CELL + PAD) + PAD
    lines = caption_lines(caption, width - 2 * PAD)
    head = 42 + 16 * len(lines)
    height = rows * (CELL + PAD) + PAD + head
    vm = VM()

    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="ui-sans-serif, system-ui, sans-serif">',
        f'<rect width="{width}" height="{height}" fill="#ffffff"/>',
        f'<text x="{PAD}" y="22" font-size="15" font-weight="600" fill="#111">{title}</text>',
    ] + [
        f'<text x="{PAD}" y="{42 + 15 * i}" font-size="11.5" fill="#666">{line}</text>'
        for i, line in enumerate(lines)
    ]
    for i, program in enumerate(programs):
        x = PAD + (i % cols) * (CELL + PAD)
        y = head + (i // cols) * (CELL + PAD)
        trace = vm.run(program)
        # `to_svg` emits a full document; nesting it keeps the ISA's coordinate
        # system intact and makes the cell independent of the sheet's layout.
        inner = to_svg(trace, size=CELL).split("\n", 1)[1].rsplit("</svg>", 1)[0]
        stroke = "#d94b3a" if trace.faults else "#e6e6e6"
        out.append(
            f'<g transform="translate({x},{y})">'
            f'<svg width="{CELL}" height="{CELL}" viewBox="0 0 {CANVAS} {CANVAS}">{inner}</svg>'
            f'<rect width="{CELL}" height="{CELL}" fill="none" stroke="{stroke}"/>'
            f"</g>"
        )
    out.append("</svg>")
    return "\n".join(out)


def compare_sheet(blocks: list[tuple[str, str, list[bytes]]], cols: int,
                  title: str, caption: str) -> str:
    """Two arms' samples in one frame, one block of rows each.

    Two sheets side by side in a document is not the same figure: the contrast
    between what a flat-trace model draws and what a `REPEATX` model draws is the
    reading, and a reader comparing two files scrolls instead of seeing it.
    """
    vm = VM()
    per = (max(len(p) for _, _, p in blocks) + cols - 1) // cols
    width = cols * (CELL + PAD) + PAD
    lines = caption_lines(caption, width - 2 * PAD)
    head = 42 + 16 * len(lines)
    block_h = per * (CELL + PAD) + 38
    height = len(blocks) * block_h + PAD + head

    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="ui-sans-serif, system-ui, sans-serif">',
        f'<rect width="{width}" height="{height}" fill="#ffffff"/>',
        f'<text x="{PAD}" y="22" font-size="15" font-weight="600" fill="#111">{title}</text>',
    ] + [
        f'<text x="{PAD}" y="{42 + 15 * i}" font-size="11.5" fill="#666">{line}</text>'
        for i, line in enumerate(lines)
    ]
    for b, (label, sub, programs) in enumerate(blocks):
        top = head + b * block_h
        out.append(f'<text x="{PAD}" y="{top + 16}" font-size="13" font-weight="600" '
                   f'fill="#111">{label}</text>')
        out.append(f'<text x="{PAD}" y="{top + 32}" font-size="11" fill="#666">{sub}</text>')
        for i, program in enumerate(programs):
            x = PAD + (i % cols) * (CELL + PAD)
            y = top + 38 + (i // cols) * (CELL + PAD)
            trace = vm.run(program)
            inner = to_svg(trace, size=CELL).split("\n", 1)[1].rsplit("</svg>", 1)[0]
            edge = "#d94b3a" if trace.faults else "#e6e6e6"
            out.append(
                f'<g transform="translate({x},{y})">'
                f'<svg width="{CELL}" height="{CELL}" viewBox="0 0 {CANVAS} {CANVAS}">'
                f'{inner}</svg>'
                f'<rect width="{CELL}" height="{CELL}" fill="none" stroke="{edge}"/></g>'
            )
    out.append("</svg>")
    return "\n".join(out)


def real_programs(corpus: str, n: int) -> tuple[list[bytes], str]:
    """A val split, through the same loader training used."""
    from dm.data import synthetic

    if corpus == "synthetic":
        return synthetic.dataset(n, seed=99, tier=Tier.L1, depth=2), "Tier A val (generated)"
    if corpus in ("composed", "composed_control"):
        from dm.data import composed

        scenes = composed.build(n, "valid", seed=composed.val_seed(0),
                                control=corpus.endswith("control"))
        return [s.flat for s in scenes], (
            "composed val — " + composed.policy_label(
                {"control": corpus.endswith("control")})
        )
    from dm.data import quickdraw

    return (
        quickdraw.load(("cat", "dog", "bus", "car", "tree"), "valid", limit=n),
        "QuickDraw val (real human sketches)",
    )


#: Which colour each byte class of a constructed scene is drawn in. The orbit's
#: first copy and its images are the only thing the corpus was built to contain,
#: so they are the only thing with a hue; everything else is context.
PROVENANCE = {
    "motif": ("#1a1a1a", "the motif, drawn once"),
    "copies": ("#c8654a", "its D4 images — what REPEATX folds"),
    "prefix": ("#b8bcc4", "distractors: content that is not a copy"),
    "suffix": ("#b8bcc4", None),
}


def provenance_sheet(scenes, cols: int, title: str, caption: str) -> str:
    """A grid of constructed scenes with the planted orbit picked out in colour.

    **The classes come from `dm/eval/spelling.py`, not from a second reading of
    the generator.** That module's spans are what the bit decomposition is
    measured over and are checked per scene, so a figure built on them cannot
    disagree with the numbers it illustrates — and if it ever did, `check` would
    raise before anything was drawn.

    Each class is rendered as its own program through the same VM and the same
    renderer, then overlaid. A span of a composed scene *is* a legal program: the
    motifs are `MOVE`/`LINE` bodies concatenated at instruction boundaries, so a
    slice plus `HALT` runs.
    """
    from dm.eval.spelling import check, flat_spans

    rows = (len(scenes) + cols - 1) // cols
    width = cols * (CELL + PAD) + PAD
    lines = caption_lines(caption, width - 2 * PAD)
    head = 42 + 16 * len(lines)
    height = rows * (CELL + PAD) + PAD + head + 20
    vm = VM()
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="ui-sans-serif, system-ui, sans-serif">',
        f'<rect width="{width}" height="{height}" fill="#ffffff"/>',
        f'<text x="{PAD}" y="22" font-size="15" font-weight="600" fill="#111">{title}</text>',
    ] + [
        f'<text x="{PAD}" y="{42 + 15 * i}" font-size="11.5" fill="#666">{line}</text>'
        for i, line in enumerate(lines)
    ]
    legend = PAD
    for name, (colour, label) in PROVENANCE.items():
        if label is None:
            continue
        out.append(f'<rect x="{legend}" y="{head + 1}" width="18" height="3" '
                   f'fill="{colour}"/>')
        out.append(f'<text x="{legend + 24}" y="{head + 7}" font-size="11" '
                   f'fill="#666">{label}</text>')
        legend += 30 + int(6.1 * len(label))

    for i, scene in enumerate(scenes):
        check(scene)
        spans = flat_spans(scene)
        x = PAD + (i % cols) * (CELL + PAD)
        y = head + 20 + (i // cols) * (CELL + PAD)
        layers = [f'<rect width="{CANVAS}" height="{CANVAS}" fill="white"/>']
        for name, (colour, _) in PROVENANCE.items():
            chunk = b"".join(scene.flat[s] for s in spans[name])
            if not chunk:
                continue
            svg = to_svg(vm.run(chunk + bytes([int(Op.HALT)])), size=CELL,
                         ink=colour, background=None)
            layers.append(svg.split("\n", 1)[1].rsplit("</svg>", 1)[0])
        out.append(
            f'<g transform="translate({x},{y})">'
            f'<svg width="{CELL}" height="{CELL}" viewBox="0 0 {CANVAS} {CANVAS}">'
            + "".join(layers) +
            f'</svg><rect width="{CELL}" height="{CELL}" fill="none" stroke="#e6e6e6"/>'
            f"</g>"
        )
    out.append("</svg>")
    return "\n".join(out)


def sample(checkpoint: Path, n: int, seed: int, device: str,
           cols: int = 6) -> tuple[list[bytes], str, str, int]:
    model, record = load(checkpoint)
    # `--device` was never honoured: the weights stayed on CPU and any accelerator
    # raised "Placeholder storage has not been allocated on MPS device".
    model = model.to(device)
    cfg = corpus_config(record)
    codec = CODECS[record["config"]["codec"]]
    _, val = build_data(cfg)

    torch.manual_seed(seed)
    if is_planner(record):
        out = planner_generate(model, codec, n=n, steps=record["config"]["gen_steps"],
                               device=device, top_k=40, order="random")
        programs = out.programs
        arm = f"{record['config']['comp_objective']} composition + stroke decoder"
    else:
        from dm.data.dataset import ProgramDataset

        p99 = ProgramDataset(val, codec, cfg.max_len).length_stats()["p99"]
        # A conditional arm is laid out **one class per row**, which is the whole
        # point of having labels: the sheet then reads as "asked for each of these,
        # got these" rather than as an unlabelled grid. `cols` is the caller's
        # layout, so the class of cell i is its row index.
        n_classes = record["model"].get("n_classes", 0)
        classes = (torch.arange(n, device=device) // cols % n_classes
                   if n_classes else None)
        ids = model.generate(n, min(cfg.max_len, int(cfg.gen_cap * p99)), top_k=40,
                             device=device, monitor=codec.halt_monitor(n),
                             classes=classes)
        programs = [codec.decode(row.tolist()) for row in ids.cpu()]
        arm = "flat autoregressive"
        if n_classes:
            names = ", ".join(record["config"]["categories"][:n_classes])
            arm = f"class-conditional · one row per class, in order: {names}"
        extra = record["config"].get("extra") or {}
        if record["config"]["data"] == "composed":
            # Two composed records can differ in nothing a sheet shows except the
            # policy, and one of them is spelled with REPEATX. "flat
            # autoregressive" is dropped here because it names the model kind and
            # would sit next to "flat trace" naming the spelling.
            from dm.data import composed

            arm = f"{codec.name} alphabet · {composed.policy_label(extra)}"
    return programs, arm, record["name"], record["model"]["params"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoints", type=Path, nargs="*")
    ap.add_argument("--corpus", choices=("quickdraw", "synthetic", "composed",
                                        "composed_control"),
                    help="also emit a sheet of real programs from this corpus. The "
                         "two composed values draw the planted orbit in colour")
    ap.add_argument("--rows", type=int, default=3)
    ap.add_argument("--cols", type=int, default=8)
    ap.add_argument("--compare", action="store_true",
                    help="two checkpoints in one sheet, one block of rows each. The "
                         "contrast between two spellings of one corpus is the reading, "
                         "and two files make a reader scroll instead of see it")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", type=Path, default=ROOT / "docs" / "figs")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    n = args.rows * args.cols

    if args.corpus:
        path = args.out / f"corpus_{args.corpus}.svg"
        if args.corpus.startswith("composed"):
            # The constructed corpus is the one whose *provenance* is the point:
            # a plain sheet shows a scene, and what a reader needs to see is which
            # of its bytes the generator planted.
            from dm.data import composed

            scenes = composed.build(n, "valid", seed=composed.val_seed(0),
                                    control=args.corpus.endswith("control"))
            ceiling = composed.ceiling(scenes)
            path.write_text(provenance_sheet(
                scenes, args.cols,
                "composed val — " + composed.policy_label(
                    {"control": args.corpus.endswith("control")}),
                f"NO MODEL INVOLVED — this is the training corpus. {n} scenes "
                "assembled by dm/data/composed.py from real QuickDraw sketches: one "
                "sketch (black) placed once, the same sketch again as an exact mirror "
                "or quarter-turn of itself (orange), and unrelated sketches in the "
                f"other cells (grey). These {n}: {ceiling['saved_frac']:.1%} of bytes "
                f"foldable to REPEATX by construction, ratio {ceiling['ratio']:.3f}; "
                "the 1,000-scene val split the runs were scored on: 40.9%"))
        else:
            programs, title = real_programs(args.corpus, n)
            path.write_text(sheet(programs, args.cols, title,
                                  f"{n} programs, executed by dm/vm/interp.py"))
        print(f"  {named(path)}")

    vm = VM()
    if args.compare:
        if len(args.checkpoints) != 2:
            raise SystemExit("--compare takes exactly two checkpoints")
        blocks, names = [], []
        for checkpoint in args.checkpoints:
            programs, arm, name, params = sample(checkpoint, n, args.seed,
                                                 args.device, args.cols)
            traces = [vm.run(p) for p in programs]
            names.append(name)
            blocks.append((
                arm,
                f"{name} · {params:,} params · "
                f"{sum(t.valid for t in traces)}/{len(programs)} valid · "
                f"{sum(map(len, programs)) / len(programs):.0f} bytes/drawing mean",
                programs,
            ))
        path = args.out / f"compare_{names[0]}_vs_{names[1]}_s{args.seed}.svg"
        path.write_text(compare_sheet(
            blocks, args.cols, "One corpus, two spellings — what each model draws",
            f"EVERY DRAWING IS MACHINE-GENERATED, {n} samples per block · seed "
            f"{args.seed} · top-k 40 · executed by dm/vm/interp.py · red border = VM "
            "fault · an illustration, not evidence"))
        print(f"  {named(path)}")
        return 0

    for checkpoint in args.checkpoints:
        programs, arm, name, params = sample(checkpoint, n, args.seed, args.device,
                                             args.cols)
        traces = [vm.run(p) for p in programs]
        valid = sum(t.valid for t in traces)
        bytes_mean = sum(map(len, programs)) / max(1, len(programs))
        caption = (
            f"EVERY DRAWING IS MACHINE-GENERATED — sampled from this checkpoint, "
            f"nothing is a corpus drawing. {arm} · {params:,} params · seed "
            f"{args.seed} · top-k 40 · {valid}/{len(programs)} valid · "
            f"{bytes_mean:.0f} bytes/drawing mean · red border = VM fault"
        )
        path = args.out / f"{name}_s{args.seed}.svg"
        path.write_text(sheet(programs, args.cols, name, caption))
        print(f"  {named(path)}  ({valid}/{len(programs)} valid)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
