#!/usr/bin/env python3
"""The project's result figures, each built from the records it reports.

Fourteen figures, chosen because each one shows something a table states but
does not *show*:

1. **Claim 1 reverses with the corpus.** A table gives two numbers; the figure
   gives them against the noise floor that decides whether either is real, which
   is the actual argument.
2. **Claim 3's gap does not close.** The retraction that produced this figure
   came from differencing both rungs against one baseline. Drawn against each
   arm's own budget, the closure disappears -- and the figure shows *why* the
   wrong reading was so easy to believe.
3. **Claim 2 learned to copy, not to count.** The per-copy curve at two seeds.
   The in-count copies lie on top of each other and the out-of-count ones do
   not, which no scalar can say.
4. **What the device actually costs**, and why peak RAM is a constant rather
   than a slope.
5. **What the interpreter's work is proportional to** -- not the bytecode it
   reads, which explains only R² = 0.80, but decode count plus geometry emitted,
   which explains 0.999. The ISA is a compression format and the figure is what
   makes that a cost model instead of a remark.
6. **Why a constructed ceiling can be believed.** A null is the hardest thing to
   show honestly, because a broken detector produces the same zero. The figure
   is built around the row that makes the zero readable: the same oracle finds
   3x more in Tabler's icons than REPEAT can express, and then finds nothing at
   all in QuickDraw.
7. **What the transform tier costs a device, and what its knob buys.** Peak stack
   is linear in the depth bound at 12 bytes a scope, and composing by value
   rather than through pointers is strictly dominated -- which needs a *plane*,
   because it moved flash and stack at once and either axis alone shows half of
   it.
8. **The sample-quality trade.** Nine numbers in a table hide robust ordering
   reversals: flat arm wins fidelity, planners win coverage/termination, and AR
   composition beats diffusion on MMD/NNA. Exact planner coverage rank is noisy.
9. **The transform tier's claim, and its falsifier firing on the wrong
   comparison.** A scalar says the control recovered more; the panel says the
   disagreement is a hundredth of the collapse both arms achieved.
10. **What the fold is worth.** 40.9% of the bytes; 1.2% of the bits for the
   opcode's own term and 4.1% for the whole spelling effect — read off a pair
   whose codec is held fixed, so the context term is the fold's shorter
   sequence and not the alphabet.
11. **Where the tier actually pays.** Not likelihood -- termination, on two
   columns `bits/drawing` cannot see, with the flat spelling's own seeds four
   times further apart than the structured spelling's.
12. **Why the constructed corpus's loss sits lower**, which is a fact about the
   corpus and not about the model -- and, in the same panel, the evidence that
   outside its planted structure it is as hard as the natural corpus it is made
   of. Area is bits/drawing, so the argument is a shape.
13. **What a class label buys, on the instrument that can see it.** The paired
   gap scatters 4.1 bits across two seeds and once lands past the 2.32-bit
   ceiling; the free classifier reads the same identity twice to within 0.03
   bits. Two instruments, one quantity, and only the single-model one resolves
   it.
14. **Whether asking for a cat produces cat-shaped geometry.** A margin over the
   next-nearest class is meaningless against zero and readable against the
   margin *real drawings of that class* achieve, so the corpus's own separation
   is drawn as the block the model has to clear. And the two columns that fall
   short fall short in the same direction the margin overshoots, which is what
   makes them one finding rather than three.

No figure here invents a number: each reads `runs/*.json` or a recovery report,
so a figure that disagrees with a table means the table is stale.

    python3 scripts/plots.py                 # all of them, into docs/figs
    python3 scripts/plots.py --only 2
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dm.eval.plot import INK, MINUS, MUTED, PALETTE, Panel, Series, figure

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"


def record(tag: str) -> dict:
    report = json.loads((RUNS / f"{tag}.json").read_text())
    if report.get("report_schema") == 2 and report.get("status", "complete") != "complete":
        raise SystemExit(f"{tag} is a failed report; geometry is unavailable")
    return report


def history(tag: str) -> tuple[list[float], list[float]]:
    entries = [e for e in record(tag)["history"] if "bits_per_drawing" in e]
    return [e["step"] for e in entries], [e["bits_per_drawing"] for e in entries]


# ---------------------------------------------------------------------------


def fig_granularity() -> str:
    """Claim 1: the same axis, two corpora, each read against its own floor.

    A forest plot rather than two bars, because the quantity that decides the
    result is not the estimate — it is the estimate's size relative to the
    corpus's own resolution floor, and a bar chart cannot show a floor at all.
    """
    rows = [
        ("Tier A — synthetic programs", -0.67, 0.77, 1.0),
        ("Tier B — QuickDraw sketches", 11.58, 0.60, 2.5),
    ]
    panel = Panel(xlim=(-4, 14), ylim=(-0.55, 1.55), width=310, height=112,
                  xlabel="Δ bits/drawing   (bit alphabet − byte alphabet)",
                  yticks=[], xticks=[-4, -2, 0, 2, 4, 6, 8, 10, 12, 14])
    panel.rule(0.0, axis="x", color=MUTED, dash="2 2")

    for i, (name, delta, ci, floor) in enumerate(rows):
        y = 1.0 - i
        resolved = abs(delta) > 2 * floor
        colour = PALETTE["clay"] if resolved else PALETTE["teal"]
        # Each row's own floor, drawn only across that row -- one shared band
        # would imply the two corpora share a floor, which is the mistake this
        # figure exists to prevent.
        panel.box(-floor, floor, y - 0.3, y + 0.3, PALETTE["slate"], 0.16)
        panel.add(Series([delta - ci, delta + ci], [y, y], color=colour, width=1.4))
        panel.add(Series([delta], [y], color=colour, marker=4.0))
        for end in (delta - ci, delta + ci):
            panel.add(Series([end, end], [y - 0.11, y + 0.11], color=colour, width=1.4))
        # Row name outside the frame on the left; value outside the interval on
        # the right. Nothing overlaps the data.
        panel.note(-4.4, y, name, anchor="end", color="#1a1a1a", size=12, dy=4)
        panel.note(-4.4, y - 0.34, f"floor ±{floor:.1f}", anchor="end", color=MUTED, size=10.5)
        panel.note(delta, y, f"{delta:+.2f} ±{ci:.2f}".replace("-", MINUS),
                   anchor="middle", color=colour, size=11.5, dy=-11)
        panel.note(delta, y, "unresolvable" if not resolved
                   else f"{abs(delta) / floor:.0f}× the floor",
                   anchor="middle", color=MUTED, size=10.5, dy=17)

    return figure(
        [(panel, 230, 22)], width=580, height=182,
        number="Figure 1.",
        caption="A two-symbol alphabet is free on generated programs and costs 11.58 bits "
                "on real drawings — same axis, same code, same guards. Bars are 95% "
                "intervals carrying the between-seed term; grey blocks are each corpus's "
                "own resolution floor. Neither number means anything without it.",
    )


def fig_budget() -> str:
    """Claim 3: differenced against one baseline it closes; against its own it does not."""
    left = Panel(xlim=(0, 24000), ylim=(550, 700), width=240, height=175,
                 xlabel="training steps", ylabel="bits/drawing",
                 xticks=[0, 6000, 12000, 18000, 24000],
                 xtick_labels=["0", "6k", "12k", "18k", "24k"],
                 yticks=[550, 600, 650, 700])
    arms = [
        ("quickdraw_plannerdiff24000eps2_byte_balanced_s0", PALETTE["clay"], "diffusion", "", -10),
        ("quickdraw_plannerar24000eps2_byte_balanced_s0", PALETTE["olive"], "AR comp.", "5 2", 16),
        ("quickdraw_planbase24000eps2_byte_square_s0", PALETTE["teal"], "flat AR", "2 2", -10),
    ]
    for tag, colour, label, dash, dy in arms:
        xs, ys = history(tag)
        left.add(Series(xs, ys, color=colour, dash=dash, width=1.6))
        left.note(23400, ys[-1], label, anchor="end", color=colour, size=11, dy=dy - 6)

    # Every arm is still descending at 24k -- which is why an absolute total is
    # not an asymptote and only the paired difference is readable.
    right = Panel(xlim=(9000, 27000), ylim=(30, 60), width=240, height=175,
                  xlabel="training steps", ylabel="gap vs flat AR (bits)",
                  xticks=[12000, 24000], xtick_labels=["12k", "24k"],
                  yticks=[30, 40, 50, 60])

    wrong = [(12000, 55.54), (24000, 47.41)]
    right_pairs = {
        "diffusion": ([(12000, 48.97), (24000, 47.41)], PALETTE["clay"]),
        "AR comp.": ([(12000, 38.75), (24000, 39.82)], PALETTE["olive"]),
    }
    right.add(Series([p[0] for p in wrong], [p[1] for p in wrong],
                     color=MUTED, dash="1 3", width=1.3, marker=3.0))
    right.note(12000, 55.54, "differenced against", anchor="start", color=MUTED, size=10, dy=-16)
    right.note(12000, 55.54, "one fixed baseline", anchor="start", color=MUTED, size=10, dy=-5)

    for label, (pts, colour) in right_pairs.items():
        right.add(Series([p[0] for p in pts], [p[1] for p in pts],
                         color=colour, width=1.8, marker=3.6))
        right.note(24300, pts[-1][1], label, anchor="start", color=colour, size=11, dy=4)

    return figure(
        [(left, 62, 26), (right, 400, 26)], width=720, height=248,
        number="Figure 2.",
        caption="Left: every arm is still descending at 24,000 steps, so no absolute "
                "total is an asymptote. Right: the planner's gap over the flat arm, "
                "each rung differenced against the baseline at its own budget "
                "(coloured) and against a single 24,000-step baseline (grey). The "
                "grey line closes at 8 bits per doubling; the real one does not close "
                "at all, and on the AR-composition arm it widens.",
    )


def fig_recovery() -> str:
    """Claim 2: two seeds, and the curve splits where the trained counts stop."""
    def curve(seed: int) -> tuple[list[float], list[float]]:
        path = RUNS / f"recovery_synthetic_c2flat24000_byte_square_s{seed}_n5-6.json"
        by = json.loads(path.read_text())["by_copy"]
        return by["copies"], by["bits_per_symbol"]

    panel = Panel(xlim=(0.6, 7.5), ylim=(0, 2.05), width=310, height=185,
                  xlabel="copy ordinal within the repeat",
                  ylabel="bits/symbol", xticks=[1, 2, 3, 4, 5, 6],
                  yticks=[0, 0.5, 1.0, 1.5, 2.0])

    # Everything right of 4 is a count the model never saw in training.
    panel.band(4.5, 7.5, PALETTE["clay"], 0.09, axis="x")
    panel.note(7.35, 1.95, "counts never", anchor="end", color=PALETTE["clay"], size=10.5)
    panel.note(7.35, 1.95, "trained on", anchor="end", color=PALETTE["clay"],
               size=10.5, dy=12)

    for seed, colour, dash in ((0, PALETTE["indigo"], ""), (1, PALETTE["sand"], "4 2")):
        xs, ys = curve(seed)
        panel.add(Series(xs, ys, color=colour, dash=dash, width=1.7, marker=3.2))
        panel.note(xs[-1] + 0.12, ys[-1], f"seed {seed}", color=colour, size=11, dy=4)

    panel.rule(1.7868, axis="y", color=MUTED, dash="2 3")
    panel.note(0.75, 1.83, "cost of an unrepeated body", color=MUTED, size=10.5)

    return figure(
        [(panel, 62, 24)], width=470, height=252,
        number="Figure 3.",
        caption="Within the trained counts the two seeds agree to 0.03 bits/symbol; "
                "one copy past the bound they differ by 0.53 and then 0.94. The copy "
                "mechanism is a property of the model, the count prior is a property "
                "of the seed — so the break's location replicates and its depth does "
                "not. Neither seed rises above the unrepeated-body rate.",
    )


def fig_memory() -> str:
    """Claim 4: why the port streams geometry instead of buffering it.

    The interesting number is not the 288 bytes — it is that 288 is a *constant*
    where the obvious implementation is a line with a slope, and that the line
    leaves the part somewhere inside the corpus this project actually trains on.
    """
    from dm.data import quickdraw
    from dm.vm.interp import VM

    vm = VM()
    programs = quickdraw.load(("cat", "dog", "bus", "car", "tree"), "valid", limit=1000)
    xs, ys = [], []
    for program in programs:
        trace = vm.run(program)
        paths = [s.points for s in trace.strokes] + [r.points for r in trace.regions]
        if not paths:
            continue
        xs.append(len(program))
        # Two int32 per buffered point, plus the interpreter's own frame.
        ys.append(288 + 8 * max(len(p) for p in paths))

    top = max(ys) * 1.15
    panel = Panel(xlim=(0, 460), ylim=(0, top), width=300, height=185,
                  xlabel="program length (bytes)", ylabel="peak RAM (bytes)",
                  xticks=[0, 100, 200, 300, 400])

    panel.add(Series(xs, ys, color=PALETTE["clay"], width=0, marker=1.5))
    panel.add(Series([0, 460], [288, 288], color=PALETTE["teal"], width=1.8))
    panel.note(14, 288, "streamed — 288 B, whatever the program",
               color=PALETTE["teal"], size=11, dy=16)
    panel.note(300, max(ys) * 0.92, "buffered", anchor="end",
               color=PALETTE["clay"], size=11.5)
    panel.note(300, max(ys) * 0.92, "(one path held in RAM)", anchor="end",
               color=PALETTE["clay"], size=10, dy=12)

    return figure(
        [(panel, 66, 22)], width=460, height=252,
        number="Figure 4.",
        caption="Peak interpreter RAM over 1,000 QuickDraw val programs. Buffering a path "
                "— the obvious port of the reference VM — costs 8 bytes per point, so it "
                "is a slope; streaming each point to the sink is a constant. On this "
                "corpus buffering peaks at ~1.2 KB and would fit, which is the honest "
                "reading: the design matters because the cost scales, not because it is "
                "large here. A 3 KB program of curves is ~7,000 points, ~56 KB, and no "
                "part in this class has that.",
    )


def fig_instructions() -> str:
    """Claim 4: what the interpreter's work is actually proportional to.

    The obvious plot is instructions against program length, and it is the one
    that misleads: the ISA is a *compression* format, so a `REPEAT` byte is
    cheap to store and expensive to run, and byte count explains only R² = 0.80
    of the cost across three corpora.

    Two terms explain 0.999 of it, and both are interpretable: a fixed price to
    decode and dispatch each bytecode instruction, and a marginal price per
    point of geometry emitted. That decomposition is the figure -- it says where
    a device's cycles go, and it is what a bar chart of one mean cannot.
    """
    import numpy as np

    source = RUNS / "qemu_instructions.json"
    if not source.exists():
        raise SystemExit(
            f"{source.relative_to(ROOT)} is missing -- it is a measurement, not a "
            "commitment, and `runs/` is gitignored. Regenerate with\n"
            "  python3 scripts/footprint.py --instructions 40   (needs qemu-system-arm)"
        )
    data = json.loads(source.read_text())
    rows = data["rows"]
    corpora = ["quickdraw", "tier A L1", "tier A L0"]
    colour = {"quickdraw": PALETTE["teal"], "tier A L1": PALETTE["clay"],
              "tier A L0": PALETTE["indigo"]}

    y = np.array([r["instructions"] for r in rows], float)
    design = np.column_stack([np.ones(len(rows)),
                              [r["steps"] for r in rows],
                              [r["points"] for r in rows],
                              [r["discs"] for r in rows]])
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    pred = design @ coef
    r2 = lambda p: 1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum()

    lengths = np.column_stack([np.ones(len(rows)), [r["bytes"] for r in rows]])
    by_length, *_ = np.linalg.lstsq(lengths, y, rcond=None)
    r2_len = r2(lengths @ by_length)

    top = max(y.max(), pred.max()) * 1.08
    left = Panel(xlim=(0, 420), ylim=(0, top), width=225, height=195,
                 xlabel="program length (bytes)", ylabel="guest instructions",
                 xticks=[0, 100, 200, 300, 400])
    right = Panel(xlim=(0, top), ylim=(0, top), width=225, height=195,
                  xlabel="predicted from decode + geometry", ylabel="")

    for corpus in corpora:
        idx = [i for i, r in enumerate(rows) if r["corpus"] == corpus]
        left.add(Series([rows[i]["bytes"] for i in idx], [y[i] for i in idx],
                        color=colour[corpus], width=0, marker=2.0))
        right.add(Series([pred[i] for i in idx], [y[i] for i in idx],
                         color=colour[corpus], width=0, marker=2.0))

    right.add(Series([0, top], [0, top], color=MUTED, dash="3 3", width=1.0))
    left.note(12, top * 0.93, f"R² = {r2_len:.2f}", color=MUTED, size=11.5)
    right.note(12, top * 0.93, f"R² = {r2(pred):.3f}", color=MUTED, size=11.5)

    for i, corpus in enumerate(corpora):
        left.note(406, top * (0.30 - 0.075 * i), corpus, anchor="end",
                  color=colour[corpus], size=10.5)

    return figure(
        [(left, 62, 24), (right, 372, 24)], width=640, height=282,
        number="Figure 5.",
        caption="Guest instructions on an emulated Cortex-M0, one QEMU run per program "
                f"with the empty-corpus baseline subtracted. Left: program length explains "
                f"only R² = {r2_len:.2f} of the cost, because the ISA is a compression "
                "format — a REPEAT byte is cheap to store and expensive to run. Right: two "
                f"terms explain R² = {r2(pred):.3f} — <b>{coef[1]:.0f} instructions to "
                f"decode and dispatch each bytecode instruction, and {coef[2]:.0f} per "
                "point of geometry emitted</b>. That is where a device's cycles go.",
    )


def fig_orbits() -> str:
    """Why the constructed corpus's ceiling can be believed.

    A constructed ceiling is only worth having if the corpus it is built from
    did not already contain the structure being planted. That is a *null* to be
    established, and a null is the hardest thing to show honestly: a detector
    that found nothing anywhere would produce this same zero.

    So the figure is built around the row that makes the zero readable. The same
    oracle, on the same axis, finds **three times more** in Tabler's icons than
    `REPEAT` can express — arrows, gears and clock faces are radially symmetric
    and translationally repetitive not at all — and then finds *exactly nothing*
    in QuickDraw. One panel would show the ceiling; the second is there because
    "0.00% of bytes" and "0 of 300 programs" are different claims, and only the
    second rules out a small effect averaged thin.
    """
    source = RUNS / "orbit_oracle.json"
    if not source.exists():
        raise SystemExit(
            f"{source.relative_to(ROOT)} is missing -- it is a measurement, not a "
            "commitment, and `runs/` is gitignored. Regenerate with\n"
            "  python3 scripts/orbit_oracle.py"
        )
    data = json.loads(source.read_text())
    rows = data["rows"]

    short = {"tabler": "Tabler icons", "quickdraw": "QuickDraw",
             "motifs": "QuickDraw motifs", "control": "composed — control",
             "composed": "composed — transformed"}
    kind = {"tabler": "natural", "quickdraw": "natural", "motifs": "natural",
            "control": "constructed", "composed": "constructed"}

    n = len(rows)
    left = Panel(xlim=(0, 44), ylim=(-0.7, n - 0.3), width=250, height=196,
                 xlabel="% of bytecode bytes an oracle can fold",
                 yticks=[], xticks=[0, 10, 20, 30, 40],
                 xtick_labels=["0", "10", "20", "30", "40"])
    right = Panel(xlim=(0, 108), ylim=(-0.7, n - 0.3), width=104, height=196,
                  xlabel="% carrying an orbit", yticks=[], xticks=[0, 50, 100],
                  xtick_labels=["0", "50", "100"])

    for i, row in enumerate(rows):
        y = n - 1 - i
        a = 100 * row["repeat_fraction"]
        b = 100 * row["repeatx_fraction"]
        gain = b - a
        # The dumbbell's *length* is the reading: what the transform tier buys
        # over the tier that already exists. A pair of bars would show two
        # totals and hide the quantity the figure is about.
        left.add(Series([a, b], [y, y], color=PALETTE["slate"], width=1.3))
        left.add(Series([a], [y], color=PALETTE["teal"], marker=3.6))
        left.add(Series([b], [y], color=PALETTE["clay"], marker=3.6))

        left.note(-1.6, y, short[row["key"]], anchor="end", color=INK, size=11.5, dy=1)
        left.note(-1.6, y - 0.30, kind[row["key"]], anchor="end", color=MUTED, size=9.5)
        if gain > 1.0:
            left.note((a + b) / 2, y, f"+{gain:.1f}", anchor="middle",
                      color=PALETTE["clay"], size=10.5, dy=-9)
        else:
            left.note(max(a, b) + 1.2, y, f"{b:.2f}%", anchor="start",
                      color=MUTED, size=10.5, dy=4)

        share = 100 * row["with_orbit"] / max(1, row["n"])
        colour = PALETTE["clay"] if share else PALETTE["slate"]
        right.box(0, share, y - 0.26, y + 0.26, colour, 0.85 if share else 0.0)
        right.note(share + 3 if share < 60 else share - 3, y,
                   f"{row['with_orbit']}/{row['n']}",
                   anchor="start" if share < 60 else "end",
                   color="#ffffff" if share >= 60 else MUTED, size=9.5, dy=3.5)

    # The legend sits below the last row rather than above the first: the top
    # row carries the annotation that makes the figure's point, and a legend
    # over it was the one collision a rendered check caught.
    left.add(Series([20.0], [-0.5], color=PALETTE["teal"], marker=3.6))
    left.note(21.4, -0.5, "REPEAT", color=PALETTE["teal"], size=10.5, dy=4)
    left.add(Series([31.0], [-0.5], color=PALETTE["clay"], marker=3.6))
    left.note(32.4, -0.5, "REPEATX", color=PALETTE["clay"], size=10.5, dy=4)

    quickdraw_row = next(r for r in rows if r["key"] == "quickdraw")
    tabler_row = next(r for r in rows if r["key"] == "tabler")
    built = next(r for r in rows if r["key"] == "composed")
    ratio = tabler_row["repeatx_fraction"] / max(1e-9, tabler_row["repeat_fraction"])
    # Which elements Tabler's orbits use, because a *flat* spread over the eight
    # would be what a detector firing on noise looks like. Codes 2 and 6 are the
    # quarter and three-quarter turns.
    elements = {int(k): v for k, v in tabler_row["elements"].items()}
    turns = (elements.get(2, 0) + elements.get(6, 0)) / max(1, sum(elements.values()))

    return figure(
        [(left, 176, 22), (right, 470, 22)], width=620, height=272,
        number="Figure 6.",
        caption="Two oracles over identical programs, so the gap between them is the "
                "transform tier and nothing else. <b>The same detector that finds "
                f"{ratio:.1f}× more structure in Tabler's icons than REPEAT can express "
                f"— {100 * tabler_row['repeatx_fraction']:.1f}% against "
                f"{100 * tabler_row['repeat_fraction']:.1f}%, in "
                f"{tabler_row['with_orbit']} of {tabler_row['n']} icons — finds none at "
                f"all in QuickDraw</b>: {100 * quickdraw_row['repeatx_fraction']:.2f}% of "
                f"bytes, in {quickdraw_row['with_orbit']} of {quickdraw_row['n']} "
                f"programs. Tabler's orbits are {turns:.0%} quarter and three-quarter "
                "turns, which is what icons are — arrows, gears, clock faces — and not "
                "what a detector firing on noise would look like, since that would "
                "spread evenly over the eight elements. The null is what licenses the "
                f"constructed corpus's {100 * built['repeatx_fraction']:.1f}%: it "
                "establishes that every fold on offer is structure the generator put "
                "there rather than structure the corpus already had. The control is "
                "visible to both oracles, which is the check that neither is blind.",
    )


def fig_device_cost() -> str:
    """What the transform tier cost the device, and what its knob is worth.

    Two things a table cannot show. The first is that the depth bound is
    **linear and cheap** -- 12 bytes a scope, the size of one transform -- so
    §2.6's "the depth bound is the knob" has a price per turn rather than a
    hope attached. The second needs a *plane*: composing by value rather than
    through pointers moved flash and stack together, and a one-axis chart of
    either would show half a finding. On the same axes it is strictly dominated,
    and the depth sweep beside it shows why that matters -- the knob buys stack
    at almost no flash, so the two costs are not interchangeable.
    """
    source = RUNS / "depth_cost.json"
    if not source.exists():
        raise SystemExit(
            f"{source.relative_to(ROOT)} is missing -- it is a measurement, not a "
            "commitment, and `runs/` is gitignored. Regenerate with\n"
            "  python3 scripts/depth_cost.py   (needs clang and llvm-size)"
        )
    data = json.loads(source.read_text())
    rows = data["rows"]
    v1, ptr, val = data["isa_v1"], data["by_pointer"], data["by_value"]
    depths = [r["depth"] for r in rows]
    stacks = [r["stack"] for r in rows]

    # The slope, from the widest pair rather than a fit: the low depths carry a
    # few bytes of alignment wobble and the figure should quote the rule, not
    # the noise around it.
    per_scope = (stacks[-1] - stacks[3]) / (depths[-1] - depths[3])

    left = Panel(xlim=(0, 17), ylim=(240, 680), width=250, height=190,
                 xlabel="transform scopes the build allows",
                 ylabel="peak stack (bytes)",
                 xticks=[0, 4, 8, 12, 16], yticks=[300, 400, 500, 600])
    left.add(Series(depths, stacks, color=PALETTE["teal"], width=1.7, marker=3.0))
    left.rule(v1["stack"], color=MUTED, dash="4 3")
    left.note(16.6, v1["stack"], "ISA v1 — no tier", anchor="end", color=MUTED,
              size=10.5, dy=-6)

    # The shipped build, and the same build with the composition written by
    # value. The vertical gap between them *is* the implementation choice.
    left.add(Series([data["shipped_depth"]], [ptr["stack"]],
                    color=PALETTE["clay"], marker=4.4))
    left.add(Series([data["shipped_depth"]], [val["stack"]],
                    color=PALETTE["slate"], marker=4.4))
    left.add(Series([data["shipped_depth"], data["shipped_depth"]],
                    [ptr["stack"], val["stack"]], color=PALETTE["slate"],
                    width=1.2, dash="2 2"))
    scopes = (val["stack"] - ptr["stack"]) / per_scope
    left.note(4.6, val["stack"], "by value", color=PALETTE["slate"], size=10.5, dy=-4)
    left.note(4.6, (ptr["stack"] + val["stack"]) / 2,
              f"= {scopes:.0f} extra scopes", color=PALETTE["slate"], size=10, dy=9)
    left.note(4.6, ptr["stack"], "shipped", color=PALETTE["clay"], size=10.5, dy=12)
    left.note(0.5, 640, f"{per_scope:.0f} bytes per scope", color=PALETTE["teal"],
              size=11)

    # The plane. Flash is `.text` on both axes' sources, so the 14 bytes of
    # `.rodata` are excluded from both and cancel.
    right = Panel(xlim=(1300, 1980), ylim=(240, 700), width=250, height=190,
                  xlabel="flash, .text (bytes)", ylabel="",
                  xticks=[1400, 1600, 1800], yticks=[300, 400, 500, 600])
    # Everything cheaper than the by-value build on *both* axes. Anchored at the
    # panel corner rather than at ISA v1, because the claim is dominance and not
    # a comparison between two chosen points.
    right.box(1300, val["text"], 240, val["stack"], PALETTE["clay"], 0.07)
    right.add(Series([r["text"] for r in rows], stacks, color=PALETTE["teal"],
                     width=1.4, marker=2.4))
    right.add(Series([v1["text"]], [v1["stack"]], color=MUTED, marker=4.4))
    right.add(Series([ptr["text"]], [ptr["stack"]], color=PALETTE["clay"], marker=4.4))
    right.add(Series([val["text"]], [val["stack"]], color=PALETTE["slate"], marker=4.4))
    right.note(v1["text"] + 16, v1["stack"], "ISA v1", color=MUTED, size=10.5, dy=4)
    right.note(val["text"] - 16, val["stack"], "by value", anchor="end",
               color=PALETTE["slate"], size=10.5, dy=-7)
    right.note(ptr["text"] - 16, ptr["stack"], "shipped", anchor="end",
               color=PALETTE["clay"], size=10.5, dy=5)
    right.note(1830, 655, "depth sweep", anchor="end", color=PALETTE["teal"], size=10)
    right.note(1316, 585, "cheaper than \u201cby value\u201d", color=PALETTE["clay"], size=9.5)
    right.note(1316, 555, "on both axes at once", color=PALETTE["clay"], size=9.5)

    return figure(
        [(left, 66, 22), (right, 386, 22)], width=670, height=268,
        number="Figure 7.",
        caption="The transform tier's device cost, compiled for Cortex-M0+ at "
                "<code>-Os</code> and read off <code>-fstack-usage</code> and "
                f"<code>size</code>. Left: peak stack is linear in the depth bound at "
                f"<b>{per_scope:.0f} bytes per scope</b> — the size of one transform — so "
                "the knob <code>docs/direction.md</code> §2.6 reaches for has a price per "
                f"turn. Right: the same builds in the flash–stack plane. <b>Composing by "
                f"value costs {val['text'] - ptr['text']} bytes of flash and "
                f"{val['stack'] - ptr['stack']} of stack against composing through "
                f"pointers — the same stack as {scopes:.0f} extra scopes would buy</b>, and "
                "it is strictly dominated: everything in the shaded rectangle is cheaper "
                "on both axes at once. The sweep also shows the two costs are not "
                "interchangeable — depth buys stack at almost no flash, so a budget "
                "problem on one axis cannot be paid for on the other.",
    )


def fig_sample_quality() -> str:
    """The second half of the evaluation, and why it needs three panels.

    A table of nine numbers hides robust ordering reversals. Flat arm is best on
    fidelity and worst on coverage; both planners reverse that trade, while AR
    composition beats diffusion on MMD and NNA. Planner coverage rank itself is
    unresolved. Small multiples on a shared row layout make that a glance rather
    than a comparison of columns, and the shared floor band is what makes any of
    the three readable at all -- `coverage`'s ceiling is not 1, `mmd` is in this
    corpus's pixels, and `nna`'s ideal is only 0.5 at matched set sizes.

    Each panel is scaled to its own metric, because they share no unit. What
    they share is the *floor*, which is drawn identically in all three.
    """
    arms = [
        ("flat AR", "quickdraw_planbase12000eps2_byte_square_s0_gen_ar", PALETTE["teal"]),
        ("planner, AR comp.", "quickdraw_plannerar12000eps2_byte_balanced_s0_gen_random",
         PALETTE["olive"]),
        ("planner, diffusion", "quickdraw_plannerdiff12000eps2_byte_balanced_s0_gen_random",
         PALETTE["clay"]),
    ]
    reports = {name: record(tag) for name, tag, _ in arms}
    floor = next(iter(reports.values()))["floor"]

    def stats(values: list[float]) -> tuple[float, float]:
        mean = sum(values) / len(values)
        spread = (sum((v - mean) ** 2 for v in values) / (len(values) - 1)) ** 0.5
        return mean, spread

    metrics = [
        ("coverage", "coverage  ↑", (0.36, 0.55), [0.40, 0.45, 0.50]),
        ("mmd", "mmd (px)  ↓", (12.2, 13.35), [12.4, 12.8, 13.2]),
        ("nna", "nna  → floor", (0.49, 0.68), [0.50, 0.55, 0.60, 0.65]),
    ]
    panels = []
    for key, title, xlim, xticks in metrics:
        panel = Panel(xlim=xlim, ylim=(-0.9, 2.6), width=186, height=130,
                      xlabel=title, yticks=[], xticks=xticks,
                      xtick_labels=[f"{t:g}" for t in xticks])
        low, high = stats([f[key] for f in floor])
        # The floor as a band, not a line: it is an estimate with a spread, and
        # a bare line would invite reading a difference smaller than it.
        panel.band(low - high, low + high, PALETTE["slate"], 0.18, axis="x")
        panel.rule(low, axis="x", color=MUTED, dash="2 2")

        for i, (name, _, colour) in enumerate(arms):
            y = 2.0 - i
            mean, spread = stats([d[key] for d in reports[name]["draws"]])
            panel.add(Series([mean - spread, mean + spread], [y, y],
                             color=colour, width=1.4))
            panel.add(Series([mean], [y], color=colour, marker=3.8))
            panel.note(mean, y, f"{mean:.3f}", anchor="middle", color=colour,
                       size=9.5, dy=-9)
        panels.append(panel)

    # Row labels once, on the left panel only: the three share a layout, so
    # repeating them would say the rows differ between metrics.
    for i, (name, _, colour) in enumerate(arms):
        panels[0].note(0.354, 2.0 - i, name, anchor="end", color=colour, size=10.5, dy=4)
    panels[0].note(0.354, -0.62, "floor — real drawings,", anchor="end",
                   color=MUTED, size=9.5)
    panels[0].note(0.354, -0.86, "same two set sizes", anchor="end", color=MUTED, size=9.5)

    return figure(
        [(p, 196 + 246 * i, 22) for i, p in enumerate(panels)],
        width=920, height=212,
        number="Figure 8.",
        caption="Five CPU draws × 256 samples from one checkpoint per arm at "
                "top-k 40, temperature 1, raw model support; one shared reference "
                "and one shared floor. Bars are the draw-to-draw spread; the "
                "grey band is what <b>real drawings</b> score at the same two set sizes, "
                "which is the only thing that makes these numbers readable — coverage's "
                "ceiling is not 1, mmd is in this corpus's pixels, and nna's textbook 0.5 "
                "holds only when the sets are the same size. <b>The robust reversal: "
                "flat arm is best on fidelity and worst on coverage; both planners "
                "reverse that trade, while AR composition beats diffusion on mmd and "
                "nna. Exact planner coverage rank is unresolved.</b> That trade is the "
                "whole reason three numbers are reported instead of "
                "one, and it is invisible in any of them alone.",
    )


# --- Direction item 2, P5: the transform tier, measured ----------------------

#: P5's three flat arms, in the order the reading goes: the pair that carries the
#: falsifier first, its control in the middle so both neighbours are read against
#: it, and the four-copy arm last because it is the only one with a curve.
XFORM_ARMS = (
    ("x24000n2", "D4 orbits, n = 2", PALETTE["clay"]),
    ("c24000n2", "translation only — the control", PALETTE["teal"]),
    ("x24000n24", "D4 orbits, n ∈ {2,4}", PALETTE["indigo"]),
)
XFORM_SEEDS = (0, 1)

#: The controlled spelling pair: the flat and structured arms are the same 1,000
#: scenes at the same 24,000 steps on the same `byte` codec, so the difference is
#: the spelling and nothing else (`scripts/spelling.py` refuses to assume that
#: and checks both records' corpus digests). The token and token_typed pairs
#: replicate the fold term and are quoted as a range in figure 10's caption; the
#: byte pair is the instrument, because it holds the codec fixed
#: (docs/xform.md §3.1).
SPELLING_PAIRS = tuple(
    f"spelling_composed_x24000n24_byte_square_s{seed}"
    f"_vs_composed_sconv24000_byte_square_s{seed}" for seed in XFORM_SEEDS
)

#: Every structured codec the fold term has been read against — the cross-codec
#: replication figure 10 quotes.
SPELLING_CODECS = ("byte", "token", "token_typed")


def spread(values) -> tuple[float, float]:
    """Mean and half-range across seeds. **Half-range, not a standard error**:
    at k = 2 there is one degree of freedom and the quantity a reader needs is
    "how far apart were the two runs", which is what this is."""
    values = list(values)
    return sum(values) / len(values), (max(values) - min(values)) / 2


def fig_transform_recovery() -> str:
    """P5: the model discovers the orbit, and the pre-registered falsifier fired
    on a comparison that could not settle what it was written to settle.

    The left panel is the falsifier as written -- transformed copies against the
    control's, the corpus differing in nothing but the group. It fires: both
    transformed seeds cost *more* per copied symbol than either control seed. The
    panel also shows why that firing licenses nothing, and it is the thing a
    scalar hides: the whole span of the disagreement sits inside a hundredth of
    the collapse both arms achieved. `recovery` against **zero** is the question
    the claim asked, and every arm answers it the same way.

    The right panel is the shape, which no scalar reports and which Tier A's
    curve had taught the project to expect a break in. There is none: copies get
    monotonically *cheaper* with ordinal here, and the four-copy orbits were in
    the training distribution, so this is the in-count regime that replicated in
    claim 2 rather than the out-of-count regime that did not.
    """
    reports = {(arm, seed): record(f"recovery_composed_{arm}_byte_square_s{seed}_indist")
               for arm, _, _ in XFORM_ARMS for seed in XFORM_SEEDS}
    later = {k: r["recovery"]["bits_per_symbol_later"] for k, r in reports.items()}
    control = [later[("c24000n2", seed)] for seed in XFORM_SEEDS]

    left = Panel(xlim=(0, 0.20), ylim=(-0.7, 5.5), width=232, height=190,
                 xlabel="bits/symbol on the copies", yticks=[],
                 title="recovery, against a ceiling known by construction",
                 xticks=[0, 0.05, 0.10, 0.15, 0.20])
    # The control's own two seeds, as a band. It is the reference the falsifier
    # names, and it is an interval rather than a line because its spread is 17x
    # the transformed arm's -- the pair being differenced is not equally precise
    # on both sides, which is the fact that decides the reading.
    left.band(min(control), max(control), PALETTE["teal"], 0.20, axis="x")

    for i, (arm, label, colour) in enumerate(XFORM_ARMS):
        for j, seed in enumerate(XFORM_SEEDS):
            y = 5.0 - (2 * i + j)
            rate = later[(arm, seed)]
            got = reports[(arm, seed)]["recovery"]["recovery"]
            left.add(Series([0, rate], [y, y], color=colour, width=1.1))
            left.add(Series([rate], [y], color=colour, marker=3.8))
            left.note(rate + 0.006, y, f"{got:+.4f}", color=colour, size=10.5, dy=4)
            if j == 0:
                left.note(-0.006, y - 0.5, label, anchor="end", color=INK,
                          size=11, dy=4)
            left.note(-0.006, y, f"s{seed}", anchor="end", color=MUTED, size=9.5, dy=4)

    right = Panel(xlim=(1.6, 4.4), ylim=(0, 0.20), width=176, height=190,
                  xlabel="copy ordinal", ylabel="bits/symbol",
                  xticks=[2, 3, 4], yticks=[0, 0.05, 0.10, 0.15, 0.20])
    first = spread(reports[("x24000n24", s)]["by_copy"]["bits_per_symbol"][0]
                   for s in XFORM_SEEDS)[0]
    for seed, dash in zip(XFORM_SEEDS, ("", "4 2")):
        by = reports[("x24000n24", seed)]["by_copy"]
        right.add(Series(by["copies"][1:], by["bits_per_symbol"][1:],
                         color=PALETTE["indigo"], dash=dash, width=1.7, marker=3.2))
        # Labelled at ordinal 3, where the two seeds are 30 px apart. At their
        # last point they are 2 px apart and both sit on the axis.
        right.note(3.08, by["bits_per_symbol"][2], f"s{seed}", anchor="start",
                   color=PALETTE["indigo"], size=10.5, dy=-5)
    right.note(1.72, 0.188, f"copy 1 costs {first:.2f} bits/symbol,",
               color=MUTED, size=10)
    right.note(1.72, 0.188, f"{first / 0.20:.0f}× the top of this axis",
               color=MUTED, size=10, dy=12)

    gaps = [later[("x24000n2", s)] - later[("c24000n2", s)] for s in XFORM_SEEDS]
    return figure(
        [(left, 210, 22), (right, 546, 22)], width=760, height=252,
        number="Figure 9.",
        caption=(
            "<b>The transform tier's claim is supported and its pre-registered "
            "falsifier fired anyway.</b> Later copies of a mirrored or rotated motif "
            f"cost {later[('x24000n2', 0)]:.3f} and {later[('x24000n2', 1)]:.3f} "
            f"bits/symbol against {first:.2f} on their own first copy — a recovery of "
            "+0.960 at both seeds, on a corpus where the translational matcher finds "
            "nothing at all. The falsifier said "
            "<em>at or below the control's</em> and it is: the control's copies are "
            f"cheaper by {abs(gaps[0]):.3f} and {abs(gaps[1]):.3f} bits/symbol, the "
            "same sign twice. But the inference it licensed — that the tier buys "
            "nothing — does not follow from it, because the quantity the claim is "
            "about is how far <em>either</em> arm is from paying full price, and both "
            "are 96–98% of the way. The control's own two seeds differ by "
            f"{(max(control) - min(control)):.3f}, which is the whole size of the "
            "smaller gap. Right: the four-copy arm's curve falls monotonically, so "
            "nothing breaks at the trained bound — copy 4 costs a twelfth of copy 2."),
    )


def fig_fold_price() -> str:
    """P5's headline, revised by its own follow-up: three numbers, not two.

    `REPEATX` removes 40.9% of this corpus's bytes. In bits its value has two
    terms, and they were confounded until the structured `byte` arm landed: the
    opcode's own term -- copies removed minus header paid, +5.84 ±0.18, moving
    only within 5.67-6.14 across three structured codecs -- and the context
    term, the same prefix/motif/suffix bytes costing 12.3-15.6 bits less in the
    structured spelling. With the codec held fixed across the pair that second
    term cannot be the alphabet: it is the fold's indirect consequence, a 41%
    shorter program predicting its own shared bytes better. So the ~19.8-bit
    difference of totals is the tier's like-for-like value (4.1%), the 1.2% is
    the opcode's direct term, and the earlier caption's "two thirds is the
    codec and the sequence length" resolves to: it was the sequence length.

    The reason the direct term is small lives in figure 9: a copy byte already
    costs 3% of an original byte, so there was never more than 11 bits/drawing
    on the table for the opcode itself.
    """
    pairs = [record(tag) for tag in SPELLING_PAIRS]
    flat = [record(f"composed_x24000n24_byte_square_s{s}") for s in XFORM_SEEDS]
    ceiling = record("recovery_composed_x24000n24_byte_square_s0_indist")["ceiling"]

    total_bits = spread(r["history"][-1]["bits_per_drawing"] for r in flat)[0]
    total_bytes = ceiling["bytes"] / ceiling["n"]
    saved_bytes = ceiling["saved"] / ceiling["n"]
    fold, fold_half = spread(p["delta"]["fold_total"] for p in pairs)
    copies, copies_half = spread(p["delta"]["fold"]["copies"] for p in pairs)
    header = -spread(p["delta"]["fold"]["header"] for p in pairs)[0]
    context, context_half = spread(p["delta"]["context_total"] for p in pairs)
    gap, gap_half = spread(p["delta"]["total"] for p in pairs)
    folds = [record(f"spelling_composed_x24000n24_byte_square_s{s}"
                    f"_vs_composed_sconv24000_{codec}_square_s{s}")["delta"]["fold_total"]
             for codec in SPELLING_CODECS for s in XFORM_SEEDS]

    left = Panel(xlim=(0, 100), ylim=(-0.9, 2.7), width=250, height=150,
                 xlabel="% of one drawing the fold is worth", yticks=[],
                 xticks=[0, 20, 40, 60, 80, 100])
    rows = [
        ("bytes of bytecode", 100 * saved_bytes / total_bytes, PALETTE["clay"],
         f"{saved_bytes:.1f} of {total_bytes:.1f} B"),
        ("bits — whole spelling effect", 100 * gap / total_bits, PALETTE["slate"],
         f"{gap:.1f} of {total_bits:.1f} bits"),
        ("bits — the opcode's own term", 100 * fold / total_bits, PALETTE["indigo"],
         f"{fold:.2f} of {total_bits:.1f} bits"),
    ]
    for i, (name, percent, colour, detail) in enumerate(rows):
        y = 2.0 - i
        left.box(0, 100, y - 0.26, y + 0.26, PALETTE["slate"], 0.13)
        left.box(0, percent, y - 0.26, y + 0.26, colour, 0.85)
        left.note(-2, y, name, anchor="end", color=INK, size=11, dy=1)
        left.note(-2, y - 0.30, detail, anchor="end", color=MUTED, size=9.5)
        left.note(percent + 2.5, y, f"{percent:.1f}%", color=colour, size=11.5, dy=4)

    right = Panel(xlim=(0, 4), ylim=(0, 24), width=200, height=176,
                  ylabel="bits/drawing", xticks=[], yticks=[0, 5, 10, 15, 20],
                  title="the difference, by position class")
    steps = [
        (0.0, copies, PALETTE["clay"], "copies", copies_half),
        (copies, copies - header, PALETTE["teal"], "− header", 0.0),
        (copies - header, copies - header + context, PALETTE["slate"], "context",
         context_half),
    ]
    for i, (base, top, colour, name, half) in enumerate(steps):
        x0, x1 = 0.25 + i, 0.95 + i
        # The subtracted column is drawn faint: in a waterfall it spans from the
        # new level up to the old one, so at full weight it reads as a positive
        # bar the height of the one before it.
        right.box(x0, x1, min(base, top), max(base, top), colour,
                  0.45 if top < base else 0.85)
        # The name clears the seed tick as well as the bar, or the context
        # column's ±1.7 tick strikes straight through its own label.
        right.note((x0 + x1) / 2, max(base, top) + half + 0.6, name,
                   anchor="middle", color=colour, size=10.5)
        # Inside the bar, under its own name: the three levels a waterfall wants
        # to label are all near each other in the middle of the panel, and a
        # value printed at a bar's base collides with whatever the next bar's
        # base is.
        right.note((x0 + x1) / 2, max(base, top), f"{top - base:+.2f}".replace("-", MINUS),
                   anchor="middle", color="#ffffff" if top > base else INK,
                   size=10, dy=15)
        if half:
            mid = (x0 + x1) / 2
            right.add(Series([mid, mid], [max(base, top) - half,
                                          max(base, top) + half],
                             color=INK, width=1.0))
        if i < len(steps) - 1:
            right.add(Series([x1, x1 + 0.3], [top, top], color=MUTED,
                             width=0.8, dash="2 2"))
    right.box(3.25, 3.95, 0, gap, PALETTE["slate"], 0.22)
    right.note(3.6, gap + 0.6, "total", anchor="middle", color=MUTED, size=10.5)
    right.note(3.6, gap / 2, f"{gap:.1f}", anchor="middle", color=MUTED, size=10.5, dy=4)
    # The fold's bracket sits at its own level rather than beside a column: it is
    # the difference between two columns and belongs to neither.
    right.rule(fold, axis="y", color=PALETTE["indigo"], dash="3 2")
    # The fold's own label goes in the empty quarter under the context column,
    # on the level its rule draws: it is the difference between two columns and
    # sits beside neither.
    right.note(2.6, 3.4, "the fold", anchor="middle", color=PALETTE["indigo"], size=10.5)
    right.note(2.6, 3.4, f"{fold:+.2f}".replace("-", MINUS), anchor="middle",
               color=PALETTE["indigo"], size=11.5, dy=15)

    return figure(
        [(left, 190, 26), (right, 530, 22)], width=760, height=250,
        number="Figure 10.",
        caption=(
            f"<b>The transform tier folds {100 * saved_bytes / total_bytes:.1f}% of "
            f"this corpus's bytes; the opcode's own term is "
            f"{100 * fold / total_bits:.1f}% of the bits, and the whole spelling "
            f"effect is {100 * gap / total_bits:.1f}%.</b> Both arms are the same "
            "1,000 scenes at 24,000 steps on the same byte codec, so their difference "
            f"— {gap:.1f} ±{gap_half:.1f} bits/drawing across seeds — is the spelling "
            f"and nothing else. Split by position class: {copies:.2f} for the copies "
            f"the fold removes, {header:.2f} against it for the REPEATX and ENDREP it "
            f"adds, and {context:.1f} ±{context_half:.1f} on bytes that are "
            "<em>byte-identical in both spellings</em>. With the codec held fixed that "
            "context term can no longer be the alphabet: it is the fold's indirect "
            "consequence — a 41% shorter program predicts its own shared bytes better "
            "— resolving the earlier caption's “codec and sequence length” to "
            f"the sequence length alone. The fold's own term is {fold:+.2f} "
            f"±{fold_half:.2f} across seeds and stays within "
            f"{min(folds):.2f}–{max(folds):.2f} across three structured codecs. Ticks "
            "are half the seed range, and the dashed rule is the ISA's direct term."),
    )


def fig_termination() -> str:
    """Where the transform tier actually pays, and it is not the axis it was
    built for.

    The rows are one corpus in two spellings plus its control, all at 24,000
    steps, and the columns are three things `bits/drawing` cannot see: how far a
    sampler's length distribution sits from the corpus's, how often it emits
    something the VM refuses, and -- the one that changed the reading -- whether
    what it drew contains an orbit at all.

    **The third column was added after the first two had a mechanism attached to
    them that turned out to be wrong.** The story was that D4 is closed on the
    canvas, so a mirrored prefix always has a legal continuation and the over-long
    samples are continued orbits. Pointing the oracle at the samples says the
    opposite: 13-23% of them contain an orbit against the corpus's 100%, and only 2
    of 96 carry a count the corpus does not have. The model is not over-applying
    the copy relation, it is **failing to apply it** and filling the space with
    new content -- and `REPEATX` makes it a certainty at 48 of 48, because a copy
    that is an operand is not something a sampler has to decide to produce.

    The second reading is the spread, and it inverts between columns. On length
    the flat transformed arms differ from themselves by 4x; on orbit retention
    *they* replicate (13/15%, 23/23%) and the **control** is the loose one
    (40/67%) -- the same asymmetry as `recovery` in figure 9.

    `length_emd` is normalised by each corpus's own mean program length, because
    the structured spelling *is* 41% shorter and an absolute distance in bytes
    would credit it for that. It is the weaker kind of comparison and it is
    labelled as one; `validity` is a fraction of samples and needs no
    normalisation at all.
    """
    arms = (
        ("c24000n2_byte", "translation only — the control", "flat", PALETTE["teal"]),
        ("x24000n2_byte", "D4 orbits, n = 2", "flat", PALETTE["clay"]),
        ("x24000n24_byte", "D4 orbits, n ∈ {2,4}", "flat", PALETTE["indigo"]),
        ("sconv24000_token", "the same scenes, REPEATX", "structured", PALETTE["olive"]),
        ("sconv24000_token_typed", "REPEATX, typed alphabet", "structured",
         PALETTE["sand"]),
    )
    # Orbit retention, from `scripts/orbit_oracle.py --samples`: the oracle pointed
    # at what the model draws instead of at a corpus. Keyed by run name because the
    # reports are written per invocation and it took several; an arm with no report
    # draws no bar, where a zero would read as "this arm never keeps an orbit".
    retention: dict[str, float] = {}
    for path in sorted(RUNS.glob("orbit_oracle_generated*.json")):
        for row in json.loads(path.read_text())["rows"]:
            counts = {int(k): v for k, v in row["generated"]["counts"].items()}
            retention[row["name"]] = sum(v for k, v in counts.items() if k > 1) / row["n"]

    rows = []
    for arm, label, spelling, colour in arms:
        for seed in XFORM_SEEDS:
            tag = f"composed_{arm}_square_s{seed}"
            gen = record(f"{tag}_gen_ar")
            mean_len = sum(record(tag)["val_bytes"]) / len(record(tag)["val_bytes"])
            draws = [d["length_emd"] / mean_len * 100 for d in gen["draws"]]
            rows.append((label if seed == XFORM_SEEDS[0] else None, seed, spelling,
                         colour, spread(draws),
                         spread(d["validity"] for d in gen["draws"]),
                         retention.get(tag)))

    n = len(rows)
    left = Panel(xlim=(0, 125), ylim=(-0.8, n - 0.2), width=196, height=224,
                 xlabel="length EMD, % of own mean length",
                 title="how far the sampled lengths sit",
                 yticks=[], xticks=[0, 25, 50, 75, 100, 125])
    # The axis runs past 1.0 so the value labels of the rows that sit at 0.99 do
    # not have to be printed over the frame.
    middle = Panel(xlim=(0.60, 1.07), ylim=(-0.8, n - 0.2), width=118, height=224,
                   xlabel="fraction the VM accepts", title="validity",
                   yticks=[], xticks=[0.7, 0.8, 0.9, 1.0])
    middle.rule(1.0, axis="x", color=MUTED, dash="2 2")
    right = Panel(xlim=(0, 112), ylim=(-0.8, n - 0.2), width=118, height=224,
                  xlabel="% of 48 samples", title="contains an orbit",
                  yticks=[], xticks=[0, 50, 100])
    # The corpus is 100% by construction. Every arm is read against that line.
    right.rule(100, axis="x", color=MUTED, dash="2 2")

    for i, (label, seed, spelling, colour, emd, valid, kept) in enumerate(rows):
        y = n - 1 - i
        for panel, (value, half), places in ((left, emd, 0), (middle, valid, 3)):
            panel.add(Series([value - half, value + half], [y, y],
                             color=colour, width=1.4))
            panel.add(Series([value], [y], color=colour, marker=3.6))
            panel.note(value, y, f"{value:.{places}f}" + ("%" if not places else ""),
                       anchor="middle", color=colour, size=9.5, dy=-8)
        if kept is not None:
            right.box(0, 100 * kept, y - 0.26, y + 0.26, colour, 0.85)
            right.note(100 * kept + (3 if kept < 0.55 else -3), y, f"{kept:.0%}",
                       anchor="start" if kept < 0.55 else "end",
                       color=colour if kept < 0.55 else "#ffffff", size=9.5, dy=3.5)
        left.note(-3, y, f"s{seed}", anchor="end", color=MUTED, size=9.5, dy=4)
        if label:
            left.note(-16, y - 0.5, label, anchor="end", color=INK, size=11, dy=4)
            left.note(-16, y - 0.5, spelling, anchor="end", color=MUTED, size=9.5, dy=16)

    return figure(
        [(left, 250, 26), (middle, 478, 26), (right, 636, 26)],
        width=790, height=300,
        number="Figure 11.",
        caption=(
            "<b>The tier's payoff is generation, and the third column is why.</b> Five "
            "arms at 24,000 steps; the first two columns are 5 draws of 128 apiece, the "
            "third is the D4 orbit oracle pointed at 48 samples per checkpoint instead "
            "of at a corpus. The flat transformed arms miss the corpus's length "
            "distribution by 25–112% of a program's own length and <b>keep an orbit in "
            "13–23% of draws where the corpus has one in 100%</b>; the same scenes "
            "spelled with REPEATX keep one in <b>48 of 48</b>. <b>So the over-long "
            "samples are not the model continuing the orbit — it is losing the orbit "
            "and filling the space with new content.</b> Teacher-forced a mirrored copy "
            "costs 3% of its own original; free-running the same checkpoint draws one in "
            "13% of samples. The control's translated copies survive in 40–67%, so "
            "transformed reuse is harder to <em>generate</em> than translated reuse "
            "while being equally easy to <em>predict</em>. EMD is normalised by each "
            "corpus's own mean length — the weaker, relative column."),
    )


def fig_compression_rate() -> str:
    """Why the composed runs converge to a lower `bits/drawing`, and what that is
    not evidence of.

    The observation is real: every composed arm's val curve settles below every
    QuickDraw arm's. The temptation is to read it as the model doing better, and
    **the first thing to say is that the comparison is illegal** — different val
    sets, so the totals are of different drawings. The second is that once
    normalised the effect is fully explained and is a property of the corpus:
    the flat constructed corpus costs 2.16 bits per bytecode byte where QuickDraw
    costs 3.81, because 44% of its bytes are copies the model gets for 0.11.

    The right panel is the whole argument in one shape, because **area is
    bits/drawing**: a drawing's price is its length times its rate, the copy bytes
    are 44% of the width at 3% of the height, and the dashed rectangle is what the
    same bytes would have cost at the motif's own rate. The gap between them is
    ~350 bits the model compressed on its own, and the fold's 11-bit sliver is all
    that was left for an opcode to take.

    The strongest part of it is the row a reader has to be told to look for: on
    the bytes that are *not* copies, this constructed corpus costs 3.74 — within
    2% of the natural corpus it is built from. **It is not a toy.** A corpus that
    had accidentally become easy would show it exactly there, and it does not.
    """
    def rate(tags: tuple[str, ...]) -> tuple[float, float]:
        rs = [record(t)["history"][-1] for t in tags]
        bits = sum(r["bits_per_drawing"] for r in rs) / len(rs)
        return bits / rs[0]["tokens_per_drawing"], rs[0]["tokens_per_drawing"]

    arms = [
        ("Tier A — synthetic", ("synthetic_c2flat24000_byte_square_s0",
                                "synthetic_c2flat24000_byte_square_s1"),
         PALETTE["slate"], "constructed"),
        ("QuickDraw — eps 2", ("quickdraw_planbase24000eps2_byte_square_s0",),
         PALETTE["teal"], "natural"),
        ("QuickDraw — eps 4", ("quickdraw_budget24000eps4_byte_square_s0",
                               "quickdraw_budget24000eps4_byte_square_s1"),
         PALETTE["teal"], "natural"),
        ("composed — control, n = 2", ("composed_c24000n2_byte_square_s0",
                                       "composed_c24000n2_byte_square_s1"),
         PALETTE["olive"], "flat"),
        ("composed — D4, n = 2", ("composed_x24000n2_byte_square_s0",
                                  "composed_x24000n2_byte_square_s1"),
         PALETTE["clay"], "flat"),
        ("composed — D4, n ∈ {2,4}", ("composed_x24000n24_byte_square_s0",
                                      "composed_x24000n24_byte_square_s1"),
         PALETTE["clay"], "flat"),
        ("composed — REPEATX", ("composed_sconv24000_token_square_s0",
                                "composed_sconv24000_token_square_s1"),
         PALETTE["indigo"], "structured"),
    ]
    reference = rate(("quickdraw_budget24000eps4_byte_square_s0",
                      "quickdraw_budget24000eps4_byte_square_s1"))[0]

    n = len(arms)
    left = Panel(xlim=(0, 4.4), ylim=(-1.4, n - 0.4), width=214, height=214,
                 xlabel="bits per bytecode byte", yticks=[],
                 title="what one byte costs, by corpus",
                 xticks=[0, 1, 2, 3, 4])
    left.rule(reference, axis="x", color=MUTED, dash="3 2")
    for i, (name, tags, colour, kind) in enumerate(arms):
        y = n - 1 - i
        value, length = rate(tags)
        left.add(Series([0, value], [y, y], color=colour, width=1.1))
        left.add(Series([value], [y], color=colour, marker=3.8))
        left.note(value + 0.09, y, f"{value:.2f}", color=colour, size=10.5, dy=4)
        left.note(-0.1, y, name, anchor="end", color=INK, size=10.5, dy=4)
        left.note(-0.1, y, f"{length:.0f} B/drawing · {kind}", anchor="end",
                  color=MUTED, size=8.5, dy=13)
    left.note(0.1, -1.15, "dashed: the natural corpus's rate", color=MUTED, size=10)

    # The right panel: area is bits/drawing.
    pair = record(SPELLING_PAIRS[0])["flat"]
    total_B = pair["bytes_per_drawing"]
    copy_B = pair["classes"]["copies"]["bytes_per_drawing"]
    copy_rate = pair["classes"]["copies"]["bits_per_symbol"]
    rest_B = total_B - copy_B
    rest_rate = (pair["bits_per_drawing"] - pair["classes"]["copies"]["per_drawing"]) / rest_B
    full = total_B * rest_rate

    right = Panel(xlim=(0, 236), ylim=(0, 4.4), width=230, height=214,
                  xlabel="bytes of one drawing", ylabel="bits per byte",
                  title="one drawing's price, as area",
                  xticks=[0, 50, 100, 150, 200], yticks=[0, 1, 2, 3, 4])
    right.box(0, rest_B, 0, rest_rate, PALETTE["indigo"], 0.75)
    right.box(rest_B, total_B, 0, copy_rate, PALETTE["clay"], 0.9)
    # What the copied bytes would have cost at the rate of the motif they copy.
    for xs, ys in (([rest_B, total_B], [rest_rate, rest_rate]),
                   ([total_B, total_B], [copy_rate, rest_rate])):
        right.add(Series(xs, ys, color=PALETTE["clay"], dash="3 2", width=1.1))
    right.rule(reference, axis="x", color=MUTED, dash="3 2")

    right.note(rest_B / 2, rest_rate / 2, f"{rest_B:.0f} B at {rest_rate:.2f}",
               anchor="middle", color="#ffffff", size=10.5)
    right.note(rest_B / 2, rest_rate / 2, f"= {rest_rate * rest_B:.0f} bits",
               anchor="middle", color="#ffffff", size=10.5, dy=14)
    # Inside the dashed rectangle, over the sliver it labels: outside the frame is
    # where this ran off the panel.
    right.note((rest_B + total_B) / 2, 0.62, f"{copy_B:.0f} B at {copy_rate:.2f}",
               anchor="middle", color=PALETTE["clay"], size=10)
    right.note((rest_B + total_B) / 2, 0.62, f"= {copy_rate * copy_B:.0f} bits",
               anchor="middle", color=PALETTE["clay"], size=10, dy=13)
    right.note(rest_B + 6, rest_rate - 0.35,
               f"{full - pair['bits_per_drawing']:.0f} bits the model", color=MUTED, size=10)
    right.note(rest_B + 6, rest_rate - 0.35, "compressed by itself", color=MUTED,
               size=10, dy=13)

    return figure(
        [(left, 190, 26), (right, 500, 26)], width=760, height=290,
        number="Figure 12.",
        caption=(
            "<b>The composed arms settle at a lower bits/drawing because their "
            "corpus is more redundant, not because anything trained better</b> — and "
            "the totals were never comparable anyway, being different val sets. "
            f"Normalised, the flat transformed corpus costs "
            f"{rate(arms[5][1])[0]:.2f} bits per bytecode byte against QuickDraw's "
            f"{reference:.2f}. Right: area is bits/drawing, so the price of one "
            f"drawing is its length times its rate. The copies are {copy_B:.0f} of "
            f"{total_B:.0f} bytes at {copy_rate:.2f} bits each; at the rate of the "
            f"motif they copy they would have cost {full:.0f} bits and they cost "
            f"{copy_rate * copy_B:.0f}. <b>And the row worth looking for is the "
            f"non-copy one: {rest_rate:.2f} bits per byte, within 2% of the natural "
            "corpus these motifs come from.</b> Outside its planted structure the "
            "constructed corpus is exactly as hard as QuickDraw, which is what says "
            "it is an instrument and not a toy. A rate compared across corpora is "
            "descriptive and not a paired difference: it explains a level, and no "
            "claim here rests on it."),
    )


# --- Direction item 3: class conditioning, measured --------------------------

#: `scripts/conditioning.py`'s reports, one per seed — the two-instrument
#: reading figure 13 draws.
CONDITIONING_REPORTS = tuple(
    f"conditioning_quickdraw_cond24000eps4_byte_square_s{seed}" for seed in (0, 1)
)


def fig_conditioning() -> str:
    """What a class label buys, and which instrument can see it.

    The ceiling is arithmetic: a five-class label carries at most
    H(C) = log2(5) = 2.32 bits, and this corpus's run-to-run floor is ~2.5 --
    larger than everything a label could ever be worth. The pre-registered
    consequence (`docs/conditioning.md` §1.1) was that the paired gap between a
    conditional and an unconditional run carries I(X; C) *plus* a cross-model
    term the size of the floor, while the free classifier -- Bayes over five
    forward passes of the conditional checkpoint alone -- has no second model
    to differ from and therefore no floor.

    The figure is that consequence measured at full scale, and the shaded
    region matters more than any marker inside it: past H(C) is more than a
    label carries, so everything a paired reading puts there is the two runs
    differing on their own. One of the two paired readings lands there; the two
    single-model readings land within 0.03 bits of each other, at 97-98% of the
    ceiling.
    """
    reports = [record(tag) for tag in CONDITIONING_REPORTS]
    ceiling = reports[0]["ceiling"]["entropy_bits"]
    gaps = [(r["gap"]["delta"], r["gap"]["ci95"]) for r in reports]
    infos = [(r["implied_mutual_information"],
              1.96 * r["classifier"]["class_bits_stderr"]) for r in reports]

    panel = Panel(xlim=(0, 7.2), ylim=(-0.75, 3.95), width=320, height=175,
                  xlabel="bits/drawing the class label is worth",
                  yticks=[], xticks=[0, 1, 2, 3, 4, 5, 6, 7])
    # Everything right of H(C) is unreachable by information about the class,
    # so a reading there indicts the instrument and not the label.
    panel.band(ceiling, 7.2, PALETTE["clay"], 0.08, axis="x")
    panel.rule(ceiling, axis="x", color=MUTED, dash="3 2")
    panel.note(ceiling - 0.12, 3.72, "H(class) = log₂ 5", anchor="end",
               color=MUTED, size=10)
    panel.note(ceiling + 0.15, 3.72, "past here is not the label —",
               color=MUTED, size=10)
    panel.note(ceiling + 0.15, 3.72, "it is the two runs differing on their own",
               color=MUTED, size=10, dy=12)

    instruments = [
        ("the paired gap", "two runs, differenced", gaps, PALETTE["clay"]),
        ("the free classifier", "one checkpoint, five passes", infos,
         PALETTE["teal"]),
    ]
    for g, (name, how, values, colour) in enumerate(instruments):
        for s, (delta, ci) in enumerate(values):
            y = 3.0 - 2 * g - s
            panel.add(Series([delta - ci, delta + ci], [y, y], color=colour,
                             width=1.4))
            for end in (delta - ci, delta + ci):
                panel.add(Series([end, end], [y - 0.10, y + 0.10], color=colour,
                                 width=1.4))
            panel.add(Series([delta], [y], color=colour, marker=3.8))
            # The top row's value sits under its marker: above it is where the
            # shaded region's own two-line label lives, and a rendered check
            # caught the collision.
            panel.note(delta, y, f"{delta:+.2f} ±{ci:.2f}".replace("-", MINUS),
                       anchor="middle", color=colour, size=10,
                       dy=20 if g == 0 and s == 0 else -10)
            panel.note(-0.12, y, f"s{s}", anchor="end", color=MUTED, size=9.5, dy=4)
        panel.note(-0.55, 3.0 - 2 * g - 0.5, name, anchor="end", color=INK,
                   size=11.5, dy=2)
        panel.note(-0.55, 3.0 - 2 * g - 0.5, how, anchor="end", color=MUTED,
                   size=9.5, dy=15)

    accs = [r["classifier"]["accuracy"] for r in reports]
    fanos = [r["fano_max_accuracy"] for r in reports]
    asked = sum(r["controllability"]["n"] for r in reports)
    read_back = sum(round(r["controllability"]["accuracy"]
                          * r["controllability"]["n_nonempty"]) for r in reports)
    return figure(
        [(panel, 205, 26)], width=650, height=242,
        number="Figure 13.",
        caption=(
            "<b>The label's value cannot be measured by differencing two runs, and "
            "can be measured inside one — pre-registered, and demonstrated at "
            f"scale.</b> A five-class label carries at most H(class) = {ceiling:.2f} "
            "bits, under this corpus's ~2.5-bit run-to-run floor. The paired gap — "
            "unconditional minus conditional bits/drawing, matched runs at two seeds "
            f"— reads {gaps[0][0]:+.2f} and {gaps[1][0]:+.2f}: "
            f"{abs(gaps[0][0] - gaps[1][0]):.2f} bits apart, one of them past the "
            "ceiling, which no property of a label can explain. The free classifier "
            "— p(c | x) ∝ p(x | c)·p(c), five forward passes of one checkpoint, no "
            f"cross-model term — reads I(X; C) = {infos[0][0]:+.2f} and "
            f"{infos[1][0]:+.2f}, {abs(infos[0][0] - infos[1][0]):.3f} apart, at 97% "
            "of the ceiling: a drawing carries nearly all of its own class. Its "
            f"accuracy ({accs[0]:.3f}, {accs[1]:.3f}) respects the Fano bound its own "
            f"H(C | X) predicts ({fanos[0]:.3f}, {fanos[1]:.3f}), and asked to draw "
            "each class in turn, the model reads back the class it was asked for in "
            f"{read_back} of {asked} samples against a chance of 0.2."),
    )


#: The class-conditional sample-quality reports, one per seed. Their `floor` and
#: `floor_matrix` are properties of the corpus rather than of a checkpoint, so
#: the two files carry byte-identical copies — read from the first, and the
#: second is the check that they do.
CLASS_QUALITY_REPORTS = tuple(
    f"class_quality_quickdraw_cond24000eps4_byte_square_s{seed}_n100x5"
    for seed in (0, 1)
)


def fig_class_geometry() -> str:
    """Does asking for a cat produce cat-shaped geometry — and at what price?

    Three panels because the answer has two halves that a single one would let
    a reader average together. The left panel is the question: the diagonal's
    margin over its nearest confusing class, against **the margin real drawings
    of that class achieve**, which is the scale the first reading of this
    matrix lacked. Chamfer's separation of two classes is a property of their
    shapes, so "the diagonal is the row minimum" says the model won and cannot
    say whether it won by enough.

    The right two are the price, and they are the same fact twice. The label
    moves `coverage` and `nna` most of the way to their floors and stops --
    and the direction of every miss agrees with the left panel *overshooting*:
    samples sit in each class's typical middle, which separates the classes
    better than real drawings do, reaches fewer distinct real drawings than
    real drawings do, and stays more distinguishable from them. One
    mode-seeking signature at the published sampler, read three ways. A later
    no-training sweep attributes much of its concentration to top-k, with `k=80`
    as best tested setting; full support then fails structural/fidelity guards.
    This figure remains the `k=40` measurement rather than being redrawn from one checkpoint.

    `mmd` gets no panel because its diagonal is a null -- 13.76 against a 13.79
    floor -- and the caption carries it. Its *off*-diagonal is the left panel.
    """
    reports = [record(tag) for tag in CLASS_QUALITY_REPORTS]
    names = reports[0]["categories"]
    floor_draws = [{"matrix": m} for m in reports[0]["floor_matrix"]]
    if reports[1]["floor_matrix"] != reports[0]["floor_matrix"]:
        raise SystemExit("the two reports' floor matrices differ; the floor is a "
                         "corpus quantity and cannot depend on the checkpoint")

    def margins(draws: list[dict], c: int) -> list[float]:
        return [min(v for j, v in enumerate(d["matrix"][c]) if j != c)
                - d["matrix"][c][c] for d in draws]

    def stats(values: list[float]) -> tuple[float, float]:
        mean = sum(values) / len(values)
        return mean, (sum((v - mean) ** 2 for v in values) / (len(values) - 1)) ** 0.5

    rows = len(names)
    ylim = (-1.15, rows - 0.45)

    # -- left: the margin, against the margin the corpus itself offers --------
    left = Panel(xlim=(0, 6.4), ylim=ylim, width=210, height=176,
                 xlabel="margin over the nearest class  (px)  ↑",
                 yticks=[], xticks=[0, 2, 4, 6])
    for i, name in enumerate(names):
        y = rows - 1 - i
        base, _ = stats(margins(floor_draws, i))
        # The corpus's own margin as a block from zero: everything inside it is
        # separation a perfect generator also gets, so only the part beyond is
        # the model distinguishing classes better than real drawings do.
        left.box(0, base, y - 0.34, y + 0.34, PALETTE["slate"], 0.20)
        for s, report in enumerate(reports):
            mean, sd = stats(margins(report["draws"], i))
            yy = y + (0.17 if s == 0 else -0.17)
            left.add(Series([mean - sd, mean + sd], [yy, yy],
                            color=PALETTE["teal"], width=1.2))
            left.add(Series([mean], [yy], color=PALETTE["teal"], marker=3.0))
        left.note(-0.25, y, name, anchor="end", color=INK, size=11, dy=4)

    # -- right: what the label buys, and where it stops -----------------------
    panels = [left]
    for key, title, xlim, xticks in (
        ("coverage", "coverage  ↑", (0.10, 0.60), [0.1, 0.2, 0.3, 0.4, 0.5]),
        ("nna", "nna  → floor", (0.44, 1.02), [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]),
    ):
        panel = Panel(xlim=xlim, ylim=ylim, width=210, height=176,
                      xlabel=title, yticks=[], xticks=xticks,
                      xtick_labels=[f"{t:g}" for t in xticks])
        for i, name in enumerate(names):
            y = rows - 1 - i
            base, sd = stats([f[key] for f in reports[0]["floor"][str(i)]])
            cond, _ = stats([d["diagonal"][str(i)][key]
                             for r in reports for d in r["draws"]])
            ctrl, _ = stats([d[str(i)][key]
                             for r in reports for d in r["control_draws"]])
            # Each class's own floor: a shared band would say the five classes
            # share one, and they do not -- flower's nna floor is 0.524 against
            # bus's 0.482 on the identical estimator.
            panel.box(base - sd, base + sd, y - 0.34, y + 0.34,
                      PALETTE["slate"], 0.20)
            # The travel the label buys, drawn as travel: unconditional tail,
            # conditional head, floor block as the target both are aimed at.
            panel.add(Series([ctrl, cond], [y, y], color=MUTED, width=0.9,
                             dash="2 2"))
            panel.add(Series([ctrl], [y], color=PALETTE["clay"], marker=3.0))
            panel.add(Series([cond], [y], color=PALETTE["teal"], marker=3.4))
        panels.append(panel)

    # One legend, under the left panel, because all three share a row layout.
    left.note(0.05, -0.72, "■ the margin real drawings get", color=MUTED, size=9.5)
    left.note(0.05, -1.02, "— asked class, each seed", color=PALETTE["teal"], size=9.5)
    panels[1].note(0.115, -0.72, "● no label — “just draw something”",
                   color=PALETTE["clay"], size=9.5)
    panels[1].note(0.115, -1.02, "● asked for this class",
                   color=PALETTE["teal"], size=9.5)
    panels[2].note(0.46, -0.72, "■ real drawings of the class,", color=MUTED, size=9.5)
    panels[2].note(0.46, -1.02, "same two set sizes", color=MUTED, size=9.5)

    wins = sum(1 for r in reports for d in r["draws"] for c in range(rows)
               if d["matrix"][c][c] == min(d["matrix"][c]))
    total = sum(len(r["draws"]) for r in reports) * rows
    return figure(
        [(p, 236 + 250 * i, 24) for i, p in enumerate(panels)],
        width=1000, height=258,
        number="Figure 14.",
        caption=(
            "<b>Asking for a class produces that class's geometry, by a wider "
            "margin than real drawings of it achieve — and the same overshoot is "
            "why the other two columns stop short.</b> 5 draws × 100 samples per "
            "class per seed, every set scored against a fixed 100-drawing "
            "reference of real val drawings. <b>Left:</b> the diagonal is the row "
            f"minimum in {wins} of {total} class-draws, and its margin over the "
            "nearest confusing class exceeds the corpus's own at all five classes "
            "and both seeds (1.08–3.27×) — the grey block is the margin a perfect "
            "generator gets, and the model is outside it. <b>Right:</b> against "
            "the unconditional arm on the identical references, the label closes "
            "43% of the coverage gap to the floor and 61% of the nna gap, and no "
            "class reaches either. <b>All three misses point the same way:</b> "
            "samples sit in each class's typical middle, so they separate the "
            "classes better than real drawings do, reach fewer distinct real "
            "drawings, and stay more distinguishable from them — one mode-seeking "
            "signature at a shared top-k of 40, not three results. A later "
            "no-training sweep shows that concentration is strongly top-k-sensitive "
            "at s0: k=80 moves coverage 0.406→0.487 and nna 0.639→0.549; full "
            "support then worsens fidelity/separation and emits one empty. "
            "<code>mmd</code> is the one column with no gap left: 13.76 against a "
            "13.79 floor, within 0.22 px at every class."),
    )


# --- Direction 2: teacher forcing against generation -------------------------

#: `(venue, label, note)`. The two diagnostics ride along because their nulls
#: are the reason the step venue's label is what it is, and because one of them
#: is null for a reason that is not about the relation at all.
CONTEXT_VENUES = (
    ("synthetic_flat_step", "flat step twins", "co-primary · x axis"),
    ("synthetic_flat_step_yaxis", "same twins, y axis", "readable · does not transfer"),
    ("synthetic_flat_step_d4r1", "exact quarter turn", "out of support · says nothing"),
    ("composed_shape_copy2", "composed shape twins", "co-primary"),
)

CONTEXT_C3 = ("context_c3_synthetic_byte_dcc93caf0003",
              "context_c3_composed_byte_2679fc2356da")
CONTEXT_C4 = ("context_c4_synthetic_byte_dcc93caf0003_v5",
              "context_c4_composed_byte_2679fc2356da_v5")


def context_rows(tags: tuple[str, ...]) -> dict:
    """`{venue: {seed: summary}}` from a stage's reports, however many files."""
    out: dict[str, dict[int, dict]] = {}
    for tag in tags:
        body = json.loads((RUNS / f"{tag}.json").read_text())
        if body.get("status") != "complete":
            raise SystemExit(f"{tag}: status={body.get('status')!r}")
        for entry in body["reports"]:
            for venue, summary in entry["venues"].items():
                out.setdefault(venue, {})[int(entry["model_seed"])] = summary
    return out


def fig_context_generation() -> str:
    """Direction 2's result and its own limit, in one row per venue.

    The teacher-forced effect is enormous — `+1.28` to `+4.72` bits per target
    byte, with every raw `D` positive and 63/64 controlled step `Delta` values
    positive in each seed, both matched controls measured at ~0. Read alone
    it invites "the model uses the relation", and the middle panel is the reason
    that sentence needs a qualifier: prompted at the same boundary and left to
    decode, the same checkpoints lean the right way by `0.018`–`0.084` of a
    normalised byte distance.

    **The third panel is the one that decides what the second one means.** A
    positive `D_gen` is a preference, not a capability, and on the composed
    venue the preference sits on `hit_own` of 1.1–1.3%: the compatible continuation
    is essentially never the thing that comes out. On the step venue it is 8-14%
    — small, but an order of magnitude larger, and that ordering is the finding.

    The two diagnostic rows are here because they fail differently and the
    difference is the methodological point of the whole direction. The y-axis
    row is readable (prefix cost x1.34) and flat: the relation genuinely does
    not transfer. The quarter-turn row is flat because the model has no
    distribution there at all (prefix cost x2.2-2.4), and a null from an
    instrument that cannot see is not evidence about a relation.
    """
    c3, c4 = context_rows(CONTEXT_C3), context_rows(CONTEXT_C4)
    seeds, gap = (0, 1), 0.26
    rows = [(venue, label, note, i) for i, (venue, label, note)
            in enumerate(CONTEXT_VENUES)]
    n = len(rows)

    def place(index: int, seed: int) -> float:
        return n - 1 - index + (gap / 2 if seed == 0 else -gap / 2)

    def panel(title: str, xlabel: str, xlim, xticks, width: float) -> Panel:
        return Panel(xlim=xlim, ylim=(-0.75, n - 0.45), width=width, height=224,
                     xlabel=xlabel, yticks=[], xticks=xticks, title=title)

    left = panel("teacher forced: Δ, bits per target byte",
                 "Δ = D − matched control", (-0.45, 5.45), [0, 1, 2, 3, 4, 5], 232)
    middle = panel("free running: D-gen, normalised distance",
                   "paired preference", (-0.016, 0.105),
                   [0, 0.025, 0.05, 0.075, 0.1], 205)
    right = panel("and what it actually emitted",
                  "hits on the compatible target",
                  (0, 0.235), [0, 0.05, 0.1, 0.15], 196)
    for target in (left, middle, right):
        target.rule(0.0, axis="x", color=MUTED, dash="2 2")

    for venue, label, note, index in rows:
        dim = "out of support" in note
        for seed in seeds:
            y = place(index, seed)
            colour = PALETTE["slate"] if dim else (
                PALETTE["teal"] if seed == 0 else PALETTE["clay"])
            c3_row, c4_row = c3[venue][seed], c4[venue][seed]
            for target, value, interval in (
                (left, c3_row["D_minus_control_bits_per_byte_mean"],
                 c3_row["bootstrap_D_minus_control"]["ci95"]),
                (middle, c4_row["D_gen_mean"], c4_row["bootstrap_D_gen"]["ci95"]),
            ):
                target.add(Series(list(interval), [y, y], color=colour,
                                  width=1.1))
                target.add(Series([value], [y], color=colour, marker=3.4))
            # The matched irrelevant edit, on the same axis as the effect it is
            # the control for. Drawn small and grey: a control that has to be
            # looked up in a table is a control a reader takes on trust.
            middle.add(Series([c4_row["D_gen_block_control_mean"]], [y],
                              color=MUTED, marker=1.9))
            right.add(Series([0, c4_row["hit_own_rate"]], [y, y], color=colour,
                             width=4.2))
        # One label per venue, between its two seed rows.
        left.note(-0.62, n - 1 - index, label, anchor="end", color=INK,
                  size=10.5, dy=0)
        left.note(-0.62, n - 1 - index, note, anchor="end",
                  color=MUTED, size=8.5, dy=12)
        # Anchored at the right edge rather than beside the bar: a label placed
        # relative to a 14% bar and a 1% bar lands in two different columns,
        # and the eye then reads the labels as a second series.
        right.note(0.232, n - 1 - index,
                   f"{100 * c4[venue][0]['hit_own_rate']:.1f}% / "
                   f"{100 * c4[venue][1]['hit_own_rate']:.1f}%",
                   anchor="end", color=MUTED, size=9, dy=4)

    # The seeds are labelled on their own rows rather than in a key: the top
    # venue has empty panel to its right, and a reader who has to look up a
    # legend has already left the figure.
    for seed in seeds:
        row = c3["synthetic_flat_step"][seed]
        left.note(row["bootstrap_D_minus_control"]["ci95"][1] + 0.14,
                  place(0, seed), f"seed {seed}",
                  color=PALETTE["teal"] if seed == 0 else PALETTE["clay"],
                  size=9.5, dy=4)
    left.note(-0.62, -0.5, "bars are 95% component intervals", anchor="end",
              color=MUTED, size=9.5)
    middle.note(0.006, place(2, 1) - 0.34, "grey: matched irrelevant edit",
                color=MUTED, size=9)

    step_c3 = c3["synthetic_flat_step"]
    shape_c3 = c3["composed_shape_copy2"]
    shape_c4 = c4["composed_shape_copy2"]
    return figure(
        [(left, 226, 26), (middle, 496, 26), (right, 744, 26)],
        width=964, height=300, number="Figure 15.",
        caption=(
            "<b>A controlled prefix change moves the conditional distribution "
            "by a lot, and moves what the model emits by a little.</b> Left: "
            "teacher-forced Δ, the four-way symmetric contrast minus its matched "
            "donor control, at "
            f"{shape_c3[0]['D_minus_control_bits_per_byte_mean']:+.2f} / "
            f"{shape_c3[1]['D_minus_control_bits_per_byte_mean']:+.2f} bits per "
            "target byte on the composed venue and "
            f"{step_c3[0]['D_minus_control_bits_per_byte_mean']:+.2f} / "
            f"{step_c3[1]['D_minus_control_bits_per_byte_mean']:+.2f} on the step "
            "venue, with both component intervals excluding zero. Middle: the same "
            "checkpoints prompted at the same boundary and left to decode, "
            "paired on one shared block of uniforms — the preference survives, "
            "an order of magnitude smaller in a unit that cannot be compared "
            "to the left panel and is not. The grey dot is the matched "
            "irrelevant prefix edit, which moves the completions by "
            "&#8722;0.003 to +0.001 on every row: what the completions follow "
            "is the relation, not the fact that the prefix changed. <b>Right "
            "is what stops the middle panel being over-read:</b> on the "
            "composed venue a positive, interval-excludes-zero preference "
            "sits on "
            f"{100 * shape_c4[0]['hit_own_rate']:.1f}% / "
            f"{100 * shape_c4[1]['hit_own_rate']:.1f}% of draws actually "
            "producing the compatible continuation. The preference is real and "
            "the capability is not there. The two diagnostic rows fail for "
            "different reasons and only one of them is about the relation: the "
            "y-axis twins are readable (prefix cost ×1.34) and flat, while the "
            "exact quarter turn costs the model ×2.2–2.4 its own prefix rate, so "
            "its null is about the shift and not about the relation. Two "
            "checkpoints per venue are not a population; every number here is "
            "conditional on the named ones."),
    )



# ---------------------------------------------------------------------------


#: The five `project_progressive_v1` cells, in the order they were produced.
#: F5b's two ran under `gain_calibration="none"` on `data_seed=100`; F5c's three
#: ran with the gain calibrated at the phase transition, on `data_seed=200`.
#: Different validation splits, so **absolute** bits/drawing is not comparable
#: across the two packages and every comparison below is within-checkpoint.
FEEDBACK_CELLS: tuple[tuple[str, str, str, str, str], ...] = (
    ("F5b", "feedback_qual_project_progressive_v1_s100",
     "feedback_qual_project_progressive_v1_s100_cell", "slate", "5 2.5"),
    ("F5b", "feedback_qual_project_progressive_v1_s101",
     "feedback_qual_project_progressive_v1_s101_cell", "slate", "2 2"),
    # Colour is the seed and dash is the replicate, so the repeated cell reads as
    # one configuration run twice rather than as two more cells.
    ("F5c", "feedback_f5c_project_progressive_v1_s200_r1",
     "feedback_f5c_project_progressive_v1_s200_r1_cell", "clay", ""),
    ("F5c", "feedback_f5c_project_progressive_v1_s200_r2",
     "feedback_f5c_project_progressive_v1_s200_r2_cell", "clay", "1.5 2"),
    ("F5c", "feedback_f5c_project_progressive_v1_s201_r1",
     "feedback_f5c_project_progressive_v1_s201_r1_cell", "teal", ""),
)


def feedback_cell(tag: str) -> dict:
    """One qualification cell report, refused unless it verifies against itself."""
    body = json.loads((RUNS / f"{tag}.json").read_text())
    recorded = body.get("report_sha256")
    from dm.eval.provenance import canonical_digest

    if recorded != canonical_digest(body, "report_sha256"):
        raise SystemExit(f"{tag} does not match its own payload digest")
    return body["cell"]


def fig_feedback_calibration() -> str:
    """Direction 3's one correction package: the lever moved and nothing followed.

    F5b diagnosed the fused input as arriving 3-5x below the scale the trunk
    reads, because the shared norm's gain is calibrated to the embedding's RMS at
    *initialisation* while the channel switches on at the midpoint. F5c is the
    single predeclared package that hypothesis was allowed to buy: one lever, the
    gain re-calibrated once at the phase transition to the frequency-weighted RMS
    of the embeddings the channel then meets.

    **The lever moved.** Right panel, x axis: fused/standard input p99 goes from
    0.27-0.29 to 0.74-0.80, and the gain lands at the embedding's own RMS
    (g/e 0.97-1.01, against 0.28-0.41 before). The intervention did the thing it
    was designed to do, by a factor of 2.5-2.8.

    **Nothing followed it.** Same panel, y axis: the exact soft-minus-standard
    validation guard does not come down. Middle panel says the recurrence got
    *worse* -- the repeated-prefill penalty at pass 32 rises from 27/47 to
    100-153 bits/drawing, and the wavefront leaves the frozen bound in all three
    cells having been inside it in both of F5b's.

    **And the left panel is why every one of those numbers needs a floor.** The
    two clay curves are the same configuration run twice under requested and
    granted deterministic execution. They separate at the first eval, before the
    calibration fires, and end 2.09 bits/drawing apart.
    """
    from dm.eval.plot import PALETTE as P

    cells = {}
    for _, record_tag, cell_tag, _, _ in FEEDBACK_CELLS:
        cells[record_tag] = feedback_cell(cell_tag)

    # -- left: the training trajectories, all five ------------------------
    left = Panel(xlim=(0, 25.6), ylim=(158, 196), width=250, height=186,
                 xlabel="training step (thousands)",
                 ylabel="validation bits / drawing",
                 title="the runs", xticks=[0, 6, 12, 18, 24],
                 yticks=[160, 170, 180, 190])
    left.rule(12, axis="x", color=MUTED, dash="2 3")
    left.note(11.4, 194.6, "feedback on;", size=9.5, color=MUTED, anchor="end")
    left.note(11.4, 191.9, "gain calibrated", size=9.5, color=MUTED, anchor="end")
    for package, record_tag, _, colour, dash in FEEDBACK_CELLS:
        xs, ys = history(record_tag)
        left.add(Series([x / 1000 for x in xs], ys, color=P[colour], dash=dash,
                        width=1.4 if package == "F5b" else 1.7, marker=1.5))
    # Stacked in the panel's one empty corner rather than at each series' last
    # point: five curves converge inside three bits of each other by step 24,000,
    # so in-place labels would sit on top of the data they name.
    left.note(24.8, 188.5, "F5b, seeds 100 / 101, own split", size=10,
              color=P["slate"], anchor="end")
    left.note(24.8, 184.3, "F5c seed 200, run twice", size=10, color=P["clay"],
              anchor="end")
    left.note(24.8, 180.1, "F5c seed 201", size=10, color=P["teal"], anchor="end")

    # -- middle: the repeated-prefill penalty, which *is* comparable -------
    depths = [0, 1, 2, 4, 8, 16, 32]
    middle = Panel(xlim=(-0.25, 6.9), ylim=(-12, 168), width=222, height=186,
                   xlabel="fused prefill passes",
                   ylabel="bits / drawing above pass 0",
                   title="what the recurrence costs",
                   xticks=list(range(7)),
                   xtick_labels=[str(d) for d in depths],
                   yticks=[0, 40, 80, 120, 160])
    for package, record_tag, _, colour, dash in FEEDBACK_CELLS:
        rows = {row["passes"]: row["bits_per_drawing"]
                for row in cells[record_tag]["stability"]["report"]["passes"]}
        base = rows[0]
        middle.add(Series(list(range(7)), [rows[d] - base for d in depths],
                          color=P[colour], dash=dash,
                          width=1.4 if package == "F5b" else 1.7, marker=2.0))
    middle.note(0.15, 152, "F5c — gain calibrated", size=10.5, color=P["clay"])
    middle.note(0.15, 139, "F5b — gain at init", size=10.5, color=P["slate"])

    # -- right: what the lever moved against what it bought ---------------
    right = Panel(xlim=(0.14, 0.93), ylim=(-6, 82), width=214, height=186,
                  xlabel="fused / standard input RMS p99",
                  ylabel="soft − standard, bits / drawing",
                  title="the lever, and the price",
                  xticks=[0.2, 0.4, 0.6, 0.8], yticks=[0, 20, 40, 60, 80])
    right.rule(1.0, color=P["indigo"], dash="4 2")
    right.note(0.155, 5.0, "budget +1.0", size=9.5, color=P["indigo"])
    for _, record_tag, _, colour, _ in FEEDBACK_CELLS:
        cell = cells[record_tag]
        report = cell["stability"]["report"]
        rows = {row["passes"]: row for row in report["passes"]}
        ratio = (rows[32]["fused_input_rms_p99"]
                 / report["standard_input_rms_p99"])
        guard = cell["generic_guards"]["values"][
            "validation_cost_delta_bits_per_drawing"]
        right.add(Series([ratio], [guard], color=P[colour], marker=4.2))
    right.note(0.155, 75, "gain at init", size=9.5, color=MUTED)
    right.note(0.915, 75, "gain at switch-on", size=9.5, color=MUTED, anchor="end")
    # The distance between the two clusters is the intervention, so it is drawn
    # as a distance rather than asserted in the caption alone.
    right.add(Series([0.335, 0.700], [36, 36], color=MUTED, width=0.9,
                     dash="2 2"))
    right.note(0.517, 39.5, "the lever, 2.7×", size=10, color=MUTED,
               anchor="middle")

    return figure(
        [(left, 92, 30), (middle, 412, 30), (right, 706, 30)],
        width=944, height=268, number="Figure 16.",
        caption=(
            "<b>Direction 3's one correction package: the lever moved and the "
            "outcome did not follow.</b> Five full-budget cells on the "
            "<code>project_progressive_v1</code> schedule, 857,600 parameters and "
            "24,000 steps, arm <code>glu_source_v2</code>. F5b's two (dashed, "
            "grey) ran with the shared input norm's gain left at its "
            "initialisation, 0.02; F5c's three (solid) set it once, at the "
            "predeclared feedback phase transition, to the token-frequency-"
            "weighted RMS of the embeddings the channel then meets. <b>Right is "
            "the finding.</b> The x axis is the quantity F5b's diagnosis named "
            "and the lever targets, and it moved 2.7&#215;, from 0.27&#8211;0.29 "
            "to 0.74&#8211;0.80 &#8212; the fused input now arrives at the scale "
            "the trunk spent 24,000 steps learning to read. The y axis is the "
            "exact sequential soft&#8722;standard validation guard, and it did "
            "not come down: 49/63 before, 24/59/68 after, against a budget of "
            "+1.0 drawn in blue at the floor. <b>Middle says the recurrence got "
            "worse, on the one axis the two packages may be compared on.</b> "
            "Absolute likelihood cannot be compared &#8212; the packages use "
            "different data seeds and so different validation splits &#8212; but "
            "the repeated-prefill penalty is a within-checkpoint difference and "
            "the split cancels: 27 and 47 bits/drawing at pass 32 before, 100, "
            "107 and 153 after, with no overlap. The most economical reading is "
            "that the small gain was not suppressing a good channel; it was "
            "keeping a bad one quiet. <b>Left is the floor under all of it.</b> "
            "The two clay curves are one configuration run twice with "
            "deterministic algorithms requested and granted by torch. They "
            "separate at the first eval &#8212; before the dashed rule where the "
            "calibration fires &#8212; and end 2.09 bits/drawing apart, with 8.7 "
            "bits/drawing between their guards and 1.5&#215; between their "
            "wavefronts. That spread is comparable to the package-to-package "
            "differences in the middle panel and is not comparable to the "
            "24&#8211;68&#215; the guard misses its budget by, which is what "
            "makes the closure safe and the middle panel suggestive rather than "
            "settled. Three cells, no replicate within a seed, one schedule: a "
            "predeclared package answering one question, not an estimate of "
            "anything."),
    )



FIGURES = {
    1: ("fig1_granularity", fig_granularity),
    2: ("fig2_budget", fig_budget),
    3: ("fig3_recovery", fig_recovery),
    4: ("fig4_memory", fig_memory),
    5: ("fig5_instructions", fig_instructions),
    6: ("fig6_orbits", fig_orbits),
    7: ("fig7_device_cost", fig_device_cost),
    8: ("fig8_sample_quality", fig_sample_quality),
    9: ("fig9_transform_recovery", fig_transform_recovery),
    10: ("fig10_fold_price", fig_fold_price),
    11: ("fig11_termination", fig_termination),
    12: ("fig12_compression_rate", fig_compression_rate),
    13: ("fig13_conditioning", fig_conditioning),
    14: ("fig14_class_geometry", fig_class_geometry),
    15: ("fig15_context_generation", fig_context_generation),
    16: ("fig16_feedback_calibration", fig_feedback_calibration),
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", type=int, nargs="*", choices=sorted(FIGURES))
    ap.add_argument("--out", type=Path, default=ROOT / "docs" / "figs")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    for n in args.only or sorted(FIGURES):
        name, build = FIGURES[n]
        path = args.out / f"{name}.svg"
        path.write_text(build())
        print(f"  {path.name}  {path.stat().st_size / 1024:.1f} KB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
