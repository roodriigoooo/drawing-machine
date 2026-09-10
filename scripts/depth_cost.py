#!/usr/bin/env python3
"""What the transform tier costs the device, and what its depth bound buys.

`docs/direction.md` §2.6 names the transform depth as **the knob** to turn if
peak stack ever exceeds a budget. A knob whose travel nobody has measured is a
hope, so this compiles the interpreter at a range of depths and reads the cost
off `-fstack-usage` and `size`.

It also measures the two implementation variants against each other, because
`dm_xform_then` composing by value rather than through pointers was worth real
bytes on both axes and that is the kind of claim which decays into folklore
unless a record carries it.

Nothing here is a build the project ships. `DM_MAX_XFORM_DEPTH` below the
reference's bound makes the device fault where `dm/vm/interp.py` does not, so a
lowered build is a *different specification* and `scripts/conformance.py` will
say so. These are cost probes, and the record says as much.

    python3 scripts/depth_cost.py           # writes runs/depth_cost.json
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

ROOT = Path(__file__).resolve().parent.parent
PORT = ROOT / "port"
RUNS = ROOT / "runs"

#: The device build, minus the parts that would drag in a runtime. Kept in one
#: place because a probe compiled with different flags from the shipped build
#: measures a different program.
CFLAGS = (
    "--target=thumbv6m-none-eabi", "-mcpu=cortex-m0plus", "-std=c11", "-Os",
    "-ffreestanding", "-fno-builtin", "-fno-unwind-tables",
    "-fno-asynchronous-unwind-tables", "-fstack-usage",
)


def measure(out: Path) -> tuple[int, int]:
    """`(peak stack, .text bytes)` for one compiled interpreter.

    Peak stack is the **sum of every frame**, which is `scripts/footprint.py`'s
    methodology and is exact here because the interpreter does not recurse: the
    call graph is a straight line, so the sum is the bound rather than an
    estimate. Reading `dm_vm_run` alone would miss the helpers, and the helpers
    are where the by-value variant spent most of what it cost -- a probe that
    looked only at the main frame reported the two builds as identical.
    """
    frames = 0
    for line in out.with_suffix(".su").read_text().splitlines():
        # `file:line:col:function\tbytes\tqualifier`
        fields = line.split("\t")
        if len(fields) < 3:
            continue
        if fields[2].strip() != "static":
            raise SystemExit(f"{fields[0]} has a {fields[2]!r} frame, not static -- "
                             "the bound stops being exact and the figure a guess")
        frames += int(fields[1])
    text = subprocess.run(["xcrun", "llvm-size", "-A", str(out)],
                          check=True, capture_output=True, text=True).stdout
    size = sum(int(m.group(1)) for m in re.finditer(r"^\.text\S*\s+(\d+)", text, re.MULTILINE))
    return frames, size


def compile_probe(out: Path, defines: tuple[str, ...] = (),
                  source: Path | None = None) -> tuple[int, int]:
    subprocess.run(
        ["clang", *CFLAGS, *(f"-D{d}" for d in defines), f"-I{PORT / 'include'}",
         "-c", str(source or PORT / "src" / "dm_vm.c"), "-o", str(out)],
        check=True, capture_output=True,
    )
    return measure(out)


def by_value_source(text: str) -> str:
    """The interpreter with `dm_xform_then` taking and returning by value.

    Rewritten here rather than kept as a second file: two copies of an
    interpreter is exactly the drift this project spends its time preventing, and
    the variant exists only to be measured. The substitution is asserted rather
    than assumed -- a silent no-op would report the two builds as identical and
    look like a null result.
    """
    swaps = [
        ("""static void dm_xform_then(dm_xform *first, const dm_xform *second)
{
    dm_d4_linear(second->code, &first->dx, &first->dy);
    first->dx += second->dx;
    first->dy += second->dy;
    first->code = dm_d4_then(first->code, second->code);
}""",
         """static dm_xform dm_xform_then(dm_xform first, dm_xform second)
{
    dm_d4_linear(second.code, &first.dx, &first.dy);
    return (dm_xform){first.dx + second.dx, first.dy + second.dy,
                      dm_d4_then(first.code, second.code)};
}"""),
        ("        dm_xform_then(&out, &st->xf_stack[i]);",
         "        out = dm_xform_then(out, st->xf_stack[i]);"),
        ("                    dm_xform_then(&st.xf_stack[frame->xf_slot], &step);",
         """                    st.xf_stack[frame->xf_slot] =
                        dm_xform_then(st.xf_stack[frame->xf_slot], step);"""),
    ]
    for old, new in swaps:
        if old not in text:
            raise SystemExit("dm_xform_then is not in the shape this probe rewrites; "
                             "the by-value comparison would measure nothing")
        text = text.replace(old, new, 1)
    return text


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--depths", type=int, nargs="+", default=[1, 2, 3, 4, 6, 8, 12, 16])
    ap.add_argument("--out", type=Path, default=RUNS / "depth_cost.json")
    args = ap.parse_args()

    if not shutil.which("clang"):
        raise SystemExit("no clang -- the device cost is not measurable here")

    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        rows = []
        for depth in args.depths:
            stack, text = compile_probe(work / f"d{depth}.o",
                                        (f"DM_MAX_XFORM_DEPTH={depth}",))
            rows.append({"depth": depth, "stack": stack, "text": text})
            print(f"  depth {depth:>2}   stack {stack:>4} B   .text {text:>5} B")

        # The shipped depth, both ways round.
        shipped = next(r for r in rows if r["depth"] == 4)
        variant = work / "byvalue.c"
        variant.write_text(by_value_source((PORT / "src" / "dm_vm.c").read_text()))
        by_value_stack, by_value_text = compile_probe(work / "byvalue.o",
                                                      source=variant)
        print(f"  by value      stack {by_value_stack:>4} B   .text {by_value_text:>5} B")

    report = {
        "rows": rows,
        "shipped_depth": 4,
        "by_pointer": {"stack": shipped["stack"], "text": shipped["text"]},
        "by_value": {"stack": by_value_stack, "text": by_value_text},
        #: ISA v1, before the tier existed. Measured on the same toolchain and
        #: flags, and quoted here so the figure's baseline is a number rather
        #: than a memory: `docs/claim4.md` carries the full table.
        "isa_v1": {"stack": 288, "text": 1354},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(f"\n  -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
