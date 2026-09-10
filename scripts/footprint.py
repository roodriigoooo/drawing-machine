#!/usr/bin/env python3
"""Claim 4's second gate: what does the interpreter cost on a Cortex-M0+?

Three of claim 4's four numbers are static, and static numbers do not need a
board. This measures them from a real Thumb build:

- **flash** -- `.text` plus `.rodata`, the bytes the part must store.
- **static RAM** -- `.data` plus `.bss`. Asserted to be **zero**: the
  interpreter keeps no globals, so two traces can run concurrently and a second
  instance costs nothing but stack.
- **peak stack** -- from `-fstack-usage`. Exact rather than estimated, because
  the interpreter does not recurse: `REPEAT` nesting lives in a fixed array
  (`DM_MAX_REPEAT_DEPTH`), so the call graph is a straight line and the bound is
  a sum instead of a fixed point. This is also the number the streaming sink in
  `dm_vm.h` was designed for -- it does not grow with the program.

The fourth number, **cycles**, is not here and must not be guessed. A cycle
count depends on the part's flash wait states and prefetch buffer, which are a
vendor choice rather than a core property, so it needs silicon or a
cycle-accurate model. `docs/claim4.md` says what is still owed and what it costs.

    python3 scripts/footprint.py

Exits non-zero when nothing could be measured, rather than reporting a
footprint this project has not measured as though it were small.
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
BUILD = PORT / "build"
QEMU = "qemu-system-arm"
#: Where `--instructions` writes its measurement, so a figure reads data rather
#: than re-running an emulator.
INSTRUCTIONS = ROOT / "runs" / "qemu_instructions.json"

#: Where a section lands on a bare-metal part, by name *prefix* -- the build
#: uses `-ffunction-sections -fdata-sections`, so the real names are
#: `.text.dm_vm_run` and `.rodata.dm_instr_size` and an exact match silently
#: measures nothing. Order matters: the first matching prefix wins.
SECTION_CLASS = (
    (".ARM.exidx", "unwind"),   # discarded by any bare-metal linker script
    (".ARM.attributes", None),  # toolchain metadata, never loaded
    (".comment", None),
    (".note", None),
    (".llvm_addrsig", None),
    (".text", "flash"),
    (".rodata", "flash"),
    (".data", "ram"),
    (".bss", "ram"),
)


def classify(section: str) -> str | None:
    for prefix, where in SECTION_CLASS:
        if section.startswith(prefix):
            return where
    return "unclassified"

SIZE_LINE = re.compile(r"^(\.\S+)\s+(\d+)")
SU_LINE = re.compile(r"^(?P<file>[^:]+):(?P<line>\d+):(?P<name>\S+)\s+(?P<bytes>\d+)\s+(?P<qual>\S+)")


def size_tool() -> list[str] | None:
    for candidate in (["arm-none-eabi-size"], ["llvm-size"], ["xcrun", "llvm-size"]):
        if shutil.which(candidate[0]):
            probe = subprocess.run([*candidate, "--version"], capture_output=True, check=False)
            if probe.returncode == 0:
                return candidate
    return None


def sections(tool: list[str], objects: list[Path]) -> dict[str, dict[str, int]]:
    """`{object: {section: bytes}}` from `size -A`, which both tools speak."""
    out: dict[str, dict[str, int]] = {}
    for obj in objects:
        proc = subprocess.run([*tool, "-A", str(obj)], capture_output=True, text=True, check=True)
        found: dict[str, int] = {}
        for line in proc.stdout.splitlines():
            match = SIZE_LINE.match(line.strip())
            if match and match.group(1) != ".comment":
                found[match.group(1)] = int(match.group(2))
        out[obj.name] = found
    return out


def stack_usage() -> list[tuple[str, int, str]]:
    frames: list[tuple[str, int, str]] = []
    for su in sorted(BUILD.glob("*.su")):
        for line in su.read_text().splitlines():
            match = SU_LINE.match(line.strip())
            if match:
                frames.append((match["name"], int(match["bytes"]), match["qual"]))
    return frames


# ---------------------------------------------------------------------------
# instructions, per program


def instructions_for(programs: list[bytes], fuel: int = 100_000) -> int:
    """Guest instructions retired running `programs`, via QEMU's exec log.

    `one-insn-per-tb` makes each translation block one instruction, so the log
    has one `Trace` line per instruction executed. The **null sink** is selected
    with `quiet.txt`, so this counts the interpreter and not the harness's
    decimal formatter, which would otherwise dominate.
    """
    from scripts.conformance import framed

    with tempfile.TemporaryDirectory() as work:
        d = Path(work)
        (d / "frames.bin").write_bytes(framed(programs))
        (d / "fuel.txt").write_text(str(fuel))
        (d / "quiet.txt").write_text("1")
        proc = subprocess.run(
            [QEMU, "-M", "microbit", "-nographic",
             "-semihosting-config", "enable=on,target=native",
             "-accel", "tcg,one-insn-per-tb=on", "-d", "exec,nochain", "-D", "e.log",
             "-kernel", str((BUILD / "dm_qemu.elf").resolve())],
            cwd=d, capture_output=True, check=False,
        )
        if proc.returncode != 0:
            raise SystemExit(f"qemu failed: {proc.stderr.decode()[:400]}")
        with open(d / "e.log", errors="ignore") as log:
            return sum(1 for line in log if line.startswith("Trace"))


def measure_instructions(n: int) -> int:
    """Per-program instruction counts, and the geometry each program produced.

    One QEMU run per program, with the empty-corpus run subtracted, because the
    question this answers is *relational*: whether the interpreter's work tracks
    the bytecode it reads or the geometry it emits. An aggregate cannot separate
    those -- the two are correlated across a corpus and dissociate only within
    one, on the programs where `REPEAT` or `CURVE` makes them disagree.

    Corpora are chosen to make them disagree: Tier A L1 carries `REPEAT`, whose
    whole purpose is to emit more geometry than it costs in bytes.
    """
    from dm.data import quickdraw, synthetic
    from dm.isa.spec import Tier
    from dm.vm.interp import VM

    if shutil.which(QEMU) is None:
        raise SystemExit(f"{QEMU} not found -- instructions NOT measured")

    corpora = {
        "quickdraw": quickdraw.load(("cat", "dog", "bus", "car", "tree"), "valid",
                                    limit=n),
        "tier A L1": synthetic.dataset(n, seed=5, tier=Tier.L1, depth=2),
        "tier A L0": synthetic.dataset(n, seed=6, tier=Tier.L0, depth=1),
    }

    baseline = instructions_for([])
    vm = VM()
    rows = []
    for corpus, programs in corpora.items():
        for program in programs:
            trace = vm.run(program)
            points = sum(len(s.points) for s in trace.strokes)
            points += sum(len(r.points) for r in trace.regions)
            rows.append({
                "corpus": corpus,
                "bytes": len(program),
                # What the VM *decoded*: one per bytecode instruction executed,
                # which a REPEAT body inflates far past the byte count.
                "steps": trace.steps,
                # What it *emitted*. Discs are geometry with no points, so they
                # are counted separately rather than silently missing -- a
                # CIRCLE-only program has real cost and zero points.
                "points": points,
                "discs": len(trace.discs),
                "instructions": instructions_for([program]) - baseline,
            })

    INSTRUCTIONS.write_text(json.dumps(
        {"baseline": baseline, "machine": "qemu microbit (cortex-m0)", "rows": rows},
        indent=1))
    print(f"  {len(rows)} programs, baseline {baseline:,} instructions")
    print(f"  -> {INSTRUCTIONS.relative_to(ROOT)}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no-build", action="store_true")
    ap.add_argument("--instructions", type=int, metavar="N", default=0,
                    help="also measure guest instructions for N programs per corpus")
    args = ap.parse_args()

    if args.instructions:
        return measure_instructions(args.instructions)

    if not args.no_build:
        build = subprocess.run(["make", "-s", "-C", str(PORT), "device"], check=False)
        if build.returncode != 0:
            print("no cross compiler -- device footprint NOT measured.\n"
                  "  install either: brew install --cask gcc-arm-embedded\n"
                  "  or any clang new enough for --target=thumbv6m-none-eabi")
            return 2

    tool = size_tool()
    objects = sorted(BUILD.glob("*.o"))
    if tool is None or not objects:
        print("no size tool or no objects -- device footprint NOT measured")
        return 2

    table = sections(tool, objects)
    totals: dict[str, int] = {}
    print(f"cortex-m0plus, -Os, size via {' '.join(tool)}\n")
    print(f"  {'object':<12} {'section':<26} {'bytes':>6}  where")
    for name, found in table.items():
        for section, size in sorted(found.items()):
            where = classify(section)
            if size == 0 or where is None:
                continue
            totals[where] = totals.get(where, 0) + size
            print(f"  {name:<12} {section:<26} {size:>6}  {where}")

    frames = stack_usage()
    total_frames = sum(size for _, size, _ in frames)

    print(f"\n  flash (.text + .rodata)        {totals.get('flash', 0):>7} bytes")
    print(f"  static RAM (.data + .bss)      {totals.get('ram', 0):>7} bytes")
    print(f"  peak stack (no recursion)      {total_frames:>7} bytes")
    if totals.get("unwind"):
        print(f"  unwind metadata, discardable   {totals['unwind']:>7} bytes")

    print("\n  stack frames:")
    for name, size, qual in frames:
        print(f"    {name:<24} {size:>5}  {qual}")

    # The two structural claims `dm_vm.h` makes, checked rather than asserted in
    # prose. Either failing means the port stopped being the thing this project
    # says runs on a 32 KB part.
    problems = []
    if totals.get("ram"):
        problems.append(
            f"static RAM is {totals['ram']} bytes; the interpreter is supposed to keep none"
        )
    if totals.get("unclassified"):
        problems.append(
            f"{totals['unclassified']} bytes in sections this script cannot place; "
            "the footprint above is incomplete"
        )
    if any(qual != "static" for _, _, qual in frames):
        problems.append("a frame is 'dynamic' or 'bounded': stack no longer has an exact bound")
    if not frames:
        problems.append("no .su output; peak stack was not measured")
    if not totals.get("flash"):
        problems.append("no flash sections found; nothing was measured")

    for problem in problems:
        print(f"\n  FAIL: {problem}")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
