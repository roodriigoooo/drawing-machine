#!/usr/bin/env python3
"""Claim 4's third number: cycles per drawing, on a real Cortex-M0+.

`docs/claim4.md` has said since 2026-08-10 that this one cannot be honestly
emulated, and the reason is specific rather than cautious. Instruction timing on
a Cortex-M0+ *is* deterministic -- no cache, no branch prediction, a single-cycle
multiplier -- so a cycle count is computable from a disassembly. What is not a
core property is the **flash wait states and prefetch buffer**, which are a
vendor choice: the same binary on two M0+ parts at one clock does not take the
same number of cycles.

**This measurement removes that term rather than characterising it.** The image
is linked into SRAM (`port/pico/link.ld`), so every instruction fetch is a
single-cycle access and what comes back is the core's own cost. That makes the
number comparable to any other M0+ running from zero-wait-state memory, and it
makes it a *lower bound* for the same interpreter running from XIP -- which is
the honest shape, because the XIP figure would be about Raspberry Pi's flash
configuration.

    python3 scripts/cycles.py                    # 40 programs per corpus, 5 reps
    python3 scripts/cycles.py --programs 120 --reps 9

Three guards travel with the number, and each of them has fired on a project
like this one:

- **The clock source is asserted, not inferred.** The firmware reads back the
  RP2040 clock mux controls, their hardware `SELECTED` acknowledgements, and
  both dividers, and emits `clock_source xosc`. The host requires that record.
  The SysTick/TIMER ratio remains a secondary frequency sanity check; because
  both counters could otherwise follow the same wrong source, the ratio alone is
  not treated as proof.
- **The instrument's own cost is measured and subtracted**, the same way
  `scripts/footprint.py` subtracts QEMU's empty-corpus baseline.
- **A wrapped counter is a failure, not a data point.** SysTick is 24 bits.

The join against `runs/qemu_instructions.json` is what turns cycles into cycles
*per instruction*, and it is only legitimate because the two builds compile the
interpreter to byte-identical instructions -- `tests/test_pico.py` asserts that,
and this script refuses the join if the corpora it rebuilds do not match the
record row for row.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

ROOT = Path(__file__).resolve().parent.parent
RECORD = ROOT / "runs" / "pico_cycles.json"
INSTRUCTIONS = ROOT / "runs" / "qemu_instructions.json"

#: `dm_machine.h`'s sentinel for a SysTick wrap, as the harness prints it.
OVERFLOW = 0xFFFFFFFF

#: How far the recovered core frequency may differ from the configured one
#: before the run is refused. The direct mux assertion is the source proof; this
#: secondary check catches a bad divider or timebase in either direction.
CLOCK_TOLERANCE = 0.25


def corpora(n: int) -> dict[str, list[bytes]]:
    """The same three corpora, in the same order, as `footprint.py` measured
    instructions over. Identical seeds on purpose: it is what makes a per-program
    join against that record possible instead of a comparison of two averages,
    and averages cannot separate bytes decoded from geometry emitted.
    """
    from dm.data import quickdraw, synthetic
    from dm.isa.spec import Tier

    return {
        "quickdraw": quickdraw.load(("cat", "dog", "bus", "car", "tree"), "valid",
                                    limit=n),
        "tier A L1": synthetic.dataset(n, seed=5, tier=Tier.L1, depth=2),
        "tier A L0": synthetic.dataset(n, seed=6, tier=Tier.L0, depth=1),
    }


def parse(text: str, expected: int) -> dict:
    """One framed pass of the cycles protocol back into a record.

    `BEGIN` … four header lines … one `c` line per program … `elapsed_us` …
    `END n`. The framing is what lets a reader that attached mid-pass throw the
    partial one away, and it is required here rather than tolerated so that a
    capture missing it is a failure instead of a short corpus.
    """
    lines = text.splitlines()
    if len(lines) < 2 or not lines[0].startswith("BEGIN ") or not lines[-1].startswith("END "):
        raise SystemExit("the device's output is not framed by BEGIN/END")
    declared = int(lines[-1].split()[1])
    if declared != expected:
        raise SystemExit(f"the device measured {declared} programs, not {expected}")

    header: dict[str, str] = {}
    rows: list[dict[str, int]] = []
    elapsed: int | None = None

    for line in lines[1:-1]:
        tag, _, rest = line.partition(" ")
        if tag == "c":
            index, cycles, steps = (int(v) for v in rest.split())
            rows.append({"index": index, "cycles": cycles, "steps": steps})
        elif tag == "elapsed_us":
            if rest == "overflow":
                raise SystemExit("the microsecond timer wrapped; the run was too long")
            elapsed = int(rest)
        elif tag in ("clock_source", "machine", "clk_hz", "reps", "overhead"):
            header[tag] = rest
        elif line.startswith("FAILED"):
            raise SystemExit(f"firmware reported: {line}")
        else:
            raise SystemExit(f"unparsable cycles output: {line!r}")

    missing = {"clock_source", "machine", "clk_hz", "reps", "overhead"} - set(header)
    if missing or elapsed is None:
        raise SystemExit(f"the cycles header is incomplete: missing {missing or 'elapsed_us'}")
    if header["clock_source"] != "xosc":
        raise SystemExit(
            f"the firmware reported clock_source {header['clock_source']!r}, not 'xosc'"
        )
    if len(rows) != expected:
        raise SystemExit(f"{len(rows)} measurements for {expected} programs")

    wrapped = [r["index"] for r in rows if r["cycles"] == OVERFLOW]
    if wrapped:
        raise SystemExit(
            f"SysTick wrapped on {len(wrapped)} program(s), first at index {wrapped[0]}. "
            "A 24-bit counter spans 16,777,215 cycles; either the program is longer "
            "than that or the core is far slower than configured."
        )
    return {
        "clock_source": header["clock_source"],
        "machine": header["machine"],
        "clk_hz": int(header["clk_hz"]),
        "reps": int(header["reps"]),
        "overhead": int(header["overhead"]),
        "elapsed_us": elapsed,
        "rows": rows,
    }


def check_clock(measured: dict) -> float:
    """Use the second timebase as a frequency consistency check.

    The firmware's direct mux assertion is the source proof. SysTick counts core
    cycles and TIMER counts microseconds; their ratio recovers the apparent core
    frequency and catches a bad divider or timebase. The firmware accumulates
    that span across the *measured*
    repetitions only and never across the output between them -- a span that
    included a UART draining at 115200 baud would recover something four orders
    of magnitude too low and make this check vacuous. What remains inside it is
    loop overhead, which biases the recovered figure slightly *down*: the useful
    direction, since it cannot make a slow part look fast.
    """
    total = sum(r["cycles"] for r in measured["rows"]) * measured["reps"]
    recovered = total / max(1, measured["elapsed_us"]) * 1e6
    claimed = measured["clk_hz"]
    if not (claimed * (1 - CLOCK_TOLERANCE) <= recovered <=
            claimed * (1 + CLOCK_TOLERANCE)):
        raise SystemExit(
            f"the recovered core frequency was {recovered / 1e6:.2f} MHz but the firmware "
            f"configured {claimed / 1e6:.2f} MHz. The cycle counter and the "
            "microsecond timer disagree, so one of them is not measuring what it "
            "claims in the expected range and neither number is publishable."
        )
    return recovered


def join_instructions(rows: list[dict]) -> str | None:
    """Attach QEMU's per-program instruction count, or say why it cannot be.

    Refused rather than approximated when the shapes disagree: cycles per
    instruction across two runs of *different* programs is a ratio of two
    unrelated numbers, and it would look exactly like a measurement.
    """
    if not INSTRUCTIONS.exists():
        return "runs/qemu_instructions.json does not exist"
    recorded = json.loads(INSTRUCTIONS.read_text())["rows"]
    if len(recorded) != len(rows):
        return (f"the instruction record has {len(recorded)} programs and this run "
                f"has {len(rows)}; rerun `scripts/footprint.py --instructions` at "
                "the same count")
    for mine, theirs in zip(rows, recorded):
        if (mine["corpus"], mine["bytes"], mine["steps"]) != (
            theirs["corpus"], theirs["bytes"], theirs["steps"]
        ):
            return (f"program {mine['index']} of {mine['corpus']} differs between the "
                    "two records; the corpora are not the same programs")
    for mine, theirs in zip(rows, recorded):
        mine["instructions"] = theirs["instructions"]
        mine["points"] = theirs["points"]
        mine["discs"] = theirs["discs"]
    return None


def join(passes: list[dict]) -> dict:
    """Several batches back into one record.

    A batch boundary is not a state boundary -- `dm_vm_run` is re-entrant and
    keeps nothing between programs, which `scripts/footprint.py` checks by
    asserting the interpreter has zero static RAM -- so the rows concatenate.
    The *headers* must not merely concatenate: every batch ran on the same board
    at the same clock with the same instrument, and a batch that disagrees means
    the board was re-flashed or reset in the middle, which would make the halves
    incomparable. So they are checked for equality rather than taken from the
    first.
    """
    first = passes[0]
    for other in passes[1:]:
        for field in ("clock_source", "machine", "clk_hz", "reps", "overhead"):
            if other[field] != first[field]:
                raise SystemExit(
                    f"batches disagree on {field}: {first[field]!r} then "
                    f"{other[field]!r}. They are not one measurement."
                )
    return {
        **first,
        "elapsed_us": sum(p["elapsed_us"] for p in passes),
        "rows": [row for p in passes for row in p["rows"]],
    }


def measure(n: int, reps: int, fuel: int) -> dict:
    from dm.vm.interp import VM
    from scripts import pico
    from scripts.uf2 import batches, capacity

    pico.build()
    built = corpora(n)
    programs = [p for group in built.values() for p in group]

    groups = batches(programs, capacity())
    texts = pico.run_batches(groups, fuel=fuel, reps=reps)
    measured = join([parse(text, len(group)) for text, group in zip(texts, groups)])

    recovered = check_clock(measured)

    vm = VM()
    rows: list[dict] = []
    cursor = 0
    for corpus, group in built.items():
        for index, program in enumerate(group):
            row = measured["rows"][cursor]
            trace = vm.run(program)
            if trace.steps != row["steps"]:
                raise SystemExit(
                    f"{corpus}[{index}]: the device executed {row['steps']} "
                    f"instructions and the reference {trace.steps}. Conformance "
                    "should have caught this; run `conformance.py --pico`."
                )
            rows.append({
                "corpus": corpus,
                "index": index,
                "bytes": len(program),
                "steps": trace.steps,
                # The instrument's own cost, removed the same way footprint.py
                # removes QEMU's empty-corpus baseline.
                "cycles": row["cycles"] - measured["overhead"],
            })
            cursor += 1

    note = join_instructions(rows)
    return {
        "clock_source": measured["clock_source"],
        "machine": measured["machine"],
        "clk_hz": measured["clk_hz"],
        "clk_hz_recovered": round(recovered),
        "reps": measured["reps"],
        "overhead_cycles": measured["overhead"],
        "statistic": "minimum over reps, instrument overhead subtracted",
        "resident": "sram",
        "instructions_joined": note is None,
        "instructions_note": note,
        "rows": rows,
    }


def report(record: dict) -> None:
    print(f"\n  {record['machine']}, {record['clk_hz'] / 1e6:.3f} MHz, "
          f"{record['reps']} reps, instrument overhead {record['overhead_cycles']} cycles")
    print(f"  clock source: {record['clock_source']}; frequency cross-check: "
          f"{record['clk_hz_recovered'] / 1e6:.2f} MHz from the timer\n")

    print(f"  {'corpus':<12} {'n':>4} {'bytes':>8} {'cycles':>10} {'cyc/byte':>9} "
          f"{'cyc/instr':>10}")
    for corpus in dict.fromkeys(row["corpus"] for row in record["rows"]):
        group = [row for row in record["rows"] if row["corpus"] == corpus]
        cycles = statistics.mean(row["cycles"] for row in group)
        size = statistics.mean(row["bytes"] for row in group)
        cpi = (f"{statistics.mean(row['cycles'] / row['instructions'] for row in group):>10.3f}"
               if record["instructions_joined"] else f"{'—':>10}")
        print(f"  {corpus:<12} {len(group):>4} {size:>8.1f} {cycles:>10,.0f} "
              f"{cycles / size:>9.1f}{cpi}")

    if not record["instructions_joined"]:
        print(f"\n  cycles/instruction NOT computed: {record['instructions_note']}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--programs", type=int, default=40,
                    help="programs per corpus; must match the instruction record to join")
    # Odd by default and small: the core is deterministic, so repetitions buy a
    # guard against interference rather than statistical power, and the
    # statistic taken over them is the minimum rather than the mean.
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--fuel", type=int, default=100_000)
    ap.add_argument("--out", type=Path, default=RECORD)
    args = ap.parse_args()

    from scripts import pico

    try:
        record = measure(args.programs, args.reps, args.fuel)
    except pico.PicoError as failure:
        print(f"\n{failure}", file=sys.stderr)
        return 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(record, indent=1))
    tmp.replace(args.out)
    report(record)
    print(f"\n  -> {args.out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
