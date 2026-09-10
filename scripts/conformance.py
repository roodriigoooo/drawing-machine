#!/usr/bin/env python3
"""Claim 4's first gate: does the C port compute the *same* geometry as `dm/vm/interp.py`?

Not "close enough". The same. `port/include/dm_vm.h` argues that the reference
VM's output is exactly integral at a scale of `DM_ONE`, so a fixed-point port on
a chip with no FPU can reproduce it bit for bit rather than approximately, and
this script is what turns that argument into a check. Every comparison here is
integer equality; there is no tolerance anywhere in the file, and if one ever
appears the claim it was protecting has already been given up.

The exactness has one precondition -- `CURVE_STEPS` must be a power of two, so
the Bernstein coefficients are binary fractions -- and the header lines the
harness prints are checked against the reference's constants before any program
runs. A port built for a different curve count is a different specification and
is refused rather than diffed.

**What is compared is the reference's specification, not the port's
convenience:** stroke, disc and region lists in their own order, each fault's
kind and pc, the instruction count, and the halted flag. Fault `detail` strings
are diagnostic and have no device counterpart, so they are excluded and that is
the only thing excluded.

    python3 scripts/conformance.py                  # every corpus on this machine
    python3 scripts/conformance.py --quick          # the synthetic corpora only
    python3 scripts/conformance.py --programs 5000

Corpora are whatever is on disk: the two Tier A tiers and the flat expansion
always, QuickDraw and Tabler when their caches are present, plus a fuzz corpus
and a hand-written set of one program per fault path. **The fuzz corpus is not
decoration.** A structured corpus reaches almost none of the fault handling, and
the fault paths are exactly where an interpreter and its port drift apart.
"""

from __future__ import annotations

import argparse
import random
import shutil
import struct
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dm.isa.spec import SPECS, Op, Tier
from dm.vm.interp import CURVE_STEPS, DEFAULT_FUEL, VM

ROOT = Path(__file__).resolve().parent.parent
PORT = ROOT / "port"
QEMU = "qemu-system-arm"

#: Fixed-point scale the port emits. Derived, not chosen: it is the curve
#: denominator `CURVE_STEPS ** 3`, which is what makes flattening exact.
FRAC_BITS = 3 * CURVE_STEPS.bit_length() - 3
ONE = 1 << FRAC_BITS


@dataclass(frozen=True)
class Trace:
    """Both VMs reduced to the same tuple, so `==` is the whole comparison."""

    strokes: tuple[tuple[int, tuple[int, ...]], ...]  # (width, flattened coordinates)
    discs: tuple[tuple[int, int, int, int], ...]
    regions: tuple[tuple[int, ...], ...]
    faults: tuple[tuple[str, int], ...]
    steps: int
    halted: bool


def to_fixed(value: float) -> int:
    """A reference coordinate as the integer the port emits.

    Raises rather than rounds. Rounding here would convert a real divergence --
    the port and the reference disagreeing about a coordinate -- into a silent
    pass, and it would also hide the more interesting failure: a reference
    coordinate that is *not* a multiple of 1/ONE at all, which would mean the
    exactness argument in `dm_vm.h` is wrong for some input.
    """
    scaled = value * ONE
    if scaled != int(scaled):
        raise AssertionError(
            f"reference coordinate {value!r} is not a multiple of 1/{ONE}: "
            "the fixed-point port cannot be exact and dm_vm.h's argument is wrong"
        )
    return int(scaled)


def reference_trace(program: bytes, fuel: int) -> Trace:
    tr = VM(fuel=fuel).run(program)
    return Trace(
        strokes=tuple(
            (s.width, tuple(to_fixed(v) for pt in s.points for v in pt)) for s in tr.strokes
        ),
        discs=tuple((to_fixed(d.cx), to_fixed(d.cy), d.r, d.width) for d in tr.discs),
        regions=tuple(tuple(to_fixed(v) for pt in r.points for v in pt) for r in tr.regions),
        faults=tuple((f.kind.value, f.pc) for f in tr.faults),
        steps=tr.steps,
        halted=tr.halted,
    )


def parse_port_traces(stdout: str, expected: int) -> list[Trace]:
    """The harness's line protocol back into `Trace`.

    It streams points and only then says what the path became, which is the sink
    contract `dm_vm.h` adopts so the device needs no path buffer. Reassembling
    that here rather than in C is deliberate: the host is where an array is
    free, and doing it in the harness would leave the streaming contract itself
    untested.
    """
    lines = stdout.splitlines()
    # The pass is framed. On a semihosted machine that framing is redundant --
    # the file begins where the run began -- but on a Pico the harness loops
    # forever because it cannot know when a reader attached, and an unframed
    # capture would open mid-drawing and parse perfectly to the end. Required on
    # both paths so neither can quietly lose it.
    if len(lines) < 4 or not lines[0].startswith("BEGIN ") or not lines[-1].startswith("END "):
        raise SystemExit("the harness's output is not framed by BEGIN/END")
    declared = int(lines[-1].split()[1])
    if declared != expected:
        raise SystemExit(f"the harness ran {declared} programs, not {expected}")
    lines = lines[1:-1]

    if not lines[0].startswith("frac_bits ") or not lines[1].startswith("curve_steps "):
        raise SystemExit("harness did not print its fixed-point header")
    port_frac, port_steps = int(lines[0].split()[1]), int(lines[1].split()[1])
    if (port_frac, port_steps) != (FRAC_BITS, CURVE_STEPS):
        raise SystemExit(
            f"port built for frac_bits={port_frac} curve_steps={port_steps}, reference is "
            f"{FRAC_BITS}/{CURVE_STEPS}. A different curve count is a different specification."
        )

    traces: list[Trace] = []
    strokes: list[tuple[int, tuple[int, ...]]] = []
    discs: list[tuple[int, int, int, int]] = []
    regions: list[tuple[int, ...]] = []
    faults: list[tuple[str, int]] = []
    path: list[int] = []

    for line in lines[2:]:
        tag, _, rest = line.partition(" ")
        arg = rest.split()
        if tag.startswith("#"):
            strokes, discs, regions, faults, path = [], [], [], [], []
        elif tag == "b":
            path = []
        elif tag == "p":
            path += [int(arg[0]), int(arg[1])]
        elif tag == "e":
            kind = arg[0]
            if kind == "stroke":
                strokes.append((int(arg[2]), tuple(path)))
            elif kind == "region":
                regions.append(tuple(path))
            path = []
        elif tag == "d":
            discs.append((int(arg[0]), int(arg[1]), int(arg[2]), int(arg[3])))
        elif tag == "f":
            faults.append((arg[0], int(arg[1])))
        elif tag == "=":
            traces.append(
                Trace(tuple(strokes), tuple(discs), tuple(regions), tuple(faults),
                      int(arg[0]), arg[1] == "1")
            )
        else:
            raise SystemExit(f"unparsable harness output: {line!r}")

    if len(traces) != expected:
        raise SystemExit(f"harness returned {len(traces)} traces for {expected} programs")
    return traces


def framed(programs: list[bytes]) -> bytes:
    return b"".join(struct.pack("<I", len(p)) + p for p in programs)


def run_native(harness: Path, programs: list[bytes], fuel: int) -> list[Trace]:
    """One process for the whole corpus; framing is `u32` length then bytes."""
    proc = subprocess.run(
        [str(harness), str(fuel)], input=framed(programs), capture_output=True, check=False
    )
    if proc.returncode != 0:
        raise SystemExit(f"harness failed ({proc.returncode}): {proc.stderr.decode()}")
    if proc.stderr:
        # The sanitiser build reports on stderr and still exits 0, so a silent
        # pass here would be the worst possible outcome of running it.
        raise SystemExit(f"harness wrote to stderr:\n{proc.stderr.decode()}")
    return parse_port_traces(proc.stdout.decode(), len(programs))


def bare_metal_inputs(work: Path, programs: list[bytes], fuel: int) -> None:
    """The three files a machine with no argv is given its corpus through.

    Written into a temporary directory rather than the repository because both
    runners resolve the harness's file opens relative to their own working
    directory, and a stray `frames.bin` next to this script would otherwise
    decide what a conformance run tested.
    """
    (work / "frames.bin").write_bytes(framed(programs))
    (work / "fuel.txt").write_text(str(fuel))


def run_qemu(image: Path, programs: list[bytes], fuel: int) -> list[Trace]:
    """The same corpus through the *Thumb* build, on an emulated Cortex-M0.

    Until `--pico` existed this was the only path in the project where the
    port's own instructions executed; it remains the one that needs no hardware.
    Everything else -- the native diff, the sanitiser, the footprint -- runs or
    measures native code, and what carries those across to a target is an
    argument about undefined behaviour rather than a measurement.
    """
    if shutil.which(QEMU) is None:
        raise SystemExit(f"{QEMU} not found -- the Thumb build was NOT executed")
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        bare_metal_inputs(work, programs, fuel)
        proc = subprocess.run(
            [QEMU, "-M", "microbit", "-nographic",
             "-semihosting-config", "enable=on,target=native",
             "-kernel", str(image.resolve())],
            cwd=work, capture_output=True, check=False,
        )
        if proc.returncode != 0:
            raise SystemExit(
                f"qemu failed ({proc.returncode}): {proc.stderr.decode()[:2000]}")
        console = proc.stdout.decode()
        if "FAILED" in console:
            raise SystemExit(f"firmware reported: {console[:2000]}")
        text = (work / "trace.txt").read_text()
    return parse_port_traces(text, len(programs))


def run_pico(image: Path, programs: list[bytes], fuel: int) -> list[Trace]:
    """The same corpus on an **RP2040**. Real Cortex-M0+ silicon, no debugger.

    This is the measurement `docs/claim4.md` says the QEMU run could not be:
    QEMU is a functional model, and a functional model agreeing with the
    reference says nothing about the part. The differ is unchanged and the line
    protocol is unchanged, because `port/bare/dm_harness.c` is one file serving
    both machines -- which is the only reason a silicon trace and an emulated
    one are comparable at all.

    The corpus goes in packed into the image's own UF2 and the trace comes back
    over a UART pin; `scripts/pico.py` explains why there is no channel inward.
    The image is unused here beyond existing -- the packer reads it -- so it is
    accepted for symmetry with the other runners rather than passed on.
    """
    from scripts import pico

    if not image.exists():
        raise SystemExit(f"{image} does not exist -- run `make -C port pico` first")
    try:
        text = pico.run(programs, fuel=fuel, no_build=True)
    except pico.PicoError as failure:
        raise SystemExit(str(failure)) from failure
    return parse_port_traces(text, len(programs))


def build_harness(target: str) -> None:
    subprocess.run(["make", "-s", "-C", str(PORT), target], check=True)


# --------------------------------------------------------------------------
# corpora


def edge_cases() -> list[bytes]:
    """One program per fault path, plus the boundaries the fuzzer hits rarely.

    Written out rather than generated because each one is an assertion about the
    specification, and a name in a list is easier to check against
    `dm/vm/interp.py` than a seed is.
    """
    halt, fill, endrep = bytes([Op.HALT]), bytes([Op.FILL]), bytes([Op.ENDREP])
    move = lambda x, y: bytes([Op.MOVE, x, y])
    line = lambda x, y: bytes([Op.LINE, x, y])
    repeat = lambda n, dx, dy: bytes([Op.REPEAT, n, dx & 0xFF, dy & 0xFF])
    curve = lambda *a: bytes([Op.CURVE, *a])

    return [
        b"",                                          # no halt, no instructions
        halt,                                         # the shortest valid program
        bytes([0xFF]),                                # unknown opcode
        bytes([Op.MOVE, 0x10]),                       # truncated operands
        bytes([Op.CURVE, 1, 2, 3]),                   # truncated, longest instruction
        endrep + halt,                                # unmatched ENDREP
        repeat(3, 4, 4) + move(1, 1) + line(2, 2) + halt,   # unterminated REPEAT
        repeat(0, 1, 1) + move(1, 1) + endrep + halt,       # zero count
        repeat(1, 0, 0) * 5 + halt,                   # one past MAX_REPEAT_DEPTH
        repeat(0, 1, 1) * 5 + halt,                   # zero count *and* depth overflow
        bytes([Op.CALL, 0]) + halt,                   # unsupported
        bytes([Op.COLOR, 7]) + move(1, 1) + line(9, 9) + halt,  # reserved, no-op
        move(1, 1) + halt,                            # single-point path, discarded
        move(1, 1) + line(2, 2) + fill + halt,        # two-point FILL, discarded
        move(1, 1) + line(2, 2) + line(3, 3) + fill + halt,    # three-point region
        line(5, 5) + halt,                            # LINE with no MOVE: opens at (0,0)
        curve(1, 2, 3, 4, 5, 6) + halt,               # CURVE with no MOVE
        bytes([Op.WIDTH, 0]) + move(1, 1) + line(2, 2) + halt,  # WIDTH clamps to 1
        move(1, 1) + line(2, 2) + bytes([Op.WIDTH, 9]) + line(3, 3) + halt,  # flush order
        bytes([Op.CIRCLE, 200]) + halt,               # disc at the origin
        # Clamping, in both directions, through a nested REPEAT's offset.
        repeat(200, 127, 127) + move(250, 250) + line(255, 255) + endrep + halt,
        repeat(200, -128, -128) + move(5, 5) + line(0, 0) + endrep + halt,
        # A repeat whose body ends mid-path, so the flush happens on ENDREP.
        repeat(4, 10, 0) + move(1, 1) + line(2, 2) + endrep + halt,
    ]


#: Opcodes the reference has and `port/src/dm_vm.c` does not. **Empty since
#: 2026-08-11**, when the transform tier landed on the device: the port now
#: implements every opcode the reference does, so the fuzzer draws from the whole
#: table again and a divergence is a divergence rather than a known gap.
#:
#: Kept rather than deleted because the next opcode will need it, and because a
#: filter that has to be *re-derived* under time pressure is how a differ ends up
#: forgiving a real mismatch. `tests/test_port.py:NOT_PORTED` is the same set,
#: asserted against the C header.
UNPORTED: frozenset[int] = frozenset()


def fuzz(n: int, seed: int) -> list[bytes]:
    """Two fuzzers, because they fail in different places.

    Uniform bytes reach UNKNOWN_OPCODE almost immediately and test little else.
    The structured fuzzer emits real instructions with real operand counts, so
    it runs deep enough to reach the repeat machinery, and truncates a fraction
    of programs mid-instruction so TRUNCATED is reached at every instruction
    width rather than only at the widest.

    Both draw from the ISA the *port* implements. Random bytes that happen to
    spell an unported opcode are dropped, because the reference would execute
    them and the port would fault -- which is a true statement about the gap and
    a false one about conformance.
    """
    rng = random.Random(seed)
    sizes = {op: spec.size for op, spec in SPECS.items() if int(op) not in UNPORTED}
    programs: list[bytes] = []

    for _ in range(n // 2):
        program = bytes(rng.randrange(256) for _ in range(rng.randrange(0, 64)))
        if not UNPORTED & set(program):
            programs.append(program)

    for _ in range(n - n // 2):
        out = bytearray()
        for _ in range(rng.randrange(1, 40)):
            op = rng.choice(list(sizes))
            out.append(int(op))
            out += bytes(rng.randrange(256) for _ in range(sizes[op] - 1))
        if rng.random() < 0.25 and len(out) > 1:
            del out[rng.randrange(1, len(out)):]
        if not UNPORTED & set(out):
            programs.append(bytes(out))

    return programs


def corpora(n: int, quick: bool) -> dict[str, list[bytes]]:
    from dm.data import synthetic
    from dm.isa.unroll import unroll

    out: dict[str, list[bytes]] = {
        "edge cases": edge_cases(),
        "fuzz": fuzz(n, seed=0),
        "tier A L0": synthetic.dataset(n, seed=1, tier=Tier.L0, depth=2),
        "tier A L1": synthetic.dataset(n, seed=2, tier=Tier.L1, depth=2),
    }
    out["tier A flat"] = [
        flat for p in out["tier A L1"] if (flat := unroll(p)) is not None
    ]
    # The transform tier on *structured* programs, which is what the fuzzer
    # cannot supply: it reaches XFORM and REPEATX often enough to hit the fault
    # paths and almost never in a shape that draws anything. Both forms go in --
    # the structured program exercises the tier, and its own flat expansion is
    # the drawing the device has to agree with.
    try:
        from dm.data import composed

        scenes = composed.build(min(n, 400), "valid", seed=11)
        out["composed L2"] = [s.structured for s in scenes]
        out["composed flat"] = [s.flat for s in scenes]
    except Exception as exc:  # noqa: BLE001 -- an absent QuickDraw cache is not a failure
        print(f"  composed skipped: {exc}")
    if quick:
        return out

    try:
        from dm.data import quickdraw

        out["quickdraw"] = quickdraw.load(("cat", "dog", "bus", "car", "tree"), "valid", limit=n)
    except Exception as exc:  # noqa: BLE001 -- an absent cache is not a failure here
        print(f"  quickdraw skipped: {exc}")
    try:
        from dm.data import tabler

        # The val half, which is grouped by icon family and so is the more
        # varied of the two per program read.
        out["tabler"] = tabler.split()[1][:n]
    except Exception as exc:  # noqa: BLE001
        print(f"  tabler skipped: {exc}")
    return out


# --------------------------------------------------------------------------


def describe(trace: Trace) -> str:
    return (
        f"steps={trace.steps} halted={trace.halted} strokes={len(trace.strokes)} "
        f"discs={len(trace.discs)} regions={len(trace.regions)} faults={list(trace.faults)}"
    )


def check(runner, harness: Path, name: str, programs: list[bytes], fuel: int) -> int:
    """Returns the number of divergences, and prints the first one in full."""
    port = runner(harness, programs, fuel)
    failures = 0
    for i, (program, actual) in enumerate(zip(programs, port)):
        expected = reference_trace(program, fuel)
        if actual == expected:
            continue
        failures += 1
        if failures == 1:
            print(f"\n  DIVERGED on {name}[{i}], fuel={fuel}, {len(program)} bytes")
            print(f"    program   {program.hex()}")
            print(f"    reference {describe(expected)}")
            print(f"    port      {describe(actual)}")
            for field in Trace.__dataclass_fields__:
                a, b = getattr(expected, field), getattr(actual, field)
                if a != b:
                    print(f"    first differing field: {field}\n      ref  {a}\n      port {b}")
                    break
    return failures


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--programs", type=int, default=1000, help="programs per generated corpus")
    ap.add_argument("--quick", action="store_true", help="synthetic corpora only")
    ap.add_argument("--no-build", action="store_true")
    # The sanitiser build is the same C with UBSan and ASan on. It matters
    # because conformance runs *native* code while the part runs Thumb: what
    # carries a host result over to a target it has never executed on is the
    # absence of undefined behaviour, not the passing diff. Signed overflow and
    # out-of-range shifts are exactly the constructs that can differ between two
    # targets while both look right on one.
    ap.add_argument("--sanitise", action="store_true",
                    help="run the UBSan/ASan build instead of the optimised one")
    # The Thumb build, on an emulated Cortex-M0. Slower by orders of magnitude
    # -- every byte of output crosses a semihosting trap -- so it is opt-in and
    # usually run on a smaller corpus than the native sweep.
    ap.add_argument("--qemu", action="store_true",
                    help="run the Cortex-M0 build under QEMU instead of natively")
    # The part itself. Slower again than QEMU, and for a different reason: there
    # is no debugger, so the trace leaves on a 115200-baud UART pin and the
    # corpus is re-flashed per batch. The only run in this project whose result
    # is about silicon rather than about a model of it.
    ap.add_argument("--pico", action="store_true",
                    help="run the Cortex-M0+ build on an RP2040, trace over UART")
    args = ap.parse_args()

    chosen = [name for name in ("qemu", "pico", "sanitise") if getattr(args, name)]
    if len(chosen) > 1:
        raise SystemExit(f"{', '.join('--' + c for c in chosen)} build different "
                         "images; pick one")

    target, harness_name, runner, where = {
        "qemu": ("qemu", "dm_qemu.elf", run_qemu, "cortex-m0 under qemu"),
        "pico": ("pico", "dm_pico.elf", run_pico, "rp2040 cortex-m0+, sram, uart"),
        "sanitise": ("host-san", "dm_trace_san", run_native, "native, ubsan"),
    }.get(chosen[0] if chosen else "", ("host", "dm_trace", run_native, "native"))
    harness = PORT / "build" / harness_name
    if not args.no_build:
        build_harness(target)

    print(f"port {harness.relative_to(ROOT)} ({where}), scale 1/{ONE}, "
          f"curve steps {CURVE_STEPS}\n")
    total = failures = 0
    for name, programs in corpora(args.programs, args.quick).items():
        if not programs:
            continue
        # A second pass at a fuel far below any program's instruction count, so
        # OUT_OF_FUEL is reached mid-path rather than only in the fuzz corpus --
        # it is the one fault the reference can raise at any instruction.
        for fuel in (DEFAULT_FUEL, 25):
            bad = check(runner, harness, name, programs, fuel)
            failures += bad
            total += len(programs)
            flag = "FAIL" if bad else "ok"
            print(f"  {flag:>4}  {name:<14} {len(programs):>5} programs  fuel {fuel:>6}")

    print(f"\n{total - failures}/{total} traces identical")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
