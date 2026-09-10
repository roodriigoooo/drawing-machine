"""The drawing arrives from the board, point by point, while the wire carries it.

**This module is the demo's only geometry source.** `docs/claim4-bringup.md`
§11.1 states the honesty contract and it is worth restating where it is
enforced rather than only where it was written: the reference VM may compute an
equality assertion, and it must never supply the geometry being rendered.
Rendering a host trace beside a "Pico" badge would demonstrate the renderer.

So the path is:

    sampled bytecode -> UF2 -> SRAM -> RP2040 VM -> UART fixed-point trace
        -> this parser -> events -> page

and a mismatch, fault, incomplete frame or failed BOOTSEL return is surfaced as
a refusal rather than repaired with host geometry.

## Why the loop needs no hands

Each corpus is packed with `passes=1`, so the firmware calls the bootrom's
`reset_usb_boot` when its pass ends and the board returns to BOOTSEL by itself.
Only the *first* load of a session needs someone to hold the button; after that
the host writes the next word's UF2 onto a drive that remounted on its own.
That is what makes a type-a-word-and-watch loop possible at all on a part with
no debugger and no inward channel.

## Coordinates

The wire carries integers in units of `1 / (1 << FRAC_BITS)` canvas pixels --
the fixed-point form the port emits, whose scale is the curve denominator
rather than a choice. Events carry both: the raw integer, which is what the
equality assertion compares, and the canvas float, which is what the page
draws.
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import pico, uf2  # noqa: E402
from scripts.conformance import FRAC_BITS, ONE, Trace, reference_trace  # noqa: E402

#: Fuel for one demo drawing. The same value the conformance sweep's headline
#: runs used, so a program that executes here executes there.
FUEL = 100_000


@dataclass
class DrawEvent:
    """One thing that happened on the wire, ready for the page's event stream."""

    kind: str
    data: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {"t": self.kind, **self.data}


class DeviceRefusal(RuntimeError):
    """The device did not produce a frame this demo is allowed to render."""


def _canvas(value: int) -> float:
    return value / ONE


class DeviceSession:
    """One open serial port, one built image, many drawings.

    The port is opened once and held. `scripts/pico.py` documents why it must be
    open *before* a UF2 is written -- the board starts executing the instant its
    last block lands, so a host that opened the port afterwards would race the
    output of a short corpus -- and a demo that reopened per word would lose that
    race on every drawing rather than on some of them.

    The batch counter increments per drawing. A stale frame from the previous
    word can still be draining out of the adapter when the next one is written,
    and matching by position instead of by batch identity would animate the
    previous drawing under the new word.
    """

    #: Set once the port is open; named here so a caller can read it before
    #: `__enter__` without an AttributeError.
    device_path: Path | None = None

    def __init__(self, port: Path | None = None, build: bool = True,
                 timeout_s: float = 60.0, tamper=None) -> None:
        self.timeout_s = timeout_s
        self._batch = 0
        self._fd: int | None = None
        self._port = port
        self._build = build
        self._tamper = tamper

    def __enter__(self) -> "DeviceSession":
        if self._build:
            # Always `-B`: a bring-up run leaves a checkpointed image whose
            # timestamp is newer than every source, and `make` cannot see a
            # `-D`. `scripts/pico.py` carries the full scar.
            pico.build()
        device = pico.choose_port(self._port)
        self._fd = pico.open_serial(device)
        self.device_path = device
        return self

    def __exit__(self, *exc) -> None:
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None

    #: What the record calls this geometry's origin. The page shows the exact-match
    #: badge for this value and for no other.
    source = "rp2040"

    def draw(self, program: bytes) -> Iterator[DrawEvent]:
        """Execute one program on silicon and yield its geometry as it arrives.

        The final event carries the parsed device `Trace` and whether it equals
        the host reference. Callers that want the badge read that event; callers
        that want the picture have already drawn it from the points.
        """
        if self._fd is None:
            raise RuntimeError("use DeviceSession as a context manager")
        batch = self._batch
        self._batch += 1

        if len(program) + 4 > uf2.capacity():
            raise DeviceRefusal(
                f"a {len(program)}-byte program does not fit the reserved corpus region"
            )

        started = time.monotonic()
        # `pico.load` blocks while the BOOTSEL volume appears and settles, which
        # is seconds the page would otherwise spend looking frozen. The event
        # goes out first so the view can say what it is waiting for.
        yield DrawEvent("loading", {"bytes": len(program), "batch": batch})
        pico.load(uf2.pack([program], fuel=FUEL, reps=0, quiet=False, passes=1,
                           batch=batch))
        yield DrawEvent("loaded", {"seconds": round(time.monotonic() - started, 3)})
        lines = pico.stream_pass(self._fd, self.timeout_s, batch=batch)
        yield from events(self._tamper(lines) if self._tamper else lines,
                          program, started)


class NativeRehearsal:
    """The identical protocol, produced by the native harness instead of a board.

    It exists so the page, the event stream, the renderer and the refusal paths
    can be built and tested with no hardware attached, and so a demo can be
    rehearsed on a train. It is **not** a fallback: a record it produces is
    stamped `source: "native"`, the page refuses to show the silicon badge for
    anything but `rp2040`, and `tests/test_demo.py` asserts that a native record
    cannot acquire one.

    `docs/claim4-bringup.md` §8 already wrote the rule this obeys: *a visual demo
    must render the parsed device trace; rendering a host VM trace and placing a
    "Pico" badge beside it would demonstrate the renderer, not Claim 4.* The
    honest way to have a hardware-free rehearsal is to make the difference
    machine-readable rather than to make it invisible.
    """

    source = "native"
    device_path = None

    def __init__(self, harness: Path | None = None, build: bool = True,
                 tamper=None) -> None:
        self.harness = harness or (ROOT / "port" / "build" / "dm_trace")
        self._build = build
        self._tamper = tamper

    def __enter__(self) -> "NativeRehearsal":
        if self._build and not self.harness.exists():
            import subprocess
            subprocess.run(["make", "-s", "-C", str(ROOT / "port"), "host"], check=True)
        if not self.harness.exists():
            raise DeviceRefusal(f"{self.harness} does not exist -- run `make -C port host`")
        return self

    def __exit__(self, *exc) -> None:
        return None

    def draw(self, program: bytes) -> Iterator[DrawEvent]:
        import struct
        import subprocess

        started = time.monotonic()
        proc = subprocess.run([str(self.harness), str(FUEL)],
                              input=struct.pack("<I", len(program)) + program,
                              capture_output=True, check=False)
        if proc.returncode != 0:
            raise DeviceRefusal(f"the native harness failed ({proc.returncode})")
        yield DrawEvent("loaded", {"seconds": round(time.monotonic() - started, 3)})
        lines = _lines(proc.stdout.decode())
        yield from events(self._tamper(lines) if self._tamper else lines,
                          program, started)


def _lines(text: str):
    """Native-harness stdout as the same `(tag, args, line)` stream the wire yields."""
    started = False
    for raw in text.splitlines():
        line = raw.strip()
        if not started:
            started = line.startswith("BEGIN ")
            continue
        tag, _, remainder = line.partition(" ")
        yield tag, remainder.split(), line
        if tag == "END":
            return


def events(lines, program: bytes, started: float) -> Iterator[DrawEvent]:
    """`(tag, args)` from either source, reassembled into geometry and events.

    One reassembler for both sources is the point: a rehearsal that parsed the
    protocol differently from the board would test the rehearsal.
    """
    strokes: list[tuple[int, tuple[int, ...]]] = []
    discs: list[tuple[int, int, int, int]] = []
    regions: list[tuple[int, ...]] = []
    faults: list[tuple[str, int]] = []
    path: list[int] = []
    header: dict[str, int] = {}
    seen_program = False
    finished: Trace | None = None
    count: int | None = None

    received: list[str] = []
    for tag, args, raw in lines:
        received.append(raw)
        if tag == "frac_bits":
            header["frac_bits"] = int(args[0])
        elif tag == "curve_steps":
            header["curve_steps"] = int(args[0])
        elif tag.startswith("#"):
            # A second `#` in a one-program corpus would mean the frame
            # carried a drawing this word did not ask for.
            if seen_program:
                raise DeviceRefusal("the device reported more than one program")
            seen_program = True
            _require_header(header)
            yield DrawEvent("begin", dict(header))
        elif tag == "b":
            path = []
        elif tag == "p":
            x, y = int(args[0]), int(args[1])
            path += [x, y]
            yield DrawEvent("point", {"x": _canvas(x), "y": _canvas(y),
                                      "fx": x, "fy": y})
        elif tag == "e":
            kind = args[0]
            if kind == "stroke":
                width = int(args[2])
                strokes.append((width, tuple(path)))
                yield DrawEvent("stroke", {"width": width,
                                           "points": _pairs(path)})
            elif kind == "region":
                regions.append(tuple(path))
                yield DrawEvent("region", {"points": _pairs(path)})
            else:
                raise DeviceRefusal(f"unknown path terminator {kind!r} on the wire")
            path = []
        elif tag == "d":
            disc = (int(args[0]), int(args[1]), int(args[2]), int(args[3]))
            discs.append(disc)
            yield DrawEvent("disc", {"cx": _canvas(disc[0]), "cy": _canvas(disc[1]),
                                     "r": disc[2], "width": disc[3]})
        elif tag == "f":
            fault = (args[0], int(args[1]))
            faults.append(fault)
            yield DrawEvent("fault", {"kind": fault[0], "pc": fault[1]})
        elif tag == "=":
            finished = Trace(tuple(strokes), tuple(discs), tuple(regions),
                             tuple(faults), int(args[0]), args[1] == "1")
        elif tag == "END":
            count = int(args[0])
        else:
            raise DeviceRefusal(f"unparsable line on the wire: {tag} {' '.join(args)}")

    if finished is None or count is None:
        raise DeviceRefusal("the frame ended without a complete trace record")
    if count != 1:
        raise DeviceRefusal(f"the device reported {count} programs, not 1")

    expected = reference_trace(program, FUEL)
    yield DrawEvent("done", {
        "steps": finished.steps,
        "halted": finished.halted,
        "strokes": len(finished.strokes),
        "regions": len(finished.regions),
        "discs": len(finished.discs),
        "faults": [{"kind": k, "pc": pc} for k, pc in finished.faults],
        "matches_reference": finished == expected,
        "seconds": round(time.monotonic() - started, 3),
        "frac_bits": FRAC_BITS,
        "trace_text": "\n".join(received),
        "geometry": {
            "strokes": [{"width": w, "points": _pairs(list(pts))} for w, pts in strokes],
            "regions": [{"points": _pairs(list(pts))} for pts in regions],
            "discs": [{"cx": _canvas(c[0]), "cy": _canvas(c[1]), "r": c[2],
                       "width": c[3]} for c in discs],
        },
    })


def tamper_first_point(lines):
    """Move one coordinate by one canvas pixel, on purpose.

    The refusal path is the least visible and most important guarantee this
    demo makes, and a guarantee nobody has watched fail is a guarantee nobody
    believes. This makes the failure a one-command demonstration: the wire is
    perturbed by exactly `1 << FRAC_BITS` -- one whole canvas pixel, so the
    change is real rather than a rounding argument -- and the page refuses.

    It is never on by default, it is stamped into the record's `note` by the
    caller, and it cannot produce a record at all, because the frame it
    produces is the one `capture.draw` declines to write.
    """
    perturbed = False
    for tag, args, raw in lines:
        if tag == "p" and not perturbed:
            perturbed = True
            moved = f"p {int(args[0]) + ONE} {args[1]}"
            yield tag, [str(int(args[0]) + ONE), args[1]], moved
            continue
        yield tag, args, raw


def _pairs(path: list[int]) -> list[list[float]]:
    return [[_canvas(path[i]), _canvas(path[i + 1])] for i in range(0, len(path), 2)]


def _require_header(header: dict) -> None:
    """The port's fixed-point header, checked before a single point is drawn.

    A port built for a different `curve_steps` is a different specification, and
    its coordinates would land on the canvas looking entirely plausible.
    `scripts/conformance.py` refuses that case for a measurement; a demo that
    did not would be drawing geometry from an ISA it never verified.
    """
    from scripts.conformance import CURVE_STEPS

    if header.get("frac_bits") != FRAC_BITS or header.get("curve_steps") != CURVE_STEPS:
        raise DeviceRefusal(
            f"the device reports frac_bits={header.get('frac_bits')} "
            f"curve_steps={header.get('curve_steps')}; this host is "
            f"{FRAC_BITS}/{CURVE_STEPS}, which is a different specification"
        )
