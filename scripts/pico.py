#!/usr/bin/env python3
"""Running bytecode on a Raspberry Pi Pico with nothing but a USB cable.

Claim 4's silicon half needs the interpreter executing on a real Cortex-M0+.
The standard way to arrange that is SWD, which needs a second piece of hardware
— a debug probe, or a spare Pico flashed as one. This module is the path that
needs neither, and the constraint it works around is worth stating plainly:

| direction | available? | how |
|---|---|---|
| host → part | yes | BOOTSEL, then a UF2 dropped on a USB drive |
| part → host | **no** | once the image runs, BOOTSEL is gone |

So getting a program *in* is easy and getting anything *out* is the whole
problem. Three decisions follow from it, and together they are the design:

1. **The corpus travels with the image.** `scripts/uf2.py` packs both into one
   UF2 at the address `port/pico/link.ld` reserves, so the part wakes with its
   work already in SRAM. Nothing needs to reach it after it starts.
2. **Output leaves on a pin.** `GP0` is UART TX at 3.3 V — the simplest thing a
   chip can emit, no USB stack, no debugger. Any USB-serial adapter reads it,
   including an Arduino UNO with `RESET` jumpered to `GND`, which turns the
   board into a bare bridge.
3. **The firmware repeats forever.** It cannot know when a reader attached, so
   `port/bare/dm_harness.c` brackets each pass with `BEGIN`/`END` and this
   module keeps the first *whole* pass it sees. A reader that attached
   mid-drawing would otherwise capture something well-formed and truncated.

    python3 scripts/pico.py --ports          # which serial device is the adapter
    python3 scripts/pico.py --loopback       # the host side alone, no Pico involved
    python3 scripts/pico.py --listen         # whatever is on the wire, framed or not
    python3 scripts/pico.py --bss-check      # full .bss clear + one-HALT reboot gate
    python3 scripts/pico.py --run            # one HALT, on silicon, traced back

The middle two exist because `--run` fails the same way for four different
reasons — dead adapter, wrong pin, wrong ground, wrong baud — and each retry
costs a BOOTSEL press. `--loopback` proves the host, the driver, the adapter and
the baud rate with the Pico unplugged; `--listen` shows the bytes that are
arriving when framing fails, which is the only way to tell a baud error (bytes,
never framed) from a wiring error (no bytes at all).

**Nothing is ever written to the board's flash.** The image is linked into SRAM,
which is what makes the cycle count core-limited rather than a measurement of
Raspberry Pi's XIP configuration, and it also means a power cycle restores the
part exactly as it shipped.
"""

from __future__ import annotations

import argparse
import errno
import os
import re
import select
import subprocess
import sys
import termios
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

ROOT = Path(__file__).resolve().parent.parent
PORT = ROOT / "port"
IMAGE = PORT / "build" / "dm_pico.elf"

#: What an RP2040 in BOOTSEL mode mounts as.
BOOTSEL_VOLUME = Path("/Volumes/RPI-RP2")

# macOS has produced EIO both while opening a newly remounted BOOTSEL volume and
# while attempting its first write. Both are retryable for the bounded mount
# readiness interval, but a write remains non-retryable after any byte succeeds.
UF2_OPEN_TIMEOUT_S = 5.0
UF2_OPEN_RETRY_S = 0.1
TRANSIENT_UF2_OPEN_ERRNOS = frozenset({
    errno.EACCES,
    errno.EBUSY,
    errno.EIO,
    errno.ENOENT,
    errno.EROFS,
})

TRANSIENT_UF2_ZERO_WRITE_ERRNOS = TRANSIENT_UF2_OPEN_ERRNOS

#: Must match `DM_UART_BAUD` in `port/pico/machine.c`. 115200 rather than the
#: 750,000 the clock could reach exactly, because this is read by whatever
#: adapter the user has and 115200 is the rate every driver supports —
#: `termios` on macOS does not even name a constant for the faster one.
BAUD = 115200

#: Serial devices that are never the adapter. macOS presents Bluetooth and the
#: debug console as `/dev/cu.*` too, and opening one of those waits forever.
NOT_ADAPTERS = ("Bluetooth", "debug-console", "wlan-debug")

#: What a USB-attached serial adapter is called on macOS: `usbmodem` for a
#: CDC-ACM device (an official UNO's ATmega16U2), `usbserial`/`wchusbserial` for
#: a CH340 clone, `SLAB_USBtoUART` for a CP210x. Used to *prefer* a candidate
#: rather than to filter, because a paired Bluetooth device with an arbitrary
#: name — `/dev/cu.RoBose` on this machine — is otherwise indistinguishable from
#: an adapter, and it would make every unattended run fail as ambiguous.
USB_ADAPTERS = ("usbmodem", "usbserial", "wchusbserial", "SLAB_USBtoUART")

#: Names the port outright, for the callers that have no `--port` flag of their
#: own: `scripts/conformance.py --pico` and `scripts/cycles.py` both reach the
#: board through this module and neither should have to grow an argument to
#: disambiguate a Bluetooth headset.
PORT_ENV = "DM_PICO_PORT"

PASS = re.compile(r"^BEGIN (\d+)\n(.*?)^END (\d+)\n", re.DOTALL | re.MULTILINE)
# Stage 17 is successful only when the UART carries this complete record. The
# marker is emitted before the frame is decoded, and the host supplies the
# expected PC from the stage ELF; a reboot by itself is not enough because a
# reset can also follow an unrelated path through the boot ROM.
FAULT = re.compile(
    r"\A!\nFAULT pc=0x([0-9a-fA-F]{8}) lr=0x([0-9a-fA-F]{8})\n\Z"
)
BSS_SUCCESS = b"BSS_CLEAR ok\nBEGIN 0\nEND 1\n"


def bringup_stage_success(stage: int, reached_bootsel: bool, seen: bytes,
                          expected_pc: int | None = None) -> bool:
    """Apply the extra evidence required by the stage-17 instrument test."""
    if stage != 17:
        return reached_bootsel
    text = seen.decode("ascii", "replace")
    record = FAULT.fullmatch(text)
    return (reached_bootsel and record is not None and expected_pc is not None and
            int(record.group(1), 16) == expected_pc)


def bss_check_success(reached_bootsel: bool, seen: bytes) -> bool:
    """Require the diagnostic verdict, the quiet one-HALT pass, and reboot."""
    return reached_bootsel and seen == BSS_SUCCESS


class PicoError(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# getting the image in


#: Whether this process has already delivered a UF2. Before the first one the
#: board can only be in BOOTSEL because someone put it there; after it, every
#: further remount is the firmware's own `reset_usb_boot` at the end of a pass.
#:
#: The distinction is not cosmetic. `docs/claim4-bringup.md` §8 records that
#: unplugging in response to the generic prompt during an automatic return
#: turns readiness into `ENOENT` and kills the sweep — so a prompt that says
#: "hold BOOTSEL and plug in the cable" while the board is *already* coming
#: back by itself is an instruction to break the run, printed by the run.
_DELIVERED_A_UF2 = False


def bootsel_volume(timeout_s: float = 120.0) -> Path:
    """Wait for the board to appear as a USB drive, prompting for the right thing.

    Polled rather than required up front because entering BOOTSEL is a physical
    act — hold the button, plug the cable — and a script that fails instantly
    with "not found" makes the user run it twice for no reason.

    Which prompt is printed depends on whose job the remount is. The first load
    of a process needs a person; every one after it needs a person to do
    nothing at all, and says so.
    """
    if BOOTSEL_VOLUME.exists():
        return BOOTSEL_VOLUME
    if _DELIVERED_A_UF2:
        print(f"  waiting for {BOOTSEL_VOLUME} to come back — the board returns "
              "by itself at the end of a pass. Wait; do not unplug.")
    else:
        print(f"  waiting for {BOOTSEL_VOLUME} — hold the Pico's BOOTSEL button, "
              "plug in its USB cable, then release")
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if BOOTSEL_VOLUME.exists():
            return BOOTSEL_VOLUME
        time.sleep(0.5)
    if _DELIVERED_A_UF2:
        raise PicoError(
            f"{BOOTSEL_VOLUME} did not come back within {timeout_s:.0f}s. The "
            "board should have rebooted to BOOTSEL by itself when the pass "
            "ended; if it was unplugged during the return, hold BOOTSEL while "
            "reconnecting and start again."
        )
    raise PicoError(
        f"{BOOTSEL_VOLUME} never appeared. Hold BOOTSEL *while* connecting the "
        "cable, and check the cable carries data rather than power only."
    )


def open_uf2_target(
    target: Path,
    timeout_s: float = UF2_OPEN_TIMEOUT_S,
) -> int:
    """Open the UF2 destination after a newly mounted BOOTSEL volume settles.

    Disk Arbitration can create /Volumes/RPI-RP2 before the mounted FAT
    filesystem accepts file creation. Retry only mount-transition errors for a
    bounded interval; unrelated failures remain immediate.
    """
    deadline = time.monotonic() + timeout_s

    while True:
        try:
            return os.open(
                target,
                os.O_WRONLY | os.O_CREAT | os.O_TRUNC,
                0o644,
            )
        except OSError as failure:
            if failure.errno not in TRANSIENT_UF2_OPEN_ERRNOS:
                raise PicoError(f"could not open {target}: {failure}") from failure

            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise PicoError(
                    f"{target.parent} appeared but did not become writable "
                    f"within {timeout_s:.1f}s; last error: {failure}"
                ) from failure

            time.sleep(min(UF2_OPEN_RETRY_S, remaining))


def write_uf2_target(
    target: Path,
    uf2: bytes,
    timeout_s: float = UF2_OPEN_TIMEOUT_S,
) -> None:
    """Write a UF2 once the newly mounted BOOTSEL volume accepts data.

    A failed first write is safe to retry because no UF2 block reached the
    bootrom. Once any byte has been accepted, failures remain fatal: retrying a
    partially delivered UF2 would conceal a materially different condition.
    """
    deadline = time.monotonic() + timeout_s

    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise PicoError(
                f"{target.parent} did not become writable within "
                f"{timeout_s:.1f}s"
            )

        handle = open_uf2_target(target, timeout_s=remaining)
        written = 0
        caught: OSError | None = None

        try:
            while written < len(uf2):
                count = os.write(handle, uf2[written:written + 65536])
                if count <= 0:
                    raise OSError(errno.EIO, "zero-byte UF2 write")
                written += count
        except OSError as failure:
            caught = failure
        finally:
            try:
                os.close(handle)
            except OSError:
                # A successful UF2 transfer reboots and unmounts the board
                # underneath the descriptor.
                pass

        if caught is None:
            return

        if written != 0 or caught.errno not in TRANSIENT_UF2_ZERO_WRITE_ERRNOS:
            raise PicoError(
                f"writing {target} failed after {written:,} bytes: {caught}"
            ) from caught

        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise PicoError(
                f"{target.parent} opened but did not accept its first byte "
                f"within {timeout_s:.1f}s; last error: {caught}"
            ) from caught

        time.sleep(min(UF2_OPEN_RETRY_S, remaining))


def load(uf2: bytes, timeout_s: float = 120.0) -> None:
    """Write the UF2 to the mounted board.

    The board reboots and unmounts the moment the last block lands, so the close
    can fail on a volume that no longer exists. That is success, not an error,
    and distinguishing it from a real write failure is the only subtlety here:
    a real one fails on the *write*, which is still checked.
    """
    global _DELIVERED_A_UF2

    volume = bootsel_volume(timeout_s)
    target = volume / "dm.uf2"
    write_uf2_target(target, uf2)
    # Set only after the write succeeded. A load that failed leaves the board
    # wherever it was, and the next attempt is still a first delivery.
    _DELIVERED_A_UF2 = True
    print(f"  loaded {len(uf2):,} bytes ({len(uf2) // 512} blocks) — the board is running")


# ---------------------------------------------------------------------------
# getting the trace out


def serial_ports() -> list[Path]:
    return sorted(
        path for path in Path("/dev").glob("cu.*")
        if not any(skip in path.name for skip in NOT_ADAPTERS)
    )


def open_serial(path: Path) -> int:
    """Raw 8N1 at `BAUD`, via `termios` rather than a dependency.

    `O_NONBLOCK` on open matters on macOS: a `/dev/cu.*` open otherwise blocks
    waiting for carrier detect, which a USB-serial adapter with nothing asserting
    DCD never provides — the script would hang with no output rather than fail.
    """
    fd = os.open(path, os.O_RDWR | os.O_NOCTTY | os.O_NONBLOCK)
    iflag, oflag, cflag, lflag, _, _, cc = termios.tcgetattr(fd)
    iflag &= ~(termios.IGNBRK | termios.BRKINT | termios.PARMRK | termios.ISTRIP |
               termios.INLCR | termios.IGNCR | termios.ICRNL | termios.IXON)
    oflag &= ~termios.OPOST
    lflag &= ~(termios.ECHO | termios.ECHONL | termios.ICANON | termios.ISIG |
               termios.IEXTEN)
    cflag &= ~(termios.CSIZE | termios.PARENB | termios.CRTSCTS)
    cflag |= termios.CS8 | termios.CREAD | termios.CLOCAL
    cc[termios.VMIN] = 0
    cc[termios.VTIME] = 0
    termios.tcsetattr(fd, termios.TCSANOW,
                      [iflag, oflag, cflag, lflag, BAUD, BAUD, cc])
    termios.tcflush(fd, termios.TCIFLUSH)
    return fd


def usb_ports() -> list[Path]:
    """The candidates whose names say they arrived over USB.

    Preferred over the full list rather than substituted for it: a USB-TTL
    adapter this table has not seen still works if it is the only device, and
    the table only has to be right about the ones it does name.
    """
    return [p for p in serial_ports() if any(tag in p.name for tag in USB_ADAPTERS)]


def choose_port(explicit: Path | None) -> Path:
    """The adapter, by argument, by environment, or by being the obvious one.

    The environment variable exists because two callers that need a board —
    `conformance.py --pico` and `cycles.py` — reach it through `run()` and have
    no port argument of their own. Growing one on each would put the same flag
    in three places; `DM_PICO_PORT` puts the answer where all three read it.
    """
    if explicit:
        return explicit
    named = os.environ.get(PORT_ENV)
    if named:
        chosen = Path(named)
        if not chosen.exists():
            raise PicoError(f"{PORT_ENV}={named} does not exist")
        return chosen
    candidates = serial_ports()
    if not candidates:
        raise PicoError(
            "no USB-serial adapter found under /dev/cu.*. Plug in the Arduino "
            "(or any USB-TTL adapter) and check it enumerates."
        )
    usb = usb_ports()
    if len(usb) == 1:
        return usb[0]
    remaining = usb or candidates
    if len(remaining) > 1:
        raise PicoError(
            "more than one serial device is present; name the right one with "
            f"--port or {PORT_ENV}. Candidates: "
            f"{', '.join(str(c) for c in remaining)}"
        )
    return remaining[0]


def read_pass(fd: int, timeout_s: float, batch: int = 0,
              progress: bool = True) -> tuple[str, int]:
    """The first complete `BEGIN`..`END` block on the wire.

    Complete is the operative word. A pass can begin before this call does --
    the board starts the instant its last UF2 block lands -- so the stream may
    open mid-drawing, and taking the first bytes seen would yield a trace that
    begins in the middle and parses perfectly all the way to the end.
    """
    buffer = bytearray()
    quiet_since = last_report = time.monotonic()
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        ready, _, _ = select.select([fd], [], [], 0.5)
        if ready:
            chunk = os.read(fd, 1 << 16)
            if chunk:
                buffer += chunk
                quiet_since = time.monotonic()
        for found in PASS.finditer(buffer.decode("ascii", "replace")):
            if int(found.group(1)) != batch:
                # A block from the *previous* batch, still draining out of the
                # adapter's buffer when this one was written. Matching by
                # position instead would diff one batch against another's
                # reference programs and report the divergence as real.
                continue
            # The whole frame, brackets included. The semihosted machine writes
            # the same brackets into `trace.txt`, so both paths hand
            # `parse_port_traces` identical text and it can *require* the
            # framing rather than tolerate it.
            return found.group(0), int(found.group(3))
        now = time.monotonic()
        if progress and now - last_report > 5.0:
            print(f"    {len(buffer):,} bytes so far…")
            last_report = now
        if now - quiet_since > 20.0 and not buffer:
            raise PicoError(
                "nothing arrived on the serial port in 20s. Check: the Pico's GP0 "
                "is wired to the adapter's RX (on an Arduino UNO that is the pin "
                "marked TX/D1, with RESET jumpered to GND), the grounds are "
                "joined, and the board is powered."
            )
    raise PicoError(
        f"no complete BEGIN..END pass in {timeout_s:.0f}s ({len(buffer):,} bytes "
        "seen). If bytes are arriving but never frame, the baud rate is wrong."
    )


def stream_pass(fd: int, timeout_s: float, batch: int = 0):
    """`read_pass`, yielding each trace line the instant it lands.

    The measurement path wants one complete frame and nothing before it, which
    is what `read_pass` gives it and why that function is left exactly as it is.
    A demo wants the opposite: the whole value of a live view is that the
    strokes appear while the wire is still carrying them, and a caller that
    waited for `END` would be animating a recording of something that finished
    a second ago.

    So this is the same framing rule with a different sink. It yields complete
    lines only -- a partial line is held until its newline arrives, because half
    of `p 12345 67890` parses as a coordinate that was never sent -- and it
    still refuses to open mid-drawing: nothing is emitted until a `BEGIN` for
    this batch has been seen, and the generator ends at that batch's `END`.

    Yields `(tag, args, line)` per line, then finally the `END` line. The raw
    line is carried alongside its split form because a demo shows the actual
    UART record beside the picture, and reconstructing that text from parsed
    fields would be showing a rendering of the trace rather than the trace.
    """
    begin = re.compile(rf"^BEGIN {batch}$")
    buffer = bytearray()
    started = False
    quiet_since = time.monotonic()
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        ready, _, _ = select.select([fd], [], [], 0.2)
        if ready:
            chunk = os.read(fd, 1 << 16)
            if chunk:
                buffer += chunk
                quiet_since = time.monotonic()
        while b"\n" in buffer:
            raw, _, rest = buffer.partition(b"\n")
            buffer = bytearray(rest)
            line = raw.decode("ascii", "replace").strip()
            if not started:
                # Everything before this batch's BEGIN is the tail of the
                # previous batch still draining out of the adapter, or the
                # middle of a drawing that started before the reader attached.
                started = bool(begin.match(line))
                continue
            tag, _, remainder = line.partition(" ")
            yield tag, remainder.split(), line
            if tag == "END":
                return
        if time.monotonic() - quiet_since > 20.0 and not started:
            raise PicoError(
                "nothing framed on the serial port in 20s. Check: the Pico's GP0 "
                "is wired to the adapter's RX (on an Arduino UNO that is the pin "
                "marked TX/D1, with RESET jumpered to GND), the grounds are "
                "joined, and the board is powered."
            )
    raise PicoError(
        f"no complete BEGIN..END pass in {timeout_s:.0f}s. If bytes are arriving "
        "but never frame, the baud rate is wrong."
    )


# ---------------------------------------------------------------------------
# bring-up diagnostics: the host side, and the wire, separately


#: What `--loopback` sends. Every bit position toggles and no byte repeats, so a
#: return that is merely *plausible* — a stuck line, an echo of one character, a
#: framing error that drops the top bit — cannot be mistaken for a pass.
PROBE = bytes(range(0x20, 0x60))


def loopback(port: Path | None, seconds: float = 3.0) -> int:
    """Send bytes out of the adapter and require them back, with no Pico at all.

    This is the one test that separates "the host, the driver, the adapter and
    the baud rate are fine" from "the Pico is not talking", and those two fail
    identically at `--run`. On an UNO held in reset it needs one extra jumper —
    `D0` to `D1` — which ties the bridge chip's TX to its own RX through the two
    1 kΩ series resistors the board already has.

    A pass here means every remaining failure is on the Pico side of `GP0`.
    """
    device = choose_port(port)
    fd = open_serial(device)
    print(f"  {device} at {BAUD} baud — jumper D0 to D1 with RESET tied to GND")
    try:
        os.write(fd, PROBE)
        seen = bytearray()
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline and len(seen) < len(PROBE):
            ready, _, _ = select.select([fd], [], [], 0.2)
            if ready:
                seen += os.read(fd, 1 << 12)
    finally:
        os.close(fd)

    if not seen:
        print("\n  nothing came back. Either the D0-D1 jumper is missing, or this "
              "device is not the adapter.", file=sys.stderr)
        return 1
    if bytes(seen) != PROBE:
        # Named separately because the two failures have different repairs: a
        # short read is a jumper that is not seated, and a corrupted one is a
        # baud or framing mismatch that would also corrupt the Pico's trace.
        kind = ("short" if bytes(seen) == PROBE[:len(seen)]
                else "corrupted")
        print(f"\n  {kind} return: {len(seen)} of {len(PROBE)} bytes, "
              f"{seen[:16].hex(' ')}…", file=sys.stderr)
        return 1
    print(f"  {len(seen)} of {len(PROBE)} bytes returned byte-identical — the host "
          "side works")
    return 0


def listen(port: Path | None, seconds: float = 15.0) -> int:
    """Whatever is on the wire, printed, whether or not it frames.

    `read_pass` discards its buffer when it gives up, which is right for a
    measurement and useless for bring-up: the buffer is the diagnosis. Bytes
    that never frame are a baud mismatch; no bytes at all are a wiring or power
    fault; framed bytes that `--run` still rejects are a firmware finding.
    """
    device = choose_port(port)
    fd = open_serial(device)
    print(f"  listening on {device} at {BAUD} baud for {seconds:.0f}s — Ctrl-C to stop")
    seen = bytearray()
    deadline = time.monotonic() + seconds
    try:
        while time.monotonic() < deadline:
            ready, _, _ = select.select([fd], [], [], 0.5)
            if ready:
                chunk = os.read(fd, 1 << 16)
                if chunk:
                    seen += chunk
                    sys.stdout.write(chunk.decode("ascii", "replace"))
                    sys.stdout.flush()
    except KeyboardInterrupt:
        pass
    finally:
        os.close(fd)

    text = seen.decode("ascii", "replace")
    frames = PASS.findall(text)
    printable = sum(1 for b in seen if 0x20 <= b < 0x7F or b in (0x0A, 0x0D))
    print(f"\n\n  {len(seen):,} bytes, {printable:,} printable, {len(frames)} complete "
          "BEGIN..END pass(es)")
    if not seen:
        print("  nothing arrived. Check GP0 to the adapter's RX, the shared ground, "
              "and that the board is powered.", file=sys.stderr)
        return 1
    if printable < len(seen) * 0.9:
        print(f"  mostly non-printable ({len(seen) - printable:,} bytes) — the baud "
              f"rate is wrong. First bytes: {seen[:16].hex(' ')}", file=sys.stderr)
        return 1
    return 0 if frames else 1


# ---------------------------------------------------------------------------
# bring-up bisection: locating a hang on a part that cannot report one


#: What each `DM_BRINGUP_STAGE` in `port/pico/machine.c` proves, in the order the
#: firmware reaches them. The wording is the *repair* domain, not the source
#: line: knowing it dies in the clock switch is what narrows the next edit.
BRINGUP_STAGES = {
    1: "RAM entry trampoline reached reset_handler; VTOR/MSP/.data/.bss startup completed",
    2: "crystal oscillator started",
    3: "clk_sys and clk_ref moved onto the crystal",
    4: "microsecond timer running",
    5: "SysTick running",
    6: "clk_peri enabled and UART0/IO_BANK0/PADS_BANK0 out of reset",
    7: "UART0 programmed and GP0 muxed to it",
    8: "corpus header found at _corpus_base",
    9: "one byte left the transmitter",
    10: "64 bytes sustained through the 32-deep FIFO — it drains",
    11: "64 bytes sustained with the FIFO disabled",
    12: "dm_io_begin returned and the shared harness entered its pass loop",
    13: 'puts_("BEGIN ") completed — six buffered put()s',
    14: "the harness's first flush reached dm_io_write",
    15: "trace_corpus header formatted; about to walk the corpus",
    16: "a whole pass: corpus walked, programs interpreted, trace flushed",
    # Not a rung: it tests the instrument. A deliberate UDF #0 must produce a
    # `FAULT pc=` line and then return to BOOTSEL. Reaching the checkpoint would
    # mean the undefined instruction did not fault.
    17: "INSTRUMENT TEST: deliberate HardFault, expect FAULT and BOOTSEL",
    18: "stage-13 microscope read out_len",
    19: "stage-13 microscope verified out_len == 0",
    20: "stage-13 microscope wrote and read out[0] canary",
    21: 'stage-13 microscope completed all six puts_("BEGIN ") writes',
    22: "stage-13 microscope reached dm_config_batch reboot checkpoint",
}

# The fault instrument is a precondition for interpreting later silence. Run it
# before the numeric ladder even though its source checkpoint is reached there.
BRINGUP_ORDER = [17] + [stage for stage in sorted(BRINGUP_STAGES) if stage != 17]


def build_stage(stage: int, out_max: int | None = None) -> None:
    """`-B` because the stage lives in a `-D`, which no timestamp reflects."""
    extra = [f"PICO_OUT_MAX={out_max}"] if out_max else []
    subprocess.run(["make", "-s", "-B", "-C", str(PORT), "pico",
                    f"PICO_EXTRA=-DDM_BRINGUP_STAGE={stage}", *extra], check=True)


def returned_to_bootsel(timeout_s: float = 25.0, fd: int | None = None) -> tuple[bool, bytes]:
    """Did the firmware reach its checkpoint, and what reached the wire?

    Two channels, watched together, because each answers half of the question and
    neither answers it alone. The reboot says the firmware got there; the serial
    bytes say the pin did. A stage that reboots with nothing on the wire puts the
    fault outside the chip — which no amount of re-seating connectors could
    establish, and no reboot on its own can either.

    Two waits on the volume, not one. It is still mounted at the moment the last
    UF2 block lands, so a single presence check reads the *old* mount and calls
    every stage a pass. Absence first, then presence.
    """
    seen = bytearray()

    def sip() -> None:
        if fd is None:
            return
        ready, _, _ = select.select([fd], [], [], 0.0)
        if ready:
            seen.extend(os.read(fd, 1 << 16))

    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline and BOOTSEL_VOLUME.exists():
        sip()
        time.sleep(0.2)
    deadline = time.monotonic() + timeout_s
    found = False
    while time.monotonic() < deadline:
        sip()
        if BOOTSEL_VOLUME.exists():
            found = True
            break
        time.sleep(0.25)
    # The board keeps transmitting right up to its reboot, and macOS takes a
    # moment to mount, so the last bytes land after the volume appears.
    grace = time.monotonic() + 1.0
    while time.monotonic() < grace:
        sip()
        time.sleep(0.1)
    return found, bytes(seen)


def probe(verbose: bool, out_max: int | None = None) -> int:
    """The real image on a one-`HALT` corpus, watched on both channels.

    The ladder can only checkpoint code `port/pico/machine.c` owns, and the
    harness is shared with every other machine — so the last stretch, from the
    end of `dm_io_begin` to `dm_io_end`, has no rung. This is how it is observed
    instead: `passes=1` means a clean finish reboots the board, so "did it come
    back" and "what reached the wire" bracket the whole harness.

    Quiet and verbose are separate runs because they exercise different code. The
    quiet path formats fourteen bytes and runs the interpreter against the null
    sink; the verbose path adds the reporting sink and the integer formatter. A
    quiet pass with a verbose hang localises the fault to the difference.
    """
    from dm.isa.spec import Op
    from scripts.uf2 import pack

    build(out_max=out_max)
    fd = None
    try:
        device = choose_port(None)
        fd = open_serial(device)
        print(f"  watching {device} at {BAUD} baud")
    except (PicoError, OSError) as failure:
        print(f"  no serial adapter to watch ({failure}); reboot channel only")

    buffering = f"out[{out_max}]" if out_max else "out[32768]"
    print(f"  measured image, one HALT, {'verbose' if verbose else 'quiet'}, {buffering}")
    try:
        load(pack([bytes([int(Op.HALT)])], passes=1, quiet=not verbose))
        reached, seen = returned_to_bootsel(fd=fd)
    finally:
        if fd is not None:
            os.close(fd)

    print(f"    wire: {len(seen)} bytes  "
          f"{seen.decode('ascii', 'replace')[:200]!r}" if seen else "    wire: nothing")
    print(f"    reboot: {'the harness ran to dm_io_end' if reached else 'NO — it hung'}")
    return 0 if reached else 1


def bss_check(port: Path | None, out_max: int | None = None) -> int:
    """Prefill and scan every BSS word, then require its exact one-HALT record."""
    from dm.isa.spec import Op
    from scripts.uf2 import pack

    build(out_max=out_max, pico_extra="-DDM_BSS_PREFILL=1")
    device = choose_port(port)
    fd = open_serial(device)
    print(f"  watching {device} at {BAUD} baud")
    try:
        load(pack([bytes([int(Op.HALT)])], passes=1, quiet=True))
        reached, seen = returned_to_bootsel(fd=fd)
    finally:
        os.close(fd)

    print(f"    wire: {len(seen)} bytes  "
          f"{seen.decode('ascii', 'replace')!r}" if seen else "    wire: nothing")
    print(f"    reboot: {'reported back' if reached else 'NO — it hung'}")
    if bss_check_success(reached, seen):
        print("    exact BSS_CLEAR verdict and one-HALT return verified")
        return 0
    print("\n  .bss diagnostic failed: expected exactly "
          f"{BSS_SUCCESS!r} plus BOOTSEL reboot.", file=sys.stderr)
    return 1


def bringup(only: int | None, verbose: bool = False,
            out_max: int | None = None, port: Path | None = None) -> int:
    """Walk the checkpoints until one fails to report, and name it.

    A passing stage reboots the board to BOOTSEL by itself, so the whole ladder
    runs on **one** button press. The first stage that does not come back is the
    one that hangs — and it is also the one that leaves the board needing a
    manual press, which is why the sweep stops there rather than continuing.
    """
    from dm.isa.spec import Op
    from scripts import uf2

    stages = [only] if only else BRINGUP_ORDER
    program = [bytes([int(Op.HALT)])]

    # Best effort: the reboot channel works with no adapter attached at all, and
    # losing the serial half must not stop the bisection that does not need it.
    fd = None
    try:
        device = choose_port(port)
        fd = open_serial(device)
        print(f"  also watching {device} at {BAUD} baud\n")
    except (PicoError, OSError) as failure:
        if 17 in stages:
            raise PicoError(
                "stage 17 requires a serial adapter: its exact !/FAULT record "
                "is part of the pass condition"
            ) from failure
        print(f"  no serial adapter to watch ({failure}); reboot channel only\n")

    try:
        for stage in stages:
            if stage not in BRINGUP_STAGES:
                raise PicoError(f"stage {stage} is not one of {sorted(BRINGUP_STAGES)}")
            print(f"  stage {stage} — {BRINGUP_STAGES[stage]}")
            build_stage(stage, out_max)
            expected_pc = None
            if stage == 17:
                try:
                    expected_pc = uf2.read_elf(IMAGE).symbol("dm_stage17_udf") & ~1
                except uf2.Uf2Error as failure:
                    raise PicoError(
                        "stage 17 image has no named deliberate UDF symbol"
                    ) from failure
            # Quiet is the default because the early rungs do not care, but the
            # stages inside the harness must run the *failing* path: quiet mode
            # skips the reporting sink and the integer formatter entirely, so a
            # quiet pass would clear code the verbose run never executes.
            load(uf2.pack(program, passes=1, quiet=not verbose))
            reached, seen = returned_to_bootsel(fd=fd)
            text = seen.decode("ascii", "replace") if seen else ""
            if seen:
                print(f"    wire: {len(seen)} bytes  {text[:64]!r}")
            elif fd is not None:
                print("    wire: nothing")
            if stage == 17:
                fault = FAULT.fullmatch(text) if seen else None
                if bringup_stage_success(stage, reached, seen, expected_pc):
                    print(f"    expected FAULT pc=0x{fault.group(1)} "
                          f"lr=0x{fault.group(2)} and reboot reported back\n")
                    continue
                if fault:
                    actual_pc = int(fault.group(1), 16)
                    if actual_pc != expected_pc:
                        print(f"\n  stage 17 FAULT stacked PC is 0x{actual_pc:08x}; "
                              f"expected deliberate UDF at 0x{expected_pc:08x}.",
                              file=sys.stderr)
                    else:
                        print("\n  stage 17 emitted the exact FAULT record but did "
                              "not return to BOOTSEL.", file=sys.stderr)
                elif reached:
                    print("\n  stage 17 returned to BOOTSEL without the exact "
                          "!/FAULT record.", file=sys.stderr)
                else:
                    print("\n  stage 17 produced neither the exact !/FAULT record "
                          "nor a BOOTSEL reboot.", file=sys.stderr)
                return 1
            if reached:
                print("    reported back\n")
                continue
            print(f"\n  stage {stage} never reported back. The firmware dies at or "
                  f"before:\n    {BRINGUP_STAGES[stage]}\n"
                  "  Hold BOOTSEL and replug to recover the board.", file=sys.stderr)
            return 1
    finally:
        if fd is not None:
            os.close(fd)
    print("  every checkpoint reported. Rebuild the measured image with:\n"
          "    make -B -C port pico")
    return 0


# ---------------------------------------------------------------------------


def build(quiet: bool = True, out_max: int | None = None,
          pico_extra: str | None = None) -> None:
    """Always `-B`, and that is a scar rather than a preference.

    `build_stage` compiles the same sources with `-DDM_BRINGUP_STAGE=N`, and a
    timestamp cannot see a `-D`. So an incremental `make` after any bring-up run
    says "nothing to be done" and hands the *checkpointed* image to `--run`,
    `conformance.py --pico` and `cycles.py` — an image that reboots to BOOTSEL
    partway through `dm_io_begin` and never executes a program. That is a
    measurement taken with the wrong instrument reporting success, which is the
    exact failure this project keeps paying to remove. Two seconds of rebuild
    closes it.
    """
    flags = ["-s", "-B"] if quiet else ["-B"]
    extra = [f"PICO_OUT_MAX={out_max}"] if out_max else []
    if pico_extra:
        extra.append(f"PICO_EXTRA={pico_extra}")
    subprocess.run(["make", *flags, "-C", str(PORT), "pico", *extra], check=True)


def budget(programs: list[bytes], reps: int, quiet: bool) -> float:
    """A timeout proportional to what the wire has to carry.

    At 115200 baud a QuickDraw drawing's trace is a few kilobytes, so a
    thousand-program corpus is minutes and a fixed 60s would fail on the size of
    the corpus rather than on anything being wrong. Cycle and quiet modes emit a
    line per program instead of a point per point, hence the two rates.
    """
    per_program = 40.0 if (reps or quiet) else 4000.0
    return 60.0 + len(programs) * per_program / 11_500.0


def run_batches(groups: list[list[bytes]], fuel: int = 100_000, reps: int = 0,
                quiet: bool = False, port: Path | None = None,
                no_build: bool = False) -> list[str]:
    """One pass per batch, over a single open serial port.

    Two things make a multi-batch sweep survivable, and both are here rather
    than in the firmware:

    **The port is opened before the first batch is written.** The board starts
    executing the instant its last UF2 block lands, so a host that opened the
    port afterwards would race the output of a short corpus.

    **The board returns to BOOTSEL by itself.** `passes=1` makes the firmware
    call the bootrom's `reset_usb_boot` when the pass ends, so only the *first*
    batch needs someone to hold the button.
    """
    from scripts.uf2 import pack

    if not no_build:
        build()
    device = choose_port(port)
    fd = open_serial(device)
    print(f"  reading {device} at {BAUD} baud")
    out: list[str] = []
    try:
        for index, programs in enumerate(groups):
            if len(groups) > 1:
                print(f"  batch {index + 1}/{len(groups)}: {len(programs)} programs")
            load(pack(programs, fuel=fuel, reps=reps, quiet=quiet, passes=1,
                      batch=index))
            text, count = read_pass(fd, budget(programs, reps, quiet), batch=index)
            if count != len(programs):
                raise PicoError(
                    f"the device reported {count} programs and batch {index} has "
                    f"{len(programs)}"
                )
            out.append(text)
    finally:
        os.close(fd)
    return out


def run(programs: list[bytes], fuel: int = 100_000, reps: int = 0,
        quiet: bool = False, port: Path | None = None,
        no_build: bool = False) -> str:
    """Build, pack, load, and return one complete pass of output.

    Splits into batches when the corpus outgrows the region `port/pico/link.ld`
    reserves, and rejoins the passes into one framed block so callers see a
    single run. Rejoining is legitimate because a batch boundary is not a state
    boundary: `dm_vm_run` is re-entrant and keeps nothing between programs, which
    `port/include/dm_vm.h` states and `scripts/footprint.py` checks by asserting
    the interpreter has no static RAM at all.
    """
    from scripts.uf2 import batches, capacity

    if reps:
        raise PicoError("cycle mode is per batch; call run_batches and join the "
                        "records, not the text -- `elapsed_us` does not concatenate")

    groups = batches(programs, capacity())
    passes = run_batches(groups, fuel=fuel, reps=reps, quiet=quiet, port=port,
                         no_build=no_build)
    if len(passes) == 1:
        return passes[0]

    body: list[str] = []
    for index, text in enumerate(passes):
        lines = text.splitlines()[1:-1]
        # Every batch states the fixed-point scale it was built with, and the
        # joined pass should state it once. Dropped rather than deduplicated:
        # `scripts/conformance.py` refuses a port whose scale differs from the
        # reference's, and it checks the line it is given -- so a *differing*
        # second header must not be silently discarded here. It cannot be:
        # every batch came from one image in one session.
        if index and lines[:1] and lines[0].startswith("frac_bits "):
            lines = lines[2:]
        body += lines
    return "BEGIN 0\n" + "\n".join(body) + f"\nEND {len(programs)}\n"


def smoke(port: Path | None, no_build: bool) -> int:
    """One `HALT`, executed on silicon, traced back. The whole path in one line.

    Deliberately the smallest possible corpus: if this works, the UF2 packing,
    the bootrom's SRAM load, the linker script, the clock bring-up, the UART,
    the adapter, the framing and the harness all work — and anything that fails
    afterwards is a finding rather than bring-up.
    """
    from dm.isa.spec import Op

    text = run([bytes([int(Op.HALT)])], port=port, no_build=no_build)
    print(text, end="")
    expected = ["BEGIN 0", "frac_bits 12", "curve_steps 16", "#0", "= 1 1 0", "END 1"]
    if text.splitlines() != expected:
        print(f"\n  UNEXPECTED — a single HALT should trace as {expected}")
        return 1
    print("\n  one HALT executed on an RP2040 and traced back exactly")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ports", action="store_true", help="list candidate serial devices")
    ap.add_argument("--loopback", action="store_true",
                    help="prove the adapter with D0 jumpered to D1 and no Pico")
    ap.add_argument("--listen", action="store_true",
                    help="print whatever arrives, framed or not")
    ap.add_argument("--seconds", type=float, default=15.0, help="how long to --listen")
    ap.add_argument("--bringup", nargs="?", type=int, const=0, default=None,
                    metavar="STAGE",
                    help="walk the firmware checkpoints (or run just STAGE) and "
                         "name the first that never reports back")
    ap.add_argument("--probe", action="store_true",
                    help="run the measured image on one HALT, watching wire and reboot")
    ap.add_argument("--bss-check", action="store_true",
                    help="prefill/scan all .bss words and require its exact one-HALT reboot")
    ap.add_argument("--verbose", action="store_true", help="--probe with the reporting sink")
    ap.add_argument("--out-max", type=int, default=None, metavar="N",
                    help="--probe/--bss-check with an N-byte output buffer; 1 flushes every byte, "
                         "so the last byte received is the instruction before a hang")
    ap.add_argument("--run", action="store_true", help="execute a one-HALT corpus")
    ap.add_argument("--port", type=Path, help="the serial device, if more than one")
    ap.add_argument("--no-build", action="store_true")
    args = ap.parse_args()

    if args.ports:
        found = serial_ports()
        usb = set(usb_ports())
        for path in found:
            print(f"  {path}{'   <- USB adapter' if path in usb else ''}")
        if not found:
            print("  no candidate serial devices under /dev/cu.*")
        return 0 if found else 1
    if not (args.loopback or args.listen or args.run or args.probe or args.bss_check
            or args.bringup is not None):
        ap.print_help()
        return 2
    try:
        if args.loopback:
            return loopback(args.port)
        if args.listen:
            return listen(args.port, args.seconds)
        if args.bringup is not None:
            return bringup(args.bringup or None, args.verbose, args.out_max, args.port)
        if args.bss_check:
            return bss_check(args.port, args.out_max)
        if args.probe:
            return probe(args.verbose, args.out_max)
        return smoke(args.port, args.no_build)
    except PicoError as failure:
        print(f"\n{failure}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
