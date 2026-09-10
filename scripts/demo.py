#!/usr/bin/env python3
"""Type a word; a microcontroller draws it. And the evidence that it did.

This is the demo path. A class-conditional 825k-parameter transformer emits
**bytecode**, the bytecode is packed into a UF2 and dropped onto a Raspberry Pi
Pico in BOOTSEL, the RP2040 executes it out of SRAM, and the drawing on the
screen is reassembled from the fixed-point trace that comes back over a UART.
The host reference VM checks the result and draws nothing.

    # the live page: type a word, watch the board draw it
    python3 scripts/demo.py live

    # the same thing with no hardware, for building and rehearsing
    python3 scripts/demo.py --native live

    # capture a set of records on the bench, in one hands-off pass
    python3 scripts/demo.py --novelty capture --words cat bus flower sailboat bicycle --repeat 4

    # continue held-out human drawings instead of starting from nothing
    python3 scripts/demo.py capture --prefix-panel --repeat 2

    # show the refusal path firing: one coordinate is perturbed, the page refuses
    python3 scripts/demo.py --prove-refusal live

    # measure novelty on every captured record (no hardware, slow the first time)
    python3 scripts/demo.py novelty

    # build the offline gallery, which replays those records forever
    python3 scripts/demo.py gallery --out artifacts/demo/gallery.html

    # draw the four still figures straight from the records, into docs/media
    python3 scripts/demo.py figures

**The first load of a session needs BOOTSEL held once.** After that the firmware
returns the board to BOOTSEL by itself at the end of every pass, so the loop
runs hands-off -- and during an automatic remount the right thing to do is
*wait*, never unplug (`docs/claim4-bringup.md` §8).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dm.demo import capture as capture_mod  # noqa: E402
from dm.demo import page as page_mod  # noqa: E402
from dm.demo.device import (DeviceSession, NativeRehearsal,  # noqa: E402
                             tamper_first_point)
from dm.demo.novelty import NoveltyIndex  # noqa: E402
from dm.demo.record import DemoRecord, load_all  # noqa: E402
from dm.demo.sample import (DEFAULT_TEMPERATURE, DEFAULT_TOP_K,  # noqa: E402
                            load_checkpoint)

#: Named here rather than imported so `--help` does not pay for PIL.
FIGURES = ("gallery", "novelty", "prefix", "program")

DEFAULT_CHECKPOINT = ROOT / "runs" / "quickdraw_cond24000eps4_byte_square_s0.pt"
ARTIFACTS = ROOT / "artifacts" / "demo"
RECORDS = ARTIFACTS / "records"
CACHE = ARTIFACTS / "cache"


def open_device(args):
    """The board, or the hardware-free rehearsal, named so the record can say which."""
    tamper = tamper_first_point if getattr(args, "prove_refusal", False) else None
    if args.native:
        return NativeRehearsal(tamper=tamper)
    return DeviceSession(port=args.port, build=not args.no_build, tamper=tamper)


def novelty_index(checkpoint, build: bool) -> NoveltyIndex | None:
    """The cached index, built on demand, or None when it is not wanted.

    Built once and cached under a key that changes whenever anything the numbers
    depend on changes, so a stale bank cannot be read against a new corpus.
    """
    index = NoveltyIndex(checkpoint.classes,
                         rdp_eps=(checkpoint.record["config"].get("extra") or {}).get("rdp_eps", 4.0))
    path = CACHE / f"{index.key}.npz"
    if path.exists():
        return NoveltyIndex.load(path)
    if not build:
        return None
    print(f"  building the novelty index (once) -> {path.name}")
    index.build(progress=lambda m: print(f"    {m}", flush=True))
    index.save(CACHE)
    return index


# --------------------------------------------------------------------------


def cmd_live(args) -> int:
    from dm.demo.server import DemoServer

    checkpoint = load_checkpoint(args.checkpoint)
    index = novelty_index(checkpoint, build=args.novelty)
    with open_device(args) as device:
        server = DemoServer(("127.0.0.1", args.port_number), checkpoint, device,
                            RECORDS, novelty=index, top_k=args.top_k,
                            temperature=args.temperature)
        url = f"http://127.0.0.1:{args.port_number}/"
        print(f"  {checkpoint.record['name']} — {checkpoint.parameters:,} parameters")
        print(f"  categories: {', '.join(checkpoint.classes)}")
        print(f"  source: {device.source}"
              + (f" on {device.device_path}" if device.device_path else ""))
        print(f"  novelty panel: {'on' if index else 'off (pass --novelty to build it)'}")
        print(f"\n  open {url}\n  Ctrl-C to stop")
        if device.source == "rp2040":
            print("  hold BOOTSEL and plug the Pico in for the first drawing; "
                  "after that it returns by itself — wait, do not unplug")
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("\n  stopped")
    return 0


def cmd_capture(args) -> int:
    checkpoint = load_checkpoint(args.checkpoint)
    index = novelty_index(checkpoint, build=args.novelty)
    words = args.words or list(checkpoint.classes)

    prefixes: list[tuple[bytes, dict]] = []
    if args.prefix_panel:
        prefixes = held_out_prefixes(checkpoint, args.prefix_fraction, args.repeat)

    written, refused = [], []
    with open_device(args) as device:
        if device.source == "rp2040":
            print("  hold BOOTSEL and plug the Pico in; after the first load the "
                  "board returns by itself — wait during a remount, do not unplug")
        plan = ([(w, s, None, None) for w in words for s in range(args.repeat)]
                if not args.prefix_panel else
                [(checkpoint.classes[m["class_index"]], m["seed"], p, m)
                 for p, m in prefixes])
        for word, seed, prompt, meta in plan:
            label = f"{word} seed {seed}" + (" (continuation)" if prompt else "")
            print(f"  {label} … ", end="", flush=True)
            started = time.monotonic()
            outcome = None
            for event in capture_mod.draw(checkpoint, device, word, seed,
                                          top_k=args.top_k,
                                          temperature=args.temperature,
                                          novelty=index, prompt=prompt,
                                          prefix_meta=meta, note=args.note):
                if event["t"] in {"record", "error"}:
                    outcome = event
            if outcome is None or outcome["t"] == "error":
                message = outcome["message"] if outcome else "no outcome"
                print(f"REFUSED — {message}")
                refused.append((label, message))
                continue
            path = capture_mod.save(outcome["record"], RECORDS)
            print(f"ok  {time.monotonic() - started:.1f}s  -> {path.name}")
            written.append(path)

    print(f"\n  {len(written)} records written to {RECORDS}")
    if refused:
        print(f"  {len(refused)} refused:")
        for label, message in refused:
            print(f"    {label}: {message}")
    return 0 if written else 1


def held_out_prefixes(checkpoint, fraction: float, per_class: int):
    """Val drawings the model never saw, cut to a prefix for it to finish.

    The prefix has to be a whole number of *instructions*, not bytes: cutting
    mid-instruction hands the model an operand as an opcode, and the drawing it
    finishes would be a continuation of a program the corpus does not contain.
    """
    from dm.data import quickdraw
    from dm.isa.asm import parse
    from dm.isa.spec import spec_for

    extra = checkpoint.record["config"].get("extra") or {}
    programs, labels = quickdraw.load_labelled(checkpoint.classes, "valid",
                                               **{k: v for k, v in extra.items()})
    out: list[tuple[bytes, dict]] = []
    taken = {i: 0 for i in range(checkpoint.n_classes)}
    for position, (program, label) in enumerate(zip(programs, labels)):
        if taken[label] >= per_class:
            continue
        try:
            instructions = parse(program)
        except Exception:                                 # noqa: BLE001
            continue
        if len(instructions) < 8:
            continue
        cut = max(2, int(len(instructions) * fraction))
        offset = 0
        for instruction in instructions[:cut]:
            offset += spec_for(_opcode(program, offset)).size
        prefix = program[:offset]
        out.append((prefix, {
            "source": "quickdraw valid split",
            "position": position,
            "class_index": int(label),
            "seed": taken[label],
            "bytes": len(prefix),
            "instructions": cut,
            "of_instructions": len(instructions),
            "fraction": fraction,
            "truth_hex": program.hex(),
        }))
        taken[label] += 1
    return out


def _opcode(program: bytes, offset: int) -> int:
    return program[offset]


def cmd_novelty(args) -> int:
    checkpoint = load_checkpoint(args.checkpoint)
    index = novelty_index(checkpoint, build=True)
    records = sorted(RECORDS.glob("*.json"))
    if not records:
        print(f"  no records under {RECORDS}")
        return 1
    verbatim = 0
    for path in records:
        record = DemoRecord.load(path)
        if record.novelty and not args.force:
            continue
        klass = record.routing.get("index")
        if klass is None:
            continue
        record.novelty = index.nearest(record.bytecode, klass)
        if args.runs:
            record.novelty["longest_shared_run"] = longest_shared_run(
                record.bytecode, checkpoint, klass)
        verbatim += bool(record.novelty.get("verbatim"))
        record.save(RECORDS)
        print(f"  {path.name}: {record.novelty['distance']:.2f} px, "
              f"{record.novelty['floor_percentile']:.0f}th percentile of the "
              f"held-out floor, verbatim={record.novelty['verbatim']}")
    print(f"\n  verbatim training programs among {len(records)} records: {verbatim}")
    return 0


def longest_shared_run(program: bytes, checkpoint, class_index: int) -> int:
    """The longest run of bytes this program shares with the training corpus.

    Novel geometry and novel *program text* are different claims, and the
    Chamfer distance cannot separate "drew something new" from "reassembled
    memorised fragments". This one can: it is the largest `k` for which some `k`
    consecutive bytes of the sample occur anywhere in the class's training
    programs, found by binary search over a single concatenated blob so the
    search is one `bytes.find` per probe rather than a Python scan.

    The separator byte is `0xff`, which is not a valid opcode, so a run can
    never straddle two training programs and be counted as one.
    """
    from dm.data import quickdraw

    extra = checkpoint.record["config"].get("extra") or {}
    programs, labels = quickdraw.load_labelled(checkpoint.classes, "train", **extra)
    blob = b"\xff".join(p for p, label in zip(programs, labels) if label == class_index)
    low, high = 0, len(program)
    while low < high:
        mid = (low + high + 1) // 2
        if any(blob.find(program[i:i + mid]) >= 0 for i in range(len(program) - mid + 1)):
            low = mid
        else:
            high = mid - 1
    return low


def cmd_gallery(args) -> int:
    records = load_all(RECORDS)
    if not records:
        print(f"  no records under {RECORDS} — run `demo.py capture` first")
        return 1
    if args.silicon_only:
        records = [r for r in records if r.on_silicon]
    blobs = [capture_mod.view(r) for r in records]
    checkpoint = records[0].checkpoint
    html = page_mod.render("gallery", checkpoint=checkpoint, records=blobs,
                           title=args.title)
    out = page_mod.write(args.out, html)
    silicon = sum(r.on_silicon for r in records)
    print(f"  {out}  —  {len(records)} records, {silicon} on silicon, "
          f"{out.stat().st_size / 1024:.0f} KB, no network dependency")
    return 0


def cmd_figures(args) -> int:
    """The four paper figures, from the records, with no browser in the loop.

    A screenshot of the gallery would carry the browser's chrome, the host's
    theme and the window size of the day; these are drawn from
    `record.device.geometry` and are byte-identical on every rebuild.
    """
    from dm.demo import figures as figures_mod

    records = load_all(RECORDS)
    if not records:
        print(f"  no records under {RECORDS} — run `demo.py capture` first")
        return 1
    if not any(r.on_silicon for r in records):
        print("  no record on this bench earned the silicon badge; the figures "
              "would be captioned for hardware that was not involved")
        return 1
    written = figures_mod.build(records, args.out, only=args.only,
                                columns=args.columns)
    for path in written:
        print(f"  {path}  —  {path.stat().st_size / 1024:.0f} KB")
    return 0


def cmd_status(args) -> int:
    records = load_all(RECORDS) if RECORDS.exists() else []
    silicon = sum(r.on_silicon for r in records)
    novel = sum(bool(r.novelty) for r in records)
    verbatim = sum(bool((r.novelty or {}).get("verbatim")) for r in records)
    print(f"  records      {len(records)}")
    print(f"  on silicon   {silicon}")
    print(f"  novelty run  {novel}")
    print(f"  verbatim     {verbatim}  (training programs reproduced byte for byte)")
    by_class: dict[str, int] = {}
    for record in records:
        by_class[record.routing.get("class") or "refused"] = (
            by_class.get(record.routing.get("class") or "refused", 0) + 1)
    for name, count in sorted(by_class.items()):
        print(f"    {name:10s} {count}")
    if args.json:
        print(json.dumps({"records": len(records), "silicon": silicon,
                          "novelty": novel, "verbatim": verbatim,
                          "by_class": by_class}, indent=1))
    return 0


# --------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    ap.add_argument("--native", action="store_true",
                    help="use the native harness instead of a board; records are "
                         "stamped `native` and never earn the silicon badge")
    ap.add_argument("--port", type=Path, help="the serial device, if more than one")
    ap.add_argument("--no-build", action="store_true", help="skip `make -C port pico`")
    ap.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    ap.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    ap.add_argument("--novelty", action="store_true",
                    help="build/load the novelty index and attach it to each record")
    ap.add_argument("--prove-refusal", action="store_true",
                    help="move one wire coordinate by a whole canvas pixel, so the "
                         "refusal path can be demonstrated rather than asserted. "
                         "No record is written; that is the point")
    sub = ap.add_subparsers(dest="command", required=True)

    live = sub.add_parser("live", help="serve the interactive page")
    live.add_argument("--port-number", type=int, default=8765)
    live.set_defaults(func=cmd_live)

    cap = sub.add_parser("capture", help="capture records hands-off")
    cap.add_argument("--words", nargs="*", help="defaults to the checkpoint's classes")
    cap.add_argument("--repeat", type=int, default=3, help="samples per word")
    cap.add_argument("--note", default="")
    cap.add_argument("--prefix-panel", action="store_true",
                     help="continue held-out human drawings instead of sampling free")
    cap.add_argument("--prefix-fraction", type=float, default=0.4)
    cap.set_defaults(func=cmd_capture)

    nov = sub.add_parser("novelty", help="attach novelty measurements to records")
    nov.add_argument("--force", action="store_true", help="recompute existing ones")
    nov.add_argument("--runs", action="store_true",
                     help="also compute the longest shared byte run (slow)")
    nov.set_defaults(func=cmd_novelty)

    gal = sub.add_parser("gallery", help="build the offline gallery")
    gal.add_argument("--out", type=Path, default=ARTIFACTS / "gallery.html")
    gal.add_argument("--title", default="drawing-machine — captured silicon runs")
    gal.add_argument("--silicon-only", action="store_true",
                     help="drop rehearsal records from the published page")
    gal.set_defaults(func=cmd_gallery)

    fig = sub.add_parser("figures", help="draw the still figures from the records")
    fig.add_argument("--out", type=Path, default=ROOT / "docs" / "media")
    fig.add_argument("--only", nargs="*", choices=sorted(FIGURES),
                     help="defaults to all four")
    fig.add_argument("--columns", type=int, default=6,
                     help="samples per category on the gallery sheet")
    fig.set_defaults(func=cmd_figures)

    st = sub.add_parser("status", help="what has been captured so far")
    st.add_argument("--json", action="store_true")
    st.set_defaults(func=cmd_status)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
