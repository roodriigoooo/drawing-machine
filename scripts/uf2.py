#!/usr/bin/env python3
"""Packing an image and its corpus into one UF2, for a Pico with no debugger.

Claim 4's silicon half needs bytecode executing on a real Cortex-M0+. The
obvious way in is SWD, and SWD needs a second piece of hardware. This is the
way in that needs none.

## The idea

An RP2040 in BOOTSEL mode appears as a USB drive. Drop a UF2 on it and the
bootrom writes the blocks wherever they say, then runs the result. That is
normally used to program flash — but the loader accepts blocks addressed to
**SRAM**, which is what the SDK calls a `no_flash` binary, and an SRAM-resident
image is exactly what `port/pico/link.ld` produces and exactly what a
core-limited cycle count wants.

That solves getting a program *in*. It does not solve feeding it data while it
runs, because once the image starts, BOOTSEL is gone and there is no channel
inward at all. So the corpus does not stream: **it is packed into the same UF2**,
at the address `port/pico/link.ld` reserved for it, and the firmware wakes up
with its work already in memory.

    python3 scripts/uf2.py --out run.uf2 --programs 40
    python3 scripts/uf2.py --out cycles.uf2 --reps 5 --programs 40

## What the format is

A UF2 is a sequence of 512-byte blocks, each carrying 256 bytes of payload and
its own target address. The redundancy is the point: it survives a USB mass
storage stack that may deliver blocks out of order or twice, which is why it
exists rather than a plain binary.

The span written here is deliberately **contiguous** — image, then padding to
the 256-aligned corpus base, then corpus — because the bootrom is loading an
image rather than patching memory. `port/pico/link.ld` places the corpus region
between `.data` and `.bss` for that reason: `.bss` is 37 KB that is not in the
file, and a corpus after it would leave a 37 KB hole mid-download.
"""

from __future__ import annotations

import argparse
import struct
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

ROOT = Path(__file__).resolve().parent.parent
PORT = ROOT / "port"
IMAGE = PORT / "build" / "dm_pico.elf"

# --- UF2, from the format's own specification ------------------------------
UF2_MAGIC_START0 = 0x0A324655  # "UF2\n"
UF2_MAGIC_START1 = 0x9E5D5157
UF2_MAGIC_END = 0x0AB16F30
UF2_FLAG_FAMILY_ID = 0x00002000
#: The RP2040's family ID. A UF2 without it, or with someone else's, is refused
#: by the bootrom rather than half-written, which is the good failure.
RP2040_FAMILY_ID = 0xE48BFF56
UF2_PAYLOAD = 256
UF2_BLOCK = 512

# --- the corpus header the firmware reads at `_corpus_base` ----------------
#: 'D','C','M','0'. A magic number rather than a length, because an image
#: started without a corpus finds whatever the previous run left in SRAM — which
#: is a *valid* corpus and would be silently measured.
CORPUS_MAGIC = 0x304D4344
CORPUS_VERSION = 1
CORPUS_HEADER = struct.Struct("<9I")

PT_LOAD = 1
SHT_SYMTAB = 2


class Uf2Error(RuntimeError):
    pass


@dataclass(frozen=True)
class Elf:
    """Only the ELF entry, loadable segments, and symbols a loader needs.

    The Command Line Tools ship `llvm-size` and `llvm-objdump` and *not*
    `llvm-objcopy`, so a packer that shelled out to one would work on this
    machine and not on the next. Fifty lines of `struct` has no such dependency.
    """

    entry: int
    segments: tuple[tuple[int, bytes], ...]  # (virtual address, contents)
    symbols: dict[str, int]

    @property
    def base(self) -> int:
        return min(address for address, _ in self.segments)

    def symbol(self, name: str) -> int:
        if name not in self.symbols:
            raise Uf2Error(
                f"{name} is not in the image's symbol table. The linker script and "
                "this packer disagree about the memory map."
            )
        return self.symbols[name]


def read_elf(path: Path) -> Elf:
    blob = path.read_bytes()
    if blob[:4] != b"\x7fELF" or blob[4] != 1 or blob[5] != 1:
        raise Uf2Error(f"{path} is not a little-endian 32-bit ELF")

    e_entry, e_phoff, e_shoff = struct.unpack_from("<III", blob, 0x18)
    e_phentsize, e_phnum, e_shentsize, e_shnum, e_shstrndx = struct.unpack_from(
        "<HHHHH", blob, 0x2A
    )

    segments = []
    for i in range(e_phnum):
        p_type, p_offset, p_vaddr, _, p_filesz, _, _, _ = struct.unpack_from(
            "<8I", blob, e_phoff + i * e_phentsize
        )
        if p_type == PT_LOAD and p_filesz:
            segments.append((p_vaddr, blob[p_offset:p_offset + p_filesz]))

    sections = [
        struct.unpack_from("<10I", blob, e_shoff + i * e_shentsize) for i in range(e_shnum)
    ]

    def name_at(table_offset: int, index: int) -> str:
        end = blob.index(b"\0", table_offset + index)
        return blob[table_offset + index:end].decode()

    symbols: dict[str, int] = {}
    for _, sh_type, _, _, sh_offset, sh_size, sh_link, _, _, sh_entsize in sections:
        if sh_type != SHT_SYMTAB or not sh_entsize:
            continue
        strtab = sections[sh_link][4]
        for offset in range(sh_offset, sh_offset + sh_size, sh_entsize):
            st_name, st_value = struct.unpack_from("<II", blob, offset)
            if st_name:
                symbols[name_at(strtab, st_name)] = st_value

    if not segments:
        raise Uf2Error(f"{path} has no loadable segment")
    _ = e_shstrndx  # the section-name table is not needed; symbols carry names
    return Elf(entry=e_entry, segments=tuple(segments), symbols=symbols)


def framed(programs: list[bytes]) -> bytes:
    """The same `u32` length prefix `scripts/conformance.py` frames with, so one
    corpus format serves the semihosted machine and this one."""
    return b"".join(struct.pack("<I", len(p)) + p for p in programs)


def corpus_blob(programs: list[bytes], *, fuel: int, reps: int, quiet: bool,
                passes: int = 1, batch: int = 0) -> bytes:
    """The configuration a semihosted machine reads from `fuel.txt` and
    `cycles.txt`, in front of the programs it reads from `frames.bin`.

    All of it travels together because a part with no debugger cannot be told
    anything after it starts — so 'which corpus' and 'how to run it' have to be
    the same artifact, and a run is fully described by the file that was dropped
    on the drive.

    `passes` is the one field with no counterpart on the semihosted side, and it
    is what makes a multi-batch sweep bearable. After a finite number of passes
    the firmware calls the bootrom's `reset_usb_boot`, so the board returns to
    BOOTSEL by itself and the host can write the next batch without anyone
    touching the button. **Zero** means run forever and never reboot, which is
    what a manual look or an energy measurement wants — there the load has to
    stay on for as long as the meter is watching.
    """
    payload = framed(programs)
    return CORPUS_HEADER.pack(
        CORPUS_MAGIC, CORPUS_VERSION, fuel, reps, int(quiet), len(programs),
        len(payload), passes, batch,
    ) + payload


def capacity(image: Path = IMAGE) -> int:
    """How many bytes of framed programs fit in the reserved region.

    Read from the image rather than restated, because it is the linker script
    that decides. A corpus larger than this is split into batches by the caller;
    silently truncating one would be a conformance run that passed on the part
    it managed to reach.
    """
    elf = read_elf(image)
    return elf.symbol("_corpus_end") - elf.symbol("_corpus_base") - CORPUS_HEADER.size


def batches(programs: list[bytes], limit: int) -> list[list[bytes]]:
    """Split so each batch's framed size fits `limit`, preserving order."""
    out: list[list[bytes]] = [[]]
    used = 0
    for program in programs:
        cost = len(program) + 4
        if cost > limit:
            raise Uf2Error(
                f"a single {len(program)}-byte program does not fit the {limit:,}-byte "
                "corpus region; raise DM_CORPUS_RESERVE in port/pico/link.ld"
            )
        if used + cost > limit and out[-1]:
            out.append([])
            used = 0
        out[-1].append(program)
        used += cost
    return [batch for batch in out if batch]


def blocks(base: int, span: bytes, family: int = RP2040_FAMILY_ID) -> bytes:
    """`span` as UF2 blocks starting at `base`. `base` must be 256-aligned."""
    if base % UF2_PAYLOAD:
        raise Uf2Error(f"UF2 blocks are {UF2_PAYLOAD}-byte aligned; {base:#x} is not")
    total = (len(span) + UF2_PAYLOAD - 1) // UF2_PAYLOAD
    out = bytearray()
    for index in range(total):
        chunk = span[index * UF2_PAYLOAD:(index + 1) * UF2_PAYLOAD]
        out += struct.pack(
            "<8I", UF2_MAGIC_START0, UF2_MAGIC_START1, UF2_FLAG_FAMILY_ID,
            base + index * UF2_PAYLOAD, UF2_PAYLOAD, index, total, family,
        )
        out += chunk.ljust(476, b"\0")
        out += struct.pack("<I", UF2_MAGIC_END)
    return bytes(out)


def build_uf2(elf: Elf, corpus: bytes) -> bytes:
    """One contiguous span: the image, padding, then the corpus.

    Flattened by address rather than emitted per segment, so a gap between
    `.data` and the 256-aligned corpus base becomes zero padding inside a block
    instead of a hole between two runs of blocks.
    """
    base = elf.base
    if not (elf.entry & 1):
        raise Uf2Error(f"ELF entry {elf.entry:#x} is not a Thumb address")
    if (elf.entry & ~1) != base:
        raise Uf2Error(
            f"ELF entry {elf.entry:#x} is not at the lowest image address {base:#x}; "
            "a RAM UF2 must begin with its executable trampoline"
        )
    corpus_base = elf.symbol("_corpus_base")
    corpus_end = elf.symbol("_corpus_end")

    if corpus_base % UF2_PAYLOAD:
        raise Uf2Error(f"_corpus_base is {corpus_base:#x}, which is not 256-aligned")
    if len(corpus) > corpus_end - corpus_base:
        raise Uf2Error(
            f"the corpus is {len(corpus):,} bytes and the reserved region is "
            f"{corpus_end - corpus_base:,}. Pack fewer programs, or raise "
            "DM_CORPUS_RESERVE in port/pico/link.ld and rebuild."
        )

    span = bytearray(corpus_base - base)
    for address, contents in elf.segments:
        offset = address - base
        if offset < 0 or offset + len(contents) > len(span):
            raise Uf2Error(
                f"a loadable segment at {address:#x} does not fit below _corpus_base "
                f"({corpus_base:#x}); the image has outgrown its region"
            )
        span[offset:offset + len(contents)] = contents
    span += corpus
    return blocks(base, bytes(span))


def pack(programs: list[bytes], *, fuel: int = 100_000, reps: int = 0,
         quiet: bool = False, passes: int = 1, batch: int = 0,
         image: Path = IMAGE) -> bytes:
    """Keyword-only past `programs`, and that is a scar rather than a style.

    Every argument here but the first is a small integer or a flag, so inserting
    one shifted `image` into `passes` and produced a `struct.error` from a Path
    — three frames away from the call that was wrong. Positionally
    indistinguishable arguments are how that happens twice.
    """
    if not image.exists():
        raise Uf2Error(f"{image} does not exist -- run `make -C port pico` first")
    blob = corpus_blob(programs, fuel=fuel, reps=reps, quiet=quiet, passes=passes,
                       batch=batch)
    return build_uf2(read_elf(image), blob)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--programs", type=int, default=40, help="QuickDraw val programs")
    ap.add_argument("--fuel", type=int, default=100_000)
    ap.add_argument("--reps", type=int, default=0,
                    help="non-zero selects cycle mode with this many repetitions")
    ap.add_argument("--quiet", action="store_true", help="run with the null sink")
    # Zero is the right default for a UF2 dragged by hand: the board keeps
    # running the corpus, so a terminal opened at any point sees a whole pass.
    # The scripted paths ask for 1, which reboots to BOOTSEL when the pass ends.
    ap.add_argument("--passes", type=int, default=0,
                    help="0 runs forever; N reboots to BOOTSEL after N passes")
    ap.add_argument("--batch", type=int, default=0,
                    help="identity this corpus echoes in its BEGIN line")
    ap.add_argument("--image", type=Path, default=IMAGE)
    args = ap.parse_args()

    from dm.data import quickdraw

    programs = quickdraw.load(("cat", "dog", "bus", "car", "tree"), "valid",
                              limit=args.programs)
    data = pack(programs, fuel=args.fuel, reps=args.reps, quiet=args.quiet,
                passes=args.passes, batch=args.batch, image=args.image)
    args.out.write_bytes(data)
    print(f"  {len(programs)} programs, {len(data) // UF2_BLOCK} blocks, "
          f"{len(data):,} bytes -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
