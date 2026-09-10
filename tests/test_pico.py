"""The RP2040 path, checked without a board attached.

Claim 4's silicon half has one property that makes it hard to test: the
interesting failures happen on hardware, and hardware is not in CI. What *is*
checkable here is everything between the source and the wire — and on this path
that is more than usual, because there is no debugger to notice a mistake. A
UF2 with a wrong address, a corpus header whose fields the C reads in a
different order, or a cycle counter clocked off the wrong source all produce a
board that runs and answers, incorrectly.

So the tests below are mirrors and invariants rather than behaviour: the C and
the Python agree about the corpus header; the packer and the linker agree about
the memory map; the two builds compile the interpreter identically, without
which cycles-per-instruction is a ratio of two different programs.
"""

from __future__ import annotations

import errno
import re
import shutil
import struct
import subprocess
import sys
from pathlib import Path

import pytest

# `scripts/` is a directory of entry points rather than an installed package, so
# it is reached the same way the scripts reach each other.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts import cycles, pico, uf2

ROOT = Path(__file__).resolve().parent.parent
PORT = ROOT / "port"
PICO_MACHINE = (PORT / "pico" / "machine.c").read_text()
QEMU_MACHINE = (PORT / "qemu" / "machine.c").read_text()
MACHINE_H = (PORT / "bare" / "dm_machine.h").read_text()
IO_H = (PORT / "bare" / "dm_io.h").read_text()
HARNESS = (PORT / "bare" / "dm_harness.c").read_text()
RP2040_H = (PORT / "pico" / "rp2040.h").read_text()
LINK_LD = (PORT / "pico" / "link.ld").read_text()
MAKEFILE = (PORT / "Makefile").read_text()

IMAGE_BASE = 0x20000000


# --------------------------------------------------------------------------
# one harness, two machines


def test_there_is_exactly_one_bare_metal_harness():
    """The anti-drift property, as an assertion rather than a convention.

    `scripts/conformance.py` diffs a silicon trace and an emulated one against
    the same reference with the same parser, and that is only meaningful while
    both came from one line protocol. A second bare-metal harness would let the
    two drift and then agree about different things — a failure that reports
    success.

    `port/host/dm_trace.c` is deliberately outside this: it is the *native*
    harness, it speaks stdio and argv rather than either of these channels, and
    its agreement is not a convention but a checked fact — all three are read by
    the same `parse_port_traces`, so a divergence in the protocol is a parse
    error or a failed diff on the very next run.
    """
    assert (PORT / "bare" / "dm_harness.c").exists()
    bare = [p for d in ("bare", "qemu", "pico") for p in (PORT / d).rglob("*.c")]
    strays = [p for p in bare
              if p.name != "dm_harness.c" and "dm_vm_run(program" in p.read_text()]
    assert strays == [], f"a second bare-metal harness runs the corpus: {strays}"


def declared_functions(header: str) -> set[str]:
    return set(re.findall(r"^(?:\w[\w ]*?[ *])(\w+)\(", header, re.MULTILINE))


@pytest.mark.parametrize("machine", [PICO_MACHINE, QEMU_MACHINE], ids=["pico", "qemu"])
def test_every_machine_implements_both_contracts(machine):
    """A missing backend function is a link error for the machine being built
    and nothing at all for the other one, so the machine nobody built today is
    the one that breaks. Checked as text, for both, always."""
    timing = declared_functions(MACHINE_H)
    assert timing == {"dm_machine_init", "dm_clk_hz", "dm_cycles_available",
                      "dm_clock_source_valid",
                      "dm_cycles_start", "dm_cycles_stop", "dm_us_now"}
    io = declared_functions(IO_H)
    assert io == {"dm_io_begin", "dm_config_fuel", "dm_config_reps", "dm_config_quiet",
                  "dm_config_batch", "dm_corpus_rewind", "dm_corpus_next",
                  "dm_io_write", "dm_io_again", "dm_io_end"}
    for name in timing | io:
        assert re.search(rf"^\w[\w ]*?[ *]{name}\(", machine, re.MULTILINE), name
    assert "dm_machine_name" in machine


def test_qemu_reports_that_it_has_no_cycle_counter():
    """`docs/claim4.md` argues that a cycle count from QEMU would be an
    assumption wearing a measurement's clothes — the emulator models neither
    flash wait states nor the prefetch buffer, and both are a vendor choice.
    The backend's job is to say so, and this is that decision written down
    somewhere it can fail."""
    body = QEMU_MACHINE.split("int dm_cycles_available(void)")[1]
    assert body.split("}")[0].strip().endswith("return 0;")


# --------------------------------------------------------------------------
# the instrument


def test_the_pico_counts_the_processor_clock_and_not_the_watchdog_tick():
    """SysTick's clock source is one bit. Selecting the watchdog's 1 MHz tick
    instead of the core produces a count that is twelve times too small at this
    part's crystal frequency, is internally consistent, and looks like a
    Cortex-M0+ that is very fast."""
    assert re.search(r"SYST_CSR = SYST_CSR_CLKSOURCE \| SYST_CSR_ENABLE;", PICO_MACHINE)
    assert re.search(r"#define SYST_CSR_CLKSOURCE \(1u << 2\)", RP2040_H)


def test_a_wrapped_cycle_count_is_reported_as_a_wrap():
    """The counter is 24 bits. A difference taken across a wrap is a small
    positive number and is indistinguishable from a fast program, so the
    backend has to return the sentinel and the harness has to refuse it."""
    assert "return DM_CYCLES_OVERFLOW;" in PICO_MACHINE
    assert "SYST_CSR_COUNTFLAG" in PICO_MACHINE
    assert "DM_CYCLES_OVERFLOW" in HARNESS


def test_the_microsecond_timer_is_read_low_word_first():
    """Reading `TIMELR` latches `TIMEHR`; the reverse order straddles a carry
    once every 71 minutes and reports a timestamp an hour away. The order is the
    entire atomicity guarantee, so it is asserted rather than commented."""
    read = re.search(r"const uint32_t low = (\w+);\s*\n\s*const uint32_t high = (\w+);",
                     PICO_MACHINE)
    assert read, "dm_us_now does not read the timer as a low/high pair"
    assert (read.group(1), read.group(2)) == ("TIMER_LR", "TIMER_HR")


def test_the_pico_asserts_the_selected_xosc_clock_source_directly():
    """A shared wrong source makes two derived counters agree by construction.

    The firmware therefore has to inspect the RP2040 mux controls and their
    hardware-selected status, not infer the source from SysTick/TIMER arithmetic.
    """
    assert "int dm_clock_source_valid(void)" in PICO_MACHINE
    for register in ("CLK_REF_CTRL", "CLK_REF_SELECTED", "CLK_SYS_CTRL",
                     "CLK_SYS_SELECTED", "CLK_REF_DIV", "CLK_SYS_DIV"):
        assert register in PICO_MACHINE
    assert "clock_source xosc" in HARNESS
    assert "dm_clock_source_valid()" in HARNESS


def test_the_host_clock_guard_rejects_both_frequency_directions():
    def measured(cycles_count: int, elapsed_us: int) -> dict:
        return {
            "clk_hz": 12_000_000,
            "reps": 1,
            "elapsed_us": elapsed_us,
            "rows": [{"cycles": cycles_count}],
        }

    assert cycles.check_clock(measured(12_000, 1_000)) == 12_000_000
    with pytest.raises(SystemExit, match="recovered core frequency"):
        cycles.check_clock(measured(6_000, 1_000))
    with pytest.raises(SystemExit, match="recovered core frequency"):
        cycles.check_clock(measured(16_000, 1_000))


def test_the_cycles_protocol_requires_the_direct_clock_source_record():
    frame = (
        "BEGIN 0\n"
        "clock_source xosc\n"
        "machine rp2040-cortex-m0plus-sram\n"
        "clk_hz 12000000\n"
        "reps 1\n"
        "overhead 10\n"
        "c 0 12000 1\n"
        "elapsed_us 1000\n"
        "END 1\n"
    )
    assert cycles.parse(frame, 1)["clock_source"] == "xosc"
    with pytest.raises(SystemExit, match="clock_source"):
        cycles.parse(frame.replace("clock_source xosc\n", "clock_source rosc\n"), 1)
    with pytest.raises(SystemExit, match="header is incomplete"):
        cycles.parse(frame.replace("clock_source xosc\n", ""), 1)


def test_the_elapsed_span_excludes_the_output():
    """The clock cross-check divides accumulated cycles by accumulated
    microseconds to recover the core frequency. At 115200 baud the UART is four
    orders of magnitude slower than the measurement, so a span that enclosed the
    output would recover ~0.001 MHz against a configured 12 and the guard would
    either be vacuous or always fire."""
    body = HARNESS.split("static uint32_t cycle_corpus(")[1]
    started = body.index("const uint64_t started = dm_us_now();")
    accumulated = body.index("measured_us += dm_us_now() - started;")
    between = body[started:accumulated]
    assert "put_u32" not in between and "puts_" not in between, (
        "output happens inside the span the clock check divides by"
    )


def test_the_host_and_the_firmware_agree_about_the_baud_rate():
    """Two constants in two languages describing one wire. Disagreeing produces
    a stream of bytes that arrives and never frames, which reads as a hardware
    fault and is not one."""
    declared = re.search(r"#define DM_UART_BAUD (\d+)u", PICO_MACHINE)
    assert declared, "DM_UART_BAUD is not defined in the machine backend"
    assert int(declared.group(1)) == pico.BAUD


# --------------------------------------------------------------------------
# the corpus that travels with the image


def test_the_corpus_header_is_the_same_struct_on_both_sides():
    """**The mirror that matters most on this path.**

    There is no debugger here, so the header is the *entire* channel between
    host and part: which programs, how much fuel, how many repetitions, how many
    passes. If the C reads field 3 where the packer wrote field 4, the board runs
    a valid corpus with the wrong settings and reports a well-formed answer to a
    question nobody asked. A link error cannot catch that and neither can a
    trace diff.
    """
    fields = dict(re.findall(r"HDR_(\w+) = (\d+)", PICO_MACHINE))
    assert {name: int(index) for name, index in fields.items()} == {
        "MAGIC": 0, "VERSION": 1, "FUEL": 2, "REPS": 3,
        "QUIET": 4, "N_PROGRAMS": 5, "BYTES": 6, "PASSES": 7, "BATCH": 8,
    }
    assert uf2.CORPUS_HEADER.size == 9 * 4

    magic = re.search(r"#define DM_CORPUS_MAGIC (0x[0-9A-Fa-f]+)u", PICO_MACHINE)
    assert magic and int(magic.group(1), 16) == uf2.CORPUS_MAGIC
    version = re.search(r"#define DM_CORPUS_VERSION (\d+)u", PICO_MACHINE)
    assert version and int(version.group(1)) == uf2.CORPUS_VERSION
    words = re.search(r"#define DM_CORPUS_HEADER_WORDS (\d+)u", PICO_MACHINE)
    assert words and int(words.group(1)) * 4 == uf2.CORPUS_HEADER.size


def test_the_corpus_blob_frames_programs_the_way_the_firmware_walks_them():
    programs = [b"\x0d", b"\x01\x02\x03", b""]
    blob = uf2.corpus_blob(programs, fuel=99, reps=3, quiet=True, passes=7, batch=5)
    (magic, version, fuel, reps, quiet,
     count, size, passes, batch) = uf2.CORPUS_HEADER.unpack_from(blob)
    assert (magic, version, fuel, reps, quiet, count, passes, batch) == (
        uf2.CORPUS_MAGIC, uf2.CORPUS_VERSION, 99, 3, 1, 3, 7, 5)

    payload = blob[uf2.CORPUS_HEADER.size:]
    assert len(payload) == size
    cursor, walked = 0, []
    while cursor + 4 <= len(payload):
        length, = struct.unpack_from("<I", payload, cursor)
        cursor += 4
        walked.append(payload[cursor:cursor + length])
        cursor += length
    assert walked == programs


def test_a_corpus_of_zeros_is_refused_rather_than_executed():
    """An image started without a corpus finds whatever the previous run left in
    SRAM. Zeroed SRAM decodes as a run of `MOVE 0,0` — a *valid* program that
    traces, diffs and means nothing. The magic word is what makes that a stop
    rather than a measurement."""
    assert "halt_forever(\"FAILED: no corpus at _corpus_base" in PICO_MACHINE
    assert uf2.CORPUS_MAGIC != 0


# --------------------------------------------------------------------------
# the packer and the linker agree about the map


@pytest.fixture(scope="module")
def image() -> Path:
    if shutil.which("clang") is None or shutil.which("make") is None:
        pytest.skip("no cross toolchain")
    built = subprocess.run(["make", "-s", "-B", "-C", str(PORT), "pico"],
                           capture_output=True, text=True, check=False)
    if built.returncode != 0:
        pytest.skip(f"pico image did not build: {built.stderr[-400:]}")
    return PORT / "build" / "dm_pico.elf"


def test_the_elf_reader_finds_the_symbols_the_packer_needs(image):
    elf = uf2.read_elf(image)
    assert elf.base == IMAGE_BASE
    assert elf.entry == elf.symbol("ram_entry")
    assert elf.entry & 1
    base, end = elf.symbol("_corpus_base"), elf.symbol("_corpus_end")
    assert base % uf2.UF2_PAYLOAD == 0, "the corpus must start on a UF2 block boundary"
    assert end > base
    assert uf2.capacity(image) == end - base - uf2.CORPUS_HEADER.size


def test_the_elf_reader_refuses_something_that_is_not_an_image(tmp_path):
    """A parser that returns a plausible answer for the wrong file is worse than
    one that raises: the failure would be a board that loads and hangs."""
    not_an_elf = tmp_path / "corpus.bin"
    not_an_elf.write_bytes(b"\x01\x00\x00\x00\x0d")
    with pytest.raises(uf2.Uf2Error):
        uf2.read_elf(not_an_elf)


def _elf_bytes(elf: uf2.Elf, address: int, length: int) -> bytes:
    for base, contents in elf.segments:
        offset = address - base
        if 0 <= offset <= len(contents) - length:
            return contents[offset:offset + length]
    raise AssertionError(f"ELF has no loadable bytes at {address:#010x}")


def test_the_ram_entry_trampoline_and_relocated_vector_table_are_distinct(image):
    """A RAM UF2 starts with executable entry code, not Cortex reset words.

    The trampoline is the physical boot target. It installs the 256-byte-aligned
    vector table, loads its SP and reset target, and branches through that target;
    the table is therefore checked as data at its relocated address rather than
    assumed to be what the boot ROM consumes at the image base.
    """
    elf = uf2.read_elf(image)
    entry = elf.symbol("ram_entry")
    vector = elf.symbol("vector_table")
    estack = elf.symbol("_estack")
    reset = elf.symbol("reset_handler")
    fault = elf.symbol("fault_handler")

    entry_address = entry & ~1
    assert elf.base == entry_address == IMAGE_BASE
    assert vector % 0x100 == 0
    assert vector == IMAGE_BASE + 0x100
    assert vector != entry

    words = struct.unpack("<16I", _elf_bytes(elf, vector, 16 * 4))
    assert words[0] == estack
    assert words[1] == reset | 1
    assert words[2] == fault | 1
    assert words[3] == fault | 1
    assert words[1] & 1 and words[2] & 1 and words[3] & 1

    entry_bytes = _elf_bytes(elf, entry_address, 0x100)
    assert entry_bytes[:2] == b"\x72\xb6", "entry does not begin with CPSID i"
    assert struct.pack("<I", vector) in entry_bytes
    assert struct.pack("<I", 0xE000ED08) in entry_bytes


def test_the_generated_uf2_starts_with_the_ram_trampoline(image):
    data = uf2.pack([b"\x0d"], image=image)
    _, _, _, address, size, _, _, _ = struct.unpack_from("<8I", data)
    payload = data[32:32 + size]
    assert address == IMAGE_BASE
    assert payload[:2] == b"\x72\xb6"
    assert payload[:4] != struct.pack("<I", uf2.read_elf(image).symbol("_estack"))


def test_the_uf2_is_one_contiguous_aligned_span(image):
    """The bootrom is loading an image, not patching memory. `.bss` is ~37 KB
    that the file does not carry, so a corpus placed after it would leave a
    37 KB hole mid-download — which is why `port/pico/link.ld` puts the reserved
    region between `.data` and `.bss`."""
    data = uf2.pack([b"\x0d"] * 8, image=image)
    assert len(data) % uf2.UF2_BLOCK == 0
    count = len(data) // uf2.UF2_BLOCK

    addresses = []
    for index in range(count):
        m0, m1, flags, addr, size, block_no, total, family = struct.unpack_from(
            "<8I", data, index * uf2.UF2_BLOCK)
        end, = struct.unpack_from("<I", data, index * uf2.UF2_BLOCK + 508)
        assert (m0, m1, end) == (uf2.UF2_MAGIC_START0, uf2.UF2_MAGIC_START1,
                                 uf2.UF2_MAGIC_END)
        assert family == uf2.RP2040_FAMILY_ID
        assert flags & uf2.UF2_FLAG_FAMILY_ID
        assert (size, block_no, total) == (uf2.UF2_PAYLOAD, index, count)
        addresses.append(addr)

    assert addresses[0] == IMAGE_BASE
    assert addresses == list(range(IMAGE_BASE, IMAGE_BASE + count * uf2.UF2_PAYLOAD,
                                   uf2.UF2_PAYLOAD))
    assert uf2.read_elf(image).symbol("_corpus_base") in addresses


def test_a_corpus_larger_than_the_region_is_refused(image):
    over = uf2.capacity(image) + 1
    with pytest.raises(uf2.Uf2Error, match="reserved region"):
        uf2.pack([b"\x0d" * (over - 4)], image=image)


def test_batches_fit_the_region_and_preserve_order():
    """A conformance sweep is several times the reserved region, so it is split.
    Losing or reordering a program between batches would be a diff against the
    wrong reference program, which is a divergence report that is not one."""
    programs = [bytes([i % 251]) * (i % 40 + 1) for i in range(500)]
    limit = 1024
    groups = uf2.batches(programs, limit)
    assert [p for g in groups for p in g] == programs
    for group in groups:
        assert sum(len(p) + 4 for p in group) <= limit


def test_a_single_oversized_program_is_a_failure_not_a_batch():
    with pytest.raises(uf2.Uf2Error, match="does not fit"):
        uf2.batches([b"\x00" * 5000], 1024)


# --------------------------------------------------------------------------
# the memory map


def test_the_pico_image_is_linked_into_sram_with_no_flash_region():
    """The reason the cycle number means anything. An XIP-resident link measures
    Raspberry Pi's flash wait states and cache as much as the interpreter, and
    `docs/claim4.md` says at length why that is a vendor number rather than a
    core one."""
    regions = re.findall(r"^\s*(\w+)\s*\([rwx]+\)\s*:\s*ORIGIN\s*=\s*(0x[0-9a-fA-F]+)",
                         LINK_LD, re.MULTILINE)
    assert regions == [("RAM", hex(IMAGE_BASE))], regions


def test_the_bss_clear_cannot_reach_the_corpus():
    """`startup.c` zeroes `_sbss`..`_ebss` on entry. If the reserved region ever
    moved inside that span, every run would execute a corpus of zeros — valid
    `MOVE 0,0` instructions, so it would trace and diff and be wrong. The linker
    script asserts it; this asserts the assertion."""
    assert re.search(r"ASSERT\(_sbss >= _corpus_end,", LINK_LD)


def test_the_reserved_corpus_region_is_not_carried_in_the_file(image):
    """`.corpus` is `NOLOAD`, and this is what makes that load-bearing.

    If it ever became a loadable section, every UF2 would carry 180 KB of
    padding, every batch would take fifteen times as long to write, and the
    packer would be filling a region it is also supposed to fill — with the
    linker's zeros winning or losing depending on block order.
    """
    elf = uf2.read_elf(image)
    carried = sum(len(contents) for _, contents in elf.segments)
    assert carried < 32 * 1024, f"the file carries {carried:,} bytes; the corpus leaked in"
    assert elf.symbol("_corpus_base") >= IMAGE_BASE + carried


def test_the_image_leaves_room_for_a_stack(image):
    """The interpreter's worst case is 492 bytes and the harness's buffers are
    static, so the margin here is enormous by design. It is checked because the
    thing that would eat it — a buffer that grows with the program — is exactly
    the design `dm_vm.h` exists to avoid."""
    elf = uf2.read_elf(image)
    stack = elf.symbol("_estack") - elf.symbol("_ebss")
    assert stack > 8 * 1024, f"only {stack:,} bytes of stack below _estack"


def test_the_two_bare_builds_compile_the_interpreter_identically(image):
    """**The precondition for dividing cycles by instructions.**

    Cycles per instruction is measured across two machines: QEMU counts the
    instructions an emulated Cortex-M0 retires, and the RP2040 counts the cycles
    a Cortex-M0+ takes. That ratio is only a property of the *part* while both
    are counting the same instruction stream — and the two images are compiled
    with different `-mcpu` values, which is exactly the sort of difference that
    changes a schedule without changing a result.

    ARMv6-M is one instruction set and clang's two tunings agree here, so the
    objects disassemble the same. If a future toolchain ever separates them, CPI
    stops being a ratio of two measurements of one thing and this fails, which
    is the point: the alternative is a number that is quietly about the compiler.
    """
    subprocess.run(["make", "-s", "-C", str(PORT), "qemu"], check=True)
    build = image.parent  # the fixture is what guarantees the pico objects exist
    disassembly = []
    for machine in ("qemu", "pico"):
        proc = subprocess.run(["xcrun", "llvm-objdump", "-d",
                               str(build / machine / "dm_vm.o")],
                              capture_output=True, text=True, check=False)
        if proc.returncode != 0:
            pytest.skip("no llvm-objdump")
        # The first two lines name the file, which is the one thing that differs.
        disassembly.append(proc.stdout.split("\n", 2)[2])
    assert disassembly[0] == disassembly[1], (
        "the cortex-m0 and cortex-m0plus builds of dm_vm.c are different "
        "instructions, so cycles/instruction would compare two programs"
    )


def test_the_pico_build_asks_for_the_larger_output_buffer():
    """32 KB rather than the emulator's 1 KB. The Pico has 264 KB of SRAM and
    the nRF51 has 16, and a bigger buffer is fewer, larger writes down a slow
    wire.

    It is a variable rather than a literal so a bring-up build can set it to 1
    and turn the buffer off — the harness flushes at the end of a pass, so a
    hang inside `trace_corpus` otherwise takes the whole trace with it and the
    host cannot tell that from a dead UART. The **default** is what the
    measurement uses, so the default is what this pins.
    """
    assert re.search(r"pico,-mcpu=cortex-m0plus -DDM_OUT_MAX=\$\(PICO_OUT_MAX\)", MAKEFILE)
    assert re.search(r"^PICO_OUT_MAX \?= 32768$", MAKEFILE, re.MULTILINE)


def test_a_capture_holding_several_passes_is_matched_by_identity():
    """The reader picks a block by the identity the device echoed, not by
    position.

    A capture legitimately contains more than one pass: the adapter is still
    draining the previous batch when the next is written, and a re-drag after a
    missed pass leaves both in the stream. Taking the first complete block would
    then diff batch 1 against batch 0's reference programs and report every
    drawing as a divergence — a failure that looks like a broken port.
    """
    stream = (
        "curve_st\n"                               # tail of a pass already in flight
        "BEGIN 0\nfrac_bits 12\nEND 3\n"
        "BEGIN 1\nfrac_bits 12\nEND 7\n"
    )
    blocks = {int(m.group(1)): int(m.group(3)) for m in pico.PASS.finditer(stream)}
    assert blocks == {0: 3, 1: 7}

    partial = "BEGIN 1\nfrac_bits 12\n#0\n"
    assert pico.PASS.search(partial) is None, "an unterminated pass must not match"


# --------------------------------------------------------------------------
# the bring-up diagnostics
#
# A pty is not a UART, and the difference is exactly the thing that cannot be
# tested here: no baud rate, no line discipline on the wire, no adapter. What a
# pty *does* exercise is everything this project wrote — the `termios` raw
# setup, the non-blocking open, the `select` loop and the verdict each
# diagnostic returns — and those are the parts that would otherwise first run
# with a board attached and a person waiting.


@pytest.fixture
def wire():
    """A `/dev/cu.*`-shaped file descriptor pair, master side returned.

    `open_serial` is given the slave path, so the code under test opens a real
    terminal device and configures it exactly as it would the adapter.
    """
    import os
    import pty

    master, slave = pty.openpty()
    yield master, Path(os.ttyname(slave))
    for fd in (master, slave):
        try:
            os.close(fd)
        except OSError:
            pass


@pytest.mark.parametrize("transient_errno", [errno.EACCES, errno.EIO])
def test_uf2_open_retries_transient_mount_denials(
    monkeypatch,
    tmp_path,
    transient_errno,
):
    attempts = 0

    def fake_open(path, flags, mode):
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise OSError(transient_errno, "BOOTSEL volume not ready")
        return 17

    monkeypatch.setattr(pico.os, "open", fake_open)
    monkeypatch.setattr(pico.time, "sleep", lambda _seconds: None)

    target = tmp_path / "dm.uf2"
    assert pico.open_uf2_target(target, timeout_s=1.0) == 17
    assert attempts == 3


def test_uf2_open_times_out_on_persistent_mount_denial(monkeypatch, tmp_path):
    attempts = 0
    ticks = iter((0.0, 0.0, 0.2))

    def fake_open(path, flags, mode):
        nonlocal attempts
        attempts += 1
        raise OSError(errno.EACCES, "BOOTSEL volume still not ready")

    monkeypatch.setattr(pico.os, "open", fake_open)
    monkeypatch.setattr(pico.time, "monotonic", lambda: next(ticks))
    monkeypatch.setattr(pico.time, "sleep", lambda _seconds: None)

    with pytest.raises(
        pico.PicoError,
        match=r"appeared but did not become writable within 0\.1s",
    ):
        pico.open_uf2_target(tmp_path / "dm.uf2", timeout_s=0.1)

    assert attempts == 2


def test_uf2_open_does_not_retry_unrelated_errors(monkeypatch, tmp_path):
    attempts = 0

    def fake_open(path, flags, mode):
        nonlocal attempts
        attempts += 1
        raise OSError(errno.EINVAL, "invalid open request")

    monkeypatch.setattr(pico.os, "open", fake_open)

    with pytest.raises(pico.PicoError, match="could not open"):
        pico.open_uf2_target(tmp_path / "dm.uf2")

    assert attempts == 1


def test_uf2_write_retries_eio_before_the_first_byte(monkeypatch, tmp_path):
    opened: list[int] = []
    closed: list[int] = []
    writes = 0
    descriptors = iter((17, 18))

    def fake_open(path, flags, mode):
        descriptor = next(descriptors)
        opened.append(descriptor)
        return descriptor

    def fake_write(descriptor, data):
        nonlocal writes
        writes += 1
        if writes == 1:
            raise OSError(errno.EIO, "mount not ready")
        return len(data)

    monkeypatch.setattr(pico.os, "open", fake_open)
    monkeypatch.setattr(pico.os, "write", fake_write)
    monkeypatch.setattr(pico.os, "close", closed.append)
    monkeypatch.setattr(pico.time, "sleep", lambda _seconds: None)

    pico.write_uf2_target(
        tmp_path / "dm.uf2",
        b"x" * 512,
        timeout_s=1.0,
    )

    assert opened == [17, 18]
    assert closed == [17, 18]
    assert writes == 2


def test_uf2_write_never_retries_after_partial_transfer(monkeypatch, tmp_path):
    opens = 0
    writes = 0

    def fake_open(path, flags, mode):
        nonlocal opens
        opens += 1
        return 17

    def fake_write(descriptor, data):
        nonlocal writes
        writes += 1
        if writes == 1:
            return 256
        raise OSError(errno.EIO, "device disappeared")

    monkeypatch.setattr(pico.os, "open", fake_open)
    monkeypatch.setattr(pico.os, "write", fake_write)
    monkeypatch.setattr(pico.os, "close", lambda _descriptor: None)

    with pytest.raises(pico.PicoError, match="failed after 256 bytes"):
        pico.write_uf2_target(
            tmp_path / "dm.uf2",
            b"x" * 512,
            timeout_s=1.0,
        )

    assert opens == 1
    assert writes == 2


def test_every_declared_bringup_stage_exists_in_the_firmware():
    """The Python table and the C checkpoints are the same set.

    This is the one way the bisection can lie. A stage listed here with no
    `DM_BRINGUP(n)` in `machine.c` builds an image with *no* checkpoint at all,
    which runs the whole firmware, hangs, and gets reported as "dies at stage n"
    — pointing the repair at code that is fine. The reverse leaves a checkpoint
    the sweep never selects, so a hang hides behind a stage nobody runs.
    """
    in_c = {int(n) for n in re.findall(r"DM_BRINGUP\((\d+)\)", PICO_MACHINE)}
    in_harness = {int(n) for n in re.findall(r"DM_HARNESS_BRINGUP\((\d+)\)", HARNESS)}
    declared = in_c | in_harness
    assert declared == set(pico.BRINGUP_STAGES), (
        f"machine/harness have {sorted(declared)}, pico.BRINGUP_STAGES has "
        f"{sorted(pico.BRINGUP_STAGES)}"
    )


def test_unrestricted_bringup_runs_stage_17_as_the_first_precondition():
    assert pico.BRINGUP_ORDER[0] == 17
    assert set(pico.BRINGUP_ORDER) == set(pico.BRINGUP_STAGES)


def test_stage_17_requires_both_fault_record_and_bootsel_reboot():
    expected_pc = 0x200001F2
    fault = b"!\nFAULT pc=0x200001f2 lr=0x200016a1\n"
    assert pico.bringup_stage_success(17, True, fault, expected_pc)
    assert not pico.bringup_stage_success(17, False, fault, expected_pc)
    assert not pico.bringup_stage_success(17, True, fault[2:], expected_pc)
    wrong_pc = b"!\nFAULT pc=0x200001f0 lr=0x200016a1\n"
    assert not pico.bringup_stage_success(17, True, wrong_pc, expected_pc)
    assert not pico.bringup_stage_success(17, True, fault)
    assert pico.bringup_stage_success(16, True, b"")


def test_bss_diagnostic_requires_exact_verdict_and_bootsel():
    assert pico.bss_check_success(True, pico.BSS_SUCCESS)
    assert not pico.bss_check_success(False, pico.BSS_SUCCESS)
    assert not pico.bss_check_success(True, pico.BSS_SUCCESS + b"noise")
    assert "bss_nonzero()" in HARNESS
    assert HARNESS.index("dm_io_begin();") < HARNESS.index('"BSS_CLEAR ok\\n"')


def test_the_measured_image_carries_no_checkpoint():
    """Stage 0 is the default and compiles to nothing.

    The bisection exists to find a hang, not to ride along in the image whose
    cycle count is the result. A checkpoint left in would put a bootrom call and
    a comparison inside the measured build.
    """
    assert re.search(r"#ifndef DM_BRINGUP_STAGE\s*\n#define DM_BRINGUP_STAGE 0", PICO_MACHINE)
    assert "#define DM_BRINGUP(stage) ((void)0)" in PICO_MACHINE
    assert re.search(r"^PICO_EXTRA \?=\s*$", MAKEFILE, re.MULTILINE), \
        "PICO_EXTRA must default to empty or every build carries a stage"


def test_a_bluetooth_device_does_not_make_the_adapter_ambiguous(monkeypatch):
    """A paired Bluetooth device is a `/dev/cu.*` with an arbitrary name.

    `NOT_ADAPTERS` can only exclude the ones Apple names predictably, and this
    machine has a `/dev/cu.RoBose`. Without a preference for USB-shaped names,
    every run through `conformance.py --pico` and `cycles.py` — neither of which
    has a `--port` flag — fails as ambiguous the moment the UNO enumerates.
    """
    monkeypatch.setattr(pico, "serial_ports",
                        lambda: [Path("/dev/cu.RoBose"), Path("/dev/cu.usbmodem1101")])
    assert pico.choose_port(None) == Path("/dev/cu.usbmodem1101")


def test_two_usb_adapters_are_still_refused(monkeypatch):
    """Preference is not a guess. Two plausible adapters means the run must ask
    rather than pick, because picking wrong reads the wrong board in silence."""
    monkeypatch.setattr(pico, "serial_ports",
                        lambda: [Path("/dev/cu.usbmodem1101"), Path("/dev/cu.usbserial-A9")])
    with pytest.raises(pico.PicoError, match="more than one"):
        pico.choose_port(None)


def test_the_environment_names_the_port_for_the_callers_that_cannot(monkeypatch):
    """`conformance.py --pico` and `cycles.py` reach the board through `run()`.

    Neither takes a port argument, so the disambiguation has to live somewhere
    all three read; a flag on each would be the same answer in three places.
    """
    monkeypatch.setenv(pico.PORT_ENV, "/dev/null")
    monkeypatch.setattr(pico, "serial_ports", list)
    assert pico.choose_port(None) == Path("/dev/null")

    monkeypatch.setenv(pico.PORT_ENV, "/dev/cu.nothing-here")
    with pytest.raises(pico.PicoError, match=pico.PORT_ENV):
        pico.choose_port(None)


def test_the_probe_pattern_cannot_be_passed_by_accident():
    """Every byte distinct and every bit position exercised.

    A loopback that returns a stuck line, one repeated character, or bytes with
    the top bit cleared by a framing error must *fail*. A probe of identical
    bytes, or one confined to a few bit positions, passes all three.
    """
    assert len(set(pico.PROBE)) == len(pico.PROBE), "a repeated byte hides a stuck line"
    ones = zeros = 0
    for byte in pico.PROBE:
        ones |= byte
        zeros |= ~byte & 0xFF
    assert ones == 0x7F and zeros == 0xFF, "some bit never toggles in both directions"


def test_a_clean_loopback_passes_and_says_so(wire, capsys):
    import os
    import threading

    master, device = wire
    # The adapter's job, in one line: whatever it is handed, it hands back.
    def echo():
        try:
            os.write(master, os.read(master, 1 << 12))
        except OSError:
            pass

    threading.Thread(target=echo, daemon=True).start()
    assert pico.loopback(device, seconds=5.0) == 0
    assert "byte-identical" in capsys.readouterr().out


def test_a_silent_adapter_fails_rather_than_hanging(wire, capsys):
    """Nothing echoing is the missing-jumper case, and it must terminate."""
    _, device = wire
    assert pico.loopback(device, seconds=0.5) == 1
    assert "nothing came back" in capsys.readouterr().err


def test_a_corrupted_return_is_named_apart_from_a_short_one(wire, capsys):
    """The two have different repairs, so they may not share a message.

    A short return is a jumper that is not seated; a corrupted one is a baud or
    framing mismatch, which would equally corrupt a real trace and is the more
    dangerous of the two because `read_pass` would report it as "bytes arriving
    but never framing".
    """
    import os
    import threading

    master, device = wire

    def mangle():
        try:
            data = bytearray(os.read(master, 1 << 12))
            data[3] ^= 0x80  # exactly what a wrong baud does to one byte
            os.write(master, bytes(data))
        except OSError:
            pass

    threading.Thread(target=mangle, daemon=True).start()
    assert pico.loopback(device, seconds=5.0) == 1
    assert "corrupted return" in capsys.readouterr().err


def transmit(master: int, data: bytes) -> None:
    """Write once the listener has opened the port, not before.

    `open_serial` ends with a `TCIFLUSH`, which is right — a run must not parse
    whatever the adapter buffered before it attached — and it means a test that
    writes first measures the flush instead of the diagnostic.
    """
    import os
    import threading
    import time

    def later():
        time.sleep(0.3)
        try:
            os.write(master, data)
        except OSError:
            pass

    threading.Thread(target=later, daemon=True).start()


def test_listen_accepts_a_framed_pass_and_rejects_an_unframed_one(wire, capsys):
    master, device = wire
    transmit(master, b"BEGIN 0\nfrac_bits 12\ncurve_steps 16\n#0\n= 1 1 0\nEND 1\n")
    assert pico.listen(device, seconds=3.0) == 0
    assert "1 complete BEGIN..END pass" in capsys.readouterr().out

    transmit(master, b"#0\n= 1 1 0\n")  # well-formed lines, no frame
    assert pico.listen(device, seconds=3.0) == 1


def test_listen_calls_a_wrong_baud_by_its_name(wire, capsys):
    """Bytes arriving that are mostly non-printable is one fault with one repair,
    and it is the fault `read_pass` reports as "never frames"."""
    master, device = wire
    transmit(master, bytes([0xFE, 0x81, 0x00, 0xC3] * 16))
    assert pico.listen(device, seconds=3.0) == 1
    assert "baud rate is wrong" in capsys.readouterr().err
