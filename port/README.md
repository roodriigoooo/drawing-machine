# `port/` — the drawing VM in C, for a Cortex-M0+

`dm/vm/interp.py` is the specification. This is the device implementation, and
it is required to match it **exactly** rather than closely. The full record —
what is measured, what is not, and what is still owed — is
[`docs/claim4.md`](../docs/claim4.md); the argument for each design decision is
in the headers, next to the code it justifies.

| file | what is in it |
|---|---|
| `include/dm_isa.h`, `src/dm_isa.c` | the opcode table, mirrored from `dm/isa/spec.py` |
| `include/dm_vm.h` | the public API, **and the argument for fixed point and for streaming** |
| `src/dm_vm.c` | the interpreter — no allocation, no recursion, no libc, no float |
| `host/dm_trace.c` | native harness: bytecode on stdin, trace on stdout, for the differ |
| `bare/dm_harness.c` | **the same harness, bare-metal — one copy, every machine** |
| `bare/dm_machine.h` | the six functions a machine supplies, and nothing more |
| `bare/dm_io.h` | how a corpus gets in and a trace gets out, which differs per machine |
| `bare/semihost.h`, `bare/runtime.c` | the debugger-side syscalls; the three AEABI helpers |
| `qemu/` | nRF51822 under QEMU: linker script, startup, machine backend |
| `pico/` | RP2040, linked into SRAM, no debugger: the same three files plus its registers |

## The three things worth knowing before reading the code

**There is no floating point, and nothing was given up to remove it.** The
reference computes geometry in `double`, but every value it produces is an exact
multiple of `1/CURVE_STEPS³`: corner points are integers, and Bézier samples are
taken at `t = i/S` with `S` a power of two, so the Bernstein coefficients are
binary fractions. Scaling by `S³` makes the numerator *be* the fixed-point
coordinate — no division, no rounding, no shift. The port is therefore bit-exact
with the reference, and `scripts/conformance.py` compares with `==`.

**Geometry is streamed, so peak RAM does not depend on the program.** A literal
port would buffer each path and blow past the part's SRAM on a curve-heavy
drawing. Points are emitted as produced; `path_end` reports what the path turned
out to be. Measured consequence: **0 bytes of static RAM, 492 bytes of stack,
whatever the program.**

**There is exactly one bare-metal harness, and that is the whole reason a
silicon trace can be compared with an emulated one.** `bare/dm_harness.c` owns
the line protocol, the framing and the formatter; a machine directory owns a
linker script, a startup file and six functions that know nothing about any of
it. Two harnesses would drift and then agree about different things — a failure
that reports success — so `tests/test_pico.py` asserts there is one.

## Build and check

```bash
make host                              # the harness the differ drives
make device                            # Cortex-M0+ objects and .su stack reports
make qemu                              # bare-metal image for QEMU's microbit
make pico                              # bare-metal image for an RP2040, into SRAM

python3 scripts/conformance.py         # every corpus on this machine, exact diff
python3 scripts/conformance.py --sanitise   # the same, under UBSan
python3 scripts/conformance.py --qemu       # the same, as Thumb on an emulated M0
python3 scripts/conformance.py --pico       # the same, on RP2040 silicon, trace over UART
python3 scripts/footprint.py           # flash, static RAM, peak stack
python3 scripts/cycles.py              # cycles per drawing, on the part
python3 -m pytest tests/test_port.py tests/test_pico.py
```

`make device` prefers `arm-none-eabi-gcc` and falls back to `clang
--target=thumbv6m-none-eabi`, so a machine with Xcode needs nothing installed.
`make qemu` and `make pico` use clang and `ld.lld` directly and need no GNU
toolchain at all.

## Running on the part, with no debugger

The usual way to get code onto a Cortex-M0+ is SWD, which needs a second piece
of hardware. This path needs none, and the constraint it works around is worth
stating plainly:

| direction | available? | how |
|---|---|---|
| host → part | yes | BOOTSEL, then a UF2 dropped on a USB drive |
| part → host | **no** | once the image runs, BOOTSEL is gone |

So getting a program *in* is easy and getting anything *out* is the whole
problem. Three decisions follow, and together they are the design:

1. **The corpus travels with the image.** `scripts/uf2.py` packs both into one
   UF2 — the image at `0x20000000`, the programs at the address `pico/link.ld`
   reserves — so the part wakes with its work already in SRAM. The bootrom's
   drag-and-drop loader accepts blocks addressed to SRAM, which is what makes
   this possible at all.
2. **Output leaves on a pin.** `GP0` is UART TX at 3.3 V, read by any USB-serial
   adapter — including an Arduino UNO with `RESET` jumpered to `GND`, which
   turns that board into a bare bridge. Only the 3.3 V → 5 V direction is ever
   used, so no level shifter is needed and nothing can drive 5 V into a pin that
   is not tolerant of it.
3. **The board returns to BOOTSEL by itself.** The reserved region holds 180 KB
   and the full conformance corpus is several times that, so a sweep is split
   into batches. After each pass the firmware calls the bootrom's
   `reset_usb_boot`, so only the *first* batch needs someone to hold the button.

**Nothing is ever written to the board's flash.** `pico/link.ld` has no flash
region, which is why the cycle number means anything — an XIP-resident image
measures the vendor's wait states and cache as much as the interpreter — and
also why a failed experiment costs a power cycle rather than a board.

```bash
python3 scripts/pico.py --ports        # which serial device is the adapter
python3 scripts/pico.py --run          # one HALT, executed on silicon, traced back
python3 scripts/uf2.py --out run.uf2   # just build a UF2, to drag by hand
```

No installs: `clang` and `ld.lld` build the image, and the host side is Python's
`termios`.

## What this does not establish

Everything before `--pico` measured or reasoned about code that had never run on
the part: conformance runs the native build, `--sanitise` rules out the
undefined behaviour that could let two targets disagree, and `--qemu` runs Thumb
on a *functional model*. Those are an argument and a model, not silicon.
[`docs/claim4.md`](../docs/claim4.md) says which of claim 4's numbers each one
settles.

(ASan is opt-in via `make host-san SAN=undefined,address`: its runtime hangs at
process start on this project's macOS/arm64 host, which is an environment fault
and not a finding.)
