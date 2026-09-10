# Claim 4 — execution, run by run

> **Update:** the later [silicon run](claim4-bringup.md) completed
> 12,670/12,670 exact RP2040 traces and measured timing. The pre-silicon status
> below is historical. The preserved [120-row timing record](../artifacts/silicon/pico_cycles.json)
> includes 40 QuickDraw programs averaging 7,334 cycles/drawing from SRAM at
> 12 MHz. Energy remains unmeasured. The device executes the VM, not the model.

"The emitted bytecode runs on a Cortex-M0+ within budget." Until 2026-08-10 this
claim had nothing on disk at all, and `PLAN.md` said so in four separate places.
It now has a port, an exactness result, a conformance harness and two of its four
numbers. It still has **no measurement taken on silicon**, and the two numbers
that need silicon are named below rather than estimated.

Section references of the form §N are to `PLAN.md`.

---

## What the claim is about, since §9.11's wording invites one wrong reading

§9.11 lists the work as "Quantize, C port, Cortex-M0+ measurement". **The thing
that executes on the part is the VM, not the model.** The model emits bytecode;
the device interprets it. An 825k-parameter model is ~825 KB at int8, which is
25× the flash of the part class this project names, so on-device inference was
never what claim 4 asserted and no amount of quantization would make it so.

**"Quantize" therefore means the geometry pipeline, and the result is stronger
than quantization usually is: the error is zero.** See below.

## The exactness result — the port loses nothing by having no FPU

A Cortex-M0+ has no floating-point unit, so porting a VM that computes in
`double` is normally a choice between soft-float (kilobytes of flash, hundreds of
cycles per operation) and fixed point (fast, but a different answer, and then
every comparison against the reference needs a tolerance and every tolerance
needs a justification).

Neither is necessary here, because **the reference VM's geometry is already
integral at a scale of `CURVE_STEPS³`**:

- Every corner point is an integer. `place()` clamps an 8-bit operand plus an
  integer offset, and the only writer of the current point is a `place()` result.
- Curve samples are taken at `t = i/S` for `i = 1..S`. With `S` a power of two,
  `u³, 3u²t, 3ut², t³` are exactly `A/S³, B/S³, C/S³, D/S³` for integers with
  `A+B+C+D = S³`.

So a flattened coordinate is `(A·x₀ + B·x₁ + C·x₂ + D·x₃)/S³` with an integer
numerator, and choosing `DM_FRAC_BITS = 3·log₂S` makes that numerator *be* the
fixed-point value — no division, no rounding, not even a shift. The reference
computes the same quantity in `double`, where every intermediate is a small
multiple of a power of two and is therefore also exact.

**Consequence: the port is bit-exact with its specification, and
`scripts/conformance.py` compares with `==`.** There is no tolerance anywhere in
that file, which is the point — a tolerance is where a port and its
specification are permitted to disagree, and this one is not.

> **The precondition is `CURVE_STEPS` being a power of two, and it is a real
> constraint rather than a formality.** At `S = 10` the reference's own
> coefficients stop being binary fractions, the reference starts rounding, and no
> integer port can match it. `DM_CURVE_STEPS_LOG2` is the only knob and there is
> deliberately no way to ask for a non-power-of-two. `tests/test_port.py` checks
> the relation is *derived* in the header rather than a literal that happens to
> agree today.

## The second design decision: peak RAM does not depend on the program

The reference accumulates a `path` list and turns it into a stroke or a region
when it ends. Ported literally that is unbounded — a 3 KB program of curves is
~7,000 points, ~56 KB — which does not fit the part this project claims to
target, and it would have been the sort of failure that only appears on the
largest input.

The port streams instead: each point is emitted as it is produced and `path_end`
reports what the path turned out to be (`stroke`, `region`, or `discarded`, since
the reference drops paths under two and three points respectively). A consumer
that must know up-front can buffer; a plotter or rasteriser that can defer does
not have to. **The interpreter's own footprint becomes a constant**, and the
measurement below is that constant.

## Measured 2026-08-10 — footprint

`scripts/footprint.py`, Cortex-M0+, `-Os`, `clang --target=thumbv6m-none-eabi`
(the `arm-none-eabi-gcc` path is supported and was not available on this machine):

| | ISA v1 | **ISA v2** | note |
|---|---|---|---|
| flash, `.text` | 1,354 | **1,848** | the interpreter, plus the transform tier's three helpers |
| flash, `.rodata` | 11 | **14** | `dm_instr_size[]`, the only ISA table on the device |
| **flash, total** | **1,365** | **1,862** | +497 bytes, +36% |
| **static RAM (`.data` + `.bss`)** | **0** | **0** | no globals; two traces can run concurrently |
| **peak stack** | **288** | **492** | +204 bytes, and still **independent of program length** |
| unwind metadata | 8 | 32 | `.ARM.exidx`, discarded by any bare-metal linker script |

[Figure 7](figs/fig7_device_cost.svg) draws both axes and the depth sweep.

**The transform tier cost 2× what `docs/direction.md` §2.4 predicted, and the
prediction is worth reading against the measurement.** It budgeted "~6 ints × 4 =
~96 bytes" for a depth-4 transform stack. The stack itself is 48, and the other
156 are the things a sketch does not see: the cached composition the hot path
reads (12), a step and a scope floor in every `REPEAT` frame (16), and the rest
compiler spills from a wider dispatch and a `place()` that now returns two
coordinates because a rotation mixes them. **No falsifier fires** — §2.6 said the
depth bound would have to shrink if peak stack exceeded a budget, and 492 bytes
is 3% of the smallest target's SRAM and 0.2% of an RP2040's.

Composing through pointers rather than by value was worth **56 bytes of flash and
36 of stack** against the first version, which is the whole of `dm_xform_then`'s
argument copying; it is also the clearer spelling, since composing in place is
what the operation is. **Peak stack is linear in the depth bound at 12 bytes per
scope** — the size of one `dm_xform` — measured over depths 1 to 16
(`scripts/depth_cost.py`), so §2.6's knob has a price per turn: the by-value
version cost the same stack as *three extra scopes* would have bought.

Two of those are checked rather than reported: `scripts/footprint.py` fails if
static RAM is non-zero, and fails if any frame is `dynamic` or `bounded` instead
of `static` — the latter is what makes 288 an exact worst case rather than a
typical one. **The interpreter does not recurse** (`REPEAT` nesting lives in a
fixed `DM_MAX_REPEAT_DEPTH` array), so the call graph is a straight line and the
stack bound is a sum instead of a fixed point. The transform tier keeps that
property: its scopes live in a second fixed array bounded by
`DM_MAX_XFORM_DEPTH`, so a program with four nested transforms and four nested
loops costs exactly what one with none does.

**For scale.** A QuickDraw program in this project averages 112.7 bytes (§3), so
**the interpreter costs about twelve drawings.** Against a 32 KB / 8 KB part —
the small end of the M0+ range — that is ~4% of flash and ~3.5% of SRAM, leaving
the rest for the drawings and the output device. Against the RP2040 it is noise.

## Measured 2026-08-10 — conformance

`scripts/conformance.py`. Every trace compared field by field: stroke, disc and
region lists in their own order, each fault's kind and `pc`, the instruction
count, and the `halted` flag. Fault `detail` strings are excluded and are the
only thing excluded.

| corpus | programs | result |
|---|---|---|
| edge cases | 23 | identical |
| fuzz (uniform + structured) | 1,000 | identical |
| Tier A L0 / L1 / flat | 3,000 | identical |
| QuickDraw val | 1,000 | identical |
| Tabler val | 512 | identical |
| **total, at two fuel settings** | **12,670 traces** | **identical** |

Three things about that table are deliberate:

- **The fuzz corpus is not decoration.** A structured corpus reaches almost none
  of the fault handling, and the fault paths are where an interpreter and its
  port drift apart. Two fuzzers, because they fail in different places: uniform
  bytes reach `UNKNOWN_OPCODE` almost immediately, and a structured fuzzer emits
  real instructions with real operand counts so it runs deep enough to reach the
  repeat machinery, truncating a quarter of programs so `TRUNCATED` is reached at
  every instruction width rather than only the widest.
- **Every corpus runs twice, at `DEFAULT_FUEL` and at 25.** `OUT_OF_FUEL` is the
  one fault the reference can raise at *any* instruction, and at the default it
  is never reached by anything except the fuzzer.
- **The edge cases are written out rather than generated**, one per fault path
  plus the boundaries a fuzzer hits rarely: zero-count `REPEAT` *and* depth
  overflow at once, `HALT` inside an open repeat, a two-point `FILL`, `WIDTH`
  before a flush, clamping in both directions through a nested offset, `COLOR`
  as a known no-op. Each is an assertion about the specification, and a name in
  a list is easier to check against `dm/vm/interp.py` than a seed is.

## Measured 2026-08-10 — the Thumb build executes

`scripts/conformance.py --qemu`. The port is built for `thumbv6m-none-eabi`,
linked against a hand-written vector table and linker script, and run on QEMU's
`microbit` machine — an nRF51822, **Cortex-M0, 256 KB flash and 16 KB SRAM**.
Corpus and fuel arrive over ARM semihosting; the trace comes back the same way,
in the same line protocol the native harness emits, and is diffed by the same
code.

**12,670/12,670 traces identical**, across every corpus and both fuel settings —
the same total as the native sweep, now with the port's own instructions
actually executing.

Three things the port had to grow to run bare-metal, each worth recording
because none is visible in the source:

- **`-ffreestanding` stops the compiler assuming a library, not emitting calls
  into one.** Zeroing a struct became `__aeabi_memclr4`, and on ARMv6-M — which
  has `MULS` and **no divide instruction** — any `/` became `__aeabi_uidiv`.
  `port/qemu/runtime.c` supplies them. `dm_vm.c` needs only the first: it
  contains no division at all, because the fixed-point scheme was chosen so
  flattening is a multiply-add.
- **The harness streams the corpus** rather than loading it. A buffer-the-corpus
  harness does not fit in 16 KB, which is the footprint argument arriving as a
  build failure.
- **32-bit ARM's `SYS_EXIT` takes the reason in `r1`, not a pointer to a block.**
  Passing a pointer makes the emulator read the pointer value as the reason and
  exit non-zero on a successful run. `SYS_EXIT_EXTENDED` is the form that carries
  a status.

### Instructions per drawing — and why this is not a cycle count

Measured with `-accel tcg,one-insn-per-tb=on -d exec`, on QuickDraw val
programs averaging 160 bytes, with a **null sink** so the count is the
interpreter's rather than the harness's decimal formatter's. Taken as the
difference between a 10-program and a 30-program run, so fixed startup cancels:

| | |
|---|---|
| per drawing | **4,414 instructions** |
| per bytecode byte | **27.4 instructions** |
| fixed startup | ~5,621 instructions |

> **This is an instruction count. It is not cycles, and the difference is not
> pedantry.** QEMU models neither flash wait states nor the prefetch buffer, and
> both are a vendor choice rather than a core property — the same binary on two
> M0+ parts at one clock does not take the same number of cycles. Multiplying
> 4,414 by an assumed cycles-per-instruction would produce a number that looks
> like a measurement and is an assumption, which is the shape of every
> instrument fault this project has had. **The board is what settles it.**
>
### What the work is actually proportional to

Program length is the obvious predictor and it is the misleading one. Per-program
counts over 120 programs from three corpora, one QEMU run each with the
empty-corpus baseline subtracted:

| model | R² |
|---|---|
| program length alone | 0.80 |
| bytecode instructions decoded alone | 0.98 |
| **decode count + points emitted + discs** | **0.999** |

```
instructions ≈ 220 + 49.6·(instructions decoded) + 17.9·(points emitted) + 33.9·(discs)
```

**The ISA is a compression format, so byte count is the wrong unit for cost.** A
`REPEAT` byte is cheap to store and expensive to run, which is exactly the
property the representation exists for — and it means a device's budget has to be
computed from what a program *expands to*, never from its size on flash. The two
coefficients say where the cycles go: ~50 instructions to decode and dispatch one
bytecode instruction, ~18 per point of geometry emitted.
[Figure 5](figs/fig5_instructions.svg)

> One honest note on that fit: it was taken *after* `port/qemu/runtime.c`'s
> `memset` was made word-wise. With the byte-at-a-time version `dm_vm_run`'s
> ~150-byte state zero cost ~600 instructions, inflating a 160-byte drawing by
> ~14%. A measurement taken through a deliberately naive runtime measures the
> runtime.

> A second gap: QEMU's `microbit` is a Cortex-**M0**, not an M0+. The
> instruction set is identical (ARMv6-M) so the trace comparison is unaffected,
> but the pipelines differ — which is another reason the cycle figure is not
> available here.

## What this does **not** establish, stated plainly

**The port has never executed on real silicon**; QEMU is a functional model.
Everything about timing and energy remains unmeasured.

What carries a host result over to a target that has never run it is the absence
of undefined behaviour — signed overflow and out-of-range shifts are exactly the
constructs that can differ between two targets while both look correct on one.
So `scripts/conformance.py --sanitise` reruns the whole comparison against a
UBSan build with `-fno-sanitize-recover=all`, and the differ fails on *any*
byte written to stderr rather than only on a non-zero exit — a sanitiser that
reports and continues is otherwise indistinguishable from a clean run.

**Result: 12,670/12,670 traces identical under UBSan, no diagnostics**, plus a
separate 8,400-program fuzz pass (412,387 emitted records, 0.36 s, clean).

> **ASan is not in the default sanitiser set and the reason is environmental.**
> Its runtime hangs at process start on this project's macOS/arm64 host — on a
> one-byte `HALT` program, where the same binary without it runs in 3 ms — so it
> is a fault in the environment rather than a finding, and leaving it in would
> have meant a check nobody could run. `make host-san SAN=undefined,address`
> asks for it where it works. The interpreter indexes only a caller-supplied
> buffer and a fixed-size frame array, so UBSan is the one carrying the argument
> either way.

**All of this is an argument, not a measurement.** It says the C has no
construct whose meaning could change on another target. It does not say the
Thumb build produces these traces, because the Thumb build has not been run.

## What is still owed, and what it costs

| number | status | what it needs |
|---|---|---|
| flash bytes | **measured**, 1,862 at ISA v2 (1,365 at v1) | nothing |
| peak SRAM | **measured**, 0 static + 492 stack | nothing |
| instructions/drawing | **measured**, 4,414 | nothing |
| cycles/instruction | **not measured** | silicon or a cycle-accurate model |
| J/drawing | **not measured** | silicon and a current probe |

**Cycles cannot be honestly emulated, and the reason is specific.** Cortex-M0+
instruction timing is deterministic — no cache, no branch prediction, a
single-cycle multiplier — so a cycle count *is* computable from a disassembly and
an execution trace. What is not a core property is the **flash wait states and
prefetch buffer**, which are a vendor choice: the same binary on two M0+ parts at
the same clock does not take the same number of cycles. So "cycles on a Cortex-M0+"
is part-specific by construction, and any number this project quotes has to name
the part.

## The instrument built for the board — and the constraint that shaped it

**Built 2026-08-14; RAM-entry contract corrected 2026-08-20; hardware rerun
pending.** Recorded here because the design is a claim about what will be
measured, and it is falsifiable before any number exists.

The obvious way onto a Cortex-M0+ is SWD, which needs a debug probe. This
project had one Pico and no probe, so the path built instead needs neither — and
the constraint is worth stating because everything else follows from it:

| direction | available? | how |
|---|---|---|
| host → part | yes | BOOTSEL, then a UF2 dropped on a USB drive |
| part → host | **no** | once the image runs, BOOTSEL is gone |

Getting a program *in* is easy; getting anything *out* is the whole problem.

- **The corpus travels with the image.** `scripts/uf2.py` packs both into one
  UF2 — the RP2040 bootrom's loader accepts blocks addressed to SRAM, which is
  what the SDK calls a `no_flash` binary — so the part wakes with its work
  already in memory and nothing needs to reach it after it starts.
- **The trace leaves on `GP0`** as 3.3 V UART, read by any USB-serial adapter.
  Only that direction is used, so nothing can drive 5 V into a pin that is not
  tolerant of it.
- **The board returns to BOOTSEL by itself** via the bootrom's
  `reset_usb_boot`, so a sweep that outgrows the 180 KB corpus region is split
  into batches without anyone touching the button between them.

**`port/pico/link.ld` has no flash region, and that is the measurement
decision.** An XIP-resident image would time Raspberry Pi's QSPI wait states and
cache alongside the interpreter — the vendor term this document has said since
2026-08-10 is not a core property. Running from SRAM removes it rather than
characterising it, which makes the result comparable to any M0+ with
zero-wait-state memory and a **lower bound** for the same code out of XIP.

### RAM-UF2 startup contract

The RP2040 boot ROM enters a RAM-only UF2 at the lowest address present in the
download. It does not consume that address as the Cortex reset vector. The image
therefore starts with `port/pico/entry.S` at `0x20000000`: it masks interrupts,
writes `VTOR`, executes synchronization barriers, loads MSP from the table at
`0x20000100`, and branches through its reset word. `startup.c` then copies
`.data`, clears `.bss`, and calls `main`.

This mirrors the Pico SDK's no-flash crt0 layout, where executable reset-entry
code precedes the VTOR-aligned vectors. The linker entry is the trampoline as
well, while `scripts/uf2.py` deliberately uses the lowest load address because
that is what the boot ROM physically enters. The ELF/UF2 tests assert both
contracts and reject an image whose entry is not the first downloaded address.

Primary references: [RP2040 datasheet, UF2 boot behavior](https://datasheets.raspberrypi.com/rp2040/rp2040-datasheet.pdf)
and [Pico SDK `crt0.S`](https://github.com/raspberrypi/pico-sdk/blob/master/src/rp2_common/pico_crt0/crt0.S).

Four guards ride with the number:

- **The clock source is asserted directly.** The Pico backend reads back the
  XOSC-selected `clk_ref` and `clk_sys` mux state, their hardware `SELECTED`
  acknowledgements, and both undivided dividers, then emits `clock_source xosc`.
  The host requires that record. SysTick/TIMER frequency recovery remains a
  secondary check, but it cannot by itself prove the source because both counters
  could follow the same wrong clock.
- **The frequency cross-check is two-sided.** SysTick counts core cycles and the
  TIMER block counts microseconds; the measured span is accumulated across the
  program only, never across UART output. A recovered frequency outside the
  configured tolerance in either direction refuses publication.
- **The instrument's own cost is measured and subtracted.** The empty-corpus
  minimum is recorded by the firmware and removed by the host rather than
  treated as interpreter work.
- **A wrapped counter is a failure, not a data point.** SysTick is 24 bits and
  the RP2040's M0+ has no DWT, so the backend returns a sentinel and the harness
  refuses it.

**Why cycles/instruction is a legitimate ratio here:** it divides RP2040 cycles
by QEMU instructions, which are two runs of two builds. `-mcpu=cortex-m0` and
`-mcpu=cortex-m0plus` compile `dm_vm.c` to **byte-identical instructions** on
this toolchain, so the two are counting one program. `tests/test_pico.py`
asserts it; if a future toolchain separates them, the ratio stops being a
property of the part and the test fails rather than the number quietly changing.

One refactor landed with it, and it is the reason a silicon trace is comparable
to an emulated one at all: **`port/bare/dm_harness.c` is now one harness serving
both machines**, with `dm_machine.h` (clock, cycles) and `dm_io.h` (corpus in,
trace out) as the only things a machine supplies. All three harnesses — native,
QEMU, Pico — now bracket a run with `BEGIN`/`END`, which the Pico needs because
a firmware with no channel inward cannot know when a reader attached, and which
the other two carry so no path can quietly lose the framing.

Three ways forward, cheapest first:

1. ~~**QEMU**~~ — **DONE 2026-08-10.** 12,670 traces identical on an emulated
   Cortex-M0, and 4,414 instructions per drawing. It gave what it can give.
2. **An RP2040 board (~$4).** A real Cortex-M0+ with a cycle counter reachable
   from `SysTick`, and the only cheap path to a real cycles/instruction figure.
   With a USB power meter or an INA219 shunt it also gives J/drawing, the one
   number nothing else can produce. **Instrument built 2026-08-14; board present,
   not yet run.** No debug probe is needed — see the section above.
3. **A cycle-accurate model** (Arm Fast Models / Cycle Models). Exact and
   licensed; worth it only if a board is not an option.

**The order is not by cost, it is by what each one settles.** QEMU settles
"does it run on ARM at all"; only the board settles the two numbers the budget
argument is actually made of.
