# Claim 4 on silicon — the bring-up log

**Status 2026-08-20: CLOSED — PASS on silicon; visual demo not yet packaged.**
The corrected RAM image completed the full startup gate, **12,670/12,670**
functional traces matched the reference on the RP2040, and the guarded cycle
record joined all 120 programs to their QEMU instruction counts. The earlier
board observations remain below as explicitly superseded history.

`docs/claim4.md` holds the claim and its numbers, but has not yet been
synchronised to this hardware result. This file holds the debugging, the bench
gate, and the next demo handoff.

> Sections 2–8 describe the pre-correction image as it was observed on
> 2026-08-15. Section 9 supersedes their startup, fault-instrument, and stage-13
> conclusions. Section 10 is the post-correction silicon result. Section 11 is
> the current resume point.

---

## 0. What this changed

**The model, ISA and claims 1–3 remain untouched; Claim 4 gained the silicon
evidence it was missing.** The implementation remains confined to Pico-specific
startup/backend code and its host instruments:

- `port/pico/machine.c` — the RP2040 machine backend
- `port/pico/entry.S` — the RAM-UF2 entry trampoline
- `port/pico/link.ld` — the RAM entry/vector layout
- `port/pico/startup.c` — its C runtime and fault vector table
- `scripts/pico.py` — the host-side driver

Plus a `PICO_OUT_MAX`/`PICO_EXTRA` variable in `port/Makefile` (pico target only),
`tests/test_pico.py`, and the optional `.bss`/stage-13 diagnostics in the shared
harness.

**Untouched:** all of `dm/` (model, ISA, codecs, evaluation), `port/src/dm_vm.c`
and `dm_isa.c` (the interpreter), `port/host/`, and `port/qemu/`.

The native, QEMU, UBSan and footprint results remain unchanged: **1,862 B
flash / 0 static RAM / 492 B peak stack**, and the historical QEMU instruction
record remains the join source. What is new is a full **12,670/12,670**
reference diff on real RP2040 silicon and a guarded SRAM-resident timing record:
QuickDraw averages **7,334 cycles/drawing** and **1.959 cycles/instruction** over
40 programs, after subtracting 13 cycles of instrument overhead.

This retracts every "never executed on real silicon" and "cycles not measured"
statement for Claim 4. It does not add an energy result: **J/drawing remains
explicitly descoped**, and claims 1–3, Direction 1's state work, attribution,
Bézier and conditioning still never involved a board.

What was *gained*: the `clk_peri` bug in §5 is a real defect in the port's device
code that no host-side check could have found, because nothing in a functional
model has a reset that waits on a clock. That is a footnote worth writing.

## 1. What is being attempted

Execute the ISA interpreter on a real Cortex-M0+ and read the trace back. No
debug probe exists, so:

- image and corpus travel together in one UF2 dropped on `RPI-RP2`, linked into
  SRAM (`port/pico/link.ld` has no flash region);
- the trace leaves on `GP0` as 115200-baud UART;
- an ELEGOO UNO R3 with `RESET` tied to `GND` is the bridge, its ATmega16U2
  reading `GP0` on `TX→1`;
- a finite run calls the bootrom's `reset_usb_boot`, handing the board back to
  BOOTSEL.

**J/drawing was descoped 2026-08-15** — no meter, and `docs/directions.md` puts
silicon in feasibility rather than the paper core. Claim 4 targets 3 of 4
numbers, the fourth deliberately dropped rather than owed.

## 2. The symptom

`scripts/pico.py --run` loads, the board starts, and **nothing arrives on the
serial port and the board never returns to BOOTSEL.** Both channels silent.

## 3. The instruments built to find it

None existed on 2026-08-14; the path had never met hardware.

| instrument | what it observes |
|---|---|
| `pico.py --loopback` | host, driver, adapter, baud with **no Pico attached**, via a `D0`–`D1` jumper |
| `pico.py --listen` | raw bytes, framed or not — "no bytes" vs "bytes that never frame" |
| `pico.py --bringup [N]` | stage 17 is an explicit first precondition; it requires the named UDF's exact `!`/`FAULT pc=... lr=...` record and BOOTSEL, then the remaining checkpoints use the reboot channel. |
| `pico.py --probe [--verbose]` | the **measured** image (no checkpoints) on one `HALT`, watching wire *and* reboot |
| `pico.py --bss-check` | a diagnostic image pre-fills and scans every word of `.bss`, then requires the exact `BSS_CLEAR ok` + quiet one-HALT record and BOOTSEL |
| `--out-max N` | rebuilds with `DM_OUT_MAX=N`; at 1 the buffer is off, so the last byte received is the instruction before the fault |
| `dm_pico_fault` | fault handler reporting the **stacked PC** over UART instead of semihosting; early faults stay off UART until `uart_ready` |

Stages 1–11 are `machine.c`'s own bring-up. Stages 12–16 are the only hooks this
machine has inside the *shared* harness — `dm_corpus_rewind`, `dm_config_batch`,
`dm_io_write`, `dm_corpus_next`, `dm_io_again`. Resolution inside `main`'s pass
loop cannot be improved without putting machine-specific code in
`port/bare/dm_harness.c`, where it does not belong. Stage 17 is not a rung: it
tests the instrument.

## 4. Ruled out — and by which observation

Each row is an observation, not an argument.

| ruled out | observation |
|---|---|
| host, macOS driver, adapter, 115200 | `--loopback`, Pico unplugged: **64/64 bytes byte-identical** |
| the dupont wire, its crimps, the `TX→1` socket | loopback **chained through that wire**: 64/64 byte-identical |
| bootrom launch, `link.ld`, vector table, `startup.c`, `.bss` clear | stage 1 |
| crystal oscillator | stage 2 |
| `clk_sys`/`clk_ref` onto the crystal | stage 3 |
| microsecond timer, SysTick | stages 4, 5 |
| `clk_peri`, UART0/IO_BANK0/PADS_BANK0 reset release | stage 6 (after the §5 fix) |
| UART0 programming, `GP0` pad mux | stage 7 |
| UF2 corpus at `_corpus_base`, header intact | stage 8 |
| the transmitter | stage 9: one byte drained |
| FIFO drain, sustained throughput | stages 10, 11: 64 bytes, FIFO on *and* off |
| baud divisor, `GP0` routing, 3.3 V vs the 16U2's 3.0 V `V_IH` | stage 10 **with the wire attached**: `'0123456789…'`, 64 bytes byte-identical on the host |
| `dm_io_begin` returning; the harness entering its pass loop | stage 12 |
| `flush`, `dm_io_write`, `uart_put` as the cause of the `puts_` hang | stage 13 fails at **default** `DM_OUT_MAX=32768`, where `"BEGIN "` never flushes |
| the memory map | read from the **ELF** — §4.1 |
| the interpreter differing from spec | 12,670/12,670 QEMU traces identical, UBSan clean; `-mcpu=cortex-m0` and `-mcpu=cortex-m0plus` compile `dm_vm.c` byte-identically (`tests/test_pico.py` asserts it) |

> **Historical conclusion, narrowed by §9.** Stages 9–11 proved the complete
> host/adapter/wire/UART path and made further cable or voltage speculation
> unproductive. They did **not** prove the boot-ROM entry contract, VTOR/MSP or
> startup execution; treating stage 1 as that proof was the error §9 corrected.

### 4.1 The pre-correction memory map, from the ELF

Parsed out of `port/build/dm_pico.elf`, because a map that is *reasoned about* is
a map that can be wrong:

```
vector_table   0x20000000   64 B      _corpus_base   0x20001900   (256-aligned)
_sdata=_edata  0x20001844   (.data empty)
                                      _corpus_end    0x2002e900
_sbss          0x2002e900  ← exactly _corpus_end, so the ASSERT holds
_ebss          0x2003791c            _estack        0x20040000   (~34 KB stack)
out_len        0x2002e900   out       0x2002e904     program  0x2002e905
uart_ready     0x20037908
```

Everything is inside SRAM0–3 (`0x20000000`–`0x20040000`). `.bss` begins exactly
where the corpus ends. `out_len` is inside the zeroed span, so it is 0 at the
first `put()`. Nothing overlaps, nothing is truncated. **Not the bug.**

Vector table contents also verified: `SP=0x20040000`, `reset=0x2000156d`,
`HardFault=0x200015bd`, both with the Thumb bit set. `reset_handler`
disassembles to a correct `*0xe000ed08 = 0x20000000` VTOR write, and its `.data`
copy and `.bss` clear both terminate.

## 5. Pre-correction bugs found and fixed

**`clk_peri` ordering** (`machine.c`, `uart_start`) — **real, fixed, confirmed by
stage 6 going from fail to pass.** The original released UART0's reset and *then*
enabled `clk_peri`. `RESET_DONE` does not assert until the block has a clock, so
the wait spun forever. The pico-sdk never meets this because `clocks_init()` runs
long before any peripheral touches `RESETS` — reading the SDK does not reveal
that the order is load-bearing, and nothing in QEMU models a reset that waits on
a clock. Fixed: `clk_peri` first, disabled-then-enabled because its mux is not
glitchless.

**Semihosting in the fault handler** (`startup.c`) — **real, fixed, but did not
resolve the silence.** Faults were reported with `sh_write`/`sh_exit`, which are
`BKPT`; on ARMv6-M with no debugger attached `BKPT` escalates to HardFault, so
the fault handler faults and the board locks up silently. Replaced with a `naked`
handler passing the exception frame to `dm_pico_fault`. The handler still
produces no output — see §6.

**`build()` was incremental** (`scripts/pico.py`) — a workflow defect created by
this debugging. `build_stage()` compiles with `-DDM_BRINGUP_STAGE=N` using `-B`,
but a timestamp cannot see a `-D`. After any bring-up run, `make` reported
"nothing to be done" and would hand the **checkpointed** image to `--run`,
`conformance.py --pico` and `cycles.py` — an image that reboots partway through
`dm_io_begin` and never executes a program. A measurement taken with the wrong
instrument, reporting success. `build()` now always forces `-B`.

**`uart_ready` made `volatile`** — correct on principle, **not demonstrated to
have been the bug.** It is written on the normal path and read only from an
exception handler, a control flow the compiler cannot see, so a plain store may
legally be sunk past the `volatile asm` block that triggers the test fault.
`volatile` forbids that motion, and the current build provably stores 1 before
the `UDF`. It did **not** change the outcome: stage 17 is still silent.

> An earlier draft of this file claimed the non-volatile build demonstrably
> stored 0. That was read off the *stage-17* ELF and does not survive rebuilding
> both ways — register allocation shuffles between dumps. The claim is withdrawn.

## 6. Pre-correction stopping point — superseded 2026-08-20

> Historical only. The RAM-entry correction in §9 and the silicon result in §10
> supersede both failures below. They remain because this is the failure record,
> not because either is still open.

The pre-correction image left two failures, and they could have been one bug.

**(a) The fault path does not report.** Stage 17 executes a deliberate `UDF #0`
— the architecturally-defined undefined instruction, a guaranteed HardFault at a
known PC, placed after `uart_start()` so the transmitter is up. **The wire stays
silent.** Verified with `uart_ready` both plain and `volatile`.

This is the more important of the two, because it invalidates an inference used
throughout §4: "no `FAULT` line" was read as "not a fault". On ARMv6-M a fault
taken inside the HardFault handler, or an exception whose own register push
fails, enters **LOCKUP** with no output. A silent fault and a hang are then the
same observation. **Every "hang" below stage 17 is therefore unproven as a hang.**

**(b) The harness hangs inside `puts_("BEGIN ")`.** Bounded by stage 12 passing
(`dm_corpus_rewind`) and stage 13 failing (`dm_config_batch`, called immediately
after). No bytes at `--out-max 1`, where the second `put()` should flush `B`.
Stage 13 also fails at the **default** buffer size, where `"BEGIN "` never
flushes at all — so `flush`, `dm_io_write` and `uart_put` are not involved.

`puts_` was disassembled and is correct: a clean `ldrb`/`cmp`/`strb` loop,
`out_len` at `0x2002e900`, `out` at `0x2002e904`, flush only at the threshold.
**`put()` and `flush()` contain no loop.** Two stores cannot spin — but given
(a), they can fault silently. (a) and (b) are consistent with a single
memory-access fault whose reporter is broken.

## 7. Pre-correction next tests — superseded

Do not resume from this list. It records the tests owed by the old image; the
startup-contract review in §9 replaced them, and §10 records the completed gate.

Ordered so the instrument is fixed before it is trusted again.

1. **Print unconditionally.** Remove the `uart_ready` guard from `dm_pico_fault`
   entirely and re-run stage 17. If it still says nothing, no instruction of the
   handler is executing.
2. **Prove the handler is entered at all**, independent of the UART: have the
   handler call `reboot_to_bootsel()` instead of printing. `RPI-RP2` reappearing
   after a `UDF` would prove the vector is taken; silence would localise the
   failure to exception entry itself — stack push, VTOR, or LOCKUP.
3. **Only then** re-run stage 13, and read its result as a fault rather than a
   hang if a PC appears.
4. Turn any reported PC into a source line with
   `xcrun llvm-objdump -d --source port/build/dm_pico.elf`.

## 8. Rules this episode earned

- **A part with no debugger needs a second output channel before it needs a
  measurement.** `reset_usb_boot` was that channel and was already there; nobody
  had thought of a reboot as a bit of information.
- **A checkpoint proves only what it executes.** Stage 9 sent one byte and was
  read as "the transmitter works"; the failing case needed 50 bytes and a FIFO.
  A rung that does not exercise the failing case is a rung that lies.
- **Verify the instrument before trusting its silence.** Stage 17 should have
  been the *first* rung, not the fifteenth. Several rounds of reasoning rested on
  "no `FAULT` line means no fault"; the corrected ladder now runs it first, and
  §10 records its exact-PC silicon pass.
- **Never report a fault through a channel that requires the thing that faulted.**
  Semihosting needs a debugger; a `BKPT` without one is another HardFault.
- **A `-D` is invisible to `make`.** Any build that varies by macro must be `-B`,
  or a later run silently measures the earlier instrument.
- **Read the map out of the ELF, not the linker script.** §4.1 took two minutes
  and retired a hypothesis three rounds of reasoning had left alive.
- **State written on the normal path and read from an exception handler must be
  `volatile`.** The compiler cannot see that control flow.
- **Silence has two causes on ARMv6-M and they are not the same bug.** A hang
  spins; a fault inside the HardFault handler enters LOCKUP. Both look like a
  dead board.
- **QEMU models the core, not the chip.** The first confirmed silicon bug was
  peripheral bring-up ordering — precisely the layer a functional model omits.
  The later RAM-entry defect was a boot-ROM contract error. The board was what
  settled both the port and its timing.
- **During an automatic conformance sweep, never unplug in response to the
  generic BOOTSEL prompt.** A successful finite run returns by itself; unplugging
  while the host is opening the new mount turns readiness into `ENOENT`.
- **Retry a remount failure only before the first successful byte.** `EACCES`,
  `EIO`, `ENOENT`, `EBUSY` and `EROFS` occurred while the macOS FAT mount settled;
  a partial UF2 transfer remains fatal rather than silently restarted.
- **A visual demo must render the parsed device trace.** Rendering a host VM
  trace and placing a "Pico" badge beside it would demonstrate the renderer, not
  Claim 4.

## 9. Corrective implementation — 2026-08-20

The independent review found one high-confidence unifying defect: the RP2040
RAM-UF2 boot contract had been treated as a conventional Cortex reset-vector
launch. The boot ROM actually enters the lowest downloaded address directly.
The old image therefore executed the `_estack` word and the reset-vector word as
Thumb instructions, eventually falling through into `main` by accident. That
made stage 1 a false proof and explains the skipped VTOR, missing MSP setup,
unproven `.bss` clear, and silent deliberate faults.

The correction is now in the tree:

- `port/pico/entry.S` is linked at `0x20000000` and performs `cpsid i`, loads the
  table address, writes VTOR, executes `dsb`/`isb`, loads MSP, and branches
  through the reset vector.
- `vector_table` is explicitly 256-byte aligned at `0x20000100`; `startup.c`
  owns only `.data`/`.bss` runtime initialization and `main`.
- `scripts/uf2.py` records the ELF entry and refuses to pack an image whose
  Thumb entry is not the lowest load address.
- The named `dm_stage17_udf` emits the deliberate UDF. The handler emits `!`
  before touching the frame, prints stacked PC/LR as `FAULT pc=... lr=...`,
  drains UART, and reboots to BOOTSEL. The host reads the UDF address from the
  stage ELF and accepts stage 17 only for the exact record, exact PC, and reboot.
- Early faults are protected by a volatile `uart_ready` guard and therefore do
  not touch UART registers before `uart_start()`.
- `DM_BSS_PREFILL=1` dirties the complete BSS span before the real clear; `main`
  scans every word before machine init, keeps the result on its stack, and the
  host `--bss-check` path reports only after UART setup and requires the exact
  `BSS_CLEAR ok\nBEGIN 0\nEND 1\n` record plus BOOTSEL.
- Optional stage-13 microscope checkpoints 18–22 separately cover the `out_len`
  read, the zero invariant, an `out[0]` canary, all six `BEGIN` writes, and the
  existing `dm_config_batch` reboot checkpoint.

The cycle publication guard was also repaired: the Pico backend now reads back
the XOSC-selected `clk_ref`/`clk_sys` mux state and both undivided dividers, emits
`clock_source xosc`, and the host requires that record. The SysTick/TIMER ratio
is retained only as a secondary frequency check and is rejected in either
direction outside its tolerance; it is no longer treated as proof of the source.

Pre-hardware evidence after the correction was 44 focused Pico tests plus
successful stage-17 and BSS diagnostic builds with the trampoline and relocated
vectors. The following gate was frozen before touching the corrected image to
silicon; §10 records its completion rather than changing it after the result:

1. `python3 scripts/pico.py --bringup` must run stage 17 first; it must show
   `!`, one exact `FAULT pc=...` record at the ELF's `dm_stage17_udf` address,
   and a BOOTSEL return.
2. `python3 scripts/pico.py --bss-check --out-max 1` and again with the default
   buffer must report the exact BSS verdict and reboot.
3. Run stage 13 with `--out-max 1` and with the default buffer. If either fails,
   run stages 18–22 in order at the same buffer size.
4. Run quiet and verbose one-HALT probes at both buffer sizes, then obtain ten
   consecutive warm runs and three cold-power runs.
5. Require the exact one-HALT output from `python3 scripts/pico.py --run`.
6. Run Pico conformance on the edge corpus and then the full corpus; only after
   those pass, run and publish `scripts/cycles.py`.

## 10. Post-correction silicon result — CLOSED — PASS

Run on 2026-08-20 on the documented Raspberry Pi Pico and ELEGOO UNO R3 UART
bridge, with both boards separately USB-powered and the image executing from
SRAM. Every predeclared gate in §9 passed:

| gate | silicon observation |
|---|---|
| startup/fault instrument | Stage 17 emitted exactly `!\nFAULT pc=0x200012f8 lr=0x2000134f\n`; the PC matched the ELF's `dm_stage17_udf`, then the board returned to BOOTSEL. Stages 1–16 and 18–22 all reported back. |
| transmitter | Stage 9 returned `U`; stages 10 and 11 returned the same 64-byte `0123456789…` sequence with the FIFO enabled and disabled. |
| full-span BSS | Both `--out-max 1` and the production buffer returned the exact 27-byte `BSS_CLEAR ok\nBEGIN 0\nEND 1\n` record and rebooted. |
| former stage-13 failure | Stage 13 reported at both buffer sizes. At `out[1]`, `BEGIN` reached the wire and the sixth byte remained buffered at the checkpoint, exactly as the write order predicts. |
| measured image | Quiet and verbose one-HALT probes passed at `out[1]` and `out[32768]`; the verbose record was exactly `BEGIN`, `frac_bits 12`, `curve_steps 16`, `#0`, `= 1 1 0`, `END 1`. |
| repeatability | Ten consecutive warm one-HALT runs and three cold-power runs returned the exact record and BOOTSEL. |
| edge/quick conformance | The edge set passed 23/23 at fuel 100,000 and 23/23 at fuel 25. The final uninterrupted quick sweep was **58/58**. |
| full conformance | **12,670/12,670 traces identical** across edge, fuzz, Tier A L0/L1/flat, composed L2/flat, QuickDraw and Tabler, each at both fuel values. |
| cycle publication guard | `clock_source xosc`; configured clock 12,000,000 Hz; timer-recovered clock 11,917,964 Hz (−0.68%); five repetitions; 13-cycle instrument overhead; no wrap; all 120 rows joined to the exact QEMU instruction record. |

The cycle statistic is the minimum over five repetitions for each program, with
the 13-cycle instrument overhead subtracted; the table reports the mean across
40 programs per corpus:

| corpus | bytes/drawing | cycles/drawing | time at 12 MHz | cycles/instruction |
|---|---:|---:|---:|---:|
| QuickDraw | 157.0 | **7,333.9** | **0.611 ms** | **1.959** |
| Tier A L1 | 100.8 | **7,063.5** | **0.589 ms** | **1.921** |
| Tier A L0 | 43.5 | **2,633.7** | **0.219 ms** | **1.903** |

The row-level record is `runs/pico_cycles.json`, SHA-256
`8c6966425df28ac196ae0961f59e2b74f992c32466d88f0df436ccb5d71a47b3`.
It contains `instructions_joined: true`, no instruction note, and 120 rows.
**It is currently under the repository-wide `/runs/` ignore rule and is not yet
a durable tracked artifact.** The conformance terminal transcript was also not
written to a repository log. Preserve the JSON in a tracked artifact location
and synchronise `docs/claim4.md`/`docs/evidence.md` before treating the result as
published rather than locally reproduced.

### 10.1 Host remount defect found by the sweep

The first edge attempt passed its silicon diff and then exposed a separate host
race: macOS `fskit` can expose `/Volumes/RPI-RP2` before the FAT mount accepts a
create or its first byte. The observed transient errors were `EACCES` and `EIO`
at `open`, and `EIO` at a zero-byte first write. A manual unplug during an
automatic return additionally produced `ENOENT`.

`scripts/pico.py` now retries the observed mount-transition errors for a bounded
five-second readiness interval. A zero-byte write may close and retry; after any
successful byte, an error is fatal because silently restarting a partial UF2
would change the transaction being claimed. The focused Pico suite now passes
**50 tests**, including transient recovery, bounded timeout, unrelated-error
refusal, zero-byte retry and partial-transfer refusal. The final 58/58 quick run
crossed thirteen automatic remounts without operator intervention, and the full
sweep crossed the production sequence.

### 10.2 What the result does and does not establish

Claim 4's device implementation is now functionally exact on the tested RP2040
and measured at the core from zero-wait-state SRAM. The timing is a lower bound
for the same interpreter executing from vendor XIP flash; it is not a flash
configuration measurement. The conformance result is broad, but the timing
record is one physical board/session with 40 programs per corpus and five
within-session repetitions. No energy claim follows: J/drawing remains
descoped. Nothing here says the 825k-parameter model runs on the Pico; the model
emits bytecode and the Pico executes the VM.

## 11. Resume — package an honest visual silicon demo

The current minimal demonstration is runnable and has been executed:

```bash
python3 scripts/pico.py --run
```

It proves one real `HALT` went through UF2 → RP2040 → UART → exact host parser,
but terminal text does not make the result legible to an audience. The next
milestone is a self-contained browser view whose displayed geometry comes from
the Pico's UART trace.

### 11.1 Honesty contract

The demo data path must be:

```
frozen QuickDraw bytecode → UF2/SRAM → RP2040 VM → UART fixed-point trace
    → host parser → SVG/HTML
```

The host reference VM may compute an equality assertion, but **must not supply
the geometry being rendered**. A mismatch, fault, incomplete frame or failed
BOOTSEL return must refuse the page rather than fall back to a host drawing.
Normal one-HALT output does not contain a per-program cycle count, so the page
may show §10's aggregate timing table in a clearly separate methodology panel;
it must not label that aggregate as the selected drawing's measured latency.

### 11.2 MVP design

Implement `scripts/pico_demo.py` with one board load containing five fixed
QuickDraw validation programs — one each from `cat`, `dog`, `bus`, `car` and
`tree`. One batch avoids making a UI demo depend on repeated Finder remounts.
For each program:

1. display category, validation index, byte length, guest instruction count and
   bytecode hash/disassembly;
2. parse the actual fixed-point device trace through the conformance parser;
3. assert it equals the reference trace, then convert that **device object** to
   `dm.vm.render` geometry;
4. embed its SVG in a self-contained HTML gallery, with an expandable raw UART
   record and an explicit `RP2040 trace: exact match` badge;
5. animate the returned polylines as a replay, labelled **replay of the captured
   silicon trace**, not “live”, because today's `pico.run()` returns a complete
   frame rather than yielding points incrementally.

Target CLI — this is a design, **not implemented yet**:

```bash
python3 scripts/pico_demo.py \
  --categories cat dog bus car tree --split valid --index 0 \
  --out build/pico-demo.html
```

The HTML should have no network dependency and remain viewable after the boards
are disconnected. A later live mode would require refactoring `read_pass()` to
yield validated records incrementally; it is optional and should not block the
captured-trace MVP.

### 11.3 Acceptance and ordered next steps

1. **Preserve the measurement first:** move/copy the exact cycle JSON to a
   tracked artifact path without changing its bytes, record its SHA-256, then
   update `docs/claim4.md` and `docs/evidence.md`.
2. **Fix the operator prompt:** automatic post-pass remounts must say “wait; do
   not unplug”; BOOTSEL/replug instructions belong only to initial load or
   recovery after the process exits.
3. **Build the five-drawing MVP:** device trace is the renderer input; reference
   trace is assertion-only; one UF2 carries all five programs.
4. **Test the evidence boundary:** fixed-point conversion, category/index
   selection, mismatch/fault refusal, self-contained HTML, and a test that makes
   host-VM rendering fail if it is accidentally used as the visual source.
5. **Bench acceptance:** one command and one BOOTSEL entry produce five visible
   drawings, five exact-match badges and one raw trace record; disconnecting the
   boards does not change the page.
