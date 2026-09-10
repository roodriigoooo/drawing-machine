# Four builds of the same two source files.
#
#   make host      the conformance harness, built natively; what the differ runs
#   make device    Cortex-M0+ objects and their .su stack reports, into build/
#   make qemu      bare-metal Cortex-M0 image for QEMU's `microbit` machine
#   make pico      bare-metal Cortex-M0+ image for an RP2040, linked into SRAM
#
# `device` prefers arm-none-eabi-gcc and falls back to a clang cross build,
# which needs nothing installed on a machine that already has Xcode. Both
# produce real Thumb objects; `scripts/footprint.py` does the measuring and the
# asserting, because a footprint is a claim and claims belong where they can be
# checked rather than in a build log.
#
# The two bare-metal images share everything above the metal: `bare/` holds the
# harness, the semihosting calls and the freestanding runtime, and each machine
# directory supplies only a linker script, a startup file and the `dm_machine.h`
# backend. One harness is the point — claim 4's value is that the same differ
# compares the same line protocol whether the trace came from an emulator or
# from silicon, and two copies of it would drift.

CC ?= cc
CROSS_CC := $(shell command -v arm-none-eabi-gcc 2>/dev/null)
ifeq ($(CROSS_CC),)
  CROSS_CC := $(shell command -v clang 2>/dev/null)
  CROSS_TARGET := --target=thumbv6m-none-eabi
endif

# -Wconversion is load-bearing here rather than decorative: the interpreter
# moves between uint8_t operands, int32_t canvas units and int32_t fixed point,
# and an implicit narrowing anywhere in that chain is exactly how a port stops
# matching its specification.
WARNINGS := -Wall -Wextra -Wpedantic -Wconversion -Wsign-conversion -Wshadow \
            -Wstrict-prototypes -Wcast-qual -Wundef -Wvla
CFLAGS ?= -std=c11 -Os $(WARNINGS)
INCLUDE := -Iinclude

# No libc, no allocator, no floating point. -ffreestanding and the soft-float
# ABI are assertions as much as settings: if anything in the interpreter ever
# needs a double, this build pulls in soft-float support and the footprint
# report shows it rather than the fact going unnoticed.
DEVICE_FLAGS := $(CROSS_TARGET) -mcpu=cortex-m0plus -mthumb -mfloat-abi=soft \
                -ffreestanding -fno-builtin -fno-common \
                -fno-unwind-tables -fno-asynchronous-unwind-tables \
                -ffunction-sections -fdata-sections -fstack-usage

SRC := src/dm_vm.c src/dm_isa.c
HOST_SRC := $(SRC) host/dm_trace.c
HEADERS := include/dm_vm.h include/dm_isa.h
BARE := bare/dm_harness.c bare/runtime.c
BARE_HEADERS := bare/dm_machine.h bare/semihost.h
BUILD := build
DEVICE_OBJ := $(BUILD)/dm_vm.o $(BUILD)/dm_isa.o

.PHONY: all host host-san device qemu pico clean

all: host

host: $(BUILD)/dm_trace
host-san: $(BUILD)/dm_trace_san

$(BUILD)/dm_trace: $(HOST_SRC) $(HEADERS) | $(BUILD)
	$(CC) $(CFLAGS) $(INCLUDE) -o $@ $(HOST_SRC)

# The same C with UndefinedBehaviorSanitizer on. Conformance runs native code
# while the part runs Thumb, so what carries a host result over to a target that
# has never executed it is the absence of undefined behaviour -- signed overflow
# and out-of-range shifts are what would let two targets disagree while both
# look correct on one. -fno-sanitize-recover makes the first finding fatal, so a
# diagnostic cannot scroll past in a corpus of thousands. -O1 keeps the
# sanitiser's view of the code close to the source.
#
# AddressSanitizer is deliberately *not* in the default set: its runtime hangs
# at process start on this project's macOS/arm64 host, on a program as small as
# a single HALT, which is an environment fault rather than a finding. Ask for it
# where it works with `make host-san SAN=undefined,address`. The interpreter
# indexes only a caller-supplied buffer and a fixed frame array, so UBSan is the
# one carrying the argument here regardless.
SAN ?= undefined

$(BUILD)/dm_trace_san: $(HOST_SRC) $(HEADERS) | $(BUILD)
	$(CC) -std=c11 -O1 -g $(WARNINGS) -fsanitize=$(SAN) \
	    -fno-sanitize-recover=all -fno-omit-frame-pointer $(INCLUDE) -o $@ $(HOST_SRC)

device: $(DEVICE_OBJ)

# Compiled from $(BUILD) so the .su reports land beside the objects rather than
# in the working directory, which is the one thing gcc and clang agree to do
# only relative to the current one.
$(BUILD)/%.o: src/%.c $(HEADERS) | $(BUILD)
	@test -n "$(CROSS_CC)" || { echo "no cross compiler found"; exit 1; }
	cd $(BUILD) && $(CROSS_CC) $(CFLAGS) $(DEVICE_FLAGS) -I../include -c ../$< -o $(@F)

$(BUILD):
	@mkdir -p $(BUILD)

clean:
	@rm -rf $(BUILD)

# ---------------------------------------------------------------------------
# The two bare-metal images.
#
# ld.lld rather than a GNU toolchain: clang already cross-compiles thumbv6m, so
# a linker is the only missing piece and lld is 6 MB against a full
# arm-none-eabi install. Objects go into a per-machine subdirectory because the
# three cross builds compile the *same* file names with different flags, and a
# shared directory means `scripts/footprint.py` can measure whichever object
# happened to be built last.
BARE_CC := $(shell command -v clang 2>/dev/null)
BARE_LD := $(shell command -v ld.lld 2>/dev/null || echo /opt/homebrew/opt/lld/bin/ld.lld)
BARE_FLAGS := --target=thumbv6m-none-eabi -mthumb -mfloat-abi=soft \
              -ffreestanding -fno-builtin -fno-common -fno-unwind-tables \
              -fno-asynchronous-unwind-tables -Os -g $(WARNINGS)

# $(1) machine directory, $(2) extra compiler flags. One rule body, two
# machines, so a flag that matters can only be given to both or named for one.
define BARE_IMAGE
	@test -x "$(BARE_LD)" || { echo "ld.lld not found (brew install lld)"; exit 1; }
	@mkdir -p $(BUILD)/$(1)
	@for f in $(SRC) $(BARE) $(1)/machine.c $(1)/startup.c \
	          $(if $(filter pico,$(1)),$(1)/entry.S,); do \
	  $(BARE_CC) $(BARE_FLAGS) $(2) $(INCLUDE) -Ibare -I$(1) -c $$f \
	      -o $(BUILD)/$(1)/$$(basename $${f%.c}).o || exit 1; \
	done
	$(BARE_LD) -T $(1)/link.ld -o $@ $(BUILD)/$(1)/*.o
endef

# QEMU's `microbit`: an nRF51822, Cortex-M0, 16 KB SRAM. The point of this
# target is narrow and worth stating: the port is verified against
# `dm/vm/interp.py` by running *native* code, and until this existed the Thumb
# build had never executed at all. What carries a native result across is the
# absence of undefined behaviour, which is an argument; this is the measurement.
# It gives an instruction count and **not** a cycle count -- QEMU models neither
# flash wait states nor the prefetch buffer, and those are a vendor choice
# rather than a core property.
qemu: $(BUILD)/dm_qemu.elf

$(BUILD)/dm_qemu.elf: $(SRC) $(BARE) qemu/machine.c qemu/startup.c \
                      $(HEADERS) $(BARE_HEADERS) qemu/link.ld | $(BUILD)
	$(call BARE_IMAGE,qemu,-mcpu=cortex-m0)

# An RP2040, Cortex-M0+, linked entirely into SRAM -- see `pico/link.ld` for why
# there is no flash region. This is the target that produces the two numbers
# QEMU cannot: cycles per drawing, and the energy measurement that hangs off it.
#
# The output buffer is 32x the emulator's. Every semihosting write is a `bkpt`
# that QEMU traps as a function call; on the Pico the corresponding write is a
# UART transfer. 264 KB of SRAM can pay 32 KB to make that transfer 32x rarer,
# and 16 KB of nRF51 could not.
#
# PICO_EXTRA carries bring-up and diagnostic defines. A part with no debugger
# and a silent UART cannot say where it died, so `pico/machine.c` turns "the board
# rebooted to BOOTSEL" into a checkpoint and this is how a stage is selected.
# `DM_BSS_PREFILL=1` is the separate full-span startup diagnostic. It is empty by
# default and the measured image carries neither instrument.
pico: $(BUILD)/dm_pico.elf

PICO_EXTRA ?=

# 32768 is the measurement value: every write down a 115200-baud wire should
# carry as much as the part can spare, and 264 KB of SRAM can spare 32 KB.
#
# It is a variable because the buffer is also the thing that hides a hang. The
# harness flushes at the *end of a pass*, so a firmware that dies inside
# `trace_corpus` takes the whole trace to the grave and the host sees silence —
# indistinguishable from a dead UART. Building at 1 turns the buffer off, and
# then the last byte to arrive is the instruction before the fault.
PICO_OUT_MAX ?= 32768

$(BUILD)/dm_pico.elf: $(SRC) $(BARE) pico/machine.c pico/startup.c pico/entry.S \
                      $(HEADERS) $(BARE_HEADERS) pico/rp2040.h pico/link.ld | $(BUILD)
	$(call BARE_IMAGE,pico,-mcpu=cortex-m0plus -DDM_OUT_MAX=$(PICO_OUT_MAX) $(PICO_EXTRA))
