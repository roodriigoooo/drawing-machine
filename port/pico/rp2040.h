/* The RP2040 registers this port touches, and no others.
 *
 * Written out rather than pulled from the pico-sdk, for the same reason
 * `port/` has no libc: the thing under measurement is the interpreter, and an
 * SDK would put its own runtime, its own startup and its own clock policy
 * underneath it. `docs/claim4.md` already records one measurement distorted by
 * a naive `memset` in the harness's own runtime; a whole SDK is that failure
 * with more surface. Six peripherals, forty lines, and every offset carries the
 * datasheet section it came from.
 *
 * Section numbers are RP2040 datasheet, release 2.1.
 */

#ifndef DM_RP2040_H
#define DM_RP2040_H

#include <stdint.h>

#define DM_REG(address) (*(volatile uint32_t *)(uintptr_t)(address))

/* §2.1.2 — every peripheral is aliased three more times, so a field can be set
 * or cleared without a read-modify-write that an interrupt could sit inside. */
#define RP_SET 0x2000u
#define RP_CLR 0x3000u

/* §2.14.3 RESETS. Peripherals come out of reset one bit at a time; four of
 * them do anything here. */
#define RESETS_BASE 0x4000c000u
#define RESETS_RESET DM_REG(RESETS_BASE + 0x00u)
#define RESETS_RESET_CLR DM_REG(RESETS_BASE + RP_CLR + 0x00u)
#define RESETS_RESET_DONE DM_REG(RESETS_BASE + 0x08u)
#define RESETS_BIT_IO_BANK0 (1u << 5)
#define RESETS_BIT_PADS_BANK0 (1u << 8)
#define RESETS_BIT_TIMER (1u << 21)
#define RESETS_BIT_UART0 (1u << 22)

/* §2.16.7 XOSC — the 12 MHz crystal on a Pico board. */
#define XOSC_BASE 0x40024000u
#define XOSC_CTRL DM_REG(XOSC_BASE + 0x00u)
#define XOSC_CTRL_SET DM_REG(XOSC_BASE + RP_SET + 0x00u)
#define XOSC_STATUS DM_REG(XOSC_BASE + 0x04u)
#define XOSC_STARTUP DM_REG(XOSC_BASE + 0x0cu)
#define XOSC_CTRL_FREQ_RANGE_1_15MHZ 0xaa0u
#define XOSC_CTRL_ENABLE (0xfabu << 12)
#define XOSC_STATUS_STABLE (1u << 31)

/* §2.15.7 CLOCKS. Only the two glitchless muxes on the path to the core:
 * clk_ref selects the reference, clk_sys selects what the processor runs on.
 * `SELECTED` is one-hot over `SRC` and is the hardware's acknowledgement --
 * writing `CTRL` requests a switch, reading `SELECTED` observes it. */
#define CLOCKS_BASE 0x40008000u
#define CLK_REF_CTRL DM_REG(CLOCKS_BASE + 0x30u)
#define CLK_REF_DIV DM_REG(CLOCKS_BASE + 0x34u)
#define CLK_REF_SELECTED DM_REG(CLOCKS_BASE + 0x38u)
#define CLK_SYS_CTRL DM_REG(CLOCKS_BASE + 0x3cu)
#define CLK_SYS_DIV DM_REG(CLOCKS_BASE + 0x40u)
#define CLK_SYS_SELECTED DM_REG(CLOCKS_BASE + 0x44u)
#define CLK_REF_SRC_MASK 0x3u
#define CLK_SYS_SRC_MASK 0x1u
#define CLK_REF_SRC_XOSC 2u
#define CLK_SYS_SRC_REF 0u
#define CLK_DIV_1 (1u << 8) /* integer part is [31:8], fraction [7:0] */

/* clk_peri feeds the UART. It has no glitchless mux and no `SELECTED`: it is
 * simply gated, so enabling it with `AUXSRC` at 0 runs the peripheral off
 * clk_sys, which here is the crystal. That is why `DM_UART_DIV` below can be a
 * compile-time constant rather than something the firmware has to discover. */
#define CLK_PERI_CTRL DM_REG(CLOCKS_BASE + 0x48u)
#define CLK_PERI_CTRL_ENABLE (1u << 11)

/* §2.19.6 IO_BANK0 and §2.19.6.3 PADS_BANK0. Each GPIO has an 8-byte control
 * pair and a 4-byte pad register; only GPIO0 is used, as UART0 TX. */
#define IO_BANK0_BASE 0x40014000u
#define IO_BANK0_GPIO0_CTRL DM_REG(IO_BANK0_BASE + 0x004u)
#define IO_BANK0_FUNC_UART 2u

#define PADS_BANK0_BASE 0x4001c000u
#define PADS_BANK0_GPIO0 DM_REG(PADS_BANK0_BASE + 0x04u)
#define PADS_BANK0_IE (1u << 6)
#define PADS_BANK0_OD (1u << 7)

/* §4.2.8 UART0, an ARM PL011. Transmit only: there is no wire back. A Pico
 * GPIO is not 5 V tolerant, so accepting input from a 5 V USB-serial adapter
 * would need a level shifter, and the design here needs no input at all --
 * `scripts/uf2.py` puts the corpus in SRAM before the core starts. */
#define UART0_BASE 0x40034000u
#define UART0_DR DM_REG(UART0_BASE + 0x00u)
#define UART0_FR DM_REG(UART0_BASE + 0x18u)
#define UART0_IBRD DM_REG(UART0_BASE + 0x24u)
#define UART0_FBRD DM_REG(UART0_BASE + 0x28u)
#define UART0_LCR_H DM_REG(UART0_BASE + 0x2cu)
#define UART0_CR DM_REG(UART0_BASE + 0x30u)
#define UART0_FR_BUSY (1u << 3)
#define UART0_FR_TXFF (1u << 5)
#define UART0_LCR_H_8BIT (3u << 5)
#define UART0_LCR_H_FIFO (1u << 4)
#define UART0_CR_UARTEN (1u << 0)
#define UART0_CR_TXE (1u << 8)

/* §4.7.6 WATCHDOG. The tick generator is here rather than in TIMER: it divides
 * clk_ref down to the 1 MHz the timer counts. */
#define WATCHDOG_BASE 0x40058000u
#define WATCHDOG_TICK DM_REG(WATCHDOG_BASE + 0x2cu)
#define WATCHDOG_TICK_ENABLE (1u << 9)

/* §4.6.5 TIMER. The *latched* pair, not the RAW one: reading `TIMELR` latches
 * `TIMEHR` at the same instant, so the two halves cannot straddle a carry. The
 * RAW registers exist for a single-word read and using them for a 64-bit one is
 * how a timestamp ends up an hour out once every 71 minutes. */
#define TIMER_BASE 0x40054000u
#define TIMER_HR DM_REG(TIMER_BASE + 0x08u)
#define TIMER_LR DM_REG(TIMER_BASE + 0x0cu)

/* ARMv6-M B3.3 — SysTick, and the vector table offset register.
 *
 * SysTick is the *only* cycle counter this part has: the Cortex-M0+ on an
 * RP2040 does not implement the DWT unit, so there is no `DWT_CYCCNT` and a
 * 24-bit down-counter is the instrument. That is why `dm_cycles_stop` has to
 * report a wrap rather than return a difference. */
#define SYST_CSR DM_REG(0xe000e010u)
#define SYST_RVR DM_REG(0xe000e014u)
#define SYST_CVR DM_REG(0xe000e018u)
#define SYST_CSR_ENABLE (1u << 0)
#define SYST_CSR_CLKSOURCE (1u << 2) /* 0 = watchdog tick, 1 = processor clock */
#define SYST_CSR_COUNTFLAG (1u << 16)
#define SYST_MASK 0x00ffffffu

#define SCB_VTOR DM_REG(0xe000ed08u)

#endif /* DM_RP2040_H */
