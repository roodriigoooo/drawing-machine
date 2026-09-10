/* C runtime bring-up for an RP2040 image that lives entirely
 * in SRAM.
 *
 * `entry.S` is the physical RAM-UF2 entry point. It disables interrupts,
 * installs this table in VTOR, loads its initial MSP, and branches through the
 * reset vector. This file then performs the ordinary C runtime work: `.data`
 * copy, `.bss` clear, and the call to `main`.
 *
 * Everything else is deliberately absent: no clock setup (that is
 * `machine.c`'s, and it is called from `main` so the harness owns the order),
 * no peripheral init, no heap, no second core. The claim under test is that
 * the interpreter behaves identically on this part; anything else executing is
 * a confound.
 */

#include <stdint.h>

extern uint32_t _sidata, _sdata, _edata, _sbss, _ebss, _estack;

int main(void);

/* `machine.c` owns the UART, so it owns what a fault is able to say. */
void dm_pico_fault(const uint32_t *frame);

void reset_handler(void);
void fault_handler(void);

/* The table is in the VTOR-aligned slot at 0x20000100. The RAM entry
 * trampoline, rather than the boot ROM, consumes its first two words and
 * establishes the reset contract before entering C. */
__attribute__((section(".isr_vector"), used, aligned(256)))
void (*const vector_table[])(void) = {
    (void (*)(void)) & _estack,
    reset_handler,
    fault_handler, /* NMI */
    fault_handler, /* HardFault */
    0, 0, 0, 0, 0, 0, 0,
    fault_handler, /* SVCall */
    0, 0,
    fault_handler, /* PendSV */
    fault_handler, /* SysTick */
};

void reset_handler(void)
{
    /* A no-op in this link, where `.data` is loaded at the address it runs at,
     * and kept because it is the loop that stops being a no-op the moment
     * anything is linked to run from XIP flash — which is the second arm this
     * measurement wants. Costing nothing and being correct beats being absent
     * and being noticed later. */
    for (uint32_t *src = &_sidata, *dst = &_sdata; dst < &_edata;) {
        *dst++ = *src++;
    }
    /* Diagnostic builds deliberately dirty the complete BSS span before the
     * real clear. This makes a missing or partial clear observable on silicon,
     * rather than relying on whatever pattern a preceding RAM image left. */
#ifdef DM_BSS_PREFILL
    for (uint32_t *dst = &_sbss; dst < &_ebss;) {
        *dst++ = 0xa5a5a5a5u;
    }
#endif
    for (uint32_t *dst = &_sbss; dst < &_ebss;) {
        *dst++ = 0;
    }
    (void)main();
    /* `dm_io_end` never returns, so this is unreachable — and it is a spin
     * rather than a semihosted exit because there is no debugger to take one.
     * A `BKPT` here would escalate to HardFault and lock the part up silently. */
    for (;;) {
    }
}

/* Any fault here is a bug in the port or the linker script, and on a part with
 * no debugger a silent spin is indistinguishable from a slow run — the worst
 * failure available on something being timed.
 *
 * `naked`, because the frame the exception pushed is what we want to read and a
 * compiler prologue would move `sp` off it first. ARMv6-M pushes
 * {R0,R1,R2,R3,R12,LR,PC,xPSR}; `machine.c` prints the PC out of it. `bl` rather
 * than `b`, because ARMv6-M has no wide unconditional branch and the handler
 * never returns, so clobbering LR costs nothing.
 */
__attribute__((naked)) void fault_handler(void)
{
    __asm__ volatile("mrs r0, msp\n\t"
                     "bl  dm_pico_fault\n\t");
}
