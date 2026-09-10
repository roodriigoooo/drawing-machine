/* Reset vector and C runtime bring-up for a bare Cortex-M0.
 *
 * There is no bootloader and no libc, so this is everything between power-on
 * and `main`: the vector table the core fetches at address 0, a reset handler
 * that establishes the two things C guarantees and nothing else does -- `.data`
 * initialised and `.bss` zeroed -- and a fault handler that says so rather than
 * hanging.
 *
 * Keeping it this small is the point. The claim being tested is that the
 * interpreter behaves identically as Thumb, so anything else running on the
 * part is a confound; there is no clock setup, no peripheral init, no heap.
 */

#include <stdint.h>

#include "semihost.h"

extern uint32_t _sidata, _sdata, _edata, _sbss, _ebss, _estack;

int main(void);

/* Not static: the linker script's ENTRY() names it, and a Cortex-M actually
 * boots from the vector table below rather than from the ELF entry point --
 * ENTRY is there so a debugger and `objdump` agree with the hardware. */
void reset_handler(void)
{
    /* `.data` lives in flash and runs from RAM. Copying it is the compiler's
     * assumption, not an optimisation. */
    for (uint32_t *src = &_sidata, *dst = &_sdata; dst < &_edata;) {
        *dst++ = *src++;
    }
    for (uint32_t *dst = &_sbss; dst < &_ebss;) {
        *dst++ = 0;
    }
    sh_exit(main());
}

/* Any fault here is a bug in the port or the linker script, and a silent
 * spin would look exactly like a slow run. Exit loudly instead. */
static void fault_handler(void)
{
    static const char message[] = "FAULT: hard fault or unhandled exception\n";
    (void)sh_write(2, message, sizeof message - 1);
    sh_exit(70);
}

/* Cortex-M0 fetches the initial stack pointer from offset 0 and the reset
 * vector from offset 4, so this must be the first thing in flash -- which is
 * what the linker script's `.isr_vector` placement guarantees. */
__attribute__((section(".isr_vector"), used))
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
