/* RP2040 RAM-UF2 entry contract.
 *
 * The boot ROM enters a RAM-only UF2 at its lowest downloaded address. It does
 * not interpret that address as a Cortex reset vector. This trampoline is the
 * first downloaded code and performs the small part of reset entry that the
 * ROM intentionally does not: mask interrupts, install our vector table, load
 * its initial MSP, and branch through its reset vector.
 */

.syntax unified
.cpu cortex-m0plus
.thumb

.section .text.entry, "ax", %progbits
.align 2
.global ram_entry
.type ram_entry, %function
.thumb_func
ram_entry:
    cpsid i
    ldr r0, .Lvector_table
    ldr r1, .Lvtor
    str r0, [r1]
    dsb sy
    isb sy
    ldr r1, [r0, #0]
    msr msp, r1
    ldr r1, [r0, #4]
    bx r1

.align 2
.Lvector_table:
    .word vector_table
.Lvtor:
    .word 0xe000ed08

.size ram_entry, .-ram_entry
