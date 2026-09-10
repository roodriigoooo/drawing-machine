/* Memory map for QEMU's `microbit` machine: an nRF51822, Cortex-M0,
 * 256 KB flash at 0 and 16 KB SRAM at 0x20000000.
 *
 * The part is chosen for being *small*. A generous machine would run the
 * harness without proving anything about the footprint claim; 16 KB of SRAM
 * means the streaming sink in `dm_vm.h` is load-bearing here rather than
 * theoretical -- a buffering port would not fit its own worst case.
 */

MEMORY
{
  FLASH (rx)  : ORIGIN = 0x00000000, LENGTH = 256K
  RAM   (rwx) : ORIGIN = 0x20000000, LENGTH = 16K
}

ENTRY(reset_handler)

SECTIONS
{
  .text :
  {
    KEEP(*(.isr_vector))   /* must be at address 0: the core fetches SP and PC from it */
    *(.text*)
    *(.rodata*)
    . = ALIGN(4);
  } > FLASH

  _sidata = LOADADDR(.data);

  .data :
  {
    . = ALIGN(4);
    _sdata = .;
    *(.data*)
    . = ALIGN(4);
    _edata = .;
  } > RAM AT > FLASH

  .bss :
  {
    . = ALIGN(4);
    _sbss = .;
    *(.bss*)
    *(COMMON)
    . = ALIGN(4);
    _ebss = .;
  } > RAM

  /* Whatever RAM is left is stack. `_estack` is the initial SP, and the
   * assertion below is what turns "it fits" into something the build checks:
   * the interpreter needs 288 bytes and the harness's buffers are static. */
  . = ALIGN(8);
  _heap_start = .;
  _estack = ORIGIN(RAM) + LENGTH(RAM);

  ASSERT(_estack - _heap_start > 2K, "less than 2 KB of stack left")

  /DISCARD/ : { *(.ARM.exidx*) *(.comment) }
}
