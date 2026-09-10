/* Memory map for an RP2040 image that never touches flash.
 *
 * ## Why there is no FLASH region here
 *
 * The number this build exists to produce is **cycles per drawing on a
 * Cortex-M0+**, and `docs/claim4.md` argues at length that such a number is
 * part-specific by construction: instruction timing on the core is
 * deterministic — no cache, no branch prediction, a single-cycle multiplier —
 * but flash wait states and the prefetch buffer are a vendor choice, so the
 * same binary on two M0+ parts at one clock does not take the same number of
 * cycles.
 *
 * On an RP2040 that choice is a QSPI part behind an XIP cache, and it is
 * configurable. Linking into SRAM removes it from the measurement entirely
 * rather than characterising it: every instruction fetch is a single-cycle
 * SRAM access, so what comes out is the core's own cost and is comparable to
 * any other M0+ running from zero-wait-state memory. An XIP-resident arm is a
 * separate and genuinely interesting measurement — the delta between the two
 * *is* the flash cost — and it needs a second-stage bootloader this build
 * deliberately does not have.
 *
 * ## The map
 *
 * SRAM0–SRAM3 are 256 KB at 0x20000000, striped by word across four banks so a
 * single core sees no contention. SRAM4 and SRAM5 are two further 4 KB banks at
 * 0x20040000 that are *not* striped; they are left out because nothing here
 * needs them and an image that fits the striped region has one uniform
 * timing behaviour, which a measurement wants and a general-purpose map does
 * not care about.
 *
 * ## Why the corpus has a region, and why it sits *before* `.bss`
 *
 * There is no debugger on this path and therefore no channel into the running
 * part, so the corpus travels in the same UF2 as the image and the bootrom's
 * drag-and-drop loader writes both into SRAM before the core starts.
 *
 * Its placement is not cosmetic. A UF2 written into RAM should describe one
 * contiguous span -- the bootrom is loading an image, not patching memory --
 * and `.bss` occupies ~37 KB that is *not* in the file, because it is zeroed at
 * run time. Putting the corpus after `.bss` would leave a 37 KB hole in the
 * middle of the download. So the reserved region goes between `.data` and
 * `.bss`: everything the UF2 carries is one run of addresses, and everything it
 * does not carry is after it.
 *
 * `_corpus_base` is read out of the ELF's symbol table by `scripts/uf2.py`
 * rather than duplicated there as a constant. The linker decides where the
 * region is; the packer asks.
 *
 * Nothing is written to the board's flash at any point, by this or by
 * `scripts/pico.py`. A power cycle restores the part exactly as it shipped.
 */

MEMORY
{
  RAM (rwx) : ORIGIN = 0x20000000, LENGTH = 256K
}

/* 180 KB, which is ~1,600 QuickDraw programs at the corpus's 112-byte mean and
 * every synthetic corpus this project has whole. Reserved rather than sized to
 * fit, because the image must be one artifact across every run: a linker script
 * that changed with the corpus would move `_corpus_base` and make two runs
 * incomparable for a reason nobody would look for. */
DM_CORPUS_RESERVE = 184320;

/* A RAM UF2 is entered at the lowest downloaded address, not by fetching the
 * first two words as a Cortex reset vector. The entry trampoline therefore
 * owns address 0x20000000 and vectors live at the first 256-byte-aligned
 * address after it. */
ENTRY(ram_entry)

SECTIONS
{
  .text :
  {
    /* RP2040 RAM-image boot enters the lowest downloaded address directly. Keep
     * executable entry code there; putting `.isr_vector` first makes its SP/PC
     * words execute as Thumb instructions and skips reset_handler entirely. */
    KEEP(*(.text.entry))
    . = ALIGN(4);
    ASSERT(. <= ORIGIN(RAM) + 0x100,
           "RAM entry trampoline does not fit before the vector table")
    . = ORIGIN(RAM) + 0x100;
    /* VTOR uses bits [31:8], so this table is deliberately 256-byte aligned. */
    KEEP(*(.isr_vector))
    *(.text*)
    *(.rodata*)
    . = ALIGN(4);
  } > RAM

  /* Load address equals run address here, so the copy in `startup.c` is a
   * no-op. Kept as a real symbol rather than elided, because it is what an
   * XIP-resident link would change and one that silently lacked it would fail
   * in the least legible way available. */
  _sidata = LOADADDR(.data);

  .data :
  {
    . = ALIGN(4);
    _sdata = .;
    *(.data*)
    . = ALIGN(4);
    _edata = .;
  } > RAM

  /* NOLOAD: the ELF carries no corpus, only the address one will be written to.
   * 256-aligned so the region starts on a UF2 block boundary and the packer
   * never has to split a block between the image and the corpus. */
  .corpus (NOLOAD) :
  {
    . = ALIGN(256);
    _corpus_base = .;
    . = . + DM_CORPUS_RESERVE;
    _corpus_end = .;
  } > RAM

  .bss :
  {
    . = ALIGN(4);
    _sbss = .;
    *(.bss*)
    *(COMMON)
    . = ALIGN(4);
    _ebss = .;
  } > RAM

  /* The corpus is never zeroed, and this is what keeps that true. `startup.c`
   * clears `_sbss`..`_ebss`; if the region ever moved to overlap that span,
   * every run would execute a corpus of zeros -- which is a corpus of valid
   * `MOVE 0,0` instructions, so it would trace, and diff, and be wrong. */
  ASSERT(_sbss >= _corpus_end, "the .bss clear would erase the corpus")

  /* Whatever is left is stack. The interpreter's worst case is 492 bytes and
   * the harness's buffers are static, so this is enormous headroom — the
   * assertion is here to fail loudly if a future output buffer ever eats the
   * region rather than to defend a tight margin. */
  . = ALIGN(8);
  _heap_start = .;
  _estack = ORIGIN(RAM) + LENGTH(RAM);

  ASSERT(_estack - _heap_start > 8K, "less than 8 KB of stack left")

  /DISCARD/ : { *(.ARM.exidx*) *(.comment) }
}
