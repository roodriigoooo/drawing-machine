/* How a bare-metal machine gets its corpus in and its trace out.
 *
 * Split out from `dm_machine.h` because the two axes are genuinely independent
 * and the project has one machine that moves on each: QEMU has file I/O through
 * a debugger and no clock worth reporting; an RP2040 with no debugger attached
 * has a real cycle counter and no filesystem at all.
 *
 * ## The two implementations, and why they look nothing alike
 *
 * **Under QEMU** the corpus arrives over ARM semihosting -- the emulator opens
 * `frames.bin` on the host and the firmware reads it a program at a time. That
 * needs a debugger on the other end of the wire, which is exactly what a bare
 * Pico with one USB cable does not have.
 *
 * **On the Pico** there is no channel inward at all, so the corpus is *already
 * in SRAM*: `scripts/uf2.py` packs it into the same UF2 as the image, at a fixed
 * address the linker script reserves, behind a header carrying the run's
 * configuration. The bootrom's drag-and-drop loader writes both, and the
 * firmware finds its corpus by looking at an address. Output leaves over a UART
 * pin at 3.3 V, which any USB-serial adapter can read.
 *
 * The consequence for the harness is one function: `dm_io_again`. A semihosted
 * run exits when the corpus ends, because the host was there for all of it. A
 * Pico run has no idea when the host attached, so it repeats forever and the
 * host takes whichever `BEGIN`..`END` block it catches whole. That is also why
 * the harness brackets its output -- the framing is what makes a mid-stream
 * attach recoverable rather than a truncated trace that parses.
 */

#ifndef DM_IO_H
#define DM_IO_H

#include <stdint.h>

/* Bring up the channel and locate the corpus. May not return, if there is no
 * corpus to find. */
void dm_io_begin(void);

/* The run's configuration, wherever this machine keeps it. */
uint32_t dm_config_fuel(void);
uint32_t dm_config_reps(void);  /* 0 selects trace mode, non-zero cycle mode */
int dm_config_quiet(void);

/* Which batch of a split sweep this corpus is, echoed in the `BEGIN` line.
 *
 * A machine whose host is present for the whole run has one batch and reports
 * zero. The offline flow does not: the corpus is larger than the part's SRAM,
 * so it is dropped on the board one file at a time by hand, and a capture may
 * end up holding them out of order, twice, or with a partial retry in the
 * middle. Matching blocks by position would then silently diff batch 3 against
 * batch 4's reference. An identity the device echoes makes a re-drag free. */
uint32_t dm_config_batch(void);

#define DM_CORPUS_END 0xFFFFFFFFu

void dm_corpus_rewind(void);

/* The next program into `dst`, its length, or `DM_CORPUS_END` at the end of the
 * corpus. A program longer than `max` is a fault rather than a truncation --
 * silently running a prefix of a program would produce a trace that differs
 * from the reference for a reason no diff could name. */
uint32_t dm_corpus_next(uint8_t *dst, uint32_t max);

void dm_io_write(const char *data, uint32_t length);

/* Non-zero when the corpus should be run again from the start. */
int dm_io_again(void);

/* Flush and stop. Never returns. */
void dm_io_end(int status);

#endif /* DM_IO_H */
