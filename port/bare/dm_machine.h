/* Everything `dm_harness.c` needs from the part it is running on, and nothing
 * else.
 *
 * The harness exists so that one set of bytes -- one line protocol, one framing,
 * one formatter -- is what `scripts/conformance.py` diffs against
 * `dm/vm/interp.py`, whether those bytes came from an emulated Cortex-M0 or from
 * silicon. Claim 4's whole value is that the *same* differ compares the *same*
 * protocol; two harnesses that drifted apart would compare two things and say
 * they agreed.
 *
 * So the machine-specific part is pushed behind this interface and kept as small
 * as it can be: bring-up, a clock, a cycle counter, and an independent timebase
 * to check the cycle counter against. An emulator implements the first and
 * stubs the rest, which is honest -- QEMU has no cycles to report and
 * `docs/claim4.md` says at length why a guessed one would be worse than none.
 */

#ifndef DM_MACHINE_H
#define DM_MACHINE_H

#include <stdint.h>

/* How the machine names itself in the cycles protocol. Not decoration: a
 * cycle count is part-specific by construction (flash wait states and the
 * prefetch buffer are a vendor choice, not a core property), so a number that
 * does not carry its part is not a measurement. */
extern const char *const dm_machine_name;

/* Called once before any I/O. Clocks, resets, timebases. */
void dm_machine_init(void);

/* Core clock in Hz, or 0 when the machine does not have a defined one. */
uint32_t dm_clk_hz(void);

/* Non-zero when `dm_cycles_start`/`dm_cycles_stop` measure real core cycles. */
int dm_cycles_available(void);

/* Non-zero only when the machine has directly verified the clock source used
 * by its cycle counter. A ratio of two counters is not enough when both can
 * follow the same wrong source. */
int dm_clock_source_valid(void);

void dm_cycles_start(void);

/* Core cycles since `dm_cycles_start`, or `DM_CYCLES_OVERFLOW` if the counter
 * wrapped. A wrapped 24-bit count is indistinguishable from a small one, so it
 * is reported as a distinct value rather than returned as a measurement. */
#define DM_CYCLES_OVERFLOW 0xFFFFFFFFu
uint32_t dm_cycles_stop(void);

/* Microseconds since bring-up, from a second timebase rather than the cycle
 * counter, or 0 when there is none. The source of both counters is asserted
 * separately by dm_clock_source_valid(); this timebase is a consistency check,
 * not proof of the selected clock. */
uint64_t dm_us_now(void);

#endif /* DM_MACHINE_H */
