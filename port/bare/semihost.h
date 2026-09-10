/* ARM semihosting: the debugger-side syscalls, which is all the I/O a
 * bare-metal conformance run needs.
 *
 * The harness has no libc, no filesystem and no UART. Semihosting gives it the
 * host's, through a `bkpt 0xAB` the emulator traps: the guest puts the
 * operation in r0 and a pointer to its arguments in r1, and QEMU performs it.
 * That is enough to read the corpus and write the trace, so the *same*
 * comparison `scripts/conformance.py` already runs natively can run against
 * code that has actually executed as Thumb.
 *
 * Deliberately not a libc shim. Four operations, each a direct call, so nothing
 * between `dm_vm_run` and the emulator can buffer, translate or reorder --
 * which matters because the thing under test is whether the *interpreter*
 * behaves identically, not whether a printf does.
 */

#ifndef DM_SEMIHOST_H
#define DM_SEMIHOST_H

#include <stdint.h>

#define SYS_OPEN 0x01
#define SYS_CLOSE 0x02
#define SYS_WRITE 0x05
#define SYS_READ 0x06
#define SYS_EXIT 0x18
/* 32-bit ARM's SYS_EXIT takes the *reason* in r1, not a pointer to a block, so
 * it cannot carry an exit status. SYS_EXIT_EXTENDED takes the two-word block
 * and can. Passing a pointer to plain SYS_EXIT makes the emulator read the
 * pointer value as the reason, fail to recognise it, and exit non-zero on a
 * successful run -- which is exactly how this was found. */
#define SYS_EXIT_EXTENDED 0x20
#define ADP_STOPPED_APPLICATION_EXIT 0x20026u

/* Semihosting mode numbers for `SYS_OPEN`. 0 is "r", 4 is "rb", 6 is "wb". */
#define SH_MODE_RB 1
#define SH_MODE_WB 6

static inline int32_t semihost(uint32_t op, void *args)
{
    register uint32_t r0 __asm__("r0") = op;
    register void *r1 __asm__("r1") = args;

    /* The AAPCS clobber list matters: QEMU may modify r0/r1, and the compiler
     * must not assume anything survives the breakpoint. */
    __asm__ volatile("bkpt 0xAB" : "+r"(r0) : "r"(r1) : "memory");
    return (int32_t)r0;
}

static inline int32_t sh_open(const char *name, uint32_t length, uint32_t mode)
{
    uint32_t args[3] = {(uint32_t)name, mode, length};
    return semihost(SYS_OPEN, args);
}

static inline int32_t sh_read(int32_t handle, void *buffer, uint32_t length)
{
    /* Returns bytes *not* read, which is the semihosting convention and the
     * opposite of read(2). Callers here compare against `length`. */
    uint32_t args[3] = {(uint32_t)handle, (uint32_t)buffer, length};
    return semihost(SYS_READ, args);
}

static inline int32_t sh_write(int32_t handle, const void *buffer, uint32_t length)
{
    uint32_t args[3] = {(uint32_t)handle, (uint32_t)buffer, length};
    return semihost(SYS_WRITE, args);
}

static inline void sh_close(int32_t handle)
{
    uint32_t args[1] = {(uint32_t)handle};
    (void)semihost(SYS_CLOSE, args);
}

static inline void sh_exit(int code)
{
    /* The extended form, so the emulator's own return code carries the
     * harness's verdict rather than only "it stopped". */
    uint32_t args[2] = {ADP_STOPPED_APPLICATION_EXIT, (uint32_t)code};
    (void)semihost(SYS_EXIT_EXTENDED, args);
    /* Older monitors ignore the extended call; fall back to the plain one,
     * which reports the reason and loses the status. */
    (void)semihost(SYS_EXIT, (void *)(uintptr_t)ADP_STOPPED_APPLICATION_EXIT);
    for (;;) {
    }
}

#endif /* DM_SEMIHOST_H */
