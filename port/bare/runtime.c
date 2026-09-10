/* The three things a freestanding C program still owes the compiler.
 *
 * `-ffreestanding` stops the compiler *assuming* a library; it does not stop it
 * emitting calls into one. Zeroing a struct becomes `__aeabi_memclr4`,
 * assigning one becomes `__aeabi_memcpy`, and on ARMv6-M -- which has `MULS`
 * but **no divide instruction** -- any `/` becomes `__aeabi_uidiv`. All three
 * turned up as link errors the first time the port was built for the target,
 * which is itself worth recording: they are invisible in the source.
 *
 * `dm_vm.c` needs only the first. It contains no division at all, by design,
 * because the fixed-point scheme in `dm_vm.h` was chosen so curve flattening is
 * a multiply-add and never a divide -- so the division helper below exists for
 * the *harness's* decimal formatting and for nothing on the measured path.
 */

#include <stddef.h>
#include <stdint.h>

/* Word-at-a-time once aligned. Not premature: `dm_vm_run` zeroes its ~150-byte
 * state on entry, so a byte loop here put ~600 instructions in front of every
 * program and would have inflated the measured cost of the interpreter by
 * ~14% on a 160-byte drawing. A measurement taken through a deliberately naive
 * runtime is a measurement of the runtime. */
void *memset(void *dst, int value, size_t n)
{
    uint8_t *p = dst;
    const uint8_t byte = (uint8_t)value;

    while (n && ((uintptr_t)p & 3u)) {
        *p++ = byte;
        n--;
    }
    const uint32_t word = 0x01010101u * byte;
    for (; n >= 4u; n -= 4u, p += 4) {
        *(uint32_t *)(void *)p = word;
    }
    while (n--) {
        *p++ = byte;
    }
    return dst;
}

void *memcpy(void *restrict dst, const void *restrict src, size_t n)
{
    uint8_t *d = dst;
    const uint8_t *s = src;

    if ((((uintptr_t)d | (uintptr_t)s) & 3u) == 0u) {
        for (; n >= 4u; n -= 4u, d += 4, s += 4) {
            *(uint32_t *)(void *)d = *(const uint32_t *)(const void *)s;
        }
    }
    while (n--) {
        *d++ = *s++;
    }
    return dst;
}

/* The AEABI spellings. Argument order differs from the C ones for memclr/memset
 * (`__aeabi_memset` takes the value last), which is exactly the sort of detail
 * that produces a silent wrong answer rather than a link error. */
void __aeabi_memclr(void *dst, size_t n) { (void)memset(dst, 0, n); }
void __aeabi_memclr4(void *dst, size_t n) { (void)memset(dst, 0, n); }
void __aeabi_memclr8(void *dst, size_t n) { (void)memset(dst, 0, n); }
void __aeabi_memset(void *dst, size_t n, int value) { (void)memset(dst, value, n); }
void __aeabi_memset4(void *dst, size_t n, int value) { (void)memset(dst, value, n); }
void __aeabi_memcpy(void *dst, const void *src, size_t n) { (void)memcpy(dst, src, n); }
void __aeabi_memcpy4(void *dst, const void *src, size_t n) { (void)memcpy(dst, src, n); }
void __aeabi_memcpy8(void *dst, const void *src, size_t n) { (void)memcpy(dst, src, n); }

/* Restoring-division, shift-subtract, 32 iterations. Not fast and not on any
 * measured path; correctness and legibility are the whole specification. */
unsigned __aeabi_uidiv(unsigned numerator, unsigned denominator)
{
    if (denominator == 0) {
        return 0; /* undefined in C; a defined answer beats a fault here */
    }
    unsigned quotient = 0, remainder = 0;
    for (int bit = 31; bit >= 0; bit--) {
        remainder = (remainder << 1) | ((numerator >> bit) & 1u);
        if (remainder >= denominator) {
            remainder -= denominator;
            quotient |= 1u << bit;
        }
    }
    return quotient;
}

/* `__aeabi_uidivmod` returns the quotient in r0 and the remainder in r1, which
 * C cannot express, so the register pair is built by hand. */
unsigned long long __aeabi_uidivmod(unsigned numerator, unsigned denominator)
{
    const unsigned quotient = __aeabi_uidiv(numerator, denominator);
    const unsigned remainder = numerator - quotient * denominator;
    return ((unsigned long long)remainder << 32) | quotient;
}
