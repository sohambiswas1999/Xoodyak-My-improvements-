#include <stdio.h>
#include <string.h>
#include <stddef.h>
#include <stdint.h>
#include <smmintrin.h>
#include <wmmintrin.h>
#include <immintrin.h>
#include <emmintrin.h>
#include "brg_endian.h"

#define ALIGN(x) __declspec(align(x))

#define MAXROUNDS 12
#define NROWS 3
#define NCOLUMS 4
#define NLANES (NCOLUMS * NROWS)

/*    Round constants    */
#define _rc12 0x00000058
#define _rc11 0x00000038
#define _rc10 0x000003C0
#define _rc9 0x000000D0
#define _rc8 0x00000120
#define _rc7 0x00000014
#define _rc6 0x00000060
#define _rc5 0x0000002C
#define _rc4 0x00000380
#define _rc3 0x000000F0
#define _rc2 0x000001A0
#define _rc1 0x00000012

#if (VERBOSE > 0)
#define Dump(__t)                                                                                               \
    Vars2State;                                                                                                 \
    printf(__t "\n");                                                                                           \
    printf("a00 %08x, a01 %08x, a02 %08x, a03 %08x\n", state[0 + 0], state[0 + 1], state[0 + 2], state[0 + 3]); \
    printf("a10 %08x, a11 %08x, a12 %08x, a13 %08x\n", state[4 + 0], state[4 + 1], state[4 + 2], state[4 + 3]); \
    printf("a20 %08x, a21 %08x, a22 %08x, a23 %08x\n\n", state[8 + 0], state[8 + 1], state[8 + 2], state[8 + 3]);

#define DumpLanes(__t, l0, l1, l2)                                         \
    {                                                                      \
        uint32_t buf[4];                                                   \
        printf(__t "\n");                                                  \
        STORE128u(buf[0], l0);                                             \
        printf("%08x %08x %08x %08x\n", buf[0], buf[1], buf[2], buf[3]);   \
        STORE128u(buf[0], l1);                                             \
        printf("%08x %08x %08x %08x\n", buf[0], buf[1], buf[2], buf[3]);   \
        STORE128u(buf[0], l2);                                             \
        printf("%08x %08x %08x %08x\n\n", buf[0], buf[1], buf[2], buf[3]); \
    }
#else
#define Dump(__t)
#define DumpLanes(__t, l0, l1, l2)
#endif

#if (VERBOSE >= 1)
#define Dump1(__t) Dump(__t)
#else
#define Dump1(__t)
#endif

#if (VERBOSE >= 2)
#define Dump2(__t) Dump(__t)
#else
#define Dump2(__t)
#endif

#if (VERBOSE >= 3)
#define Dump3(__t) Dump(__t)
#else
#define Dump3(__t)
#endif

typedef struct
{
    ALIGN(16)
    uint32_t A[12];
} Xoodoo_align128plain32_state;

typedef Xoodoo_align128plain32_state Xoodoo_state;

typedef __m128i V128;
typedef __m256i V256;
typedef __m512i V512;

void Xoodoo_Initialize(Xoodoo_align128plain32_state *state)
{
    memset(state, 0, sizeof(Xoodoo_align128plain32_state));
}

/* ---------------------------------------------------------------- */

void Xoodoo_AddBytes(Xoodoo_align128plain32_state *argState, const unsigned char *data, unsigned int offset, unsigned int length)
{
#if (PLATFORM_BYTE_ORDER == IS_LITTLE_ENDIAN)
    unsigned int sizeLeft = length;
    unsigned int lanePosition = offset / 4; // offset sinifies how much of the state is filled in terms of bytes
    unsigned int offsetInLane = offset % 4; // offset % 4 is the ammount of data in last lane in  terms of bytes
    const unsigned char *curData = data;
    uint32_t *state = argState->A;

    state += lanePosition;
    if ((sizeLeft > 0) && (offsetInLane != 0))
    {
        unsigned int bytesInLane = 4 - offsetInLane; // space available in lane
        uint32_t lane = 0;                           // lane is initialised at 0 so during xor only input data gets added
        if (bytesInLane > sizeLeft)
            bytesInLane = sizeLeft;
        memcpy((unsigned char *)&lane + offsetInLane, curData, bytesInLane);
        *state++ ^= lane;
        sizeLeft -= bytesInLane;
        curData += bytesInLane;
    }

    while (sizeLeft >= 4)
    {
        *state++ ^= READ32_UNALIGNED(curData);
        sizeLeft -= 4;
        curData += 4;
    }

    if (sizeLeft > 0)
    {
        uint32_t lane = 0;
        memcpy(&lane, curData, sizeLeft);
        *state ^= lane;
    }
#else
#error "Not yet implemented"
#endif
}

/* ---------------------------------------------------------------- */

void Xoodoo_OverwriteBytes(Xoodoo_align128plain32_state *state, const unsigned char *data, unsigned int offset, unsigned int length)
{
#if (PLATFORM_BYTE_ORDER == IS_LITTLE_ENDIAN)
    memcpy((unsigned char *)state + offset, data, length);
#else
#error "Not yet implemented"
#endif
}

/* ---------------------------------------------------------------- */

void Xoodoo_OverwriteWithZeroes(Xoodoo_align128plain32_state *state, unsigned int byteCount)
{
#if (PLATFORM_BYTE_ORDER == IS_LITTLE_ENDIAN)
    memset(state, 0, byteCount);
#else
#error "Not yet implemented"
#endif
}

/* ---------------------------------------------------------------- */

void Xoodoo_ExtractBytes(const Xoodoo_align128plain32_state *state, unsigned char *data, unsigned int offset, unsigned int length)
{
#if (PLATFORM_BYTE_ORDER == IS_LITTLE_ENDIAN)
    memcpy(data, (const unsigned char *)state + offset, length);
#else
#error "Not yet implemented"
#endif
}

/* ---------------------------------------------------------------- */

void Xoodoo_ExtractAndAddBytes(const Xoodoo_align128plain32_state *argState, const unsigned char *input, unsigned char *output, unsigned int offset, unsigned int length)
{
#if (PLATFORM_BYTE_ORDER == IS_LITTLE_ENDIAN)
    unsigned int sizeLeft = length;
    unsigned int lanePosition = offset / 4;
    unsigned int offsetInLane = offset % 4;
    const unsigned char *curInput = input;
    unsigned char *curOutput = output;
    const uint32_t *state = argState->A;

    state += lanePosition;
    if ((sizeLeft > 0) && (offsetInLane != 0))
    {
        unsigned int bytesInLane = 4 - offsetInLane;
        uint32_t lane = *state++ >> (offsetInLane * 8);
        if (bytesInLane > sizeLeft)
            bytesInLane = sizeLeft;
        sizeLeft -= bytesInLane;
        do
        {
            *curOutput++ = (*curInput++) ^ (unsigned char)lane;
            lane >>= 8;
        } while (--bytesInLane != 0);
    }

    while (sizeLeft >= 4)
    {
        WRITE32_UNALIGNED(curOutput, READ32_UNALIGNED(curInput) ^ *state++);
        sizeLeft -= 4;
        curInput += 4;
        curOutput += 4;
    }

    if (sizeLeft > 0)
    {
        uint32_t lane = *state;
        do
        {
            *curOutput++ = (*curInput++) ^ (unsigned char)lane;
            lane >>= 8;
        } while (--sizeLeft != 0);
    }
#else
#error "Not yet implemented"
#endif

    ALIGN(16)
    static const uint8_t maskRhoEast2[16] = {
        11,
        8,
        9,
        10,
        15,
        12,
        13,
        14,
        3,
        0,
        1,
        2,
        7,
        4,
        5,
        6,
    };

#define CONST128(a) _mm_load_si128((const V128 *)&(a))
#define LOAD128(a) _mm_load_si128((const V128 *)&(a))
#define LOAD128u(a) _mm_loadu_si128((const V128 *)&(a))
#define LOAD4_32(a, b, c, d) _mm_setr_epi32(a, b, c, d)
#define ROL32(a, o) _mm_rol_epi32(a, o)
#define SHL32(a, o) _mm_slli_epi32(a, o)
#define STORE128(a, b) _mm_store_si128((V128 *)&(a), b)
#define STORE128u(a, b) _mm_storeu_si128((V128 *)&(a), b)
#define AND(a, b) _mm_and_si128(a, b)
#define XOR(a, b) _mm_xor_si128(a, b)
#define XOR3(a, b, c) _mm_ternarylogic_epi32(a, b, c, 0x96)
#define Chi(a, b, c) _mm_ternarylogic_epi32(a, b, c, 0xD2)

#define LOAD256u(a) _mm256_loadu_si256((const V256 *)&(a))
#define STORE256u(a, b) _mm256_storeu_si256((V256 *)&(a), b)
#define XOR256(a, b) _mm256_xor_si256(a, b)

#define LOAD512u(a) _mm512_loadu_si512((const V512 *)&(a))
#define STORE512u(a, b) _mm512_storeu_si512((V512 *)&(a), b)
#define XOR512(a, b) _mm512_xor_si512(a, b)

#define DeclareVars V128 a0, a1, a2, p, e, rhoEast2 = CONST128(maskRhoEast2);
#define State2Vars a0 = LOAD128(state->A[0]), a1 = LOAD128(state->A[4]), a2 = LOAD128(state->A[8]);
#define Vars2State STORE128(state->A[0], a0), STORE128(state->A[4], a1), STORE128(state->A[8], a2);

#define Round(__rc)                                                                       \
    /* Theta: Column Parity Mixer */                                                      \
    p = XOR3(a0, a1, a2);                                                                 \
    p = _mm_shuffle_epi32(p, 0x93);                                                       \
    e = ROL32(p, 5);                                                                      \
    p = ROL32(p, 14);                                                                     \
    a0 = XOR3(a0, e, p);                                                                  \
    a1 = XOR3(a1, e, p);                                                                  \
    a2 = XOR3(a2, e, p);                                                                  \
    Dump3("Theta"); /*dump prints the state with the string in dump function parameters*/ \
                                                                                          \
    /* Rho-west: Plane shift */                                                           \
    a1 = _mm_shuffle_epi32(a1, 0x93);                                                     \
    a2 = ROL32(a2, 11);                                                                   \
    Dump3("Rho-west");                                                                    \
                                                                                          \
    /* Iota: round constants */                                                           \
    a0 = XOR(a0, _mm_set_epi32(0, 0, 0, (__rc)));                                         \
    Dump3("Iota");                                                                        \
                                                                                          \
    /* Chi: non linear step, on colums */                                                 \
    a0 = Chi(a0, a1, a2);                                                                 \
    a1 = Chi(a1, a2, a0);                                                                 \
    a2 = Chi(a2, a0, a1);                                                                 \
    Dump3("Chi");                                                                         \
                                                                                          \
    /* Rho-east: Plane shift */                                                           \
    a1 = ROL32(a1, 1);                                                                    \
    a2 = _mm_shuffle_epi8(a2, rhoEast2);                                                  \
    Dump3("Rho-east")

    static const uint32_t RC[MAXROUNDS] = {
        _rc12,
        _rc11,
        _rc10,
        _rc9,
        _rc8,
        _rc7,
        _rc6,
        _rc5,
        _rc4,
        _rc3,
        _rc2,
        _rc1};
