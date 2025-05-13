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
    ALIGN(32)
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

    ALIGN(32)
    static const uint8_t maskRhoEast2[32] = {
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
        11 + 16,
        8 + 16,
        9 + 16,
        10 + 16,
        15 + 16,
        12 + 16,
        13 + 16,
        14 + 16,
        3 + 16,
        0 + 16,
        1 + 16,
        2 + 16,
        7 + 16,
        4 + 16,
        5 + 16,
        6 + 16};

#define CONST256(a) _mm256_load_si256((const V256 *)&(a))
#define LOAD256(a) _mm256_load_si256((const V256 *)&(a))
#define LOAD256u(a) _mm256_loadu_si256((const V256 *)&(a))
#define LOAD4_32(a, b, c, d) _mm_setr_epi32(a, b, c, d)
#define ROL32(a, o) _mm256_rol_epi32(a, o)
#define SHL32(a, o) _mm256_slli_epi32(a, o)
#define STORE256(a, b) _mm256_store_si256((V256 *)&(a), b)
#define STORE256u(a, b) _mm256_storeu_si256((V256 *)&(a), b)
#define AND(a, b) _mm256_and_si256(a, b)
#define XOR(a, b) _mm256_xor_si256(a, b)
#define XOR3(a, b, c) _mm256_ternarylogic_epi32(a, b, c, 0x96)
#define Chi(a, b, c) _mm256_ternarylogic_epi32(a, b, c, 0xD2)
#define set(a) _mm256_set_m128i

#define XOR256(a, b) _mm256_xor_si256(a, b)

#define Flip256(a) _mm256_permutex_epi64(a, 0x8e)

#define DeclareVars
    V256 a01, a10, a22, a21, a02, a20, p, e, rhoEast2 = CONST256(maskRhoEast2);
    V128 a2;
#define State2Vars                                                                 \
    a10 = LOAD256(state->A[0]), a2 = _mm_load_si128((const V128 *)&(state->A[8])); \
    a22 = set(a2), a01 = Flip256(a10);
#define Vars2State STORE256(state->A[0], a01), STORE256(state->A[8], a22);

#define Round(__rc)                                                                       \
    /* Theta: Column Parity Mixer */                                                      \
    p = XOR3(a10, a22, a01);                                                              \
    p = _mm256_shuffle_epi32(p, 0x93);                                                    \
    e = ROL32(p, 5);                                                                      \
    p = ROL32(p, 14);                                                                     \
    a10 = XOR3(a10, e, p);                                                                \
    a22 = XOR3(a22, e, p);                                                                \
                                                                                          \
    Dump3("Theta"); /*dump prints the state with the string in dump function parameters*/ \
                                                                                          \
    /* Rho-west: Plane shift */                                                           \
    a10 = _mm256_mask_shuffle_epi32(a10, 0xf0, a10, 0x93);                                \
    a22 = _mm256_rol_epi32(a22, 11);                                                      \
    Dump3("Rho-west");                                                                    \
                                                                                          \
    /* Iota: round constants */                                                           \
    a10 = XOR(a10, _mm256_set_epi32(0, 0, 0, 0, 0, 0, 0, (__rc)));                        \
    Dump3("Iota");                                                                        \
                                                                                          \
    /* Chi: non linear step, on colums */                                                 \
    a01 = Flip256(a10);                                                                   \
    a21 = _mm256_blend_epi32(a01, a22, 0xf0);                                             \
    a20 = _mm256_blend_epi32(a22, a01, 0xf0);                                             \
    a10 = Chi(a10, a21, a20);                                                             \
    a22 = Chi(a22, a10, a01);                                                             \
                                                                                          \
    Dump3("Chi");                                                                         \
                                                                                          \
    /* Rho-east: Plane shift */                                                           \
    a10 = ROL32(a1, 1);                                                                   \
    a22 = _mm256_shuffle_epi8(a2, rhoEast2);                                              \
    Dump3("Rho-east")

    void main()
    {
        DeclareVars;
        State2Vars;

        Round(_rc1);
    }
