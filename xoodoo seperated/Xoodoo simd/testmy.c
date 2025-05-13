#include <stdio.h>
#include <string.h>
#include <stddef.h>
#include <stdint.h>
#include <smmintrin.h>
#include <wmmintrin.h>
#include <immintrin.h>
#include <emmintrin.h>
#include "measurement.h"

#if defined(_MSC_VER)
#define CGAL_ALIGN_32 __declspec(align(32))
#elif defined(__GNUC__)
#define CGAL_ALIGN_32 __attribute__((aligned(32)))
#endif

#define MAXROUNDS 12
#define NROWS 3
#define NCOLUMS 4
#define NLANES (NCOLUMS * NROWS)

#define READ32_UNALIGNED(argAddress) (*((const uint32_t *)(argAddress)))

//__m256i _mm256_permutex_epi64(__m256i, const int);

#define Dump(__t)                                                                                                           \
    Vars2State;                                                                                                             \
    printf(__t "\n");                                                                                                       \
    printf("a00 %08x, a01 %08x, a02 %08x, a03 %08x\n", state->A[0 + 0], state->A[0 + 1], state->A[0 + 2], state->A[0 + 3]); \
    printf("a10 %08x, a11 %08x, a12 %08x, a13 %08x\n", state->A[4 + 0], state->A[4 + 1], state->A[4 + 2], state->A[4 + 3]); \
    printf("a20 %08x, a21 %08x, a22 %08x, a23 %08x\n\n", state->A[8 + 0], state->A[8 + 1], state->A[8 + 2], state->A[8 + 3]);\
    printf("a20 %08x, a21 %08x, a22 %08x, a23 %08x\n\n", state->A[12 + 0], state->A[12 + 1], state->A[12 + 2], state->A[12 + 3]);

#define Dump3(__t) Dump(__t)

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

typedef struct
{
    CGAL_ALIGN_32
    uint32_t A[16];
} Xoodoo_align128plain32_state;

typedef Xoodoo_align128plain32_state Xoodoo_state;

typedef __m128i V128;
typedef __m256i V256;
typedef __m512i V512;

uint32_t input[12] = {0xB2275BE8, 0x6A657F3D, 0x636CA821, 0xFF7DDEDD,
                      0x222CB942, 0x03019ACC, 0x32F2D96E, 0x9B047B36,
                      0x79CE4436, 0xE6687CC7, 0x1B62F7AD, 0x38D0C129};
                      
uint32_t input1[12] = {0x6c69dcb5, 0x2bffdf49, 0x0537a1f6, 0x81073228,
0x121aeba3, 0x9acc68c7, 0xac60e464, 0xe85ab8e9,
0x95ace5b1, 0x56b663b0,0x5e9cee13, 0xa97d9918};                      

CGAL_ALIGN_32
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
    27,
    24,
    25,
    26,
    31,
    28,
    29,
    30,
    19,
    16,
    17,
    18,
    23,
    20,
    21,
    22,
};

CGAL_ALIGN_32
static const uint32_t maskfornot[8] = {0,0,0,0,0xffffffff,0xffffffff,0xffffffff,0xffffffff};


CGAL_ALIGN_32
static const uint8_t shuffle_mask_for_a10[32] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                                                 12, 13, 14, 15, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};

CGAL_ALIGN_32 static const uint32_t sl1[8] = {0, 0, 0, 0, 1, 1, 1, 1};
CGAL_ALIGN_32 static const uint32_t sr1[8] = {32, 32, 32, 32, 31, 31, 31, 31};

#define CONST256(a) _mm256_load_si256((const V256 *)&(a))
#define LOAD256(a) _mm256_load_si256((const V256 *)&(a))
#define LOAD256u(a) _mm256_loadu_si256((const V256 *)&(a))
#define LOAD4_32(a, b, c, d) _mm_setr_epi32(a, b, c, d)
#define ROL32(a, o) _mm256_or_si256(_mm256_slli_epi32(a, o), _mm256_srli_epi32(a, 32 - (o)))
#define SHL32(a, o) _mm256_slli_epi32(a, o)
#define STORE256(a, b) _mm256_store_si256((V256 *)&(a), b)
#define STORE256u(a, b) _mm256_storeu_si256((V256 *)&(a), b)

#define AND(a, b) _mm256_and_si256(a, b)
#define XOR(a, b) _mm256_xor_si256(a, b)
#define XOR3(a, b, c) _mm256_ternarylogic_epi32(a, b, c, 0x96)
#define Chi(a, b, c) _mm256_ternarylogic_epi32(a, b, c, 0xD2)
#define set(a) _mm256_insertf128_si256(_mm256_castsi128_si256(a), (a), 0x1)

#define XOR256(a, b) _mm256_xor_si256(a, b)

#define Flip256(a) _mm256_permute4x64_epi64(a, 0x4e)

#define ANDnot256(a, b) _mm256_andnot_si256(a, b)

#define DeclareVars                                                                                                                                                              \
    V256 a01, a10, a22, a21, a02, a20, p, e,a2not2, rhoEast2 = CONST256(maskRhoEast2), mask_for_a10 = CONST256(shuffle_mask_for_a10), sl1mask = CONST256(sl1), sr1mask = CONST256(sr1),notmask=CONST256(maskfornot); \
    V128 a2;
#define State2Vars                                                                 \
    a10 = LOAD256(state->A[0]), a2 = _mm_load_si128((const V128 *)&(state->A[8])); \
    a22 = set(a2);
#define Vars2State STORE256(state->A[0], a10), STORE256(state->A[8], a22);

#define Round(__rc)                                                                          \
    /* Theta: Column Parity Mixer */							     \
    a01 = Flip256(a10);                                                                      \
    p = XOR(a10, XOR(a22, a01));                                                             \
    p = _mm256_shuffle_epi32(p, 0x93);                                                       \
    e = ROL32(p, 5);                                                                         \
    p = ROL32(p, 14);                                                                        \
    p = XOR(p, e);                                                                           \
    a10 = XOR(a10, p);                                                                       \
    a22 = XOR(a22, p);                                                                       \
                                                                                             \
    /*Dump3("Theta");*/                                                                         \
                                                                                             \
    /* Rho-west: Plane shift */                                                              \
    a10 = _mm256_shuffle_epi8(a10, mask_for_a10);                                            \
    a22 = ROL32(a22, 11);                                                                    \
    /*Dump3("Rho-west"); */                                                                      \
                                                                                             \
    /* Iota: round constants */                                                              \
    a10 = XOR(a10, _mm256_set_epi32(0, 0, 0, 0, 0, 0, 0, (__rc)));                           \
    /*Dump3("Iota");*/                                                                           \
                                                                                             \
    /* Chi: non linear step, on colums */                                                    \
    a01 = Flip256(a10);                                                                      \
    /*a21 = _mm256_blend_epi32(a01, a22, 0xf0); */                                           \
    /*a02 = _mm256_blend_epi32(a22, a01, 0xf0);*/                                            \
    a01=XOR(a01,notmask);								     \
    								     \
    											     \
    a10 = XOR(a10, ANDnot256(a01, XOR(a22,notmask)));                                                     \
    a22 = XOR(a22, ANDnot256(XOR(a10,notmask), a01));                                                     \
                                                                                             \
   /* Dump3("Chi");     */                                                                       \
                                                                                             \
    /* Rho-east: Plane shift */                                                              \
    a10 = _mm256_or_si256(_mm256_sllv_epi32(a10, sl1mask), _mm256_srlv_epi32(a10, sr1mask)); \
    a22 = _mm256_shuffle_epi8(a22, rhoEast2);                                                \
    /*Dump3("rho east")*/

;

#define Twelve_rounds \
    Round(_rc1);      \
    Round(_rc2);      \
    Round(_rc3);      \
    Round(_rc4);      \
    Round(_rc5);      \
    Round(_rc6);      \
    Round(_rc7);      \
    Round(_rc8);      \
    Round(_rc9);      \
    Round(_rc10);     \
    Round(_rc11);     \
    Round(_rc12);

void main()
{
    Xoodoo_align128plain32_state data;
    memset(&data, 0, sizeof(Xoodoo_align128plain32_state));
    printf("%ld\n", sizeof(Xoodoo_align128plain32_state));

    Xoodoo_align128plain32_state *state = &data;

    uint32_t *tempst = state->A;

    for (size_t i = 0; i < 12; i++)
    {
        *tempst++ = input[i];
    }

    // memcpy(tempst, &input, 4);

    printf("%08x\n", input[1]);

    printf("%08x", state->A[1]);

    DeclareVars;
    State2Vars;
    Vars2State;
    Dump3("initial");
    MEASURE(Twelve_rounds);
    printf("%f,%f,%f",RDTSC_clk_min ,RDTSC_clk_median,RDTSC_clk_max );
    /*for(int count=0;count<10;count++)
    {
    
    	printf("Round:%d\n",count);
    	Round(_rc1);
	
    }*/
    //Twelve_rounds
    Dump3("rho east");

    Vars2State;

    printf("%08x", state->A[1]);
}
