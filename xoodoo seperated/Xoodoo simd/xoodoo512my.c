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
#define CGAL_ALIGN_64 __declspec(align(64))
#elif defined(__GNUC__)
#define CGAL_ALIGN_64 __attribute__((aligned(64)))
#endif

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

#define Dump(__t)                                                                                                           \
    Vars2State;                                                                                                             \
    printf(__t "\n");                                                                                                       \
    printf("a00 %08x, a01 %08x, a02 %08x, a03 %08x\n", state->A[0 + 0], state->A[0 + 1], state->A[0 + 2], state->A[0 + 3]); \
    printf("a10 %08x, a11 %08x, a12 %08x, a13 %08x\n", state->A[4 + 0], state->A[4 + 1], state->A[4 + 2], state->A[4 + 3]); \
    printf("a20 %08x, a21 %08x, a22 %08x, a23 %08x\n\n", state->A[8 + 0], state->A[8 + 1], state->A[8 + 2], state->A[8 + 3]);

#define Dump3(__t) Dump(__t)

typedef struct
{
    CGAL_ALIGN_64
    uint32_t A[12];
} Xoodoo_align128plain32_state;

typedef Xoodoo_align128plain32_state Xoodoo_state;

typedef __m128i V128;
typedef __m256i V256;
typedef __m512i V512;

uint32_t input[12] = {0x6c69dcb5, 0x2bffdf49, 0x0537a1f6, 0x81073228,
0x121aeba3, 0x9acc68c7, 0xac60e464, 0xe85ab8e9,
0x95ace5b1, 0x56b663b0,0x5e9cee13, 0xa97d9918};

CGAL_ALIGN_64
static const uint8_t maskRhoEast2[64] = {
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
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
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
};

CGAL_ALIGN_64
static const uint8_t maskforrotstate[64] = {
    0+16,1+16,2+16,3+16,4+16,5+16,6+16,7+16,8+16,9+16,10+16,11+16,12+16,13+16,14+16,15+16,
    0+32,1+32,2+32,3+32,4+32,5+32,6+32,7+32,8+32,9+32,10+32,11+32,12+32,13+32,14+32,15+32,
    0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
};

#define CONST512(a) _mm512_load_si512((const V512 *)&(a))
#define LOAD512(a) _mm512_load_si512((const V512 *)&(a))
#define LOAD128u(a) _mm_loadu_si128((const V128 *)&(a))
#define LOAD4_32(a, b, c, d) _mm_setr_epi32(a, b, c, d)
#define ROL32(a, o) _mm512_rol_epi32(a, o)
#define MASKROL32(src,mask,a,o) _mm_mask_rol_epi32 (src,mask,a,o)
#define SHL32(a, o) _mm_slli_epi32(a, o)
#define STORE512(a, b) _mm512_store_si512((V512 *)&(a), b)
#define STORE128u(a, b) _mm_storeu_si128((V128 *)&(a), b)
#define AND(a, b) _mm512_and_si512 (a, b)
#define XOR(a, b) _mm512_xor_si512(a, b)
#define XOR3(a, b, c) _mm512_ternarylogic_epi32 (a, b, c, 0x96)
#define Chi(a, b, c) _mm512_ternarylogic_epi32 (a, b, c, 0xD2)

#define LOAD256u(a) _mm256_loadu_si256((const V256 *)&(a))
#define STORE256u(a, b) _mm256_storeu_si256((V256 *)&(a), b)
#define XOR256(a, b) _mm256_xor_si256(a, b)

#define LOAD512u(a) _mm512_loadu_si512((const V512 *)&(a))
#define STORE512u(a, b) _mm512_storeu_si512((V512 *)&(a), b)


#define DeclareVars V512 s,s1,s2, p, e, rhoEast2 = CONST512(maskRhoEast2),rotstate=CONST512(maskforrotstate);
#define State2Vars s = LOAD512(state->A[0]);
#define Vars2State STORE512(state->A[0], s);

#define Round(__rc)                                                                       \
    /* Theta: Column Parity Mixer */                                                      \
    s1=__m512i _mm512_shuffle_epi8(s,rotstate);						  \
    s2=__m512i _mm512_shuffle_epi8(s1,rotstate);						  \
    p = XOR3(s, s1, s2);                                                                 \
    p = _mm512_shuffle_epi32(p, 0x93);                                                       \
    e = ROL32(p, 5);                                                                      \
    p = ROL32(p, 14);                                                                     \
    s = XOR3(s, e, p);                                                                  \
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
     s1=__m512i _mm512_shuffle_epi8(s,rotstate);						  \
    s2=__m512i _mm512_shuffle_epi8(s1,rotstate);                                                                \
    s = Chi(s, s1, s2);                                                                 \
    Dump3("Chi");                                                                         \
                                                                                          \
    /* Rho-east: Plane shift */                                                           \
    a1 = ROL32(a1, 1);                                                                    \
    a2 = _mm_shuffle_epi8(a2, rhoEast2);                                                  \
    Dump3("Rho-east")

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

    printf("%08x\n", input[4]);

    printf("%08x", state->A[4]);

    DeclareVars;
    State2Vars;

    Dump3("initial");

   //MEASURE(Twelve_rounds);
   //printf("%f,%f,%f",RDTSC_clk_min ,RDTSC_clk_median,RDTSC_clk_max );
	
	for(int count=0;count<10;count++)
    {
    	printf("Round:%d\n",count);
    	Round(_rc1);
	
    }	
    Dump3("Rho-east")
    Vars2State;
}
