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
#define CGAL_ALIGN_16 __declspec(align(16))
#elif defined(__GNUC__)
#define CGAL_ALIGN_16 __attribute__((aligned(16)))
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
    CGAL_ALIGN_16
    uint32_t A[12];
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

CGAL_ALIGN_16
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

#define ANDnu128(a, b) _mm_andnot_si128(a, b)
#define CONST128(a) _mm_load_si128((const V128 *)&(a))
#define LOAD128(a) _mm_load_si128((const V128 *)&(a))
#if defined(Waffel_useXOP)
#define ROL32in128(a, o) _mm_roti_epi32(a, o)
#else
#define ROL32in128(a, o) _mm_or_si128(_mm_slli_epi32(a, o), _mm_srli_epi32(a, 32 - (o)))
#endif
#define STORE128(a, b) _mm_store_si128((V128 *)&(a), b)
#define XOR128(a, b) _mm_xor_si128(a, b)

#define DeclareVars        \
    V128 a0, a1, a2, p, e; \
    V128 rhoEast2 = CONST128(maskRhoEast2)

#define State2Vars a0 = LOAD128(state->A[0]), a1 = LOAD128(state->A[4]), a2 = LOAD128(state->A[8]);

#define Vars2State STORE128(state->A[0], a0), STORE128(state->A[4], a1), STORE128(state->A[8], a2);

/*
** Theta: Column Parity Mixer
*/
#define Theta()                     \
    p = XOR128(a0, a1);             \
    p = XOR128(p, a2);              \
    p = _mm_shuffle_epi32(p, 0x93); \
    e = ROL32in128(p, 5);           \
    p = ROL32in128(p, 14);          \
    e = XOR128(e, p);               \
    a0 = XOR128(a0, e);             \
    a1 = XOR128(a1, e);             \
    a2 = XOR128(a2, e);

/*
** Rho-west: Plane shift
*/
#define Rho_west()                    \
    a1 = _mm_shuffle_epi32(a1, 0x93); \
    a2 = ROL32in128(a2, 11);

/*
** Iota: round constants
*/
#define Iota(__rc) a0 = XOR128(a0, _mm_set_epi32(0, 0, 0, (__rc)));

/*
** Chi: non linear step, on colums
*/
#define Chi()                          \
    a0 = XOR128(a0, ANDnu128(a1, a2)); \
    a1 = XOR128(a1, ANDnu128(a2, a0)); \
    a2 = XOR128(a2, ANDnu128(a0, a1));

/*
** Rho-east: Plane shift
*/
#define Rho_east()          \
    a1 = ROL32in128(a1, 1); \
    a2 = _mm_shuffle_epi8(a2, rhoEast2);

#define Round(__rc)    \
    Theta();           \
    /*Dump3("Theta"); */   \
    Rho_west();        \
    /*Dump3("Rho-west");*/ \
    Iota(__rc);        \
    /*Dump3("Iota");  */   \
    Chi();             \
    /*Dump3("Chi");    */ \
    Rho_east();        \
    /*Dump3("Rho-east")*/

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
    
    Twelve_rounds;

   /*MEASURE(Chi());
   printf("%f,%f,%f",RDTSC_clk_min ,RDTSC_clk_median,RDTSC_clk_max );
	
	for(int count=0;count<10;count++)
    {
    	printf("Round:%d\n",count);
    	Round(_rc1);
	
    }	*/
    Dump3("Rho-east")
    Vars2State;
}
