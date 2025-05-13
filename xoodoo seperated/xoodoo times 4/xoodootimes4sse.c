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

typedef __m128i V128;
typedef __m256i V256;
typedef __m512i V512;


typedef struct
{
    CGAL_ALIGN_32
    V128 A[12];
} Xoodoo_align128plain32_state;

typedef Xoodoo_align128plain32_state Xoodoo_state;

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


CGAL_ALIGN_32
static const uint8_t maskRhoEast2[32] = {
    11,8,9,10,15,12,13,14,3,0,1,2,7,4,5,6,
    11+(16*1),8+(16*1),9+(16*1),10+(16*1),15+(16*1),12+(16*1),13+(16*1),14+(16*1),3+(16*1),0+(16*1),1+(16*1),2+(16*1),7+(16*1),4+(16*1),5+(16*1),6+(16*1)};


#define Dump(t)						\
{							\
	CGAL_ALIGN_32 uint32_t buf[48];			\
	STORE256(buf[0], a0);				\
	STORE256(buf[8], a1);				\
	STORE256(buf[16], a2);				\
	STORE256(buf[24], a3);				\
	STORE256(buf[32], a4);				\
	STORE256(buf[40], a5);				\
	printf("%s\n",t);\
	for(int j=0;j<3;j++)				\
	{						\
		for(int i=0;i<16;i++)			\
		{					\
			if(i%4==0) printf("\n");\
			printf("%d : %08x ",j*16+i,buf[j*16+i]);	\
					\
		}					\
		printf("\n");				\
	}						\
}


#define CONST256(a) _mm256_load_si256((const V256 *)&(a))
#define LOAD256(a) _mm256_load_si256((const V256 *)&(a))
#define LOAD256u(a) _mm256_loadu_si256((const V256 *)&(a))
#define LOAD4_32(a, b, c, d) _mm_setr_epi32(a, b, c, d)
#define ROL(a, o) _mm256_or_si256(_mm256_slli_epi32(a, o), _mm256_srli_epi32(a, 32 - (o)))
#define SHL32(a, o) _mm256_slli_epi32(a, o)
#define STORE256(a, b) _mm256_store_si256((V256 *)&(a), b)
#define STORE256u(a, b) _mm256_storeu_si256((V256 *)&(a), b)

#define LOAD(a) _mm_load_si128((const V128 *)&(a)) 

#define AND(a, b) _mm256_and_si256(a, b)
#define XOR(a, b) _mm256_xor_si256(a, b)
#define XOR3(a, b, c) _mm256_ternarylogic_epi32(a, b, c, 0x96)
#define Chi(a, b, c) _mm256_ternarylogic_epi32(a, b, c, 0xD2)
#define set(a) _mm256_insertf128_si256(_mm256_castsi128_si256(a), (a), 0x1)

#define XOR256(a, b) _mm256_xor_si256(a, b)

#define Flip256(a) _mm256_permute4x64_epi64(a, 0x4e)

#define ANDnot256(a, b) _mm256_andnot_si256(a, b)


#define DeclareVars V256 a0,a1,a2,a3,a4,a5,p0,p1,e0,e1,rhoEast2=CONST256(maskRhoEast2);

#define State2Vars a0 = LOAD256(state->A[0]), a1 = LOAD256(state->A[2]), a2 = LOAD256(state->A[4]),a3 = LOAD256(state->A[6]), a4 = LOAD256(state->A[8]), a5 = LOAD256(state->A[10]);
#define Vars2State STORE256(state->A[0], a0), STORE256(state->A[2], a1), STORE256(state->A[4], a2),STORE256(state->A[6], a3), STORE256(state->A[8], a4), STORE256(state->A[10], a5);


#define Round(__rc)									\
											\
											\
p0=XOR256(a1,XOR256(a3,a5));								\
p1=XOR256(a0,XOR256(a2,a4));								\
											\
p0=_mm256_shuffle_epi32(p0,0x93);							\
p1=_mm256_shuffle_epi32(p1,0x93);							\
											\
e0=ROL(p0,5);										\
e1=ROL(p1,5);										\
p0=ROL(p0,14);										\
p1=ROL(p1,14);										\
											\
p0=XOR256(p0,e0);									\
p1=XOR256(p1,e1);									\
											\
a0=XOR256(p0,a0);									\
a2=XOR256(p0,a2);									\
a4=XOR256(p0,a4);									\
a1=XOR256(p1,a1);									\
a3=XOR256(p1,a3);									\
a5=XOR256(p1,a5);									\
											\
/*Dump("theta");*/										\
											\
a2=_mm256_shuffle_epi32(a2,0x93);							\
a3=_mm256_shuffle_epi32(a3,0x93);							\
a4=ROL(a4,11);										\
a5=ROL(a5,11);										\
											\
/*Dump("rho-east");*/									\
											\
a0=XOR256(a0,_mm256_set_epi32(0,0,0,(__rc),0,0,0,(__rc)));				\
a1=XOR256(a1,_mm256_set_epi32(0,0,0,(__rc),0,0,0,(__rc)));				\
											\
/*Dump("iota");*/										\
											\
a0=XOR256(a0,ANDnot256(a2,a4));								\
a1=XOR256(a1,ANDnot256(a3,a5));								\
a2=XOR256(a2,ANDnot256(a4,a0));								\
a3=XOR256(a3,ANDnot256(a5,a1));								\
a4=XOR256(a4,ANDnot256(a0,a2));								\
a5=XOR256(a5,ANDnot256(a1,a3));								\
											\
/*Dump("chi");*/										\
											\
a2=ROL(a2,1);										\
a3=ROL(a3,1);										\
a4=_mm256_shuffle_epi8(a4,rhoEast2);							\
a5=_mm256_shuffle_epi8(a5,rhoEast2);							\
/*Dump("rho-west");*/


uint32_t input[48] = {0xB2275BE8, 0x6A657F3D, 0x636CA821, 0xFF7DDEDD,
			0xB2275BE8, 0x6A657F3D, 0x636CA821, 0xFF7DDEDD,
			0xB2275BE8, 0x6A657F3D, 0x636CA821, 0xFF7DDEDD,
			0xB2275BE8, 0x6A657F3D, 0x636CA821, 0xFF7DDEDD,
			
			0x222CB942, 0x03019ACC, 0x32F2D96E, 0x9B047B36,
			0x222CB942, 0x03019ACC, 0x32F2D96E, 0x9B047B36,
			0x222CB942, 0x03019ACC, 0x32F2D96E, 0x9B047B36,
			0x222CB942, 0x03019ACC, 0x32F2D96E, 0x9B047B36,
			
			0x79CE4436, 0xE6687CC7, 0x1B62F7AD, 0x38D0C129,
			0x79CE4436, 0xE6687CC7, 0x1B62F7AD, 0x38D0C129,
			0x79CE4436, 0xE6687CC7, 0x1B62F7AD, 0x38D0C129,
			0x79CE4436, 0xE6687CC7, 0x1B62F7AD, 0x38D0C129};
			
			
#define Twelve_rounds				\
	Round(_rc1);					\
	Round(_rc2);					\
	Round(_rc3);					\
	Round(_rc4);					\
	Round(_rc5);					\
	Round(_rc6);					\
	Round(_rc7);					\
	Round(_rc8);					\
	Round(_rc9);					\
	Round(_rc10);					\
	Round(_rc11);					\
	Round(_rc12);
	
	
void main()
{
	Xoodoo_align128plain32_state data;
	Xoodoo_align128plain32_state *state = &data;
	
	state->A[0]=LOAD(input[0]);
	state->A[1]=LOAD(input[4]);
	state->A[2]=LOAD(input[8]);
	state->A[3]=LOAD(input[12]);
	state->A[4]=LOAD(input[16]);
	state->A[5]=LOAD(input[20]);
	state->A[6]=LOAD(input[24]);
	state->A[7]=LOAD(input[28]);
	state->A[8]=LOAD(input[32]);
	state->A[9]=LOAD(input[36]);
	state->A[10]=LOAD(input[40]);
	state->A[11]=LOAD(input[44]);
	
	
	DeclareVars;
	State2Vars ;
	Dump("initial");
	
	//Twelve_rounds
	MEASURE(Twelve_rounds);
	printf("%f,%f,%f",RDTSC_clk_min ,RDTSC_clk_median,RDTSC_clk_max );
	
	Dump("Final")
}									
