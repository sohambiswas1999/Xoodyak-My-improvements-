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

typedef __m128i V128;
typedef __m256i V256;
typedef __m512i V512;



typedef __m512i V512;

typedef struct {
    V512 A[12];
} Xoodootimes16_SIMD512_states;

typedef Xoodootimes16_SIMD512_states Xoodootimes16_states;


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



#define Dump(t)						\
{							\
	uint32_t buf[96*2];			\
	STORE512(buf[0], a0);				\
	STORE512(buf[16*1], a1);				\
	STORE512(buf[16*2], a2);				\
	STORE512(buf[16*3], a3);				\
	STORE512(buf[16*4], a4);				\
	STORE512(buf[16*5], a5);				\
	STORE512(buf[16*6], a6);				\
	STORE512(buf[16*7], a7);				\
	STORE512(buf[16*8], a8);				\
	STORE512(buf[16*9], a9);				\
	STORE512(buf[16*10], a10);				\
	STORE512(buf[16*11], a11);				\
	printf("%s\n",t);				\
	for(int j=0;j<12;j++)				\
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


CGAL_ALIGN_64
static const uint8_t maskRhoEast2[64] = {
    11,8,9,10,15,12,13,14,3,0,1,2,7,4,5,6,
    11+(16*1),8+(16*1),9+(16*1),10+(16*1),15+(16*1),12+(16*1),13+(16*1),14+(16*1),3+(16*1),0+(16*1),1+(16*1),2+(16*1),7+(16*1),4+(16*1),5+(16*1),6+(16*1),
    11+(16*2),8+(16*2),9+(16*2),10+(16*2),15+(16*2),12+(16*2),13+(16*2),14+(16*2),3+(16*2),0+(16*2),1+(16*2),2+(16*2),7+(16*2),4+(16*2),5+(16*2),6+(16*2),
    11+(16*3),8+(16*3),9+(16*3),10+(16*3),15+(16*3),12+(16*3),13+(16*3),14+(16*3),3+(16*3),0+(16*3),1+(16*3),2+(16*3),7+(16*3),4+(16*3),5+(16*3),6+(16*3),
    
};



#define Chi(a, b, c) _mm512_ternarylogic_epi32(a, b, c, 0xD2)

#define LOAD(a) _mm_load_si128((const V128 *)&(a)) 

#define CONST4_32(a) _mm_set1_epi32(a)
#define LOAD256u(a) _mm512_loadu_si256((const V256 *)&(a))

#define LOAD512(a) _mm512_load_si512((const V512 *)&(a))
#define LOAD512u(a) _mm512_loadu_si512((const V512 *)&(a))

#define LOAD_GATHER4_32(idx, p) _mm_i32gather_epi32((const int *)(p), idx, 4)
#define STORE_SCATTER4_32(idx, a, p) _mm_i32scatter_epi32((void *)(p), idx, a, 4)
#define LOAD4_32(a, b, c, d) _mm_setr_epi32(a, b, c, d)

#define SHUFFLE_LANES_RIGHT(idx, a) _mm_permutexvar_epi32(idx, a)

#define ROL32(a, o) _mm512_rol_epi32(a, o)
#define SHL32(a, o) _mm512_slli_epi32(a, o)

#define SET4_32 _mm_setr_epi32

#define STORE128(a, b) _mm_store_si128((V128 *)&(a), b)
#define STORE128u(a, b) _mm_storeu_si128((V128 *)&(a), b)
#define STORE256u(a, b) _mm256_storeu_si256((V256 *)&(a), b)
#define STORE256(a, b) _mm256_store_si256((V256 *)&(a), b)
#define STORE512(a, b) _mm512_store_si512((V512 *)&(a), b)
#define STORE512u(a, b) _mm512_storeu_si512((V512 *)&(a), b)
#define CONST512(a) _mm512_load_si512((const V512 *)&(a))

#define AND512(a, b) _mm512_and_si128(a, b)
#define XOR(a, b) _mm_xor_si128(a, b)
#define XOR256(a, b) _mm256_xor_si256(a, b)
#define XOR512(a, b) _mm512_xor_si512(a, b)
#define XOR3(a, b, c) _mm512_ternarylogic_epi32(a, b, c, 0x96)



#define DeclareVars V512 a0,a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,p0,p1,p2,p3,e0,e1,e2,e3,rconst,rhoEast2=CONST512(maskRhoEast2);

#define State2Vars a0 = LOAD512(state->A[0]), a1 = LOAD512(state->A[1]), a2 = LOAD512(state->A[2]),			\
		   a3 = LOAD512(state->A[3]), a4 = LOAD512(state->A[4]), a5 = LOAD512(state->A[5]),			\
		   a6 = LOAD512(state->A[6]), a7 = LOAD512(state->A[7]), a8 = LOAD512(state->A[8]),			\
		   a9 = LOAD512(state->A[9]), a10 = LOAD512(state->A[10]), a11 = LOAD512(state->A[11]);			


#define Vars2State STORE512(state->A[0], a0), STORE512(state->A[1], a1), STORE512(state->A[2], a2),			\
		   STORE512(state->A[3], a3), STORE512(state->A[4], a4), STORE512(state->A[5], a5),			\
		   STORE512(state->A[6], a6), STORE512(state->A[7], a7), STORE512(state->A[8], a8),			\
		   STORE512(state->A[9], a9), STORE512(state->A[10], a10), STORE512(state->A[11], a11);	
		   
		   
		  
# define Round(__rc)									\
											\
											\
	p0=XOR3(a0,a4,a8);								\
	p1=XOR3(a1,a5,a9);								\
	p2=XOR3(a2,a6,a10);								\
	p3=XOR3(a3,a7,a11);								\
											\
											\
	p0=_mm512_shuffle_epi32(p0,0x93);						\
	p1=_mm512_shuffle_epi32(p1,0x93);						\
	p2=_mm512_shuffle_epi32(p2,0x93);						\
	p3=_mm512_shuffle_epi32(p3,0x93);						\
											\
											\
	e0=ROL32(p0,5);									\
	e1=ROL32(p1,5);									\
	e2=ROL32(p2,5);									\
	e3=ROL32(p3,5);									\
	p0=ROL32(p0,14);								\
	p1=ROL32(p1,14);								\
	p2=ROL32(p2,14);								\
	p3=ROL32(p3,14);								\
											\
											\
	a0=XOR3(a0,p0,e0);								\
	a4=XOR3(a4,p0,e0);								\
	a8=XOR3(a8,p0,e0);								\
	a1=XOR3(a1,p1,e1);								\
	a5=XOR3(a5,p1,e1);								\
	a9=XOR3(a9,p1,e1);								\
	a2=XOR3(a2,p2,e2);								\
	a6=XOR3(a6,p2,e2);								\
	a10=XOR3(a10,p2,e2);								\
	a3=XOR3(a3,p3,e3);								\
	a7=XOR3(a7,p3,e3);								\
	a11=XOR3(a11,p3,e3);								\
											\
											\
										\
	a4=_mm512_shuffle_epi32(a4,0x93);						\
	a5=_mm512_shuffle_epi32(a5,0x93);						\
	a6=_mm512_shuffle_epi32(a6,0x93);						\
	a7=_mm512_shuffle_epi32(a7,0x93);						\
	a8=ROL32(a8,11);								\
	a9=ROL32(a9,11);								\
	a10=ROL32(a10,11);								\
	a11=ROL32(a11,11);								\
											\
											\
	rconst=_mm512_set_epi32(0,0,0,(__rc),0,0,0,(__rc),0,0,0,(__rc),0,0,0,(__rc));	\
	a0=XOR512(a0,rconst);								\
	a1=XOR512(a1,rconst);								\
	a2=XOR512(a2,rconst);								\
	a3=XOR512(a3,rconst);								\
											\
											\
	a0=Chi(a0,a4,a8);								\
	a1=Chi(a1,a5,a9);								\
	a2=Chi(a2,a6,a10);								\
	a3=Chi(a3,a7,a11);								\
	a4=Chi(a4,a8,a0);								\
	a5=Chi(a5,a9,a1);								\
	a6=Chi(a6,a10,a2);								\
	a7=Chi(a7,a11,a3);								\
	a8=Chi(a8,a0,a4);								\
	a9=Chi(a9,a1,a5);								\
	a10=Chi(a10,a2,a6);								\
	a11=Chi(a11,a3,a7);								\
											\
											\
	a4=ROL32(a4,1);									\
	a5=ROL32(a5,1);									\
	a6=ROL32(a6,1);									\
	a7=ROL32(a7,1);									\
	a8=_mm512_shuffle_epi8(a8,rhoEast2);						\
	a9=_mm512_shuffle_epi8(a9,rhoEast2);						\
	a10=_mm512_shuffle_epi8(a10,rhoEast2);						\
	a11=_mm512_shuffle_epi8(a11,rhoEast2);						\
	
											
											
uint32_t input[48*4] = {0xB2275BE8, 0x6A657F3D, 0x636CA821, 0xFF7DDEDD,
			0xB2275BE8, 0x6A657F3D, 0x636CA821, 0xFF7DDEDD,
			0xB2275BE8, 0x6A657F3D, 0x636CA821, 0xFF7DDEDD,
			0xB2275BE8, 0x6A657F3D, 0x636CA821, 0xFF7DDEDD,
			0xB2275BE8, 0x6A657F3D, 0x636CA821, 0xFF7DDEDD,
			0xB2275BE8, 0x6A657F3D, 0x636CA821, 0xFF7DDEDD,
			0xB2275BE8, 0x6A657F3D, 0x636CA821, 0xFF7DDEDD,
			0xB2275BE8, 0x6A657F3D, 0x636CA821, 0xFF7DDEDD,
			0xB2275BE8, 0x6A657F3D, 0x636CA821, 0xFF7DDEDD,
			0xB2275BE8, 0x6A657F3D, 0x636CA821, 0xFF7DDEDD,
			0xB2275BE8, 0x6A657F3D, 0x636CA821, 0xFF7DDEDD,
			0xB2275BE8, 0x6A657F3D, 0x636CA821, 0xFF7DDEDD,
			0xB2275BE8, 0x6A657F3D, 0x636CA821, 0xFF7DDEDD,
			0xB2275BE8, 0x6A657F3D, 0x636CA821, 0xFF7DDEDD,
			0xB2275BE8, 0x6A657F3D, 0x636CA821, 0xFF7DDEDD,
			0xB2275BE8, 0x6A657F3D, 0x636CA821, 0xFF7DDEDD,
			
			0x222CB942, 0x03019ACC, 0x32F2D96E, 0x9B047B36,
			0x222CB942, 0x03019ACC, 0x32F2D96E, 0x9B047B36,
			0x222CB942, 0x03019ACC, 0x32F2D96E, 0x9B047B36,
			0x222CB942, 0x03019ACC, 0x32F2D96E, 0x9B047B36,
			0x222CB942, 0x03019ACC, 0x32F2D96E, 0x9B047B36,
			0x222CB942, 0x03019ACC, 0x32F2D96E, 0x9B047B36,
			0x222CB942, 0x03019ACC, 0x32F2D96E, 0x9B047B36,
			0x222CB942, 0x03019ACC, 0x32F2D96E, 0x9B047B36,
			0x222CB942, 0x03019ACC, 0x32F2D96E, 0x9B047B36,
			0x222CB942, 0x03019ACC, 0x32F2D96E, 0x9B047B36,
			0x222CB942, 0x03019ACC, 0x32F2D96E, 0x9B047B36,
			0x222CB942, 0x03019ACC, 0x32F2D96E, 0x9B047B36,
			0x222CB942, 0x03019ACC, 0x32F2D96E, 0x9B047B36,
			0x222CB942, 0x03019ACC, 0x32F2D96E, 0x9B047B36,
			0x222CB942, 0x03019ACC, 0x32F2D96E, 0x9B047B36,
			0x222CB942, 0x03019ACC, 0x32F2D96E, 0x9B047B36,
			
			0x79CE4436, 0xE6687CC7, 0x1B62F7AD, 0x38D0C129,
			0x79CE4436, 0xE6687CC7, 0x1B62F7AD, 0x38D0C129,
			0x79CE4436, 0xE6687CC7, 0x1B62F7AD, 0x38D0C129,
			0x79CE4436, 0xE6687CC7, 0x1B62F7AD, 0x38D0C129,
			0x79CE4436, 0xE6687CC7, 0x1B62F7AD, 0x38D0C129,
			0x79CE4436, 0xE6687CC7, 0x1B62F7AD, 0x38D0C129,
			0x79CE4436, 0xE6687CC7, 0x1B62F7AD, 0x38D0C129,
			0x79CE4436, 0xE6687CC7, 0x1B62F7AD, 0x38D0C129,
			0x79CE4436, 0xE6687CC7, 0x1B62F7AD, 0x38D0C129,
			0x79CE4436, 0xE6687CC7, 0x1B62F7AD, 0x38D0C129,
			0x79CE4436, 0xE6687CC7, 0x1B62F7AD, 0x38D0C129,
			0x79CE4436, 0xE6687CC7, 0x1B62F7AD, 0x38D0C129,
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
	Xoodootimes16_SIMD512_states data;
	Xoodootimes16_SIMD512_states *state = &data;

	state->A[0]=LOAD512(input[0]);
	state->A[1]=LOAD512(input[16*1]);
	state->A[2]=LOAD512(input[16*2]);
	state->A[3]=LOAD512(input[16*3]);
	state->A[4]=LOAD512(input[16*4]);
	state->A[5]=LOAD512(input[16*5]);
	state->A[6]=LOAD512(input[16*6]);
	state->A[7]=LOAD512(input[16*7]);
	state->A[8]=LOAD512(input[16*8]);
	state->A[9]=LOAD512(input[16*9]);
	state->A[10]=LOAD512(input[16*10]);
	state->A[11]=LOAD512(input[16*11]);
	
	V512* states=state->A;
	
	DeclareVars;
	State2Vars ;
	Dump("initial");
	
	//Round(_rc1);
	//Twelve_rounds
	MEASURE(Twelve_rounds);
	printf("%f,%f,%f",RDTSC_clk_min ,RDTSC_clk_median,RDTSC_clk_max );
	
	Dump("Final")
	
	
	
		
}		
