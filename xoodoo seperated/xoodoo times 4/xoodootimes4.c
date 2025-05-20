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


typedef struct
{
    CGAL_ALIGN_64
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



CGAL_ALIGN_64
static const uint8_t maskRhoEast2[64] = {
    11,8,9,10,15,12,13,14,3,0,1,2,7,4,5,6,
    11+(16*1),8+(16*1),9+(16*1),10+(16*1),15+(16*1),12+(16*1),13+(16*1),14+(16*1),3+(16*1),0+(16*1),1+(16*1),2+(16*1),7+(16*1),4+(16*1),5+(16*1),6+(16*1),
    11+(16*2),8+(16*2),9+(16*2),10+(16*2),15+(16*2),12+(16*2),13+(16*2),14+(16*2),3+(16*2),0+(16*2),1+(16*2),2+(16*2),7+(16*2),4+(16*2),5+(16*2),6+(16*2),
    11+(16*3),8+(16*3),9+(16*3),10+(16*3),15+(16*3),12+(16*3),13+(16*3),14+(16*3),3+(16*3),0+(16*3),1+(16*3),2+(16*3),7+(16*3),4+(16*3),5+(16*3),6+(16*3),
    
};


#define Dump(t)						\
{							\
	CGAL_ALIGN_64 uint32_t buf[48];			\
	STORE512(buf[0], a0);				\
	STORE512(buf[16], a1);				\
	STORE512(buf[32], a2);				\
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




#define DeclareVars V512 a0,a1,a2,p,e,rhoEast2=CONST512(maskRhoEast2);

#define State2Vars a0 = LOAD512(state->A[0]), a1 = LOAD512(state->A[4]), a2 = LOAD512(state->A[8]);
#define Vars2State STORE512(state->A[0], a0), STORE512(state->A[4], a1), STORE512(state->A[8], a2);


#define Round(__rc)										\
												\
	p=XOR3(a0,a1,a2);									\
	p=_mm512_shuffle_epi32(p,0x93);								\
	e=ROL32(p,5);										\
	p=ROL32(p,14);										\
	a0=XOR3(a0,e,p);									\
	a1=XOR3(a1,e,p);									\
	a2=XOR3(a2,e,p);									\
	/*Dump("theta");*/									\
												\
	a1=_mm512_shuffle_epi32(a1,0x93);							\
	a2=ROL32(a2,11);									\
	/*Dump("rho-east");*/									\
												\
												\
	a0=XOR512(a0,_mm512_set_epi32(0,0,0,(__rc),0,0,0,(__rc),0,0,0,(__rc),0,0,0,(__rc)));	\
	/*Dump("iota");*/									\
												\
												\
	a0=Chi(a0,a1,a2);									\
	a1=Chi(a1,a2,a0);									\
	a2=Chi(a2,a0,a1);									\
	/*Dump("chi");*/									\
												\
												\
	a1=ROL32(a1,1);										\
	a2=_mm512_shuffle_epi8(a2,rhoEast2);							\
	/*Dump("rho-west");*/									\


 uint32_t input1[48] = {0xB2275BE8, 0x6A657F3D, 0x636CA821, 0xFF7DDEDD,
                       0x222CB942, 0x03019ACC, 0x32F2D96E, 0x9B047B36,
                       0x79CE4436, 0xE6687CC7, 0x1B62F7AD, 0x38D0C129,
                       0x6c69dcb5, 0x2bffdf49, 0x0537a1f6, 0x81073228,
                       0xB2275BE8, 0x6A657F3D, 0x636CA821, 0xFF7DDEDD,
                       0x222CB942, 0x03019ACC, 0x32F2D96E, 0x9B047B36,
                       0x79CE4436, 0xE6687CC7, 0x1B62F7AD, 0x38D0C129,
                       0x6c69dcb5, 0x2bffdf49, 0x0537a1f6, 0x81073228,
                       0xB2275BE8, 0x6A657F3D, 0x636CA821, 0xFF7DDEDD,
                       0x222CB942, 0x03019ACC, 0x32F2D96E, 0x9B047B36,
                       0x79CE4436, 0xE6687CC7, 0x1B62F7AD, 0x38D0C129,
                       0x6c69dcb5, 0x2bffdf49, 0x0537a1f6, 0x81073228};
                       
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

#define TimesTwelve_rounds				\
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
	Round(_rc12);					\
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
	Round(_rc12);					\
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
	Round(_rc12);					\
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
	
	//Round(_rc1);
	//Twelve_rounds
	//MEASURE(TimesTwelve_rounds);
	MEASURE(Twelve_rounds);
	printf("%f,%f,%f",RDTSC_clk_min ,RDTSC_clk_median,RDTSC_clk_max );
	
	Dump("Final")
	
	
	
		
}	
	
	
	
	
	
