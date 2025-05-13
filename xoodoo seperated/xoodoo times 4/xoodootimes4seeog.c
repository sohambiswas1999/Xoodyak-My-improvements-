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

typedef __m128i V128;
typedef __m256i V256;
typedef __m512i V512;


typedef struct
{
    CGAL_ALIGN_16
    V128 A[12];
} Xoodoo_align128plain32_state;

typedef Xoodoo_align128plain32_state Xoodoo_state;

#define Dump(t)						\
{							\
	CGAL_ALIGN_16 uint32_t buf[48];			\
	STORE128(buf[0], a00);				\
	STORE128(buf[4], a01);				\
	STORE128(buf[8], a02);				\
	STORE128(buf[12], a03);				\
	STORE128(buf[16], a10);				\
	STORE128(buf[20], a11);				\
	STORE128(buf[24], a12);				\
	STORE128(buf[28], a13);				\
	STORE128(buf[32], a20);				\
	STORE128(buf[36], a21);				\
	STORE128(buf[40], a22);				\
	STORE128(buf[44], a23);				\
	printf("%s\n",t);				\
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



#define ANDnu128(a, b)              _mm_andnot_si128(a, b)
#define LOAD128(a)                  _mm_load_si128((const V128 *)&(a))
#define LOAD4_32(a,b,c,d)           _mm_setr_epi32(a,b,c,d)
#if defined(Waffel_useXOP)
    #define ROL32in128(a, o)    _mm_roti_epi32(a, o)
//        #define ROL32in128_8(a)     ROL32in128(a, 8)
#else
    #define ROL32in128(a, o)        _mm_or_si128(_mm_slli_epi32(a, o), _mm_srli_epi32(a, 32-(o)))
//        #define ROL32in128_8(a)     _mm_shuffle_epi8(a, CONST128(rho8))
//static const uint64_t rho8[2] = {0x0605040302010007, 0x0E0D0C0B0A09080F};
#endif
#define STORE128(a, b)              _mm_store_si128((V128 *)&(a), b)
#if defined(__SSE41__) || defined(__SSE4_1__)
#define STORE4_32(r, a, b, c, d)    a = _mm_extract_epi32(r, 0), b = _mm_extract_epi32(r, 1), c = _mm_extract_epi32(r, 2), d = _mm_extract_epi32(r, 3)
#else
#define STORE4_32(r, a, b, c, d)    a = _mm_cvtsi128_si32(r), b = _mm_cvtsi128_si32(_mm_srli_si128(r,4)), c = _mm_cvtsi128_si32(_mm_srli_si128(r,8)), d = _mm_cvtsi128_si32(_mm_srli_si128(r,12))
#endif
#define XOR128(a, b)                _mm_xor_si128(a, b)
#define XOReq128(a, b)              a = XOR128(a, b)




#define DeclareVars     V128    a00, a01, a02, a03; \
                        V128    a10, a11, a12, a13; \
                        V128    a20, a21, a22, a23; \
                        V128    v1, v2

#define State2Vars      a00 = LOAD128(state->A[0]), a01 = LOAD128(state->A[1]), a02 = LOAD128(state->A[2]), a03 = LOAD128(state->A[3]); \
                        a10 = LOAD128(state->A[4]), a11 = LOAD128(state->A[5]), a12 = LOAD128(state->A[6]), a13 = LOAD128(state->A[7]); \
                        a20 = LOAD128(state->A[8]), a21 = LOAD128(state->A[9]), a22 = LOAD128(state->A[10]), a23 = LOAD128(state->A[11])

#define State2Vars2     a00 = LOAD128(state->A[(0+0)]), a01 = LOAD128(state->A[(0+1)]), a02 = LOAD128(state->A[(0+2)]), a03 = LOAD128(state->A[(0+3)]); \
                        a12 = LOAD128(state->A[(4+0)]), a13 = LOAD128(state->A[(4+1)]), a10 = LOAD128(state->A[(4+2)]), a11 = LOAD128(state->A[(4+3)]); \
                        a20 = LOAD128(state->A[(8+0)]), a21 = LOAD128(state->A[(8+1)]), a22 = LOAD128(state->A[(8+2)]), a23 = LOAD128(state->A[(8+3)])

#define Vars2State      STORE128(state->A[(0+0)], a00), STORE128(state->A[(0+1)], a01), STORE128(state->A[(0+2)], a02), STORE128(state->A[(0+3)], a03); \
                        STORE128(state->A[(4+0)], a10), STORE128(state->A[(4+1)], a11), STORE128(state->A[(4+2)], a12), STORE128(state->A[(4+3)], a13); \
                        STORE128(state->A[(8+0)], a20), STORE128(state->A[(8+1)], a21), STORE128(state->A[(8+2)], a22), STORE128(state->A[(8+3)], a23)

#define Round(a10i, a11i, a12i, a13i, a10w, a11w, a12w, a13w, a20i, a21i, a22i, a23i, __rc) \
                                                            \
    /* Theta: Column Parity Mixer */                        \
    v1 = XOR128( a03, XOR128( a13i, a23i ) );               \
    v2 = XOR128( a00, XOR128( a10i, a20i ) );               \
    v1 = XOR128( ROL32in128(v1, 5), ROL32in128(v1, 14) );  \
    a00 = XOR128( a00, v1 );                                \
    a10i = XOR128( a10i, v1 );                              \
    a20i = XOR128( a20i, v1 );                              \
    v1 = XOR128( a01, XOR128( a11i, a21i ) );               \
    v2 = XOR128( ROL32in128(v2, 5), ROL32in128(v2, 14) );  \
    a01 = XOR128( a01, v2 );                                \
    a11i = XOR128( a11i, v2 );                              \
    a21i = XOR128( a21i, v2 );                              \
    v2 = XOR128( a02, XOR128( a12i, a22i ) );               \
    v1 = XOR128( ROL32in128(v1, 5), ROL32in128(v1, 14) );  \
    a02 = XOR128( a02, v1 );                                \
    a12i = XOR128( a12i, v1 );                              \
    a22i = XOR128( a22i, v1 );                              \
    v2 = XOR128( ROL32in128(v2, 5), ROL32in128(v2, 14) );  \
    a03 = XOR128( a03, v2 );                                \
    a13i = XOR128( a13i, v2 );                              \
    a23i = XOR128( a23i, v2 );                              \
    /*Dump("Theta");*/                                         \
                                                            \
    /* Rho-west: Plane shift */                             \
    a20i = ROL32in128(a20i, 11);                            \
    a21i = ROL32in128(a21i, 11);                            \
    a22i = ROL32in128(a22i, 11);                            \
    a23i = ROL32in128(a23i, 11);                            \
    /*Dump("Rho-west"); */                                     \
                                                            \
    /* Iota: round constants */                             \
    a00 = XOR128( a00, _mm_set1_epi32( __rc ) );            \
    /*Dump("Iota");*/                                          \
                                                            \
    /* Chi: non linear step, on colums */                   \
    a00 = XOR128( a00, ANDnu128( a10w, a20i ) );            \
    a01 = XOR128( a01, ANDnu128( a11w, a21i ) );            \
    a02 = XOR128( a02, ANDnu128( a12w, a22i ) );            \
    a03 = XOR128( a03, ANDnu128( a13w, a23i ) );            \
    a10w = XOR128( a10w, ANDnu128( a20i, a00 ) );           \
    a11w = XOR128( a11w, ANDnu128( a21i, a01 ) );           \
    a12w = XOR128( a12w, ANDnu128( a22i, a02 ) );           \
    a13w = XOR128( a13w, ANDnu128( a23i, a03 ) );           \
    a20i = XOR128( a20i, ANDnu128( a00, a10w ) );           \
    a21i = XOR128( a21i, ANDnu128( a01, a11w ) );           \
    a22i = XOR128( a22i, ANDnu128( a02, a12w ) );           \
    a23i = XOR128( a23i, ANDnu128( a03, a13w ) );           \
    /*Dump("Chi"); */                                          \
                                                            \
    /* Rho-east: Plane shift */                             \
    a10w = ROL32in128(a10w, 1);                             \
    a11w = ROL32in128(a11w, 1);                             \
    a12w = ROL32in128(a12w, 1);                             \
    a13w = ROL32in128(a13w, 1);                             \
    /* todo!! optimization for ROTL multiple of 8  */       \
    a20i = ROL32in128(a20i, 8);                             \
    a21i = ROL32in128(a21i, 8);                             \
    a22i = ROL32in128(a22i, 8);                             \
    a23i = ROL32in128(a23i, 8);                             \
   /* Dump("Rho-east");*/
    
    
    
    
uint32_t input[48] = {0xB2275BE8,0xB2275BE8,0xB2275BE8,0xB2275BE8,
		      0x6A657F3D,0x6A657F3D,0x6A657F3D,0x6A657F3D,
		      0x636CA821,0x636CA821,0x636CA821,0x636CA821,
		      0xFF7DDEDD,0xFF7DDEDD,0xFF7DDEDD,0xFF7DDEDD,  
			
			0x222CB942,0x222CB942,0x222CB942,0x222CB942,
			0x03019ACC,0x03019ACC,0x03019ACC,0x03019ACC,
			0x32F2D96E,0x32F2D96E,0x32F2D96E,0x32F2D96E,
			0x9B047B36,0x9B047B36,0x9B047B36,0x9B047B36,
			
			0x79CE4436,0x79CE4436,0x79CE4436,0x79CE4436,
			0xE6687CC7,0xE6687CC7,0xE6687CC7,0xE6687CC7,
			0x1B62F7AD,0x1B62F7AD,0x1B62F7AD,0x1B62F7AD,
			0x38D0C129,0x38D0C129,0x38D0C129,0x38D0C129 }; 

#define Twelve_rounds				\
	Round(  a10, a11, a12, a13,    a13, a10, a11, a12,    a20, a21, a22, a23,    _rc1 );			\
    Round(  a13, a10, a11, a12,    a12, a13, a10, a11,    a22, a23, a20, a21,    _rc2 );			\
    Round(  a12, a13, a10, a11,    a11, a12, a13, a10,    a20, a21, a22, a23,    _rc3 );			\
    Round(  a11, a12, a13, a10,    a10, a11, a12, a13,    a22, a23, a20, a21,    _rc4 );			\
    Round(  a10, a11, a12, a13,    a13, a10, a11, a12,    a20, a21, a22, a23,    _rc5 );			\
    Round(  a13, a10, a11, a12,    a12, a13, a10, a11,    a22, a23, a20, a21,    _rc6 );			\
    Round(  a12, a13, a10, a11,    a11, a12, a13, a10,    a20, a21, a22, a23,    _rc7 );			\
    Round(  a11, a12, a13, a10,    a10, a11, a12, a13,    a22, a23, a20, a21,    _rc8 );			\
    Round(  a10, a11, a12, a13,    a13, a10, a11, a12,    a20, a21, a22, a23,    _rc9 );			\
    Round(  a13, a10, a11, a12,    a12, a13, a10, a11,    a22, a23, a20, a21,    _rc10 );			\
    Round(  a12, a13, a10, a11,    a11, a12, a13, a10,    a20, a21, a22, a23,    _rc11 );			\
    Round(  a11, a12, a13, a10,    a10, a11, a12, a13,    a22, a23, a20, a21,    _rc12 );			\
				
			
void main()
{
	Xoodoo_align128plain32_state data;
	Xoodoo_align128plain32_state *state = &data;
	
	state->A[0]=LOAD128(input[0]);
	state->A[1]=LOAD128(input[4]);
	state->A[2]=LOAD128(input[8]);
	state->A[3]=LOAD128(input[12]);
	state->A[4]=LOAD128(input[16]);
	state->A[5]=LOAD128(input[20]);
	state->A[6]=LOAD128(input[24]);
	state->A[7]=LOAD128(input[28]);
	state->A[8]=LOAD128(input[32]);
	state->A[9]=LOAD128(input[36]);
	state->A[10]=LOAD128(input[40]);
	state->A[11]=LOAD128(input[44]);
	
	//V128* states=state->A;
	
	
	
	DeclareVars ;
	State2Vars ;
	Dump("initial");
	
	 //Round(  a10, a11, a12, a13,    a13, a10, a11, a12,    a20, a21, a22, a23,    _rc1 );
	 
	 MEASURE(Twelve_rounds);
	printf("%f,%f,%f",RDTSC_clk_min ,RDTSC_clk_median,RDTSC_clk_max );
	
	Dump("Final");
}			   
