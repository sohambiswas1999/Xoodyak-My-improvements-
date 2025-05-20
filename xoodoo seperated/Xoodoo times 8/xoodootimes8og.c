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
    V256 A[12];
} Xoodootimes8_align512SIMD256_states;

typedef Xoodootimes8_align512SIMD256_states Xoodoo_state;

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
	CGAL_ALIGN_64 uint32_t buf[96];			\
	STORE256(buf[0], a00);				\
	STORE256(buf[8*1], a01);				\
	STORE256(buf[8*2], a02);				\
	STORE256(buf[8*3], a03);				\
	STORE256(buf[8*4], a10);				\
	STORE256(buf[8*5], a11);				\
	STORE256(buf[8*6], a12);				\
	STORE256(buf[8*7], a13);				\
	STORE256(buf[8*8], a20);				\
	STORE256(buf[8*9], a21);				\
	STORE256(buf[8*10], a22);				\
	STORE256(buf[8*11], a23);				\
	printf("%s\n",t);				\
	for(int j=0;j<6;j++)				\
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


typedef __m128i V128;
typedef __m256i V256;
typedef __m512i V512;

#define SnP_laneLengthInBytes   4
#define laneIndex(instanceIndex, lanePosition) ((lanePosition)*8 + instanceIndex)

#define Chi(a,b,c)                  _mm256_ternarylogic_epi32(a,b,c,0xD2)

#define LOAD(a) _mm256_load_si256((const V256 *)&(a))

#define CONST8_32(a)                _mm256_set1_epi32(a)
#define LOAD256u(a)                 _mm256_loadu_si256((const V256 *)&(a))

#define LOAD512(a)                  _mm512_load_si512((const V512 *)&(a))
#define LOAD512u(a)                 _mm512_loadu_si512((const V512 *)&(a))

#define LOAD_GATHER8_32(idx,p)      _mm256_i32gather_epi32((const int*)(p), idx, 4)
#define STORE_SCATTER8_32(idx,a,p)  _mm256_i32scatter_epi32((void*)(p), idx, a, 4)
#define LOAD8_32(a,b,c,d,e,f,g,h)   _mm256_setr_epi32(a,b,c,d,e,f,g,h)


#define SHUFFLE_LANES_RIGHT(idx, a) _mm256_permutexvar_epi32(idx, a)

#define ROL32(a, o)                 _mm256_rol_epi32(a, o)
#define SHL32(a, o)                 _mm256_slli_epi32(a, o)

#define SET8_32                     _mm256_setr_epi32

#define STORE128(a, b)              _mm_store_si128((V128 *)&(a), b)
#define STORE128u(a, b)             _mm_storeu_si128((V128 *)&(a), b)
#define STORE256u(a, b)             _mm256_storeu_si256((V256 *)&(a), b)
#define STORE256(a, b)              _mm256_store_si256((V256 *)&(a), b)
#define STORE512(a, b)              _mm512_store_si512((V512 *)&(a), b)
#define STORE512u(a, b)             _mm512_storeu_si512((V512 *)&(a), b)

#define AND(a, b)                   _mm256_and_si256(a, b)
#define XOR128(a, b)                _mm_xor_si128(a, b)
#define XOR256(a, b)                _mm256_xor_si256(a, b)
#define XOR512(a, b)                _mm512_xor_si512(a, b)
#define XOR(a, b)                   XOR256(a, b)
#define XOR3(a,b,c)                 _mm256_ternarylogic_epi32(a,b,c,0x96)


#ifndef _mm256_storeu2_m128i
#define _mm256_storeu2_m128i(hi, lo, a)    _mm_storeu_si128((V128*)(lo), _mm256_castsi256_si128(a)), _mm_storeu_si128((V128*)(hi), _mm256_extracti128_si256(a, 1))
#endif



#define DeclareVars     V256    a00, a01, a02, a03; \
                        V256    a10, a11, a12, a13; \
                        V256    a20, a21, a22, a23; \
                        V256    v1, v2;

#define State2Vars2     a00 = states[0], a01 = states[1], a02 = states[ 2], a03 = states[ 3]; \
                        a12 = states[4], a13 = states[5], a10 = states[ 6], a11 = states[ 7]; \
                        a20 = states[8], a21 = states[9], a22 = states[10], a23 = states[11]

#define State2Vars      a00 = states[0], a01 = states[1], a02 = states[ 2], a03 = states[ 3]; \
                        a10 = states[4], a11 = states[5], a12 = states[ 6], a13 = states[ 7]; \
                        a20 = states[8], a21 = states[9], a22 = states[10], a23 = states[11]

#define Vars2State      states[0] = a00, states[1] = a01, states[ 2] = a02, states[ 3] = a03; \
                        states[4] = a10, states[5] = a11, states[ 6] = a12, states[ 7] = a13; \
                        states[8] = a20, states[9] = a21, states[10] = a22, states[11] = a23

#define Round(a10i, a11i, a12i, a13i, a10w, a11w, a12w, a13w, a20i, a21i, a22i, a23i, __rc) \
                                                            \
    /* Theta: Column Parity Mixer */                        \
    /* Iota: round constants */                             \
    v1 = XOR3( a03, a13i, a23i );                           \
    v2 = XOR3( a00, a10i, a20i );                           \
    v1 = XOR( ROL32(v1, 5), ROL32(v1, 14) );               \
    a00  = XOR3( a00,  v1, CONST8_32(__rc) ); /* Iota */    \
    a10i = XOR( a10i, v1 );                                 \
    a20i = XOR( a20i, v1 );                                 \
    v1 = XOR3( a01, a11i, a21i );                           \
    v2 = XOR( ROL32(v2, 5), ROL32(v2, 14) );               \
    a01  = XOR( a01,  v2 );                                 \
    a11i = XOR( a11i, v2 );                                 \
    a21i = XOR( a21i, v2 );                                 \
    v2 = XOR3( a02, a12i, a22i );                           \
    v1 = XOR( ROL32(v1, 5), ROL32(v1, 14) );               \
    a02  = XOR( a02,  v1 );                                 \
    a12i = XOR( a12i, v1 );                                 \
    a22i = XOR( a22i, v1 );                                 \
    v2 = XOR( ROL32(v2, 5), ROL32(v2, 14) );               \
    a03  = XOR( a03,  v2 );                                 \
    a13i = XOR( a13i, v2 );                                 \
    a23i = XOR( a23i, v2 );                                 \
    /*Dump("Theta");*/                                       \
                                                            \
    /* Rho-west: Plane shift */                             \
    a20i = ROL32(a20i, 11);                                 \
    a21i = ROL32(a21i, 11);                                 \
    a22i = ROL32(a22i, 11);                                 \
    a23i = ROL32(a23i, 11);                                 \
    /*Dump("Rho-west"); */                                   \
                                                            \
    /* Chi: non linear step, on colums */                   \
    a00  = Chi(a00,  a10w, a20i);                           \
    a01  = Chi(a01,  a11w, a21i);                           \
    a02  = Chi(a02,  a12w, a22i);                           \
    a03  = Chi(a03,  a13w, a23i);                           \
    a10w = Chi(a10w, a20i, a00);                            \
    a11w = Chi(a11w, a21i, a01);                            \
    a12w = Chi(a12w, a22i, a02);                            \
    a13w = Chi(a13w, a23i, a03);                            \
    a20i = Chi(a20i, a00,  a10w);                           \
    a21i = Chi(a21i, a01,  a11w);                           \
    a22i = Chi(a22i, a02,  a12w);                           \
    a23i = Chi(a23i, a03,  a13w);                           \
    /*Dump("Chi");*/                                         \
                                                            \
    /* Rho-east: Plane shift */                             \
    a10w = ROL32(a10w, 1);                                  \
    a11w = ROL32(a11w, 1);                                  \
    a12w = ROL32(a12w, 1);                                  \
    a13w = ROL32(a13w, 1);                                  \
    a20i = ROL32(a20i, 8);                                  \
    a21i = ROL32(a21i, 8);                                  \
    a22i = ROL32(a22i, 8);                                  \
    a23i = ROL32(a23i, 8);                                  \
    /*Dump("Rho-east")*/


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
    

uint32_t input[48*2] = {0xB2275BE8,0xB2275BE8,0xB2275BE8,0xB2275BE8,0xB2275BE8,0xB2275BE8,0xB2275BE8,0xB2275BE8,
		      0x6A657F3D,0x6A657F3D,0x6A657F3D,0x6A657F3D,0x6A657F3D,0x6A657F3D,0x6A657F3D,0x6A657F3D,
		      0x636CA821,0x636CA821,0x636CA821,0x636CA821,0x636CA821,0x636CA821,0x636CA821,0x636CA821,
		      0xFF7DDEDD,0xFF7DDEDD,0xFF7DDEDD,0xFF7DDEDD,0xFF7DDEDD,0xFF7DDEDD,0xFF7DDEDD,0xFF7DDEDD,  
			
			0x222CB942,0x222CB942,0x222CB942,0x222CB942,0x222CB942,0x222CB942,0x222CB942,0x222CB942,
			0x03019ACC,0x03019ACC,0x03019ACC,0x03019ACC,0x03019ACC,0x03019ACC,0x03019ACC,0x03019ACC,
			0x32F2D96E,0x32F2D96E,0x32F2D96E,0x32F2D96E,0x32F2D96E,0x32F2D96E,0x32F2D96E,0x32F2D96E,
			0x9B047B36,0x9B047B36,0x9B047B36,0x9B047B36,0x9B047B36,0x9B047B36,0x9B047B36,0x9B047B36,
			
			0x79CE4436,0x79CE4436,0x79CE4436,0x79CE4436,0x79CE4436,0x79CE4436,0x79CE4436,0x79CE4436,
			0xE6687CC7,0xE6687CC7,0xE6687CC7,0xE6687CC7,0xE6687CC7,0xE6687CC7,0xE6687CC7,0xE6687CC7,
			0x1B62F7AD,0x1B62F7AD,0x1B62F7AD,0x1B62F7AD,0x1B62F7AD,0x1B62F7AD,0x1B62F7AD,0x1B62F7AD,
			0x38D0C129,0x38D0C129,0x38D0C129,0x38D0C129,0x38D0C129,0x38D0C129,0x38D0C129,0x38D0C129 }; 

			

    
void main()
{
	Xoodootimes8_align512SIMD256_states data;
	Xoodootimes8_align512SIMD256_states *state = &data;
	
	state->A[0]=LOAD(input[0]);
	state->A[1]=LOAD(input[8*1]);
	state->A[2]=LOAD(input[8*2]);
	state->A[3]=LOAD(input[8*3]);
	state->A[4]=LOAD(input[8*4]);
	state->A[5]=LOAD(input[8*5]);
	state->A[6]=LOAD(input[8*6]);
	state->A[7]=LOAD(input[8*7]);
	state->A[8]=LOAD(input[8*8]);
	state->A[9]=LOAD(input[8*9]);
	state->A[10]=LOAD(input[8*10]);
	state->A[11]=LOAD(input[8*11]);
	
	V256* states=state->A;
	
	
	
	DeclareVars ;
	State2Vars ;
	Dump("initial");
	
	 //Round(  a10, a11, a12, a13,    a13, a10, a11, a12,    a20, a21, a22, a23,    _rc1 );
	 Twelve_rounds;
	 //MEASURE(Twelve_rounds);
	//printf("%f,%f,%f",RDTSC_clk_min ,RDTSC_clk_median,RDTSC_clk_max );
	
	Dump("Final");
}			       
