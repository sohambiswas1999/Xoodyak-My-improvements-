#include <stdio.h>
#include <string.h>
#include <stdint.h>

typedef struct
{
    uint32_t A[12];
} Xoodoo_plain32_state;

typedef Xoodoo_plain32_state Xoodoo_state;

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

#define DeclareVars              \
    uint32_t a00, a01, a02, a03; \
    uint32_t a10, a11, a12, a13; \
    uint32_t a20, a21, a22, a23; \
    uint32_t v1, v2

#define State2Vars                                                                              \
    a00 = state->A[0 + 0], a01 = state->A[0 + 1], a02 = state->A[0 + 2], a03 = state->A[0 + 3]; \
    a10 = state->A[4 + 0], a11 = state->A[4 + 1], a12 = state->A[4 + 2], a13 = state->A[4 + 3]; \
    a20 = state->A[8 + 0], a21 = state->A[8 + 1], a22 = state->A[8 + 2], a23 = state->A[8 + 3]

#define Vars2State                                                                              \
    state->A[0 + 0] = a00, state->A[0 + 1] = a01, state->A[0 + 2] = a02, state->A[0 + 3] = a03; \
    state->A[4 + 0] = a10, state->A[4 + 1] = a11, state->A[4 + 2] = a12, state->A[4 + 3] = a13; \
    state->A[8 + 0] = a20, state->A[8 + 1] = a21, state->A[8 + 2] = a22, state->A[8 + 3] = a23

/*
** Theta: Column Parity Mixer
*/
#define Theta()                          \
    v1 = a03 ^ a13 ^ a23;                \
    v2 = a00 ^ a10 ^ a20;                \
    v1 = ROTL32(v1, 5) ^ ROTL32(v1, 14); \
    a00 ^= v1;                           \
    a10 ^= v1;                           \
    a20 ^= v1;                           \
    v1 = a01 ^ a11 ^ a21;                \
    v2 = ROTL32(v2, 5) ^ ROTL32(v2, 14); \
    a01 ^= v2;                           \
    a11 ^= v2;                           \
    a21 ^= v2;                           \
    v2 = a02 ^ a12 ^ a22;                \
    v1 = ROTL32(v1, 5) ^ ROTL32(v1, 14); \
    a02 ^= v1;                           \
    a12 ^= v1;                           \
    a22 ^= v1;                           \
    v2 = ROTL32(v2, 5) ^ ROTL32(v2, 14); \
    a03 ^= v2;                           \
    a13 ^= v2;                           \
    a23 ^= v2

/*
** Rho-west: Plane shift
*/
#define Rho_west()         \
    a20 = ROTL32(a20, 11); \
    a21 = ROTL32(a21, 11); \
    a22 = ROTL32(a22, 11); \
    a23 = ROTL32(a23, 11); \
    v1 = a13;              \
    a13 = a12;             \
    a12 = a11;             \
    a11 = a10;             \
    a10 = v1

/*
** Iota: Round constants
*/
#define Iota(__rc) a00 ^= __rc

/*
** Chi: Non linear step, on colums
*/
#define Chi()          \
    a00 ^= ~a10 & a20; \
    a10 ^= ~a20 & a00; \
    a20 ^= ~a00 & a10; \
                       \
    a01 ^= ~a11 & a21; \
    a11 ^= ~a21 & a01; \
    a21 ^= ~a01 & a11; \
                       \
    a02 ^= ~a12 & a22; \
    a12 ^= ~a22 & a02; \
    a22 ^= ~a02 & a12; \
                       \
    a03 ^= ~a13 & a23; \
    a13 ^= ~a23 & a03; \
    a23 ^= ~a03 & a13

/*
** Rho-east: Plane shift
*/
#define Rho_east()        \
    a10 = ROTL32(a10, 1); \
    a11 = ROTL32(a11, 1); \
    a12 = ROTL32(a12, 1); \
    a13 = ROTL32(a13, 1); \
    v1 = ROTL32(a23, 8);  \
    a23 = ROTL32(a21, 8); \
    a21 = v1;             \
    v1 = ROTL32(a22, 8);  \
    a22 = ROTL32(a20, 8); \
    a20 = v1

#define Round(__rc)    \
    Theta();           \
    Dump3("Theta");    \
    Rho_west();        \
    Dump3("Rho-west"); \
    Iota(__rc);        \
    Dump3("Iota");     \
    Chi();             \
    Dump3("Chi");      \
    Rho_east();        \
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

void Xoodoo_Initialize(Xoodoo_plain32_state *state)
{
    memset(state, 0, sizeof(Xoodoo_plain32_state));
}

void main()
{
    
}