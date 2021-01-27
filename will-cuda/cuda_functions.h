#pragma once

#ifndef REAL_PRECISION
#define REAL_PRECISION 0
#endif

#if REAL_PRECISION == 0
typedef float real_cuda;
#elif REAL_PRECISION == 1
typedef double real_cuda;
#elif REAL_PRECISION == 2
#include "cuda_fp16.h"
typedef half real_cuda;
#endif

/*
    for convenience, all functions use similar parameters CUDA_FUNCTION{X}{Y}{H}, for examples:

    where X refers number of pointers and Y refers number of real parameters

    "H" means declaration, and here the semicolon should be followed at the end of the line

    please read the example to get more information

    add new macros if needed.
*/

#define CUDA_FUNCTION22H(name) int name(real_cuda* p1, real_cuda* p2, unsigned int size, real_cuda a1, real_cuda a2)
#define CUDA_FUNCTION23H(name) int name(real_cuda* p1, real_cuda* p2, unsigned int size, real_cuda a1, real_cuda a2, real_cuda a3)
#define CUDA_FUNCTION32H(name) int name(real_cuda* p1, real_cuda* p2, real_cuda* p3, unsigned int size, real_cuda a1, real_cuda a2)
#define CUDA_FUNCTION33H(name) int name(real_cuda* p1, real_cuda* p2, real_cuda* p3, unsigned int size, real_cuda a1, real_cuda a2, real_cuda a3)
#define CUDA_FUNCTION42H(name) int name(real_cuda* p1, real_cuda* p2, real_cuda* p3, real_cuda* p4, unsigned int size, real_cuda a1, real_cuda a2)
#define CUDA_FUNCTION43H(name) int name(real_cuda* p1, real_cuda* p2, real_cuda* p3, real_cuda* p4, unsigned int size, real_cuda a1, real_cuda a2, real_cuda a3)
#define CUDA_FUNCTION44H(name) int name(real_cuda* p1, real_cuda* p2, real_cuda* p3, real_cuda* p4, unsigned int size, real_cuda a1, real_cuda a2, real_cuda a3, real_cuda a4)

CUDA_FUNCTION22H(cuda_reciprocal);
CUDA_FUNCTION22H(cuda_addnumber);
CUDA_FUNCTION22H(cuda_pow);
CUDA_FUNCTION22H(cuda_sparse);
CUDA_FUNCTION22H(cuda_sign);
CUDA_FUNCTION32H(cuda_cross_entropy);
CUDA_FUNCTION32H(cuda_cross_entropy2);

CUDA_FUNCTION32H(cuda_add);
CUDA_FUNCTION32H(cuda_mul);
CUDA_FUNCTION33H(cuda_div);
CUDA_FUNCTION32H(cuda_sectionlimit);
CUDA_FUNCTION32H(cuda_ada_update);
CUDA_FUNCTION42H(cuda_ada_delta_update);
CUDA_FUNCTION44H(cuda_adam_update);
CUDA_FUNCTION32H(cuda_rms_prop_update);

CUDA_FUNCTION22H(cuda_sin);
CUDA_FUNCTION22H(cuda_cos);

CUDA_FUNCTION22H(cuda_zigzag);
CUDA_FUNCTION42H(cuda_zigzagb);

CUDA_FUNCTION22H(cuda_step);

CUDA_FUNCTION23H(cuda_leaky_relu);
CUDA_FUNCTION43H(cuda_leaky_relub);

#undef CUDA_FUNCTION22H
#undef CUDA_FUNCTION32H
#undef CUDA_FUNCTION33H
#undef CUDA_FUNCTION42H
#undef CUDA_FUNCTION43H
#undef CUDA_FUNCTION44H

#if REAL_PRECISION == 2
//int float2half();
int cuda_half2float(half* p1, float* p2, unsigned int size);
#endif
