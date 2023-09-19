#pragma once

#ifdef _WIN32
#define DLL_EXPORT extern "C" __declspec(dllexport)
#else
#define DLL_EXPORT extern "C"
#endif

#ifndef REAL_PRECISION
#define REAL_PRECISION 0
#endif

#if REAL_PRECISION == 0
typedef float realc;
#elif REAL_PRECISION == 1
typedef double realc;
#elif REAL_PRECISION == 2
#include "cuda_fp16.h"
typedef half realc;
#endif

/*
    for convenience, all functions use similar parameters CUDA_FUNCTION{X}{Y}{H}, for examples:

    where X refers number of pointers and Y refers number of real parameters

    "H" means declaration, and here the semicolon should be followed at the end of the line

    please read the example to get more information

    add new macros if needed.
*/

#define CUDA_FUNCTION22H(name) DLL_EXPORT int cuda_##name(realc* p1, realc* p2, unsigned int size, realc a1, realc a2)
#define CUDA_FUNCTION23H(name) DLL_EXPORT int cuda_##name(realc* p1, realc* p2, unsigned int size, realc a1, realc a2, realc a3)
#define CUDA_FUNCTION32H(name) DLL_EXPORT int cuda_##name(realc* p1, realc* p2, realc* p3, unsigned int size, realc a1, realc a2)
#define CUDA_FUNCTION33H(name) DLL_EXPORT int cuda_##name(realc* p1, realc* p2, realc* p3, unsigned int size, realc a1, realc a2, realc a3)
#define CUDA_FUNCTION42H(name) DLL_EXPORT int cuda_##name(realc* p1, realc* p2, realc* p3, realc* p4, unsigned int size, realc a1, realc a2)
#define CUDA_FUNCTION43H(name) DLL_EXPORT int cuda_##name(realc* p1, realc* p2, realc* p3, realc* p4, unsigned int size, realc a1, realc a2, realc a3)
#define CUDA_FUNCTION44H(name) DLL_EXPORT int cuda_##name(realc* p1, realc* p2, realc* p3, realc* p4, unsigned int size, realc a1, realc a2, realc a3, realc a4)
#define CUDA_FUNCTION63H(name) DLL_EXPORT int cuda_##name(realc* p1, realc* p2, realc* p3, realc* p4, realc* p5, realc* p6, unsigned int size, realc a1, realc a2, realc a3)

CUDA_FUNCTION22H(reciprocal);
CUDA_FUNCTION22H(addnumber);
CUDA_FUNCTION22H(pow);
CUDA_FUNCTION22H(sparse);
CUDA_FUNCTION22H(sign);
CUDA_FUNCTION32H(cross_entropy);
CUDA_FUNCTION32H(cross_entropy2);

CUDA_FUNCTION32H(add);
CUDA_FUNCTION32H(mul);
CUDA_FUNCTION33H(div);
CUDA_FUNCTION32H(sectionlimit);
CUDA_FUNCTION32H(ada_update);
CUDA_FUNCTION42H(ada_delta_update);
CUDA_FUNCTION44H(adam_update);
CUDA_FUNCTION32H(rms_prop_update);

CUDA_FUNCTION22H(sin);
CUDA_FUNCTION22H(cos);

CUDA_FUNCTION22H(zigzag);
CUDA_FUNCTION42H(zigzagb);

CUDA_FUNCTION22H(step);

CUDA_FUNCTION23H(leaky_relu);
CUDA_FUNCTION43H(leaky_relub);

CUDA_FUNCTION33H(max);
CUDA_FUNCTION63H(maxb);

#if REAL_PRECISION == 2
//int float2half();
int cuda_half2float(half* p1, float* p2, unsigned int size);
#endif
