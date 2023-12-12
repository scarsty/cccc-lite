#pragma once

#ifndef ENABLE_CUDA
#define ENABLE_CUDA 1
#endif
#ifndef ENABLE_HIP
#define ENABLE_HIP 1
#endif

#if ENABLE_CUDA
#include "cuda_runtime_api.h"
#define __CUDA_RUNTIME_H__
#include "cublas_v2.h"
#include "cuda_functions.h"
#include "cudnn.h"
#endif

#if (ENABLE_CUDA) && (ENABLE_HIP)
#include "hip_runtime_api_part.h"
#endif

#if (ENABLE_CUDA) && !(ENABLE_HIP)
#include "hip_runtime_type_part.h"
#endif

#if !(ENABLE_CUDA) && (ENABLE_HIP)
#define __HIP_PLATFORM_AMD__
#define __HIP_DISABLE_CPP_FUNCTIONS__
#include "hip/hip_runtime_api.h"
#include "cuda_type_fake.h"
#endif

#if ENABLE_HIP
#include "rocblas/rocblas.h"
#include "hip_functions.h"
#include "miopen/miopen.h"
#endif

#ifndef _DEBUG
#ifndef AUTO_LOAD_GPU_FUNCTIONS
#define AUTO_LOAD_GPU_FUNCTIONS
#endif
#endif

namespace cccc
{
#if defined(_WIN32) && defined(AUTO_LOAD_GPU_FUNCTIONS)
#define IMPORT(func) \
    using func##_t = decltype(&func); \
    extern func##_t func;
#define UNIMPORT_OR_BLANKTEMP(func)
#if ENABLE_CUDA
#include "cuda_libs.inc"
#endif
#if ENABLE_HIP
#include "hip_libs.inc"
#endif
#undef IMPORT
#undef UNIMPORT_OR_BLANKTEMP
#endif

#define IMPORT(func) \
    template <typename... Args> \
    inline int func(Args&&... args) { return 0; }
#define UNIMPORT_OR_BLANKTEMP(func) IMPORT(func)
#if !ENABLE_CUDA
#include "cuda_libs.inc"
#endif
#if !ENABLE_HIP
#include "hip_libs.inc"
#endif
#undef IMPORT
#undef UNIMPORT_OR_BLANKTEMP

};    //namespace cccc

//After defined of the function ptrs
#include "cublas_real.h"
#include "rocblas_real.h"