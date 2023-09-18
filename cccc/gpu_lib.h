#pragma once

#ifndef ENABLE_CUDA
#define ENABLE_CUDA
#endif
#ifndef ENABLE_HIP
#define ENABLE_HIP_
#endif

#ifdef ENABLE_CUDA
#include "cublas_v2.h"
#include "cuda_functions.h"
#include "cuda_runtime.h"
#include "cudnn.h"
#if defined(_WIN32)
#include <nvml.h>
#endif
#endif

#if defined(ENABLE_CUDA) && defined(ENABLE_HIP)
#include "hip_runtime_api_part.h"
#endif

#if defined(ENABLE_CUDA) && !defined(ENABLE_HIP)
#include "hip_runtime_type_part.h"
#endif

#if !defined(ENABLE_CUDA) && defined(ENABLE_HIP)
#include "hip/hip_runtime_api.h"
#endif

#ifdef ENABLE_HIP
#include "rocblas/rocblas.h"
#include "hip_functions.h"
#endif

#ifndef AUTO_LOAD_GPU_FUNCTIONS
#define AUTO_LOAD_GPU_FUNCTIONS
#endif

namespace cccc
{
#if defined(_WIN32) && defined(AUTO_LOAD_GPU_FUNCTIONS)
#define IMPORT(func) \
    using func##_t = decltype(&func); \
    extern func##_t func;
#ifdef ENABLE_CUDA
#include "cuda_libs.inc"
#endif
#ifdef ENABLE_HIP
#include "hip_libs.inc"
#endif
#undef IMPORT

#endif

#define IMPORT(func) \
    template <typename... Args> \
    inline int func(Args&&... args) { return 0; }
#ifndef ENABLE_CUDA
#include "cuda_libs.inc"
#endif
#ifndef ENABLE_HIP
#include "hip_libs.inc"
#endif
#undef IMPORT

};    //namespace cccc

//After defined of the function ptrs
#include "cublas_real.h"
#include "rocblas_real.h"