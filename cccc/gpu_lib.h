#pragma once

#ifndef ENABLE_CUDA
#define ENABLE_CUDA 1
#endif
#ifndef ENABLE_HIP
#ifdef _WIN32
#define ENABLE_HIP 1
#else
#define ENABLE_HIP 0
#endif
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
#include "hip_type_fake.h"
#endif

#if !(ENABLE_CUDA) && (ENABLE_HIP)
#define __HIP_PLATFORM_AMD__
#define __HIP_DISABLE_CPP_FUNCTIONS__
#include "cuda_type_fake.h"
#include "hip/hip_runtime_api.h"
#endif

#if ENABLE_HIP
#include "hip_functions.h"
#include "miopen/miopen.h"
#include "rocblas/rocblas.h"
#endif

#ifndef _DEBUG
#ifndef AUTO_LOAD_GPU_FUNCTIONS
#define AUTO_LOAD_GPU_FUNCTIONS
#endif
#endif

namespace cccc
{
#if defined(_WIN32) && defined(AUTO_LOAD_GPU_FUNCTIONS)
#define IMPORT(func, ...) \
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

#define IMPORT(func, ...) \
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

#include "types.h"

namespace cccc
{

inline cudnnDataType_t toCudnnDataType(DataType dt)
{
    const cudnnDataType_t t[] = { CUDNN_DATA_FLOAT, CUDNN_DATA_DOUBLE, CUDNN_DATA_HALF };
    return t[int(dt)];
}

inline miopenDataType_t toMiopenDataType(DataType dt)
{
    const miopenDataType_t t[] = { miopenFloat, miopenDouble, miopenHalf };
    return t[int(dt)];
}

#if ENABLE_CUDA
#define CUDNN_DESC(T) \
    class cudnn##T##Desc \
    { \
    private: \
        cudnn##T##Descriptor_t desc_; \
\
    public: \
        cudnn##T##Desc() { cudnnCreate##T##Descriptor(&desc_); } \
        ~cudnn##T##Desc() { cudnnDestroy##T##Descriptor(desc_); } \
        cudnn##T##Descriptor_t operator()() { return (cudnn##T##Descriptor_t)desc_; } \
        cudnn##T##Desc(const cudnn##T##Desc&) = delete; \
        cudnn##T##Desc& operator=(const cudnn##T##Desc&) = delete; \
    };
#endif
#define CUDNN_DESC2(T) \
    class cudnn##T##Desc \
    { \
    private: \
        int desc_[64]{}; \
\
    public: \
        cudnn##T##Descriptor_t operator()() { return (cudnn##T##Descriptor_t)desc_; } \
    };

class cudnnConvolutionDesc
{
private:
    int desc_[64]{};

public:
    cudnnConvolutionDesc()
    {
        desc_[23] = 1;    //此值从cudnn的create结果推断得到，原理不负责
    }

    cudnnConvolutionDescriptor_t operator()() { return (cudnnConvolutionDescriptor_t)desc_; }
};
CUDNN_DESC2(Tensor)
CUDNN_DESC2(Filter)
CUDNN_DESC2(Pooling)
CUDNN_DESC2(Activation)
CUDNN_DESC2(Dropout)
CUDNN_DESC2(OpTensor)
CUDNN_DESC2(LRN)
CUDNN_DESC2(SpatialTransformer)
CUDNN_DESC2(CTCLoss)

//CUDNN_DESC(Convolution);
#if ENABLE_HIP
#define MIOPEN_DESC(T) \
    class miopen##T##Desc \
    { \
    private: \
        miopen##T##Descriptor_t desc_; \
\
    public: \
        miopen##T##Desc() { miopenCreate##T##Descriptor(&desc_); } \
        ~miopen##T##Desc() { miopenDestroy##T##Descriptor(desc_); } \
        miopen##T##Descriptor_t operator()() { return (cudnn##T##Descriptor_t)desc_; } \
        miopen##T##Desc(const miopen##T##Desc&) = delete; \
        miopen##T##Desc& operator=(const miopen##T##Desc&) = delete; \
    };
#endif
#define MIOPEN_DESC2(T) \
    class miopen##T##Desc \
    { \
    private: \
        int desc_[64]{}; \
\
    public: \
        miopen##T##Descriptor_t operator()() { return (miopen##T##Descriptor_t)desc_; } \
    };

MIOPEN_DESC2(Tensor)
MIOPEN_DESC2(Convolution)
MIOPEN_DESC2(Pooling)
MIOPEN_DESC2(Activation)
MIOPEN_DESC2(Dropout)
MIOPEN_DESC2(RNN)
MIOPEN_DESC2(LRN)

void find_gpu_functions();

}    //namespace cccc