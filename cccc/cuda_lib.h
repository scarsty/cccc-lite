#pragma once

#include <cstddef>

#ifndef NO_CUDA
#include "cublas_v2.h"
#include "cuda_runtime.h"
#include "cudnn.h"

#include "cuda_functions.h"
#endif

#ifndef AUTO_CUDA_VERSION
#define AUTO_CUDA_VERSION
#endif

#ifdef NO_CUDA
namespace cccc
{
#define IMPORT(func) \
    template <typename... Args> \
    inline int func(Args&&... args) { return 0; }

enum
{
    cudaSuccess = 0,
    CUDNN_DIM_MAX = 8,

    CUBLAS_STATUS_SUCCESS,
    cudaMemcpyDeviceToDevice,
    cudaMemcpyDeviceToHost,
    cudaMemcpyHostToDevice,

    CUDNN_ACTIVATION_SIGMOID,
    CUDNN_ACTIVATION_TANH,
    CUDNN_ACTIVATION_RELU,
    CUDNN_ACTIVATION_CLIPPED_RELU,
    CUDNN_ACTIVATION_ELU,

    CUDNN_BATCHNORM_PER_ACTIVATION,
    CUDNN_BATCHNORM_SPATIAL,
    CUDNN_BIDIRECTIONAL,
    CUDNN_CROSS_CORRELATION,
    CUDNN_DATA_FLOAT,
    CUDNN_GRU,
    CUDNN_LINEAR_INPUT,
    CUDNN_LRN_CROSS_CHANNEL_DIM1,
    CUDNN_LSTM,
    CUDNN_NOT_PROPAGATE_NAN,
    CUDNN_OP_TENSOR_ADD,
    CUDNN_OP_TENSOR_MUL,
    CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING,
    CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING,
    CUDNN_POOLING_MAX,
    CUDNN_RNN_ALGO_PERSIST_DYNAMIC,
    CUDNN_RNN_ALGO_PERSIST_STATIC,
    CUDNN_RNN_ALGO_STANDARD,
    CUDNN_RNN_RELU,
    CUDNN_RNN_TANH,
    CUDNN_SKIP_INPUT,
    CUDNN_SOFTMAX_ACCURATE,
    CUDNN_SOFTMAX_FAST,
    CUDNN_SOFTMAX_LOG,
    CUDNN_SOFTMAX_MODE_INSTANCE,
    CUDNN_STATUS_SUCCESS,
    CUDNN_TENSOR_NCHW,
    CUDNN_TENSOR_OP_MATH,
    CUDNN_UNIDIRECTIONAL,
};

#define CUDNN_BN_MIN_EPSILON 1e-5

struct Cheat
{
    size_t memory;
    int algo, mathType;
};

using cudaDeviceProp = int;
using cudaError = int;
using cudnnActivationDescriptor_t = void*;
using cudnnActivationMode_t = int;
using cudnnBatchNormMode_t = int;
using cudnnConvolutionBwdDataAlgo_t = int;
using cudnnConvolutionBwdDataAlgoPerf_t = Cheat;
using cudnnConvolutionBwdFilterAlgo_t = int;
using cudnnConvolutionBwdFilterAlgoPerf_t = Cheat;
using cudnnConvolutionDescriptor_t = void*;
using cudnnConvolutionFwdAlgo_t = int;
using cudnnConvolutionFwdAlgo_t = int;
using cudnnConvolutionFwdAlgoPerf_t = Cheat;
using cudnnConvolutionFwdAlgoPerf_t = Cheat;
using cudnnDataType_t = int;
using cudnnDropoutDescriptor_t = void*;
using cudnnFilterDescriptor_t = void*;
using cudnnHandle_t = void*;
using cudnnLRNDescriptor_t = void*;
using cudnnMathType_t = int;
using cudnnOpTensorDescriptor_t = void*;
using cudnnPoolingDescriptor_t = void*;
using cudnnPoolingMode_t = int;
using cudnnRNNDescriptor_t = void*;
using cudnnSpatialTransformerDescriptor_t = void*;
using cudnnStatus_t = int;
using cudnnTensorDescriptor_t = void*;

#define Cublas Cblas

IMPORT(cudaDriverGetVersion)
IMPORT(cudaFree)
IMPORT(cudaGetDevice)
IMPORT(cudaGetDeviceCount)
IMPORT(cudaGetDeviceProperties)
IMPORT(cudaMalloc)
IMPORT(cudaMemcpy)
IMPORT(cudaMemcpyPeer)
IMPORT(cudaMemset)
IMPORT(cudaRuntimeGetVersion)
IMPORT(cudaSetDevice)

IMPORT(cuda_reciprocal);
IMPORT(cuda_addnumber);
IMPORT(cuda_pow);
IMPORT(cuda_sparse);
IMPORT(cuda_sign);
IMPORT(cuda_cross_entropy);
IMPORT(cuda_cross_entropy2);

IMPORT(cuda_add);
IMPORT(cuda_mul);
IMPORT(cuda_div);
IMPORT(cuda_sectionlimit);
IMPORT(cuda_ada_update);
IMPORT(cuda_ada_delta_update);
IMPORT(cuda_adam_update);
IMPORT(cuda_rms_prop_update);

IMPORT(cuda_sin);
IMPORT(cuda_cos);

IMPORT(cuda_zigzag);
IMPORT(cuda_zigzagb);

IMPORT(cuda_step);

IMPORT(cuda_leaky_relu);
IMPORT(cuda_leaky_relub);

#include "cuda_lib.inc"

#undef IMPORT

}    // namespace cccc
#else
#if defined(_WIN32) && defined(AUTO_CUDA_VERSION)
namespace cccc
{
#define IMPORT(func) \
    using func##_t = decltype(&func); \
    extern func##_t func;
#include "cuda_lib.inc"
#undef IMPORT
};    // namespace cccc
#endif
#endif
