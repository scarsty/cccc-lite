#pragma once

//即使是没有cuda设备的情况下，仍然需要cudnn.h提供部分常数，故仍需要此头文件
#include "cudnn.h"

//在cudnn的早期版本，或者没有cuda的情况下，对原来的函数进行一些类似hack的处理

#ifdef _NO_CUDA
//在cuda不生效的时候，屏蔽所有使用过的cuda函数
//这个方法不知道好不好

typedef int* CudaHackPointer;

#define Cublas Cblas
#define CUBLAS_STATUS_SUCCESS 0

#define cudaGetDevice (0)
#define cudaSetDevice (0)
#define cudaGetDeviceCount (0)
#define cudaGetDeviceProperties (0)

#define cudaMalloc (0)
#define cudaFree (0)
#define cudaMemcpy (0)
#define cudaMemcpyPeer (0)

#define cuda_addnumber (0)
#define cuda_reciprocal (0)

#define curandGenerator_t CudaHackPointer
#define curandCreateGenerator (0)
#define curandDestroyGenerator (0)
#define curandSetPseudoRandomGeneratorSeed (0)
#define curandGenerateUniform (0)
#define curandGenerateUniformDouble (0)

#define cudnnHandle_t CudaHackPointer
#define cudnnStatus_t int
#define cudaError int
#define cudnnCreate (0)
#define cudnnDestroy (0)

#define cudnnTensorDescriptor_t CudaHackPointer
#define cudnnCreateTensorDescriptor (0)
#define cudnnDestroyTensorDescriptor (0)
#define cudnnSetTensor4dDescriptor (0)
#define cudnnSetTensorNdDescriptor (0)
#define cudnnSetTensor (0)
#define cudnnGetTensor4dDescriptor (0)

#define cudnnPoolingDescriptor_t CudaHackPointer
#define cudnnCreatePoolingDescriptor (0)
#define cudnnDestroyPoolingDescriptor (0)
#define cudnnSetPooling2dDescriptor (0)
#define cudnnSetPoolingNdDescriptor (0)
#define cudnnPoolingForward (0)
#define cudnnPoolingBackward (0)

#define cudnnConvolutionDescriptor_t CudaHackPointer
#define cudnnCreateConvolutionDescriptor (0)
#define cudnnDestroyConvolutionDescriptor (0)
#define cudnnSetConvolutionNdDescriptor (0)
#define cudnnSetConvolutionMathType (0)
#define cudnnSetConvolution2dDescriptor (0)
#define cudnnConvolutionForward (0)
#define cudnnConvolutionBackwardData (0)
#define cudnnConvolutionBackwardFilter (0)
#define cudnnConvolutionBackwardBias (0)

#define cudnnFilterDescriptor_t CudaHackPointer
#define cudnnCreateFilterDescriptor (0)
#define cudnnDestroyFilterDescriptor (0)
#define cudnnSetFilterNdDescriptor (0)
#define cudnnSetFilter4dDescriptor (0)

#define cudnnSoftmaxForward (0)
#define cudnnSoftmaxBackward (0)

#define cudnnActivationDescriptor_t CudaHackPointer
#define cudnnCreateActivationDescriptor (0)
#define cudnnDestroyActivationDescriptor (0)
#define cudnnSetActivationDescriptor (0)
#define cudnnActivationForward (0)
#define cudnnActivationBackward (0)

#define cudnnOpTensorDescriptor_t CudaHackPointer
#define cudnnCreateOpTensorDescriptor (0)
#define cudnnDestroyOpTensorDescriptor (0)
#define cudnnSetOpTensorDescriptor (0)
#define cudnnOpTensor (0)
#define cudnnAddTensor (0)

#define cuda_reciprocal (0)
#define cuda_addnumber (0)
#define cuda_pow (0)
#define cuda_sparse (0)
#define cuda_sign (0)
#define cuda_cross_entropy (0)
#define cuda_cross_entropy2 (0)
#define cuda_add (0)
#define cuda_mul (0)
#define cuda_div (0)
#define cuda_sectionlimit (0)
#define cuda_adam_update (0)
#define cuda_ada_delta_update (0)
#define cuda_rms_prop_update (0)

#endif

//判断cudnn版本2.0实际上只用于TK1，未完成
#if CUDNN_VERSION < 2000 || (defined _NO_CUDA)

#define cudnnRNNDescriptor_t CudaHackPointer
#define cudnnDropoutDescriptor_t CudaHackPointer
#define cudnnSpatialTransformerDescriptor_t CudaHackPointer
#define cudnnLRNDescriptor_t CudaHackPointer

#define cudnnCreateRNNDescriptor (0)
#define cudnnCreateDropoutDescriptor (0)
#define cudnnCreateSpatialTransformerDescriptor (0)
#define cudnnCreateLRNDescriptor (0)

#define cudnnDestroyRNNDescriptor (0)
#define cudnnDestroyDropoutDescriptor (0)
#define cudnnDestroySpatialTransformerDescriptor (0)
#define cudnnDestroyLRNDescriptor (0)

#define cudnnFindConvolutionForwardAlgorithm (0)
#define cudnnFindConvolutionBackwardDataAlgorithm (0)
#define cudnnFindConvolutionBackwardFilterAlgorithm (0)

#define cudnnSetDropoutDescriptor (0)
#define cudnnDropoutForward (0)
#define cudnnDropoutBackward (0)
#define cudnnDropoutGetStatesSize (0)
#define cudnnDropoutGetReserveSpaceSize (0)

#define cudnnDivisiveNormalizationForward (0)
#define cudnnDivisiveNormalizationBackward (0)

#define cudnnBatchNormalizationForwardInference (0)
#define cudnnBatchNormalizationForwardTraining (0)
#define cudnnBatchNormalizationBackward (0)
#define cudnnDeriveBNTensorDescriptor (0)

#define cudnnSetLRNDescriptor (0)
#define cudnnLRNCrossChannelForward (0)
#define cudnnLRNCrossChannelBackward (0)

#define cudnnSetSpatialTransformerNdDescriptor (0)
#define cudnnSpatialTfGridGeneratorForward (0)
#define cudnnSpatialTfSamplerForward (0)
#define cudnnSpatialTfSamplerBackward (0)
#define cudnnSpatialTfGridGeneratorBackward (0)

#define cudnnRNNForwardInference (0)
#define cudnnRNNBackwardData (0)
#define cudnnRNNBackwardWeights (0)

#endif

#if CUDNN_VERSION < 2000

#define CUDNN_NOT_PROPAGATE_NAN 0
#define CUDNN_ACTIVATION_CLIPPED_RELU cudnnActivationMode_t(CUDNN_ACTIVATION_TANH + 1)
#define CUDNN_LRN_CROSS_CHANNEL_DIM1 0
#define CUDNN_SAMPLER_BILINEAR 0
#define CUDNN_DIVNORM_PRECOMPUTED_MEANS 0
#define CUDNN_SOFTMAX_LOG CUDNN_SOFTMAX_FAST

//复制自cudnn5，只是为了使代码可以通过
enum cudnnBatchNormMode_t
{
    CUDNN_BATCHNORM_PER_ACTIVATION = 0,
    CUDNN_BATCHNORM_SPATIAL = 1,
};

enum cudnnRNNMode_t
{
    CUDNN_RNN_RELU = 0,
    CUDNN_RNN_TANH = 1,
    CUDNN_LSTM = 2,
    CUDNN_GRU = 3
};

enum cudnnDirectionMode_t
{
    CUDNN_UNIDIRECTIONAL = 0,
    CUDNN_BIDIRECTIONAL = 1
};

enum cudnnRNNInputMode_t
{
    CUDNN_LINEAR_INPUT = 0,
    CUDNN_SKIP_INPUT = 1
};

enum cudnnOpTensorOp_t
{
    CUDNN_OP_TENSOR_ADD = 0,
    CUDNN_OP_TENSOR_MUL = 1,
    CUDNN_OP_TENSOR_MIN = 2,
    CUDNN_OP_TENSOR_MAX = 3,
};

typedef struct
{
    cudnnConvolutionFwdAlgo_t algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
    cudnnStatus_t status;
    float time;
    size_t memory;
} cudnnConvolutionFwdAlgoPerf_t;

typedef int cudnnConvolutionBwdFilterAlgo_t;
typedef struct
{
    cudnnConvolutionBwdFilterAlgo_t algo;
    cudnnStatus_t status;
    float time;
    size_t memory;
} cudnnConvolutionBwdFilterAlgoPerf_t;

typedef int cudnnConvolutionBwdDataAlgo_t;
typedef struct
{
    cudnnConvolutionBwdDataAlgo_t algo;
    cudnnStatus_t status;
    float time;
    size_t memory;
} cudnnConvolutionBwdDataAlgoPerf_t;

typedef cudnnOpTensorOp_t* cudnnOpTensorDescriptor_t;
typedef cudnnActivationMode_t* cudnnActivationDescriptor_t;

cudnnStatus_t cudnnCreateOpTensorDescriptor(cudnnOpTensorDescriptor_t*);
cudnnStatus_t cudnnDestroyOpTensorDescriptor(cudnnOpTensorDescriptor_t);
cudnnStatus_t cudnnSetOpTensorDescriptor(cudnnOpTensorDescriptor_t, cudnnOpTensorOp_t, cudnnDataType_t, int);
/* Tensor Bias operation : C = op( alpha1 * A, alpha2 * B ) + beta * C */
cudnnStatus_t cudnnOpTensor(cudnnHandle_t, const cudnnOpTensorDescriptor_t, const void*, const cudnnTensorDescriptor_t, const void*, const void*,
    const cudnnTensorDescriptor_t, const void*, const void*, const cudnnTensorDescriptor_t, void*);

cudnnStatus_t cudnnCreateActivationDescriptor(cudnnActivationDescriptor_t*);
cudnnStatus_t cudnnDestroyActivationDescriptor(cudnnActivationDescriptor_t);
cudnnStatus_t cudnnSetActivationDescriptor(cudnnActivationDescriptor_t, cudnnActivationMode_t, int, int);

#define cudnnActivationForward(x1, x2, x3, x4, x5, x6, x7, x8) cudnnActivationForward(x1, *x2, x3, x4, x5, x6, x7, x8)
#define cudnnActivationBackward(x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12) cudnnActivationBackward(x1, *x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12)

#define cudnnSetPooling2dDescriptor(x1, x2, x3, x4, x5, x6, x7, x8, x9) cudnnSetPooling2dDescriptor(x1, x2, x4, x5, x6, x7, x8, x9)

#define cudnnSetFilter4dDescriptor(x1, x2, x3, x4, x5, x6, x7) cudnnSetFilter4dDescriptor(x1, x2, x4, x5, x6, x7)
#define cudnnConvolutionBackwardFilter(x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13) cudnnConvolutionBackwardFilter(x1, x2, x3, x4, x5, x6, x7, x11, x12, x13)
#define cudnnConvolutionBackwardData(x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13) cudnnConvolutionBackwardData(x1, x2, x3, x4, x5, x6, x7, x11, x12, x13)

#endif

#if CUDNN_VERSION < 6000
#define CUDNN_ACTIVATION_ELU 4
#define CUDNN_RNN_ALGO_STANDARD 0
#define CUDNN_RNN_ALGO_PERSIST_STATIC 1
#define CUDNN_RNN_ALGO_PERSIST_DYNAMIC 2
#define cudnnSetConvolution2dDescriptor(x1, x2, x3, x4, x5, x6, x7, x8, x9) cudnnSetConvolution2dDescriptor(x1, x2, x3, x4, x5, x6, x7, x8)
#endif
