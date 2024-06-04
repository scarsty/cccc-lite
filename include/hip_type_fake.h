#pragma once

enum
{
    miopenFloat,
    miopenDouble,
    miopenHalf,
    miopenPoolingMax,
    miopenPoolingAverage,
    miopenPoolingAverageInclusive,
    MIOPEN_SOFTMAX_ACCURATE,
    MIOPEN_SOFTMAX_FAST,
    MIOPEN_SOFTMAX_LOG,
    MIOPEN_SOFTMAX_MODE_INSTANCE,
    miopenActivationLOGISTIC,
    miopenActivationRELU,
    miopenActivationTANH,
    miopenConvolution,

};

#define CUDNN_BN_MIN_EPSILON 1e-5

struct Cheat
{
    size_t memory;
    int algo, mathType;
};

using hipError = int;
using miopenActivationDescriptor_t = void*;
using miopenActivationMode_t = int;
using miopenBatchNormMode_t = int;
using miopenConvolutionBwdDataAlgo_t = int;
using miopenConvolutionBwdDataAlgoPerf_t = Cheat;
using miopenConvolutionBwdFilterAlgo_t = int;
using miopenConvolutionBwdFilterAlgoPerf_t = Cheat;
using miopenConvolutionDescriptor_t = void*;
using miopenConvolutionFwdAlgo_t = int;
using miopenConvolutionFwdAlgo_t = int;
using miopenConvolutionFwdAlgoPerf_t = Cheat;
using miopenConvolutionFwdAlgoPerf_t = Cheat;
using miopenDataType_t = int;
using miopenDropoutDescriptor_t = void*;
using miopenFilterDescriptor_t = void*;
using miopenHandle_t = void*;
using miopenLRNDescriptor_t = void*;
using miopenMathType_t = int;
using miopenOpTensorDescriptor_t = void*;
using miopenPoolingDescriptor_t = void*;
using miopenPoolingMode_t = int;
using miopenRNNDescriptor_t = void*;
using miopenSpatialTransformerDescriptor_t = void*;
using miopenStatus_t = int;
using miopenConvFwdAlgorithm_t = int;
using miopenConvBwdDataAlgorithm_t = int;
using miopenConvBwdWeightsAlgorithm_t = int;

struct miopenTensorStruct
{
};
using miopenTensorDescriptor_t = miopenTensorStruct*;

struct miopenConvAlgoPerf_t
{
    union
    {
        miopenConvFwdAlgorithm_t fwd_algo;
        miopenConvBwdWeightsAlgorithm_t bwd_weights_algo;
        miopenConvBwdDataAlgorithm_t bwd_data_algo;
    };
    float time;
    size_t memory;
};