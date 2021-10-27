#pragma once

//包含一些类型定义

#include "cuda_lib.h"

#define VAR_NAME(a) #a

//默认使用单精度，整个工程除公用部分，均应当使用real定义浮点数
#ifndef REAL_PRECISION
#define REAL_PRECISION 0
#endif

#define WILL_NAMESPACE_BEGIN \
    namespace will \
    {
#define WILL_NAMESPACE_END }

#ifdef _WINDLL
#define CLASS class __declspec(dllexport)
#else
#define CLASS class
#endif

#if REAL_PRECISION == 2
#include "cuda_half_hack.h"
#endif

#define FMT_HEADER_ONLY
#include "fmt/format.h"
#include "fmt/ranges.h"

namespace cccc
{

//realc的作用：某些半精度函数需要单精度的常数参数，例如池化和卷积
//MYCUDNN_DATA_REAL_C作用类似
#if REAL_PRECISION == 0
using real = float;
using realc = float;
#define REAL_MAX FLT_MAX
#define MYCUDNN_DATA_REAL CUDNN_DATA_FLOAT
#define MYCUDNN_DATA_REAL_C CUDNN_DATA_FLOAT
#elif REAL_PRECISION == 1
using real = double;
using realc = double;
#define REAL_MAX DBL_MAX
#define MYCUDNN_DATA_REAL CUDNN_DATA_DOUBLE
#define MYCUDNN_DATA_REAL_C CUDNN_DATA_DOUBLE
#elif REAL_PRECISION == 2
using real = will_half;
using realc = float;
#define REAL_MAX 65504
#define MYCUDNN_DATA_REAL CUDNN_DATA_HALF
#define MYCUDNN_DATA_REAL_C CUDNN_DATA_FLOAT
#endif

//激活函数种类
//注意如果需要引用CUDNN中的值，必须要按顺序写
enum ActiveFunctionType
{
    ACTIVE_FUNCTION_NONE = -1,
    ACTIVE_FUNCTION_SIGMOID = CUDNN_ACTIVATION_SIGMOID,
    ACTIVE_FUNCTION_RELU = CUDNN_ACTIVATION_RELU,
    ACTIVE_FUNCTION_TANH = CUDNN_ACTIVATION_TANH,
    ACTIVE_FUNCTION_CLIPPED_RELU = CUDNN_ACTIVATION_CLIPPED_RELU,
    ACTIVE_FUNCTION_ELU = CUDNN_ACTIVATION_ELU,    //only GPU
    ACTIVE_FUNCTION_SOFTMAX,
    ACTIVE_FUNCTION_SOFTMAX_FAST,
    ACTIVE_FUNCTION_SOFTMAX_LOG,
    ACTIVE_FUNCTION_ABSMAX,
    ACTIVE_FUNCTION_SIGMOID_CE,    //CE为交叉熵，表示反向时误差原样回传，用于多出口网络，下同
    ACTIVE_FUNCTION_SOFTMAX_CE,
    ACTIVE_FUNCTION_SOFTMAX_FAST_CE,
};

enum ActivePhaseType
{
    ACTIVE_PHASE_TRAIN,    //训练
    ACTIVE_PHASE_TEST,     //测试
};

//池化种类，与cuDNN直接对应，可以类型转换
enum PoolingType
{
    POOLING_MAX = CUDNN_POOLING_MAX,
    POOLING_AVERAGE_PADDING = CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING,
    POOLING_AVERAGE_NOPADDING = CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING,
};

//合并种类
enum CombineType
{
    COMBINE_CONCAT,
    COMBINE_ADD,
};

//代价函数种类
enum CostFunctionType
{
    COST_FUNCTION_RMSE,
    COST_FUNCTION_CROSS_ENTROPY,
};

//for layer
//隐藏，输入，输出
enum LayerVisibleType
{
    LAYER_VISIBLE_HIDDEN,
    LAYER_VISIBLE_IN,
    LAYER_VISIBLE_OUT,
};

//连接类型
enum LayerConnectionType
{
    LAYER_CONNECTION_NONE,            //无连接，用于输入层，不需要特殊设置
    LAYER_CONNECTION_FULLCONNECT,     //全连接
    LAYER_CONNECTION_CONVOLUTION,     //卷积
    LAYER_CONNECTION_POOLING,         //池化
};

//for net

//初始化权重模式
enum RandomFillType
{
    RANDOM_FILL_CONSTANT,
    RANDOM_FILL_XAVIER,
    RANDOM_FILL_GAUSSIAN,
    RANDOM_FILL_MSRA,
};

//调整学习率模式
enum AdjustLearnRateType
{
    ADJUST_LEARN_RATE_FIXED,
    ADJUST_LEARN_RATE_SCALE_INTER,
    ADJUST_LEARN_RATE_LINEAR_INTER,
};

enum BatchNormalizationType
{
    BATCH_NORMALIZATION_PER_ACTIVATION = CUDNN_BATCHNORM_PER_ACTIVATION,
    BATCH_NORMALIZATION_SPATIAL = CUDNN_BATCHNORM_SPATIAL,
    BATCH_NORMALIZATION_AUTO,
};

enum RecurrentType
{
    RECURRENT_RELU = CUDNN_RNN_RELU,
    RECURRENT_TANH = CUDNN_RNN_TANH,
    RECURRENT_LSTM = CUDNN_LSTM,
    RECURRENT_GRU = CUDNN_GRU,
};

enum RecurrentDirectionType
{
    RECURRENT_DIRECTION_UNI = CUDNN_UNIDIRECTIONAL,
    RECURRENT_DIRECTION_BI = CUDNN_BIDIRECTIONAL,
};

enum RecurrentInputType
{
    RECURRENT_INPUT_LINEAR = CUDNN_LINEAR_INPUT,
    RECURRENT_INPUT_SKIP = CUDNN_SKIP_INPUT,
};

enum RecurrentAlgoType
{
    RECURRENT_ALGO_STANDARD = CUDNN_RNN_ALGO_STANDARD,
    RECURRENT_ALGO_PERSIST_STATIC = CUDNN_RNN_ALGO_PERSIST_STATIC,
    RECURRENT_ALGO_PERSIST_DYNAMIC = CUDNN_RNN_ALGO_PERSIST_DYNAMIC,
};

enum SolverType
{
    SOLVER_SGD,
    SOLVER_NAG,
    SOLVER_ADA_DELTA,
    SOLVER_ADAM,
    SOLVER_RMS_PROP,
};

enum WorkModeType
{
    WORK_MODE_NORMAL,
    WORK_MODE_PRUNE,
    WORK_MODE_GAN,
};

enum PruneType
{
    PRUNE_ACTIVE,
    PRUNE_WEIGHT,
};

}    // namespace cccc