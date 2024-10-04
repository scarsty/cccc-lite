#pragma once

#include "dll_export.h"
#include <cfloat>
#include <climits>
#include <cstdint>

#define VAR_NAME(a) #a

#define CCCC_NAMESPACE_BEGIN \
    namespace cccc \
    {
#define CCCC_NAMESPACE_END \
    }

namespace cccc
{
using half = float;

//数据类型，因常用于Matrix，为避免重载冲突，使用严格的枚举类型
enum class DataType
{
    FLOAT = 0,
    DOUBLE = 1,
    HALF = 2,
    CURRENT = 65535,
};

//使用设备的类型，主要决定数据位置，同上使用严格的枚举类型
enum class UnitType
{
    CPU = 0,
    GPU,
};

//激活函数种类
//注意如果需要引用CUDNN中的值，必须要按顺序写
enum ActiveFunctionType
{
    ACTIVE_FUNCTION_NONE = -1,
    ACTIVE_FUNCTION_SIGMOID = 0,
    ACTIVE_FUNCTION_RELU = 1,
    ACTIVE_FUNCTION_TANH = 2,
    ACTIVE_FUNCTION_CLIPPED_RELU = 3,
    ACTIVE_FUNCTION_ELU = 4,    //only GPU
    ACTIVE_FUNCTION_SOFTMAX,
    ACTIVE_FUNCTION_SOFTMAX_FAST,
    ACTIVE_FUNCTION_SOFTMAX_LOG,
    ACTIVE_FUNCTION_ABSMAX,
    ACTIVE_FUNCTION_DROPOUT,
    ACTIVE_FUNCTION_RECURRENT,
    ACTIVE_FUNCTION_SOFTPLUS,    //only CPU
    ACTIVE_FUNCTION_LOCAL_RESPONSE_NORMALIZATION,
    ACTIVE_FUNCTION_LOCAL_CONSTRAST_NORMALIZATION,
    ACTIVE_FUNCTION_DIVISIVE_NORMALIZATION,
    ACTIVE_FUNCTION_BATCH_NORMALIZATION,
    ACTIVE_FUNCTION_SPATIAL_TRANSFORMER,
    ACTIVE_FUNCTION_SQUARE,
    ACTIVE_FUNCTION_SUMMAX,
    ACTIVE_FUNCTION_ZERO_CHANNEL,
    ACTIVE_FUNCTION_SIGMOID_CE,    //CE为交叉熵，表示反向时误差原样回传，用于多出口网络，下同
    ACTIVE_FUNCTION_SOFTMAX_CE,
    ACTIVE_FUNCTION_SOFTMAX_FAST_CE,
    ACTIVE_FUNCTION_SIN,
    ACTIVE_FUNCTION_ZIGZAG,
    ACTIVE_FUNCTION_SIN_STEP,
    ACTIVE_FUNCTION_LEAKY_RELU,
    ACTIVE_FUNCTION_SELU,
    ACTIVE_FUNCTION_ABS,
    ACTIVE_FUNCTION_SIN_PLUS,
    ACTIVE_FUNCTION_SILU,
    ACTIVE_FUNCTION_SIGMOID3,    //CE为交叉熵，表示反向时误差原样回传，用于多出口网络，下同
    ACTIVE_FUNCTION_SOFTMAX3,
};

enum ActivePhaseType
{
    ACTIVE_PHASE_TRAIN,        //训练
    ACTIVE_PHASE_TEST,         //测试
    ACTIVE_PHASE_ONLY_TEST,    //仅测试，表示中间结果无需保留
};

//池化种类，与cuDNN直接对应，可以类型转换
enum PoolingType
{
    POOLING_MAX = 0,
    POOLING_AVERAGE_PADDING = 1,
    POOLING_AVERAGE_NOPADDING = 2,
    POOLING_MA,    //实验功能，正向max，反向average
};

//是否反卷积
enum PoolingReverseType
{
    POOLING_NOT_REVERSE = 0,
    POOLING_REVERSE = 1,
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
    LAYER_CONNECTION_DIRECT,          //直连
    LAYER_CONNECTION_CORRELATION,     //相关
    LAYER_CONNECTION_COMBINE,         //合并
    LAYER_CONNECTION_EXTRACT,         //抽取
    LAYER_CONNECTION_ROTATE_EIGEN,    //旋转
    LAYER_CONNECTION_NORM2,           //求出每组数据的模
    LAYER_CONNECTION_TRANSPOSE,       //NCHW2NHWC
    LAYER_CONNECTION_NAC,             //NAC
};

//for net

//初始化权重模式
enum RandomFillType
{
    RANDOM_FILL_CONSTANT,
    RANDOM_FILL_XAVIER,
    RANDOM_FILL_GAUSSIAN,
    RANDOM_FILL_MSRA,
    RANDOM_FILL_LECUN,
};

//调整学习率模式
enum AdjustLearnRateType
{
    ADJUST_LEARN_RATE_FIXED,
    ADJUST_LEARN_RATE_SCALE_INTER,
    ADJUST_LEARN_RATE_LINEAR_INTER,
    ADJUST_LEARN_RATE_STEPS,
    ADJUST_LEARN_RATE_STEPS_WARM,
    ADJUST_LEARN_RATE_STEPS_AUTO,
};

enum BatchNormalizationType
{
    BATCH_NORMALIZATION_PER_ACTIVATION = 0,
    BATCH_NORMALIZATION_SPATIAL = 1,
    BATCH_NORMALIZATION_AUTO,
};

enum RecurrentType
{
    RECURRENT_RELU = 0,
    RECURRENT_TANH = 1,
    RECURRENT_LSTM = 2,
    RECURRENT_GRU = 3,
};

enum RecurrentDirectionType
{
    RECURRENT_DIRECTION_UNI = 0,
    RECURRENT_DIRECTION_BI = 1,
};

enum RecurrentInputType
{
    RECURRENT_INPUT_LINEAR = 0,
    RECURRENT_INPUT_SKIP = 1,
};

enum RecurrentAlgoType
{
    RECURRENT_ALGO_STANDARD = 0,
    RECURRENT_ALGO_PERSIST_STATIC = 1,
    RECURRENT_ALGO_PERSIST_DYNAMIC = 2,
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

struct TestInfo
{
    double accuracy = 0;
    int64_t right = 0, total = 0;
};

}    // namespace cccc