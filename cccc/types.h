#pragma once

#include "cccc_export.h"
#include "half.hpp"
#include <cfloat>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>

#define VAR_NAME(a) #a

#define CCCC_NAMESPACE_BEGIN \
    namespace cccc \
    {
#define CCCC_NAMESPACE_END \
    }

namespace cccc
{

// 数据类型，C 风格枚举，与 int 可互转，方便作为函数参数传递
enum DataType
{
    FLOAT = 0,
    DOUBLE = 1,
    HALF = 2,
    BFLOAT16 = 3,
    FP8_E4M3 = 4,    // 1 byte/element; OCP FP8 E4M3 (max=448)
    FP8_E5M2 = 5,    // 1 byte/element; FP8 E5M2 (max=57344)
    FP4_E2M1 = 6,    // 0.5 byte/element packed nibble
    CURRENT = 65535,
};

using half = half_float::half;

// BF16 host type: upper 16 bits of IEEE 754 float32
struct bfloat16
{
    uint16_t bits = 0;
    bfloat16() = default;
    bfloat16(float f)
    {
        uint32_t u;
        std::memcpy(&u, &f, 4);
        bits = (uint16_t)(u >> 16);
    }
    template <typename T>
    bfloat16(T v) :
        bfloat16((float)(v)) {}
    operator float() const
    {
        uint32_t u = (uint32_t)(bits) << 16;
        float f;
        std::memcpy(&f, &u, 4);
        return f;
    }
};

// FP8 E4M3 (OCP FP8 / NVIDIA __nv_fp8_e4m3)
// bias=7; E=15,M=7 is NaN; E=15,M=0..6 are normal (max=448); no Inf
struct fp8_e4m3
{
    uint8_t bits = 0;
    fp8_e4m3() = default;
    explicit fp8_e4m3(float f) :
        bits(from_float(f)) {}
    explicit operator float() const { return to_float(bits); }

private:
    static float to_float(uint8_t b)
    {
        const uint8_t s = (b >> 7) & 1u;
        const int e = (b >> 3) & 0xF;
        const int m = b & 7;
        if (e == 15 && m == 7)
        {
            uint32_t nan = 0x7FC00000u;
            float r;
            std::memcpy(&r, &nan, 4);
            return r;
        }
        if (e == 0) { return (s ? -1.f : 1.f) * m * (1.f / 512.f); }    // subnormal: M/8 * 2^-6 = M * 2^-9
        // normal: reconstruct float32 bits directly
        uint32_t f32 = ((uint32_t)s << 31) | ((uint32_t)(e + 120) << 23) | ((uint32_t)m << 20);
        float r;
        std::memcpy(&r, &f32, 4);
        return r;
    }

    static uint8_t from_float(float f)
    {
        uint32_t u;
        std::memcpy(&u, &f, 4);
        if ((u & 0x7FFFFFFFu) > 0x7F800000u)
        {
            return 0x7Fu;    // NaN → NaN
        }
        const uint32_t s = u >> 31;
        const int fe = (int)((u >> 23) & 0xFFu) - 127;    // unbiased float exp
        const uint32_t fm = u & 0x7FFFFFu;
        if (fe > 8 || (fe == 8 && fm >= 0xC00000u))
        {
            return (uint8_t)((s << 7) | 0x7Eu);    // saturate to 448
        }
        const int e8 = fe + 7;
        if (e8 <= 0)
        {
            const int shift = 14 - fe;    // = 21 - e8
            if (shift >= 32)
            {
                return (uint8_t)(s << 7);
            }
            return (uint8_t)((s << 7) | (((fm | 0x800000u) >> shift) & 0x7u));
        }
        uint8_t m8 = (uint8_t)(fm >> 20);
        if (e8 == 15 && m8 == 7)
        {
            m8 = 6;    // avoid NaN code-point
        }
        return (uint8_t)((s << 7) | ((uint8_t)(e8 & 0xFu) << 3) | m8);
    }

public:
};

// FP8 E5M2 (standard / NVIDIA __nv_fp8_e5m2)
// bias=15; E=31,M=0: ±Inf; E=31,M≠0: NaN; max=57344
struct fp8_e5m2
{
    uint8_t bits = 0;
    fp8_e5m2() = default;
    explicit fp8_e5m2(float f) :
        bits(from_float(f)) {}
    explicit operator float() const { return to_float(bits); }

private:
    static float to_float(uint8_t b)
    {
        const uint8_t s = (b >> 7) & 1u;
        const int e = (b >> 2) & 0x1F;
        const int m = b & 3;
        if (e == 31)
        {
            if (m == 0)
            {
                uint32_t inf = ((uint32_t)s << 31) | 0x7F800000u;
                float r;
                std::memcpy(&r, &inf, 4);
                return r;
            }
            uint32_t nan = 0x7FC00000u;
            float r;
            std::memcpy(&r, &nan, 4);
            return r;
        }
        if (e == 0)
        {
            return (s ? -1.f : 1.f) * m * (1.f / 65536.f);    // subnormal: M/4 * 2^-14 = M * 2^-16
        }
        uint32_t f32 = ((uint32_t)s << 31) | ((uint32_t)(e + 112) << 23) | ((uint32_t)m << 21);
        float r;
        std::memcpy(&r, &f32, 4);
        return r;
    }

    static uint8_t from_float(float f)
    {
        uint32_t u;
        std::memcpy(&u, &f, 4);
        if ((u & 0x7FFFFFFFu) > 0x7F800000u)
        {
            return 0x7Cu;    // NaN
        }
        const uint32_t s = u >> 31;
        if ((u & 0x7FFFFFFFu) == 0x7F800000u)
        {
            return (uint8_t)((s << 7) | 0x7Cu);    // ±Inf
        }
        const int fe = (int)((u >> 23) & 0xFFu) - 127;
        const uint32_t fm = u & 0x7FFFFFu;
        if (fe > 15 || (fe == 15 && fm >= 0xE00000u))
        {
            return (uint8_t)((s << 7) | 0x7Bu);    // saturate to 57344
        }
        const int e8 = fe + 15;
        if (e8 <= 0)
        {
            const int shift = 7 - fe;    // = 22 - e8 approx
            if (shift >= 32)
            {
                return (uint8_t)(s << 7);
            }
            return (uint8_t)((s << 7) | (((fm | 0x800000u) >> shift) & 0x3u));
        }
        const uint8_t m8 = (uint8_t)(fm >> 21);
        return (uint8_t)((s << 7) | ((uint8_t)(e8 & 0x1Fu) << 2) | m8);
    }

public:
};

// FP4 E2M1 (NVIDIA Blackwell NVFP4)
// bias=1; E=0: subnormal (0, ±0.5); E=1..3: normal; no NaN/Inf
// Values: 0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6
// Storage: two nibbles packed per byte (low nibble = even element, high nibble = odd element)
struct fp4_e2m1
{
    static constexpr float kTable[16] = { 0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f,
        0.f, -0.5f, -1.f, -1.5f, -2.f, -3.f, -4.f, -6.f };

    static float to_float(uint8_t nibble)
    {
        return kTable[nibble & 0xFu];
    }

    static uint8_t from_float(float f)    // returns 4-bit nibble in lower 4 bits
    {
        const uint8_t s = (f < 0.f) ? 8u : 0u;
        if (f < 0.f)
        {
            f = -f;
        }
        if (f < 0.25f)
        {
            return s;
        }
        if (f < 0.75f)
        {
            return s | 1u;
        }
        if (f < 1.25f)
        {
            return s | 2u;
        }
        if (f < 1.75f)
        {
            return s | 3u;
        }
        if (f < 2.5f)
        {
            return s | 4u;
        }
        if (f < 3.5f)
        {
            return s | 5u;
        }
        if (f < 5.0f)
        {
            return s | 6u;
        }
        return s | 7u;
    }

    // Pack/unpack helpers for byte-packed buffers
    static float get(const uint8_t* bytes, int i)
    {
        const uint8_t nibble = (i & 1) ? (bytes[i >> 1] >> 4) : (bytes[i >> 1] & 0xFu);
        return to_float(nibble);
    }
    static void set(uint8_t* bytes, int i, float v)
    {
        const uint8_t n = from_float(v);
        if (i & 1)
        {
            bytes[i >> 1] = (uint8_t)((bytes[i >> 1] & 0x0Fu) | (n << 4));
        }
        else
        {
            bytes[i >> 1] = (uint8_t)((bytes[i >> 1] & 0xF0u) | n);
        }
    }
};

// 张量内存布局形式（影响cuDNN卷积descriptor和矩阵物理存储顺序）
enum class TensorForm
{
    NCHW = 0,    // W最快，cccc默认，dim[]={W,H,C,N}
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
    ACTIVE_FUNCTION_BATCH_NORMALIZATION_deprecated,
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
    ACTIVE_FUNCTION_COS,
    ACTIVE_FUNCTION_SIGMOID3,    //CE为交叉熵，表示反向时误差原样回传，用于多出口网络，下同
    ACTIVE_FUNCTION_SOFTMAX3,
    //axis-aware softmax: 沿 row_/channel 划分的"通道"维度做 softmax (对应 cuDNN MODE_CHANNEL)
    //用于 attention 中对每行 (key 维度) 单独归一化的场景
    ACTIVE_FUNCTION_SOFTMAX_CHANNEL,    //语言模型 per-position CE 专用: 前向同 SOFTMAX_CHANNEL (沿 width_ 维归一化)
    //反向直接将上游梯度 Y.d() 透传到 X.d() (与 SOFTMAX_CE 相同), 配合 crossEntropy 使用
    ACTIVE_FUNCTION_SOFTMAX_CHANNEL_CE,
    // Gaussian Error Linear Unit: gelu(x) = 0.5*x*(1+tanh(sqrt(2/pi)*(x+0.044715*x^3)))
    ACTIVE_FUNCTION_GELU,
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
    POOLING_MA,    //实验功能
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
    RANDOM_FILL_KAIMING,
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
    ADJUST_LEARN_RATE_STEPS_WARM2,
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
    double error = 0;
    int64_t right = 0, total = 0;
};

}    // namespace cccc