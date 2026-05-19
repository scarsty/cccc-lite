// HIP 自定义核函数（对应 cuda_functions.cu 的 HIP 移植版本）
// 宏命名约定：HIP_FUNCTIONxy，x=指针参数个数，y=浮点参数个数
// 每个宏自动生成 float/double/half 三种类型的 kernel
// 额外包含 addbias/softmax/pool/conv2d 等手写 kernel（仅 float 版本）

#include "hip_functions.h"
#include <cstring>
#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#define half _Float16
#define bfloat16 __hip_bfloat16

// On Windows HIP builds, the float->bfloat16 truncation function emitted by
// clang for host-side __float2bfloat16 calls (__truncsfbf2) may not be
// supplied by compiler-rt.  We provide a weak definition here so the linker
// can satisfy the reference without requiring a separate runtime library.
extern "C" __attribute__((weak)) unsigned short __truncsfbf2(float f)
{
    unsigned int u;
    std::memcpy(&u, &f, sizeof(u));
    return static_cast<unsigned short>(u >> 16);
}

#define blockMax 1024    // 每个 block 的最大线程数

#define cal_i() (blockIdx.x * blockDim.x + threadIdx.x)    // 计算全局线程索引

inline int blockNum(unsigned int size) { return (size + blockMax - 1) / blockMax; }

inline int getError(const char* content)
{
    hipDeviceSynchronize();
    hipError_t hipStatus = hipGetLastError();
    if (hipStatus != hipSuccess)
    {
        fprintf(stderr, "%s kernel launch failed: %s\n", content, hipGetErrorString(hipStatus));
        return 1;
    }
    return 0;
}

static __global__ void half2floatkernel(half* p1, float* p2, unsigned int size)
{
    int i = cal_i();
    if (i < size)
    {
        p2[i] = __half2float(p1[i]);
    }
}

int hip_half2float(void* p1, void* p2, unsigned int size)
{
    half2floatkernel<<<blockNum(size), blockMax>>>((half*)p1, (float*)p2, size);
    return getError("half2float");
}

// BF16 转换函数：将 uint16 bfloat16 转为 float
// bfloat16: [sign(1) | exp(8) | mantissa(7)] = 16 bits
// 转换策略：bfloat16 -> float 时，把 bf16 作为 float 的高 16 位，低 16 位填 0
static __global__ void bf162floatkernel(bfloat16* p1, float* p2, unsigned int size)
{
    int i = cal_i();
    if (i < size)
    {
        p2[i] = __bfloat162float(p1[i]);
    }
}

int hip_bf162float(void* p1, void* p2, unsigned int size)
{
    bf162floatkernel<<<blockNum(size), blockMax>>>((bfloat16*)p1, (float*)p2, size);
    return getError("bf162float");
}

// CAS-based atomicAdd for bfloat16 (used in backward kernels)
static __device__ void atomicAddBF16(bfloat16* address, float val)
{
    unsigned int* base = (unsigned int*)((uintptr_t)address & ~3u);
    unsigned int shift = (((uintptr_t)address & 2u) ? 16u : 0u);
    unsigned int mask = 0xFFFFu << shift;
    unsigned int old_word = *base;
    unsigned int assumed, new_word;
    do {
        assumed = old_word;
        unsigned short raw = (unsigned short)(assumed >> shift);
        bfloat16 old_bf16;
        __builtin_memcpy(&old_bf16, &raw, 2);
        float new_f = __bfloat162float(old_bf16) + val;
        bfloat16 new_bf16 = __float2bfloat16(new_f);
        unsigned short new_raw;
        __builtin_memcpy(&new_raw, &new_bf16, 2);
        new_word = (assumed & ~mask) | ((unsigned int)new_raw << shift);
        old_word = atomicCAS(base, assumed, new_word);
    } while (old_word != assumed);
}

// FP32 -> BF16: 使用HIP原生转换函数
static __global__ void float2bf16kernel(float* p1, bfloat16* p2, unsigned int size)
{
    int i = cal_i();
    if (i < size)
    {
        p2[i] = __float2bfloat16(p1[i]);
    }
}

int hip_float2bf16(void* p1, void* p2, unsigned int size)
{
    float2bf16kernel<<<blockNum(size), blockMax>>>((float*)p1, (bfloat16*)p2, size);
    return getError("float2bf16");
}

#define PATCH_HALF1(func) \
    inline __device__ half func(half a) { return h##func(a); }
#define PATCH_HALF11(func) \
    inline __device__ half func(half a) { return __h##func(a); }
// PATCH_HALF2: use float path to avoid hlog/hsin ambiguity in ROCm 7.1
#define PATCH_HALF2(func) \
    inline __device__ half func(half a) { return __float2half(func(__half2float(a))); }

// Using PATCH_HALF2 (via float) for all to avoid ROCm 7.1 h* overload ambiguity
PATCH_HALF2(log)
PATCH_HALF2(floor)
PATCH_HALF2(round)
PATCH_HALF2(sin)
PATCH_HALF2(cos)
PATCH_HALF2(sqrt)
// abs and pow need special handling (fabsf/powf)
inline __device__ half abs(half a) { return __float2half(fabsf(__half2float(a))); }
inline __device__ half pow(half a, half b) { return __float2half(powf(__half2float(a), __half2float(b))); }

//inline __device__ half pow(half a, half b) { return __float2half(pow(__half2float(a), __half2float(b))); }

// BF16 math helpers for use inside bf16 kernels.
// Use prefixed names to avoid ambiguity with CUDA math overloads.
__device__ inline bfloat16 bf16m_abs(bfloat16 a) { return __float2bfloat16(fabsf(__bfloat162float(a))); }
__device__ inline bfloat16 bf16m_log(bfloat16 a) { return __float2bfloat16(logf(__bfloat162float(a))); }
__device__ inline bfloat16 bf16m_floor(bfloat16 a) { return __float2bfloat16(floorf(__bfloat162float(a))); }
__device__ inline bfloat16 bf16m_round(bfloat16 a) { return __float2bfloat16(roundf(__bfloat162float(a))); }
__device__ inline bfloat16 bf16m_sin(bfloat16 a) { return __float2bfloat16(sinf(__bfloat162float(a))); }
__device__ inline bfloat16 bf16m_cos(bfloat16 a) { return __float2bfloat16(cosf(__bfloat162float(a))); }
__device__ inline bfloat16 bf16m_sqrt(bfloat16 a) { return __float2bfloat16(sqrtf(__bfloat162float(a))); }
__device__ inline bfloat16 bf16m_pow(bfloat16 a, bfloat16 b) { return __float2bfloat16(powf(__bfloat162float(a), __bfloat162float(b))); }

// Lambda locals shadow global math names inside bf16 kernels.
// Lambda call goes through operator() - no overload resolution, no ambiguity.
#define BF16_MATH_LOCALS \
    auto abs = [](bfloat16 x) -> bfloat16 { \
        return bf16m_abs(x); \
    }; \
    auto log = [](bfloat16 x) -> bfloat16 { \
        return bf16m_log(x); \
    }; \
    auto floor = [](bfloat16 x) -> bfloat16 { \
        return bf16m_floor(x); \
    }; \
    auto round = [](bfloat16 x) -> bfloat16 { \
        return bf16m_round(x); \
    }; \
    auto sin = [](bfloat16 x) -> bfloat16 { \
        return bf16m_sin(x); \
    }; \
    auto cos = [](bfloat16 x) -> bfloat16 { \
        return bf16m_cos(x); \
    }; \
    auto sqrt = [](bfloat16 x) -> bfloat16 { \
        return bf16m_sqrt(x); \
    }; \
    auto pow = [](bfloat16 a, bfloat16 b) -> bfloat16 { \
        return bf16m_pow(a, b); \
    };

#define PATCH_BF162(func)    // no-op

#define HIP_FUNCTION22(name, function) \
    static __global__ void name##kernel##float(float* p1, float* p2, unsigned int size, float a1, float a2) \
    { \
        int i = cal_i(); \
        using fp = float; \
        if (i < size) { function; } \
    } \
    static __global__ void name##kernel##double(double* p1, double* p2, unsigned int size, double a1, double a2) \
    { \
        int i = cal_i(); \
        using fp = double; \
        if (i < size) { function; } \
    } \
    static __global__ void name##kernel##half(half* p1, half* p2, unsigned int size, half a1, half a2) \
    { \
        int i = cal_i(); \
        using fp = half; \
        if (i < size) { function; } \
    } \
    static __global__ void name##kernel##bf16(bfloat16* p1, bfloat16* p2, unsigned int size, float _a1, float _a2) \
    { \
        int i = cal_i(); \
        using fp = bfloat16; \
        BF16_MATH_LOCALS \
        bfloat16 a1 = __float2bfloat16(_a1), a2 = __float2bfloat16(_a2); \
        if (i < size) { function; } \
    } \
    int hip_##name(int type, void* p1, void* p2, unsigned int size, float a1, float a2) \
    { \
        if (type == 0) { name##kernel##float<<<blockNum(size), blockMax>>>((float*)p1, (float*)p2, size, a1, a2); } \
        else if (type == 1) { name##kernel##double<<<blockNum(size), blockMax>>>((double*)p1, (double*)p2, size, a1, a2); } \
        else if (type == 2) { name##kernel##half<<<blockNum(size), blockMax>>>((half*)p1, (half*)p2, size, a1, a2); } \
        else if (type == 3) { name##kernel##bf16<<<blockNum(size), blockMax>>>((bfloat16*)p1, (bfloat16*)p2, size, a1, a2); } \
        return getError(#name); \
    }
#define HIP_FUNCTION23(name, function) \
    static __global__ void name##kernel##float(float* p1, float* p2, unsigned int size, float a1, float a2, float a3) \
    { \
        int i = cal_i(); \
        using fp = float; \
        if (i < size) { function; } \
    } \
    static __global__ void name##kernel##double(double* p1, double* p2, unsigned int size, double a1, double a2, double a3) \
    { \
        int i = cal_i(); \
        using fp = double; \
        if (i < size) { function; } \
    } \
    static __global__ void name##kernel##half(half* p1, half* p2, unsigned int size, half a1, half a2, half a3) \
    { \
        int i = cal_i(); \
        using fp = half; \
        if (i < size) { function; } \
    } \
    static __global__ void name##kernel##bf16(bfloat16* p1, bfloat16* p2, unsigned int size, float _a1, float _a2, float _a3) \
    { \
        int i = cal_i(); \
        using fp = bfloat16; \
        BF16_MATH_LOCALS \
        bfloat16 a1 = __float2bfloat16(_a1), a2 = __float2bfloat16(_a2), a3 = __float2bfloat16(_a3); \
        if (i < size) { function; } \
    } \
    int hip_##name(int type, void* p1, void* p2, unsigned int size, float a1, float a2, float a3) \
    { \
        if (type == 0) { name##kernel##float<<<blockNum(size), blockMax>>>((float*)p1, (float*)p2, size, a1, a2, a3); } \
        else if (type == 1) { name##kernel##double<<<blockNum(size), blockMax>>>((double*)p1, (double*)p2, size, a1, a2, a3); } \
        else if (type == 2) { name##kernel##half<<<blockNum(size), blockMax>>>((half*)p1, (half*)p2, size, a1, a2, a3); } \
        else if (type == 3) { name##kernel##bf16<<<blockNum(size), blockMax>>>((bfloat16*)p1, (bfloat16*)p2, size, a1, a2, a3); } \
        return getError(#name); \
    }
#define HIP_FUNCTION32(name, function) \
    static __global__ void name##kernel##float(float* p1, float* p2, float* p3, unsigned int size, float a1, float a2) \
    { \
        int i = cal_i(); \
        using fp = float; \
        if (i < size) { function; } \
    } \
    static __global__ void name##kernel##double(double* p1, double* p2, double* p3, unsigned int size, double a1, double a2) \
    { \
        int i = cal_i(); \
        using fp = double; \
        if (i < size) { function; } \
    } \
    static __global__ void name##kernel##half(half* p1, half* p2, half* p3, unsigned int size, half a1, half a2) \
    { \
        int i = cal_i(); \
        using fp = half; \
        if (i < size) { function; } \
    } \
    static __global__ void name##kernel##bf16(bfloat16* p1, bfloat16* p2, bfloat16* p3, unsigned int size, float _a1, float _a2) \
    { \
        int i = cal_i(); \
        using fp = bfloat16; \
        BF16_MATH_LOCALS \
        bfloat16 a1 = __float2bfloat16(_a1), a2 = __float2bfloat16(_a2); \
        if (i < size) { function; } \
    } \
    int hip_##name(int type, void* p1, void* p2, void* p3, unsigned int size, float a1, float a2) \
    { \
        if (type == 0) { name##kernel##float<<<blockNum(size), blockMax>>>((float*)p1, (float*)p2, (float*)p3, size, a1, a2); } \
        else if (type == 1) { name##kernel##double<<<blockNum(size), blockMax>>>((double*)p1, (double*)p2, (double*)p3, size, a1, a2); } \
        else if (type == 2) { name##kernel##half<<<blockNum(size), blockMax>>>((half*)p1, (half*)p2, (half*)p3, size, a1, a2); } \
        else if (type == 3) { name##kernel##bf16<<<blockNum(size), blockMax>>>((bfloat16*)p1, (bfloat16*)p2, (bfloat16*)p3, size, a1, a2); } \
        return getError(#name); \
    }
#define HIP_FUNCTION33(name, function) \
    static __global__ void name##kernel##float(float* p1, float* p2, float* p3, unsigned int size, float a1, float a2, float a3) \
    { \
        int i = cal_i(); \
        using fp = float; \
        if (i < size) { function; } \
    } \
    static __global__ void name##kernel##double(double* p1, double* p2, double* p3, unsigned int size, double a1, double a2, double a3) \
    { \
        int i = cal_i(); \
        using fp = double; \
        if (i < size) { function; } \
    } \
    static __global__ void name##kernel##half(half* p1, half* p2, half* p3, unsigned int size, half a1, half a2, half a3) \
    { \
        int i = cal_i(); \
        using fp = half; \
        if (i < size) { function; } \
    } \
    static __global__ void name##kernel##bf16(bfloat16* p1, bfloat16* p2, bfloat16* p3, unsigned int size, float _a1, float _a2, float _a3) \
    { \
        int i = cal_i(); \
        using fp = bfloat16; \
        BF16_MATH_LOCALS \
        bfloat16 a1 = __float2bfloat16(_a1), a2 = __float2bfloat16(_a2), a3 = __float2bfloat16(_a3); \
        if (i < size) { function; } \
    } \
    int hip_##name(int type, void* p1, void* p2, void* p3, unsigned int size, float a1, float a2, float a3) \
    { \
        if (type == 0) { name##kernel##float<<<blockNum(size), blockMax>>>((float*)p1, (float*)p2, (float*)p3, size, a1, a2, a3); } \
        else if (type == 1) { name##kernel##double<<<blockNum(size), blockMax>>>((double*)p1, (double*)p2, (double*)p3, size, a1, a2, a3); } \
        else if (type == 2) { name##kernel##half<<<blockNum(size), blockMax>>>((half*)p1, (half*)p2, (half*)p3, size, a1, a2, a3); } \
        else if (type == 3) { name##kernel##bf16<<<blockNum(size), blockMax>>>((bfloat16*)p1, (bfloat16*)p2, (bfloat16*)p3, size, a1, a2, a3); } \
        return getError(#name); \
    }
#define HIP_FUNCTION42(name, function) \
    static __global__ void name##kernel##float(float* p1, float* p2, float* p3, float* p4, unsigned int size, float a1, float a2) \
    { \
        int i = cal_i(); \
        using fp = float; \
        if (i < size) { function; } \
    } \
    static __global__ void name##kernel##double(double* p1, double* p2, double* p3, double* p4, unsigned int size, double a1, double a2) \
    { \
        int i = cal_i(); \
        using fp = double; \
        if (i < size) { function; } \
    } \
    static __global__ void name##kernel##half(half* p1, half* p2, half* p3, half* p4, unsigned int size, half a1, half a2) \
    { \
        int i = cal_i(); \
        using fp = half; \
        if (i < size) { function; } \
    } \
    static __global__ void name##kernel##bf16(bfloat16* p1, bfloat16* p2, bfloat16* p3, bfloat16* p4, unsigned int size, float _a1, float _a2) \
    { \
        int i = cal_i(); \
        using fp = bfloat16; \
        BF16_MATH_LOCALS \
        bfloat16 a1 = __float2bfloat16(_a1), a2 = __float2bfloat16(_a2); \
        if (i < size) { function; } \
    } \
    int hip_##name(int type, void* p1, void* p2, void* p3, void* p4, unsigned int size, float a1, float a2) \
    { \
        if (type == 0) { name##kernel##float<<<blockNum(size), blockMax>>>((float*)p1, (float*)p2, (float*)p3, (float*)p4, size, a1, a2); } \
        else if (type == 1) { name##kernel##double<<<blockNum(size), blockMax>>>((double*)p1, (double*)p2, (double*)p3, (double*)p4, size, a1, a2); } \
        else if (type == 2) { name##kernel##half<<<blockNum(size), blockMax>>>((half*)p1, (half*)p2, (half*)p3, (half*)p4, size, a1, a2); } \
        else if (type == 3) { name##kernel##bf16<<<blockNum(size), blockMax>>>((bfloat16*)p1, (bfloat16*)p2, (bfloat16*)p3, (bfloat16*)p4, size, a1, a2); } \
        return getError(#name); \
    }
#define HIP_FUNCTION43(name, function) \
    static __global__ void name##kernel##float(float* p1, float* p2, float* p3, float* p4, unsigned int size, float a1, float a2, float a3) \
    { \
        int i = cal_i(); \
        using fp = float; \
        if (i < size) { function; } \
    } \
    static __global__ void name##kernel##double(double* p1, double* p2, double* p3, double* p4, unsigned int size, double a1, double a2, double a3) \
    { \
        int i = cal_i(); \
        using fp = double; \
        if (i < size) { function; } \
    } \
    static __global__ void name##kernel##half(half* p1, half* p2, half* p3, half* p4, unsigned int size, half a1, half a2, half a3) \
    { \
        int i = cal_i(); \
        using fp = half; \
        if (i < size) { function; } \
    } \
    static __global__ void name##kernel##bf16(bfloat16* p1, bfloat16* p2, bfloat16* p3, bfloat16* p4, unsigned int size, float _a1, float _a2, float _a3) \
    { \
        int i = cal_i(); \
        using fp = bfloat16; \
        BF16_MATH_LOCALS \
        bfloat16 a1 = __float2bfloat16(_a1), a2 = __float2bfloat16(_a2), a3 = __float2bfloat16(_a3); \
        if (i < size) { function; } \
    } \
    int hip_##name(int type, void* p1, void* p2, void* p3, void* p4, unsigned int size, float a1, float a2, float a3) \
    { \
        if (type == 0) { name##kernel##float<<<blockNum(size), blockMax>>>((float*)p1, (float*)p2, (float*)p3, (float*)p4, size, a1, a2, a3); } \
        else if (type == 1) { name##kernel##double<<<blockNum(size), blockMax>>>((double*)p1, (double*)p2, (double*)p3, (double*)p4, size, a1, a2, a3); } \
        else if (type == 2) { name##kernel##half<<<blockNum(size), blockMax>>>((half*)p1, (half*)p2, (half*)p3, (half*)p4, size, a1, a2, a3); } \
        else if (type == 3) { name##kernel##bf16<<<blockNum(size), blockMax>>>((bfloat16*)p1, (bfloat16*)p2, (bfloat16*)p3, (bfloat16*)p4, size, a1, a2, a3); } \
        return getError(#name); \
    }
#define HIP_FUNCTION44(name, function) \
    static __global__ void name##kernel##float(float* p1, float* p2, float* p3, float* p4, unsigned int size, float a1, float a2, float a3, float a4) \
    { \
        int i = cal_i(); \
        using fp = float; \
        if (i < size) { function; } \
    } \
    static __global__ void name##kernel##double(double* p1, double* p2, double* p3, double* p4, unsigned int size, double a1, double a2, double a3, double a4) \
    { \
        int i = cal_i(); \
        using fp = double; \
        if (i < size) { function; } \
    } \
    static __global__ void name##kernel##half(half* p1, half* p2, half* p3, half* p4, unsigned int size, half a1, half a2, half a3, half a4) \
    { \
        int i = cal_i(); \
        using fp = half; \
        if (i < size) { function; } \
    } \
    static __global__ void name##kernel##bf16(bfloat16* p1, bfloat16* p2, bfloat16* p3, bfloat16* p4, unsigned int size, float _a1, float _a2, float _a3, float _a4) \
    { \
        int i = cal_i(); \
        using fp = bfloat16; \
        BF16_MATH_LOCALS \
        bfloat16 a1 = __float2bfloat16(_a1), a2 = __float2bfloat16(_a2), a3 = __float2bfloat16(_a3), a4 = __float2bfloat16(_a4); \
        if (i < size) { function; } \
    } \
    int hip_##name(int type, void* p1, void* p2, void* p3, void* p4, unsigned int size, float a1, float a2, float a3, float a4) \
    { \
        if (type == 0) { name##kernel##float<<<blockNum(size), blockMax>>>((float*)p1, (float*)p2, (float*)p3, (float*)p4, size, a1, a2, a3, a4); } \
        else if (type == 1) { name##kernel##double<<<blockNum(size), blockMax>>>((double*)p1, (double*)p2, (double*)p3, (double*)p4, size, a1, a2, a3, a4); } \
        else if (type == 2) { name##kernel##half<<<blockNum(size), blockMax>>>((half*)p1, (half*)p2, (half*)p3, (half*)p4, size, a1, a2, a3, a4); } \
        else if (type == 3) { name##kernel##bf16<<<blockNum(size), blockMax>>>((bfloat16*)p1, (bfloat16*)p2, (bfloat16*)p3, (bfloat16*)p4, size, a1, a2, a3, a4); } \
        return getError(#name); \
    }
#define HIP_FUNCTION63(name, function) \
    static __global__ void name##kernel##float(float* p1, float* p2, float* p3, float* p4, float* p5, float* p6, unsigned int size, float a1, float a2, float a3) \
    { \
        int i = cal_i(); \
        using fp = float; \
        if (i < size) { function; } \
    } \
    static __global__ void name##kernel##double(double* p1, double* p2, double* p3, double* p4, double* p5, double* p6, unsigned int size, double a1, double a2, double a3) \
    { \
        int i = cal_i(); \
        using fp = double; \
        if (i < size) { function; } \
    } \
    static __global__ void name##kernel##half(half* p1, half* p2, half* p3, half* p4, half* p5, half* p6, unsigned int size, half a1, half a2, half a3) \
    { \
        int i = cal_i(); \
        using fp = half; \
        if (i < size) { function; } \
    } \
    static __global__ void name##kernel##bf16(bfloat16* p1, bfloat16* p2, bfloat16* p3, bfloat16* p4, bfloat16* p5, bfloat16* p6, unsigned int size, float _a1, float _a2, float _a3) \
    { \
        int i = cal_i(); \
        using fp = bfloat16; \
        BF16_MATH_LOCALS \
        bfloat16 a1 = __float2bfloat16(_a1), a2 = __float2bfloat16(_a2), a3 = __float2bfloat16(_a3); \
        if (i < size) { function; } \
    } \
    int hip_##name(int type, void* p1, void* p2, void* p3, void* p4, void* p5, void* p6, unsigned int size, float a1, float a2, float a3) \
    { \
        if (type == 0) { name##kernel##float<<<blockNum(size), blockMax>>>((float*)p1, (float*)p2, (float*)p3, (float*)p4, (float*)p5, (float*)p6, size, a1, a2, a3); } \
        else if (type == 1) { name##kernel##double<<<blockNum(size), blockMax>>>((double*)p1, (double*)p2, (double*)p3, (double*)p4, (double*)p5, (double*)p6, size, a1, a2, a3); } \
        else if (type == 2) { name##kernel##half<<<blockNum(size), blockMax>>>((half*)p1, (half*)p2, (half*)p3, (half*)p4, (half*)p5, (half*)p6, size, a1, a2, a3); } \
        else if (type == 3) { name##kernel##bf16<<<blockNum(size), blockMax>>>((bfloat16*)p1, (bfloat16*)p2, (bfloat16*)p3, (bfloat16*)p4, (bfloat16*)p5, (bfloat16*)p6, size, a1, a2, a3); } \
        return getError(#name); \
    }

// Macro to add BF16 support to existing 2-pointer functions (HIP_FUNCTION22 extension)
#define HIP_FUNCTION22_BF16(name, function) \
    static __global__ void name##kernel##bf16(bfloat16* p1, bfloat16* p2, unsigned int size, float _a1, float _a2) \
    { \
        int i = cal_i(); \
        using fp = bfloat16; \
        BF16_MATH_LOCALS \
        bfloat16 a1 = __float2bfloat16(_a1), a2 = __float2bfloat16(_a2); \
        if (i < size) { function; } \
    }

// Macro to add BF16 support to existing 3-pointer functions (HIP_FUNCTION32 extension)
#define HIP_FUNCTION32_BF16(name, function) \
    static __global__ void name##kernel##bf16(bfloat16* p1, bfloat16* p2, bfloat16* p3, unsigned int size, float _a1, float _a2) \
    { \
        int i = cal_i(); \
        using fp = bfloat16; \
        BF16_MATH_LOCALS \
        bfloat16 a1 = __float2bfloat16(_a1), a2 = __float2bfloat16(_a2); \
        if (i < size) { function; } \
    }

// Extend dispatcher function for HIP_FUNCTION32 to include BF16
#define HIP_FUNCTION32_ADD_BF16(name) \
    static int hip_##name##_orig(int type, void* p1, void* p2, void* p3, unsigned int size, float a1, float a2); \
    int hip_##name(int type, void* p1, void* p2, void* p3, unsigned int size, float a1, float a2) \
    { \
        if (type == 3) \
        { \
            name##kernel##bf16<<<blockNum(size), blockMax>>>((bfloat16*)p1, (bfloat16*)p2, (bfloat16*)p3, size, a1, a2); \
            return getError(#name); \
        } \
        return hip_##name##_orig(type, p1, p2, p3, size, a1, a2); \
    }

// R = scale / (A + epsilon)
HIP_FUNCTION22(reciprocal,
    {
        p2[i] = a1 / (p1[i] + a2);
    });

// R = number + A * scale
HIP_FUNCTION22(addnumber, { p2[i] = a1 + p1[i] * a2; });

// R = |A + bias|^e × sign(A + bias)  —— 幂函数，对负数取绝对值后恢复符号
HIP_FUNCTION22(pow,
    {
        // 与 CUDA 版本对齐：对负数取绝对值后求幂，再恢复符号
        p2[i] = pow(abs(p1[i] + a2), a1);
        if (p1[i] < -a2)
        {
            p2[i] *= -1;
        }
    });

// 稀疏惩罚梯度（KL散度的导数）
HIP_FUNCTION22(sparse,
    {
        p2[i] = ((fp(1) - a1) / (fp(1) - p1[i]) - a1 / p1[i]) * a2;
    });

HIP_FUNCTION22(sign,
    {
        if (p1[i] > a2)
        {
            p2[i] = a1;
            return;
        }
        if (p1[i] < -a2)
        {
            p2[i] = -a1;
            return;
        }
        p2[i] = 0;
    });

HIP_FUNCTION32(cross_entropy, { p3[i] = -a2 * p2[i] * log(p1[i] + a1); });

HIP_FUNCTION32(cross_entropy2, { p3[i] = -a2 * (p2[i] * log(p1[i] + a1) + (fp(1) - p2[i]) * log(fp(1) - p1[i] + a1)); });

HIP_FUNCTION32(add, { p3[i] = p1[i] * a1 + p2[i] * a2; });

HIP_FUNCTION32(mul, { p3[i] = p1[i] * p2[i] * a1 + p3[i] * a2; });

HIP_FUNCTION33(div, { p3[i] = a3 * (p1[i] + a1) / (p2[i] + a2); });

HIP_FUNCTION32(sectionlimit,
    {
        if (p3 != p1)
        {
            p3[i] = p1[i];
        }
        if (p3[i] < a1)
        {
            p3[i] = a1;
        }
        if (p3[i] > a2)
        {
            p3[i] = a2;
        }
    });

// AdaGrad 更新
HIP_FUNCTION32(ada_update,
    {
        fp& rou = a1;
        fp& epsilon = a2;
        p2[i] = p2[i] * rou + p3[i] * p3[i] * (fp(1) - rou);
        p3[i] = p3[i] * sqrt((p1[i] + epsilon) / (p2[i] + epsilon));
        p1[i] = p1[i] * rou + p3[i] * p3[i] * (fp(1) - rou);
    });

// AdaDelta 更新
HIP_FUNCTION42(ada_delta_update,
    {
        fp& rou = a1;
        fp& epsilon = a2;
        p1[i] = p1[i] * rou + p3[i] * p3[i] * (fp(1) - rou);
        p4[i] = p3[i] * sqrt((p2[i] + epsilon) / (p1[i] + epsilon));
        p2[i] = p2[i] * rou + p4[i] * p4[i] * (fp(1) - rou);
    });

// Adam 优化器更新
HIP_FUNCTION44(adam_update,
    {
        fp& beta1 = a1;
        fp& beta2 = a2;
        fp& epsilon = a3;
        fp& t = a4;
        p1[i] = p1[i] * beta1 + p3[i] * (fp(1) - beta1);
        p2[i] = p2[i] * beta2 + p3[i] * p3[i] * (fp(1) - beta2);
        p4[i] = p1[i] / (fp(1) - pow(beta1, t)) / (sqrt(p2[i] / (fp(1) - pow(beta2, t))) + epsilon);
    });

// RMSProp 更新
HIP_FUNCTION32(rms_prop_update,
    {
        fp& rou = a1;
        fp& epsilon = a2;
        p1[i] = p1[i] * rou + p2[i] * p2[i] * (fp(1) - rou);
        p3[i] = p2[i] / sqrt(p1[i] + epsilon);
    });

HIP_FUNCTION22(sin,
    {
        p2[i] = sin(a1 * p1[i] + a2);
    });

HIP_FUNCTION22(cos,
    {
        p2[i] = cos(a1 * p1[i] + a2);
    });

HIP_FUNCTION22(zigzag,
    {
        p2[i] = a1 * (p1[i] + a2 - fp(2) * floor((p1[i] + a2 - fp(1)) / fp(2)) - fp(2));
    });

HIP_FUNCTION42(zigzagb,
    {
        if (abs(p1[i]) > fp(1 - 1e-2))
        {
            p2[i] = -p4[i] * fp(100);
            return;
        }
        p2[i] = p4[i];
    });

HIP_FUNCTION22(step,
    {
        p2[i] = round(p1[i] * fp(256)) / fp(256);
    });

HIP_FUNCTION23(leaky_relu,
    {
        if (p1[i] >= fp(0))
        {
            p2[i] = p1[i] * a2 + p2[i] * a3;
        }
        else
        {
            p2[i] = p1[i] * a1 * a2 + p2[i] * a3;
        }
    });

HIP_FUNCTION43(leaky_relub,
    {
        if (p1[i] >= fp(0))
        {
            p2[i] = p4[i] * a2 + p2[i] * a3;
        }
        else
        {
            p2[i] = p4[i] * a1 * a2 + p2[i] * a3;
        }
    });

HIP_FUNCTION33(max,
    {
        p3[i] = p1[i] > p2[i] ? p1[i] : p2[i];
    });
HIP_FUNCTION63(maxb,
    {
        if (p1[i] == p5[i])
        {
            p2[i] = a3 * p6[i] + a1 * p2[i];
            p4[i] = a2 * p4[i];
        }
        else
        {
            p2[i] = a1 * p2[i];
            p4[i] = a3 * p6[i] + a2 * p4[i];
        }
    });

HIP_FUNCTION32(zero_limit,
    {
        if (p2[i] > a1)
        {
            p3[i] = 0;
        }
        else
        {
            p3[i] = p1[i] - p2[i];
        }
    });

// ====== 以下为 HIP 特有的手写 kernel（仅 float），用于不依赖 MIOpen 的场景 ======

// 加偏置（前向）：r[i] = a2*r[i] + a1*b[channel_of(i)]
static __global__ void addbiaskernel(float* m, float* b, float* r, unsigned int size_m, unsigned int size_mchannel, unsigned int size_b, float a1, float a2)
{
    int i = cal_i();
    if (i < size_m)
    {
        r[i] *= a2;
        r[i] += a1 * b[i / size_mchannel % size_b];
    }
}

int hip_addbias(float* m, float* b, float* r, unsigned int size_m, unsigned int size_mchannel, unsigned int size_b, float a1, float a2)
{
    addbiaskernel<<<blockNum(size_m), blockMax>>>(m, b, r, size_m, size_mchannel, size_b, a1, a2);
    return getError("addbias");
}

static __global__ void addbiaskernel_bf16(bfloat16* r, bfloat16* b, unsigned int size_m, unsigned int size_mchannel, unsigned int size_b, float a1, float a2)
{
    int i = cal_i();
    if (i < size_m)
    {
        float ri = __bfloat162float(r[i]);
        float bi = __bfloat162float(b[i / size_mchannel % size_b]);
        r[i] = __float2bfloat16(ri * a2 + a1 * bi);
    }
}

int hip_addbias_bf16(void* r, void* b, unsigned int size_m, unsigned int size_mchannel, unsigned int size_b, float a1, float a2)
{
    addbiaskernel_bf16<<<blockNum(size_m), blockMax>>>((bfloat16*)r, (bfloat16*)b, size_m, size_mchannel, size_b, a1, a2);
    return getError("addbias_bf16");
}

// BF16 activation kernels (bypass MIOpen which may not support bfloat16)
static __global__ void sigmoid_bf16_kernel(bfloat16* output, const bfloat16* input, unsigned int size, float alpha, float beta)
{
    int i = cal_i();
    if (i < size)
    {
        float x = __bfloat162float(input[i]);
        float sig = 1.0f / (1.0f + expf(-x));
        float prev = (beta != 0.0f) ? __bfloat162float(output[i]) : 0.0f;
        output[i] = __float2bfloat16(alpha * sig + beta * prev);
    }
}
int hip_sigmoid_bf16(void* output, const void* input, unsigned int size, float alpha, float beta)
{
    sigmoid_bf16_kernel<<<blockNum(size), blockMax>>>((bfloat16*)output, (const bfloat16*)input, size, alpha, beta);
    return getError("sigmoid_bf16");
}

static __global__ void relu_bf16_kernel(bfloat16* output, const bfloat16* input, unsigned int size, float alpha, float beta)
{
    int i = cal_i();
    if (i < size)
    {
        float x = __bfloat162float(input[i]);
        float relu_x = x > 0.0f ? x : 0.0f;
        float prev = (beta != 0.0f) ? __bfloat162float(output[i]) : 0.0f;
        output[i] = __float2bfloat16(alpha * relu_x + beta * prev);
    }
}
int hip_relu_bf16(void* output, const void* input, unsigned int size, float alpha, float beta)
{
    relu_bf16_kernel<<<blockNum(size), blockMax>>>((bfloat16*)output, (const bfloat16*)input, size, alpha, beta);
    return getError("relu_bf16");
}

static __global__ void tanh_bf16_kernel(bfloat16* output, const bfloat16* input, unsigned int size, float alpha, float beta)
{
    int i = cal_i();
    if (i < size)
    {
        float x = __bfloat162float(input[i]);
        float tanh_x = tanhf(x);
        float prev = (beta != 0.0f) ? __bfloat162float(output[i]) : 0.0f;
        output[i] = __float2bfloat16(alpha * tanh_x + beta * prev);
    }
}
int hip_tanh_bf16(void* output, const void* input, unsigned int size, float alpha, float beta)
{
    tanh_bf16_kernel<<<blockNum(size), blockMax>>>((bfloat16*)output, (const bfloat16*)input, size, alpha, beta);
    return getError("tanh_bf16");
}

// BF16 softmax: Y = a * softmax(X) + r * Y
// Each group of group_size elements is softmax'd independently.
// Layout: group g occupies X[g*group_size .. g*group_size+group_size-1]
static __global__ void softmax_bf16_kernel(
    bfloat16* Y, const bfloat16* X, unsigned int group_size, unsigned int num_groups, float a, float r)
{
    unsigned int g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= num_groups)
    {
        return;
    }
    const bfloat16* x = X + g * group_size;
    bfloat16* y = Y + g * group_size;
    // find max for numerical stability
    float maxv = __bfloat162float(x[0]);
    for (unsigned int i = 1; i < group_size; i++)
    {
        maxv = fmaxf(maxv, __bfloat162float(x[i]));
    }
    // sum of exp
    float sum = 0.f;
    for (unsigned int i = 0; i < group_size; i++)
    {
        sum += expf(__bfloat162float(x[i]) - maxv);
    }
    float inv_sum = (sum > 0.f) ? (a / sum) : 0.f;
    // write output
    for (unsigned int i = 0; i < group_size; i++)
    {
        y[i] = __float2bfloat16(expf(__bfloat162float(x[i]) - maxv) * inv_sum + r * __bfloat162float(y[i]));
    }
}

int hip_softmax_bf16(void* Y, const void* X, unsigned int group_size, unsigned int num_groups, float a, float r)
{
    softmax_bf16_kernel<<<blockNum(num_groups), blockMax>>>((bfloat16*)Y, (const bfloat16*)X, group_size, num_groups, a, r);
    return getError("softmax_bf16");
}

// BF16 log-softmax: Y = log(softmax(X)) in place
static __global__ void log_softmax_bf16_kernel(
    bfloat16* Y, const bfloat16* X, unsigned int group_size, unsigned int num_groups, float a, float r)
{
    unsigned int g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= num_groups)
    {
        return;
    }
    const bfloat16* x = X + g * group_size;
    bfloat16* y = Y + g * group_size;
    float maxv = __bfloat162float(x[0]);
    for (unsigned int i = 1; i < group_size; i++)
    {
        maxv = fmaxf(maxv, __bfloat162float(x[i]));
    }
    float sum = 0.f;
    for (unsigned int i = 0; i < group_size; i++)
    {
        sum += expf(__bfloat162float(x[i]) - maxv);
    }
    float log_sum = logf(sum > 0.f ? sum : 1e-30f);
    for (unsigned int i = 0; i < group_size; i++)
    {
        y[i] = __float2bfloat16(a * (__bfloat162float(x[i]) - maxv - log_sum) + r * __bfloat162float(y[i]));
    }
}

int hip_log_softmax_bf16(void* Y, const void* X, unsigned int group_size, unsigned int num_groups, float a, float r)
{
    log_softmax_bf16_kernel<<<blockNum(num_groups), blockMax>>>((bfloat16*)Y, (const bfloat16*)X, group_size, num_groups, a, r);
    return getError("log_softmax_bf16");
}

// 加偏置（反向）：将梯度汇聚到偏置向量
static __global__ void addbiasbkernel(float* bd, float* rd, unsigned int size_m, unsigned int size_mchannel, unsigned int size_b, float a1, float a2)
{
    int i = cal_i();
    if (i < size_b)
    {
        bd[i] *= a2;
        for (int i1 = 0; i1 < size_m / size_mchannel / size_b; i1++)
        {
            for (int i2 = 0; i2 < size_mchannel; i2++)
            {
                bd[i] += a1 * rd[i2 + i1 * size_b * size_mchannel + i * size_mchannel];
            }
        }
    }
}

int hip_addbiasb(float* bd, float* rd, unsigned int size_m, unsigned int size_mchannel, unsigned int size_b, float a1, float a2)
{
    addbiasbkernel<<<blockNum(size_b), blockMax>>>(bd, rd, size_m, size_mchannel, size_b, a1, a2);
    return getError("addbiasb");
}

// --- element-wise add: R = a*A + b*B + r*R ---
static __global__ void elementwise_add_float_kernel(const float* A, const float* B, float* R,
    unsigned int size, float a, float b, float ar)
{
    int i = cal_i();
    if (i < (int)size)
    {
        R[i] = a * A[i] + b * B[i] + ar * R[i];
    }
}

static __global__ void elementwise_add_half_kernel(const _Float16* A, const _Float16* B, _Float16* R,
    unsigned int size, float a, float b, float ar)
{
    int i = cal_i();
    if (i < (int)size)
    {
        float val = a * __half2float(A[i]) + b * __half2float(B[i]);
        if (ar != 0.0f)
        {
            val += ar * __half2float(R[i]);
        }
        R[i] = __float2half(val);
    }
}

static __global__ void elementwise_add_bf16_kernel(const bfloat16* A, const bfloat16* B, bfloat16* R,
    unsigned int size, float a, float b, float ar)
{
    int i = cal_i();
    if (i < (int)size)
    {
        float val = a * __bfloat162float(A[i]) + b * __bfloat162float(B[i]);
        if (ar != 0.0f)
        {
            val += ar * __bfloat162float(R[i]);
        }
        R[i] = __float2bfloat16(val);
    }
}

int hip_elementwise_add(int type, void* A, void* B, void* R, unsigned int size, float a, float b, float ar)
{
    if (type == 2)
    {
        elementwise_add_half_kernel<<<blockNum(size), blockMax>>>((const _Float16*)A, (const _Float16*)B, (_Float16*)R, size, a, b, ar);
    }
    else if (type == 3)
    {
        elementwise_add_bf16_kernel<<<blockNum(size), blockMax>>>((const bfloat16*)A, (const bfloat16*)B, (bfloat16*)R, size, a, b, ar);
    }
    else
    {
        elementwise_add_float_kernel<<<blockNum(size), blockMax>>>((const float*)A, (const float*)B, (float*)R, size, a, b, ar);
    }
    return getError("elementwise_add");
}

// --- element-wise multiply: R = a*A*B + r*R ---
static __global__ void elementwise_mul_float_kernel(const float* A, const float* B, float* R,
    unsigned int size, unsigned int b_size, float a, float ar)
{
    int i = cal_i();
    if (i < (int)size)
    {
        R[i] = a * A[i] * B[i % b_size] + ar * R[i];
    }
}

static __global__ void elementwise_mul_half_kernel(const _Float16* A, const _Float16* B, _Float16* R,
    unsigned int size, unsigned int b_size, float a, float ar)
{
    int i = cal_i();
    if (i < (int)size)
    {
        float val = a * __half2float(A[i]) * __half2float(B[i % b_size]);
        if (ar != 0.0f)
        {
            val += ar * __half2float(R[i]);
        }
        R[i] = __float2half(val);
    }
}

static __global__ void elementwise_mul_bf16_kernel(const bfloat16* A, const bfloat16* B, bfloat16* R,
    unsigned int size, unsigned int b_size, float a, float ar)
{
    int i = cal_i();
    if (i < (int)size)
    {
        float val = a * __bfloat162float(A[i]) * __bfloat162float(B[i % b_size]);
        if (ar != 0.0f)
        {
            val += ar * __bfloat162float(R[i]);
        }
        R[i] = __float2bfloat16(val);
    }
}

int hip_elementwise_mul(int type, void* A, void* B, void* R, unsigned int size, unsigned int b_size, float a, float ar)
{
    if (type == 2)
    {
        elementwise_mul_half_kernel<<<blockNum(size), blockMax>>>((const _Float16*)A, (const _Float16*)B, (_Float16*)R, size, b_size, a, ar);
    }
    else if (type == 3)
    {
        elementwise_mul_bf16_kernel<<<blockNum(size), blockMax>>>((const bfloat16*)A, (const bfloat16*)B, (bfloat16*)R, size, b_size, a, ar);
    }
    else
    {
        elementwise_mul_float_kernel<<<blockNum(size), blockMax>>>((const float*)A, (const float*)B, (float*)R, size, b_size, a, ar);
    }
    return getError("elementwise_mul");
}

// 2D 池化前向：仅支持正方形窗口、stride=窗口大小、无padding
// type=0: 最大池化, type=1: 平均池化
static __global__ void poolkernel(float* x, float* y, unsigned int w0, unsigned w0h0, unsigned int w1, unsigned int w1h1, unsigned int size_win, unsigned int size, int type, float a1, float a2)
{
    int i = cal_i();
    if (i < size)
    {
        int n = i / w1h1;
        int x1 = i % w1h1 / w1;
        int y1 = i % w1;
        int x0 = x1 * size_win;
        int y0 = y1 * size_win;
        float v = type == 0 ? -1e8 : 0;
        for (int ix = x0; ix < x0 + size_win; ix++)
        {
            for (int iy = y0; iy < y0 + size_win; iy++)
            {
                int index = n * w0h0 + ix * w0 + iy;
                if (type == 0 && x[index] > v)
                {
                    v = x[index];
                }
                else if (type == 1)
                {
                    v += x[index] / size_win / size_win;
                }
            }
        }
        y[i] = a1 * v + a2 * y[i];
    }
}

int hip_pool(float* x, float* y, unsigned int w0, unsigned int h0, unsigned int c, unsigned n, unsigned int w1, unsigned int h1, unsigned int size_win, int type, float a1, float a2)
{
    poolkernel<<<blockNum(w1 * h1 * c * n), blockMax>>>(x, y, w0, w0 * h0, w1, w1 * h1, size_win, w1 * h1 * c * n, type, a1, a2);
    return getError("pool");
}

// 2D 池化反向
static __global__ void poolbkernel(float* x, float* dx, float* y, float* dy, unsigned int w0, unsigned w0h0, unsigned int w1, unsigned int w1h1, unsigned int size_win, unsigned int size, int type, float a1, float a2)
{
    int i = cal_i();
    if (i < size)
    {
        int n = i / w0h0;
        int x0 = i % w0h0 / w0;
        int y0 = i % w0;
        int x1 = x0 / size_win;
        int y1 = y0 / size_win;
        int index = n * w1h1 + x1 * w1 + y1;
        if (type == 0)
        {
            if (x[i] == y[index])
            {
                dx[i] = a1 * dy[index] + a2 * dx[i];
            }
            else
            {
                dx[i] = a2 * dx[i];
            }
        }
        else
        {
            dx[i] = a1 * dy[index] / size_win / size_win + a2 * dx[i];
        }
    }
}

int hip_poolb(float* x, float* dx, float* y, float* dy, unsigned int w0, unsigned int h0, unsigned int c, unsigned n, unsigned int w1, unsigned int h1, unsigned int size_win, int type, float a1, float a2)
{
    poolbkernel<<<blockNum(w0 * h0 * c * n), blockMax>>>(x, dx, y, dy, w0, w0 * h0, w1, w1 * h1, size_win, w0 * h0 * c * n, type, a1, a2);
    return getError("poolb");
}

// 2D 卷积前向（朴素实现）
static __global__ void conv2dkernel(float* x, float* w, float* y, int w0, int h0, int c0, int n, int w1, int h1, int c1, int winw, int winh, int stride, int padding, float a1, float a2)
{
    int i = cal_i();
    if (i < w1 * h1 * c1 * n)
    {
        int in = i / (w1 * h1 * c1);
        int ic1 = i / (w1 * h1) % c1;
        int ih1 = i / w1 % h1;
        int iw1 = i % w1;

        int iw0_begin = iw1 * stride - padding;
        int ih0_begin = ih1 * stride - padding;

        float v = 0;
        for (int ic0 = 0; ic0 < c0; ic0++)
        {
            for (int iwinh = 0; iwinh < winh; iwinh++)
            {
                for (int iwinw = 0; iwinw < winw; iwinw++)
                {
                    int ih0 = ih0_begin + iwinh;
                    int iw0 = iw0_begin + iwinw;
                    if (iw0 >= 0 && iw0 < w0 && ih0 >= 0 && ih0 < h0)
                    {
                        int index_x = in * w0 * h0 * c0 + ic0 * w0 * h0 + ih0 * w0 + iw0;
                        int index_w = ic1 * winw * winh * c0 + ic0 * winw * winh + iwinh * winw + iwinw;
                        v += x[index_x] * w[index_w];
                    }
                }
            }
        }
        y[i] = a1 * v + a2 * y[i];
    }
}

int hip_conv2d(float* x, float* w, float* y, int w0, int h0, int c0, int n, int w1, int h1, int c1, int winw, int winh, int stride, int padding, float a1, float a2)
{
    conv2dkernel<<<blockNum(w1 * h1 * c1 * n), blockMax>>>(x, w, y, w0, h0, c0, n, w1, h1, c1, winw, winh, stride, padding, a1, a2);
    return getError("conv2d");
}

// 2D 卷积反向 - 计算输入梯度 dx
static __global__ void conv2db_dkernel(float* dx, float* w, float* dy, int w0, int h0, int c0, int n, int w1, int h1, int c1, int winw, int winh, int stride, int padding, float a1, float a2)
{
    int i = cal_i();
    if (i < w0 * h0 * c0 * n)
    {
        int in = i / (w0 * h0 * c0);
        int ic0 = i / (w0 * h0) % c0;
        int ih0 = i / w0 % h0;
        int iw0 = i % w0;

        int iw1_begin = iw0 * stride + padding;
        int ih1_begin = ih0 * stride + padding;

        float v = 0;
        for (int ic1 = 0; ic1 < c1; ic1++)
        {
            for (int iwinh = 0; iwinh < winh; iwinh++)
            {
                for (int iwinw = 0; iwinw < winw; iwinw++)
                {
                    int ih1 = ih1_begin - iwinh;
                    int iw1 = iw1_begin - iwinw;
                    if (iw1 >= 0 && iw1 < w1 && ih1 >= 0 && ih1 < h1)
                    {
                        int index_y = in * w1 * h1 * c1 + ic1 * w1 * h1 + ih1 * w1 + iw1;
                        int index_w = ic1 * winw * winh * c0 + ic0 * winw * winh + iwinh * winw + iwinw;
                        v += dy[index_y] * w[index_w];
                    }
                }
            }
        }
        dx[i] = a1 * v + a2 * dx[i];
    }
}

int hip_conv2db_d(float* dx, float* w, float* dy, int w0, int h0, int c0, int n, int w1, int h1, int c1, int winw, int winh, int stride, int padding, float a1, float a2)
{
    conv2db_dkernel<<<blockNum(w0 * h0 * c0 * n), blockMax>>>(dx, w, dy, w0, h0, c0, n, w1, h1, c1, winw, winh, stride, padding, a1, a2);
    return getError("conv2db_d");
}

// 2D 卷积反向 - 计算权重梯度 dw
static __global__ void conv2db_wkernel(float* x, float* dw, float* dy, int w0, int h0, int c0, int n, int w1, int h1, int c1, int winw, int winh, int stride, int padding, float a1, float a2)
{
    int i = cal_i();
    if (i < winw * winh * c0 * c1)
    {
        int ic1 = i / (winw * winh * c0);
        int ic0 = i / (winw * winh) % c0;
        int iwinh = i / winw % winh;
        int iwinw = i % winw;

        float v = 0;
        for (int in = 0; in < n; in++)
        {
            for (int iw1 = 0; iw1 < w1; iw1++)
            {
                for (int ih1 = 0; ih1 < h1; ih1++)
                {
                    int ih0 = ih1 + iwinh - padding;
                    int iw0 = iw1 + iwinw - padding;
                    if (iw0 >= 0 && iw0 < w0 && ih0 >= 0 && ih0 < h0)
                    {
                        int index_x = in * w0 * h0 * c0 + ic0 * w0 * h0 + ih0 * w0 + iw0;
                        int index_y = in * w1 * h1 * c1 + ic1 * w1 * h1 + ih1 * w1 + iw1;
                        v += dy[index_y] * x[index_x];
                    }
                }
            }
        }
        dw[i] = a1 * v + a2 * dw[i];
    }
}

int hip_conv2db_w(float* x, float* dw, float* dy, int w0, int h0, int c0, int n, int w1, int h1, int c1, int winw, int winh, int stride, int padding, float a1, float a2)
{
    conv2db_wkernel<<<blockNum(winw * winh * c0 * c1), blockMax>>>(x, dw, dy, w0, h0, c0, n, w1, h1, c1, winw, winh, stride, padding, a1, a2);
    return getError("conv2db_w");
}

// ===========================================================================
// LayerNorm: normalize along inner dimension
// block = LN_BLOCK threads, one block per outer group; shared mem reduction
// ===========================================================================

#define LN_BLOCK 256

static __global__ void layer_norm_fwd_float_kernel(const float* X, float* Y,
    const float* scale, const float* bias, float* mean_out, float* invstd_out,
    unsigned int outer, unsigned int inner, float epsilon)
{
    unsigned int g = blockIdx.x;
    if (g >= outer) { return; }
    const float* x = X + g * inner;
    float* y = Y + g * inner;

    __shared__ float s_sum[LN_BLOCK];
    __shared__ float s_sqsum[LN_BLOCK];

    float local_sum = 0.f, local_sqsum = 0.f;
    for (unsigned int i = threadIdx.x; i < inner; i += blockDim.x)
    {
        float v = x[i];
        local_sum += v;
        local_sqsum += v * v;
    }
    s_sum[threadIdx.x] = local_sum;
    s_sqsum[threadIdx.x] = local_sqsum;
    __syncthreads();
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (threadIdx.x < s)
        {
            s_sum[threadIdx.x] += s_sum[threadIdx.x + s];
            s_sqsum[threadIdx.x] += s_sqsum[threadIdx.x + s];
        }
        __syncthreads();
    }
    float mean = s_sum[0] / (float)inner;
    float var = s_sqsum[0] / (float)inner - mean * mean;
    float invstd = rsqrtf(var + epsilon);
    if (threadIdx.x == 0)
    {
        if (mean_out) { mean_out[g] = mean; }
        if (invstd_out) { invstd_out[g] = invstd; }
    }
    for (unsigned int i = threadIdx.x; i < inner; i += blockDim.x)
    {
        float xhat = (x[i] - mean) * invstd;
        float sc = scale ? scale[i] : 1.f;
        float b = bias ? bias[i] : 0.f;
        y[i] = xhat * sc + b;
    }
}

// ========== Layer Norm Forward: Half Precision ==========

static __global__ void layer_norm_fwd_half_kernel(const half* X, half* Y,
    const half* scale, const half* bias, float* mean_out, float* invstd_out,
    unsigned int outer, unsigned int inner, float epsilon)
{
    unsigned int g = blockIdx.x;
    if (g >= outer) { return; }
    const half* x = X + g * inner;
    half* y = Y + g * inner;

    __shared__ float s_sum[LN_BLOCK];
    __shared__ float s_sqsum[LN_BLOCK];

    float local_sum = 0.f, local_sqsum = 0.f;
    for (unsigned int i = threadIdx.x; i < inner; i += blockDim.x)
    {
        float v = __half2float(x[i]);
        local_sum += v;
        local_sqsum += v * v;
    }
    s_sum[threadIdx.x] = local_sum;
    s_sqsum[threadIdx.x] = local_sqsum;
    __syncthreads();
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (threadIdx.x < s)
        {
            s_sum[threadIdx.x] += s_sum[threadIdx.x + s];
            s_sqsum[threadIdx.x] += s_sqsum[threadIdx.x + s];
        }
        __syncthreads();
    }
    float mean = s_sum[0] / (float)inner;
    float var = s_sqsum[0] / (float)inner - mean * mean;
    float invstd = rsqrtf(var + epsilon);
    if (threadIdx.x == 0)
    {
        if (mean_out) { mean_out[g] = mean; }
        if (invstd_out) { invstd_out[g] = invstd; }
    }
    for (unsigned int i = threadIdx.x; i < inner; i += blockDim.x)
    {
        float xhat = (__half2float(x[i]) - mean) * invstd;
        float sc = scale ? __half2float(scale[i]) : 1.f;
        float b = bias ? __half2float(bias[i]) : 0.f;
        y[i] = __float2half(xhat * sc + b);
    }
}

// ========== Layer Norm Forward: BFloat16 Precision ==========

static __global__ void layer_norm_fwd_bf16_kernel(const bfloat16* X, bfloat16* Y,
    const bfloat16* scale, const bfloat16* bias, float* mean_out, float* invstd_out,
    unsigned int outer, unsigned int inner, float epsilon)
{
    unsigned int g = blockIdx.x;
    if (g >= outer) { return; }
    const bfloat16* x = X + g * inner;
    bfloat16* y = Y + g * inner;

    __shared__ float s_sum[LN_BLOCK];
    __shared__ float s_sqsum[LN_BLOCK];

    float local_sum = 0.f, local_sqsum = 0.f;
    for (unsigned int i = threadIdx.x; i < inner; i += blockDim.x)
    {
        float v = __bfloat162float(x[i]);
        local_sum += v;
        local_sqsum += v * v;
    }
    s_sum[threadIdx.x] = local_sum;
    s_sqsum[threadIdx.x] = local_sqsum;
    __syncthreads();
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (threadIdx.x < s)
        {
            s_sum[threadIdx.x] += s_sum[threadIdx.x + s];
            s_sqsum[threadIdx.x] += s_sqsum[threadIdx.x + s];
        }
        __syncthreads();
    }
    float mean = s_sum[0] / (float)inner;
    float var = s_sqsum[0] / (float)inner - mean * mean;
    float invstd = rsqrtf(fmaxf(var, 0.f) + epsilon);
    if (threadIdx.x == 0)
    {
        if (mean_out) { mean_out[g] = mean; }
        if (invstd_out) { invstd_out[g] = invstd; }
    }
    for (unsigned int i = threadIdx.x; i < inner; i += blockDim.x)
    {
        float xv = __bfloat162float(x[i]);
        float xhat = (xv - mean) * invstd;

        float sc = 1.f;
        if (scale)
        {
            sc = __bfloat162float(scale[i]);
        }

        float b = 0.f;
        if (bias)
        {
            b = __bfloat162float(bias[i]);
        }

        float result = xhat * sc + b;
        y[i] = __float2bfloat16(result);
    }
}

int hip_layer_norm_fwd(int type, void* X, void* Y, void* scale, void* bias,
    void* mean_out, void* invstd_out, unsigned int outer, unsigned int inner, float epsilon)
{
    if (type == 2)    // half
    {
        layer_norm_fwd_half_kernel<<<outer, LN_BLOCK>>>((half*)X, (half*)Y,
            (half*)scale, (half*)bias, (float*)mean_out, (float*)invstd_out,
            outer, inner, epsilon);
        return getError("layer_norm_fwd_half");
    }
    if (type == 3)    // bfloat16
    {
        layer_norm_fwd_bf16_kernel<<<outer, LN_BLOCK>>>((bfloat16*)X, (bfloat16*)Y,
            (bfloat16*)scale, (bfloat16*)bias, (float*)mean_out, (float*)invstd_out,
            outer, inner, epsilon);
        return getError("layer_norm_fwd_bf16");
    }
    if (type != 0)
    {
        fprintf(stderr, "hip_layer_norm_fwd: only float/half/bfloat16 supported (type=%d)\n", type);
        return 1;
    }
    layer_norm_fwd_float_kernel<<<outer, LN_BLOCK>>>((float*)X, (float*)Y,
        (float*)scale, (float*)bias, (float*)mean_out, (float*)invstd_out,
        outer, inner, epsilon);
    return getError("layer_norm_fwd");
}

static __global__ void layer_norm_bwd_float_kernel(const float* X, const float* dY, float* dX,
    const float* scale, const float* mean, const float* invstd,
    float* dscale, float* dbias, unsigned int outer, unsigned int inner)
{}

static __global__ void layer_norm_bwd_bf16_kernel(
    const bfloat16* X, const bfloat16* dY, bfloat16* dX,
    const bfloat16* scale, const float* mean, const float* invstd,
    bfloat16* dscale, bfloat16* dbias, unsigned int outer, unsigned int inner)
{}

int hip_layer_norm_bwd(int type, void* X, void* dY, void* dX,
    void* scale, void* mean, void* invstd, void* dscale, void* dbias,
    unsigned int outer, unsigned int inner)
{
    return 0;
}

// ===========================================================================
// RMS Normalization
// ===========================================================================

static __global__ void rms_norm_fwd_float_kernel(const float* X, float* Y,
    const float* scale, float* invstd_out,
    unsigned int outer, unsigned int inner, float epsilon)
{
    unsigned int g = blockIdx.x;
    if (g >= outer) { return; }
    const float* x = X + g * inner;
    float* y = Y + g * inner;

    __shared__ float s_sqsum[LN_BLOCK];
    float local_sqsum = 0.f;
    for (unsigned int i = threadIdx.x; i < inner; i += blockDim.x)
    {
        float v = x[i];
        local_sqsum += v * v;
    }
    s_sqsum[threadIdx.x] = local_sqsum;
    __syncthreads();
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (threadIdx.x < s) { s_sqsum[threadIdx.x] += s_sqsum[threadIdx.x + s]; }
        __syncthreads();
    }
    float invstd = rsqrtf(s_sqsum[0] / (float)inner + epsilon);
    if (threadIdx.x == 0 && invstd_out) { invstd_out[g] = invstd; }
    for (unsigned int i = threadIdx.x; i < inner; i += blockDim.x)
    {
        float sc = scale ? scale[i] : 1.f;
        y[i] = x[i] * invstd * sc;
    }
}

static __global__ void rms_norm_fwd_half_kernel(const half* X, half* Y,
    const half* scale, float* invstd_out,
    unsigned int outer, unsigned int inner, float epsilon)
{
    unsigned int g = blockIdx.x;
    if (g >= outer) { return; }
    const half* x = X + g * inner;
    half* y = Y + g * inner;
    __shared__ float s_sqsum[LN_BLOCK];
    float local_sqsum = 0.f;
    for (unsigned int i = threadIdx.x; i < inner; i += blockDim.x)
    {
        float v = __half2float(x[i]);
        local_sqsum += v * v;
    }
    s_sqsum[threadIdx.x] = local_sqsum;
    __syncthreads();
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (threadIdx.x < s) { s_sqsum[threadIdx.x] += s_sqsum[threadIdx.x + s]; }
        __syncthreads();
    }
    float invstd = rsqrtf(s_sqsum[0] / (float)inner + epsilon);
    if (threadIdx.x == 0 && invstd_out) { invstd_out[g] = invstd; }
    for (unsigned int i = threadIdx.x; i < inner; i += blockDim.x)
    {
        float sc = scale ? __half2float(scale[i]) : 1.f;
        y[i] = __float2half(__half2float(x[i]) * invstd * sc);
    }
}

static __global__ void rms_norm_fwd_bf16_kernel(const bfloat16* X, bfloat16* Y,
    const bfloat16* scale, float* invstd_out,
    unsigned int outer, unsigned int inner, float epsilon)
{
    unsigned int g = blockIdx.x;
    if (g >= outer) { return; }
    const bfloat16* x = X + g * inner;
    bfloat16* y = Y + g * inner;
    __shared__ float s_sqsum[LN_BLOCK];
    float local_sqsum = 0.f;
    for (unsigned int i = threadIdx.x; i < inner; i += blockDim.x)
    {
        float v = __bfloat162float(x[i]);
        local_sqsum += v * v;
    }
    s_sqsum[threadIdx.x] = local_sqsum;
    __syncthreads();
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (threadIdx.x < s) { s_sqsum[threadIdx.x] += s_sqsum[threadIdx.x + s]; }
        __syncthreads();
    }
    float invstd = rsqrtf(s_sqsum[0] / (float)inner + epsilon);
    if (threadIdx.x == 0 && invstd_out) { invstd_out[g] = invstd; }
    for (unsigned int i = threadIdx.x; i < inner; i += blockDim.x)
    {
        float sc = scale ? __bfloat162float(scale[i]) : 1.f;
        y[i] = __float2bfloat16(__bfloat162float(x[i]) * invstd * sc);
    }
}

int hip_rms_norm_fwd(int type, void* X, void* Y, void* scale,
    void* invstd_out, unsigned int outer, unsigned int inner, float epsilon)
{
    if (type == 0)
    {
        rms_norm_fwd_float_kernel<<<outer, LN_BLOCK>>>((float*)X, (float*)Y,
            (float*)scale, (float*)invstd_out, outer, inner, epsilon);
        return getError("rms_norm_fwd");
    }
    else if (type == 2)
    {
        rms_norm_fwd_half_kernel<<<outer, LN_BLOCK>>>((half*)X, (half*)Y,
            (half*)scale, (float*)invstd_out, outer, inner, epsilon);
        return getError("rms_norm_fwd_half");
    }
    else if (type == 3)
    {
        rms_norm_fwd_bf16_kernel<<<outer, LN_BLOCK>>>((bfloat16*)X, (bfloat16*)Y,
            (bfloat16*)scale, (float*)invstd_out, outer, inner, epsilon);
        return getError("rms_norm_fwd_bf16");
    }
    fprintf(stderr, "hip_rms_norm_fwd: unsupported type=%d\n", type);
    return 1;
}

static __global__ void rms_norm_bwd_float_kernel(const float* X, const float* dY, float* dX,
    const float* scale, const float* invstd,
    float* dscale, unsigned int outer, unsigned int inner)
{}

static __global__ void rms_norm_bwd_bf16_kernel(
    const bfloat16* X, const bfloat16* dY, bfloat16* dX,
    const bfloat16* scale, const float* invstd, bfloat16* dscale,
    unsigned int outer, unsigned int inner)
{}

int hip_rms_norm_bwd(int type, void* X, void* dY, void* dX,
    void* scale, void* invstd, void* dscale,
    unsigned int outer, unsigned int inner)
{
    return 0;
}

// ===========================================================================
// 4D permute
// ===========================================================================

static __global__ void permute4d_float_kernel(const float* X, float* Y,
    int in_d0, int in_d1, int in_d2, int in_d3,
    int out_d0, int out_d1, int out_d2, int out_d3,
    int p0, int p1, int p2, int p3)
{
    unsigned int idx = cal_i();
    unsigned int total = (unsigned int)out_d0 * out_d1 * out_d2 * out_d3;
    if (idx >= total) { return; }
    int o[4];
    o[0] = idx % out_d0;
    unsigned int t = idx / out_d0;
    o[1] = t % out_d1;
    t /= out_d1;
    o[2] = t % out_d2;
    t /= out_d2;
    o[3] = t % out_d3;
    int in_coord[4] = { 0, 0, 0, 0 };
    in_coord[p0] = o[0];
    in_coord[p1] = o[1];
    in_coord[p2] = o[2];
    in_coord[p3] = o[3];
    unsigned int in_lin = in_coord[0]
        + in_coord[1] * in_d0
        + in_coord[2] * in_d0 * in_d1
        + in_coord[3] * in_d0 * in_d1 * in_d2;
    Y[idx] = X[in_lin];
}

static __global__ void permute4d_half_kernel(const half* X, half* Y,
    int in_d0, int in_d1, int in_d2, int in_d3,
    int out_d0, int out_d1, int out_d2, int out_d3,
    int p0, int p1, int p2, int p3)
{
    unsigned int idx = cal_i();
    unsigned int total = (unsigned int)out_d0 * out_d1 * out_d2 * out_d3;
    if (idx >= total) { return; }
    int o[4];
    o[0] = idx % out_d0;
    unsigned int t = idx / out_d0;
    o[1] = t % out_d1;
    t /= out_d1;
    o[2] = t % out_d2;
    t /= out_d2;
    o[3] = t % out_d3;
    int in_coord[4] = { 0, 0, 0, 0 };
    in_coord[p0] = o[0];
    in_coord[p1] = o[1];
    in_coord[p2] = o[2];
    in_coord[p3] = o[3];
    unsigned int in_lin = in_coord[0]
        + in_coord[1] * in_d0
        + in_coord[2] * in_d0 * in_d1
        + in_coord[3] * in_d0 * in_d1 * in_d2;
    Y[idx] = X[in_lin];
}

int hip_permute4d(int type, const void* X, void* Y,
    int in_d0, int in_d1, int in_d2, int in_d3,
    int p0, int p1, int p2, int p3)
{
    int dims[4] = { in_d0, in_d1, in_d2, in_d3 };
    int out_d0 = dims[p0];
    int out_d1 = dims[p1];
    int out_d2 = dims[p2];
    int out_d3 = dims[p3];
    unsigned int total = (unsigned int)out_d0 * out_d1 * out_d2 * out_d3;
    if (type == 0)
    {
        permute4d_float_kernel<<<blockNum(total), blockMax>>>((const float*)X, (float*)Y,
            in_d0, in_d1, in_d2, in_d3,
            out_d0, out_d1, out_d2, out_d3,
            p0, p1, p2, p3);
        return getError("permute4d");
    }
    else if (type == 2 || type == 3)
    {
        // half and bfloat16 are both 16-bit values; permute is a pure index copy
        permute4d_half_kernel<<<blockNum(total), blockMax>>>((const half*)X, (half*)Y,
            in_d0, in_d1, in_d2, in_d3,
            out_d0, out_d1, out_d2, out_d3,
            p0, p1, p2, p3);
        return getError("permute4d_16bit");
    }
    fprintf(stderr, "hip_permute4d: unsupported type=%d\n", type);
    return 1;
}

// ===========================================================================
// RoPE (half-rotate / Qwen style)
// ===========================================================================

static __global__ void rope_fwd_float_kernel(const float* X, float* Y,
    const float* cos_tab, const float* sin_tab,
    unsigned int D, unsigned int T, unsigned int B, unsigned int pos_offset)
{
    unsigned int half_d = D / 2;
    unsigned int total = half_d * T * B;
    unsigned int idx = cal_i();
    if (idx >= total) { return; }
    unsigned int i = idx % half_d;
    unsigned int t = idx / half_d;
    unsigned int b = t / T;
    t = t % T;
    unsigned int base = (b * T + t) * D;
    unsigned int tab = (t + pos_offset) * half_d + i;
    float c = cos_tab[tab];
    float s = sin_tab[tab];
    float xl = X[base + i];
    float xr = X[base + half_d + i];
    Y[base + i] = xl * c - xr * s;
    Y[base + half_d + i] = xr * c + xl * s;
}

static __global__ void rope_fwd_half_kernel(const half* X, half* Y,
    const half* cos_tab, const half* sin_tab,
    unsigned int D, unsigned int T, unsigned int B, unsigned int pos_offset)
{
    unsigned int half_d = D / 2;
    unsigned int total = half_d * T * B;
    unsigned int idx = cal_i();
    if (idx >= total) { return; }
    unsigned int i = idx % half_d;
    unsigned int t = idx / half_d;
    unsigned int b = t / T;
    t = t % T;
    unsigned int base = (b * T + t) * D;
    unsigned int tab = (t + pos_offset) * half_d + i;
    float c = __half2float(cos_tab[tab]);
    float s = __half2float(sin_tab[tab]);
    float xl = __half2float(X[base + i]);
    float xr = __half2float(X[base + half_d + i]);
    Y[base + i] = __float2half(xl * c - xr * s);
    Y[base + half_d + i] = __float2half(xr * c + xl * s);
}

static __global__ void rope_fwd_bf16_kernel(const bfloat16* X, bfloat16* Y,
    const bfloat16* cos_tab, const bfloat16* sin_tab,
    unsigned int D, unsigned int T, unsigned int B, unsigned int pos_offset)
{
    unsigned int half_d = D / 2;
    unsigned int idx = cal_i();
    if (idx >= half_d * T * B) { return; }
    unsigned int i = idx % half_d;
    unsigned int t = idx / half_d;
    unsigned int b = t / T;
    t = t % T;
    unsigned int base = (b * T + t) * D;
    unsigned int tab = (t + pos_offset) * half_d + i;
    float c = __bfloat162float(cos_tab[tab]);
    float s = __bfloat162float(sin_tab[tab]);
    float xl = __bfloat162float(X[base + i]);
    float xr = __bfloat162float(X[base + half_d + i]);
    Y[base + i] = __float2bfloat16(xl * c - xr * s);
    Y[base + half_d + i] = __float2bfloat16(xr * c + xl * s);
}

int hip_rope_fwd(int type, const void* X, void* Y,
    const void* cos_tab, const void* sin_tab,
    unsigned int D, unsigned int T, unsigned int B,
    unsigned int pos_offset)
{
    if (D % 2 != 0)
    {
        fprintf(stderr, "hip_rope_fwd: D must be even (D=%u)\n", D);
        return 1;
    }
    unsigned int total = (D / 2) * T * B;
    if (type == 0)
    {
        rope_fwd_float_kernel<<<blockNum(total), blockMax>>>((const float*)X, (float*)Y,
            (const float*)cos_tab, (const float*)sin_tab, D, T, B, pos_offset);
        return getError("rope_fwd");
    }
    else if (type == 2)
    {
        rope_fwd_half_kernel<<<blockNum(total), blockMax>>>((const half*)X, (half*)Y,
            (const half*)cos_tab, (const half*)sin_tab, D, T, B, pos_offset);
        return getError("rope_fwd_half");
    }
    else if (type == 3)
    {
        rope_fwd_bf16_kernel<<<blockNum(total), blockMax>>>((const bfloat16*)X, (bfloat16*)Y,
            (const bfloat16*)cos_tab, (const bfloat16*)sin_tab, D, T, B, pos_offset);
        return getError("rope_fwd_bf16");
    }
    fprintf(stderr, "hip_rope_fwd: unsupported type=%d\n", type);
    return 1;
}

// ========== RoPE Interleaved: y[2i] = x[2i]*cos - x[2i+1]*sin; y[2i+1] = x[2i+1]*cos + x[2i]*sin ==========

static __global__ void rope_interleaved_fwd_float_kernel(const float* X, float* Y,
    const float* cos_tab, const float* sin_tab,
    unsigned int D, unsigned int T, unsigned int B, unsigned int pos_offset)
{
    unsigned int half_d = D / 2;
    unsigned int total = half_d * T * B;
    unsigned int idx = cal_i();
    if (idx >= total) { return; }
    unsigned int i = idx % half_d;
    unsigned int t = idx / half_d;
    unsigned int b = t / T;
    t = t % T;
    unsigned int base = (b * T + t) * D;
    unsigned int tab = (t + pos_offset) * half_d + i;
    float c = cos_tab[tab];
    float s = sin_tab[tab];
    float xl = X[base + 2 * i];
    float xr = X[base + 2 * i + 1];
    Y[base + 2 * i] = xl * c - xr * s;
    Y[base + 2 * i + 1] = xr * c + xl * s;
}

static __global__ void rope_interleaved_fwd_half_kernel(const half* X, half* Y,
    const half* cos_tab, const half* sin_tab,
    unsigned int D, unsigned int T, unsigned int B, unsigned int pos_offset)
{
    unsigned int half_d = D / 2;
    unsigned int total = half_d * T * B;
    unsigned int idx = cal_i();
    if (idx >= total) { return; }
    unsigned int i = idx % half_d;
    unsigned int t = idx / half_d;
    unsigned int b = t / T;
    t = t % T;
    unsigned int base = (b * T + t) * D;
    unsigned int tab = (t + pos_offset) * half_d + i;
    float c = __half2float(cos_tab[tab]);
    float s = __half2float(sin_tab[tab]);
    float xl = __half2float(X[base + 2 * i]);
    float xr = __half2float(X[base + 2 * i + 1]);
    Y[base + 2 * i] = __float2half(xl * c - xr * s);
    Y[base + 2 * i + 1] = __float2half(xr * c + xl * s);
}

static __global__ void rope_interleaved_fwd_bf16_kernel(const bfloat16* X, bfloat16* Y,
    const bfloat16* cos_tab, const bfloat16* sin_tab,
    unsigned int D, unsigned int T, unsigned int B, unsigned int pos_offset)
{
    unsigned int half_d = D / 2;
    unsigned int total = half_d * T * B;
    unsigned int idx = cal_i();
    if (idx >= total) { return; }
    unsigned int i = idx % half_d;
    unsigned int t = idx / half_d;
    unsigned int b = t / T;
    t = t % T;
    unsigned int base = (b * T + t) * D;
    unsigned int tab = (t + pos_offset) * half_d + i;
    float c = __bfloat162float(cos_tab[tab]);
    float s = __bfloat162float(sin_tab[tab]);
    float xl = __bfloat162float(X[base + 2 * i]);
    float xr = __bfloat162float(X[base + 2 * i + 1]);
    Y[base + 2 * i] = __float2bfloat16(xl * c - xr * s);
    Y[base + 2 * i + 1] = __float2bfloat16(xr * c + xl * s);
}

int hip_rope_interleaved_fwd(int type, const void* X, void* Y,
    const void* cos_tab, const void* sin_tab,
    unsigned int D, unsigned int T, unsigned int B,
    unsigned int pos_offset)
{
    if (D % 2 != 0)
    {
        fprintf(stderr, "hip_rope_interleaved_fwd: D must be even (D=%u)\n", D);
        return 1;
    }
    unsigned int total = (D / 2) * T * B;
    if (type == 0)
    {
        rope_interleaved_fwd_float_kernel<<<blockNum(total), blockMax>>>((const float*)X, (float*)Y,
            (const float*)cos_tab, (const float*)sin_tab, D, T, B, pos_offset);
        return getError("rope_interleaved_fwd");
    }
    else if (type == 2)
    {
        rope_interleaved_fwd_half_kernel<<<blockNum(total), blockMax>>>((const half*)X, (half*)Y,
            (const half*)cos_tab, (const half*)sin_tab, D, T, B, pos_offset);
        return getError("rope_interleaved_fwd_half");
    }
    else if (type == 3)
    {
        rope_interleaved_fwd_bf16_kernel<<<blockNum(total), blockMax>>>((const bfloat16*)X, (bfloat16*)Y,
            (const bfloat16*)cos_tab, (const bfloat16*)sin_tab, D, T, B, pos_offset);
        return getError("rope_interleaved_fwd_bf16");
    }
    fprintf(stderr, "hip_rope_interleaved_fwd: unsupported type=%d\n", type);
    return 1;
}

static __global__ void rope_bwd_float_kernel(const float* dY, float* dX,
    const float* cos_tab, const float* sin_tab,
    unsigned int D, unsigned int T, unsigned int B)
{}

static __global__ void rope_bwd_bf16_kernel(const bfloat16* dY, bfloat16* dX,
    const bfloat16* cos_tab, const bfloat16* sin_tab,
    unsigned int D, unsigned int T, unsigned int B)
{}

int hip_rope_bwd(int type, const void* dY, void* dX,
    const void* cos_tab, const void* sin_tab,
    unsigned int D, unsigned int T, unsigned int B)
{
    return 0;
}

// ===========================================================================
// Pixel Shuffle
// ===========================================================================

static __global__ void pixel_shuffle_fwd_kernel(
    const float* X, float* Y,
    unsigned int W, unsigned int H, unsigned int C_out, unsigned int r, unsigned int N)
{
    unsigned int W_out = W * r;
    unsigned int H_out = H * r;
    unsigned int C_in = C_out * r * r;
    unsigned int total = W_out * H_out * C_out * N;
    unsigned int idx = cal_i();
    if (idx >= total)
    {
        return;
    }

    unsigned int x_out = idx % W_out;
    unsigned int y_out = (idx / W_out) % H_out;
    unsigned int c_out = (idx / (W_out * H_out)) % C_out;
    unsigned int n = idx / (W_out * H_out * C_out);

    unsigned int x_in = x_out / r;
    unsigned int y_in = y_out / r;
    unsigned int dx = x_out % r;
    unsigned int dy = y_out % r;
    unsigned int c_in = c_out * r * r + dy * r + dx;

    Y[idx] = X[x_in + y_in * W + c_in * W * H + n * W * H * C_in];
}

static __global__ void pixel_shuffle16_fwd_kernel(
    const unsigned short* X, unsigned short* Y,
    unsigned int W, unsigned int H, unsigned int C_out, unsigned int r, unsigned int N)
{
    unsigned int W_out = W * r;
    unsigned int H_out = H * r;
    unsigned int C_in = C_out * r * r;
    unsigned int total = W_out * H_out * C_out * N;
    unsigned int idx = cal_i();
    if (idx >= total)
    {
        return;
    }

    unsigned int x_out = idx % W_out;
    unsigned int y_out = (idx / W_out) % H_out;
    unsigned int c_out = (idx / (W_out * H_out)) % C_out;
    unsigned int n = idx / (W_out * H_out * C_out);

    unsigned int x_in = x_out / r;
    unsigned int y_in = y_out / r;
    unsigned int dx = x_out % r;
    unsigned int dy = y_out % r;
    unsigned int c_in = c_out * r * r + dy * r + dx;

    Y[idx] = X[x_in + y_in * W + c_in * W * H + n * W * H * C_in];
}

int hip_pixel_shuffle_fwd(int type, const void* X, void* Y,
    unsigned int W, unsigned int H, unsigned int C_out, unsigned int r, unsigned int N)
{
    unsigned int total = W * r * H * r * C_out * N;
    if (type == 2 || type == 3)
    {
        pixel_shuffle16_fwd_kernel<<<blockNum(total), blockMax>>>(
            (const unsigned short*)X, (unsigned short*)Y, W, H, C_out, r, N);
        return getError("pixel_shuffle_fwd_16bit");
    }
    if (type != 0)
    {
        fprintf(stderr, "hip_pixel_shuffle_fwd: unsupported type=%d\n", type);
        return 1;
    }
    pixel_shuffle_fwd_kernel<<<blockNum(total), blockMax>>>(
        (const float*)X, (float*)Y, W, H, C_out, r, N);
    return getError("pixel_shuffle_fwd");
}

static __global__ void pixel_shuffle_bwd_kernel(
    const float* dY, float* dX,
    unsigned int W, unsigned int H, unsigned int C_out, unsigned int r, unsigned int N)
{}

static __global__ void pixel_shuffle16_bwd_kernel(
    const unsigned short* dY, unsigned short* dX,
    unsigned int W, unsigned int H, unsigned int C_out, unsigned int r, unsigned int N)
{}

int hip_pixel_shuffle_bwd(int type, const void* dY, void* dX,
    unsigned int W, unsigned int H, unsigned int C_out, unsigned int r, unsigned int N)
{
    return 0;
}

// ===========================================================================
// Embedding lookup
// ===========================================================================

static __global__ void embed_fwd_float_kernel(
    const float* ids, const float* W, float* Y,
    int D, int T, int B)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = D * T * B;
    if (idx >= total)
    {
        return;
    }
    int d = idx % D;
    int tb = idx / D;
    int t = tb % T;
    int b = tb / T;
    int id = (int)ids[t + b * T];
    Y[idx] = W[d + (long long)id * D];
}

static __global__ void embed_fwd_half_kernel(
    const float* ids, const half* W, half* Y,
    int D, int T, int B)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = D * T * B;
    if (idx >= total)
    {
        return;
    }
    int d = idx % D;
    int tb = idx / D;
    int t = tb % T;
    int b = tb / T;
    int id = (int)ids[t + b * T];
    Y[idx] = W[d + (long long)id * D];
}

static __global__ void embed_fwd_bf16_kernel(
    const float* ids, const bfloat16* W, bfloat16* Y,
    int D, int T, int B)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = D * T * B;
    if (idx >= total)
    {
        return;
    }
    int d = idx % D;
    int tb = idx / D;
    int t = tb % T;
    int b = tb / T;
    int id = (int)ids[t + b * T];
    Y[idx] = W[d + (long long)id * D];
}

int hip_embed_fwd(int type, const void* ids, const void* W, void* Y,
    int D, int T, int B)
{
    int total = D * T * B;
    if (type == 0)
    {
        embed_fwd_float_kernel<<<blockNum(total), blockMax>>>(
            (const float*)ids, (const float*)W, (float*)Y, D, T, B);
        return getError("embed_fwd");
    }
    else if (type == 2)
    {
        embed_fwd_half_kernel<<<blockNum(total), blockMax>>>(
            (const float*)ids, (const half*)W, (half*)Y, D, T, B);
        return getError("embed_fwd_half");
    }
    else if (type == 3)
    {
        embed_fwd_bf16_kernel<<<blockNum(total), blockMax>>>(
            (const float*)ids, (const bfloat16*)W, (bfloat16*)Y, D, T, B);
        return getError("embed_fwd_bf16");
    }
    fprintf(stderr, "hip_embed_fwd: unsupported type=%d\n", type);
    return 1;
}

static __global__ void embed_bwd_float_kernel(
    const float* ids, const float* dY, float* dW,
    int D, int T, int B)
{}

static __global__ void embed_bwd_bf16_kernel(
    const float* ids, const bfloat16* dY, bfloat16* dW,
    int D, int T, int B)
{}

int hip_embed_bwd(int type, const void* ids, const void* dY, void* dW,
    int D, int T, int B)
{
    return 0;
}

// ===========================================================================
// Tile
// ===========================================================================

static __global__ void tile_fwd_float_kernel(
    const float* X, float* Y,
    int W_in, int H_in, int C_in, int N_in,
    int W_out, int H_out, int C_out, int N_out)
{
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long total = (long long)W_out * H_out * C_out * N_out;
    if (idx >= total)
    {
        return;
    }
    long long t = idx;
    int w = (int)(t % W_out);
    t /= W_out;
    int h = (int)(t % H_out);
    t /= H_out;
    int c = (int)(t % C_out);
    t /= C_out;
    int n = (int)t;
    long long x_idx = (w % W_in)
        + (long long)(h % H_in) * W_in
        + (long long)(c % C_in) * W_in * H_in
        + (long long)(n % N_in) * W_in * H_in * C_in;
    Y[idx] = X[x_idx];
}

static __global__ void tile_fwd_half_kernel(
    const half* X, half* Y,
    int W_in, int H_in, int C_in, int N_in,
    int W_out, int H_out, int C_out, int N_out)
{
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long total = (long long)W_out * H_out * C_out * N_out;
    if (idx >= total)
    {
        return;
    }
    long long t = idx;
    int w = (int)(t % W_out);
    t /= W_out;
    int h = (int)(t % H_out);
    t /= H_out;
    int c = (int)(t % C_out);
    t /= C_out;
    int n = (int)t;
    long long x_idx = (w % W_in)
        + (long long)(h % H_in) * W_in
        + (long long)(c % C_in) * W_in * H_in
        + (long long)(n % N_in) * W_in * H_in * C_in;
    Y[idx] = X[x_idx];
}

int hip_tile_fwd(int type, const void* X, void* Y,
    int W_in, int H_in, int C_in, int N_in,
    int W_out, int H_out, int C_out, int N_out)
{
    long long total = (long long)W_out * H_out * C_out * N_out;
    int grid = (int)((total + blockMax - 1) / blockMax);
    if (type == 0)
    {
        tile_fwd_float_kernel<<<grid, blockMax>>>(
            (const float*)X, (float*)Y,
            W_in, H_in, C_in, N_in,
            W_out, H_out, C_out, N_out);
        return getError("tile_fwd");
    }
    else if (type == 2 || type == 3)
    {
        // half and bfloat16 are both 16-bit plain copies
        tile_fwd_half_kernel<<<grid, blockMax>>>(
            (const half*)X, (half*)Y,
            W_in, H_in, C_in, N_in,
            W_out, H_out, C_out, N_out);
        return getError("tile_fwd_16bit");
    }
    fprintf(stderr, "hip_tile_fwd: unsupported type=%d\n", type);
    return 1;
}

static __global__ void tile_bwd_float_kernel(
    const float* dY, float* dX,
    int W_in, int H_in, int C_in, int N_in,
    int W_out, int H_out, int C_out, int N_out)
{}

static __global__ void tile_bwd_bf16_kernel(
    const bfloat16* dY, bfloat16* dX,
    int W_in, int H_in, int C_in, int N_in,
    int W_out, int H_out, int C_out, int N_out)
{}

int hip_tile_bwd(int type, const void* dY, void* dX,
    int W_in, int H_in, int C_in, int N_in,
    int W_out, int H_out, int C_out, int N_out)
{
    return 0;
}

// ===========================================================================
// Causal mask
// ===========================================================================

static __global__ void causal_mask_float_kernel(float* scores, int T_q, int T_k, long long total, int pos_offset)
{
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total)
    {
        return;
    }
    int qk = (int)(idx % ((long long)T_q * T_k));
    int q = qk / T_k;
    int k = qk % T_k;
    if (k > q + pos_offset)
    {
        scores[idx] = -1e9f;
    }
}

static __global__ void causal_mask_half_kernel(half* scores, int T_q, int T_k, long long total, int pos_offset)
{
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total)
    {
        return;
    }
    int qk = (int)(idx % ((long long)T_q * T_k));
    int q = qk / T_k;
    int k = qk % T_k;
    if (k > q + pos_offset)
    {
        scores[idx] = __float2half(-1e4f);
    }
}

static __global__ void causal_mask_bf16_kernel(bfloat16* scores, int T_q, int T_k, long long total, int pos_offset)
{
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total)
    {
        return;
    }
    int qk = (int)(idx % ((long long)T_q * T_k));
    int q = qk / T_k;
    int k = qk % T_k;
    if (k > q + pos_offset)
    {
        scores[idx] = __float2bfloat16(-1e9f);
    }
}

int hip_causal_mask(int type, void* scores, int T_q, int T_k, int B, int pos_offset)
{
    long long total = (long long)T_q * T_k * B;
    int grid = (int)((total + blockMax - 1) / blockMax);
    if (type == 2)
    {
        causal_mask_half_kernel<<<grid, blockMax>>>((half*)scores, T_q, T_k, total, pos_offset);
    }
    else if (type == 3)
    {
        causal_mask_bf16_kernel<<<grid, blockMax>>>((bfloat16*)scores, T_q, T_k, total, pos_offset);
    }
    else
    {
        causal_mask_float_kernel<<<grid, blockMax>>>((float*)scores, T_q, T_k, total, pos_offset);
    }
    return getError("causal_mask");
}

// ========== Clamp scores (half precision): Prevent softmax overflow by clamping fp16 values ==========

static __global__ void clamp_scores_half_kernel(half* scores, unsigned int n, float threshold)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n)
    {
        return;
    }
    float v = __half2float(scores[idx]);
    if (v > threshold || isnan(v))
    {
        scores[idx] = __float2half(threshold);
    }
    else if (v < -threshold)
    {
        scores[idx] = __float2half(-threshold);
    }
}

int hip_clamp_scores_half(void* scores, unsigned int n, float threshold)
{
    int grid = (int)((n + blockMax - 1) / blockMax);
    clamp_scores_half_kernel<<<grid, blockMax>>>((half*)scores, n, threshold);
    return getError("clamp_scores_half");
}

// ===========================================================================
// Group Normalization affine
// ===========================================================================

static __global__ void group_norm_affine_fwd_float(
    const float* X_hat, float* Y, const float* scale, const float* bias,
    int outer, int inner, int G, int CperG, int WH)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = outer * inner;
    if (idx >= total)
    {
        return;
    }
    int k = idx / inner;
    int i = idx % inner;
    int g = k % G;
    int c = g * CperG + i / WH;
    Y[idx] = scale[c] * X_hat[idx] + bias[c];
}

static __global__ void group_norm_affine_fwd_bf16(
    const bfloat16* X_hat, bfloat16* Y, const bfloat16* scale, const bfloat16* bias,
    int outer, int inner, int G, int CperG, int WH)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = outer * inner;
    if (idx >= total)
    {
        return;
    }
    int k = idx / inner;
    int i = idx % inner;
    int g = k % G;
    int c = g * CperG + i / WH;
    float y = (float)scale[c] * (float)X_hat[idx] + (float)bias[c];
    Y[idx] = bfloat16(y);
}

int hip_group_norm_affine_fwd(int type,
    const void* X_hat, void* Y, const void* scale, const void* bias,
    int outer, int inner, int G, int CperG, int WH)
{
    int total = outer * inner;
    if (type == 3)
    {
        group_norm_affine_fwd_bf16<<<blockNum(total), blockMax>>>(
            (const bfloat16*)X_hat, (bfloat16*)Y, (const bfloat16*)scale, (const bfloat16*)bias,
            outer, inner, G, CperG, WH);
        return getError("group_norm_affine_fwd_bf16");
    }
    if (type != 0)
    {
        fprintf(stderr, "hip_group_norm_affine_fwd: unsupported type=%d\n", type);
        return 1;
    }
    group_norm_affine_fwd_float<<<blockNum(total), blockMax>>>(
        (const float*)X_hat, (float*)Y, (const float*)scale, (const float*)bias,
        outer, inner, G, CperG, WH);
    return getError("group_norm_affine_fwd");
}

static __global__ void group_norm_affine_bwd_float(
    const float* X_hat, const float* dY, float* dX_hat,
    const float* scale, float* dscale, float* dbias,
    int outer, int inner, int G, int CperG, int WH)
{}

static __global__ void group_norm_affine_bwd_bf16(
    const bfloat16* X_hat, const bfloat16* dY, bfloat16* dX_hat,
    const bfloat16* scale, bfloat16* dscale, bfloat16* dbias,
    int outer, int inner, int G, int CperG, int WH)
{}

int hip_group_norm_affine_bwd(int type,
    const void* X_hat, const void* dY, void* dX_hat,
    const void* scale, void* dscale, void* dbias,
    int outer, int inner, int G, int CperG, int WH)
{
    return 0;
}

// ===========================================================================
// VAE Reparameterization
// ===========================================================================

static __global__ void reparam_fwd_float(
    const float* mu, const float* log_var, const float* eps, float* z, unsigned int size)
{
    int i = cal_i();
    if (i < (int)size)
    {
        z[i] = mu[i] + expf(log_var[i] * 0.5f) * eps[i];
    }
}

static __global__ void reparam_fwd_bf16(
    const bfloat16* mu, const bfloat16* log_var, const bfloat16* eps, bfloat16* z,
    unsigned int size)
{
    int i = cal_i();
    if (i < (int)size)
    {
        float m = __bfloat162float(mu[i]);
        float lv = __bfloat162float(log_var[i]);
        float e = __bfloat162float(eps[i]);
        z[i] = __float2bfloat16(m + expf(lv * 0.5f) * e);
    }
}

int hip_reparam_fwd(int type,
    const void* mu, const void* log_var, const void* eps, void* z,
    unsigned int size)
{
    if (type == 3)
    {
        reparam_fwd_bf16<<<blockNum(size), blockMax>>>(
            (const bfloat16*)mu, (const bfloat16*)log_var, (const bfloat16*)eps, (bfloat16*)z, size);
        return getError("reparam_fwd_bf16");
    }
    if (type != 0)
    {
        fprintf(stderr, "hip_reparam_fwd: unsupported type=%d\n", type);
        return 1;
    }
    reparam_fwd_float<<<blockNum(size), blockMax>>>(
        (const float*)mu, (const float*)log_var, (const float*)eps, (float*)z, size);
    return getError("reparam_fwd");
}

static __global__ void reparam_bwd_float(
    const float* log_var, const float* eps, const float* dz,
    float* dmu, float* d_log_var,
    unsigned int size, float alpha_mu, float alpha_lv)
{}

static __global__ void reparam_bwd_bf16(
    const bfloat16* log_var, const bfloat16* eps, const bfloat16* dz,
    bfloat16* dmu, bfloat16* d_log_var,
    unsigned int size, float alpha_mu, float alpha_lv)
{}

int hip_reparam_bwd(int type,
    const void* log_var, const void* eps, const void* dz,
    void* dmu, void* d_log_var,
    unsigned int size, float alpha_mu, float alpha_lv)
{
    return 0;
}

// ===========================================================================
// L1 loss backward
// ===========================================================================

static __global__ void l1_bwd_float(
    const float* A, const float* Y, float* dA,
    unsigned int size, float alpha, float beta)
{}

static __global__ void l1_bwd_bf16(
    const bfloat16* A, const bfloat16* Y, bfloat16* dA,
    unsigned int size, float alpha, float beta)
{}

int hip_l1_bwd(int type,
    const void* A, const void* Y, void* dA,
    unsigned int size, float alpha, float beta)
{
    return 0;
}

// ===========================================================================
// KL log_var backward
// ===========================================================================

static __global__ void kl_lv_bwd_float(
    const float* lv, float* dlv,
    unsigned int size, float alpha, float beta)
{}

static __global__ void kl_lv_bwd_bf16(
    const bfloat16* lv, bfloat16* dlv,
    unsigned int size, float alpha, float beta)
{}

int hip_kl_lv_bwd(int type,
    const void* log_var, void* dlv,
    unsigned int size, float alpha, float beta)
{
    return 0;
}

// ===========================================================================
// Nearest neighbor upsample
// ===========================================================================

static __global__ void upsample_nearest_fwd_float(
    const float* X, float* Y,
    unsigned int W, unsigned int H, unsigned int C, unsigned int N,
    int sh, int sw)
{
    unsigned int Wo = W * (unsigned)sw;
    unsigned int Ho = H * (unsigned)sh;
    unsigned int total = Wo * Ho * C * N;
    unsigned int idx = (unsigned)cal_i();
    if (idx >= total)
    {
        return;
    }
    unsigned int y_w = idx % Wo;
    unsigned int y_h = (idx / Wo) % Ho;
    unsigned int y_c = (idx / (Wo * Ho)) % C;
    unsigned int y_n = idx / (Wo * Ho * C);
    unsigned int x_w = y_w / (unsigned)sw;
    unsigned int x_h = y_h / (unsigned)sh;
    Y[idx] = X[x_w + x_h * W + y_c * W * H + y_n * W * H * C];
}

static __global__ void upsample_nearest16_fwd(
    const unsigned short* X, unsigned short* Y,
    unsigned int W, unsigned int H, unsigned int C, unsigned int N,
    int sh, int sw)
{
    unsigned int Wo = W * (unsigned)sw;
    unsigned int Ho = H * (unsigned)sh;
    unsigned int total = Wo * Ho * C * N;
    unsigned int idx = (unsigned)cal_i();
    if (idx >= total)
    {
        return;
    }
    unsigned int y_w = idx % Wo;
    unsigned int y_h = (idx / Wo) % Ho;
    unsigned int y_c = (idx / (Wo * Ho)) % C;
    unsigned int y_n = idx / (Wo * Ho * C);
    unsigned int x_w = y_w / (unsigned)sw;
    unsigned int x_h = y_h / (unsigned)sh;
    Y[idx] = X[x_w + x_h * W + y_c * W * H + y_n * W * H * C];
}

int hip_upsample_nearest_fwd(int type,
    const void* X, void* Y,
    unsigned int W, unsigned int H, unsigned int C, unsigned int N,
    int sh, int sw)
{
    unsigned int total = W * (unsigned)sw * H * (unsigned)sh * C * N;
    if (type == 2 || type == 3)
    {
        upsample_nearest16_fwd<<<blockNum(total), blockMax>>>(
            (const unsigned short*)X, (unsigned short*)Y, W, H, C, N, sh, sw);
        return getError("upsample_nearest_fwd_16bit");
    }
    if (type != 0)
    {
        fprintf(stderr, "hip_upsample_nearest_fwd: unsupported type=%d\n", type);
        return 1;
    }
    upsample_nearest_fwd_float<<<blockNum(total), blockMax>>>(
        (const float*)X, (float*)Y, W, H, C, N, sh, sw);
    return getError("upsample_nearest_fwd");
}

static __global__ void upsample_nearest_bwd_float(
    float* dX, const float* dY,
    unsigned int W, unsigned int H, unsigned int C, unsigned int N,
    int sh, int sw, float alpha, float beta)
{}

static __global__ void upsample_nearest_bwd_bf16(
    bfloat16* dX, const bfloat16* dY,
    unsigned int W, unsigned int H, unsigned int C, unsigned int N,
    int sh, int sw, float alpha, float beta)
{}

int hip_upsample_nearest_bwd(int type,
    void* dX, const void* dY,
    unsigned int W, unsigned int H, unsigned int C, unsigned int N,
    int sh, int sw, float alpha, float beta)
{
    return 0;
}

// ===========================================================================
// Bilinear upsample (align_corners=False)
// ===========================================================================

static __global__ void upsample_bilinear_fwd_float(
    const float* X, float* Y,
    unsigned int W_in, unsigned int H_in,
    unsigned int W_out, unsigned int H_out,
    unsigned int C, unsigned int N)
{
    unsigned int total = W_out * H_out * C * N;
    unsigned int idx = (unsigned)cal_i();
    if (idx >= total)
    {
        return;
    }
    unsigned int y_w = idx % W_out;
    unsigned int y_h = (idx / W_out) % H_out;
    unsigned int y_c = (idx / (W_out * H_out)) % C;
    unsigned int y_n = idx / (W_out * H_out * C);
    float src_w = (y_w + 0.5f) * (float)W_in / (float)W_out - 0.5f;
    float src_h = (y_h + 0.5f) * (float)H_in / (float)H_out - 0.5f;
    src_w = fmaxf(0.0f, fminf(src_w, (float)(W_in - 1)));
    src_h = fmaxf(0.0f, fminf(src_h, (float)(H_in - 1)));
    int x0 = (int)src_w, x1 = (int)fminf((float)(x0 + 1), (float)(W_in - 1));
    int h0 = (int)src_h, h1 = (int)fminf((float)(h0 + 1), (float)(H_in - 1));
    float dx = src_w - x0, dy = src_h - h0;
    unsigned int base = y_c * W_in * H_in + y_n * W_in * H_in * C;
    float v = (1 - dx) * (1 - dy) * X[base + h0 * W_in + x0]
        + dx * (1 - dy) * X[base + h0 * W_in + x1]
        + (1 - dx) * dy * X[base + h1 * W_in + x0]
        + dx * dy * X[base + h1 * W_in + x1];
    Y[idx] = v;
}

static __global__ void upsample_bilinear_fwd_bf16(
    const bfloat16* X, bfloat16* Y,
    unsigned int W_in, unsigned int H_in,
    unsigned int W_out, unsigned int H_out,
    unsigned int C, unsigned int N)
{
    unsigned int total = W_out * H_out * C * N;
    unsigned int idx = (unsigned)cal_i();
    if (idx >= total)
    {
        return;
    }
    unsigned int y_w = idx % W_out;
    unsigned int y_h = (idx / W_out) % H_out;
    unsigned int y_c = (idx / (W_out * H_out)) % C;
    unsigned int y_n = idx / (W_out * H_out * C);
    float src_w = (y_w + 0.5f) * (float)W_in / (float)W_out - 0.5f;
    float src_h = (y_h + 0.5f) * (float)H_in / (float)H_out - 0.5f;
    src_w = fmaxf(0.0f, fminf(src_w, (float)(W_in - 1)));
    src_h = fmaxf(0.0f, fminf(src_h, (float)(H_in - 1)));
    int x0 = (int)src_w, x1 = (int)fminf((float)(x0 + 1), (float)(W_in - 1));
    int h0 = (int)src_h, h1 = (int)fminf((float)(h0 + 1), (float)(H_in - 1));
    float dx = src_w - x0, dy = src_h - h0;
    unsigned int base = y_c * W_in * H_in + y_n * W_in * H_in * C;
    float v = (1 - dx) * (1 - dy) * __bfloat162float(X[base + h0 * W_in + x0])
        + dx * (1 - dy) * __bfloat162float(X[base + h0 * W_in + x1])
        + (1 - dx) * dy * __bfloat162float(X[base + h1 * W_in + x0])
        + dx * dy * __bfloat162float(X[base + h1 * W_in + x1]);
    Y[idx] = __float2bfloat16(v);
}

int hip_upsample_bilinear_fwd(int type,
    const void* X, void* Y,
    unsigned int W_in, unsigned int H_in,
    unsigned int W_out, unsigned int H_out,
    unsigned int C, unsigned int N)
{
    unsigned int total = W_out * H_out * C * N;
    if (type == 3)
    {
        upsample_bilinear_fwd_bf16<<<blockNum(total), blockMax>>>(
            (const bfloat16*)X, (bfloat16*)Y, W_in, H_in, W_out, H_out, C, N);
        return getError("upsample_bilinear_fwd_bf16");
    }
    if (type != 0)
    {
        fprintf(stderr, "hip_upsample_bilinear_fwd: unsupported type=%d\n", type);
        return 1;
    }
    upsample_bilinear_fwd_float<<<blockNum(total), blockMax>>>(
        (const float*)X, (float*)Y, W_in, H_in, W_out, H_out, C, N);
    return getError("upsample_bilinear_fwd");
}

static __global__ void upsample_bilinear_bwd_float(
    const float* dY, float* dX,
    unsigned int W_in, unsigned int H_in,
    unsigned int W_out, unsigned int H_out,
    unsigned int C, unsigned int N, float alpha)
{}

static __global__ void upsample_bilinear_bwd_bf16(
    const bfloat16* dY, bfloat16* dX,
    unsigned int W_in, unsigned int H_in,
    unsigned int W_out, unsigned int H_out,
    unsigned int C, unsigned int N, float alpha)
{}

int hip_upsample_bilinear_bwd(int type,
    const void* dY, void* dX,
    unsigned int W_in, unsigned int H_in,
    unsigned int W_out, unsigned int H_out,
    unsigned int C, unsigned int N, float alpha)
{
    return 0;
}
