// CUDA 自定义核函数
// 宏命名约定：CUDA_FUNCTIONxy，x=指针参数个数，y=浮点参数个数
// 每个宏自动生成 float/double/half/bfloat16 四种类型的 kernel

#include "cuda_bf16.h"
#include "cuda_fp16.h"
#include "cuda_functions.h"
#include <cmath>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

//using half = __half;

#define blockMax 1024    // 每个 block 的最大线程数

#define cal_i() (blockIdx.x * blockDim.x + threadIdx.x)    // 计算全局线程索引

inline int blockNum(unsigned int size)
{
    return (size + blockMax - 1) / blockMax;
}

inline int getError(const char* content)
{
    cudaDeviceSynchronize();
    cudaError_t cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "%s kernel launch failed: %s\n", content, cudaGetErrorString(cudaStatus));
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

int cuda_half2float(void* p1, void* p2, unsigned int size)
{
    half2floatkernel<<<blockNum(size), blockMax>>>((half*)p1, (float*)p2, size);
    return getError("half2float");
}

static __global__ void bf162floatkernel(__nv_bfloat16* p1, float* p2, unsigned int size)
{
    int i = cal_i();
    if (i < size)
    {
        p2[i] = __bfloat162float(p1[i]);
    }
}

int cuda_bf162float(void* p1, void* p2, unsigned int size)
{
    bf162floatkernel<<<blockNum(size), blockMax>>>((__nv_bfloat16*)p1, (float*)p2, size);
    return getError("bf162float");
}

static __global__ void float2bf16kernel(const float* p1, __nv_bfloat16* p2, unsigned int size)
{
    int i = cal_i();
    if (i < size)
    {
        p2[i] = __float2bfloat16(p1[i]);
    }
}

int cuda_float2bf16(void* p1, void* p2, unsigned int size)
{
    float2bf16kernel<<<blockNum(size), blockMax>>>((const float*)p1, (__nv_bfloat16*)p2, size);
    return getError("float2bf16");
}

#define PATCH_HALF1(func) \
    inline __device__ half func(half a) { return h##func(a); }
#define PATCH_HALF11(func) \
    inline __device__ half func(half a) { return __h##func(a); }
#define PATCH_HALF2(func) \
    inline __device__ half func(half a) { return __float2half(func(__half2float(a))); }

PATCH_HALF2(log)
PATCH_HALF2(floor)
PATCH_HALF11(abs)
PATCH_HALF2(round)
PATCH_HALF2(sin)
PATCH_HALF2(cos)
PATCH_HALF2(sqrt)
inline __device__ half pow(half a, half b) { return __float2half(pow(__half2float(a), __half2float(b))); }

// BF16 math patches (all via float conversion)
#define PATCH_BF162(func) \
    inline __device__ __nv_bfloat16 func(__nv_bfloat16 a) { return __float2bfloat16(func(__bfloat162float(a))); }

PATCH_BF162(log)
PATCH_BF162(floor)
PATCH_BF162(round)
PATCH_BF162(sin)
PATCH_BF162(cos)
PATCH_BF162(sqrt)
inline __device__ __nv_bfloat16 abs(__nv_bfloat16 a) { return __float2bfloat16(fabsf(__bfloat162float(a))); }
inline __device__ __nv_bfloat16 pow(__nv_bfloat16 a, __nv_bfloat16 b) { return __float2bfloat16(powf(__bfloat162float(a), __bfloat162float(b))); }

#define CUDA_FUNCTION22(name, function) \
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
    static __global__ void name##kernel##bfloat16(__nv_bfloat16* p1, __nv_bfloat16* p2, unsigned int size, __nv_bfloat16 a1, __nv_bfloat16 a2) \
    { \
        int i = cal_i(); \
        using fp = __nv_bfloat16; \
        if (i < size) { function; } \
    } \
    int cuda_##name(int type, void* p1, void* p2, unsigned int size, float a1, float a2) \
    { \
        if (type == 0) { name##kernel##float<<<blockNum(size), blockMax>>>((float*)p1, (float*)p2, size, a1, a2); } \
        else if (type == 1) { name##kernel##double<<<blockNum(size), blockMax>>>((double*)p1, (double*)p2, size, a1, a2); } \
        else if (type == 2) { name##kernel##half<<<blockNum(size), blockMax>>>((half*)p1, (half*)p2, size, a1, a2); } \
        else if (type == 3) { name##kernel##bfloat16<<<blockNum(size), blockMax>>>((__nv_bfloat16*)p1, (__nv_bfloat16*)p2, size, __float2bfloat16(a1), __float2bfloat16(a2)); } \
        return getError(#name); \
    }
#define CUDA_FUNCTION23(name, function) \
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
    static __global__ void name##kernel##bfloat16(__nv_bfloat16* p1, __nv_bfloat16* p2, unsigned int size, __nv_bfloat16 a1, __nv_bfloat16 a2, __nv_bfloat16 a3) \
    { \
        int i = cal_i(); \
        using fp = __nv_bfloat16; \
        if (i < size) { function; } \
    } \
    int cuda_##name(int type, void* p1, void* p2, unsigned int size, float a1, float a2, float a3) \
    { \
        if (type == 0) { name##kernel##float<<<blockNum(size), blockMax>>>((float*)p1, (float*)p2, size, a1, a2, a3); } \
        else if (type == 1) { name##kernel##double<<<blockNum(size), blockMax>>>((double*)p1, (double*)p2, size, a1, a2, a3); } \
        else if (type == 2) { name##kernel##half<<<blockNum(size), blockMax>>>((half*)p1, (half*)p2, size, a1, a2, a3); } \
        else if (type == 3) { name##kernel##bfloat16<<<blockNum(size), blockMax>>>((__nv_bfloat16*)p1, (__nv_bfloat16*)p2, size, __float2bfloat16(a1), __float2bfloat16(a2), __float2bfloat16(a3)); } \
        return getError(#name); \
    }
#define CUDA_FUNCTION32(name, function) \
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
    static __global__ void name##kernel##bfloat16(__nv_bfloat16* p1, __nv_bfloat16* p2, __nv_bfloat16* p3, unsigned int size, __nv_bfloat16 a1, __nv_bfloat16 a2) \
    { \
        int i = cal_i(); \
        using fp = __nv_bfloat16; \
        if (i < size) { function; } \
    } \
    int cuda_##name(int type, void* p1, void* p2, void* p3, unsigned int size, float a1, float a2) \
    { \
        if (type == 0) { name##kernel##float<<<blockNum(size), blockMax>>>((float*)p1, (float*)p2, (float*)p3, size, a1, a2); } \
        else if (type == 1) { name##kernel##double<<<blockNum(size), blockMax>>>((double*)p1, (double*)p2, (double*)p3, size, a1, a2); } \
        else if (type == 2) { name##kernel##half<<<blockNum(size), blockMax>>>((half*)p1, (half*)p2, (half*)p3, size, a1, a2); } \
        else if (type == 3) { name##kernel##bfloat16<<<blockNum(size), blockMax>>>((__nv_bfloat16*)p1, (__nv_bfloat16*)p2, (__nv_bfloat16*)p3, size, __float2bfloat16(a1), __float2bfloat16(a2)); } \
        return getError(#name); \
    }
#define CUDA_FUNCTION33(name, function) \
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
    static __global__ void name##kernel##bfloat16(__nv_bfloat16* p1, __nv_bfloat16* p2, __nv_bfloat16* p3, unsigned int size, __nv_bfloat16 a1, __nv_bfloat16 a2, __nv_bfloat16 a3) \
    { \
        int i = cal_i(); \
        using fp = __nv_bfloat16; \
        if (i < size) { function; } \
    } \
    int cuda_##name(int type, void* p1, void* p2, void* p3, unsigned int size, float a1, float a2, float a3) \
    { \
        if (type == 0) { name##kernel##float<<<blockNum(size), blockMax>>>((float*)p1, (float*)p2, (float*)p3, size, a1, a2, a3); } \
        else if (type == 1) { name##kernel##double<<<blockNum(size), blockMax>>>((double*)p1, (double*)p2, (double*)p3, size, a1, a2, a3); } \
        else if (type == 2) { name##kernel##half<<<blockNum(size), blockMax>>>((half*)p1, (half*)p2, (half*)p3, size, a1, a2, a3); } \
        else if (type == 3) { name##kernel##bfloat16<<<blockNum(size), blockMax>>>((__nv_bfloat16*)p1, (__nv_bfloat16*)p2, (__nv_bfloat16*)p3, size, __float2bfloat16(a1), __float2bfloat16(a2), __float2bfloat16(a3)); } \
        return getError(#name); \
    }
#define CUDA_FUNCTION42(name, function) \
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
    static __global__ void name##kernel##bfloat16(__nv_bfloat16* p1, __nv_bfloat16* p2, __nv_bfloat16* p3, __nv_bfloat16* p4, unsigned int size, __nv_bfloat16 a1, __nv_bfloat16 a2) \
    { \
        int i = cal_i(); \
        using fp = __nv_bfloat16; \
        if (i < size) { function; } \
    } \
    int cuda_##name(int type, void* p1, void* p2, void* p3, void* p4, unsigned int size, float a1, float a2) \
    { \
        if (type == 0) { name##kernel##float<<<blockNum(size), blockMax>>>((float*)p1, (float*)p2, (float*)p3, (float*)p4, size, a1, a2); } \
        else if (type == 1) { name##kernel##double<<<blockNum(size), blockMax>>>((double*)p1, (double*)p2, (double*)p3, (double*)p4, size, a1, a2); } \
        else if (type == 2) { name##kernel##half<<<blockNum(size), blockMax>>>((half*)p1, (half*)p2, (half*)p3, (half*)p4, size, a1, a2); } \
        else if (type == 3) { name##kernel##bfloat16<<<blockNum(size), blockMax>>>((__nv_bfloat16*)p1, (__nv_bfloat16*)p2, (__nv_bfloat16*)p3, (__nv_bfloat16*)p4, size, __float2bfloat16(a1), __float2bfloat16(a2)); } \
        return getError(#name); \
    }
#define CUDA_FUNCTION43(name, function) \
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
    static __global__ void name##kernel##bfloat16(__nv_bfloat16* p1, __nv_bfloat16* p2, __nv_bfloat16* p3, __nv_bfloat16* p4, unsigned int size, __nv_bfloat16 a1, __nv_bfloat16 a2, __nv_bfloat16 a3) \
    { \
        int i = cal_i(); \
        using fp = __nv_bfloat16; \
        if (i < size) { function; } \
    } \
    int cuda_##name(int type, void* p1, void* p2, void* p3, void* p4, unsigned int size, float a1, float a2, float a3) \
    { \
        if (type == 0) { name##kernel##float<<<blockNum(size), blockMax>>>((float*)p1, (float*)p2, (float*)p3, (float*)p4, size, a1, a2, a3); } \
        else if (type == 1) { name##kernel##double<<<blockNum(size), blockMax>>>((double*)p1, (double*)p2, (double*)p3, (double*)p4, size, a1, a2, a3); } \
        else if (type == 2) { name##kernel##half<<<blockNum(size), blockMax>>>((half*)p1, (half*)p2, (half*)p3, (half*)p4, size, a1, a2, a3); } \
        else if (type == 3) { name##kernel##bfloat16<<<blockNum(size), blockMax>>>((__nv_bfloat16*)p1, (__nv_bfloat16*)p2, (__nv_bfloat16*)p3, (__nv_bfloat16*)p4, size, __float2bfloat16(a1), __float2bfloat16(a2), __float2bfloat16(a3)); } \
        return getError(#name); \
    }
#define CUDA_FUNCTION44(name, function) \
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
    static __global__ void name##kernel##bfloat16(__nv_bfloat16* p1, __nv_bfloat16* p2, __nv_bfloat16* p3, __nv_bfloat16* p4, unsigned int size, __nv_bfloat16 a1, __nv_bfloat16 a2, __nv_bfloat16 a3, __nv_bfloat16 a4) \
    { \
        int i = cal_i(); \
        using fp = __nv_bfloat16; \
        if (i < size) { function; } \
    } \
    int cuda_##name(int type, void* p1, void* p2, void* p3, void* p4, unsigned int size, float a1, float a2, float a3, float a4) \
    { \
        if (type == 0) { name##kernel##float<<<blockNum(size), blockMax>>>((float*)p1, (float*)p2, (float*)p3, (float*)p4, size, a1, a2, a3, a4); } \
        else if (type == 1) { name##kernel##double<<<blockNum(size), blockMax>>>((double*)p1, (double*)p2, (double*)p3, (double*)p4, size, a1, a2, a3, a4); } \
        else if (type == 2) { name##kernel##half<<<blockNum(size), blockMax>>>((half*)p1, (half*)p2, (half*)p3, (half*)p4, size, a1, a2, a3, a4); } \
        else if (type == 3) { name##kernel##bfloat16<<<blockNum(size), blockMax>>>((__nv_bfloat16*)p1, (__nv_bfloat16*)p2, (__nv_bfloat16*)p3, (__nv_bfloat16*)p4, size, __float2bfloat16(a1), __float2bfloat16(a2), __float2bfloat16(a3), __float2bfloat16(a4)); } \
        return getError(#name); \
    }
#define CUDA_FUNCTION63(name, function) \
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
    static __global__ void name##kernel##bfloat16(__nv_bfloat16* p1, __nv_bfloat16* p2, __nv_bfloat16* p3, __nv_bfloat16* p4, __nv_bfloat16* p5, __nv_bfloat16* p6, unsigned int size, __nv_bfloat16 a1, __nv_bfloat16 a2, __nv_bfloat16 a3) \
    { \
        int i = cal_i(); \
        using fp = __nv_bfloat16; \
        if (i < size) { function; } \
    } \
    int cuda_##name(int type, void* p1, void* p2, void* p3, void* p4, void* p5, void* p6, unsigned int size, float a1, float a2, float a3) \
    { \
        if (type == 0) { name##kernel##float<<<blockNum(size), blockMax>>>((float*)p1, (float*)p2, (float*)p3, (float*)p4, (float*)p5, (float*)p6, size, a1, a2, a3); } \
        else if (type == 1) { name##kernel##double<<<blockNum(size), blockMax>>>((double*)p1, (double*)p2, (double*)p3, (double*)p4, (double*)p5, (double*)p6, size, a1, a2, a3); } \
        else if (type == 2) { name##kernel##half<<<blockNum(size), blockMax>>>((half*)p1, (half*)p2, (half*)p3, (half*)p4, (half*)p5, (half*)p6, size, a1, a2, a3); } \
        else if (type == 3) { name##kernel##bfloat16<<<blockNum(size), blockMax>>>((__nv_bfloat16*)p1, (__nv_bfloat16*)p2, (__nv_bfloat16*)p3, (__nv_bfloat16*)p4, (__nv_bfloat16*)p5, (__nv_bfloat16*)p6, size, __float2bfloat16(a1), __float2bfloat16(a2), __float2bfloat16(a3)); } \
        return getError(#name); \
    }

// R = scale / (A + epsilon)
CUDA_FUNCTION22(reciprocal,
    {
        p2[i] = a1 / (p1[i] + a2);
    });

// R = number + A * scale
CUDA_FUNCTION22(addnumber, { p2[i] = a1 + p1[i] * a2; });

// R = |A + bias|^e × sign(A + bias)  —— 幂函数，对负数取绝对值后恢复符号
CUDA_FUNCTION22(pow,
    {
        p2[i] = pow(abs(p1[i] + a2), a1);
        if (p1[i] < -a2)
        {
            p2[i] *= -1;
        }
    });

// 稀疏惩罚梯度（KL散度的导数）
CUDA_FUNCTION22(sparse,
    {
        p2[i] = ((fp(1) - a1) / (fp(1) - p1[i]) - a1 / p1[i]) * a2;
    });

CUDA_FUNCTION22(sign,
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

CUDA_FUNCTION32(cross_entropy, { p3[i] = -a2 * p2[i] * log(p1[i] + a1); });

CUDA_FUNCTION32(cross_entropy2, { p3[i] = -a2 * (p2[i] * log(p1[i] + a1) + (fp(1) - p2[i]) * log(fp(1) - p1[i] + a1)); });

CUDA_FUNCTION32(add, { p3[i] = p1[i] * a1 + p2[i] * a2; });

CUDA_FUNCTION32(mul, { p3[i] = p1[i] * p2[i] * a1 + p3[i] * a2; });

CUDA_FUNCTION33(div, { p3[i] = a3 * (p1[i] + a1) / (p2[i] + a2); });

CUDA_FUNCTION32(sectionlimit,
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
CUDA_FUNCTION32(ada_update,
    {
        fp& rou = a1;
        fp& epsilon = a2;
        p2[i] = p2[i] * rou + p3[i] * p3[i] * (fp(1) - rou);
        p3[i] = p3[i] * sqrt((p1[i] + epsilon) / (p2[i] + epsilon));
        p1[i] = p1[i] * rou + p3[i] * p3[i] * (fp(1) - rou);
    });

// AdaDelta 更新
CUDA_FUNCTION42(ada_delta_update,
    {
        fp& rou = a1;
        fp& epsilon = a2;
        p1[i] = p1[i] * rou + p3[i] * p3[i] * (fp(1) - rou);
        p4[i] = p3[i] * sqrt((p2[i] + epsilon) / (p1[i] + epsilon));
        p2[i] = p2[i] * rou + p4[i] * p4[i] * (fp(1) - rou);
    });

// Adam 优化器更新
CUDA_FUNCTION44(adam_update,
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
CUDA_FUNCTION32(rms_prop_update,
    {
        fp& rou = a1;
        fp& epsilon = a2;
        p1[i] = p1[i] * rou + p2[i] * p2[i] * (fp(1) - rou);
        p3[i] = p2[i] / sqrt(p1[i] + epsilon);
    });

CUDA_FUNCTION22(sin,
    {
        p2[i] = sin(a1 * p1[i] + a2);
    });

CUDA_FUNCTION22(cos,
    {
        p2[i] = cos(a1 * p1[i] + a2);
    });

CUDA_FUNCTION22(zigzag,
    {
        p2[i] = a1 * (p1[i] + a2 - fp(2) * floor((p1[i] + a2 - fp(1)) / fp(2)) - fp(2));
    });

CUDA_FUNCTION42(zigzagb,
    {
        if (abs(p1[i]) > fp(1 - 1e-2))
        {
            p2[i] = -p4[i] * fp(100);
            return;
        }
        p2[i] = p4[i];
    });

CUDA_FUNCTION22(step,
    {
        p2[i] = round(p1[i] * fp(256)) / fp(256);
    });

CUDA_FUNCTION23(leaky_relu,
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

CUDA_FUNCTION43(leaky_relub,
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

CUDA_FUNCTION33(max,
    {
        p3[i] = p1[i] > p2[i] ? p1[i] : p2[i];
    });
CUDA_FUNCTION63(maxb,
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

CUDA_FUNCTION32(zero_limit,
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

// ===========================================================================
// LayerNorm: 沿 inner 维度做归一化
// 内核布局: 一个 block 处理一个 outer group; 用共享内存做 sum / sum_sq 归约
// 注意: 当前仅实现 float 版本; double/half 暂时返回错误
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

    float local_sum = 0.f;
    float local_sqsum = 0.f;
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
        float s = scale ? scale[i] : 1.f;
        float b = bias ? bias[i] : 0.f;
        y[i] = xhat * s + b;
    }
}

static __global__ void layer_norm_fwd_half_kernel(const __half* X, __half* Y,
    const __half* scale, const __half* bias, float* mean_out, float* invstd_out,
    unsigned int outer, unsigned int inner, float epsilon)
{
    unsigned int g = blockIdx.x;
    if (g >= outer) { return; }
    const __half* x = X + g * inner;
    __half* y = Y + g * inner;

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
        float s = scale ? __half2float(scale[i]) : 1.f;
        float b = bias ? __half2float(bias[i]) : 0.f;
        y[i] = __float2half(xhat * s + b);
    }
}

static __global__ void layer_norm_fwd_bfloat16_kernel(const __nv_bfloat16* X, __nv_bfloat16* Y,
    const __nv_bfloat16* scale, const __nv_bfloat16* bias, float* mean_out, float* invstd_out,
    unsigned int outer, unsigned int inner, float epsilon)
{
    unsigned int g = blockIdx.x;
    if (g >= outer) { return; }
    const __nv_bfloat16* x = X + g * inner;
    __nv_bfloat16* y = Y + g * inner;

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
        float xhat = (__bfloat162float(x[i]) - mean) * invstd;
        float s = scale ? __bfloat162float(scale[i]) : 1.f;
        float b = bias ? __bfloat162float(bias[i]) : 0.f;
        y[i] = __float2bfloat16(xhat * s + b);
    }
}

int cuda_layer_norm_fwd(int type, void* X, void* Y, void* scale, void* bias,
    void* mean_out, void* invstd_out, unsigned int outer, unsigned int inner, float epsilon)
{
    if (type == 2)    // half
    {
        layer_norm_fwd_half_kernel<<<outer, LN_BLOCK>>>((__half*)X, (__half*)Y,
            (__half*)scale, (__half*)bias, (float*)mean_out, (float*)invstd_out,
            outer, inner, epsilon);
        return getError("layer_norm_fwd_half");
    }
    if (type == 3)    // bfloat16
    {
        layer_norm_fwd_bfloat16_kernel<<<outer, LN_BLOCK>>>((__nv_bfloat16*)X, (__nv_bfloat16*)Y,
            (__nv_bfloat16*)scale, (__nv_bfloat16*)bias, (float*)mean_out, (float*)invstd_out,
            outer, inner, epsilon);
        return getError("layer_norm_fwd_bfloat16");
    }
    if (type != 0)
    {
        fprintf(stderr, "cuda_layer_norm_fwd: only float/half/bfloat16 supported (type=%d)\n", type);
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

int cuda_layer_norm_bwd(int type, void* X, void* dY, void* dX,
    void* scale, void* mean, void* invstd, void* dscale, void* dbias,
    unsigned int outer, unsigned int inner)
{
    return 0;
}

// ===========================================================================
// RMS Normalization (无均值, 无 bias). 仿 layer_norm_kernel 简化得到.
// y_i = x_i * invstd * scale_i, invstd = rsqrt(mean(x^2) + eps)
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
        float s = scale ? scale[i] : 1.f;
        y[i] = x[i] * invstd * s;
    }
}

// RMS Norm half-precision: I/O in half, accumulation in float, invstd stored as float
static __global__ void rms_norm_fwd_half_kernel(const __half* X, __half* Y,
    const __half* scale, float* invstd_out,
    unsigned int outer, unsigned int inner, float epsilon)
{
    unsigned int g = blockIdx.x;
    if (g >= outer) { return; }
    const __half* x = X + g * inner;
    __half* y = Y + g * inner;
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

// RMS Norm bfloat16-precision: I/O in bfloat16, accumulation in float
static __global__ void rms_norm_fwd_bfloat16_kernel(const __nv_bfloat16* X, __nv_bfloat16* Y,
    const __nv_bfloat16* scale, float* invstd_out,
    unsigned int outer, unsigned int inner, float epsilon)
{
    unsigned int g = blockIdx.x;
    if (g >= outer) { return; }
    const __nv_bfloat16* x = X + g * inner;
    __nv_bfloat16* y = Y + g * inner;
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

int cuda_rms_norm_fwd(int type, void* X, void* Y, void* scale,
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
        rms_norm_fwd_half_kernel<<<outer, LN_BLOCK>>>((__half*)X, (__half*)Y,
            (__half*)scale, (float*)invstd_out, outer, inner, epsilon);
        return getError("rms_norm_fwd_half");
    }
    else if (type == 3)
    {
        rms_norm_fwd_bfloat16_kernel<<<outer, LN_BLOCK>>>((__nv_bfloat16*)X, (__nv_bfloat16*)Y,
            (__nv_bfloat16*)scale, (float*)invstd_out, outer, inner, epsilon);
        return getError("rms_norm_fwd_bfloat16");
    }
    fprintf(stderr, "cuda_rms_norm_fwd: unsupported type=%d\n", type);
    return 1;
}

// 反向: 设 r = mean(x^2), invstd = 1/sqrt(r+eps), xhat = x*invstd
// y_i = xhat_i * scale_i
// dL/dx_i = scale_i*dy_i*invstd - x_i * (sum_j scale_j*dy_j*x_j) * invstd^3 / inner
//        = invstd * (scale_i*dy_i - xhat_i * sum_b / inner)
//   其中 sum_b = sum_j scale_j*dy_j*xhat_j
// dscale_i += dy_i * xhat_i (跨 outer 累加)
static __global__ void rms_norm_bwd_float_kernel(const float* X, const float* dY, float* dX,
    const float* scale, const float* invstd,
    float* dscale, unsigned int outer, unsigned int inner)
{}

int cuda_rms_norm_bwd(int type, void* X, void* dY, void* dX,
    void* scale, void* invstd, void* dscale,
    unsigned int outer, unsigned int inner)
{
    return 0;
}

// ===========================================================================
// 4D 任意轴置换 permute(W,H,C,N) -> 以 perm 重排 4 个轴
// 内存布局 W 最快变化, 即 lin = w + h*W + c*W*H + n*W*H*C
// out_dims[i] = in_dims[perm[i]]; output coord o_i 对应 input axis perm[i] 的值
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
    // input coord on axis perm[i] = o[i]
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

static __global__ void permute4d_half_kernel(const __half* X, __half* Y,
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

static __global__ void permute4d_bfloat16_kernel(const __nv_bfloat16* X, __nv_bfloat16* Y,
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

int cuda_permute4d(int type, const void* X, void* Y,
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
    else if (type == 2)
    {
        permute4d_half_kernel<<<blockNum(total), blockMax>>>((const __half*)X, (__half*)Y,
            in_d0, in_d1, in_d2, in_d3,
            out_d0, out_d1, out_d2, out_d3,
            p0, p1, p2, p3);
        return getError("permute4d_half");
    }
    else if (type == 3)
    {
        permute4d_bfloat16_kernel<<<blockNum(total), blockMax>>>((const __nv_bfloat16*)X, (__nv_bfloat16*)Y,
            in_d0, in_d1, in_d2, in_d3,
            out_d0, out_d1, out_d2, out_d3,
            p0, p1, p2, p3);
        return getError("permute4d_bfloat16");
    }
    fprintf(stderr, "cuda_permute4d: unsupported type=%d\n", type);
    return 1;
}

// ===========================================================================
// RoPE (half-rotate 风格, ncnn RotaryEmbed mode=0, Llama 标准风格)
// X / Y: (D, T, 1, B), 内存按 W=D 最快, 然后 H=T, 最后 N=B
// cos / sin: (D/2, T) 共享所有 batch
// y_l[i] = x_l[i]*cos[i] - x_r[i]*sin[i]   （l = [0, D/2)）
// y_r[i] = x_r[i]*cos[i] + x_l[i]*sin[i]   （r = [D/2, D)）
// ===========================================================================

static __global__ void rope_fwd_float_kernel(const float* X, float* Y,
    const float* cos_tab, const float* sin_tab,
    unsigned int D, unsigned int T, unsigned int B, unsigned int pos_offset)
{
    unsigned int half = D / 2;
    unsigned int total = half * T * B;
    unsigned int idx = cal_i();
    if (idx >= total) { return; }
    unsigned int i = idx % half;
    unsigned int t = idx / half;
    unsigned int b = t / T;
    t = t % T;
    unsigned int base = (b * T + t) * D;
    unsigned int tab = (t + pos_offset) * half + i;
    float c = cos_tab[tab];
    float s = sin_tab[tab];
    float xl = X[base + i];
    float xr = X[base + half + i];
    Y[base + i] = xl * c - xr * s;
    Y[base + half + i] = xr * c + xl * s;
}

static __global__ void rope_fwd_half_kernel(const __half* X, __half* Y,
    const __half* cos_tab, const __half* sin_tab,
    unsigned int D, unsigned int T, unsigned int B, unsigned int pos_offset)
{
    unsigned int half = D / 2;
    unsigned int total = half * T * B;
    unsigned int idx = cal_i();
    if (idx >= total) { return; }
    unsigned int i = idx % half;
    unsigned int t = idx / half;
    unsigned int b = t / T;
    t = t % T;
    unsigned int base = (b * T + t) * D;
    unsigned int tab = (t + pos_offset) * half + i;
    float c = __half2float(cos_tab[tab]);
    float s = __half2float(sin_tab[tab]);
    float xl = __half2float(X[base + i]);
    float xr = __half2float(X[base + half + i]);
    Y[base + i] = __float2half(xl * c - xr * s);
    Y[base + half + i] = __float2half(xr * c + xl * s);
}

// ===========================================================================
// RoPE interleaved 风格 (ncnn RotaryEmbed mode=1)
// y[2i]   = x[2i]*cos[i]   - x[2i+1]*sin[i]
// y[2i+1] = x[2i+1]*cos[i] + x[2i]*sin[i]
// ===========================================================================

static __global__ void rope_interleaved_fwd_float_kernel(const float* X, float* Y,
    const float* cos_tab, const float* sin_tab,
    unsigned int D, unsigned int T, unsigned int B, unsigned int pos_offset)
{
    unsigned int half = D / 2;
    unsigned int total = half * T * B;
    unsigned int idx = cal_i();
    if (idx >= total) { return; }
    unsigned int i = idx % half;
    unsigned int t = idx / half;
    unsigned int b = t / T;
    t = t % T;
    unsigned int base = (b * T + t) * D;
    unsigned int tab = (t + pos_offset) * half + i;
    float c = cos_tab[tab];
    float s = sin_tab[tab];
    float xl = X[base + 2 * i];
    float xr = X[base + 2 * i + 1];
    Y[base + 2 * i] = xl * c - xr * s;
    Y[base + 2 * i + 1] = xr * c + xl * s;
}

static __global__ void rope_interleaved_fwd_half_kernel(const __half* X, __half* Y,
    const __half* cos_tab, const __half* sin_tab,
    unsigned int D, unsigned int T, unsigned int B, unsigned int pos_offset)
{
    unsigned int half = D / 2;
    unsigned int total = half * T * B;
    unsigned int idx = cal_i();
    if (idx >= total) { return; }
    unsigned int i = idx % half;
    unsigned int t = idx / half;
    unsigned int b = t / T;
    t = t % T;
    unsigned int base = (b * T + t) * D;
    unsigned int tab = (t + pos_offset) * half + i;
    float c = __half2float(cos_tab[tab]);
    float s = __half2float(sin_tab[tab]);
    float xl = __half2float(X[base + 2 * i]);
    float xr = __half2float(X[base + 2 * i + 1]);
    Y[base + 2 * i] = __float2half(xl * c - xr * s);
    Y[base + 2 * i + 1] = __float2half(xr * c + xl * s);
}

static __global__ void rope_fwd_bfloat16_kernel(const __nv_bfloat16* X, __nv_bfloat16* Y,
    const __nv_bfloat16* cos_tab, const __nv_bfloat16* sin_tab,
    unsigned int D, unsigned int T, unsigned int B, unsigned int pos_offset)
{
    unsigned int half = D / 2;
    unsigned int total = half * T * B;
    unsigned int idx = cal_i();
    if (idx >= total) { return; }
    unsigned int i = idx % half;
    unsigned int t = idx / half;
    unsigned int b = t / T;
    t = t % T;
    unsigned int base = (b * T + t) * D;
    unsigned int tab = (t + pos_offset) * half + i;
    float c = __bfloat162float(cos_tab[tab]);
    float s = __bfloat162float(sin_tab[tab]);
    float xl = __bfloat162float(X[base + i]);
    float xr = __bfloat162float(X[base + half + i]);
    Y[base + i] = __float2bfloat16(xl * c - xr * s);
    Y[base + half + i] = __float2bfloat16(xr * c + xl * s);
}

static __global__ void rope_interleaved_fwd_bfloat16_kernel(const __nv_bfloat16* X, __nv_bfloat16* Y,
    const __nv_bfloat16* cos_tab, const __nv_bfloat16* sin_tab,
    unsigned int D, unsigned int T, unsigned int B, unsigned int pos_offset)
{
    unsigned int half = D / 2;
    unsigned int total = half * T * B;
    unsigned int idx = cal_i();
    if (idx >= total) { return; }
    unsigned int i = idx % half;
    unsigned int t = idx / half;
    unsigned int b = t / T;
    t = t % T;
    unsigned int base = (b * T + t) * D;
    unsigned int tab = (t + pos_offset) * half + i;
    float c = __bfloat162float(cos_tab[tab]);
    float s = __bfloat162float(sin_tab[tab]);
    float xl = __bfloat162float(X[base + 2 * i]);
    float xr = __bfloat162float(X[base + 2 * i + 1]);
    Y[base + 2 * i] = __float2bfloat16(xl * c - xr * s);
    Y[base + 2 * i + 1] = __float2bfloat16(xr * c + xl * s);
}

int cuda_rope_fwd(int type, const void* X, void* Y,
    const void* cos_tab, const void* sin_tab,
    unsigned int D, unsigned int T, unsigned int B,
    unsigned int pos_offset)
{
    if (D % 2 != 0)
    {
        fprintf(stderr, "cuda_rope_fwd: D must be even (D=%u)\n", D);
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
        rope_fwd_half_kernel<<<blockNum(total), blockMax>>>((const __half*)X, (__half*)Y,
            (const __half*)cos_tab, (const __half*)sin_tab, D, T, B, pos_offset);
        return getError("rope_fwd_half");
    }
    else if (type == 3)
    {
        rope_fwd_bfloat16_kernel<<<blockNum(total), blockMax>>>((const __nv_bfloat16*)X, (__nv_bfloat16*)Y,
            (const __nv_bfloat16*)cos_tab, (const __nv_bfloat16*)sin_tab, D, T, B, pos_offset);
        return getError("rope_fwd_bfloat16");
    }
    fprintf(stderr, "cuda_rope_fwd: unsupported type=%d\n", type);
    return 1;
}

int cuda_rope_interleaved_fwd(int type, const void* X, void* Y,
    const void* cos_tab, const void* sin_tab,
    unsigned int D, unsigned int T, unsigned int B,
    unsigned int pos_offset)
{
    if (D % 2 != 0)
    {
        fprintf(stderr, "cuda_rope_interleaved_fwd: D must be even (D=%u)\n", D);
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
        rope_interleaved_fwd_half_kernel<<<blockNum(total), blockMax>>>((const __half*)X, (__half*)Y,
            (const __half*)cos_tab, (const __half*)sin_tab, D, T, B, pos_offset);
        return getError("rope_interleaved_fwd_half");
    }
    else if (type == 3)
    {
        rope_interleaved_fwd_bfloat16_kernel<<<blockNum(total), blockMax>>>((const __nv_bfloat16*)X, (__nv_bfloat16*)Y,
            (const __nv_bfloat16*)cos_tab, (const __nv_bfloat16*)sin_tab, D, T, B, pos_offset);
        return getError("rope_interleaved_fwd_bfloat16");
    }
    fprintf(stderr, "cuda_rope_interleaved_fwd: unsupported type=%d\n", type);
    return 1;
}

static __global__ void rope_bwd_float_kernel(const float* dY, float* dX,
    const float* cos_tab, const float* sin_tab,
    unsigned int D, unsigned int T, unsigned int B)
{}

int cuda_rope_bwd(int type, const void* dY, void* dX,
    const void* cos_tab, const void* sin_tab,
    unsigned int D, unsigned int T, unsigned int B)
{
    return 0;
}

//pixel_shuffle (sub-pixel convolution / ESPCN)
//Input  X: (W, H, C_out*r*r, N)  [cccc layout: x + y*W + c*W*H + n*W*H*C_in]
//Output Y: (W*r, H*r, C_out, N)
//Mapping: Y[x_out, y_out, c_out, n] = X[x_out/r, y_out/r, c_out*r*r + (y_out%r)*r + (x_out%r), n]
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

int cuda_pixel_shuffle_fwd(int type, const void* X, void* Y,
    unsigned int W, unsigned int H, unsigned int C_out, unsigned int r, unsigned int N)
{
    if (type != 0)
    {
        fprintf(stderr, "cuda_pixel_shuffle_fwd: only float supported\n");
        return 1;
    }
    unsigned int total = W * r * H * r * C_out * N;
    pixel_shuffle_fwd_kernel<<<blockNum(total), blockMax>>>(
        (const float*)X, (float*)Y, W, H, C_out, r, N);
    return getError("pixel_shuffle_fwd");
}

//pixel_shuffle backward: dX[x_in, y_in, c_in, n] = dY[x_in*r+dx, y_in*r+dy, c_out, n]
static __global__ void pixel_shuffle_bwd_kernel(
    const float* dY, float* dX,
    unsigned int W, unsigned int H, unsigned int C_out, unsigned int r, unsigned int N)
{}

int cuda_pixel_shuffle_bwd(int type, const void* dY, void* dX,
    unsigned int W, unsigned int H, unsigned int C_out, unsigned int r, unsigned int N)
{
    return 0;
}

// ============================================================
// Embedding lookup
// ids: (T*B) float-as-int, W: (D*V), Y: (D*T*B)
// Layout: ids flat as ids[t + b*T], W flat as W[d + id*D], Y flat as Y[d + t*D + b*D*T]
// ============================================================
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
    const float* ids, const __half* W, __half* Y,
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

static __global__ void embed_fwd_bfloat16_kernel(
    const float* ids, const __nv_bfloat16* W, __nv_bfloat16* Y,
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

int cuda_embed_fwd(int type, const void* ids, const void* W, void* Y,
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
        // ids are always float (token IDs encoded as float32 regardless of W type)
        embed_fwd_half_kernel<<<blockNum(total), blockMax>>>(
            (const float*)ids, (const __half*)W, (__half*)Y, D, T, B);
        return getError("embed_fwd_half");
    }
    else if (type == 3)
    {
        embed_fwd_bfloat16_kernel<<<blockNum(total), blockMax>>>(
            (const float*)ids, (const __nv_bfloat16*)W, (__nv_bfloat16*)Y, D, T, B);
        return getError("embed_fwd_bfloat16");
    }
    fprintf(stderr, "cuda_embed_fwd: unsupported type=%d\n", type);
    return 1;
}

static __global__ void embed_bwd_float_kernel(
    const float* ids, const float* dY, float* dW,
    int D, int T, int B)
{}

int cuda_embed_bwd(int type, const void* ids, const void* dY, void* dW,
    int D, int T, int B)
{
    return 0;
}

// ============================================================
// Tile: X (W_in, H_in, C_in, N_in) -> Y (W_out, H_out, C_out, N_out)
// Y[idx] = X[x%W_in + (y%H_in)*W_in + (c%C_in)*W_in*H_in + (n%N_in)*W_in*H_in*C_in]
// ============================================================
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
    const __half* X, __half* Y,
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

static __global__ void tile_fwd_bfloat16_kernel(
    const __nv_bfloat16* X, __nv_bfloat16* Y,
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

int cuda_tile_fwd(int type, const void* X, void* Y,
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
    else if (type == 2)
    {
        tile_fwd_half_kernel<<<grid, blockMax>>>(
            (const __half*)X, (__half*)Y,
            W_in, H_in, C_in, N_in,
            W_out, H_out, C_out, N_out);
        return getError("tile_fwd_half");
    }
    else if (type == 3)
    {
        tile_fwd_bfloat16_kernel<<<grid, blockMax>>>(
            (const __nv_bfloat16*)X, (__nv_bfloat16*)Y,
            W_in, H_in, C_in, N_in,
            W_out, H_out, C_out, N_out);
        return getError("tile_fwd_bfloat16");
    }
    fprintf(stderr, "cuda_tile_fwd: unsupported type=%d\n", type);
    return 1;
}

static __global__ void tile_bwd_float_kernel(
    const float* dY, float* dX,
    int W_in, int H_in, int C_in, int N_in,
    int W_out, int H_out, int C_out, int N_out)
{}

int cuda_tile_bwd(int type, const void* dY, void* dX,
    int W_in, int H_in, int C_in, int N_in,
    int W_out, int H_out, int C_out, int N_out)
{
    return 0;
}
// ============================================================
// Causal mask: set scores[k + q*T_k + b*T_q*T_k] = -1e9 where k > q
// scores: (T_k=width, T_q=height, 1, B) float tensor

static __global__ void causal_mask_float_kernel(float* scores, int T_q, int T_k, long long total, int pos_offset)
{
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total)
    {
        return;
    }
    int b = (int)(idx / ((long long)T_q * T_k));
    int qk = (int)(idx % ((long long)T_q * T_k));
    int q = qk / T_k;
    int k = qk % T_k;
    if (k > q + pos_offset)
    {
        scores[idx] = -1e9f;
    }
}

static __global__ void causal_mask_half_kernel(__half* scores, int T_q, int T_k, long long total, int pos_offset)
{
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total)
    {
        return;
    }
    int b = (int)(idx / ((long long)T_q * T_k));
    int qk = (int)(idx % ((long long)T_q * T_k));
    int q = qk / T_k;
    int k = qk % T_k;
    if (k > q + pos_offset)
    {
        scores[idx] = __float2half(-1e4f);    // half range ~±65504; -1e4 is representable
    }
}

static __global__ void causal_mask_bfloat16_kernel(__nv_bfloat16* scores, int T_q, int T_k, long long total, int pos_offset)
{
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total)
    {
        return;
    }
    int b = (int)(idx / ((long long)T_q * T_k));
    int qk = (int)(idx % ((long long)T_q * T_k));
    int q = qk / T_k;
    int k = qk % T_k;
    if (k > q + pos_offset)
    {
        scores[idx] = __float2bfloat16(-1e4f);
    }
}

int cuda_causal_mask(int type, void* scores, int T_q, int T_k, int B, int pos_offset)
{
    long long total = (long long)T_q * T_k * B;
    int grid = (int)((total + blockMax - 1) / blockMax);
    if (type == 2)
    {
        causal_mask_half_kernel<<<grid, blockMax>>>((__half*)scores, T_q, T_k, total, pos_offset);
    }
    else if (type == 3)
    {
        causal_mask_bfloat16_kernel<<<grid, blockMax>>>((__nv_bfloat16*)scores, T_q, T_k, total, pos_offset);
    }
    else
    {
        causal_mask_float_kernel<<<grid, blockMax>>>((float*)scores, T_q, T_k, total, pos_offset);
    }
    return getError("causal_mask");
}

// ===========================================================================
// Clamp half-precision scores: replace inf/nan with ±threshold to prevent
// float16 overflow → NaN in softmax (CUDNN ACCURATE mode treats inf as NaN)
// ===========================================================================

static __global__ void clamp_scores_half_kernel(__half* scores, unsigned int n, float threshold)
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

int cuda_clamp_scores_half(void* scores, unsigned int n, float threshold)
{
    int grid = (int)((n + blockMax - 1) / blockMax);
    clamp_scores_half_kernel<<<grid, blockMax>>>((__half*)scores, n, threshold);
    return getError("clamp_scores_half");
}

// ===========================================================================
// Group Normalization affine (逐通道 scale/bias)
// 数据布局: [outer=G*N, inner=W*H*CperG]
// 通道索引: c = (k%G)*CperG + i/WH
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
    int k = idx / inner;    // outer group index
    int i = idx % inner;    // position within group
    int g = k % G;          // group index within sample
    int c = g * CperG + i / WH;
    Y[idx] = scale[c] * X_hat[idx] + bias[c];
}

static __global__ void group_norm_affine_fwd_half(
    const __half* X_hat, __half* Y, const __half* scale, const __half* bias,
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
    float xv = __half2float(X_hat[idx]);
    float sv = __half2float(scale[c]);
    float bv = __half2float(bias[c]);
    Y[idx] = __float2half(sv * xv + bv);
}

static __global__ void group_norm_affine_fwd_bfloat16(
    const __nv_bfloat16* X_hat, __nv_bfloat16* Y, const __nv_bfloat16* scale, const __nv_bfloat16* bias,
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
    float xv = __bfloat162float(X_hat[idx]);
    float sv = __bfloat162float(scale[c]);
    float bv = __bfloat162float(bias[c]);
    Y[idx] = __float2bfloat16(sv * xv + bv);
}

int cuda_group_norm_affine_fwd(int type,
    const void* X_hat, void* Y, const void* scale, const void* bias,
    int outer, int inner, int G, int CperG, int WH)
{
    int total = outer * inner;
    if (type == 2)
    {
        group_norm_affine_fwd_half<<<blockNum(total), blockMax>>>(
            (const __half*)X_hat, (__half*)Y, (const __half*)scale, (const __half*)bias,
            outer, inner, G, CperG, WH);
    }
    else if (type == 3)
    {
        group_norm_affine_fwd_bfloat16<<<blockNum(total), blockMax>>>(
            (const __nv_bfloat16*)X_hat, (__nv_bfloat16*)Y, (const __nv_bfloat16*)scale, (const __nv_bfloat16*)bias,
            outer, inner, G, CperG, WH);
    }
    else if (type == 0)
    {
        group_norm_affine_fwd_float<<<blockNum(total), blockMax>>>(
            (const float*)X_hat, (float*)Y, (const float*)scale, (const float*)bias,
            outer, inner, G, CperG, WH);
    }
    else
    {
        fprintf(stderr, "cuda_group_norm_affine_fwd: unsupported type %d\n", type);
        return 1;
    }
    return getError("group_norm_affine_fwd");
}

static __global__ void group_norm_affine_bwd_float(
    const float* X_hat, const float* dY, float* dX_hat,
    const float* scale, float* dscale, float* dbias,
    int outer, int inner, int G, int CperG, int WH)
{}

int cuda_group_norm_affine_bwd(int type,
    const void* X_hat, const void* dY, void* dX_hat,
    const void* scale, void* dscale, void* dbias,
    int outer, int inner, int G, int CperG, int WH)
{
    return 0;
}

// ===========================================================================
// VAE 重参数化: z = mu + exp(log_var*0.5) * eps
// ===========================================================================

static __global__ void reparam_fwd_float(
    const float* mu, const float* log_var, const float* eps, float* z, unsigned int size)
{
    int i = cal_i();
    if (i < size)
    {
        z[i] = mu[i] + expf(log_var[i] * 0.5f) * eps[i];
    }
}

int cuda_reparam_fwd(int type,
    const void* mu, const void* log_var, const void* eps, void* z,
    unsigned int size)
{
    if (type != 0)
    {
        fprintf(stderr, "cuda_reparam_fwd: only float\n");
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

int cuda_reparam_bwd(int type,
    const void* log_var, const void* eps, const void* dz,
    void* dmu, void* d_log_var,
    unsigned int size, float alpha_mu, float alpha_lv)
{
    return 0;
}

// ============================================================
// L1 loss backward: dA[i] = beta*dA[i] + alpha*sign(A[i]-Y[i])
// ============================================================
static __global__ void l1_bwd_float(
    const float* A, const float* Y, float* dA,
    unsigned int size, float alpha, float beta)
{}

int cuda_l1_bwd(int type,
    const void* A, const void* Y, void* dA,
    unsigned int size, float alpha, float beta)
{
    return 0;
}

// ============================================================
// KL log_var backward: dlv[i] = beta*dlv[i] + alpha*0.5*(exp(lv[i])-1)
// ============================================================
static __global__ void kl_lv_bwd_float(
    const float* lv, float* dlv,
    unsigned int size, float alpha, float beta)
{}

int cuda_kl_lv_bwd(int type,
    const void* log_var, void* dlv,
    unsigned int size, float alpha, float beta)
{
    return 0;
}

// ============================================================
// Nearest neighbor upsample: X(W,H,C,N) -> Y(W*sw, H*sh, C, N)
// ============================================================
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

static __global__ void upsample_nearest_fwd_u16(
    const uint16_t* X, uint16_t* Y,
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

int cuda_upsample_nearest_fwd(int type,
    const void* X, void* Y,
    unsigned int W, unsigned int H, unsigned int C, unsigned int N,
    int sh, int sw)
{
    unsigned int total = W * (unsigned)sw * H * (unsigned)sh * C * N;
    if (type == 0)
    {
        upsample_nearest_fwd_float<<<blockNum(total), blockMax>>>(
            (const float*)X, (float*)Y, W, H, C, N, sh, sw);
    }
    else if (type == 2 || type == 3)
    {
        upsample_nearest_fwd_u16<<<blockNum(total), blockMax>>>(
            (const uint16_t*)X, (uint16_t*)Y, W, H, C, N, sh, sw);
    }
    else
    {
        fprintf(stderr, "cuda_upsample_nearest_fwd: unsupported type %d\n", type);
        return 1;
    }
    return getError("upsample_nearest_fwd");
}

// backward: one thread per X element, sums sh*sw dY values
static __global__ void upsample_nearest_bwd_float(
    float* dX, const float* dY,
    unsigned int W, unsigned int H, unsigned int C, unsigned int N,
    int sh, int sw, float alpha, float beta)
{}

int cuda_upsample_nearest_bwd(int type,
    void* dX, const void* dY,
    unsigned int W, unsigned int H, unsigned int C, unsigned int N,
    int sh, int sw, float alpha, float beta)
{
    return 0;
}

// ============================================================
// Bilinear upsample: align_corners=False
// src = (out + 0.5) * (in_size / out_size) - 0.5
// ============================================================
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
    int x0 = (int)src_w, x1 = min(x0 + 1, (int)W_in - 1);
    int h0 = (int)src_h, h1 = min(h0 + 1, (int)H_in - 1);
    float dx = src_w - x0, dy = src_h - h0;
    unsigned int base = y_c * W_in * H_in + y_n * W_in * H_in * C;
    float v = (1 - dx) * (1 - dy) * X[base + h0 * W_in + x0]
        + dx * (1 - dy) * X[base + h0 * W_in + x1]
        + (1 - dx) * dy * X[base + h1 * W_in + x0]
        + dx * dy * X[base + h1 * W_in + x1];
    Y[idx] = v;
}

int cuda_upsample_bilinear_fwd(int type,
    const void* X, void* Y,
    unsigned int W_in, unsigned int H_in,
    unsigned int W_out, unsigned int H_out,
    unsigned int C, unsigned int N)
{
    if (type != 0)
    {
        fprintf(stderr, "cuda_upsample_bilinear_fwd: only float\n");
        return 1;
    }
    unsigned int total = W_out * H_out * C * N;
    upsample_bilinear_fwd_float<<<blockNum(total), blockMax>>>(
        (const float*)X, (float*)Y, W_in, H_in, W_out, H_out, C, N);
    return getError("upsample_bilinear_fwd");
}

// bilinear backward uses atomicAdd; caller pre-scales dX by keepWeight
static __global__ void upsample_bilinear_bwd_float(
    const float* dY, float* dX,
    unsigned int W_in, unsigned int H_in,
    unsigned int W_out, unsigned int H_out,
    unsigned int C, unsigned int N, float alpha)
{}

int cuda_upsample_bilinear_bwd(int type,
    const void* dY, void* dX,
    unsigned int W_in, unsigned int H_in,
    unsigned int W_out, unsigned int H_out,
    unsigned int C, unsigned int N, float alpha)
{
    return 0;
}
