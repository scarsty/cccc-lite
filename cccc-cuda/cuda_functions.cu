// CUDA 自定义核函数
// 宏命名约定：CUDA_FUNCTIONxy，x=指针参数个数，y=浮点参数个数
// 每个宏自动生成 float/double/half/bfloat16 四种类型的 kernel

#include "cuda_bf16.h"
#include "cuda_fp16.h"
#include <cuda_fp8.h>
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
    // On Blackwell/SM100 (CUDA 13), device faults are reported via cudaDeviceSynchronize()
    // return value. cudaGetLastError() alone may return SUCCESS even when a fault occurred.
    // Check both: use whichever is non-zero.
    cudaError_t syncStatus = cudaDeviceSynchronize();
    cudaError_t lastStatus = cudaGetLastError();
    cudaError_t report = (syncStatus != cudaSuccess) ? syncStatus : lastStatus;
    if (report != cudaSuccess)
    {
        fprintf(stderr, "%s kernel launch failed: %s\n", content, cudaGetErrorString(report));
        return 1;
    }
    return 0;
}

void cuda_clear_last_error(void)
{
    cudaGetLastError();    // clears the CUDA last error state
}

// Sync device and report any pending CUDA error. Returns 0 on success, 1 on error.
int cuda_check_error(const char* label)
{
    return getError(label);
}

static __global__ void half2floatkernel(half* p1, float* p2, unsigned int size)
{
    int i = cal_i();
    if (i < size)
    {
        p2[i] = __half2float(p1[i]);
    }
}

int cuda_half_to_float(void* p1, void* p2, unsigned int size)
{
    half2floatkernel<<<blockNum(size), blockMax>>>((half*)p1, (float*)p2, size);
    return getError("half_to_float");
}

static __global__ void bf162floatkernel(__nv_bfloat16* p1, float* p2, unsigned int size)
{
    int i = cal_i();
    if (i < size)
    {
        p2[i] = __bfloat162float(p1[i]);
    }
}

int cuda_bf16_to_float(void* p1, void* p2, unsigned int size)
{
    bf162floatkernel<<<blockNum(size), blockMax>>>((__nv_bfloat16*)p1, (float*)p2, size);
    return getError("bf16_to_float");
}

static __global__ void float2bf16kernel(const float* p1, __nv_bfloat16* p2, unsigned int size)
{
    int i = cal_i();
    if (i < size)
    {
        p2[i] = __float2bfloat16(p1[i]);
    }
}

int cuda_float_to_bf16(void* p1, void* p2, unsigned int size)
{
    float2bf16kernel<<<blockNum(size), blockMax>>>((const float*)p1, (__nv_bfloat16*)p2, size);
    return getError("float_to_bf16");
}

static __global__ void half2bf16kernel(const half* p1, __nv_bfloat16* p2, unsigned int size)
{
    int i = cal_i();
    if (i < size)
    {
        p2[i] = __float2bfloat16(__half2float(p1[i]));
    }
}

int cuda_half_to_bf16(const void* p1, void* p2, unsigned int size)
{
    half2bf16kernel<<<blockNum(size), blockMax>>>((const half*)p1, (__nv_bfloat16*)p2, size);
    return getError("half_to_bf16");
}

static __global__ void bf162halfkernel(const __nv_bfloat16* p1, half* p2, unsigned int size)
{
    int i = cal_i();
    if (i < size)
    {
        p2[i] = __float2half(__bfloat162float(p1[i]));
    }
}

int cuda_bf16_to_half(const void* p1, void* p2, unsigned int size)
{
    bf162halfkernel<<<blockNum(size), blockMax>>>((const __nv_bfloat16*)p1, (half*)p2, size);
    return getError("bf16_to_half");
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

// FP8 E4M3 device helpers (forward declarations; implementations near cuda_float2fp8e4m3)
static __device__ __host__ inline float fp8e4m3_to_float_d(uint8_t b);
static __device__ __host__ inline uint8_t float_to_fp8e4m3_d(float f, float scale);

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

// GELU forward: gelu(x) = 0.5*x*(1+tanh(sqrt(2/pi)*(x+0.044715*x^3)))
// All computation in float to support half/bfloat16 inputs.
// p1=X, p2=Y, a1=scale, a2=keepWeight
CUDA_FUNCTION22(gelu,
    {
        float xf = (float)p1[i];
        float z = 0.7978845608f * (xf + 0.044715f * xf * xf * xf);
        float t = tanhf(z);
        p2[i] = fp(float(a1) * 0.5f * xf * (1.0f + t) + float(a2) * (float)p2[i]);
    });

// GELU backward: dX = alpha * gelu'(X) * dY + keepWeight * dX
// gelu'(x) = 0.5*(1+tanh(z)) + 0.5*x*(1-tanh^2(z))*sqrt(2/pi)*(1+3*0.044715*x^2)
// p1=X, p2=dX (out), p3=Y (unused), p4=dY, a1=alpha, a2=keepWeight
CUDA_FUNCTION42(gelub,
    {
        float xf = (float)p1[i];
        float z = 0.7978845608f * (xf + 0.044715f * xf * xf * xf);
        float t = tanhf(z);
        float sech2 = 1.0f - t * t;
        float grad = 0.5f * (1.0f + t) + 0.5f * xf * sech2 * 0.7978845608f * (1.0f + 3.0f * 0.044715f * xf * xf);
        p2[i] = fp(float(a1) * grad * (float)p4[i] + float(a2) * (float)p2[i]);
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

// RMS Norm FP8 E4M3: I/O in uint8_t (FP8), accumulation in float
static __global__ void rms_norm_fwd_fp8e4m3_kernel(const uint8_t* X, uint8_t* Y,
    const uint8_t* scale, float* invstd_out,
    unsigned int outer, unsigned int inner, float epsilon, float inv_scale_x)
{
    unsigned int g = blockIdx.x;
    if (g >= outer) { return; }
    const uint8_t* x = X + g * inner;
    uint8_t* y = Y + g * inner;
    __shared__ float s_sqsum[LN_BLOCK];
    float local_sqsum = 0.f;
    for (unsigned int i = threadIdx.x; i < inner; i += blockDim.x)
    {
        float v = fp8e4m3_to_float_d(x[i]) * inv_scale_x;
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
        float sc = scale ? fp8e4m3_to_float_d(scale[i]) : 1.f;
        y[i] = float_to_fp8e4m3_d(fp8e4m3_to_float_d(x[i]) * inv_scale_x * invstd * sc, 1.f);
    }
}

// FP8 activation + BF16 scale (rmsNorm weights kept in BF16 in W8A8 mode)
static __global__ void rms_norm_fwd_fp8act_bf16scale_kernel(const uint8_t* X, uint8_t* Y,
    const __nv_bfloat16* scale, float* invstd_out,
    unsigned int outer, unsigned int inner, float epsilon, float inv_scale_x)
{
    unsigned int g = blockIdx.x;
    if (g >= outer) { return; }
    const uint8_t* x = X + g * inner;
    uint8_t* y = Y + g * inner;
    __shared__ float s_sqsum[LN_BLOCK];
    float local_sqsum = 0.f;
    for (unsigned int i = threadIdx.x; i < inner; i += blockDim.x)
    {
        float v = fp8e4m3_to_float_d(x[i]) * inv_scale_x;
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
        y[i] = float_to_fp8e4m3_d(fp8e4m3_to_float_d(x[i]) * inv_scale_x * invstd * sc, 1.f);
    }
}

int cuda_rms_norm_fwd(int type, void* X, void* Y, void* scale, int scale_type,
    void* invstd_out, unsigned int outer, unsigned int inner, float epsilon, float inv_scale_x)
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
    else if (type == 4 || type == 5)
    {
        // FP8 E4M3 activation; scale weight may be BF16 (W8A8 mode) or FP8
        if (scale_type == 3)  // BFLOAT16 scale weight
        {
            rms_norm_fwd_fp8act_bf16scale_kernel<<<outer, LN_BLOCK>>>(
                (const uint8_t*)X, (uint8_t*)Y,
                (const __nv_bfloat16*)scale, (float*)invstd_out, outer, inner, epsilon, inv_scale_x);
            return getError("rms_norm_fwd_fp8e4m3_bf16scale");
        }
        rms_norm_fwd_fp8e4m3_kernel<<<outer, LN_BLOCK>>>((const uint8_t*)X, (uint8_t*)Y,
            (const uint8_t*)scale, (float*)invstd_out, outer, inner, epsilon, inv_scale_x);
        return getError("rms_norm_fwd_fp8e4m3");
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

// permute4d uint8 kernel: same logic as above but for 1-byte-per-element types (FP8/INT8)
static __global__ void permute4d_uint8_kernel(const uint8_t* X, uint8_t* Y,
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
    else if (type == 4 || type == 5 || type == 6)    // FP8 E4M3/E5M2, FP4: 1 byte per element
    {
        permute4d_uint8_kernel<<<blockNum(total), blockMax>>>((const uint8_t*)X, (uint8_t*)Y,
            in_d0, in_d1, in_d2, in_d3,
            out_d0, out_d1, out_d2, out_d3,
            p0, p1, p2, p3);
        return getError("permute4d_fp8");
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

// RoPE FP8 E4M3 (half-rotate): I/O in uint8_t, compute in float
static __global__ void rope_fwd_fp8e4m3_kernel(const uint8_t* X, uint8_t* Y,
    const uint8_t* cos_tab, const uint8_t* sin_tab,
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
    float c = fp8e4m3_to_float_d(cos_tab[tab]);
    float s = fp8e4m3_to_float_d(sin_tab[tab]);
    float xl = fp8e4m3_to_float_d(X[base + i]);
    float xr = fp8e4m3_to_float_d(X[base + half + i]);
    Y[base + i] = float_to_fp8e4m3_d(xl * c - xr * s, 1.f);
    Y[base + half + i] = float_to_fp8e4m3_d(xr * c + xl * s, 1.f);
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
    else if (type == 4 || type == 5)    // FP8 E4M3/E5M2: I/O as uint8, compute in float
    {
        rope_fwd_fp8e4m3_kernel<<<blockNum(total), blockMax>>>((const uint8_t*)X, (uint8_t*)Y,
            (const uint8_t*)cos_tab, (const uint8_t*)sin_tab, D, T, B, pos_offset);
        return getError("rope_fwd_fp8e4m3");
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

// Forward declaration (defined later in this file near other FP8 utilities)
static __device__ __host__ inline float fp8e4m3_to_float_d(uint8_t b);

// FP8 E4M3: decode weight byte → float → scale → BF16 activation
static __global__ void embed_fwd_fp8e4m3_to_bf16_kernel(
    const float* ids, const uint8_t* W, __nv_bfloat16* Y,
    int D, int T, int B, float inv_scale)
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
    uint8_t fp8_val = W[d + (long long)id * D];
    float f = fp8e4m3_to_float_d(fp8_val) * inv_scale;
    Y[idx] = __float2bfloat16(f);
}

// FP8 E4M3 → FP8 E4M3: straight byte-copy lookup (same quantization in W and Y)
// FP8 W → FP8 Y with rescaling: decode W at inv_scale_w (= absmax_W/448), re-encode at scale=1.0.
// This ensures Y uses the same scale=1.0 convention as all other FP8 activations, so ADD/SiLU/GEMM
// kernels that assume scale=1.0 work correctly on the embedding output.
static __global__ void embed_fwd_fp8e4m3_to_fp8e4m3_rescale_kernel(
    const float* ids, const __nv_fp8_e4m3* W, __nv_fp8_e4m3* Y,
    int D, int T, int B, float inv_scale_w)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = D * T * B;
    if (idx >= total) { return; }
    int d = idx % D;
    int tb = idx / D;
    int t = tb % T;
    int b = tb / T;
    int id = (int)ids[t + b * T];
    float f = (float)W[d + (long long)id * D] * inv_scale_w;
    Y[idx] = (__nv_fp8_e4m3)f;
}

C_EXPORT int cuda_embed_fwd_fp8_to_fp8(const void* ids, const void* W, void* Y, int D, int T, int B, float inv_scale_w)
{
    int total = D * T * B;
    embed_fwd_fp8e4m3_to_fp8e4m3_rescale_kernel<<<blockNum(total), blockMax>>>(
        (const float*)ids, (const __nv_fp8_e4m3*)W, (__nv_fp8_e4m3*)Y, D, T, B, inv_scale_w);
    return getError("embed_fwd_fp8_to_fp8");
}

// BF16 权重 → FP8 E4M3 激活：float2bfloat16 解码后以 scale=1.0 编码为 FP8
static __global__ void embed_fwd_bfloat16_to_fp8e4m3_kernel(
    const float* ids, const __nv_bfloat16* W, uint8_t* Y,
    int D, int T, int B)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = D * T * B;
    if (idx >= total) { return; }
    int d = idx % D;
    int tb = idx / D;
    int t = tb % T;
    int b = tb / T;
    int id = (int)ids[t + b * T];
    float f = __bfloat162float(W[d + (long long)id * D]);
    Y[idx] = float_to_fp8e4m3_d(f, 1.0f);
}

C_EXPORT int cuda_embed_fwd_bf16_to_fp8(const void* ids, const void* W, void* Y, int D, int T, int B)
{
    int total = D * T * B;
    embed_fwd_bfloat16_to_fp8e4m3_kernel<<<blockNum(total), blockMax>>>(
        (const float*)ids, (const __nv_bfloat16*)W, (uint8_t*)Y, D, T, B);
    return getError("embed_fwd_bf16_to_fp8");
}

int cuda_embed_fwd(int type, const void* ids, const void* W, void* Y,
    int D, int T, int B, float inv_scale)
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
    else if (type == 4 || type == 5)    // FP8 E4M3: decode → BF16 activation
    {
        embed_fwd_fp8e4m3_to_bf16_kernel<<<blockNum(total), blockMax>>>(
            (const float*)ids, (const uint8_t*)W, (__nv_bfloat16*)Y, D, T, B, inv_scale);
        return getError("embed_fwd_fp8e4m3");
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

// FP8 / any 1-byte-per-element type: treat as uint8_t, pure byte copy
static __global__ void tile_fwd_uint8_kernel(
    const uint8_t* X, uint8_t* Y,
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
    else if (type == 4 || type == 5)    // FP8 E4M3 / E5M2: 1 byte per element
    {
        tile_fwd_uint8_kernel<<<grid, blockMax>>>(
            (const uint8_t*)X, (uint8_t*)Y,
            W_in, H_in, C_in, N_in,
            W_out, H_out, C_out, N_out);
        return getError("tile_fwd_fp8");
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

// FP8 E4M3: most-negative finite value = 0xFE (-448.0); use as -inf for attention masking
static __global__ void causal_mask_fp8e4m3_kernel(uint8_t* scores, int T_q, int T_k, long long total, int pos_offset)
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
        scores[idx] = 0xFE;    // -448.0f in FP8 E4M3 (most negative finite value)
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
    else if (type == 4)    // FP8_E4M3
    {
        causal_mask_fp8e4m3_kernel<<<grid, blockMax>>>((uint8_t*)scores, T_q, T_k, total, pos_offset);
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

// =============================================================================
// FP8 / FP4 量化转换 kernel
// =============================================================================

// ─── 软件实现的 FP8 E4M3 转换（全设备支持，不依赖 CUDA 11.8+） ───────────────
// OCP FP8 E4M3: bias=7, E=15,M=7 为 NaN，最大值 448

// 可移植的 float ↔ uint32 位转换（host 用 memcpy，device 用内置函数）
static __device__ __host__ inline float bits_to_float(uint32_t u)
{
#ifdef __CUDA_ARCH__
    return __int_as_float((int)u);
#else
    float f;
    memcpy(&f, &u, sizeof(f));
    return f;
#endif
}
static __device__ __host__ inline uint32_t float_to_bits(float f)
{
#ifdef __CUDA_ARCH__
    return (uint32_t)__float_as_int(f);
#else
    uint32_t u;
    memcpy(&u, &f, sizeof(u));
    return u;
#endif
}

static __device__ __host__ inline float fp8e4m3_to_float_d(uint8_t b)
{
    const uint8_t s = (b >> 7) & 1u;
    const int e = (b >> 3) & 0xF;
    const int m = b & 7;
    if (e == 15 && m == 7)
    {
        return bits_to_float(0x7FC00000u);    // NaN
    }
    if (e == 0)
    {
        return (s ? -1.f : 1.f) * m * (1.f / 512.f);    // 次正规
    }
    uint32_t f32 = ((uint32_t)s << 31) | ((uint32_t)(e + 120) << 23) | ((uint32_t)m << 20);
    return bits_to_float(f32);
}
static __device__ __host__ inline uint8_t float_to_fp8e4m3_d(float f, float scale)
{
    f *= scale;
    uint32_t u = float_to_bits(f);
    if ((u & 0x7FFFFFFFu) > 0x7F800000u)
    {
        return 0x7Fu;    // NaN
    }
    const uint32_t s = u >> 31;
    const int fe = (int)((u >> 23) & 0xFFu) - 127;
    const uint32_t fm = u & 0x7FFFFFu;
    if (fe > 8 || (fe == 8 && fm >= 0xC00000u))
    {
        return (uint8_t)((s << 7) | 0x7Eu);
    }
    const int e8 = fe + 7;
    if (e8 <= 0)
    {
        const int shift = 14 - fe;
        if (shift >= 32)
        {
            return (uint8_t)(s << 7);
        }
        return (uint8_t)((s << 7) | (((fm | 0x800000u) >> shift) & 0x7u));
    }
    uint8_t m8 = (uint8_t)(fm >> 20);
    if (e8 == 15 && m8 == 7)
    {
        m8 = 6;
    }
    return (uint8_t)((s << 7) | ((uint8_t)(e8 & 0xFu) << 3) | m8);
}

static __global__ void float2fp8e4m3_kernel(const float* src, uint8_t* dst, unsigned int n, float scale)
{
    unsigned int i = cal_i();
    if (i < n)
    {
        dst[i] = float_to_fp8e4m3_d(src[i], scale);
    }
}
static __global__ void fp8e4m32float_kernel(const uint8_t* src, float* dst, unsigned int n, float inv_scale)
{
    unsigned int i = cal_i();
    if (i < n)
    {
        dst[i] = fp8e4m3_to_float_d(src[i]) * inv_scale;
    }
}

int cuda_float_to_fp8e4m3(const void* src, void* dst, unsigned int n, float scale)
{
    float2fp8e4m3_kernel<<<blockNum(n), blockMax>>>((const float*)src, (uint8_t*)dst, n, scale);
    return getError("float_to_fp8e4m3");
}
int cuda_fp8e4m3_to_float(const void* src, void* dst, unsigned int n, float inv_scale)
{
    fp8e4m32float_kernel<<<blockNum(n), blockMax>>>((const uint8_t*)src, (float*)dst, n, inv_scale);
    return getError("fp8e4m3_to_float");
}

// FP8 E4M3 elementwise: R[i] = a*A[i] + b*B[i] + r*R[i]  (compute in float)
// Uses CUDA native __nv_fp8_e4m3 type (SM89+ hardware-accelerated, SW-emulated on older arch)
// to avoid SM120 (Blackwell) NVCC AOT-compiled cubin bug with manual bit-manipulation.
static __global__ void add_fp8e4m3_kernel(const __nv_fp8_e4m3* A, const __nv_fp8_e4m3* B, __nv_fp8_e4m3* R,
    unsigned int n, float a, float b, float r)
{
    unsigned int i = cal_i();
    if (i < n)
    {
        float fr = (r != 0.f) ? (float)R[i] * r : 0.f;
        float val = a * (float)A[i] + b * (float)B[i] + fr;
        R[i] = (__nv_fp8_e4m3)val;
    }
}
C_EXPORT int cuda_add_fp8e4m3(const void* A, const void* B, void* R, unsigned int n, float a, float b, float r)
{
    add_fp8e4m3_kernel<<<blockNum(n), blockMax>>>(
        (const __nv_fp8_e4m3*)A, (const __nv_fp8_e4m3*)B, (__nv_fp8_e4m3*)R, n, a, b, r);
    return getError("add_fp8e4m3");
}

// FP8 E4M3 bias add: caller pre-copies X -> R, then this applies R[i] = a * bias[idx] + b * R[i]
// idx matches the existing CPU/HIP convention used by MatrixEx::addBias.
static __global__ void addbias_fp8e4m3_kernel(const __nv_fp8_e4m3* bias, __nv_fp8_e4m3* R,
    unsigned int n, unsigned int size_mc, unsigned int size_b, float a, float b)
{
    unsigned int i = cal_i();
    if (i < n)
    {
        unsigned int idx = (size_mc > 1) ? ((i % (size_mc * size_b)) / size_mc) : (i % size_b);
        float fbias = (float)bias[idx];
        float fr = (b != 0.0f) ? (float)R[i] * b : 0.0f;
        R[i] = (__nv_fp8_e4m3)(a * fbias + fr);
    }
}

C_EXPORT int cuda_addbias_fp8e4m3(const void* X, const void* bias, void* R,
    unsigned int n, unsigned int size_mc, unsigned int size_b, float a, float b)
{
    (void)X;
    addbias_fp8e4m3_kernel<<<blockNum(n), blockMax>>>(
        (const __nv_fp8_e4m3*)bias, (__nv_fp8_e4m3*)R, n, size_mc, size_b, a, b);
    return getError("addbias_fp8e4m3");
}

static __global__ void addbias_fp8e4m3_bf16bias_kernel(const __nv_bfloat16* bias, __nv_fp8_e4m3* R,
    unsigned int n, unsigned int size_mc, unsigned int size_b, float a, float b)
{
    unsigned int i = cal_i();
    if (i < n)
    {
        unsigned int idx = (size_mc > 1) ? ((i % (size_mc * size_b)) / size_mc) : (i % size_b);
        float fbias = __bfloat162float(bias[idx]);
        float fr = (b != 0.0f) ? (float)R[i] * b : 0.0f;
        R[i] = (__nv_fp8_e4m3)(a * fbias + fr);
    }
}

C_EXPORT int cuda_addbias_fp8e4m3_bf16bias(const void* X, const void* bias, void* R,
    unsigned int n, unsigned int size_mc, unsigned int size_b, float a, float b)
{
    (void)X;
    addbias_fp8e4m3_bf16bias_kernel<<<blockNum(n), blockMax>>>(
        (const __nv_bfloat16*)bias, (__nv_fp8_e4m3*)R, n, size_mc, size_b, a, b);
    return getError("addbias_fp8e4m3_bf16bias");
}

// FP8 E4M3 element-wise SiLU: R[i] = x * sigmoid(x),  x = A[i]
// cuDNN ops treat FP8 tensors as float (4× OOB); must use a native CUDA kernel.
static __global__ void silu_fp8e4m3_kernel(const __nv_fp8_e4m3* A, __nv_fp8_e4m3* R, unsigned int n)
{
    unsigned int i = cal_i();
    if (i < n)
    {
        float x = (float)A[i];
        R[i] = (__nv_fp8_e4m3)(x / (1.0f + expf(-x)));
    }
}
C_EXPORT int cuda_silu_fp8e4m3(const void* A, void* R, unsigned int n)
{
    silu_fp8e4m3_kernel<<<blockNum(n), blockMax>>>((const __nv_fp8_e4m3*)A, (__nv_fp8_e4m3*)R, n);
    return getError("silu_fp8e4m3");
}

// FP8 E4M3 element-wise multiply: R[i] = a * A[i] * B[i] + r * R[i]
// cuDNN ops treat FP8 tensors as float (4× OOB); must use a native CUDA kernel.
static __global__ void elementmul_fp8e4m3_kernel(const __nv_fp8_e4m3* A, const __nv_fp8_e4m3* B, __nv_fp8_e4m3* R,
    unsigned int n, float a, float r)
{
    unsigned int i = cal_i();
    if (i < n)
    {
        float fr = (r != 0.f) ? (float)R[i] * r : 0.f;
        R[i] = (__nv_fp8_e4m3)(a * (float)A[i] * (float)B[i] + fr);
    }
}
C_EXPORT int cuda_elementmul_fp8e4m3(const void* A, const void* B, void* R, unsigned int n, float a, float r)
{
    elementmul_fp8e4m3_kernel<<<blockNum(n), blockMax>>>(
        (const __nv_fp8_e4m3*)A, (const __nv_fp8_e4m3*)B, (__nv_fp8_e4m3*)R, n, a, r);
    return getError("elementmul_fp8e4m3");
}

// BF16 → FP8 E4M3 with static scale (scale=1.0 means direct clip to [-448,448])
static __global__ void bf162fp8e4m3_static_kernel(const uint16_t* src, uint8_t* dst, unsigned int n, float scale)
{
    unsigned int i = cal_i();
    if (i >= n)
    {
        return;
    }
    uint32_t u = (uint32_t)src[i] << 16;
    float f;
    memcpy(&f, &u, sizeof(f));
    dst[i] = float_to_fp8e4m3_d(f, scale);
}
int cuda_bf16_to_fp8e4m3(const void* src, void* dst, unsigned int n, float scale)
{
    bf162fp8e4m3_static_kernel<<<blockNum(n), blockMax>>>((const uint16_t*)src, (uint8_t*)dst, n, scale);
    return getError("bf16_to_fp8e4m3");
}

// FP8 E4M3 → BF16 (for W8A16 without FP32 intermediate)
static __global__ void fp8e4m3_to_bf16_kernel(const uint8_t* src, uint16_t* dst, unsigned int n, float inv_scale)
{
    unsigned int i = cal_i();
    if (i < n)
    {
        float f = fp8e4m3_to_float_d(src[i]) * inv_scale;
        uint32_t u = float_to_bits(f);
        // round-to-nearest-even toward BF16
        u += 0x7FFF + ((u >> 16) & 1u);
        dst[i] = (uint16_t)(u >> 16);
    }
}
int cuda_fp8e4m3_to_bf16(const void* src, void* dst, unsigned int n, float inv_scale)
{
    fp8e4m3_to_bf16_kernel<<<blockNum(n), blockMax>>>((const uint8_t*)src, (uint16_t*)dst, n, inv_scale);
    return getError("fp8e4m3_to_bf16");
}

static __global__ void fp8e4m3_to_half_kernel(const uint8_t* src, half* dst, unsigned int n, float inv_scale)
{
    unsigned int i = cal_i();
    if (i < n)
    {
        dst[i] = __float2half_rn(fp8e4m3_to_float_d(src[i]) * inv_scale);
    }
}
int cuda_fp8e4m3_to_half(const void* src, void* dst, unsigned int n, float inv_scale)
{
    fp8e4m3_to_half_kernel<<<blockNum(n), blockMax>>>((const uint8_t*)src, (half*)dst, n, inv_scale);
    return getError("fp8e4m3_to_half");
}

// ─── BF16 → FP8 E4M3 with dynamic activation scaling (for W8A8) ──────────────
// atomicMax for non-negative floats using bit-manipulation (IEEE 754: bit pattern monotone for positive floats)
static __device__ void atomicMaxFloat(float* addr, float val)
{
    int* ai = (int*)addr;
    int old = *ai, assumed;
    do {
        assumed = old;
        old = atomicCAS(ai, assumed, __float_as_int(fmaxf(val, __int_as_float(assumed))));
    } while (assumed != old);
}

// Parallel abs-max reduction over BF16 array, result accumulated into *out (must be pre-zeroed)
static __global__ void absmax_bf16_kernel(const uint16_t* src, float* out, unsigned int n)
{
    __shared__ float smem[1024];
    unsigned int tid = threadIdx.x;
    float val = 0.0f;
    for (unsigned int i = blockIdx.x * blockDim.x + tid; i < n; i += gridDim.x * blockDim.x)
    {
        uint32_t u = (uint32_t)src[i] << 16;
        float f;
        memcpy(&f, &u, sizeof(f));
        val = fmaxf(val, fabsf(f));
    }
    smem[tid] = val;
    __syncthreads();
    for (unsigned int s = blockDim.x >> 1; s > 0; s >>= 1)
    {
        if (tid < s)
        {
            smem[tid] = fmaxf(smem[tid], smem[tid + s]);
        }
        __syncthreads();
    }
    if (tid == 0)
    {
        atomicMaxFloat(out, smem[0]);
    }
}

// Quantize BF16→FP8 E4M3 reading abs_max from device memory; writes inv_scale=abs_max/448 to d_inv_scale_out
static __global__ void bf162fp8e4m3_devscale_kernel(const uint16_t* src, uint8_t* dst, unsigned int n, const float* d_absmax, float* d_inv_scale_out)
{
    float absmax = *d_absmax;
    float scale = (absmax > 0.f) ? (448.0f / absmax) : 1.0f;
    float inv_scale = (absmax > 0.f) ? (absmax / 448.0f) : 1.0f;
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        *d_inv_scale_out = inv_scale;
    }
    unsigned int i = cal_i();
    if (i < n)
    {
        uint32_t u = (uint32_t)src[i] << 16;
        float f;
        memcpy(&f, &u, sizeof(f));
        dst[i] = float_to_fp8e4m3_d(f, scale);
    }
}

// Dynamic BF16→FP8 E4M3: finds abs-max (step 1), then quantizes + writes inv_scale (step 2).
// d_absmax_tmp : device float* used as scratch for abs-max (will be overwritten)
// d_inv_scale_out: device float* that receives abs_max/448 (for use as cublasLt scale pointer)
C_EXPORT int cuda_bf16_to_fp8e4m3_dynamic(const void* src, void* dst_fp8, unsigned int n, void* d_absmax_tmp, void* d_inv_scale_out)
{
    cudaMemsetAsync(d_absmax_tmp, 0, sizeof(float));
    absmax_bf16_kernel<<<blockNum(n), blockMax>>>((const uint16_t*)src, (float*)d_absmax_tmp, n);
    // GPU stream ordering guarantees step 1 completes before step 2 starts (same default stream)
    bf162fp8e4m3_devscale_kernel<<<blockNum(n), blockMax>>>((const uint16_t*)src, (uint8_t*)dst_fp8, n, (const float*)d_absmax_tmp, (float*)d_inv_scale_out);
    return getError("bf162fp8e4m3_dynamic");
}

// ─── FP8 E5M2 ────────────────────────────────────────────────────────────────
// bias=15; E=31,M=0: ±Inf; E=31,M≠0: NaN; max=57344

static __device__ __host__ inline float fp8e5m2_to_float_d(uint8_t b)
{
    const uint8_t s = (b >> 7) & 1u;
    const int e = (b >> 2) & 0x1F;
    const int m = b & 3;
    if (e == 31)
    {
        if (m == 0)
        {
            return bits_to_float(((uint32_t)s << 31) | 0x7F800000u);
        }
        return bits_to_float(0x7FC00000u);
    }
    if (e == 0)
    {
        return (s ? -1.f : 1.f) * m * (1.f / 65536.f);
    }
    uint32_t f32 = ((uint32_t)s << 31) | ((uint32_t)(e + 112) << 23) | ((uint32_t)m << 21);
    return bits_to_float(f32);
}
static __device__ __host__ inline uint8_t float_to_fp8e5m2_d(float f, float scale)
{
    f *= scale;
    uint32_t u = float_to_bits(f);
    if ((u & 0x7FFFFFFFu) > 0x7F800000u)
    {
        return 0x7Cu;
    }
    const uint32_t s = u >> 31;
    if ((u & 0x7FFFFFFFu) == 0x7F800000u)
    {
        return (uint8_t)((s << 7) | 0x7Cu);
    }
    const int fe = (int)((u >> 23) & 0xFFu) - 127;
    const uint32_t fm = u & 0x7FFFFFu;
    if (fe > 15 || (fe == 15 && fm >= 0xE00000u))
    {
        return (uint8_t)((s << 7) | 0x7Bu);
    }
    const int e8 = fe + 15;
    if (e8 <= 0)
    {
        const int shift = 7 - fe;
        if (shift >= 32)
        {
            return (uint8_t)(s << 7);
        }
        return (uint8_t)((s << 7) | (((fm | 0x800000u) >> shift) & 0x3u));
    }
    const uint8_t m8 = (uint8_t)(fm >> 21);
    return (uint8_t)((s << 7) | ((uint8_t)(e8 & 0x1Fu) << 2) | m8);
}

static __global__ void float2fp8e5m2_kernel(const float* src, uint8_t* dst, unsigned int n, float scale)
{
    unsigned int i = cal_i();
    if (i < n)
    {
        dst[i] = float_to_fp8e5m2_d(src[i], scale);
    }
}
static __global__ void fp8e5m22float_kernel(const uint8_t* src, float* dst, unsigned int n, float inv_scale)
{
    unsigned int i = cal_i();
    if (i < n)
    {
        dst[i] = fp8e5m2_to_float_d(src[i]) * inv_scale;
    }
}

int cuda_float_to_fp8e5m2(const void* src, void* dst, unsigned int n, float scale)
{
    float2fp8e5m2_kernel<<<blockNum(n), blockMax>>>((const float*)src, (uint8_t*)dst, n, scale);
    return getError("float_to_fp8e5m2");
}
int cuda_fp8e5m2_to_float(const void* src, void* dst, unsigned int n, float inv_scale)
{
    fp8e5m22float_kernel<<<blockNum(n), blockMax>>>((const uint8_t*)src, (float*)dst, n, inv_scale);
    return getError("fp8e5m2_to_float");
}

// FP8 E5M2 → BF16 (for W8A16 without FP32 intermediate)
static __global__ void fp8e5m2_to_bf16_kernel(const uint8_t* src, uint16_t* dst, unsigned int n, float inv_scale)
{
    unsigned int i = cal_i();
    if (i < n)
    {
        float f = fp8e5m2_to_float_d(src[i]) * inv_scale;
        uint32_t u = float_to_bits(f);
        u += 0x7FFF + ((u >> 16) & 1u);
        dst[i] = (uint16_t)(u >> 16);
    }
}
int cuda_fp8e5m2_to_bf16(const void* src, void* dst, unsigned int n, float inv_scale)
{
    fp8e5m2_to_bf16_kernel<<<blockNum(n), blockMax>>>((const uint8_t*)src, (uint16_t*)dst, n, inv_scale);
    return getError("fp8e5m2_to_bf16");
}

static __global__ void fp8e5m2_to_half_kernel(const uint8_t* src, half* dst, unsigned int n, float inv_scale)
{
    unsigned int i = cal_i();
    if (i < n)
    {
        dst[i] = __float2half_rn(fp8e5m2_to_float_d(src[i]) * inv_scale);
    }
}
int cuda_fp8e5m2_to_half(const void* src, void* dst, unsigned int n, float inv_scale)
{
    fp8e5m2_to_half_kernel<<<blockNum(n), blockMax>>>((const uint8_t*)src, (half*)dst, n, inv_scale);
    return getError("fp8e5m2_to_half");
}

// ─── FP4 E2M1 packed nibble (NVIDIA Blackwell NVFP4) ─────────────────────────
// 低 nibble = 偶数元素，高 nibble = 奇数元素
// Values: 0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6

static __constant__ float fp4_table[16] = {
    0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f,
    0.f, -0.5f, -1.f, -1.5f, -2.f, -3.f, -4.f, -6.f
};

static __device__ inline uint8_t float_to_fp4_d(float f)
{
    uint8_t s = (f < 0.f) ? 8u : 0u;
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

// 每个线程处理 2 个 float → 1 个 packed byte（n 为元素数，即 dst 长度为 ceil(n/2)）
static __global__ void float2fp4_kernel(const float* src, uint8_t* dst, unsigned int n, float scale)
{
    unsigned int i = cal_i();    // i = byte index
    unsigned int e0 = i * 2;
    if (e0 >= n)
    {
        return;
    }
    uint8_t lo = float_to_fp4_d(src[e0] * scale);
    uint8_t hi = (e0 + 1 < n) ? float_to_fp4_d(src[e0 + 1] * scale) : 0u;
    dst[i] = lo | (hi << 4);
}

static __global__ void fp42float_kernel(const uint8_t* src, float* dst, unsigned int n, float inv_scale)
{
    unsigned int i = cal_i();    // i = element index
    if (i >= n)
    {
        return;
    }
    uint8_t nibble = (i & 1) ? (src[i >> 1] >> 4) : (src[i >> 1] & 0xFu);
    dst[i] = fp4_table[nibble] * inv_scale;
}

int cuda_float_to_fp4(const void* src, void* dst, unsigned int n, float scale)
{
    unsigned int n_bytes = (n + 1) / 2;
    float2fp4_kernel<<<blockNum(n_bytes), blockMax>>>((const float*)src, (uint8_t*)dst, n, scale);
    return getError("float_to_fp4");
}
int cuda_fp4_to_float(const void* src, void* dst, unsigned int n, float inv_scale)
{
    fp42float_kernel<<<blockNum(n), blockMax>>>((const uint8_t*)src, (float*)dst, n, inv_scale);
    return getError("fp4_to_float");
}

// FP4 E2M1 packed nibble → BF16 (n = element count; reads ceil(n/2) bytes)
static __global__ void fp4_to_bf16_kernel(const uint8_t* src, uint16_t* dst, unsigned int n, float inv_scale)
{
    unsigned int i = cal_i();    // element index
    if (i >= n)
    {
        return;
    }
    uint8_t nibble = (i & 1u) ? (src[i >> 1] >> 4) : (src[i >> 1] & 0xFu);
    float f = fp4_table[nibble] * inv_scale;
    uint32_t u = float_to_bits(f);
    u += 0x7FFF + ((u >> 16) & 1u);
    dst[i] = (uint16_t)(u >> 16);
}
int cuda_fp4_to_bf16(const void* src, void* dst, unsigned int n, float inv_scale)
{
    fp4_to_bf16_kernel<<<blockNum(n), blockMax>>>((const uint8_t*)src, (uint16_t*)dst, n, inv_scale);
    return getError("fp4_to_bf16");
}

// ─── FP4 E2M1 block-scale (group_size=16) kernels ────────────────────────────
// dst_scales stores dequant scale = absmax/kMaxFP4 for each block.
// block_size must be even (power-of-2 ≥ 2); typically 16.

// Quantize: one thread handles one block of block_size elements.
static __global__ void float_to_fp4_blockscale_kernel(
    const float* src, uint8_t* dst_nibbles, uint8_t* dst_scales,
    unsigned int n_elems, unsigned int block_size)
{
    unsigned int b = cal_i();
    unsigned int elem_start = b * block_size;
    if (elem_start >= n_elems) { return; }
    unsigned int elem_end = min(elem_start + block_size, n_elems);

    // absmax of this block
    float absmax = 0.0f;
    for (unsigned int i = elem_start; i < elem_end; i++)
    {
        absmax = fmaxf(absmax, fabsf(src[i]));
    }
    const float kMaxFP4 = 6.0f;
    float fwd_scale   = (absmax > 1e-10f) ? (kMaxFP4 / absmax) : 1.0f;
    float dq_scale    = (absmax > 1e-10f) ? (absmax / kMaxFP4) : 1.0f;  // dequant scale
    dst_scales[b]     = float_to_fp8e4m3_d(dq_scale, 1.0f);  // store as FP8 E4M3

    // quantize pairs → packed bytes
    for (unsigned int i = elem_start; i < elem_end; i += 2)
    {
        uint8_t lo = float_to_fp4_d(src[i] * fwd_scale);
        uint8_t hi = (i + 1 < n_elems) ? float_to_fp4_d(src[i + 1] * fwd_scale) : 0u;
        dst_nibbles[i >> 1] = lo | (hi << 4);
    }
}

// Dequantize: one thread per element.
static __global__ void fp4_blockscale_to_bf16_kernel(
    const uint8_t* src, const uint8_t* scales,
    uint16_t* dst, unsigned int n_elems, unsigned int block_size)
{
    unsigned int i = cal_i();
    if (i >= n_elems) { return; }
    unsigned int b = i / block_size;
    float inv_scale  = fp8e4m3_to_float_d(scales[b]);
    uint8_t nibble   = (i & 1u) ? (src[i >> 1] >> 4) : (src[i >> 1] & 0x0Fu);
    float f          = fp4_table[nibble] * inv_scale;
    uint32_t u       = float_to_bits(f);
    u += 0x7FFF + ((u >> 16) & 1u);
    dst[i] = (uint16_t)(u >> 16);
}

// float → FP4 packed nibbles + FP8 E4M3 block-dequant scales
// dst_nibbles: ceil(n_elems/2) bytes; dst_scales: ceil(n_elems/block_size) uint8_t (FP8 E4M3)
C_EXPORT int cuda_float_to_fp4_blockscale(const void* src, void* dst_nibbles, void* dst_scales,
    unsigned int n_elems, unsigned int block_size)
{
    unsigned int n_blocks = (n_elems + block_size - 1) / block_size;
    float_to_fp4_blockscale_kernel<<<blockNum(n_blocks), blockMax>>>(
        (const float*)src, (uint8_t*)dst_nibbles, (uint8_t*)dst_scales, n_elems, block_size);
    return getError("float_to_fp4_blockscale");
}

// FP4 packed nibbles + FP8 E4M3 block-dequant scales → BF16
C_EXPORT int cuda_fp4_blockscale_to_bf16(const void* src_nibbles, const void* src_scales,
    void* dst, unsigned int n_elems, unsigned int block_size)
{
    fp4_blockscale_to_bf16_kernel<<<blockNum(n_elems), blockMax>>>(
        (const uint8_t*)src_nibbles, (const uint8_t*)src_scales,
        (uint16_t*)dst, n_elems, block_size);
    return getError("fp4_blockscale_to_bf16");
}

// Unified conversion dispatcher. src_type/dst_type values match cccc::DataType enum:
//   0=float, 1=double, 2=half, 3=bfloat16, 4=fp8_e4m3, 5=fp8_e5m2, 6=fp4_e2m1
// scale: multiplied into conversion output (use 1.0f for lossless casts).
// ─── ROI Align (float only) ───────────────────────────────────────────────────
// Feature map layout: (W, H, C, B) WHCN → index = w + h*W + c*W*H + b*W*H*C
// Boxes layout:       (4, N, 1, B) WHCN → index = coord + box_n*4 + batch*4*N
//   coord in {0,1,2,3} = {x1, y1, x2, y2} in feature-space pixel coordinates
// Output layout:      (roi_size, roi_size, C, N*B) WHCN
//   roi_idx = box_n + batch * N_boxes
// Sampling: aligned=True convention (same as torchvision default since 0.7)
//   x_sample = x1 + (ox+0.5) * (x2-x1)/roi_size - 0.5

static __global__ void roi_align_fwd_float_kernel(
    const float* feat, int W, int H, int C, int B,
    const float* boxes, int N_boxes,
    float* out, int roi_size, float spatial_scale,
    int total)
{
    int idx = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (idx >= total) return;

    // Decompose output index (WHCN: fastest = ox)
    int RSRSC = roi_size * roi_size * C;
    int roi_idx = idx / RSRSC;
    int rem = idx % RSRSC;
    int c = rem / (roi_size * roi_size);
    int sp = rem % (roi_size * roi_size);
    int oy = sp / roi_size;
    int ox = sp % roi_size;

    int box_n = roi_idx % N_boxes;
    int batch_b = roi_idx / N_boxes;

    // boxes: (4, N_boxes, 1, B) → x_coord + box_n*4 + batch_b*4*N_boxes
    int box_off = box_n * 4 + batch_b * 4 * N_boxes;
    float x1 = boxes[box_off + 0] * spatial_scale;
    float y1 = boxes[box_off + 1] * spatial_scale;
    float x2 = boxes[box_off + 2] * spatial_scale;
    float y2 = boxes[box_off + 3] * spatial_scale;

    float bin_w = (x2 - x1) / roi_size;
    float bin_h = (y2 - y1) / roi_size;
    // aligned=True: sample at bin center, subtract 0.5 to convert to array coords
    float xs = x1 + (ox + 0.5f) * bin_w - 0.5f;
    float ys = y1 + (oy + 0.5f) * bin_h - 0.5f;

    if (xs < -1.0f || xs > W || ys < -1.0f || ys > H) {
        out[idx] = 0.0f;
        return;
    }
    xs = fmaxf(xs, 0.0f);
    ys = fmaxf(ys, 0.0f);

    int ix0 = (int)xs, ix1 = min(ix0 + 1, W - 1);
    int iy0 = (int)ys, iy1 = min(iy0 + 1, H - 1);
    ix0 = min(ix0, W - 1);
    iy0 = min(iy0, H - 1);
    float dx = xs - ix0, dy = ys - iy0;

    int feat_base = c * W * H + batch_b * W * H * C;
    float v00 = feat[ix0 + iy0 * W + feat_base];
    float v10 = feat[ix1 + iy0 * W + feat_base];
    float v01 = feat[ix0 + iy1 * W + feat_base];
    float v11 = feat[ix1 + iy1 * W + feat_base];
    out[idx] = (1-dx)*(1-dy)*v00 + dx*(1-dy)*v10 + (1-dx)*dy*v01 + dx*dy*v11;
}

static __global__ void roi_align_bwd_float_kernel(
    const float* grad_out, int W, int H, int C, int B,
    const float* boxes, int N_boxes,
    float* grad_feat,
    int roi_size, float spatial_scale,
    int total)
{}

C_EXPORT int cuda_roi_align_fwd(
    const void* feat, int W, int H, int C, int B,
    const void* boxes, int N_boxes,
    void* out, int roi_size, float spatial_scale)
{
    int total = roi_size * roi_size * C * N_boxes * B;
    roi_align_fwd_float_kernel<<<blockNum(total), blockMax>>>(
        (const float*)feat, W, H, C, B,
        (const float*)boxes, N_boxes,
        (float*)out, roi_size, spatial_scale, total);
    return getError("roi_align_fwd");
}

C_EXPORT int cuda_roi_align_bwd(
    const void* grad_out, int W, int H, int C, int B,
    const void* boxes, int N_boxes,
    void* grad_feat, int roi_size, float spatial_scale)
{
    return 0;
}

// Wrappers: adapt 3-param / non-const functions to the uniform ConvFn signature.
// ConvFn = int(*)(const void* src, void* dst, unsigned int n, float scale)
static int s_half_to_float(const void* s, void* d, unsigned int n, float) { return cuda_half_to_float((void*)s, d, n); }
static int s_float_to_bf16(const void* s, void* d, unsigned int n, float) { return cuda_float_to_bf16((void*)s, d, n); }
static int s_bf16_to_float(const void* s, void* d, unsigned int n, float) { return cuda_bf16_to_float((void*)s, d, n); }
static int s_half_to_bf16 (const void* s, void* d, unsigned int n, float) { return cuda_half_to_bf16(s, d, n); }
static int s_bf16_to_half (const void* s, void* d, unsigned int n, float) { return cuda_bf16_to_half(s, d, n); }

C_EXPORT int cuda_convert(const void* src, int src_type, void* dst, int dst_type, unsigned int n, float scale)
{
    if (src_type == dst_type && scale == 1.0f)
    {
        cudaMemcpy(dst, src, (size_t)n, cudaMemcpyDeviceToDevice);
        return 0;
    }
    // 2D dispatch table [src_type][dst_type]
    // DataType enum: FLOAT=0, DOUBLE=1, HALF=2, BFLOAT16=3, FP8_E4M3=4, FP8_E5M2=5, FP4_E2M1=6
    using ConvFn = int(*)(const void*, void*, unsigned int, float);
    static const ConvFn tbl[7][7] = {
        // src\dst  FLOAT              DOUBLE   HALF                  BF16                   FP8_E4M3              FP8_E5M2               FP4
        /*FLOAT  */{ nullptr,          nullptr, nullptr,              s_float_to_bf16,        cuda_float_to_fp8e4m3, cuda_float_to_fp8e5m2, cuda_float_to_fp4  },
        /*DOUBLE */{ nullptr,          nullptr, nullptr,              nullptr,                nullptr,               nullptr,               nullptr            },
        /*HALF   */{ s_half_to_float,  nullptr, nullptr,              s_half_to_bf16,         nullptr,               nullptr,               nullptr            },
        /*BF16   */{ s_bf16_to_float,  nullptr, s_bf16_to_half,       nullptr,               cuda_bf16_to_fp8e4m3,  nullptr,               nullptr            },
        /*FP8E4M3*/{ cuda_fp8e4m3_to_float, nullptr, cuda_fp8e4m3_to_half, cuda_fp8e4m3_to_bf16, nullptr,          nullptr,               nullptr            },
        /*FP8E5M2*/{ cuda_fp8e5m2_to_float, nullptr, cuda_fp8e5m2_to_half, cuda_fp8e5m2_to_bf16, nullptr,          nullptr,               nullptr            },
        /*FP4    */{ cuda_fp4_to_float, nullptr, nullptr,             cuda_fp4_to_bf16,       nullptr,               nullptr,               nullptr            },
    };
    if ((unsigned)src_type >= 7 || (unsigned)dst_type >= 7) return -1;
    auto fn = tbl[src_type][dst_type];
    if (!fn) return -1;
    return fn(src, dst, n, scale);
}
