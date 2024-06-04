
#include "cuda_fp16.h"
#include "cuda_functions.h"
#include <cmath>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

//using half = __half;

#define blockMax 1024

#define cal_i() (blockIdx.x * blockDim.x + threadIdx.x)

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
    int cuda_##name(int type, void* p1, void* p2, unsigned int size, float a1, float a2) \
    { \
        if (type == 0) { name##kernel##float<<<blockNum(size), blockMax>>>((float*)p1, (float*)p2, size, a1, a2); } \
        else if (type == 1) { name##kernel##double<<<blockNum(size), blockMax>>>((double*)p1, (double*)p2, size, a1, a2); } \
        else if (type == 2) { name##kernel##half<<<blockNum(size), blockMax>>>((half*)p1, (half*)p2, size, a1, a2); } \
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
    int cuda_##name(int type, void* p1, void* p2, unsigned int size, float a1, float a2, float a3) \
    { \
        if (type == 0) { name##kernel##float<<<blockNum(size), blockMax>>>((float*)p1, (float*)p2, size, a1, a2, a3); } \
        else if (type == 1) { name##kernel##double<<<blockNum(size), blockMax>>>((double*)p1, (double*)p2, size, a1, a2, a3); } \
        else if (type == 2) { name##kernel##half<<<blockNum(size), blockMax>>>((half*)p1, (half*)p2, size, a1, a2, a3); } \
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
    int cuda_##name(int type, void* p1, void* p2, void* p3, unsigned int size, float a1, float a2) \
    { \
        if (type == 0) { name##kernel##float<<<blockNum(size), blockMax>>>((float*)p1, (float*)p2, (float*)p3, size, a1, a2); } \
        else if (type == 1) { name##kernel##double<<<blockNum(size), blockMax>>>((double*)p1, (double*)p2, (double*)p3, size, a1, a2); } \
        else if (type == 2) { name##kernel##half<<<blockNum(size), blockMax>>>((half*)p1, (half*)p2, (half*)p3, size, a1, a2); } \
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
    int cuda_##name(int type, void* p1, void* p2, void* p3, unsigned int size, float a1, float a2, float a3) \
    { \
        if (type == 0) { name##kernel##float<<<blockNum(size), blockMax>>>((float*)p1, (float*)p2, (float*)p3, size, a1, a2, a3); } \
        else if (type == 1) { name##kernel##double<<<blockNum(size), blockMax>>>((double*)p1, (double*)p2, (double*)p3, size, a1, a2, a3); } \
        else if (type == 2) { name##kernel##half<<<blockNum(size), blockMax>>>((half*)p1, (half*)p2, (half*)p3, size, a1, a2, a3); } \
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
    int cuda_##name(int type, void* p1, void* p2, void* p3, void* p4, unsigned int size, float a1, float a2) \
    { \
        if (type == 0) { name##kernel##float<<<blockNum(size), blockMax>>>((float*)p1, (float*)p2, (float*)p3, (float*)p4, size, a1, a2); } \
        else if (type == 1) { name##kernel##double<<<blockNum(size), blockMax>>>((double*)p1, (double*)p2, (double*)p3, (double*)p4, size, a1, a2); } \
        else if (type == 2) { name##kernel##half<<<blockNum(size), blockMax>>>((half*)p1, (half*)p2, (half*)p3, (half*)p4, size, a1, a2); } \
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
    int cuda_##name(int type, void* p1, void* p2, void* p3, void* p4, unsigned int size, float a1, float a2, float a3) \
    { \
        if (type == 0) { name##kernel##float<<<blockNum(size), blockMax>>>((float*)p1, (float*)p2, (float*)p3, (float*)p4, size, a1, a2, a3); } \
        else if (type == 1) { name##kernel##double<<<blockNum(size), blockMax>>>((double*)p1, (double*)p2, (double*)p3, (double*)p4, size, a1, a2, a3); } \
        else if (type == 2) { name##kernel##half<<<blockNum(size), blockMax>>>((half*)p1, (half*)p2, (half*)p3, (half*)p4, size, a1, a2, a3); } \
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
    int cuda_##name(int type, void* p1, void* p2, void* p3, void* p4, unsigned int size, float a1, float a2, float a3, float a4) \
    { \
        if (type == 0) { name##kernel##float<<<blockNum(size), blockMax>>>((float*)p1, (float*)p2, (float*)p3, (float*)p4, size, a1, a2, a3, a4); } \
        else if (type == 1) { name##kernel##double<<<blockNum(size), blockMax>>>((double*)p1, (double*)p2, (double*)p3, (double*)p4, size, a1, a2, a3, a4); } \
        else if (type == 2) { name##kernel##half<<<blockNum(size), blockMax>>>((half*)p1, (half*)p2, (half*)p3, (half*)p4, size, a1, a2, a3, a4); } \
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
    int cuda_##name(int type, void* p1, void* p2, void* p3, void* p4, void* p5, void* p6, unsigned int size, float a1, float a2, float a3) \
    { \
        if (type == 0) { name##kernel##float<<<blockNum(size), blockMax>>>((float*)p1, (float*)p2, (float*)p3, (float*)p4, (float*)p5, (float*)p6, size, a1, a2, a3); } \
        else if (type == 1) { name##kernel##double<<<blockNum(size), blockMax>>>((double*)p1, (double*)p2, (double*)p3, (double*)p4, (double*)p5, (double*)p6, size, a1, a2, a3); } \
        else if (type == 2) { name##kernel##half<<<blockNum(size), blockMax>>>((half*)p1, (half*)p2, (half*)p3, (half*)p4, (half*)p5, (half*)p6, size, a1, a2, a3); } \
        return getError(#name); \
    }

CUDA_FUNCTION22(reciprocal,
    {
        p2[i] = a1 / (p1[i] + a2);
    });

CUDA_FUNCTION22(addnumber, { p2[i] = a1 + p1[i] * a2; });

CUDA_FUNCTION22(pow, { p2[i] = pow(p1[i] + a2, a1); });

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

CUDA_FUNCTION32(cross_entropy2, { p3[i] = -a2 * (p2[i] * log(p1[i] + a1) + (fp(1) - p2[i]) * log(fp(1) - p1[1] + a1)); });

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

CUDA_FUNCTION32(ada_update,
    {
        fp& rou = a1;
        fp& epsilon = a2;
        p2[i] = p2[i] * rou + p3[i] * p3[i] * (fp(1) - rou);
        p3[i] = p3[i] * sqrt((p1[i] + epsilon) / (p2[i] + epsilon));
        p1[i] = p1[i] * rou + p3[i] * p3[i] * (fp(1) - rou);
    });

CUDA_FUNCTION42(ada_delta_update,
    {
        fp& rou = a1;
        fp& epsilon = a2;
        p1[i] = p1[i] * rou + p3[i] * p3[i] * (fp(1) - rou);
        p4[i] = p3[i] * sqrt((p2[i] + epsilon) / (p1[i] + epsilon));
        p2[i] = p2[i] * rou + p4[i] * p4[i] * (fp(1) - rou);
    });

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
