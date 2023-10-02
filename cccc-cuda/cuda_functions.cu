
#include "cuda_functions.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

#if REAL_PRECISION == 2
#define pow(x, a) hexp(half(a) * hlog(x))
#define log hlog
#define sqrt hsqrt
#endif

#define blockMax 1024

#define cal_i() (blockIdx.x * blockDim.x + threadIdx.x)

inline int blockNum(unsigned int size) { return (size + blockMax - 1) / blockMax; }

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

#define CUDA_FUNCTION22(name, function) \
    static __global__ void name##kernel(realc* p1, realc* p2, unsigned int size, realc a1, realc a2) \
    { \
        int i = cal_i(); \
        if (i < size) \
        { \
            function \
        } \
    } \
    int cuda_##name(realc* p1, realc* p2, unsigned int size, realc a1, realc a2) \
    { \
        name##kernel<<<blockNum(size), blockMax>>>(p1, p2, size, a1, a2); \
        return getError(#name); \
    }

#define CUDA_FUNCTION23(name, function) \
    static __global__ void name##kernel(realc* p1, realc* p2, unsigned int size, realc a1, realc a2, realc a3) \
    { \
        int i = cal_i(); \
        if (i < size) \
        { \
            function \
        } \
    } \
    int cuda_##name(realc* p1, realc* p2, unsigned int size, realc a1, realc a2, realc a3) \
    { \
        name##kernel<<<blockNum(size), blockMax>>>(p1, p2, size, a1, a2, a3); \
        return getError(#name); \
    }

#define CUDA_FUNCTION32(name, function) \
    static __global__ void name##kernel(realc* p1, realc* p2, realc* p3, unsigned int size, realc a1, realc a2) \
    { \
        int i = cal_i(); \
        if (i < size) \
        { \
            function \
        } \
    } \
    int cuda_##name(realc* p1, realc* p2, realc* p3, unsigned int size, realc a1, realc a2) \
    { \
        name##kernel<<<blockNum(size), blockMax>>>(p1, p2, p3, size, a1, a2); \
        return getError(#name); \
    }

#define CUDA_FUNCTION33(name, function) \
    static __global__ void name##kernel(realc* p1, realc* p2, realc* p3, unsigned int size, realc a1, realc a2, realc a3) \
    { \
        int i = cal_i(); \
        if (i < size) \
        { \
            function \
        } \
    } \
    int cuda_##name(realc* p1, realc* p2, realc* p3, unsigned int size, realc a1, realc a2, realc a3) \
    { \
        name##kernel<<<blockNum(size), blockMax>>>(p1, p2, p3, size, a1, a2, a3); \
        return getError(#name); \
    }

#define CUDA_FUNCTION42(name, function) \
    static __global__ void name##kernel(realc* p1, realc* p2, realc* p3, realc* p4, unsigned int size, realc a1, realc a2) \
    { \
        int i = cal_i(); \
        if (i < size) \
        { \
            function \
        } \
    } \
    int cuda_##name(realc* p1, realc* p2, realc* p3, realc* p4, unsigned int size, realc a1, realc a2) \
    { \
        name##kernel<<<blockNum(size), blockMax>>>(p1, p2, p3, p4, size, a1, a2); \
        return getError(#name); \
    }

#define CUDA_FUNCTION43(name, function) \
    static __global__ void name##kernel(realc* p1, realc* p2, realc* p3, realc* p4, unsigned int size, realc a1, realc a2, realc a3) \
    { \
        int i = cal_i(); \
        if (i < size) \
        { \
            function \
        } \
    } \
    int cuda_##name(realc* p1, realc* p2, realc* p3, realc* p4, unsigned int size, realc a1, realc a2, realc a3) \
    { \
        name##kernel<<<blockNum(size), blockMax>>>(p1, p2, p3, p4, size, a1, a2, a3); \
        return getError(#name); \
    }

#define CUDA_FUNCTION44(name, function) \
    static __global__ void name##kernel(realc* p1, realc* p2, realc* p3, realc* p4, unsigned int size, realc a1, realc a2, realc a3, realc a4) \
    { \
        int i = cal_i(); \
        if (i < size) \
        { \
            function \
        } \
    } \
    int cuda_##name(realc* p1, realc* p2, realc* p3, realc* p4, unsigned int size, realc a1, realc a2, realc a3, realc a4) \
    { \
        name##kernel<<<blockNum(size), blockMax>>>(p1, p2, p3, p4, size, a1, a2, a3, a4); \
        return getError(#name); \
    }

#define CUDA_FUNCTION63(name, function) \
    static __global__ void name##kernel(realc* p1, realc* p2, realc* p3, realc* p4, realc* p5, realc* p6, unsigned int size, realc a1, realc a2, realc a3) \
    { \
        int i = cal_i(); \
        if (i < size) \
        { \
            function \
        } \
    } \
    int cuda_##name(realc* p1, realc* p2, realc* p3, realc* p4, realc* p5, realc* p6, unsigned int size, realc a1, realc a2, realc a3) \
    { \
        name##kernel<<<blockNum(size), blockMax>>>(p1, p2, p3, p4, p5, p6, size, a1, a2, a3); \
        return getError(#name); \
    }

CUDA_FUNCTION22(reciprocal, { p2[i] = a1 / (p1[i] + a2); });

CUDA_FUNCTION22(addnumber, { p2[i] = a1 + p1[i] * a2; });

CUDA_FUNCTION22(pow, { p2[i] = pow(p1[i] + a2, a1); });

CUDA_FUNCTION22(sparse,
    {
        p2[i] = ((realc(1) - a1) / (realc(1) - p1[i]) - a1 / p1[i]) * a2;
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

CUDA_FUNCTION32(cross_entropy2, { p3[i] = -a2 * (p2[i] * log(p1[i] + a1) + (realc(1) - p2[i]) * log(realc(1) - p1[1] + a1)); });

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
        realc& rou = a1;
        realc& epsilon = a2;
        p2[i] = p2[i] * rou + p3[i] * p3[i] * (realc(1) - rou);
        p3[i] = p3[i] * sqrt((p1[i] + epsilon) / (p2[i] + epsilon));
        p1[i] = p1[i] * rou + p3[i] * p3[i] * (realc(1) - rou);
    });

CUDA_FUNCTION42(ada_delta_update,
    {
        realc& rou = a1;
        realc& epsilon = a2;
        p1[i] = p1[i] * rou + p3[i] * p3[i] * (realc(1) - rou);
        p4[i] = p3[i] * sqrt((p2[i] + epsilon) / (p1[i] + epsilon));
        p2[i] = p2[i] * rou + p4[i] * p4[i] * (realc(1) - rou);
    });

CUDA_FUNCTION44(adam_update,
    {
        realc& beta1 = a1;
        realc& beta2 = a2;
        realc& epsilon = a3;
        realc& t = a4;
        p1[i] = p1[i] * beta1 + p3[i] * (realc(1) - beta1);
        p2[i] = p2[i] * beta2 + p3[i] * p3[i] * (realc(1) - beta2);
        p4[i] = p1[i] / (realc(1) - pow(beta1, t)) / (sqrt(p2[i] / (realc(1) - pow(beta2, t))) + epsilon);
    });

CUDA_FUNCTION32(rms_prop_update,
    {
        realc& rou = a1;
        realc& epsilon = a2;
        p1[i] = p1[i] * rou + p2[i] * p2[i] * (realc(1) - rou);
        p3[i] = p2[i] / sqrt(p1[i] + epsilon);
    });

#if REAL_PRECISION == 2
__global__ void half2floatkernel(half* p1, float* p2, unsigned int size)
{
    int i = cal_i();
    if (i < size)
    {
        p2[i] = p1[i];
    }
}
int half2float(half* p1, float* p2, unsigned int size)
{
    half2floatkernel<<<blockNum(size), blockMax>>>(p1, p2, size);
    return getError("half2float");
}
#endif

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
        p2[i] = a1 * (p1[i] + a2 - 2 * floor((p1[i] + a2 - 1) / 2) - 2);
    });

CUDA_FUNCTION42(zigzagb,
    {
        if (abs(p1[i]) > 1 - 1e-2)
        {
            p2[i] = -p4[i] * 100;
            return;
        }
        p2[i] = p4[i];
    });

CUDA_FUNCTION22(step,
    {
        p2[i] = round(p1[i] * 256) / 256;
    });

CUDA_FUNCTION23(leaky_relu,
    {
        if (p1[i] >= 0)
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
        if (p1[i] >= 0)
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
        p3[i] = max(p1[i], p2[i]);
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