
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
    cudaError_t cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "%s kernel launch failed: %s\n", content, cudaGetErrorString(cudaStatus));
        return 1;
    }
    return 0;
}

#define CUDA_FUNCTION22(name, function) \
    __global__ void name##kernel(real_cuda* p1, real_cuda* p2, unsigned int size, real_cuda a1, real_cuda a2) \
    { \
        int i = cal_i(); \
        if (i < size) \
        { \
            function \
        } \
    } \
    int name(real_cuda* p1, real_cuda* p2, unsigned int size, real_cuda a1, real_cuda a2) \
    { \
        name##kernel<<<blockNum(size), blockMax>>>(p1, p2, size, a1, a2); \
        return getError(#name); \
    }

#define CUDA_FUNCTION23(name, function) \
    __global__ void name##kernel(real_cuda* p1, real_cuda* p2, unsigned int size, real_cuda a1, real_cuda a2, real_cuda a3) \
    { \
        int i = cal_i(); \
        if (i < size) \
        { \
            function \
        } \
    } \
    int name(real_cuda* p1, real_cuda* p2, unsigned int size, real_cuda a1, real_cuda a2, real_cuda a3) \
    { \
        name##kernel<<<blockNum(size), blockMax>>>(p1, p2, size, a1, a2, a3); \
        return getError(#name); \
    }

#define CUDA_FUNCTION32(name, function) \
    __global__ void name##kernel(real_cuda* p1, real_cuda* p2, real_cuda* p3, unsigned int size, real_cuda a1, real_cuda a2) \
    { \
        int i = cal_i(); \
        if (i < size) \
        { \
            function \
        } \
    } \
    int name(real_cuda* p1, real_cuda* p2, real_cuda* p3, unsigned int size, real_cuda a1, real_cuda a2) \
    { \
        name##kernel<<<blockNum(size), blockMax>>>(p1, p2, p3, size, a1, a2); \
        return getError(#name); \
    }

#define CUDA_FUNCTION33(name, function) \
    __global__ void name##kernel(real_cuda* p1, real_cuda* p2, real_cuda* p3, unsigned int size, real_cuda a1, real_cuda a2, real_cuda a3) \
    { \
        int i = cal_i(); \
        if (i < size) \
        { \
            function \
        } \
    } \
    int name(real_cuda* p1, real_cuda* p2, real_cuda* p3, unsigned int size, real_cuda a1, real_cuda a2, real_cuda a3) \
    { \
        name##kernel<<<blockNum(size), blockMax>>>(p1, p2, p3, size, a1, a2, a3); \
        return getError(#name); \
    }

#define CUDA_FUNCTION42(name, function) \
    __global__ void name##kernel(real_cuda* p1, real_cuda* p2, real_cuda* p3, real_cuda* p4, unsigned int size, real_cuda a1, real_cuda a2) \
    { \
        int i = cal_i(); \
        if (i < size) \
        { \
            function \
        } \
    } \
    int name(real_cuda* p1, real_cuda* p2, real_cuda* p3, real_cuda* p4, unsigned int size, real_cuda a1, real_cuda a2) \
    { \
        name##kernel<<<blockNum(size), blockMax>>>(p1, p2, p3, p4, size, a1, a2); \
        return getError(#name); \
    }

#define CUDA_FUNCTION43(name, function) \
    __global__ void name##kernel(real_cuda* p1, real_cuda* p2, real_cuda* p3, real_cuda* p4, unsigned int size, real_cuda a1, real_cuda a2, real_cuda a3) \
    { \
        int i = cal_i(); \
        if (i < size) \
        { \
            function \
        } \
    } \
    int name(real_cuda* p1, real_cuda* p2, real_cuda* p3, real_cuda* p4, unsigned int size, real_cuda a1, real_cuda a2, real_cuda a3) \
    { \
        name##kernel<<<blockNum(size), blockMax>>>(p1, p2, p3, p4, size, a1, a2, a3); \
        return getError(#name); \
    }

#define CUDA_FUNCTION44(name, function) \
    __global__ void name##kernel(real_cuda* p1, real_cuda* p2, real_cuda* p3, real_cuda* p4, unsigned int size, real_cuda a1, real_cuda a2, real_cuda a3, real_cuda a4) \
    { \
        int i = cal_i(); \
        if (i < size) \
        { \
            function \
        } \
    } \
    int name(real_cuda* p1, real_cuda* p2, real_cuda* p3, real_cuda* p4, unsigned int size, real_cuda a1, real_cuda a2, real_cuda a3, real_cuda a4) \
    { \
        name##kernel<<<blockNum(size), blockMax>>>(p1, p2, p3, p4, size, a1, a2, a3, a4); \
        return getError(#name); \
    }

CUDA_FUNCTION22(cuda_reciprocal, { p2[i] = a1 / (p1[i] + a2); });

CUDA_FUNCTION22(cuda_addnumber, { p2[i] = a1 + p1[i] * a2; });

CUDA_FUNCTION22(cuda_pow, { p2[i] = pow(p1[i] + a2, a1); });

CUDA_FUNCTION22(cuda_sparse,
    {
        p2[i] = ((real_cuda(1) - a1) / (real_cuda(1) - p1[i]) - a1 / p1[i]) * a2;
    });

CUDA_FUNCTION22(cuda_sign,
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

CUDA_FUNCTION32(cuda_cross_entropy, { p3[i] = -a2 * p2[i] * log(p1[i] + a1); });

CUDA_FUNCTION32(cuda_cross_entropy2, { p3[i] = -a2 * (p2[i] * log(p1[i] + a1) + (real_cuda(1) - p2[i]) * log(real_cuda(1) - p1[1] + a1)); });

CUDA_FUNCTION32(cuda_add, { p3[i] = p1[i] * a1 + p2[i] * a2; });

CUDA_FUNCTION32(cuda_mul, { p3[i] = p1[i] * p2[i] * a1 + p3[i] * a2; });

CUDA_FUNCTION33(cuda_div, { p3[i] = a3 * (p1[i] + a1) / (p2[i] + a2); });

CUDA_FUNCTION32(cuda_sectionlimit,
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

CUDA_FUNCTION32(cuda_ada_update,
    {
        real_cuda& rou = a1;
        real_cuda& epsilon = a2;
        p2[i] = p2[i] * rou + p3[i] * p3[i] * (real_cuda(1) - rou);
        p3[i] = p3[i] * sqrt((p1[i] + epsilon) / (p2[i] + epsilon));
        p1[i] = p1[i] * rou + p3[i] * p3[i] * (real_cuda(1) - rou);
    });

CUDA_FUNCTION42(cuda_ada_delta_update,
    {
        real_cuda& rou = a1;
        real_cuda& epsilon = a2;
        p1[i] = p1[i] * rou + p3[i] * p3[i] * (real_cuda(1) - rou);
        p4[i] = p3[i] * sqrt((p2[i] + epsilon) / (p1[i] + epsilon));
        p2[i] = p2[i] * rou + p4[i] * p4[i] * (real_cuda(1) - rou);
    });

CUDA_FUNCTION44(cuda_adam_update,
    {
        real_cuda& beta1 = a1;
        real_cuda& beta2 = a2;
        real_cuda& epsilon = a3;
        real_cuda& t = a4;
        p1[i] = p1[i] * beta1 + p3[i] * (real_cuda(1) - beta1);
        p2[i] = p2[i] * beta2 + p3[i] * p3[i] * (real_cuda(1) - beta2);
        p4[i] = p3[i] / (real_cuda(1) - pow(beta1, t)) / (sqrt(p2[i] / (real_cuda(1) - pow(beta2, t))) + epsilon);
    });

CUDA_FUNCTION32(cuda_rms_prop_update,
    {
        real_cuda& rou = a1;
        real_cuda& epsilon = a2;
        p1[i] = p1[i] * rou + p2[i] * p2[i] * (real_cuda(1) - rou);
        p3[i] = p2[i] / sqrt(p1[i] + epsilon);
    });

#if REAL_PRECISION == 2
__global__ void cuda_half2floatkernel(half* p1, float* p2, unsigned int size)
{
    int i = cal_i();
    if (i < size)
    {
        p2[i] = p1[i];
    }
}
int cuda_half2float(half* p1, float* p2, unsigned int size)
{
    cuda_half2floatkernel<<<blockNum(size), blockMax>>>(p1, p2, size);
    return getError("cuda_half2float");
}
#endif

CUDA_FUNCTION22(cuda_sin,
    {
        p2[i] = sin(a1 * p1[i] + a2);
    });

CUDA_FUNCTION22(cuda_cos,
    {
        p2[i] = cos(a1 * p1[i] + a2);
    });

CUDA_FUNCTION22(cuda_zigzag,
    {
        p2[i] = a1 * (p1[i] + a2 - 2 * floor((p1[i] + a2 - 1) / 2) - 2);
    });

CUDA_FUNCTION42(cuda_zigzagb,
    {
        if (abs(p1[i]) > 1 - 1e-2)
        {
            p2[i] = -p4[i] * 100;
            return;
        }
        p2[i] = p4[i];
    });

CUDA_FUNCTION22(cuda_step,
    {
        p2[i] = round(p1[i] * 256) / 256;
    });

CUDA_FUNCTION23(cuda_leaky_relu,
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

CUDA_FUNCTION43(cuda_leaky_relub,
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