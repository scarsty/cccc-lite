#pragma once

#include "Log.h"
#include "Random.h"
#include "cblas_real.h"
#include "cublas_real.h"
#include "cuda_fp16.h"
#include "cuda_runtime.h"
#include "cuda_functions.h"
#include <algorithm>
#include <cstdio>

#if REAL_PRECISION == 2

namespace cccc
{

struct cccc_half : public __half
{
    cccc_half() {}
    cccc_half(const __half a)
    {
        *this = *(cccc_half*)(&a);
    }
    cccc_half(const int a)
    {
        auto f = __float2half(a);
        *this = *(cccc_half*)(&f);
    }
    cccc_half(const float a)
    {
        auto f = __float2half(a);
        *this = *(cccc_half*)(&f);
    }
    cccc_half(const double a)
    {
        auto f = __float2half(a);
        *this = *(cccc_half*)(&f);
    }
    //operator int() const { return __half2float(*this); }
    //operator float() const { return __half2float(*this); }
    //operator double() const { return __half2float(*this); }
};

inline cccc_half operator+(const cccc_half& lh, const cccc_half& rh) { return __float2half(__half2float(lh) + __half2float(rh)); }
inline cccc_half operator-(const cccc_half& lh, const cccc_half& rh) { return __float2half(__half2float(lh) - __half2float(rh)); }
inline cccc_half operator*(const cccc_half& lh, const cccc_half& rh) { return __float2half(__half2float(lh) * __half2float(rh)); }
inline cccc_half operator/(const cccc_half& lh, const cccc_half& rh) { return __float2half(__half2float(lh) / __half2float(rh)); }

inline cccc_half operator+(const int& lh, const cccc_half& rh) { return __float2half(lh + __half2float(rh)); }
inline cccc_half operator+(const cccc_half& lh, const int& rh) { return __float2half(__half2float(lh) + rh); }
inline double operator+(const double& lh, const cccc_half& rh) { return lh + __half2float(rh); }
inline double operator+(const cccc_half& lh, const double& rh) { return __half2float(lh) + rh; }

inline cccc_half operator-(const int& lh, const cccc_half& rh) { return __float2half(lh - __half2float(rh)); }
inline cccc_half operator-(const cccc_half& lh, const int& rh) { return __float2half(__half2float(lh) - rh); }
inline double operator-(const double& lh, const cccc_half& rh) { return lh - __half2float(rh); }
inline double operator-(const cccc_half& lh, const double& rh) { return __half2float(lh) - rh; }

inline cccc_half operator*(const int& lh, const cccc_half& rh) { return __float2half(lh * __half2float(rh)); }
inline cccc_half operator*(const cccc_half& lh, const int& rh) { return __float2half(__half2float(lh) * rh); }
inline double operator*(const double& lh, const cccc_half& rh) { return lh * __half2float(rh); }
inline double operator*(const cccc_half& lh, const double& rh) { return __half2float(lh) * rh; }
inline float operator*(const float& lh, const cccc_half& rh) { return lh * __half2float(rh); }
inline float operator*(const cccc_half& lh, const float& rh) { return __half2float(lh) * rh; }

inline cccc_half operator/(const int& lh, const cccc_half& rh) { return __float2half(lh / __half2float(rh)); }
inline cccc_half operator/(const cccc_half& lh, const int& rh) { return __float2half(__half2float(lh) / rh); }
inline double operator/(const double& lh, const cccc_half& rh) { return lh / __half2float(rh); }
inline double operator/(const cccc_half& lh, const double& rh) { return __half2float(lh) / rh; }
inline cccc_half operator/(const cccc_half& lh, const int64_t& rh) { return __float2half(__half2float(lh) / rh); }
inline cccc_half operator/(const uint8_t& lh, const cccc_half& rh) { return __float2half(lh / __half2float(rh)); }

inline cccc_half& operator+=(cccc_half& lh, const cccc_half& rh) { return lh = lh + rh; }
inline cccc_half& operator-=(cccc_half& lh, const cccc_half& rh) { return lh = lh - rh; }
inline cccc_half& operator*=(cccc_half& lh, const cccc_half& rh) { return lh = lh * rh; }
inline cccc_half& operator/=(cccc_half& lh, const cccc_half& rh) { return lh = lh / rh; }

inline cccc_half sqrt(const cccc_half& lh) { return sqrt(__half2float(lh)); }
inline cccc_half log(const cccc_half& lh) { return log(__half2float(lh)); }
inline cccc_half exp(const cccc_half& lh) { return exp(__half2float(lh)); }
inline cccc_half pow(const cccc_half& lh, const cccc_half& rh) { return pow(__half2float(lh), __half2float(rh)); }
inline cccc_half pow(const cccc_half& lh, const double& rh) { return pow(__half2float(lh), rh); }

class CublasHalf : public Cublas
{
protected:
    float* buffer_ = nullptr;
    const size_t buffer_size_ = 10000000;

public:
    CublasHalf() { cudaMalloc((void**)&buffer_, buffer_size_ * sizeof(float)); }
    ~CublasHalf() { cudaFree(buffer_); }

public:
    CUBLAS_FUNCTION float dot(const int N, const cccc_half* X, const int incX, const cccc_half* Y, const int incY)
    {
        cccc_half r;
        cublasDotEx(handle_, N, X, CUDA_R_16F, incX, Y, CUDA_R_16F, incY, &r, CUDA_R_16F, CUDA_R_32F);
        return r;
    }
    CUBLAS_FUNCTION float asum(const int N, const cccc_half* X, const int incX)
    {
        float r = 0;
        if (N * incX < buffer_size_)
        {
            cuda_half2float((cccc_half*)X, buffer_, N * incX);
            cublasSasum(handle_, N, buffer_, incX, &r);
        }
        return r;
    }
    CUBLAS_FUNCTION int iamax(const int N, const cccc_half* X, const int incX)
    {
        int r = 1;
        if (N * incX < buffer_size_)
        {
            cuda_half2float((cccc_half*)X, buffer_, N * incX);
            cublasIsamax(handle_, N, buffer_, incX, &r);
        }
        return r - 1;
    }
    CUBLAS_FUNCTION void scal(const int N, const cccc_half alpha, cccc_half* X, const int incX)
    {
        cublasScalEx(handle_, N, &alpha, CUDA_R_16F, X, CUDA_R_16F, incX, CUDA_R_32F);
    }
    CUBLAS_FUNCTION void gemv(const MatrixTransType TransA, const int M, const int N, const cccc_half alpha, const cccc_half* A, const int lda, const cccc_half* X, const int incX, const cccc_half beta, cccc_half* Y, const int incY)
    {
        cublasHgemm(handle_, get_trans(TransA), CUBLAS_OP_N, M, 1, N, &alpha, A, lda, X, N, &beta, Y, lda);
    }
    CUBLAS_FUNCTION void gemm(const MatrixTransType TransA, const MatrixTransType TransB, const int M, const int N, const int K, const cccc_half alpha, const cccc_half* A, const int lda, const cccc_half* B, const int ldb, const cccc_half beta, cccc_half* C, const int ldc)
    {
        cublasHgemm(handle_, get_trans(TransA), get_trans(TransB), M, N, K, &alpha, A, lda, B, ldb, &beta, C, ldc);
    }
};

class CblasHalf : public Cblas
{
public:
    CBLAS_FUNCTION float dot(const int N, const cccc_half* X, const int incX, const cccc_half* Y, const int incY)
    {
        LOG(stderr, "Unsupported dot for fp16 on CPU\n");
        return 0;
    }
    CBLAS_FUNCTION float asum(const int N, const cccc_half* X, const int incX)
    {
        LOG(stderr, "Unsupported asum for fp16 on CPU\n");
        return 0;
    }
    CBLAS_FUNCTION int iamax(const int N, const cccc_half* X, const int incX)
    {
        int p = 0;
        cccc_half v = X[0];
        for (int i = 1; i < N; i++)
        {
            if (X[i * incX] > v)
            {
                p = i;
                v = X[i * incX];
            }
        }
        return p;
    }
    CBLAS_FUNCTION void scal(const int N, const cccc_half alpha, cccc_half* X, const int incX)
    {
        LOG(stderr, "Unsupported scal for fp16 on CPU\n");
    }
    CBLAS_FUNCTION void gemv(const MatrixTransType TransA, const int M, const int N, const cccc_half alpha, const cccc_half* A, const int lda, const cccc_half* X, const int incX, const cccc_half beta, cccc_half* Y, const int incY)
    {
        LOG(stderr, "Unsupported gemv for fp16 on CPU\n");
    }
    CBLAS_FUNCTION void gemm(const MatrixTransType TransA, const MatrixTransType TransB, const int M, const int N, const int K, const cccc_half alpha, const cccc_half* A, const int lda, const cccc_half* B, const int ldb, const cccc_half beta, cccc_half* C, const int ldc)
    {
        LOG(stderr, "Unsupported gemm for fp16 on CPU\n");
    }
};

#define Cublas CublasHalf
#define Cblas CblasHalf

}    // namespace cccc

namespace std
{
inline cccc::cccc_half max(const cccc::cccc_half& lh, const double& rh)
{
    return lh > cccc::cccc_half(rh) ? lh : cccc::cccc_half(rh);
}
inline cccc::cccc_half max(const cccc::cccc_half& lh, const cccc::cccc_half& rh)
{
    return lh > rh ? lh : rh;
}
}    // namespace std

template <>
class Random<cccc::cccc_half> : public Random<float>
{
public:
    void set_parameter(cccc::cccc_half a, cccc::cccc_half b)
    {
        uniform_dist_.param(decltype(uniform_dist_.param())(__half2float(a), __half2float(b)));
        normal_dist_.param(decltype(normal_dist_.param())(__half2float(a), __half2float(b)));
    }
    cccc::cccc_half rand()
    {
        if (type_ == RANDOM_UNIFORM)
        {
            return uniform_dist_(generator_);
        }
        else if (type_ == RANDOM_NORMAL)
        {
            return normal_dist_(generator_);
        }
        return 0;
    }
};

#endif