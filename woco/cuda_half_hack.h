#pragma once

#include "Random.h"
#include "cblas_real.h"
#include "cublas_real.h"
#include "cuda_fp16.h"
#include "cuda_runtime.h"
#include "woco_cuda.h"
#include <algorithm>
#include <cstdio>

#if REAL_PRECISION == 2

namespace woco
{

struct Half : public __half
{
    Half() {}
    Half(const __half a)
    {
        *this = *(Half*)(&a);
    }
    Half(const int a)
    {
        auto f = __float2half(a);
        *this = *(Half*)(&f);
    }
    Half(const float a)
    {
        auto f = __float2half(a);
        *this = *(Half*)(&f);
    }
    Half(const double a)
    {
        auto f = __float2half(a);
        *this = *(Half*)(&f);
    }
    //operator int() const { return __half2float(*this); }
    //operator float() const { return __half2float(*this); }
    //operator double() const { return __half2float(*this); }
};

inline Half operator+(const Half& lh, const Half& rh) { return __float2half(__half2float(lh) + __half2float(rh)); }
inline Half operator-(const Half& lh, const Half& rh) { return __float2half(__half2float(lh) - __half2float(rh)); }
inline Half operator*(const Half& lh, const Half& rh) { return __float2half(__half2float(lh) * __half2float(rh)); }
inline Half operator/(const Half& lh, const Half& rh) { return __float2half(__half2float(lh) / __half2float(rh)); }

inline Half operator+(const int& lh, const Half& rh) { return __float2half(lh + __half2float(rh)); }
inline Half operator+(const Half& lh, const int& rh) { return __float2half(__half2float(lh) + rh); }
inline double operator+(const double& lh, const Half& rh) { return lh + __half2float(rh); }
inline double operator+(const Half& lh, const double& rh) { return __half2float(lh) + rh; }

inline Half operator-(const int& lh, const Half& rh) { return __float2half(lh - __half2float(rh)); }
inline Half operator-(const Half& lh, const int& rh) { return __float2half(__half2float(lh) - rh); }
inline double operator-(const double& lh, const Half& rh) { return lh - __half2float(rh); }
inline double operator-(const Half& lh, const double& rh) { return __half2float(lh) - rh; }

inline Half operator*(const int& lh, const Half& rh) { return __float2half(lh * __half2float(rh)); }
inline Half operator*(const Half& lh, const int& rh) { return __float2half(__half2float(lh) * rh); }
inline double operator*(const double& lh, const Half& rh) { return lh * __half2float(rh); }
inline double operator*(const Half& lh, const double& rh) { return __half2float(lh) * rh; }
inline float operator*(const float& lh, const Half& rh) { return lh * __half2float(rh); }
inline float operator*(const Half& lh, const float& rh) { return __half2float(lh) * rh; }

inline Half operator/(const int& lh, const Half& rh) { return __float2half(lh / __half2float(rh)); }
inline Half operator/(const Half& lh, const int& rh) { return __float2half(__half2float(lh) / rh); }
inline double operator/(const double& lh, const Half& rh) { return lh / __half2float(rh); }
inline double operator/(const Half& lh, const double& rh) { return __half2float(lh) / rh; }
inline Half operator/(const Half& lh, const int64_t& rh) { return __float2half(__half2float(lh) / rh); }
inline Half operator/(const uint8_t& lh, const Half& rh) { return __float2half(lh / __half2float(rh)); }

inline Half& operator+=(Half& lh, const Half& rh) { return lh = lh + rh; }
inline Half& operator-=(Half& lh, const Half& rh) { return lh = lh - rh; }
inline Half& operator*=(Half& lh, const Half& rh) { return lh = lh * rh; }
inline Half& operator/=(Half& lh, const Half& rh) { return lh = lh / rh; }

inline Half sqrt(const Half& lh) { return sqrt(__half2float(lh)); }
inline Half log(const Half& lh) { return log(__half2float(lh)); }
inline Half exp(const Half& lh) { return exp(__half2float(lh)); }
inline Half pow(const Half& lh, const Half& rh) { return pow(__half2float(lh), __half2float(rh)); }
inline Half pow(const Half& lh, const double& rh) { return pow(__half2float(lh), rh); }

class CublasHalf : public Cublas
{
protected:
    float* buffer_ = nullptr;
    const size_t buffer_size_ = 10000000;

public:
    CublasHalf() { cudaMalloc((void**)&buffer_, buffer_size_ * sizeof(float)); }
    ~CublasHalf() { cudaFree(buffer_); }

public:
    CUBLAS_FUNCTION float dot(const int N, const Half* X, const int incX, const Half* Y, const int incY)
    {
        Half r;
        cublasDotEx(handle_, N, X, CUDA_R_16F, incX, Y, CUDA_R_16F, incY, &r, CUDA_R_16F, CUDA_R_32F);
        return r;
    }
    CUBLAS_FUNCTION float asum(const int N, const Half* X, const int incX)
    {
        float r = 0;
        if (N * incX < buffer_size_)
        {
            cuda_half2float((Half*)X, buffer_, N * incX);
            cublasSasum(handle_, N, buffer_, incX, &r);
        }
        return r;
    }
    CUBLAS_FUNCTION int iamax(const int N, const Half* X, const int incX)
    {
        int r = 1;
        if (N * incX < buffer_size_)
        {
            cuda_half2float((Half*)X, buffer_, N * incX);
            cublasIsamax(handle_, N, buffer_, incX, &r);
        }
        return r - 1;
    }
    CUBLAS_FUNCTION void scal(const int N, const Half alpha, Half* X, const int incX)
    {
        cublasScalEx(handle_, N, &alpha, CUDA_R_16F, X, CUDA_R_16F, incX, CUDA_R_32F);
    }
    CUBLAS_FUNCTION void gemv(const MatrixTransType TransA, const int M, const int N, const Half alpha, const Half* A, const int lda, const Half* X, const int incX, const Half beta, Half* Y, const int incY)
    {
        cublasHgemm(handle_, get_trans(TransA), CUBLAS_OP_N, M, 1, N, &alpha, A, lda, X, N, &beta, Y, lda);
    }
    CUBLAS_FUNCTION void gemm(const MatrixTransType TransA, const MatrixTransType TransB, const int M, const int N, const int K, const Half alpha, const Half* A, const int lda, const Half* B, const int ldb, const Half beta, Half* C, const int ldc)
    {
        cublasHgemm(handle_, get_trans(TransA), get_trans(TransB), M, N, K, &alpha, A, lda, B, ldb, &beta, C, ldc);
    }
};

class CblasHalf : public Cblas
{
public:
    CBLAS_FUNCTION float dot(const int N, const Half* X, const int incX, const Half* Y, const int incY)
    {
        fprintf(stderr, "Unsupported dot for fp16 on CPU\n");
        return 0;
    }
    CBLAS_FUNCTION float asum(const int N, const Half* X, const int incX)
    {
        fprintf(stderr, "Unsupported asum for fp16 on CPU\n");
        return 0;
    }
    CBLAS_FUNCTION int iamax(const int N, const Half* X, const int incX)
    {
        int p = 0;
        Half v = X[0];
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
    CBLAS_FUNCTION void scal(const int N, const Half alpha, Half* X, const int incX)
    {
        fprintf(stderr, "Unsupported scal for fp16 on CPU\n");
    }
    CBLAS_FUNCTION void gemv(const MatrixTransType TransA, const int M, const int N, const Half alpha, const Half* A, const int lda, const Half* X, const int incX, const Half beta, Half* Y, const int incY)
    {
        fprintf(stderr, "Unsupported gemv for fp16 on CPU\n");
    }
    CBLAS_FUNCTION void gemm(const MatrixTransType TransA, const MatrixTransType TransB, const int M, const int N, const int K, const Half alpha, const Half* A, const int lda, const Half* B, const int ldb, const Half beta, Half* C, const int ldc)
    {
        fprintf(stderr, "Unsupported gemm for fp16 on CPU\n");
    }
};

#define Cublas CublasHalf
#define Cblas CblasHalf

}    // namespace woco

namespace std
{
inline woco::Half max(const woco::Half& lh, const double& rh)
{
    return lh > woco::Half(rh) ? lh : woco::Half(rh);
}
inline woco::Half max(const woco::Half& lh, const woco::Half& rh)
{
    return lh > rh ? lh : rh;
}
}    // namespace std

template <>
class Random<woco::Half> : public Random<float>
{
public:
    void set_parameter(woco::Half a, woco::Half b)
    {
        uniform_dist_.param(decltype(uniform_dist_.param())(__half2float(a), __half2float(b)));
        normal_dist_.param(decltype(normal_dist_.param())(__half2float(a), __half2float(b)));
    }
    woco::Half rand()
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