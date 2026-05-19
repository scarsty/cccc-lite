#pragma once
#include "blas_types.h"
#include "cuda_functions.h"

#if ENABLE_CUDA
#include "cublas_v2.h"
#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace cccc
{

#if defined(NORMAL_BLAS)
#define CUBLAS_FUNCTION inline
#elif defined(STATIC_BLAS)
#define CUBLAS_FUNCTION static inline
#else
#define CUBLAS_FUNCTION virtual
#endif

//Class of cublas, overload functions with the same name for float and double.
class Cublas : Blas
{
public:
    Cublas() {}
    ~Cublas() { destroy(); }

protected:
    CUBLAS_FUNCTION cublasOperation_t get_trans(MatrixTransType t)
    {
        return t == MATRIX_NO_TRANS ? CUBLAS_OP_N : CUBLAS_OP_T;
    }
    CUBLAS_FUNCTION cublasFillMode_t get_uplo(MatrixFillType t)
    {
        return t == MATRIX_UPPER ? CUBLAS_FILL_MODE_UPPER : CUBLAS_FILL_MODE_LOWER;
    }
    CUBLAS_FUNCTION cublasDiagType_t get_diag(MatrixDiagType t)
    {
        return t == MATRIX_NON_UNIT ? CUBLAS_DIAG_NON_UNIT : CUBLAS_DIAG_UNIT;
    }
    CUBLAS_FUNCTION cublasSideMode_t get_side(MatrixSideType t)
    {
        return t == MATRIX_LEFT ? CUBLAS_SIDE_LEFT : CUBLAS_SIDE_RIGHT;
    }

public:
    CUBLAS_FUNCTION cublasStatus_t init()
    {
        auto r = cublasCreate(&handle_);
        cublasSetMathMode(handle_, CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION);
        // cublasGemmStridedBatchedEx with cublasComputeType_t is not exported by older cublas.lib;
        // the project's cublas_v2.h doesn't declare it either, so we must load at runtime.
        static const char* cublas_libs[] = { "cublas64_13", "cublas64_12", "cublas64_11", nullptr };
        for (int i = 0; cublas_libs[i]; i++)
        {
            HMODULE h = GetModuleHandleA(cublas_libs[i]);
            if (!h) h = LoadLibraryA(cublas_libs[i]);
            if (h)
            {
                cublasGemmEx_fp_ = reinterpret_cast<cublasGemmEx_t>(
                    GetProcAddress(h, "cublasGemmEx"));
                cublasGemmStridedBatchedEx_fp_ = reinterpret_cast<cublasGemmStridedBatchedEx_t>(
                    GetProcAddress(h, "cublasGemmStridedBatchedEx"));
                if (cublasGemmStridedBatchedEx_fp_) break;
            }
        }
        return r;
    }
    CUBLAS_FUNCTION void destroy()
    {
        if (handle_)
        {
            cublasDestroy(handle_);
        }
    }
    CUBLAS_FUNCTION void set_handle(cublasHandle_t h) { handle_ = h; }
    CUBLAS_FUNCTION int get_version()
    {
        int ver;
        cublasGetVersion(handle_, &ver);
        return ver;
    }

public:
    CUBLAS_FUNCTION float dot(const int N, const float* X, const int incX, const float* Y, const int incY)
    {
        float r;
        cublasSdot(handle_, N, X, incX, Y, incY, &r);
        return r;
    }
    CUBLAS_FUNCTION double dot(const int N, const double* X, const int incX, const double* Y, const int incY)
    {
        double r;
        cublasDdot(handle_, N, X, incX, Y, incY, &r);
        return r;
    }
    CUBLAS_FUNCTION float nrm2(const int N, const float* X, const int incX)
    {
        float r;
        cublasSnrm2(handle_, N, X, incX, &r);
        return r;
    }
    CUBLAS_FUNCTION float asum(const int N, const float* X, const int incX)
    {
        float r;
        cublasSasum(handle_, N, X, incX, &r);
        return r;
    }
    CUBLAS_FUNCTION double nrm2(const int N, const double* X, const int incX)
    {
        double r;
        cublasDnrm2(handle_, N, X, incX, &r);
        return r;
    }
    CUBLAS_FUNCTION double asum(const int N, const double* X, const int incX)
    {
        double r;
        cublasDasum(handle_, N, X, incX, &r);
        return r;
    }
    CUBLAS_FUNCTION int iamax(const int N, const float* X, const int incX)
    {
        int r;
        cublasIsamax(handle_, N, X, incX, &r);
        return r - 1;
    }
    CUBLAS_FUNCTION int iamax(const int N, const double* X, const int incX)
    {
        int r;
        cublasIdamax(handle_, N, X, incX, &r);
        return r - 1;
    }
    CUBLAS_FUNCTION void swap(const int N, float* X, const int incX, float* Y, const int incY)
    {
        cublasSswap(handle_, N, X, incX, Y, incY);
    }
    CUBLAS_FUNCTION void copy(const int N, const float* X, const int incX, float* Y, const int incY)
    {
        cublasScopy(handle_, N, X, incX, Y, incY);
    }
    CUBLAS_FUNCTION void axpy(const int N, const float alpha, const float* X, const int incX, float* Y, const int incY)
    {
        cublasSaxpy(handle_, N, &alpha, X, incX, Y, incY);
    }
    CUBLAS_FUNCTION void swap(const int N, double* X, const int incX, double* Y, const int incY)
    {
        cublasDswap(handle_, N, X, incX, Y, incY);
    }
    CUBLAS_FUNCTION void copy(const int N, const double* X, const int incX, double* Y, const int incY)
    {
        cublasDcopy(handle_, N, X, incX, Y, incY);
    }
    CUBLAS_FUNCTION void axpy(const int N, const double alpha, const double* X, const int incX, double* Y, const int incY)
    {
        cublasDaxpy(handle_, N, &alpha, X, incX, Y, incY);
    }
    CUBLAS_FUNCTION void rotg(float* a, float* b, float* c, float* s)
    {
        cublasSrotg(handle_, a, b, c, s);
    }
    CUBLAS_FUNCTION void rotmg(float* d1, float* d2, float* b1, const float b2, float* P)
    {
        cublasSrotmg(handle_, d1, d2, b1, &b2, P);
    }
    CUBLAS_FUNCTION void rot(const int N, float* X, const int incX, float* Y, const int incY, const float c, const float s)
    {
        cublasSrot(handle_, N, X, incX, Y, incY, &c, &s);
    }
    CUBLAS_FUNCTION void rotm(const int N, float* X, const int incX, float* Y, const int incY, const float* P)
    {
        cublasSrotm(handle_, N, X, incX, Y, incY, P);
    }
    CUBLAS_FUNCTION void rotg(double* a, double* b, double* c, double* s)
    {
        cublasDrotg(handle_, a, b, c, s);
    }
    CUBLAS_FUNCTION void rotmg(double* d1, double* d2, double* b1, const double b2, double* P)
    {
        cublasDrotmg(handle_, d1, d2, b1, &b2, P);
    }
    CUBLAS_FUNCTION void rot(const int N, double* X, const int incX, double* Y, const int incY, const double c, const double s)
    {
        cublasDrot(handle_, N, X, incX, Y, incY, &c, &s);
    }
    CUBLAS_FUNCTION void rotm(const int N, double* X, const int incX, double* Y, const int incY, const double* P)
    {
        cublasDrotm(handle_, N, X, incX, Y, incY, P);
    }
    CUBLAS_FUNCTION void scal(const int N, const float alpha, float* X, const int incX)
    {
        cublasSscal(handle_, N, &alpha, X, incX);
    }
    CUBLAS_FUNCTION void scal(const int N, const double alpha, double* X, const int incX)
    {
        cublasDscal(handle_, N, &alpha, X, incX);
    }
    CUBLAS_FUNCTION void gemv(const MatrixTransType TransA, const int M, const int N, const float alpha, const float* A, const int lda, const float* X, const int incX, const float beta, float* Y, const int incY)
    {
        cublasSgemv(handle_, get_trans(TransA), M, N, &alpha, A, lda, X, incX, &beta, Y, incY);
    }
    CUBLAS_FUNCTION void gbmv(const MatrixTransType TransA, const int M, const int N, const int KL, const int KU, const float alpha, const float* A, const int lda, const float* X, const int incX, const float beta, float* Y, const int incY)
    {
        cublasSgbmv(handle_, get_trans(TransA), M, N, KL, KU, &alpha, A, lda, X, incX, &beta, Y, incY);
    }
    CUBLAS_FUNCTION void trmv(const MatrixFillType Uplo, const MatrixTransType TransA, const MatrixDiagType Diag, const int N, const float* A, const int lda, float* X, const int incX)
    {
        cublasStrmv(handle_, get_uplo(Uplo), get_trans(TransA), get_diag(Diag), N, A, lda, X, incX);
    }
    CUBLAS_FUNCTION void tbmv(const MatrixFillType Uplo, const MatrixTransType TransA, const MatrixDiagType Diag, const int N, const int K, const float* A, const int lda, float* X, const int incX)
    {
        cublasStbmv(handle_, get_uplo(Uplo), get_trans(TransA), get_diag(Diag), N, K, A, lda, X, incX);
    }
    CUBLAS_FUNCTION void tpmv(const MatrixFillType Uplo, const MatrixTransType TransA, const MatrixDiagType Diag, const int N, const float* Ap, float* X, const int incX)
    {
        cublasStpmv(handle_, get_uplo(Uplo), get_trans(TransA), get_diag(Diag), N, Ap, X, incX);
    }
    CUBLAS_FUNCTION void trsv(const MatrixFillType Uplo, const MatrixTransType TransA, const MatrixDiagType Diag, const int N, const float* A, const int lda, float* X, const int incX)
    {
        cublasStrsv(handle_, get_uplo(Uplo), get_trans(TransA), get_diag(Diag), N, A, lda, X, incX);
    }
    CUBLAS_FUNCTION void tbsv(const MatrixFillType Uplo, const MatrixTransType TransA, const MatrixDiagType Diag, const int N, const int K, const float* A, const int lda, float* X, const int incX)
    {
        cublasStbsv(handle_, get_uplo(Uplo), get_trans(TransA), get_diag(Diag), N, K, A, lda, X, incX);
    }
    CUBLAS_FUNCTION void tpsv(const MatrixFillType Uplo, const MatrixTransType TransA, const MatrixDiagType Diag, const int N, const float* Ap, float* X, const int incX)
    {
        cublasStpsv(handle_, get_uplo(Uplo), get_trans(TransA), get_diag(Diag), N, Ap, X, incX);
    }
    CUBLAS_FUNCTION void gemv(const MatrixTransType TransA, const int M, const int N, const double alpha, const double* A, const int lda, const double* X, const int incX, const double beta, double* Y, const int incY)
    {
        cublasDgemv(handle_, get_trans(TransA), M, N, &alpha, A, lda, X, incX, &beta, Y, incY);
    }
    CUBLAS_FUNCTION void gbmv(const MatrixTransType TransA, const int M, const int N, const int KL, const int KU, const double alpha, const double* A, const int lda, const double* X, const int incX, const double beta, double* Y, const int incY)
    {
        cublasDgbmv(handle_, get_trans(TransA), M, N, KL, KU, &alpha, A, lda, X, incX, &beta, Y, incY);
    }
    CUBLAS_FUNCTION void trmv(const MatrixFillType Uplo, const MatrixTransType TransA, const MatrixDiagType Diag, const int N, const double* A, const int lda, double* X, const int incX)
    {
        cublasDtrmv(handle_, get_uplo(Uplo), get_trans(TransA), get_diag(Diag), N, A, lda, X, incX);
    }
    CUBLAS_FUNCTION void tbmv(const MatrixFillType Uplo, const MatrixTransType TransA, const MatrixDiagType Diag, const int N, const int K, const double* A, const int lda, double* X, const int incX)
    {
        cublasDtbmv(handle_, get_uplo(Uplo), get_trans(TransA), get_diag(Diag), N, K, A, lda, X, incX);
    }
    CUBLAS_FUNCTION void tpmv(const MatrixFillType Uplo, const MatrixTransType TransA, const MatrixDiagType Diag, const int N, const double* Ap, double* X, const int incX)
    {
        cublasDtpmv(handle_, get_uplo(Uplo), get_trans(TransA), get_diag(Diag), N, Ap, X, incX);
    }
    CUBLAS_FUNCTION void trsv(const MatrixFillType Uplo, const MatrixTransType TransA, const MatrixDiagType Diag, const int N, const double* A, const int lda, double* X, const int incX)
    {
        cublasDtrsv(handle_, get_uplo(Uplo), get_trans(TransA), get_diag(Diag), N, A, lda, X, incX);
    }
    CUBLAS_FUNCTION void tbsv(const MatrixFillType Uplo, const MatrixTransType TransA, const MatrixDiagType Diag, const int N, const int K, const double* A, const int lda, double* X, const int incX)
    {
        cublasDtbsv(handle_, get_uplo(Uplo), get_trans(TransA), get_diag(Diag), N, K, A, lda, X, incX);
    }
    CUBLAS_FUNCTION void tpsv(const MatrixFillType Uplo, const MatrixTransType TransA, const MatrixDiagType Diag, const int N, const double* Ap, double* X, const int incX)
    {
        cublasDtpsv(handle_, get_uplo(Uplo), get_trans(TransA), get_diag(Diag), N, Ap, X, incX);
    }
    CUBLAS_FUNCTION void symv(const MatrixFillType Uplo, const int N, const float alpha, const float* A, const int lda, const float* X, const int incX, const float beta, float* Y, const int incY)
    {
        cublasSsymv(handle_, get_uplo(Uplo), N, &alpha, A, lda, X, incX, &beta, Y, incY);
    }
    CUBLAS_FUNCTION void sbmv(const MatrixFillType Uplo, const int N, const int K, const float alpha, const float* A, const int lda, const float* X, const int incX, const float beta, float* Y, const int incY)
    {
        cublasSsbmv(handle_, get_uplo(Uplo), N, K, &alpha, A, lda, X, incX, &beta, Y, incY);
    }
    CUBLAS_FUNCTION void spmv(const MatrixFillType Uplo, const int N, const float alpha, const float* Ap, const float* X, const int incX, const float beta, float* Y, const int incY)
    {
        cublasSspmv(handle_, get_uplo(Uplo), N, &alpha, Ap, X, incX, &beta, Y, incY);
    }
    CUBLAS_FUNCTION void ger(const int M, const int N, const float alpha, const float* X, const int incX, const float* Y, const int incY, float* A, const int lda)
    {
        cublasSger(handle_, M, N, &alpha, X, incX, Y, incY, A, lda);
    }
    CUBLAS_FUNCTION void syr(const MatrixFillType Uplo, const int N, const float alpha, const float* X, const int incX, float* A, const int lda)
    {
        cublasSsyr(handle_, get_uplo(Uplo), N, &alpha, X, incX, A, lda);
    }
    CUBLAS_FUNCTION void spr(const MatrixFillType Uplo, const int N, const float alpha, const float* X, const int incX, float* Ap)
    {
        cublasSspr(handle_, get_uplo(Uplo), N, &alpha, X, incX, Ap);
    }
    CUBLAS_FUNCTION void syr2(const MatrixFillType Uplo, const int N, const float alpha, const float* X, const int incX, const float* Y, const int incY, float* A, const int lda)
    {
        cublasSsyr2(handle_, get_uplo(Uplo), N, &alpha, X, incX, Y, incY, A, lda);
    }
    CUBLAS_FUNCTION void spr2(const MatrixFillType Uplo, const int N, const float alpha, const float* X, const int incX, const float* Y, const int incY, float* A)
    {
        cublasSspr2(handle_, get_uplo(Uplo), N, &alpha, X, incX, Y, incY, A);
    }
    CUBLAS_FUNCTION void symv(const MatrixFillType Uplo, const int N, const double alpha, const double* A, const int lda, const double* X, const int incX, const double beta, double* Y, const int incY)
    {
        cublasDsymv(handle_, get_uplo(Uplo), N, &alpha, A, lda, X, incX, &beta, Y, incY);
    }
    CUBLAS_FUNCTION void sbmv(const MatrixFillType Uplo, const int N, const int K, const double alpha, const double* A, const int lda, const double* X, const int incX, const double beta, double* Y, const int incY)
    {
        cublasDsbmv(handle_, get_uplo(Uplo), N, K, &alpha, A, lda, X, incX, &beta, Y, incY);
    }
    CUBLAS_FUNCTION void spmv(const MatrixFillType Uplo, const int N, const double alpha, const double* Ap, const double* X, const int incX, const double beta, double* Y, const int incY)
    {
        cublasDspmv(handle_, get_uplo(Uplo), N, &alpha, Ap, X, incX, &beta, Y, incY);
    }
    CUBLAS_FUNCTION void ger(const int M, const int N, const double alpha, const double* X, const int incX, const double* Y, const int incY, double* A, const int lda)
    {
        cublasDger(handle_, M, N, &alpha, X, incX, Y, incY, A, lda);
    }
    CUBLAS_FUNCTION void syr(const MatrixFillType Uplo, const int N, const double alpha, const double* X, const int incX, double* A, const int lda)
    {
        cublasDsyr(handle_, get_uplo(Uplo), N, &alpha, X, incX, A, lda);
    }
    CUBLAS_FUNCTION void spr(const MatrixFillType Uplo, const int N, const double alpha, const double* X, const int incX, double* Ap)
    {
        cublasDspr(handle_, get_uplo(Uplo), N, &alpha, X, incX, Ap);
    }
    CUBLAS_FUNCTION void syr2(const MatrixFillType Uplo, const int N, const double alpha, const double* X, const int incX, const double* Y, const int incY, double* A, const int lda)
    {
        cublasDsyr2(handle_, get_uplo(Uplo), N, &alpha, X, incX, Y, incY, A, lda);
    }
    CUBLAS_FUNCTION void spr2(const MatrixFillType Uplo, const int N, const double alpha, const double* X, const int incX, const double* Y, const int incY, double* A)
    {
        cublasDspr2(handle_, get_uplo(Uplo), N, &alpha, X, incX, Y, incY, A);
    }
    CUBLAS_FUNCTION void gemm(const MatrixTransType TransA, const MatrixTransType TransB, const int M, const int N, const int K, const float alpha, const float* A, const int lda, const float* B, const int ldb, const float beta, float* C, const int ldc)
    {
        cublasSgemm(handle_, get_trans(TransA), get_trans(TransB), M, N, K, &alpha, A, lda, B, ldb, &beta, C, ldc);
    }
    CUBLAS_FUNCTION void gemmStridedBatched(const MatrixTransType TransA, const MatrixTransType TransB,
        const int M, const int N, const int K,
        const float alpha, const float* A, const int lda, long long strideA,
        const float* B, const int ldb, long long strideB,
        const float beta, float* C, const int ldc, long long strideC,
        const int batchCount)
    {
        cublasSgemmStridedBatched(handle_, get_trans(TransA), get_trans(TransB),
            M, N, K, &alpha, A, lda, strideA, B, ldb, strideB, &beta, C, ldc, strideC, batchCount);
    }
    CUBLAS_FUNCTION void symm(const MatrixSideType Side, const MatrixFillType Uplo, const int M, const int N, const float alpha, const float* A, const int lda, const float* B, const int ldb, const float beta, float* C, const int ldc)
    {
        cublasSsymm(handle_, get_side(Side), get_uplo(Uplo), M, N, &alpha, A, lda, B, ldb, &beta, C, ldc);
    }
    CUBLAS_FUNCTION void syrk(const MatrixFillType Uplo, const MatrixTransType Trans, const int N, const int K, const float alpha, const float* A, const int lda, const float beta, float* C, const int ldc)
    {
        cublasSsyrk(handle_, get_uplo(Uplo), get_trans(Trans), N, K, &alpha, A, lda, &beta, C, ldc);
    }
    CUBLAS_FUNCTION void syr2k(const MatrixFillType Uplo, const MatrixTransType Trans, const int N, const int K, const float alpha, const float* A, const int lda, const float* B, const int ldb, const float beta, float* C, const int ldc)
    {
        cublasSsyr2k(handle_, get_uplo(Uplo), get_trans(Trans), N, K, &alpha, A, lda, B, ldb, &beta, C, ldc);
    }
    CUBLAS_FUNCTION void trmm(const MatrixSideType Side, const MatrixFillType Uplo, const MatrixTransType TransA, const MatrixDiagType Diag, const int M, const int N, const float alpha, const float* A, const int lda, float* B, const int ldb)
    {
        cublasStrmm(handle_, get_side(Side), get_uplo(Uplo), get_trans(TransA), get_diag(Diag), M, N, &alpha, A, lda, B, ldb, B, ldb);
    }
    CUBLAS_FUNCTION void trsm(const MatrixSideType Side, const MatrixFillType Uplo, const MatrixTransType TransA, const MatrixDiagType Diag, const int M, const int N, const float alpha, const float* A, const int lda, float* B, const int ldb)
    {
        cublasStrsm(handle_, get_side(Side), get_uplo(Uplo), get_trans(TransA), get_diag(Diag), M, N, &alpha, A, lda, B, ldb);
    }
    CUBLAS_FUNCTION void gemm(const MatrixTransType TransA, const MatrixTransType TransB, const int M, const int N, const int K, const double alpha, const double* A, const int lda, const double* B, const int ldb, const double beta, double* C, const int ldc)
    {
        cublasDgemm(handle_, get_trans(TransA), get_trans(TransB), M, N, K, &alpha, A, lda, B, ldb, &beta, C, ldc);
    }
    CUBLAS_FUNCTION void symm(const MatrixSideType Side, const MatrixFillType Uplo, const int M, const int N, const double alpha, const double* A, const int lda, const double* B, const int ldb, const double beta, double* C, const int ldc)
    {
        cublasDsymm(handle_, get_side(Side), get_uplo(Uplo), M, N, &alpha, A, lda, B, ldb, &beta, C, ldc);
    }
    CUBLAS_FUNCTION void syrk(const MatrixFillType Uplo, const MatrixTransType Trans, const int N, const int K, const double alpha, const double* A, const int lda, const double beta, double* C, const int ldc)
    {
        cublasDsyrk(handle_, get_uplo(Uplo), get_trans(Trans), N, K, &alpha, A, lda, &beta, C, ldc);
    }
    CUBLAS_FUNCTION void syr2k(const MatrixFillType Uplo, const MatrixTransType Trans, const int N, const int K, const double alpha, const double* A, const int lda, const double* B, const int ldb, const double beta, double* C, const int ldc)
    {
        cublasDsyr2k(handle_, get_uplo(Uplo), get_trans(Trans), N, K, &alpha, A, lda, B, ldb, &beta, C, ldc);
    }
    CUBLAS_FUNCTION void trmm(const MatrixSideType Side, const MatrixFillType Uplo, const MatrixTransType TransA, const MatrixDiagType Diag, const int M, const int N, const double alpha, const double* A, const int lda, double* B, const int ldb)
    {
        cublasDtrmm(handle_, get_side(Side), get_uplo(Uplo), get_trans(TransA), get_diag(Diag), M, N, &alpha, A, lda, B, ldb, B, ldb);
    }
    CUBLAS_FUNCTION void trsm(const MatrixSideType Side, const MatrixFillType Uplo, const MatrixTransType TransA, const MatrixDiagType Diag, const int M, const int N, const double alpha, const double* A, const int lda, double* B, const int ldb)
    {
        cublasDtrsm(handle_, get_side(Side), get_uplo(Uplo), get_trans(TransA), get_diag(Diag), M, N, &alpha, A, lda, B, ldb);
    }

    //extensions of cublas
    CUBLAS_FUNCTION void geam(const MatrixTransType TransA, const MatrixTransType TransB, int m, int n, const float alpha, const float* A, int lda, const float beta, const float* B, int ldb, float* C, int ldc)
    {
        cublasSgeam(handle_, get_trans(TransA), get_trans(TransB), m, n, &alpha, A, lda, &beta, B, lda, C, ldc);
    }
    CUBLAS_FUNCTION void geam(const MatrixTransType TransA, const MatrixTransType TransB, int m, int n, const double alpha, const double* A, int lda, const double beta, const double* B, int ldb, double* C, int ldc)
    {
        cublasDgeam(handle_, get_trans(TransA), get_trans(TransB), m, n, &alpha, A, lda, &beta, B, lda, C, ldc);
    }
    CUBLAS_FUNCTION void dgem(MatrixSideType Side, int m, int n, const float* A, int lda, const float* x, int incx, float* C, int ldc)
    {
        cublasSdgmm(handle_, get_side(Side), m, n, A, lda, x, incx, C, ldc);
    }
    CUBLAS_FUNCTION void dgem(MatrixSideType Side, int m, int n, const double* A, int lda, const double* x, int incx, double* C, int ldc)
    {
        cublasDdgmm(handle_, get_side(Side), m, n, A, lda, x, incx, C, ldc);
    }
    //half precision
    CUBLAS_FUNCTION float dot(const int N, const half* X, const int incX, const half* Y, const int incY)
    {
        //float r;
        //cublasDotEx(handle_, N, X, CUDA_R_16F, incX, Y, CUDA_R_16F, incY, &r, CUDA_R_32F, CUDA_R_32F);
        float* bufferX = nullptr;
        float* bufferY = nullptr;
        cudaMalloc((void**)&bufferX, N * incX * sizeof(float));
        cudaMalloc((void**)&bufferY, N * incY * sizeof(float));
        cuda_half2float((half*)X, bufferX, N * incX);
        cuda_half2float((half*)Y, bufferY, N * incY);
        float r = dot(N, bufferX, incX, bufferY, incY);
        cudaFree(bufferX);
        cudaFree(bufferY);
        return r;
    }
    CUBLAS_FUNCTION float asum(const int N, const half* X, const int incX)
    {
        float r = 0;
        float* buffer = nullptr;
        cudaMalloc((void**)&buffer, N * incX * sizeof(float));
        cuda_half2float((half*)X, buffer, N * incX);
        cublasSasum(handle_, N, buffer, incX, &r);
        cudaFree(buffer);
        return r;
    }
    CUBLAS_FUNCTION int iamax(const int N, const half* X, const int incX)
    {
        int r = 1;
        float* buffer = nullptr;
        cudaMalloc((void**)&buffer, N * incX * sizeof(float));
        cuda_half2float((half*)X, buffer, N * incX);
        cublasIsamax(handle_, N, buffer, incX, &r);
        cudaFree(buffer);
        return r - 1;
    }
    CUBLAS_FUNCTION void scal(const int N, const half alpha, half* X, const int incX)
    {
        cublasScalEx(handle_, N, &alpha, CUDA_R_16F, X, CUDA_R_16F, incX, CUDA_R_32F);
    }
    CUBLAS_FUNCTION void gemv(const MatrixTransType TransA, const int M, const int N, const half alpha, const half* A, const int lda, const half* X, const int incX, const half beta, half* Y, const int incY)
    {
        cublasHgemm(handle_, get_trans(TransA), CUBLAS_OP_N, M, 1, N, (__half*)&alpha, (__half*)A, lda, (__half*)X, N, (__half*)&beta, (__half*)Y, lda);
    }
    CUBLAS_FUNCTION void gemm(const MatrixTransType TransA, const MatrixTransType TransB, const int M, const int N, const int K, const half alpha, const half* A, const int lda, const half* B, const int ldb, const half beta, half* C, const int ldc)
    {
        cublasHgemm(handle_, get_trans(TransA), get_trans(TransB), M, N, K, (__half*)&alpha, (__half*)A, lda, (__half*)B, ldb, (__half*)&beta, (__half*)C, ldc);
    }
    CUBLAS_FUNCTION void gemmStridedBatched(const MatrixTransType TransA, const MatrixTransType TransB,
        const int M, const int N, const int K,
        const half alpha, const half* A, const int lda, long long strideA,
        const half* B, const int ldb, long long strideB,
        const half beta, half* C, const int ldc, long long strideC,
        const int batchCount)
    {
        float fa = (float)alpha, fb = (float)beta;
        if (cublasGemmStridedBatchedEx_fp_)
        {
            // Use FP32 accumulation for better precision with FP16 I/O
            cublasGemmStridedBatchedEx_fp_(handle_, get_trans(TransA), get_trans(TransB),
                M, N, K,
                &fa, A, CUDA_R_16F, lda, strideA,
                B, CUDA_R_16F, ldb, strideB,
                &fb, C, CUDA_R_16F, ldc, strideC,
                batchCount, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        }
        else
        {
            cublasHgemmStridedBatched(handle_, get_trans(TransA), get_trans(TransB),
                M, N, K, (__half*)&alpha, (__half*)A, lda, strideA,
                (__half*)B, ldb, strideB, (__half*)&beta, (__half*)C, ldc, strideC, batchCount);
        }
    }
    //bfloat16 precision
    CUBLAS_FUNCTION float dot(const int N, const bfloat16* X, const int incX, const bfloat16* Y, const int incY)
    {
        float* bufferX = nullptr;
        float* bufferY = nullptr;
        cudaMalloc((void**)&bufferX, N * incX * sizeof(float));
        cudaMalloc((void**)&bufferY, N * incY * sizeof(float));
        cuda_bf162float((void*)X, bufferX, (unsigned int)(N * incX));
        cuda_bf162float((void*)Y, bufferY, (unsigned int)(N * incY));
        float r = dot(N, bufferX, incX, bufferY, incY);
        cudaFree(bufferX);
        cudaFree(bufferY);
        return r;
    }
    CUBLAS_FUNCTION float asum(const int N, const bfloat16* X, const int incX)
    {
        float r = 0;
        float* buffer = nullptr;
        cudaMalloc((void**)&buffer, N * incX * sizeof(float));
        cuda_bf162float((void*)X, buffer, (unsigned int)(N * incX));
        cublasSasum(handle_, N, buffer, incX, &r);
        cudaFree(buffer);
        return r;
    }
    CUBLAS_FUNCTION int iamax(const int N, const bfloat16* X, const int incX)
    {
        int r = 1;
        float* buffer = nullptr;
        cudaMalloc((void**)&buffer, N * incX * sizeof(float));
        cuda_bf162float((void*)X, buffer, (unsigned int)(N * incX));
        cublasIsamax(handle_, N, buffer, incX, &r);
        cudaFree(buffer);
        return r - 1;
    }
    CUBLAS_FUNCTION void scal(const int N, const bfloat16 alpha, bfloat16* X, const int incX)
    {
        float fa = (float)alpha;
        cublasScalEx(handle_, N, &fa, CUDA_R_32F, X, CUDA_R_16BF, incX, CUDA_R_32F);
    }
    CUBLAS_FUNCTION void gemv(const MatrixTransType TransA, const int M, const int N, const bfloat16 alpha, const bfloat16* A, const int lda, const bfloat16* X, const int incX, const bfloat16 beta, bfloat16* Y, const int incY)
    {
        // BF16 gemv via float conversion: A(M×N) × X(N) → Y(M)
        float fa = (float)alpha, fb = (float)beta;
        int szA = lda * N;    // column-major: lda rows × N cols
        int szX = N * incX;
        int szY = M * incY;
        float *fA = nullptr, *fX = nullptr, *fY = nullptr;
        cudaMalloc((void**)&fA, szA * sizeof(float));
        cudaMalloc((void**)&fX, szX * sizeof(float));
        cudaMalloc((void**)&fY, szY * sizeof(float));
        cuda_bf162float((void*)A, fA, (unsigned int)szA);
        cuda_bf162float((void*)X, fX, (unsigned int)szX);
        if (fb != 0.f)
            cuda_bf162float((void*)Y, fY, (unsigned int)szY);
        else
            cudaMemset(fY, 0, szY * sizeof(float));
        cublasSgemv(handle_, get_trans(TransA), M, N, &fa, fA, lda, fX, incX, &fb, fY, incY);
        cuda_float2bf16(fY, (void*)Y, (unsigned int)szY);
        cudaFree(fA); cudaFree(fX); cudaFree(fY);
    }
    CUBLAS_FUNCTION void gemm(const MatrixTransType TransA, const MatrixTransType TransB, const int M, const int N, const int K, const bfloat16 alpha, const bfloat16* A, const int lda, const bfloat16* B, const int ldb, const bfloat16 beta, bfloat16* C, const int ldc)
    {
        // BF16 gemm via float conversion
        float fa = (float)alpha, fb = (float)beta;
        int szA = lda * (TransA == MATRIX_NO_TRANS ? K : M);
        int szB = ldb * (TransB == MATRIX_NO_TRANS ? N : K);
        int szC = ldc * N;
        float *fA = nullptr, *fB = nullptr, *fC = nullptr;
        cudaMalloc((void**)&fA, szA * sizeof(float));
        cudaMalloc((void**)&fB, szB * sizeof(float));
        cudaMalloc((void**)&fC, szC * sizeof(float));
        cuda_bf162float((void*)A, fA, (unsigned int)szA);
        cuda_bf162float((void*)B, fB, (unsigned int)szB);
        if (fb != 0.f)
            cuda_bf162float((void*)C, fC, (unsigned int)szC);
        else
            cudaMemset(fC, 0, szC * sizeof(float));
        cublasSgemm(handle_, get_trans(TransA), get_trans(TransB), M, N, K,
            &fa, fA, lda, fB, ldb, &fb, fC, ldc);
        cuda_float2bf16(fC, (void*)C, (unsigned int)szC);
        cudaFree(fA); cudaFree(fB); cudaFree(fC);
    }
    CUBLAS_FUNCTION void gemmStridedBatched(const MatrixTransType TransA, const MatrixTransType TransB,
        const int M, const int N, const int K,
        const bfloat16 alpha, const bfloat16* A, const int lda, long long strideA,
        const bfloat16* B, const int ldb, long long strideB,
        const bfloat16 beta, bfloat16* C, const int ldc, long long strideC,
        const int batchCount)
    {
        // BF16 gemmStridedBatched via float conversion
        float fa = (float)alpha, fb = (float)beta;
        long long szA = strideA > 0 ? strideA : (long long)lda * (TransA == MATRIX_NO_TRANS ? K : M);
        long long szB = strideB > 0 ? strideB : (long long)ldb * (TransB == MATRIX_NO_TRANS ? N : K);
        long long szC = strideC > 0 ? strideC : (long long)ldc * N;
        long long totalA = szA * batchCount;
        long long totalB = szB * batchCount;
        long long totalC = szC * batchCount;
        float *fA = nullptr, *fB = nullptr, *fC = nullptr;
        cudaMalloc((void**)&fA, totalA * sizeof(float));
        cudaMalloc((void**)&fB, totalB * sizeof(float));
        cudaMalloc((void**)&fC, totalC * sizeof(float));
        cuda_bf162float((void*)A, fA, (unsigned int)totalA);
        cuda_bf162float((void*)B, fB, (unsigned int)totalB);
        if (fb != 0.f)
            cuda_bf162float((void*)C, fC, (unsigned int)totalC);
        else
            cudaMemset(fC, 0, totalC * sizeof(float));
        cublasSgemmStridedBatched(handle_, get_trans(TransA), get_trans(TransB),
            M, N, K, &fa, fA, lda, szA, fB, ldb, szB, &fb, fC, ldc, szC, batchCount);
        cuda_float2bf16(fC, (void*)C, (unsigned int)totalC);
        cudaFree(fA); cudaFree(fB); cudaFree(fC);
    }

protected:
    cublasHandle_t handle_ = nullptr;

    using cublasGemmEx_t = cublasStatus_t(CUBLASWINAPI*)(
        cublasHandle_t, cublasOperation_t, cublasOperation_t,
        int, int, int,
        const void*, const void*, cudaDataType, int,
        const void*, cudaDataType, int,
        const void*, void*, cudaDataType, int,
        cublasComputeType_t, cublasGemmAlgo_t);
    cublasGemmEx_t cublasGemmEx_fp_ = nullptr;

    // cublasGemmStridedBatchedEx with cublasComputeType_t is not in the project's cublas_v2.h
    // and not exported by older cublas.lib; must be loaded at runtime via GetProcAddress.
    using cublasGemmStridedBatchedEx_t = cublasStatus_t(CUBLASWINAPI*)(
        cublasHandle_t, cublasOperation_t, cublasOperation_t,
        int, int, int,
        const void*, const void*, cudaDataType, int, long long,
        const void*, cudaDataType, int, long long,
        const void*, void*, cudaDataType, int, long long,
        int, cublasComputeType_t, cublasGemmAlgo_t);
    cublasGemmStridedBatchedEx_t cublasGemmStridedBatchedEx_fp_ = nullptr;
};

}    // namespace cccc
#else
#include "cblas_real.h"
namespace cccc
{
class Cublas : public Cblas
{
};
}    //namespace cccc
#endif
