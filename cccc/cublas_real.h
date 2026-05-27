#pragma once
#include "blas_types.h"
#include "cuda_functions.h"

#if ENABLE_CUDA
#include "cublasLt.h"
#include "cublas_v2.h"

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
        // Initialize cublasLt for FP8 W8A8 GEMM (requires cublasLt64_13.dll and SM89+)
        if (cublasLtCreate)
        {
            if (cublasLtCreate(&lt_handle_) == CUBLAS_STATUS_SUCCESS)
            {
                cudaMalloc((void**)(&d_scale_a_), sizeof(float));
                cudaMalloc((void**)(&d_scale_b_), sizeof(float));
                cudaMalloc((void**)(&d_act_absmax_), sizeof(float));
                cudaMalloc(&lt_workspace_, lt_workspace_size_);
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
        if (lt_handle_ && cublasLtDestroy)
        {
            cublasLtDestroy(lt_handle_);
        }
        lt_handle_ = nullptr;
        // Release persistent GEMM workspace buffers
        if (ws_A_)
        {
            cudaFree(ws_A_);
            ws_A_ = nullptr;
            ws_A_cap_ = 0;
        }
        if (ws_B_)
        {
            cudaFree(ws_B_);
            ws_B_ = nullptr;
            ws_B_cap_ = 0;
        }
        if (ws_C_)
        {
            cudaFree(ws_C_);
            ws_C_ = nullptr;
            ws_C_cap_ = 0;
        }
        if (ws_tmp_)
        {
            cudaFree(ws_tmp_);
            ws_tmp_ = nullptr;
            ws_tmp_cap_ = 0;
        }
        if (ws_fp8a_)
        {
            cudaFree(ws_fp8a_);
            ws_fp8a_ = nullptr;
            ws_fp8a_cap_ = 0;
        }
        if (ws_fp8b_)
        {
            cudaFree(ws_fp8b_);
            ws_fp8b_ = nullptr;
            ws_fp8b_cap_ = 0;
        }
        if (d_scale_a_)
        {
            cudaFree(d_scale_a_);
            d_scale_a_ = nullptr;
        }
        if (d_scale_b_)
        {
            cudaFree(d_scale_b_);
            d_scale_b_ = nullptr;
        }
        if (d_act_absmax_)
        {
            cudaFree(d_act_absmax_);
            d_act_absmax_ = nullptr;
        }
        if (lt_workspace_)
        {
            cudaFree(lt_workspace_);
            lt_workspace_ = nullptr;
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
        cuda_convert(X, HALF, bufferX, FLOAT, (unsigned int)(N * incX), 1.0f);
        cuda_convert(Y, HALF, bufferY, FLOAT, (unsigned int)(N * incY), 1.0f);
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
        cuda_convert(X, HALF, buffer, FLOAT, (unsigned int)(N * incX), 1.0f);
        cublasSasum(handle_, N, buffer, incX, &r);
        cudaFree(buffer);
        return r;
    }
    CUBLAS_FUNCTION int iamax(const int N, const half* X, const int incX)
    {
        int r = 1;
        float* buffer = nullptr;
        cudaMalloc((void**)&buffer, N * incX * sizeof(float));
        cuda_convert(X, HALF, buffer, FLOAT, (unsigned int)(N * incX), 1.0f);
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
        float fa = (float)alpha, fb = (float)beta;
        cublasGemmEx(handle_, get_trans(TransA), get_trans(TransB), M, N, K,
            &fa, A, CUDA_R_16F, lda, B, CUDA_R_16F, ldb,
            &fb, C, CUDA_R_16F, ldc,
            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    }
    CUBLAS_FUNCTION void gemmStridedBatched(const MatrixTransType TransA, const MatrixTransType TransB,
        const int M, const int N, const int K,
        const half alpha, const half* A, const int lda, long long strideA,
        const half* B, const int ldb, long long strideB,
        const half beta, half* C, const int ldc, long long strideC,
        const int batchCount)
    {
        float fa = (float)alpha, fb = (float)beta;
        // Use FP32 accumulation for better precision with FP16 I/O
        cublasGemmStridedBatchedEx(handle_, get_trans(TransA), get_trans(TransB),
            M, N, K,
            &fa, A, CUDA_R_16F, lda, strideA,
            B, CUDA_R_16F, ldb, strideB,
            &fb, C, CUDA_R_16F, ldc, strideC,
            batchCount, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    }
    //bfloat16 precision
    CUBLAS_FUNCTION float dot(const int N, const bfloat16* X, const int incX, const bfloat16* Y, const int incY)
    {
        float* bufferX = nullptr;
        float* bufferY = nullptr;
        cudaMalloc((void**)&bufferX, N * incX * sizeof(float));
        cudaMalloc((void**)&bufferY, N * incY * sizeof(float));
        cuda_convert(X, BFLOAT16, bufferX, FLOAT, (unsigned int)(N * incX), 1.0f);
        cuda_convert(Y, BFLOAT16, bufferY, FLOAT, (unsigned int)(N * incY), 1.0f);
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
        cuda_convert(X, BFLOAT16, buffer, FLOAT, (unsigned int)(N * incX), 1.0f);
        cublasSasum(handle_, N, buffer, incX, &r);
        cudaFree(buffer);
        return r;
    }
    CUBLAS_FUNCTION int iamax(const int N, const bfloat16* X, const int incX)
    {
        int r = 1;
        float* buffer = nullptr;
        cudaMalloc((void**)&buffer, N * incX * sizeof(float));
        cuda_convert(X, BFLOAT16, buffer, FLOAT, (unsigned int)(N * incX), 1.0f);
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
        // BF16 gemv：先将 A 和 X 转为 float 再调用 cublasSgemv，结果转回 BF16
        float fa = (float)alpha, fb = (float)beta;
        int szA = lda * N;    // column-major: lda rows × N cols
        int szX = N * incX;
        int szY = M * incY;
        ensureWS(&ws_A_, &ws_A_cap_, szA * sizeof(float));
        ensureWS(&ws_B_, &ws_B_cap_, szX * sizeof(float));
        ensureWS(&ws_C_, &ws_C_cap_, szY * sizeof(float));
        float *fA = ws_A_, *fX = ws_B_, *fY = ws_C_;
        cuda_convert(A, BFLOAT16, fA, FLOAT, (unsigned int)szA, 1.0f);
        cuda_convert(X, BFLOAT16, fX, FLOAT, (unsigned int)szX, 1.0f);
        if (fb != 0.f)
        {
            cuda_convert(Y, BFLOAT16, fY, FLOAT, (unsigned int)szY, 1.0f);
        }
        else
        {
            cudaMemset(fY, 0, szY * sizeof(float));
        }
        cublasSgemv(handle_, get_trans(TransA), M, N, &fa, fA, lda, fX, incX, &fb, fY, incY);
        cuda_convert(fY, FLOAT, Y, BFLOAT16, (unsigned int)szY, 1.0f);
    }
    CUBLAS_FUNCTION void gemm(const MatrixTransType TransA, const MatrixTransType TransB, const int M, const int N, const int K, const bfloat16 alpha, const bfloat16* A, const int lda, const bfloat16* B, const int ldb, const bfloat16 beta, bfloat16* C, const int ldc)
    {
        float fa = (float)alpha, fb = (float)beta;
        cublasGemmEx(handle_, get_trans(TransA), get_trans(TransB), M, N, K,
            &fa, A, CUDA_R_16BF, lda, B, CUDA_R_16BF, ldb,
            &fb, C, CUDA_R_16BF, ldc,
            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        cuda_check_error("cublasGemmEx_bf16");
    }
    // W8A16/W8A8：FP8-E4M3/E5M2 或 FP4 权重(A) × BF16 激活(B) → BF16 输出
    // act_scale > 0：静态激活量化（HF calibrated input_scale）；act_scale == 0：动态 absmax
    CUBLAS_FUNCTION void gemm(const MatrixTransType TransA, const MatrixTransType TransB, const int M, const int N, const int K, const float alpha, const uint8_t* A, const int lda, const bfloat16* B, const int ldb, const float beta, bfloat16* C, const int ldc, const float inv_scale = 1.0f, DataType weight_type = DataType::FP8_E4M3, const float act_scale = 0.0f)
    {
        float fa = alpha, fb = beta;
        if (lt_handle_)
        {
            // ── E5M2：尝试 cublasLt E5M2 kernel ──────────────────────────────────
            if (weight_type == DataType::FP8_E5M2)
            {
                cudaMemcpy(d_scale_a_, &inv_scale, sizeof(float), cudaMemcpyHostToDevice);
                if (lt_quant_gemm(get_trans(TransA), get_trans(TransB), M, N, K, fa, A, lda, DataType::FP8_E5M2, B, ldb, DataType::BFLOAT16, fb, C, ldc))
                {
                    if (cuda_clear_last_error)
                    {
                        cuda_clear_last_error();
                    }
                    return;
                }
                // lt failed; fall through to decode path
            }
            // ── E4M3 W8A8：将 BF16 激活量化为 FP8，再调用 cublasLt FP8×FP8 ────
            else if (weight_type == DataType::FP8_E4M3 && cuda_bf16_to_fp8e4m3_dynamic && act_data_type_ == DataType::FP8_E4M3)
            {
                int szB = ldb * (TransB == MATRIX_NO_TRANS ? N : K);
                // FP8 每元素 1 字节，分配 szB 字节（不是 *2）
                ensureWSv(&ws_fp8b_, &ws_fp8b_cap_, (size_t)szB);
                uint8_t* fp8B = (uint8_t*)(ws_fp8b_);
                if (fp8B)    // allocation succeeded
                {
                    if (act_scale > 0.0f)
                    {
                        // 静态激活量化：fp8 = bf16 / act_scale（HF calibrated input_scale）
                        cuda_convert(B, DataType::BFLOAT16, fp8B, DataType::FP8_E4M3, (unsigned int)szB, 1.0f / act_scale);
                        cudaMemcpy(d_scale_b_, &act_scale, sizeof(float), cudaMemcpyHostToDevice);
                    }
                    else
                    {
                        // 动态激活量化：运行时计算 absmax
                        cuda_bf16_to_fp8e4m3_dynamic(B, fp8B, (unsigned int)szB, d_act_absmax_, d_scale_b_);
                    }
                    cudaMemcpy(d_scale_a_, &inv_scale, sizeof(float), cudaMemcpyHostToDevice);
                    if (lt_quant_gemm(get_trans(TransA), get_trans(TransB), M, N, K, fa, A, lda, DataType::FP8_E4M3, fp8B, ldb, DataType::FP8_E4M3, fb, C, ldc))
                    {
                        if (cuda_clear_last_error)
                        {
                            cuda_clear_last_error();
                        }
                        return;
                    }
                    // lt_quant_gemm 失败；尝试 FP8×FP8 SGEMM 回退（保留激活量化精度）
                    if (cuda_clear_last_error)
                    {
                        cuda_clear_last_error();
                    }
                    float act_scale_host = act_scale;
                    if (act_scale_host == 0.0f)
                    {
                        cudaMemcpy(&act_scale_host, d_scale_b_, sizeof(float), cudaMemcpyDeviceToHost);
                    }
                    if (act_scale_host > 0.0f)
                    {
                        this->gemm(TransA, TransB, M, N, K, fa,
                            A, lda, fp8B, ldb,
                            fb, C, ldc, inv_scale, act_scale_host);
                        return;
                    }
                }
                // alloc failed or lt failed; fall through to decode path
            }
            // 清除 lt_quant_gemm 可能留下的 CUDA sticky error，避免 fallback 失败
            if (cuda_clear_last_error)
            {
                cuda_clear_last_error();
            }
        }
        // ── 通用回退：反量化权重 → BF16，再做 BF16×BF16 GEMM ─────────────
        {
            int szA = lda * (TransA == MATRIX_NO_TRANS ? K : M);
            ensureWS(&ws_A_, &ws_A_cap_, (size_t)szA * sizeof(bfloat16));
            bfloat16* bf16A = (bfloat16*)(ws_A_);
            cuda_convert(A, weight_type, bf16A, DataType::BFLOAT16, (unsigned int)szA, inv_scale);
            cublasGemmEx(handle_, get_trans(TransA), get_trans(TransB), M, N, K,
                &fa, bf16A, CUDA_R_16BF, lda, B, CUDA_R_16BF, ldb,
                &fb, C, CUDA_R_16BF, ldc,
                CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
            cuda_check_error("cublasGemmEx_w8a16_wA");
        }
    }
    // W8A16/W8A8：BF16 激活(A) × FP8-E4M3/E5M2 或 FP4 权重(B) → BF16 输出
    // act_scale > 0：静态激活量化（HF calibrated input_scale）；act_scale == 0：动态 absmax
    CUBLAS_FUNCTION void gemm(const MatrixTransType TransA, const MatrixTransType TransB, const int M, const int N, const int K, const float alpha, const bfloat16* A, const int lda, const uint8_t* B, const int ldb, const float beta, bfloat16* C, const int ldc, const float inv_scale = 1.0f, DataType weight_type = DataType::FP8_E4M3, const float act_scale = 0.0f)
    {
        float fa = alpha, fb = beta;
        // ── E4M3 W8A8：将 BF16 激活量化为 FP8，再调用 cublasLt FP8×FP8 ─────
        if (lt_handle_ && weight_type == DataType::FP8_E4M3 && cuda_bf16_to_fp8e4m3_dynamic && act_data_type_ == DataType::FP8_E4M3)
        {
            int szA = lda * (TransA == MATRIX_NO_TRANS ? K : M);
            // FP8 每元素 1 字节，分配 szA 字节（不是 *2）
            ensureWSv(&ws_fp8a_, &ws_fp8a_cap_, (size_t)szA);
            uint8_t* fp8A = (uint8_t*)(ws_fp8a_);
            if (fp8A)
            {
                if (act_scale > 0.0f)
                {
                    // 静态激活量化：fp8 = bf16 / act_scale（HF calibrated input_scale）
                    cuda_convert(A, DataType::BFLOAT16, fp8A, DataType::FP8_E4M3, (unsigned int)szA, 1.0f / act_scale);
                    cudaMemcpy(d_scale_a_, &act_scale, sizeof(float), cudaMemcpyHostToDevice);
                }
                else
                {
                    // 动态激活量化：运行时计算 absmax
                    cuda_bf16_to_fp8e4m3_dynamic(A, fp8A, (unsigned int)szA, d_act_absmax_, d_scale_a_);
                }
                cudaMemcpy(d_scale_b_, &inv_scale, sizeof(float), cudaMemcpyHostToDevice);
                if (lt_quant_gemm(get_trans(TransA), get_trans(TransB), M, N, K, fa, fp8A, lda, DataType::FP8_E4M3, B, ldb, DataType::FP8_E4M3, fb, C, ldc))
                {
                    if (cuda_clear_last_error)
                    {
                        cuda_clear_last_error();
                    }
                    return;
                }
                // lt_quant_gemm 失败；尝试 FP8×FP8 SGEMM 回退（保留激活量化精度）
                if (cuda_clear_last_error)
                {
                    cuda_clear_last_error();
                }
                float act_scale_host = act_scale;
                if (act_scale_host == 0.0f)
                {
                    cudaMemcpy(&act_scale_host, d_scale_a_, sizeof(float), cudaMemcpyDeviceToHost);
                }
                if (act_scale_host > 0.0f)
                {
                    this->gemm(TransA, TransB, M, N, K, fa,
                        fp8A, lda, B, ldb,
                        fb, C, ldc, act_scale_host, inv_scale);
                    return;
                }
                // fall through to BF16 decode path
            }
        }
        // ── 通用回退：反量化权重 → BF16，再做 BF16×BF16 GEMM ─────────────
        {
            int szB = ldb * (TransB == MATRIX_NO_TRANS ? N : K);
            ensureWS(&ws_B_, &ws_B_cap_, (size_t)szB * sizeof(bfloat16));
            bfloat16* bf16B = (bfloat16*)(ws_B_);
            cuda_convert(B, weight_type, bf16B, DataType::BFLOAT16, (unsigned int)szB, inv_scale);
            cublasGemmEx(handle_, get_trans(TransA), get_trans(TransB), M, N, K,
                &fa, A, CUDA_R_16BF, lda, bf16B, CUDA_R_16BF, ldb,
                &fb, C, CUDA_R_16BF, ldc,
                CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
            cuda_check_error("cublasGemmEx_w8a16_wB");
        }
    }
    // W8A8：A 和 B 均已是 FP8 → 输出 BF16（无需量化步骤）
    CUBLAS_FUNCTION void gemm(const MatrixTransType TransA, const MatrixTransType TransB, const int M, const int N, const int K, const float alpha, const uint8_t* A, const int lda, const uint8_t* B, const int ldb, const float beta, bfloat16* C, const int ldc, const float inv_scale_A = 1.0f, const float inv_scale_B = 1.0f)
    {
        float fa = alpha, fb = beta;
        // Probe: bypass cublasLt for W8A8->BF16 on Blackwell.
        // The Lt path can report success while producing all-zero logits; use the stable
        // FP8->float->SGEMM->BF16 fallback to verify whether Lt is the remaining root cause.
        // 回退路径：FP8 → float → cublasSgemm → BF16
        int szA = lda * (TransA == MATRIX_NO_TRANS ? K : M);
        int szB = ldb * (TransB == MATRIX_NO_TRANS ? N : K);
        int szC = ldc * N;
        ensureWS(&ws_A_, &ws_A_cap_, (size_t)szA * sizeof(float));
        ensureWS(&ws_B_, &ws_B_cap_, (size_t)szB * sizeof(float));
        ensureWS(&ws_C_, &ws_C_cap_, (size_t)szC * sizeof(float));
        if (!ws_A_ || !ws_B_ || !ws_C_)
        {
            return;    // cudaMalloc failed
        }
        float *fA = (float*)ws_A_, *fB = (float*)ws_B_, *fC = (float*)ws_C_;
        cuda_convert(A, DataType::FP8_E4M3, fA, DataType::FLOAT, (unsigned int)szA, inv_scale_A);
        cuda_convert(B, DataType::FP8_E4M3, fB, DataType::FLOAT, (unsigned int)szB, inv_scale_B);
        if (fb != 0.f)
        {
            cuda_convert(C, DataType::BFLOAT16, fC, DataType::FLOAT, (unsigned int)szC, 1.0f);
        }
        else
        {
            cudaMemset(fC, 0, (size_t)szC * sizeof(float));
        }
        cublasSgemm(handle_, get_trans(TransA), get_trans(TransB), M, N, K,
            &fa, fA, lda, fB, ldb, &fb, fC, ldc);
        cuda_check_error("cublasSgemm_w8a8_bf16");
        cuda_convert(fC, DataType::FLOAT, C, DataType::BFLOAT16, (unsigned int)szC, 1.0f);
    }
    // W8A8：A 和 B 均已是 FP8 → 输出 FP8（用独立 ws_tmp_ 中转，避免与 ws_A_/B_/C_ 冲突）
    CUBLAS_FUNCTION void gemm(const MatrixTransType TransA, const MatrixTransType TransB, const int M, const int N, const int K, const float alpha, const uint8_t* A, const int lda, const uint8_t* B, const int ldb, const float /*beta*/, uint8_t* C_fp8, const int ldc, const float inv_scale_A = 1.0f, const float inv_scale_B = 1.0f)
    {
        // Step 1: ensure ws_tmp_ can hold szC bfloat16 elements
        int szC = ldc * N;
        ensureWSv(&ws_tmp_, &ws_tmp_cap_, (size_t)szC * sizeof(bfloat16));
        bfloat16* bf16C = (bfloat16*)(ws_tmp_);
        if (!bf16C)
        {
            return;    // allocation failed; skip this GEMM
        }
        float fa = alpha;
        bool ok = false;
        if (lt_handle_)
        {
            cudaMemcpy(d_scale_a_, &inv_scale_A, sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(d_scale_b_, &inv_scale_B, sizeof(float), cudaMemcpyHostToDevice);
            ok = lt_quant_gemm(get_trans(TransA), get_trans(TransB), M, N, K, fa, A, lda, DataType::FP8_E4M3, B, ldb, DataType::FP8_E4M3, 0.f, bf16C, ldc);
            // lt_quant_gemm failed; clear any pending CUDA error before the fallback kernels
            if (cuda_clear_last_error)
            {
                cuda_clear_last_error();
            }
        }
        if (!ok)
        {
            // 回退路径：FP8→float → cublasSgemm → BF16（简化处理，忽略 beta）
            int szA = lda * (TransA == MATRIX_NO_TRANS ? K : M);
            int szB = ldb * (TransB == MATRIX_NO_TRANS ? N : K);
            ensureWS(&ws_A_, &ws_A_cap_, (size_t)szA * sizeof(float));
            ensureWS(&ws_B_, &ws_B_cap_, (size_t)szB * sizeof(float));
            ensureWS(&ws_C_, &ws_C_cap_, (size_t)szC * sizeof(float));
            if (!ws_A_ || !ws_B_ || !ws_C_)
            {
                return;    // cudaMalloc failed
            }
            float *fA = (float*)ws_A_, *fB = (float*)ws_B_, *fC = (float*)ws_C_;
            cuda_convert(A, DataType::FP8_E4M3, fA, DataType::FLOAT, (unsigned int)szA, inv_scale_A);
            cuda_convert(B, DataType::FP8_E4M3, fB, DataType::FLOAT, (unsigned int)szB, inv_scale_B);
            cudaMemset(fC, 0, (size_t)szC * sizeof(float));
            float zero = 0.f;
            cublasSgemm(handle_, get_trans(TransA), get_trans(TransB), M, N, K, &fa, fA, lda, fB, ldb, &zero, fC, ldc);
            cuda_check_error("cublasSgemm_w8a8_fp8");
            cuda_convert(fC, FLOAT, bf16C, BFLOAT16, (unsigned int)szC, 1.0f);
        }
        // Step 2: BF16→FP8 at scale=1.0，所有 FP8 激活保持 quant_scale_=1.0，
        // 使 ADD/SiLU/elementMul 等 elementwise 算子可直接操作原始 FP8 浮点值，无需 dequant。
        // 典型 Transformer 激活值 << 448（FP8 E4M3 最大值），不会溢出。
        cuda_convert(bf16C, DataType::BFLOAT16, C_fp8, DataType::FP8_E4M3, (unsigned int)szC, 1.0f);
        last_fp8_out_scale_ = 1.0f;
    }
    CUBLAS_FUNCTION void gemmStridedBatched(const MatrixTransType TransA, const MatrixTransType TransB,
        const int M, const int N, const int K,
        const bfloat16 alpha, const bfloat16* A, const int lda, long long strideA,
        const bfloat16* B, const int ldb, long long strideB,
        const bfloat16 beta, bfloat16* C, const int ldc, long long strideC,
        const int batchCount)
    {
        float fa = (float)alpha, fb = (float)beta;
        cublasGemmStridedBatchedEx(handle_, get_trans(TransA), get_trans(TransB),
            M, N, K,
            &fa, A, CUDA_R_16BF, lda, strideA,
            B, CUDA_R_16BF, ldb, strideB,
            &fb, C, CUDA_R_16BF, ldc, strideC,
            batchCount, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    }

    // ── 统一 GEMM 辅助函数 ───────────────────────────────────────────────────
    static bool isQuantized(DataType dt)
    {
        return dt == DataType::FP8_E4M3 || dt == DataType::FP8_E5M2 || dt == DataType::FP4_E2M1;
    }
    // 返回每元素字节数。FP4（nibble 打包）返回 1；需要字节步长时，调用方须自行对 FP4 除以 2。
    static size_t elemBytes(DataType dt)
    {
        switch (dt)
        {
        case DataType::DOUBLE: return 8;
        case DataType::FLOAT: return 4;
        case DataType::HALF:
        case DataType::BFLOAT16: return 2;
        default: return 1;    // FP8_E4M3, FP8_E5M2, FP4_E2M1
        }
    }
    // 统一分派：处理所有量化（FP8/FP4）权重路径及标准类型。
    // 非量化操作数的 invScaleA/invScaleB 应传 1.0f。
    // act_scale > 0 时：使用静态激活量化（HF calibrated input_scale）；
    // act_scale == 0 时：动态 absmax（默认）。
    CUBLAS_FUNCTION void gemm(MatrixTransType TransA, MatrixTransType TransB,
        int M, int N, int K, float alpha,
        const void* A, DataType typeA, int lda, float invScaleA,
        const void* B, DataType typeB, int ldb, float invScaleB,
        float beta, void* C, DataType typeC, int ldc, float act_scale = 0.0f,
        const uint8_t* blk_scA = nullptr, const uint8_t* blk_scB = nullptr)
    {
        bool aQ = isQuantized(typeA), bQ = isQuantized(typeB);
        if (aQ && bQ)
        {
            if (typeC == DataType::FP8_E4M3)
            {
                gemm(TransA, TransB, M, N, K, alpha,
                    (const uint8_t*)A, lda, (const uint8_t*)B, ldb,
                    beta, (uint8_t*)C, ldc, invScaleA, invScaleB);
            }
            else
            {
                gemm(TransA, TransB, M, N, K, alpha,
                    (const uint8_t*)A, lda, (const uint8_t*)B, ldb,
                    beta, (bfloat16*)C, ldc, invScaleA, invScaleB);
            }
        }
        else if (aQ)
        {
            // 先尝试 cublasLt 直接 FP8×HALF→HALF 路径（零转换开销）
            if (lt_handle_ && typeB == DataType::HALF && typeC == DataType::HALF)
            {
                cudaMemcpy(d_scale_a_, &invScaleA, sizeof(float), cudaMemcpyHostToDevice);
                if (lt_quant_gemm(get_trans(TransA), get_trans(TransB), M, N, K, alpha,
                        A, lda, typeA, B, ldb, DataType::HALF, beta, C, ldc, DataType::HALF))
                {
                    if (cuda_clear_last_error)
                    {
                        cuda_clear_last_error();
                    }
                    return;
                }
                if (cuda_clear_last_error)
                {
                    cuda_clear_last_error();
                }
                // cublasLt 不支持此路径；回退到 HALF 快速路径
            }
            // 快速 HALF 路径：FP8 dequant→HALF + 直接 HALF×HALF GEMM（无类型转换）
            if (typeB == DataType::HALF && typeC == DataType::HALF)
            {
                int szA = lda * (TransA == MATRIX_NO_TRANS ? K : M);
                ensureWS(&ws_A_, &ws_A_cap_, (size_t)szA * sizeof(half));
                half* halfA = (half*)(ws_A_);
                if (!halfA)
                {
                    return;
                }
                cuda_convert(A, typeA, halfA, DataType::HALF, (unsigned int)szA, invScaleA);
                float fa = alpha, fb = beta;
                cublasGemmEx(handle_, get_trans(TransA), get_trans(TransB), M, N, K,
                    &fa, halfA, CUDA_R_16F, lda,
                    B, CUDA_R_16F, ldb,
                    &fb, C, CUDA_R_16F, ldc,
                    CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
                cuda_check_error("cublasGemmEx_w8a16_half");
                return;
            }
            // FP8×BF16→BF16 W8A16：Blackwell (RTX 5090) 上 lt_quant_gemm 会返回 SUCCESS
            // 但输出垃圾（与 W8A8 问题相同）；始终走下方 dequant→BF16 回退路径。
            // if (lt_handle_ && typeB == DataType::BFLOAT16 && typeC == DataType::BFLOAT16) { ... }
            // 快速 BF16 路径：FP8/FP4 dequant→BF16 + 直接 BF16×BF16 GEMM（无类型转换）
            if (typeB == DataType::BFLOAT16 && typeC == DataType::BFLOAT16)
            {
                int szA = lda * (TransA == MATRIX_NO_TRANS ? K : M);
                ensureWS(&ws_A_, &ws_A_cap_, (size_t)szA * sizeof(bfloat16));
                bfloat16* bf16Aw = (bfloat16*)(ws_A_);
                if (!bf16Aw)
                {
                    return;
                }
                if (blk_scA && typeA == DataType::FP4_E2M1)
                    cuda_fp4_blockscale_to_bf16(A, blk_scA, bf16Aw, (unsigned int)szA, 16);
                else
                    cuda_convert(A, typeA, bf16Aw, DataType::BFLOAT16, (unsigned int)szA, invScaleA);
                float fa = alpha, fb = beta;
                cublasGemmEx(handle_, get_trans(TransA), get_trans(TransB), M, N, K,
                    &fa, bf16Aw, CUDA_R_16BF, lda,
                    B, CUDA_R_16BF, ldb,
                    &fb, C, CUDA_R_16BF, ldc,
                    CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
                cuda_check_error("cublasGemmEx_w8a16_bf16");
                return;
            }
            // HALF activation/output: convert to/from BF16 so W8A16 path works
            const bfloat16* bf16B = (const bfloat16*)B;
            bool needConvB = (typeB == DataType::HALF);
            bool needConvC = (typeC == DataType::HALF);
            int szB = ldb * (TransB == MATRIX_NO_TRANS ? N : K);
            int szC = ldc * N;
            if (needConvB)
            {
                ensureWSv(&ws_fp8a_, &ws_fp8a_cap_, (size_t)szB * sizeof(bfloat16));
                if (!ws_fp8a_)
                {
                    return;
                }
                cuda_convert(B, DataType::HALF, ws_fp8a_, DataType::BFLOAT16, (unsigned int)szB, 1.0f);
                bf16B = (const bfloat16*)(ws_fp8a_);
            }
            if (typeC == DataType::FP8_E4M3)
            {
                // W8A16 → FP8：先计算到 BF16 中间缓冲，再重新量化为 FP8
                ensureWSv(&ws_tmp_, &ws_tmp_cap_, (size_t)szC * sizeof(bfloat16));
                bfloat16* bf16Ctmp = (bfloat16*)(ws_tmp_);
                if (!bf16Ctmp)
                {
                    return;
                }
                gemm(TransA, TransB, M, N, K, alpha,
                    (const uint8_t*)A, lda, bf16B, ldb,
                    0.f, bf16Ctmp, ldc, invScaleA, typeA, act_scale);
                // BF16→FP8 at scale=1.0 (与 W8A8 路径保持一致)
                cuda_convert(bf16Ctmp, DataType::BFLOAT16, C, DataType::FP8_E4M3, (unsigned int)szC, 1.0f);
                last_fp8_out_scale_ = 1.0f;
            }
            else if (needConvC)
            {
                // W8A16 HALF output: GEMM into BF16 temp, then convert to HALF
                ensureWSv(&ws_fp8b_, &ws_fp8b_cap_, (size_t)szC * sizeof(bfloat16));
                bfloat16* bf16Cout = (bfloat16*)(ws_fp8b_);
                if (!bf16Cout)
                {
                    return;
                }
                if (beta != 0.f)
                {
                    cuda_convert(C, DataType::HALF, ws_fp8b_, DataType::BFLOAT16, (unsigned int)szC, 1.0f);
                }
                gemm(TransA, TransB, M, N, K, alpha,
                    (const uint8_t*)A, lda, bf16B, ldb,
                    beta, bf16Cout, ldc, invScaleA, typeA, act_scale);
                cuda_convert(ws_fp8b_, DataType::BFLOAT16, C, DataType::HALF, (unsigned int)szC, 1.0f);
            }
            else
            {
                gemm(TransA, TransB, M, N, K, alpha,
                    (const uint8_t*)A, lda, bf16B, ldb,
                    beta, (bfloat16*)C, ldc, invScaleA, typeA, act_scale);
            }
        }
        else if (bQ)
        {
            // 先尝试 cublasLt 直接 HALF×FP8→HALF 路径（零转换开销）
            if (lt_handle_ && typeA == DataType::HALF && typeC == DataType::HALF)
            {
                cudaMemcpy(d_scale_b_, &invScaleB, sizeof(float), cudaMemcpyHostToDevice);
                if (lt_quant_gemm(get_trans(TransA), get_trans(TransB), M, N, K, alpha,
                        A, lda, DataType::HALF, B, ldb, typeB, beta, C, ldc, DataType::HALF))
                {
                    if (cuda_clear_last_error)
                    {
                        cuda_clear_last_error();
                    }
                    return;
                }
                if (cuda_clear_last_error)
                {
                    cuda_clear_last_error();
                }
                // cublasLt 不支持此路径；回退到 HALF→BF16 转换路径
            }
            // 快速 HALF 路径：FP8 dequant→HALF + 直接 HALF×HALF GEMM（无类型转换）
            if (typeA == DataType::HALF && typeC == DataType::HALF)
            {
                int szB2 = ldb * (TransB == MATRIX_NO_TRANS ? N : K);
                ensureWS(&ws_A_, &ws_A_cap_, (size_t)szB2 * sizeof(half));
                half* halfB = (half*)(ws_A_);
                if (!halfB)
                {
                    return;
                }
                cuda_convert(B, typeB, halfB, DataType::HALF, (unsigned int)szB2, invScaleB);
                float fa = alpha, fb = beta;
                cublasGemmEx(handle_, get_trans(TransA), get_trans(TransB), M, N, K,
                    &fa, A, CUDA_R_16F, lda,
                    halfB, CUDA_R_16F, ldb,
                    &fb, C, CUDA_R_16F, ldc,
                    CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
                cuda_check_error("cublasGemmEx_w8a16_half_bQ");
                return;
            }
            // BF16×FP8→BF16 W8A16：同上 Blackwell 问题，始终走 dequant→BF16 回退路径。
            // if (lt_handle_ && typeA == DataType::BFLOAT16 && typeC == DataType::BFLOAT16) { ... }
            // 快速 BF16 路径：FP8/FP4 dequant→BF16 + 直接 BF16×BF16 GEMM（无类型转换）
            if (typeA == DataType::BFLOAT16 && typeC == DataType::BFLOAT16)
            {
                int szB2 = ldb * (TransB == MATRIX_NO_TRANS ? N : K);
                ensureWS(&ws_A_, &ws_A_cap_, (size_t)szB2 * sizeof(bfloat16));
                bfloat16* bf16Bw = (bfloat16*)(ws_A_);
                if (!bf16Bw)
                {
                    return;
                }
                if (blk_scB && typeB == DataType::FP4_E2M1)
                    cuda_fp4_blockscale_to_bf16(B, blk_scB, bf16Bw, (unsigned int)szB2, 16);
                else
                    cuda_convert(B, typeB, bf16Bw, DataType::BFLOAT16, (unsigned int)szB2, invScaleB);
                float fa = alpha, fb = beta;
                cublasGemmEx(handle_, get_trans(TransA), get_trans(TransB), M, N, K,
                    &fa, A, CUDA_R_16BF, lda,
                    bf16Bw, CUDA_R_16BF, ldb,
                    &fb, C, CUDA_R_16BF, ldc,
                    CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
                cuda_check_error("cublasGemmEx_w8a16_bf16_bQ");
                return;
            }
            // HALF activation/output: convert to/from BF16 so W8A16 path works
            const bfloat16* bf16A = (const bfloat16*)A;
            bool needConvA = (typeA == DataType::HALF);
            bool needConvC = (typeC == DataType::HALF);
            int szA = lda * (TransA == MATRIX_NO_TRANS ? K : M);
            int szC = ldc * N;
            if (needConvA)
            {
                ensureWSv(&ws_fp8a_, &ws_fp8a_cap_, (size_t)szA * sizeof(bfloat16));
                if (!ws_fp8a_)
                {
                    return;
                }
                cuda_convert(A, DataType::HALF, ws_fp8a_, DataType::BFLOAT16, (unsigned int)szA, 1.0f);
                bf16A = (const bfloat16*)(ws_fp8a_);
            }
            if (typeC == DataType::FP8_E4M3)
            {
                // W8A16 → FP8：先计算到 BF16 中间缓冲，再重新量化为 FP8
                ensureWSv(&ws_tmp_, &ws_tmp_cap_, (size_t)szC * sizeof(bfloat16));
                bfloat16* bf16Ctmp = (bfloat16*)(ws_tmp_);
                if (!bf16Ctmp)
                {
                    return;
                }
                gemm(TransA, TransB, M, N, K, alpha,
                    bf16A, lda, (const uint8_t*)B, ldb,
                    0.f, bf16Ctmp, ldc, invScaleB, typeB, act_scale);
                // BF16→FP8 at scale=1.0 (与 W8A8 路径保持一致)
                cuda_convert(bf16Ctmp, DataType::BFLOAT16, C, DataType::FP8_E4M3, (unsigned int)szC, 1.0f);
                last_fp8_out_scale_ = 1.0f;
            }
            else if (needConvC)
            {
                // W8A16 HALF output: GEMM into BF16 temp, then convert to HALF
                ensureWSv(&ws_fp8b_, &ws_fp8b_cap_, (size_t)szC * sizeof(bfloat16));
                bfloat16* bf16Cout = (bfloat16*)(ws_fp8b_);
                if (!bf16Cout)
                {
                    return;
                }
                if (beta != 0.f)
                {
                    cuda_convert(C, DataType::HALF, ws_fp8b_, DataType::BFLOAT16, (unsigned int)szC, 1.0f);
                }
                gemm(TransA, TransB, M, N, K, alpha,
                    bf16A, lda, (const uint8_t*)B, ldb,
                    beta, bf16Cout, ldc, invScaleB, typeB, act_scale);
                cuda_convert(ws_fp8b_, DataType::BFLOAT16, C, DataType::HALF, (unsigned int)szC, 1.0f);
            }
            else
            {
                gemm(TransA, TransB, M, N, K, alpha,
                    bf16A, lda, (const uint8_t*)B, ldb,
                    beta, (bfloat16*)C, ldc, invScaleB, typeB, act_scale);
            }
        }
        else if (typeC == DataType::DOUBLE)
        {
            gemm(TransA, TransB, M, N, K, (double)alpha,
                (const double*)A, lda, (const double*)B, ldb,
                (double)beta, (double*)C, ldc);
        }
        else
        {
            float fa = alpha, fb = beta;
            cudaDataType ct;
            switch (typeC)
            {
            case DataType::HALF: ct = CUDA_R_16F; break;
            case DataType::BFLOAT16: ct = CUDA_R_16BF; break;
            default: ct = CUDA_R_32F; break;
            }
            cublasGemmEx(handle_, get_trans(TransA), get_trans(TransB), M, N, K,
                &fa, A, ct, lda, B, ct, ldb,
                &fb, C, ct, ldc,
                CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        }
    }

protected:
    cublasHandle_t handle_ = nullptr;

    // Persistent float workspace for BF16/FP8 GEMM conversions.
    // Grows on demand; never shrinks (freed in destroy()).  Eliminates per-GEMM
    // cudaMalloc/cudaFree overhead which dominates decode latency for W8A16.
    float* ws_A_ = nullptr;
    size_t ws_A_cap_ = 0;
    float* ws_B_ = nullptr;
    size_t ws_B_cap_ = 0;
    float* ws_C_ = nullptr;
    size_t ws_C_cap_ = 0;
    void* ws_tmp_ = nullptr;
    size_t ws_tmp_cap_ = 0;    // FP8→FP8 输出路径的 BF16 临时缓冲区
    void* ws_fp8a_ = nullptr;
    size_t ws_fp8a_cap_ = 0;    // dedicated FP8 activation buffer for A operand (W8A8 dynamic quant)
    void* ws_fp8b_ = nullptr;
    size_t ws_fp8b_cap_ = 0;    // dedicated FP8 activation buffer for B operand (W8A8 dynamic quant)

    // Generic byte-level workspace helper (works for uint8_t, bfloat16, void*, etc.)
    CUBLAS_FUNCTION void ensureWSv(void** ptr, size_t* cap, size_t needed)
    {
        if (needed > *cap)
        {
            if (*ptr)
            {
                cudaFree(*ptr);
                *ptr = nullptr;
            }
            if (cudaMalloc(ptr, needed) == cudaSuccess)
            {
                *cap = needed;
            }
            else
            {
                *ptr = nullptr;    // malloc failed; leave *cap unchanged so next call retries
            }
        }
    }
    CUBLAS_FUNCTION void ensureWS(float** ptr, size_t* cap, size_t needed)
    {
        if (needed > *cap)
        {
            if (*ptr)
            {
                cudaFree(*ptr);
                *ptr = nullptr;
            }
            if (cudaMalloc((void**)(ptr), needed) == cudaSuccess)
            {
                *cap = needed;
            }
            else
            {
                *ptr = nullptr;    // malloc failed; leave *cap unchanged so next call retries
            }
        }
    }

    // ── W8A8 激活数据类型控制 ────────────────────────────────────────────────
    DataType act_data_type_ = DataType::BFLOAT16;
    // 最近一次 FP8 输出的动态量化 scale（由 cuda_bf16_to_fp8e4m3_dynamic 写入 d_scale_a_ 后回传）
    float last_fp8_out_scale_ = 1.0f;

public:
    void setActDataType(DataType dt) { act_data_type_ = dt; }
    // 获取最近一次 FP8 GEMM 输出的量化 scale，用于更新输出 Matrix 的 quant_scale_
    float getLastFp8OutScale() const { return last_fp8_out_scale_; }

private:
    // ── W8A8 FP8 GEMM 的 cublasLt 成员 ────────────────────────────────────
    cublasLtHandle_t lt_handle_ = nullptr;
    float* d_scale_a_ = nullptr;       // device float: scaleA for cublasLt (weight inv_scale)
    float* d_scale_b_ = nullptr;       // device float: scaleB for cublasLt (activation inv_scale)
    float* d_act_absmax_ = nullptr;    // device float: scratch for dynamic abs-max reduction
    void* lt_workspace_ = nullptr;     // 4 MB device workspace for cublasLt
    static constexpr size_t lt_workspace_size_ = 4 * 1024 * 1024;

    // ── cublasLt 句柄类型的 RAII 封装 ──────────────────────────────────────
    struct LtMatmulDesc
    {
        cublasLtMatmulDesc_t h = nullptr;
        LtMatmulDesc(cublasComputeType_t compute, cudaDataType_t scale)
        {
            cublasLtMatmulDescCreate(&h, compute, scale);
        }
        ~LtMatmulDesc()
        {
            if (h)
            {
                cublasLtMatmulDescDestroy(h);
            }
        }
        bool ok() const { return h != nullptr; }
        operator cublasLtMatmulDesc_t() const { return h; }
        LtMatmulDesc(const LtMatmulDesc&) = delete;
        LtMatmulDesc& operator=(const LtMatmulDesc&) = delete;
    };
    struct LtMatrixLayout
    {
        cublasLtMatrixLayout_t h = nullptr;
        LtMatrixLayout(cudaDataType_t type, int64_t rows, int64_t cols, int64_t ld)
        {
            cublasLtMatrixLayoutCreate(&h, type, rows, cols, ld);
        }
        ~LtMatrixLayout()
        {
            if (h)
            {
                cublasLtMatrixLayoutDestroy(h);
            }
        }
        bool ok() const { return h != nullptr; }
        operator cublasLtMatrixLayout_t() const { return h; }
        LtMatrixLayout(const LtMatrixLayout&) = delete;
        LtMatrixLayout& operator=(const LtMatrixLayout&) = delete;
    };
    struct LtMatmulPref
    {
        cublasLtMatmulPreference_t h = nullptr;
        LtMatmulPref() { cublasLtMatmulPreferenceCreate(&h); }
        ~LtMatmulPref()
        {
            if (h)
            {
                cublasLtMatmulPreferenceDestroy(h);
            }
        }
        bool ok() const { return h != nullptr; }
        operator cublasLtMatmulPreference_t() const { return h; }
        LtMatmulPref(const LtMatmulPref&) = delete;
        LtMatmulPref& operator=(const LtMatmulPref&) = delete;
    };

    // Unified cublasLt GEMM for quantized operands.
    // typeA: weight type  (FP8_E4M3 / FP8_E5M2 / FP4_E2M1)
    // typeB: activation type (BFLOAT16 for W8A16, FP8_E4M3 for W8A8)
    // Caller must set d_scale_a_ always; d_scale_b_ only when typeB is quantized.
    CUBLAS_FUNCTION bool lt_quant_gemm(
        cublasOperation_t transA, cublasOperation_t transB,
        int M, int N, int K,
        float alpha, const void* A, int lda, DataType typeA,
        const void* B, int ldb, DataType typeB,
        float beta, void* C, int ldc, DataType typeC = DataType::BFLOAT16)
    {
        if (!lt_handle_)
        {
            return false;
        }
        auto toCudaDt = [](DataType dt) -> cudaDataType_t
        {
            switch (dt)
            {
            case DataType::FP8_E4M3: return CUDA_R_8F_E4M3;
            case DataType::FP8_E5M2: return CUDA_R_8F_E5M2;
            case DataType::FP4_E2M1: return CUDA_R_4F_E2M1;
            case DataType::HALF: return CUDA_R_16F;
            case DataType::BFLOAT16: return CUDA_R_16BF;
            default: return CUDA_R_32F;
            }
        };
        // FP4 (CUDA_R_4F_E2M1) requires per-block (per-16 or per-32 elements) scale vectors via
        // CUBLASLT_MATMUL_DESC_SCALE_MODE; a single d_scale_a_/d_scale_b_ float is insufficient.
        // TODO: implement block-wise quantization and pass scale vectors before enabling FP4 here.
        if (typeA == DataType::FP4_E2M1 || typeB == DataType::FP4_E2M1)
        {
            return false;    // block-wise scale not yet implemented; caller will use decode fallback
        }
        LtMatmulDesc op_desc(CUBLAS_COMPUTE_32F, CUDA_R_32F);
        if (!op_desc.ok())
        {
            return false;
        }
        if (cublasLtMatmulDescSetAttribute(op_desc, CUBLASLT_MATMUL_DESC_TRANSA, &transA, sizeof(transA)) != CUBLAS_STATUS_SUCCESS)
        {
            return false;
        }
        if (cublasLtMatmulDescSetAttribute(op_desc, CUBLASLT_MATMUL_DESC_TRANSB, &transB, sizeof(transB)) != CUBLAS_STATUS_SUCCESS)
        {
            return false;
        }
        if (isQuantized(typeA))
        {
            if (cublasLtMatmulDescSetAttribute(op_desc, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &d_scale_a_, sizeof(d_scale_a_)) != CUBLAS_STATUS_SUCCESS)
            {
                return false;
            }
        }
        if (isQuantized(typeB))
        {
            if (cublasLtMatmulDescSetAttribute(op_desc, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &d_scale_b_, sizeof(d_scale_b_)) != CUBLAS_STATUS_SUCCESS)
            {
                return false;
            }
        }
        int64_t rowsA = (transA == CUBLAS_OP_N) ? M : K;
        int64_t colsA = (transA == CUBLAS_OP_N) ? K : M;
        int64_t rowsB = (transB == CUBLAS_OP_N) ? K : N;
        int64_t colsB = (transB == CUBLAS_OP_N) ? N : K;
        LtMatrixLayout layoutA(toCudaDt(typeA), rowsA, colsA, lda);
        LtMatrixLayout layoutB(toCudaDt(typeB), rowsB, colsB, ldb);
        LtMatrixLayout layoutC(toCudaDt(typeC), M, N, ldc);
        if (!layoutA.ok() || !layoutB.ok() || !layoutC.ok())
        {
            return false;
        }
        LtMatmulPref pref;
        if (!pref.ok())
        {
            return false;
        }
        size_t ws = lt_workspace_size_;
        if (cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws, sizeof(ws)) != CUBLAS_STATUS_SUCCESS)
        {
            return false;
        }
        cublasLtMatmulHeuristicResult_t heur{};
        int found = 0;
        if (cublasLtMatmulAlgoGetHeuristic(lt_handle_, op_desc, layoutA, layoutB, layoutC, layoutC, pref, 1, &heur, &found) != CUBLAS_STATUS_SUCCESS || found == 0)
        {
            return false;
        }
        auto st = cublasLtMatmul(lt_handle_, op_desc,
            &alpha, A, layoutA, B, layoutB,
            &beta, C, layoutC, C, layoutC,
            &heur.algo, lt_workspace_, lt_workspace_size_, nullptr);
        return (st == CUBLAS_STATUS_SUCCESS);
    }
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
