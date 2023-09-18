#pragma once
#include "blas_types.h"

#ifdef ENABLE_HIP
#include "rocblas/rocblas.h"

namespace cccc
{

#if defined(NORMAL_BLAS)
#define ROCBLAS_FUNCTION inline
#elif defined(STATIC_BLAS)
#define ROCBLAS_FUNCTION static inline
#else
#define ROCBLAS_FUNCTION virtual
#endif

//Class of rocblas, overload functions with the same name for float and double.
class Rocblas : Blas
{
public:
    Rocblas() {}
    ~Rocblas() { destroy(); }

protected:
    ROCBLAS_FUNCTION rocblas_operation get_trans(MatrixTransType t)
    {
        return t == MATRIX_NO_TRANS ? rocblas_operation_none : rocblas_operation_transpose;
    }
    ROCBLAS_FUNCTION rocblas_fill get_uplo(MatrixFillType t)
    {
        return t == MATRIX_UPPER ? rocblas_fill_upper : rocblas_fill_lower;
    }
    ROCBLAS_FUNCTION rocblas_diagonal get_diag(MatrixDiagType t)
    {
        return t == MATRIX_NON_UNIT ? rocblas_diagonal_non_unit : rocblas_diagonal_unit;
    }
    ROCBLAS_FUNCTION rocblas_side get_side(MatrixSideType t)
    {
        return t == MATRIX_LEFT ? rocblas_side_left : rocblas_side_right;
    }

public:
    ROCBLAS_FUNCTION rocblas_status init()
    {
        auto r = rocblas_create_handle(&handle_);
        //rocblas_setMathMode(handle_, CUBLAS_TENSOR_OP_MATH);
        return r;
    }
    ROCBLAS_FUNCTION void destroy()
    {
        if (handle_)
        {
            rocblas_destroy_handle(handle_);
        }
    }
    ROCBLAS_FUNCTION void set_handle(rocblas_handle h) { handle_ = h; }
    ROCBLAS_FUNCTION int get_version()
    {
        int ver = ROCBLAS_VERSION_MAJOR * 10000 + ROCBLAS_VERSION_MINOR * 100 + ROCBLAS_VERSION_PATCH;
        return ver;
    }

public:
    ROCBLAS_FUNCTION float dot(const int N, const float* X, const int incX, const float* Y, const int incY)
    {
        float r;
        rocblas_sdot(handle_, N, X, incX, Y, incY, &r);
        return r;
    }
    ROCBLAS_FUNCTION double dot(const int N, const double* X, const int incX, const double* Y, const int incY)
    {
        double r;
        rocblas_ddot(handle_, N, X, incX, Y, incY, &r);
        return r;
    }
    ROCBLAS_FUNCTION float nrm2(const int N, const float* X, const int incX)
    {
        float r;
        rocblas_snrm2(handle_, N, X, incX, &r);
        return r;
    }
    ROCBLAS_FUNCTION float asum(const int N, const float* X, const int incX)
    {
        float r;
        rocblas_sasum(handle_, N, X, incX, &r);
        return r;
    }
    ROCBLAS_FUNCTION double nrm2(const int N, const double* X, const int incX)
    {
        double r;
        rocblas_dnrm2(handle_, N, X, incX, &r);
        return r;
    }
    ROCBLAS_FUNCTION double asum(const int N, const double* X, const int incX)
    {
        double r;
        rocblas_dasum(handle_, N, X, incX, &r);
        return r;
    }
    ROCBLAS_FUNCTION int iamax(const int N, const float* X, const int incX)
    {
        int r;
        rocblas_isamax(handle_, N, X, incX, &r);
        return r - 1;
    }
    ROCBLAS_FUNCTION int iamax(const int N, const double* X, const int incX)
    {
        int r;
        rocblas_idamax(handle_, N, X, incX, &r);
        return r - 1;
    }
    ROCBLAS_FUNCTION void swap(const int N, float* X, const int incX, float* Y, const int incY)
    {
        rocblas_sswap(handle_, N, X, incX, Y, incY);
    }
    ROCBLAS_FUNCTION void copy(const int N, const float* X, const int incX, float* Y, const int incY)
    {
        rocblas_scopy(handle_, N, X, incX, Y, incY);
    }
    ROCBLAS_FUNCTION void axpy(const int N, const float alpha, const float* X, const int incX, float* Y, const int incY)
    {
        rocblas_saxpy(handle_, N, &alpha, X, incX, Y, incY);
    }
    ROCBLAS_FUNCTION void swap(const int N, double* X, const int incX, double* Y, const int incY)
    {
        rocblas_dswap(handle_, N, X, incX, Y, incY);
    }
    ROCBLAS_FUNCTION void copy(const int N, const double* X, const int incX, double* Y, const int incY)
    {
        rocblas_dcopy(handle_, N, X, incX, Y, incY);
    }
    ROCBLAS_FUNCTION void axpy(const int N, const double alpha, const double* X, const int incX, double* Y, const int incY)
    {
        rocblas_daxpy(handle_, N, &alpha, X, incX, Y, incY);
    }
    ROCBLAS_FUNCTION void rotg(float* a, float* b, float* c, float* s)
    {
        rocblas_srotg(handle_, a, b, c, s);
    }
    ROCBLAS_FUNCTION void rotmg(float* d1, float* d2, float* b1, const float b2, float* P)
    {
        rocblas_srotmg(handle_, d1, d2, b1, &b2, P);
    }
    ROCBLAS_FUNCTION void rot(const int N, float* X, const int incX, float* Y, const int incY, const float c, const float s)
    {
        rocblas_srot(handle_, N, X, incX, Y, incY, &c, &s);
    }
    ROCBLAS_FUNCTION void rotm(const int N, float* X, const int incX, float* Y, const int incY, const float* P)
    {
        rocblas_srotm(handle_, N, X, incX, Y, incY, P);
    }
    ROCBLAS_FUNCTION void rotg(double* a, double* b, double* c, double* s)
    {
        rocblas_drotg(handle_, a, b, c, s);
    }
    ROCBLAS_FUNCTION void rotmg(double* d1, double* d2, double* b1, const double b2, double* P)
    {
        rocblas_drotmg(handle_, d1, d2, b1, &b2, P);
    }
    ROCBLAS_FUNCTION void rot(const int N, double* X, const int incX, double* Y, const int incY, const double c, const double s)
    {
        rocblas_drot(handle_, N, X, incX, Y, incY, &c, &s);
    }
    ROCBLAS_FUNCTION void rotm(const int N, double* X, const int incX, double* Y, const int incY, const double* P)
    {
        rocblas_drotm(handle_, N, X, incX, Y, incY, P);
    }
    ROCBLAS_FUNCTION void scal(const int N, const float alpha, float* X, const int incX)
    {
        rocblas_sscal(handle_, N, &alpha, X, incX);
    }
    ROCBLAS_FUNCTION void scal(const int N, const double alpha, double* X, const int incX)
    {
        rocblas_dscal(handle_, N, &alpha, X, incX);
    }
    ROCBLAS_FUNCTION void gemv(const MatrixTransType TransA, const int M, const int N, const float alpha, const float* A, const int lda, const float* X, const int incX, const float beta, float* Y, const int incY)
    {
        rocblas_sgemv(handle_, get_trans(TransA), M, N, &alpha, A, lda, X, incX, &beta, Y, incY);
    }
    ROCBLAS_FUNCTION void gbmv(const MatrixTransType TransA, const int M, const int N, const int KL, const int KU, const float alpha, const float* A, const int lda, const float* X, const int incX, const float beta, float* Y, const int incY)
    {
        rocblas_sgbmv(handle_, get_trans(TransA), M, N, KL, KU, &alpha, A, lda, X, incX, &beta, Y, incY);
    }
    ROCBLAS_FUNCTION void trmv(const MatrixFillType Uplo, const MatrixTransType TransA, const MatrixDiagType Diag, const int N, const float* A, const int lda, float* X, const int incX)
    {
        rocblas_strmv(handle_, get_uplo(Uplo), get_trans(TransA), get_diag(Diag), N, A, lda, X, incX);
    }
    ROCBLAS_FUNCTION void tbmv(const MatrixFillType Uplo, const MatrixTransType TransA, const MatrixDiagType Diag, const int N, const int K, const float* A, const int lda, float* X, const int incX)
    {
        rocblas_stbmv(handle_, get_uplo(Uplo), get_trans(TransA), get_diag(Diag), N, K, A, lda, X, incX);
    }
    ROCBLAS_FUNCTION void tpmv(const MatrixFillType Uplo, const MatrixTransType TransA, const MatrixDiagType Diag, const int N, const float* Ap, float* X, const int incX)
    {
        rocblas_stpmv(handle_, get_uplo(Uplo), get_trans(TransA), get_diag(Diag), N, Ap, X, incX);
    }
    ROCBLAS_FUNCTION void trsv(const MatrixFillType Uplo, const MatrixTransType TransA, const MatrixDiagType Diag, const int N, const float* A, const int lda, float* X, const int incX)
    {
        rocblas_strsv(handle_, get_uplo(Uplo), get_trans(TransA), get_diag(Diag), N, A, lda, X, incX);
    }
    ROCBLAS_FUNCTION void tbsv(const MatrixFillType Uplo, const MatrixTransType TransA, const MatrixDiagType Diag, const int N, const int K, const float* A, const int lda, float* X, const int incX)
    {
        rocblas_stbsv(handle_, get_uplo(Uplo), get_trans(TransA), get_diag(Diag), N, K, A, lda, X, incX);
    }
    ROCBLAS_FUNCTION void tpsv(const MatrixFillType Uplo, const MatrixTransType TransA, const MatrixDiagType Diag, const int N, const float* Ap, float* X, const int incX)
    {
        rocblas_stpsv(handle_, get_uplo(Uplo), get_trans(TransA), get_diag(Diag), N, Ap, X, incX);
    }
    ROCBLAS_FUNCTION void gemv(const MatrixTransType TransA, const int M, const int N, const double alpha, const double* A, const int lda, const double* X, const int incX, const double beta, double* Y, const int incY)
    {
        rocblas_dgemv(handle_, get_trans(TransA), M, N, &alpha, A, lda, X, incX, &beta, Y, incY);
    }
    ROCBLAS_FUNCTION void gbmv(const MatrixTransType TransA, const int M, const int N, const int KL, const int KU, const double alpha, const double* A, const int lda, const double* X, const int incX, const double beta, double* Y, const int incY)
    {
        rocblas_dgbmv(handle_, get_trans(TransA), M, N, KL, KU, &alpha, A, lda, X, incX, &beta, Y, incY);
    }
    ROCBLAS_FUNCTION void trmv(const MatrixFillType Uplo, const MatrixTransType TransA, const MatrixDiagType Diag, const int N, const double* A, const int lda, double* X, const int incX)
    {
        rocblas_dtrmv(handle_, get_uplo(Uplo), get_trans(TransA), get_diag(Diag), N, A, lda, X, incX);
    }
    ROCBLAS_FUNCTION void tbmv(const MatrixFillType Uplo, const MatrixTransType TransA, const MatrixDiagType Diag, const int N, const int K, const double* A, const int lda, double* X, const int incX)
    {
        rocblas_dtbmv(handle_, get_uplo(Uplo), get_trans(TransA), get_diag(Diag), N, K, A, lda, X, incX);
    }
    ROCBLAS_FUNCTION void tpmv(const MatrixFillType Uplo, const MatrixTransType TransA, const MatrixDiagType Diag, const int N, const double* Ap, double* X, const int incX)
    {
        rocblas_dtpmv(handle_, get_uplo(Uplo), get_trans(TransA), get_diag(Diag), N, Ap, X, incX);
    }
    ROCBLAS_FUNCTION void trsv(const MatrixFillType Uplo, const MatrixTransType TransA, const MatrixDiagType Diag, const int N, const double* A, const int lda, double* X, const int incX)
    {
        rocblas_dtrsv(handle_, get_uplo(Uplo), get_trans(TransA), get_diag(Diag), N, A, lda, X, incX);
    }
    ROCBLAS_FUNCTION void tbsv(const MatrixFillType Uplo, const MatrixTransType TransA, const MatrixDiagType Diag, const int N, const int K, const double* A, const int lda, double* X, const int incX)
    {
        rocblas_dtbsv(handle_, get_uplo(Uplo), get_trans(TransA), get_diag(Diag), N, K, A, lda, X, incX);
    }
    ROCBLAS_FUNCTION void tpsv(const MatrixFillType Uplo, const MatrixTransType TransA, const MatrixDiagType Diag, const int N, const double* Ap, double* X, const int incX)
    {
        rocblas_dtpsv(handle_, get_uplo(Uplo), get_trans(TransA), get_diag(Diag), N, Ap, X, incX);
    }
    ROCBLAS_FUNCTION void symv(const MatrixFillType Uplo, const int N, const float alpha, const float* A, const int lda, const float* X, const int incX, const float beta, float* Y, const int incY)
    {
        rocblas_ssymv(handle_, get_uplo(Uplo), N, &alpha, A, lda, X, incX, &beta, Y, incY);
    }
    ROCBLAS_FUNCTION void sbmv(const MatrixFillType Uplo, const int N, const int K, const float alpha, const float* A, const int lda, const float* X, const int incX, const float beta, float* Y, const int incY)
    {
        rocblas_ssbmv(handle_, get_uplo(Uplo), N, K, &alpha, A, lda, X, incX, &beta, Y, incY);
    }
    ROCBLAS_FUNCTION void spmv(const MatrixFillType Uplo, const int N, const float alpha, const float* Ap, const float* X, const int incX, const float beta, float* Y, const int incY)
    {
        rocblas_sspmv(handle_, get_uplo(Uplo), N, &alpha, Ap, X, incX, &beta, Y, incY);
    }
    ROCBLAS_FUNCTION void ger(const int M, const int N, const float alpha, const float* X, const int incX, const float* Y, const int incY, float* A, const int lda)
    {
        rocblas_sger(handle_, M, N, &alpha, X, incX, Y, incY, A, lda);
    }
    ROCBLAS_FUNCTION void syr(const MatrixFillType Uplo, const int N, const float alpha, const float* X, const int incX, float* A, const int lda)
    {
        rocblas_ssyr(handle_, get_uplo(Uplo), N, &alpha, X, incX, A, lda);
    }
    ROCBLAS_FUNCTION void spr(const MatrixFillType Uplo, const int N, const float alpha, const float* X, const int incX, float* Ap)
    {
        rocblas_sspr(handle_, get_uplo(Uplo), N, &alpha, X, incX, Ap);
    }
    ROCBLAS_FUNCTION void syr2(const MatrixFillType Uplo, const int N, const float alpha, const float* X, const int incX, const float* Y, const int incY, float* A, const int lda)
    {
        rocblas_ssyr2(handle_, get_uplo(Uplo), N, &alpha, X, incX, Y, incY, A, lda);
    }
    ROCBLAS_FUNCTION void spr2(const MatrixFillType Uplo, const int N, const float alpha, const float* X, const int incX, const float* Y, const int incY, float* A)
    {
        rocblas_sspr2(handle_, get_uplo(Uplo), N, &alpha, X, incX, Y, incY, A);
    }
    ROCBLAS_FUNCTION void symv(const MatrixFillType Uplo, const int N, const double alpha, const double* A, const int lda, const double* X, const int incX, const double beta, double* Y, const int incY)
    {
        rocblas_dsymv(handle_, get_uplo(Uplo), N, &alpha, A, lda, X, incX, &beta, Y, incY);
    }
    ROCBLAS_FUNCTION void sbmv(const MatrixFillType Uplo, const int N, const int K, const double alpha, const double* A, const int lda, const double* X, const int incX, const double beta, double* Y, const int incY)
    {
        rocblas_dsbmv(handle_, get_uplo(Uplo), N, K, &alpha, A, lda, X, incX, &beta, Y, incY);
    }
    ROCBLAS_FUNCTION void spmv(const MatrixFillType Uplo, const int N, const double alpha, const double* Ap, const double* X, const int incX, const double beta, double* Y, const int incY)
    {
        rocblas_dspmv(handle_, get_uplo(Uplo), N, &alpha, Ap, X, incX, &beta, Y, incY);
    }
    ROCBLAS_FUNCTION void ger(const int M, const int N, const double alpha, const double* X, const int incX, const double* Y, const int incY, double* A, const int lda)
    {
        rocblas_dger(handle_, M, N, &alpha, X, incX, Y, incY, A, lda);
    }
    ROCBLAS_FUNCTION void syr(const MatrixFillType Uplo, const int N, const double alpha, const double* X, const int incX, double* A, const int lda)
    {
        rocblas_dsyr(handle_, get_uplo(Uplo), N, &alpha, X, incX, A, lda);
    }
    ROCBLAS_FUNCTION void spr(const MatrixFillType Uplo, const int N, const double alpha, const double* X, const int incX, double* Ap)
    {
        rocblas_dspr(handle_, get_uplo(Uplo), N, &alpha, X, incX, Ap);
    }
    ROCBLAS_FUNCTION void syr2(const MatrixFillType Uplo, const int N, const double alpha, const double* X, const int incX, const double* Y, const int incY, double* A, const int lda)
    {
        rocblas_dsyr2(handle_, get_uplo(Uplo), N, &alpha, X, incX, Y, incY, A, lda);
    }
    ROCBLAS_FUNCTION void spr2(const MatrixFillType Uplo, const int N, const double alpha, const double* X, const int incX, const double* Y, const int incY, double* A)
    {
        rocblas_dspr2(handle_, get_uplo(Uplo), N, &alpha, X, incX, Y, incY, A);
    }
    ROCBLAS_FUNCTION void gemm(const MatrixTransType TransA, const MatrixTransType TransB, const int M, const int N, const int K, const float alpha, const float* A, const int lda, const float* B, const int ldb, const float beta, float* C, const int ldc)
    {
        rocblas_sgemm(handle_, get_trans(TransA), get_trans(TransB), M, N, K, &alpha, A, lda, B, ldb, &beta, C, ldc);
    }
    ROCBLAS_FUNCTION void symm(const MatrixSideType Side, const MatrixFillType Uplo, const int M, const int N, const float alpha, const float* A, const int lda, const float* B, const int ldb, const float beta, float* C, const int ldc)
    {
        rocblas_ssymm(handle_, get_side(Side), get_uplo(Uplo), M, N, &alpha, A, lda, B, ldb, &beta, C, ldc);
    }
    ROCBLAS_FUNCTION void syrk(const MatrixFillType Uplo, const MatrixTransType Trans, const int N, const int K, const float alpha, const float* A, const int lda, const float beta, float* C, const int ldc)
    {
        rocblas_ssyrk(handle_, get_uplo(Uplo), get_trans(Trans), N, K, &alpha, A, lda, &beta, C, ldc);
    }
    ROCBLAS_FUNCTION void syr2k(const MatrixFillType Uplo, const MatrixTransType Trans, const int N, const int K, const float alpha, const float* A, const int lda, const float* B, const int ldb, const float beta, float* C, const int ldc)
    {
        rocblas_ssyr2k(handle_, get_uplo(Uplo), get_trans(Trans), N, K, &alpha, A, lda, B, ldb, &beta, C, ldc);
    }
    ROCBLAS_FUNCTION void trmm(const MatrixSideType Side, const MatrixFillType Uplo, const MatrixTransType TransA, const MatrixDiagType Diag, const int M, const int N, const float alpha, const float* A, const int lda, float* B, const int ldb)
    {
        rocblas_strmm(handle_, get_side(Side), get_uplo(Uplo), get_trans(TransA), get_diag(Diag), M, N, &alpha, A, lda, B, ldb);
    }
    ROCBLAS_FUNCTION void trsm(const MatrixSideType Side, const MatrixFillType Uplo, const MatrixTransType TransA, const MatrixDiagType Diag, const int M, const int N, const float alpha, const float* A, const int lda, float* B, const int ldb)
    {
        rocblas_strsm(handle_, get_side(Side), get_uplo(Uplo), get_trans(TransA), get_diag(Diag), M, N, &alpha, A, lda, B, ldb);
    }
    ROCBLAS_FUNCTION void gemm(const MatrixTransType TransA, const MatrixTransType TransB, const int M, const int N, const int K, const double alpha, const double* A, const int lda, const double* B, const int ldb, const double beta, double* C, const int ldc)
    {
        rocblas_dgemm(handle_, get_trans(TransA), get_trans(TransB), M, N, K, &alpha, A, lda, B, ldb, &beta, C, ldc);
    }
    ROCBLAS_FUNCTION void symm(const MatrixSideType Side, const MatrixFillType Uplo, const int M, const int N, const double alpha, const double* A, const int lda, const double* B, const int ldb, const double beta, double* C, const int ldc)
    {
        rocblas_dsymm(handle_, get_side(Side), get_uplo(Uplo), M, N, &alpha, A, lda, B, ldb, &beta, C, ldc);
    }
    ROCBLAS_FUNCTION void syrk(const MatrixFillType Uplo, const MatrixTransType Trans, const int N, const int K, const double alpha, const double* A, const int lda, const double beta, double* C, const int ldc)
    {
        rocblas_dsyrk(handle_, get_uplo(Uplo), get_trans(Trans), N, K, &alpha, A, lda, &beta, C, ldc);
    }
    ROCBLAS_FUNCTION void syr2k(const MatrixFillType Uplo, const MatrixTransType Trans, const int N, const int K, const double alpha, const double* A, const int lda, const double* B, const int ldb, const double beta, double* C, const int ldc)
    {
        rocblas_dsyr2k(handle_, get_uplo(Uplo), get_trans(Trans), N, K, &alpha, A, lda, B, ldb, &beta, C, ldc);
    }
    ROCBLAS_FUNCTION void trmm(const MatrixSideType Side, const MatrixFillType Uplo, const MatrixTransType TransA, const MatrixDiagType Diag, const int M, const int N, const double alpha, const double* A, const int lda, double* B, const int ldb)
    {
        rocblas_dtrmm(handle_, get_side(Side), get_uplo(Uplo), get_trans(TransA), get_diag(Diag), M, N, &alpha, A, lda, B, ldb);
    }
    ROCBLAS_FUNCTION void trsm(const MatrixSideType Side, const MatrixFillType Uplo, const MatrixTransType TransA, const MatrixDiagType Diag, const int M, const int N, const double alpha, const double* A, const int lda, double* B, const int ldb)
    {
        rocblas_dtrsm(handle_, get_side(Side), get_uplo(Uplo), get_trans(TransA), get_diag(Diag), M, N, &alpha, A, lda, B, ldb);
    }

    //extensions of rocblas
    ROCBLAS_FUNCTION void geam(const MatrixTransType TransA, const MatrixTransType TransB, int m, int n, const float alpha, const float* A, int lda, const float beta, const float* B, int ldb, float* C, int ldc)
    {
        rocblas_sgeam(handle_, get_trans(TransA), get_trans(TransB), m, n, &alpha, A, lda, &beta, B, lda, C, ldc);
    }
    ROCBLAS_FUNCTION void geam(const MatrixTransType TransA, const MatrixTransType TransB, int m, int n, const double alpha, const double* A, int lda, const double beta, const double* B, int ldb, double* C, int ldc)
    {
        rocblas_dgeam(handle_, get_trans(TransA), get_trans(TransB), m, n, &alpha, A, lda, &beta, B, lda, C, ldc);
    }
    ROCBLAS_FUNCTION void dgem(MatrixSideType Side, int m, int n, const float* A, int lda, const float* x, int incx, float* C, int ldc)
    {
        rocblas_sdgmm(handle_, get_side(Side), m, n, A, lda, x, incx, C, ldc);
    }
    ROCBLAS_FUNCTION void dgem(MatrixSideType Side, int m, int n, const double* A, int lda, const double* x, int incx, double* C, int ldc)
    {
        rocblas_ddgmm(handle_, get_side(Side), m, n, A, lda, x, incx, C, ldc);
    }

protected:
    rocblas_handle handle_ = nullptr;
};

}    // namespace cccc
#else
#include "cblas_real.h"
namespace cccc
{
class Rocblas : public Cblas
{
};
}    //namespace cccc
#endif
