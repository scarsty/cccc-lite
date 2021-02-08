#pragma once
#include "blas_types.h"
#include "cblas.h"

namespace cccc
{

//Class of blas, overload functions with the same name for float and double.
//fully static class

#if defined(NORMAL_BLAS)
#define CBLAS_FUNCTION static inline
#elif defined(STATIC_BLAS)
#define CBLAS_FUNCTION static inline
#else
#define CBLAS_FUNCTION virtual
#endif

class Cblas : Blas
{
public:
    Cblas() {}
    ~Cblas() {}

protected:
    CBLAS_FUNCTION CBLAS_TRANSPOSE get_trans(MatrixTransType t)
    {
        return t == MATRIX_NO_TRANS ? CblasNoTrans : CblasTrans;
    }
    CBLAS_FUNCTION CBLAS_UPLO get_uplo(MatrixFillType t)
    {
        return t == MATRIX_UPPER ? CblasUpper : CblasLower;
    }
    CBLAS_FUNCTION CBLAS_DIAG get_diag(MatrixDiagType t)
    {
        return t == MATRIX_NON_UNIT ? CblasNonUnit : CblasUnit;
    }
    CBLAS_FUNCTION CBLAS_SIDE get_side(MatrixSideType t)
    {
        return t == MATRIX_LEFT ? CblasLeft : CblasRight;
    }

public:
    int init() { return 0; }
    int destroy() { return 0; }
    CBLAS_FUNCTION float dot(const int N, const float* X, const int incX, const float* Y, const int incY)
    {
        return cblas_sdot(N, X, incX, Y, incY);
    }
    CBLAS_FUNCTION double dot(const int N, const double* X, const int incX, const double* Y, const int incY)
    {
        return cblas_ddot(N, X, incX, Y, incY);
    }
    CBLAS_FUNCTION float nrm2(const int N, const float* X, const int incX)
    {
        return cblas_snrm2(N, X, incX);
    }
    CBLAS_FUNCTION float asum(const int N, const float* X, const int incX)
    {
        return cblas_sasum(N, X, incX);
    }
    CBLAS_FUNCTION double nrm2(const int N, const double* X, const int incX)
    {
        return cblas_dnrm2(N, X, incX);
    }
    CBLAS_FUNCTION double asum(const int N, const double* X, const int incX)
    {
        return cblas_dasum(N, X, incX);
    }
    CBLAS_FUNCTION int iamax(const int N, const float* X, const int incX)
    {
        return int(cblas_isamax(N, X, incX));
    }
    CBLAS_FUNCTION int iamax(const int N, const double* X, const int incX)
    {
        return int(cblas_idamax(N, X, incX));
    }
    CBLAS_FUNCTION void swap(const int N, float* X, const int incX, float* Y, const int incY)
    {
        cblas_sswap(N, X, incX, Y, incY);
    }
    CBLAS_FUNCTION void copy(const int N, const float* X, const int incX, float* Y, const int incY)
    {
        cblas_scopy(N, X, incX, Y, incY);
    }
    CBLAS_FUNCTION void axpy(const int N, const float alpha, const float* X, const int incX, float* Y, const int incY)
    {
        cblas_saxpy(N, alpha, X, incX, Y, incY);
    }
    CBLAS_FUNCTION void swap(const int N, double* X, const int incX, double* Y, const int incY)
    {
        cblas_dswap(N, X, incX, Y, incY);
    }
    CBLAS_FUNCTION void copy(const int N, const double* X, const int incX, double* Y, const int incY)
    {
        cblas_dcopy(N, X, incX, Y, incY);
    }
    CBLAS_FUNCTION void axpy(const int N, const double alpha, const double* X, const int incX, double* Y, const int incY)
    {
        cblas_daxpy(N, alpha, X, incX, Y, incY);
    }
    CBLAS_FUNCTION void rotg(float* a, float* b, float* c, float* s)
    {
        cblas_srotg(a, b, c, s);
    }
    CBLAS_FUNCTION void rotmg(float* d1, float* d2, float* b1, const float b2, float* P)
    {
        cblas_srotmg(d1, d2, b1, b2, P);
    }
    CBLAS_FUNCTION void rot(const int N, float* X, const int incX, float* Y, const int incY, const float c, const float s)
    {
        cblas_srot(N, X, incX, Y, incY, c, s);
    }
    CBLAS_FUNCTION void rotm(const int N, float* X, const int incX, float* Y, const int incY, const float* P)
    {
        cblas_srotm(N, X, incX, Y, incY, P);
    }
    CBLAS_FUNCTION void rotg(double* a, double* b, double* c, double* s)
    {
        cblas_drotg(a, b, c, s);
    }
    CBLAS_FUNCTION void rotmg(double* d1, double* d2, double* b1, const double b2, double* P)
    {
        cblas_drotmg(d1, d2, b1, b2, P);
    }
    CBLAS_FUNCTION void rot(const int N, double* X, const int incX, double* Y, const int incY, const double c, const double s)
    {
        cblas_drot(N, X, incX, Y, incY, c, s);
    }
    CBLAS_FUNCTION void rotm(const int N, double* X, const int incX, double* Y, const int incY, const double* P)
    {
        cblas_drotm(N, X, incX, Y, incY, P);
    }
    CBLAS_FUNCTION void scal(const int N, const float alpha, float* X, const int incX)
    {
        cblas_sscal(N, alpha, X, incX);
    }
    CBLAS_FUNCTION void scal(const int N, const double alpha, double* X, const int incX)
    {
        cblas_dscal(N, alpha, X, incX);
    }
    CBLAS_FUNCTION void gemv(const MatrixTransType TransA, const int M, const int N, const float alpha, const float* A, const int lda, const float* X, const int incX, const float beta, float* Y, const int incY)
    {
        cblas_sgemv(CblasColMajor, get_trans(TransA), M, N, alpha, A, lda, X, incX, beta, Y, incY);
    }
    CBLAS_FUNCTION void gbmv(const MatrixTransType TransA, const int M, const int N, const int KL, const int KU, const float alpha, const float* A, const int lda, const float* X, const int incX, const float beta, float* Y, const int incY)
    {
        cblas_sgbmv(CblasColMajor, get_trans(TransA), M, N, KL, KU, alpha, A, lda, X, incX, beta, Y, incY);
    }
    CBLAS_FUNCTION void trmv(const MatrixFillType Uplo, const MatrixTransType TransA, const MatrixDiagType Diag, const int N, const float* A, const int lda, float* X, const int incX)
    {
        cblas_strmv(CblasColMajor, get_uplo(Uplo), get_trans(TransA), get_diag(Diag), N, A, lda, X, incX);
    }
    CBLAS_FUNCTION void tbmv(const MatrixFillType Uplo, const MatrixTransType TransA, const MatrixDiagType Diag, const int N, const int K, const float* A, const int lda, float* X, const int incX)
    {
        cblas_stbmv(CblasColMajor, get_uplo(Uplo), get_trans(TransA), get_diag(Diag), N, K, A, lda, X, incX);
    }
    CBLAS_FUNCTION void tpmv(const MatrixFillType Uplo, const MatrixTransType TransA, const MatrixDiagType Diag, const int N, const float* Ap, float* X, const int incX)
    {
        cblas_stpmv(CblasColMajor, get_uplo(Uplo), get_trans(TransA), get_diag(Diag), N, Ap, X, incX);
    }
    CBLAS_FUNCTION void trsv(const MatrixFillType Uplo, const MatrixTransType TransA, const MatrixDiagType Diag, const int N, const float* A, const int lda, float* X, const int incX)
    {
        cblas_strsv(CblasColMajor, get_uplo(Uplo), get_trans(TransA), get_diag(Diag), N, A, lda, X, incX);
    }
    CBLAS_FUNCTION void tbsv(const MatrixFillType Uplo, const MatrixTransType TransA, const MatrixDiagType Diag, const int N, const int K, const float* A, const int lda, float* X, const int incX)
    {
        cblas_stbsv(CblasColMajor, get_uplo(Uplo), get_trans(TransA), get_diag(Diag), N, K, A, lda, X, incX);
    }
    CBLAS_FUNCTION void tpsv(const MatrixFillType Uplo, const MatrixTransType TransA, const MatrixDiagType Diag, const int N, const float* Ap, float* X, const int incX)
    {
        cblas_stpsv(CblasColMajor, get_uplo(Uplo), get_trans(TransA), get_diag(Diag), N, Ap, X, incX);
    }
    CBLAS_FUNCTION void gemv(const MatrixTransType TransA, const int M, const int N, const double alpha, const double* A, const int lda, const double* X, const int incX, const double beta, double* Y, const int incY)
    {
        cblas_dgemv(CblasColMajor, get_trans(TransA), M, N, alpha, A, lda, X, incX, beta, Y, incY);
    }
    CBLAS_FUNCTION void gbmv(const MatrixTransType TransA, const int M, const int N, const int KL, const int KU, const double alpha, const double* A, const int lda, const double* X, const int incX, const double beta, double* Y, const int incY)
    {
        cblas_dgbmv(CblasColMajor, get_trans(TransA), M, N, KL, KU, alpha, A, lda, X, incX, beta, Y, incY);
    }
    CBLAS_FUNCTION void trmv(const MatrixFillType Uplo, const MatrixTransType TransA, const MatrixDiagType Diag, const int N, const double* A, const int lda, double* X, const int incX)
    {
        cblas_dtrmv(CblasColMajor, get_uplo(Uplo), get_trans(TransA), get_diag(Diag), N, A, lda, X, incX);
    }
    CBLAS_FUNCTION void tbmv(const MatrixFillType Uplo, const MatrixTransType TransA, const MatrixDiagType Diag, const int N, const int K, const double* A, const int lda, double* X, const int incX)
    {
        cblas_dtbmv(CblasColMajor, get_uplo(Uplo), get_trans(TransA), get_diag(Diag), N, K, A, lda, X, incX);
    }
    CBLAS_FUNCTION void tpmv(const MatrixFillType Uplo, const MatrixTransType TransA, const MatrixDiagType Diag, const int N, const double* Ap, double* X, const int incX)
    {
        cblas_dtpmv(CblasColMajor, get_uplo(Uplo), get_trans(TransA), get_diag(Diag), N, Ap, X, incX);
    }
    CBLAS_FUNCTION void trsv(const MatrixFillType Uplo, const MatrixTransType TransA, const MatrixDiagType Diag, const int N, const double* A, const int lda, double* X, const int incX)
    {
        cblas_dtrsv(CblasColMajor, get_uplo(Uplo), get_trans(TransA), get_diag(Diag), N, A, lda, X, incX);
    }
    CBLAS_FUNCTION void tbsv(const MatrixFillType Uplo, const MatrixTransType TransA, const MatrixDiagType Diag, const int N, const int K, const double* A, const int lda, double* X, const int incX)
    {
        cblas_dtbsv(CblasColMajor, get_uplo(Uplo), get_trans(TransA), get_diag(Diag), N, K, A, lda, X, incX);
    }
    CBLAS_FUNCTION void tpsv(const MatrixFillType Uplo, const MatrixTransType TransA, const MatrixDiagType Diag, const int N, const double* Ap, double* X, const int incX)
    {
        cblas_dtpsv(CblasColMajor, get_uplo(Uplo), get_trans(TransA), get_diag(Diag), N, Ap, X, incX);
    }
    CBLAS_FUNCTION void symv(const MatrixFillType Uplo, const int N, const float alpha, const float* A, const int lda, const float* X, const int incX, const float beta, float* Y, const int incY)
    {
        cblas_ssymv(CblasColMajor, get_uplo(Uplo), N, alpha, A, lda, X, incX, beta, Y, incY);
    }
    CBLAS_FUNCTION void sbmv(const MatrixFillType Uplo, const int N, const int K, const float alpha, const float* A, const int lda, const float* X, const int incX, const float beta, float* Y, const int incY)
    {
        cblas_ssbmv(CblasColMajor, get_uplo(Uplo), N, K, alpha, A, lda, X, incX, beta, Y, incY);
    }
    CBLAS_FUNCTION void spmv(const MatrixFillType Uplo, const int N, const float alpha, const float* Ap, const float* X, const int incX, const float beta, float* Y, const int incY)
    {
        cblas_sspmv(CblasColMajor, get_uplo(Uplo), N, alpha, Ap, X, incX, beta, Y, incY);
    }
    CBLAS_FUNCTION void ger(const int M, const int N, const float alpha, const float* X, const int incX, const float* Y, const int incY, float* A, const int lda)
    {
        cblas_sger(CblasColMajor, M, N, alpha, X, incX, Y, incY, A, lda);
    }
    CBLAS_FUNCTION void syr(const MatrixFillType Uplo, const int N, const float alpha, const float* X, const int incX, float* A, const int lda)
    {
        cblas_ssyr(CblasColMajor, get_uplo(Uplo), N, alpha, X, incX, A, lda);
    }
    CBLAS_FUNCTION void spr(const MatrixFillType Uplo, const int N, const float alpha, const float* X, const int incX, float* Ap)
    {
        cblas_sspr(CblasColMajor, get_uplo(Uplo), N, alpha, X, incX, Ap);
    }
    CBLAS_FUNCTION void syr2(const MatrixFillType Uplo, const int N, const float alpha, const float* X, const int incX, const float* Y, const int incY, float* A, const int lda)
    {
        cblas_ssyr2(CblasColMajor, get_uplo(Uplo), N, alpha, X, incX, Y, incY, A, lda);
    }
    CBLAS_FUNCTION void spr2(const MatrixFillType Uplo, const int N, const float alpha, const float* X, const int incX, const float* Y, const int incY, float* A)
    {
        cblas_sspr2(CblasColMajor, get_uplo(Uplo), N, alpha, X, incX, Y, incY, A);
    }
    CBLAS_FUNCTION void symv(const MatrixFillType Uplo, const int N, const double alpha, const double* A, const int lda, const double* X, const int incX, const double beta, double* Y, const int incY)
    {
        cblas_dsymv(CblasColMajor, get_uplo(Uplo), N, alpha, A, lda, X, incX, beta, Y, incY);
    }
    CBLAS_FUNCTION void sbmv(const MatrixFillType Uplo, const int N, const int K, const double alpha, const double* A, const int lda, const double* X, const int incX, const double beta, double* Y, const int incY)
    {
        cblas_dsbmv(CblasColMajor, get_uplo(Uplo), N, K, alpha, A, lda, X, incX, beta, Y, incY);
    }
    CBLAS_FUNCTION void spmv(const MatrixFillType Uplo, const int N, const double alpha, const double* Ap, const double* X, const int incX, const double beta, double* Y, const int incY)
    {
        cblas_dspmv(CblasColMajor, get_uplo(Uplo), N, alpha, Ap, X, incX, beta, Y, incY);
    }
    CBLAS_FUNCTION void ger(const int M, const int N, const double alpha, const double* X, const int incX, const double* Y, const int incY, double* A, const int lda)
    {
        cblas_dger(CblasColMajor, M, N, alpha, X, incX, Y, incY, A, lda);
    }
    CBLAS_FUNCTION void syr(const MatrixFillType Uplo, const int N, const double alpha, const double* X, const int incX, double* A, const int lda)
    {
        cblas_dsyr(CblasColMajor, get_uplo(Uplo), N, alpha, X, incX, A, lda);
    }
    CBLAS_FUNCTION void spr(const MatrixFillType Uplo, const int N, const double alpha, const double* X, const int incX, double* Ap)
    {
        cblas_dspr(CblasColMajor, get_uplo(Uplo), N, alpha, X, incX, Ap);
    }
    CBLAS_FUNCTION void syr2(const MatrixFillType Uplo, const int N, const double alpha, const double* X, const int incX, const double* Y, const int incY, double* A, const int lda)
    {
        cblas_dsyr2(CblasColMajor, get_uplo(Uplo), N, alpha, X, incX, Y, incY, A, lda);
    }
    CBLAS_FUNCTION void spr2(const MatrixFillType Uplo, const int N, const double alpha, const double* X, const int incX, const double* Y, const int incY, double* A)
    {
        cblas_dspr2(CblasColMajor, get_uplo(Uplo), N, alpha, X, incX, Y, incY, A);
    }
    CBLAS_FUNCTION void gemm(const MatrixTransType TransA, const MatrixTransType TransB, const int M, const int N, const int K, const float alpha, const float* A, const int lda, const float* B, const int ldb, const float beta, float* C, const int ldc)
    {
        cblas_sgemm(CblasColMajor, get_trans(TransA), get_trans(TransB), M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
    }
    CBLAS_FUNCTION void symm(const MatrixSideType Side, const MatrixFillType Uplo, const int M, const int N, const float alpha, const float* A, const int lda, const float* B, const int ldb, const float beta, float* C, const int ldc)
    {
        cblas_ssymm(CblasColMajor, get_side(Side), get_uplo(Uplo), M, N, alpha, A, lda, B, ldb, beta, C, ldc);
    }
    CBLAS_FUNCTION void syrk(const MatrixFillType Uplo, const MatrixTransType Trans, const int N, const int K, const float alpha, const float* A, const int lda, const float beta, float* C, const int ldc)
    {
        cblas_ssyrk(CblasColMajor, get_uplo(Uplo), get_trans(Trans), N, K, alpha, A, lda, beta, C, ldc);
    }
    CBLAS_FUNCTION void syr2k(const MatrixFillType Uplo, const MatrixTransType Trans, const int N, const int K, const float alpha, const float* A, const int lda, const float* B, const int ldb, const float beta, float* C, const int ldc)
    {
        cblas_ssyr2k(CblasColMajor, get_uplo(Uplo), get_trans(Trans), N, K, alpha, A, lda, B, ldb, beta, C, ldc);
    }
    CBLAS_FUNCTION void trmm(const MatrixSideType Side, const MatrixFillType Uplo, const MatrixTransType TransA, const MatrixDiagType Diag, const int M, const int N, const float alpha, const float* A, const int lda, float* B, const int ldb)
    {
        cblas_strmm(CblasColMajor, get_side(Side), get_uplo(Uplo), get_trans(TransA), get_diag(Diag), M, N, alpha, A, lda, B, ldb);
    }
    CBLAS_FUNCTION void trsm(const MatrixSideType Side, const MatrixFillType Uplo, const MatrixTransType TransA, const MatrixDiagType Diag, const int M, const int N, const float alpha, const float* A, const int lda, float* B, const int ldb)
    {
        cblas_strsm(CblasColMajor, get_side(Side), get_uplo(Uplo), get_trans(TransA), get_diag(Diag), M, N, alpha, A, lda, B, ldb);
    }
    CBLAS_FUNCTION void gemm(const MatrixTransType TransA, const MatrixTransType TransB, const int M, const int N, const int K, const double alpha, const double* A, const int lda, const double* B, const int ldb, const double beta, double* C, const int ldc)
    {
        cblas_dgemm(CblasColMajor, get_trans(TransA), get_trans(TransB), M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
    }
    CBLAS_FUNCTION void symm(const MatrixSideType Side, const MatrixFillType Uplo, const int M, const int N, const double alpha, const double* A, const int lda, const double* B, const int ldb, const double beta, double* C, const int ldc)
    {
        cblas_dsymm(CblasColMajor, get_side(Side), get_uplo(Uplo), M, N, alpha, A, lda, B, ldb, beta, C, ldc);
    }
    CBLAS_FUNCTION void syrk(const MatrixFillType Uplo, const MatrixTransType Trans, const int N, const int K, const double alpha, const double* A, const int lda, const double beta, double* C, const int ldc)
    {
        cblas_dsyrk(CblasColMajor, get_uplo(Uplo), get_trans(Trans), N, K, alpha, A, lda, beta, C, ldc);
    }
    CBLAS_FUNCTION void syr2k(const MatrixFillType Uplo, const MatrixTransType Trans, const int N, const int K, const double alpha, const double* A, const int lda, const double* B, const int ldb, const double beta, double* C, const int ldc)
    {
        cblas_dsyr2k(CblasColMajor, get_uplo(Uplo), get_trans(Trans), N, K, alpha, A, lda, B, ldb, beta, C, ldc);
    }
    CBLAS_FUNCTION void trmm(const MatrixSideType Side, const MatrixFillType Uplo, const MatrixTransType TransA, const MatrixDiagType Diag, const int M, const int N, const double alpha, const double* A, const int lda, double* B, const int ldb)
    {
        cblas_dtrmm(CblasColMajor, get_side(Side), get_uplo(Uplo), get_trans(TransA), get_diag(Diag), M, N, alpha, A, lda, B, ldb);
    }
    CBLAS_FUNCTION void trsm(const MatrixSideType Side, const MatrixFillType Uplo, const MatrixTransType TransA, const MatrixDiagType Diag, const int M, const int N, const double alpha, const double* A, const int lda, double* B, const int ldb)
    {
        cblas_dtrsm(CblasColMajor, get_side(Side), get_uplo(Uplo), get_trans(TransA), get_diag(Diag), M, N, alpha, A, lda, B, ldb);
    }
};

}    // namespace cccc