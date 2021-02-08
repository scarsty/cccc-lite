#pragma once

namespace cccc
{

//base class of blas
#define NORMAL_BLAS

enum MatrixTransType
{
    MATRIX_NO_TRANS = 0,
    MATRIX_TRANS = 1,
};

enum MatrixFillType
{
    MATRIX_LOWER = 0,
    MATRIX_UPPER = 1
};

enum MatrixDiagType
{
    MATRIX_NON_UNIT = 0,
    MATRIX_UNIT = 1
};

enum MatrixSideType
{
    MATRIX_LEFT = 0,
    MATRIX_RIGHT = 1
};

#ifdef VIRTUAL_BLAS
class Blas
{
public:
    Blas() {}
    virtual ~Blas() {}
    virtual float dot(const int N, const float* X, const int incX, const float* Y, const int incY) { return 0; }
    virtual double dot(const int N, const double* X, const int incX, const double* Y, const int incY) { return 0; }
    virtual float nrm2(const int N, const float* X, const int incX) { return 0; }
    virtual float asum(const int N, const float* X, const int incX) { return 0; }
    virtual double nrm2(const int N, const double* X, const int incX) { return 0; }
    virtual double asum(const int N, const double* X, const int incX) { return 0; }
    virtual int iamax(const int N, const float* X, const int incX) { return 0; }
    virtual int iamax(const int N, const double* X, const int incX) { return 0; }
    virtual void swap(const int N, float* X, const int incX, float* Y, const int incY) {}
    virtual void copy(const int N, const float* X, const int incX, float* Y, const int incY) {}
    virtual void axpy(const int N, const float alpha, const float* X, const int incX, float* Y, const int incY) {}
    virtual void swap(const int N, double* X, const int incX, double* Y, const int incY) {}
    virtual void copy(const int N, const double* X, const int incX, double* Y, const int incY) {}
    virtual void axpy(const int N, const double alpha, const double* X, const int incX, double* Y, const int incY) {}
    virtual void rotg(float* a, float* b, float* c, float* s) {}
    virtual void rotmg(float* d1, float* d2, float* b1, const float b2, float* P) {}
    virtual void rot(const int N, float* X, const int incX, float* Y, const int incY, const float c, const float s) {}
    virtual void rotm(const int N, float* X, const int incX, float* Y, const int incY, const float* P) {}
    virtual void rotg(double* a, double* b, double* c, double* s) {}
    virtual void rotmg(double* d1, double* d2, double* b1, const double b2, double* P) {}
    virtual void rot(const int N, double* X, const int incX, double* Y, const int incY, const double c, const double s) {}
    virtual void rotm(const int N, double* X, const int incX, double* Y, const int incY, const double* P) {}
    virtual void scal(const int N, const float alpha, float* X, const int incX) {}
    virtual void scal(const int N, const double alpha, double* X, const int incX) {}
    virtual void gemv(const MatrixTransType TransA, const int M, const int N, const float alpha, const float* A, const int lda, const float* X, const int incX, const float beta, float* Y, const int incY) {}
    virtual void gbmv(const MatrixTransType TransA, const int M, const int N, const int KL, const int KU, const float alpha, const float* A, const int lda, const float* X, const int incX, const float beta, float* Y, const int incY) {}
    virtual void trmv(const MatrixFillType Uplo, const MatrixTransType TransA, const MatrixDiagType Diag, const int N, const float* A, const int lda, float* X, const int incX) {}
    virtual void tbmv(const MatrixFillType Uplo, const MatrixTransType TransA, const MatrixDiagType Diag, const int N, const int K, const float* A, const int lda, float* X, const int incX) {}
    virtual void tpmv(const MatrixFillType Uplo, const MatrixTransType TransA, const MatrixDiagType Diag, const int N, const float* Ap, float* X, const int incX) {}
    virtual void trsv(const MatrixFillType Uplo, const MatrixTransType TransA, const MatrixDiagType Diag, const int N, const float* A, const int lda, float* X, const int incX) {}
    virtual void tbsv(const MatrixFillType Uplo, const MatrixTransType TransA, const MatrixDiagType Diag, const int N, const int K, const float* A, const int lda, float* X, const int incX) {}
    virtual void tpsv(const MatrixFillType Uplo, const MatrixTransType TransA, const MatrixDiagType Diag, const int N, const float* Ap, float* X, const int incX) {}
    virtual void gemv(const MatrixTransType TransA, const int M, const int N, const double alpha, const double* A, const int lda, const double* X, const int incX, const double beta, double* Y, const int incY) {}
    virtual void gbmv(const MatrixTransType TransA, const int M, const int N, const int KL, const int KU, const double alpha, const double* A, const int lda, const double* X, const int incX, const double beta, double* Y, const int incY) {}
    virtual void trmv(const MatrixFillType Uplo, const MatrixTransType TransA, const MatrixDiagType Diag, const int N, const double* A, const int lda, double* X, const int incX) {}
    virtual void tbmv(const MatrixFillType Uplo, const MatrixTransType TransA, const MatrixDiagType Diag, const int N, const int K, const double* A, const int lda, double* X, const int incX) {}
    virtual void tpmv(const MatrixFillType Uplo, const MatrixTransType TransA, const MatrixDiagType Diag, const int N, const double* Ap, double* X, const int incX) {}
    virtual void trsv(const MatrixFillType Uplo, const MatrixTransType TransA, const MatrixDiagType Diag, const int N, const double* A, const int lda, double* X, const int incX) {}
    virtual void tbsv(const MatrixFillType Uplo, const MatrixTransType TransA, const MatrixDiagType Diag, const int N, const int K, const double* A, const int lda, double* X, const int incX) {}
    virtual void tpsv(const MatrixFillType Uplo, const MatrixTransType TransA, const MatrixDiagType Diag, const int N, const double* Ap, double* X, const int incX) {}
    virtual void symv(const MatrixFillType Uplo, const int N, const float alpha, const float* A, const int lda, const float* X, const int incX, const float beta, float* Y, const int incY) {}
    virtual void sbmv(const MatrixFillType Uplo, const int N, const int K, const float alpha, const float* A, const int lda, const float* X, const int incX, const float beta, float* Y, const int incY) {}
    virtual void spmv(const MatrixFillType Uplo, const int N, const float alpha, const float* Ap, const float* X, const int incX, const float beta, float* Y, const int incY) {}
    virtual void ger(const int M, const int N, const float alpha, const float* X, const int incX, const float* Y, const int incY, float* A, const int lda) {}
    virtual void syr(const MatrixFillType Uplo, const int N, const float alpha, const float* X, const int incX, float* A, const int lda) {}
    virtual void spr(const MatrixFillType Uplo, const int N, const float alpha, const float* X, const int incX, float* Ap) {}
    virtual void syr2(const MatrixFillType Uplo, const int N, const float alpha, const float* X, const int incX, const float* Y, const int incY, float* A, const int lda) {}
    virtual void spr2(const MatrixFillType Uplo, const int N, const float alpha, const float* X, const int incX, const float* Y, const int incY, float* A) {}
    virtual void symv(const MatrixFillType Uplo, const int N, const double alpha, const double* A, const int lda, const double* X, const int incX, const double beta, double* Y, const int incY) {}
    virtual void sbmv(const MatrixFillType Uplo, const int N, const int K, const double alpha, const double* A, const int lda, const double* X, const int incX, const double beta, double* Y, const int incY) {}
    virtual void spmv(const MatrixFillType Uplo, const int N, const double alpha, const double* Ap, const double* X, const int incX, const double beta, double* Y, const int incY) {}
    virtual void ger(const int M, const int N, const double alpha, const double* X, const int incX, const double* Y, const int incY, double* A, const int lda) {}
    virtual void syr(const MatrixFillType Uplo, const int N, const double alpha, const double* X, const int incX, double* A, const int lda) {}
    virtual void spr(const MatrixFillType Uplo, const int N, const double alpha, const double* X, const int incX, double* Ap) {}
    virtual void syr2(const MatrixFillType Uplo, const int N, const double alpha, const double* X, const int incX, const double* Y, const int incY, double* A, const int lda) {}
    virtual void spr2(const MatrixFillType Uplo, const int N, const double alpha, const double* X, const int incX, const double* Y, const int incY, double* A) {}
    virtual void gemm(const MatrixTransType TransA, const MatrixTransType TransB, const int M, const int N, const int K, const float alpha, const float* A, const int lda, const float* B, const int ldb, const float beta, float* C, const int ldc) {}
    virtual void symm(const MatrixSideType Side, const MatrixFillType Uplo, const int M, const int N, const float alpha, const float* A, const int lda, const float* B, const int ldb, const float beta, float* C, const int ldc) {}
    virtual void syrk(const MatrixFillType Uplo, const MatrixTransType Trans, const int N, const int K, const float alpha, const float* A, const int lda, const float beta, float* C, const int ldc) {}
    virtual void syr2k(const MatrixFillType Uplo, const MatrixTransType Trans, const int N, const int K, const float alpha, const float* A, const int lda, const float* B, const int ldb, const float beta, float* C, const int ldc) {}
    virtual void trmm(const MatrixSideType Side, const MatrixFillType Uplo, const MatrixTransType TransA, const MatrixDiagType Diag, const int M, const int N, const float alpha, const float* A, const int lda, float* B, const int ldb) {}
    virtual void trsm(const MatrixSideType Side, const MatrixFillType Uplo, const MatrixTransType TransA, const MatrixDiagType Diag, const int M, const int N, const float alpha, const float* A, const int lda, float* B, const int ldb) {}
    virtual void gemm(const MatrixTransType TransA, const MatrixTransType TransB, const int M, const int N, const int K, const double alpha, const double* A, const int lda, const double* B, const int ldb, const double beta, double* C, const int ldc) {}
    virtual void symm(const MatrixSideType Side, const MatrixFillType Uplo, const int M, const int N, const double alpha, const double* A, const int lda, const double* B, const int ldb, const double beta, double* C, const int ldc) {}
    virtual void syrk(const MatrixFillType Uplo, const MatrixTransType Trans, const int N, const int K, const double alpha, const double* A, const int lda, const double beta, double* C, const int ldc) {}
    virtual void syr2k(const MatrixFillType Uplo, const MatrixTransType Trans, const int N, const int K, const double alpha, const double* A, const int lda, const double* B, const int ldb, const double beta, double* C, const int ldc) {}
    virtual void trmm(const MatrixSideType Side, const MatrixFillType Uplo, const MatrixTransType TransA, const MatrixDiagType Diag, const int M, const int N, const double alpha, const double* A, const int lda, double* B, const int ldb) {}
    virtual void trsm(const MatrixSideType Side, const MatrixFillType Uplo, const MatrixTransType TransA, const MatrixDiagType Diag, const int M, const int N, const double alpha, const double* A, const int lda, double* B, const int ldb) {}
};
#endif
#ifdef STATIC_BLAS
class Blas
{
};
#endif
#ifdef NORMAL_BLAS
class Blas
{
};
#endif

}    // namespace cccc