#include "AdditionalCost.h"
#include "MatrixEx.h"

namespace cccc
{

AdditionalCost::AdditionalCost()
{
}

AdditionalCost::~AdditionalCost()
{
    destory();
}

void AdditionalCost::init(Option* op, std::string section, Matrix& A)
{
    batch_ = A.getNumber();

    sparse_beta_ = op->getReal2(section, "sparse_beta", 0);
    if (sparse_beta_ != 0)
    {
        sparse_rou_ = op->getReal2(section, "sparse_rou", 0.1);
        sparse_rou_hat_.resize(A);
        sparse_rou_hat_vector_.resize(A.getRow(), 1);
        as_sparse_.resize(batch_, 1);
        as_sparse_.initData(1);
    }

    diverse_beta_ = op->getReal2(section, "diverse_beta", 0);
    if (diverse_beta_ != 0)
    {
        diverse_epsilon_ = op->getReal2(section, "diverse_epsilon", 1e-8);
        diverse_aver_.resize(A);
        diverse_aver2_.resize(A);
        diverse_aver3_.resize(A);
        as_diverse_aver_.resize(1, 1, A.getChannel() * A.getNumber(), 1);
        as_diverse_aver_.initData(1);
        diverse_A_ = A.cloneShared();
        diverse_A_.resize(1, 1, A.getChannel() * A.getNumber(), 1);
        diverse_workspace_.resize(1, int(1e6));
    }
}

void AdditionalCost::destory()
{
}

//the extra-cost function often modifies dA with A
void AdditionalCost::modifyDA(Matrix& A, Matrix& dA)
{
    //sparse
    if (sparse_beta_ != 0)
    {
        Matrix::mulVector(A, as_sparse_, sparse_rou_hat_vector_, 1.0 / batch_);
        MatrixEx::sparse(sparse_rou_hat_, sparse_rou_hat_, sparse_rou_, sparse_beta_);
        Matrix::copyData(sparse_rou_hat_vector_, sparse_rou_hat_);
        sparse_rou_hat_.repeat();
        Matrix::add(dA, sparse_rou_hat_, dA);
    }
    //diverse, dropped
    if (false && diverse_beta_ != 0)
    {
        //use convolution to calculate average cross channel and number
        //int cn = A->getChannel() * A->getNumber();
        //diverse_aver_->resize(A->getWidth(), A->getHeight(), 1, 1);
        //A->resize(A->getWidth(), A->getHeight(), cn, 1);
        //MatrixExtend::convolutionForward(A, diverse_aver_, as_diverse_aver_, diverse_workspace_, diverse_workspace2_, 1, 1, 1.0 / cn);
        //A->resize(dA);
        //diverse_aver_->resize(A->getWidth(), A->getHeight(), 1, cn);
        //diverse_aver_->repeat();

        //Matrix::add(A, diverse_aver_, diverse_aver_, 1, -1);

        //Matrix::add(dA, diverse_aver_, dA, 1, -diverse_beta_);

        /*diverse_aver_->resize(A->getWidth(), A->getHeight(), 1, 1);
        A->resize(A->getWidth(), A->getHeight(), cn, 1);
        MatrixExtend::convolutionForward(A, diverse_aver_, as_diverse_aver_, diverse_workspace_, diverse_workspace2_, 1, 1, 1.0 / cn);
        A->resize(dA);
        diverse_aver_->resize(A->getWidth(), A->getHeight(), 1, cn);
        diverse_aver_->repeat();
        Matrix::elementPow(diverse_aver_, diverse_aver2_, 2);
        Matrix::elementPow(diverse_aver_, diverse_aver3_, 3);
        Matrix::elementPow(A, diverse_aver_, 2);
        Matrix::add(diverse_aver_, diverse_aver2_, diverse_aver_, 1, -1);
        Matrix::elementDiv(diverse_aver_, diverse_aver3_, diverse_aver_, 0, diverse_epsilon_);
        Matrix::add(dA, diverse_aver_, dA);*/
    }
}

}    // namespace cccc