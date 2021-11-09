#pragma once
#include "Matrix.h"
#include "Neural.h"
#include "Option.h"

namespace cccc
{

class AdditionalCost : public Neural
{
public:
    AdditionalCost();
    virtual ~AdditionalCost();

private:
    real sparse_beta_ = 0;
    real sparse_rou_ = 0.1;
    Matrix sparse_rou_hat_;
    Matrix sparse_rou_hat_vector_;
    Matrix as_sparse_;
    int batch_ = 0;

    real diverse_beta_ = 0;
    real diverse_epsilon_;
    Matrix diverse_aver_;
    Matrix diverse_aver2_;
    Matrix diverse_aver3_;
    Matrix as_diverse_aver_;
    Matrix diverse_workspace_;
    Matrix diverse_A_;
    std::vector<int> diverse_workspace2_;

public:
    void init(Option* op, std::string section, Matrix& A);

private:
    void destory();

public:
    void modifyDA(Matrix& A, Matrix& dA);
};

}    // namespace cccc