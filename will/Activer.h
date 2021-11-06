#pragma once
#include "Matrix.h"
#include "MatrixEx.h"
#include "Neural.h"
#include "Option.h"
#include "Random.h"

namespace cccc
{

class Activer : public Neural
{
public:
    Activer();
    virtual ~Activer();

protected:
    //激活函数相关
    ActiveFunctionType active_function_ = ACTIVE_FUNCTION_NONE;
    //bool need_active_ex_ = false;
    std::vector<real> real_vector_;
    std::vector<int> int_vector_;
    std::vector<Matrix> matrix_vector_;
    //代价函数
    CostFunctionType cost_function_ = COST_FUNCTION_CROSS_ENTROPY;
    std::vector<real> cost_rate_;    //按照channel放大某些类别在损失函数中的比例，可以认为是强调某些类别的正确性
    Matrix matrix_cost_rate_;

    Random<real> random_generator_;
    ActivePhaseType active_phase_ = ACTIVE_PHASE_TRAIN;

public:
    void forward(Matrix& X, Matrix& A)
    {
        activePrepare(active_phase_);
        MatrixEx::activeForward(X, A, active_function_, int_vector_, real_vector_, matrix_vector_);
    }

    void backward(Matrix& A, Matrix& X)
    {
        MatrixEx::activeBackward(X, A, active_function_, int_vector_, real_vector_, matrix_vector_);
    }

    bool isNone() { return active_function_ == ACTIVE_FUNCTION_NONE; }
    void init(Option* op, std::string section, LayerConnectionType ct);
    void activePrepare(ActivePhaseType ap);
    void initBuffer(Matrix& X);
    void setCostFunction(CostFunctionType cf) { cost_function_ = cf; }

    void backwardCost(Matrix& A, Matrix& X, Matrix& Y, Matrix& dA, Matrix& dX);
    real calCostValue(Matrix& A, Matrix& dA, Matrix& Y, Matrix& dY);

    ActiveFunctionType getActiveFunction() { return active_function_; }
    void setActivePhase(ActivePhaseType ap) { active_phase_ = ap; }

    CostFunctionType getCostFunction_() { return cost_function_; }

private:
    bool needSave(int i);
};

}    // namespace cccc