#include "Activer.h"

namespace cccc
{

Activer::Activer()
{
}

Activer::~Activer()
{
}

void Activer::init(Option* op, std::string section, LayerConnectionType ct)
{
    active_function_ = op->getEnum(section, "active", ACTIVE_FUNCTION_NONE);
    cost_function_ = op->getEnum2(section, "cost", COST_FUNCTION_CROSS_ENTROPY);    //cost似乎是应该属于net的，待查
    cost_rate_ = op->getVector<real>(section, "cost_rate");

    real learn_rate_base = op->getReal2(section, "learn_rate_base", 1e-2);
    switch (active_function_)
    {
    case ACTIVE_FUNCTION_CLIPPED_RELU:
        real_vector_ = { op->getReal2(section, "clipped_relu", 0.5) };
        break;
    default:
        break;
    }
}

//激活前的准备工作，主要用于一些激活函数在训练和测试时有不同设置
void Activer::activePrepare(ActivePhaseType ap)
{
    switch (active_function_)
    {
    default:
        break;
    }
}

void Activer::initBuffer(Matrix& X)
{
    MatrixEx::activeBufferInit(X, active_function_, int_vector_, matrix_vector_);
    if (!cost_rate_.empty())
    {
        matrix_cost_rate_ = Matrix(X.getDim(), DeviceType::CPU);
        matrix_cost_rate_.initData(1);
        for (int ir = 0; ir < X.getRow(); ir++)
        {
            for (int in = 0; in < X.getNumber(); in++)
            {
                if (ir < cost_rate_.size())
                {
                    matrix_cost_rate_.getData(ir, in) = cost_rate_[ir];
                }
            }
        }
        matrix_cost_rate_.toGPU();
    }
}

void Activer::backwardCost(Matrix& A, Matrix& X, Matrix& Y, Matrix& dA, Matrix& dX)
{
    if (cost_function_ == COST_FUNCTION_RMSE)
    {
        Matrix::add(A, Y, dA, 1, -1);
        backward(A, X);
    }
    else
    {
        //交叉熵的推导结果就是直接相减，注意此处并未更新DA
        Matrix::add(A, Y, dX, 1, -1);
    }
    if (matrix_cost_rate_.getDataSize() > 0)
    {
        Matrix::elementMul(dX, matrix_cost_rate_, dX);
    }
}

//dY will be changed
//实际上反向传播并未直接使用，额外消耗计算资源，不要多用
real Activer::calCostValue(Matrix& A, Matrix& dA, Matrix& Y, Matrix& dY)
{
    //todo::seem not right
    if (dY.getDataSize() == 0)
    {
        return 0;
    }
    if (cost_function_ == COST_FUNCTION_RMSE)
    {
        Matrix::add(A, Y, dY, 1, -1);
        return dY.dotSelf();
    }
    else
    {
        switch (active_function_)
        {
        case ACTIVE_FUNCTION_SIGMOID:
            Matrix::crossEntropy2(A, Y, dY, 1e-5);
            break;
        case ACTIVE_FUNCTION_SOFTMAX:
        case ACTIVE_FUNCTION_SOFTMAX_FAST:
            Matrix::crossEntropy(A, Y, dY, 1e-5);
            break;
        default:
            Matrix::add(A, Y, dY, 1, -1);
            break;
        }
        return dY.sumAbs();
    }
}

bool Activer::needSave(int i)
{

}

}    // namespace cccc