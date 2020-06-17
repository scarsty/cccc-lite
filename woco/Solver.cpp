#include "Solver.h"
#include "MatrixExtend.h"
#include "Option.h"

namespace woco
{

//求解器即如何更新参数的定义
void Solver::setWeight(const Matrix& W)
{
    W_ = W;
    W0_ = W;
    DW_.resize(W.getDim());
    DW_.initData(0);

    //求解器设定
    auto& op = Option::getInstance();
    solver_type_ = op.getEnum("solver", "solver", SOLVER_SGD);
    momentum_ = op.getReal("solver", "momentum", 0.9);
    active_ = op.getEnum("solver", "active", ACTIVE_FUNCTION_NONE);

    if (active_ != ACTIVE_FUNCTION_NONE)
    {
        W0_ = W_.clone();
    }

    switch (solver_type_)
    {
    case SOLVER_SGD:

        break;
    case SOLVER_NAG:
        momentum_ = 0;
        real_vector_.resize(1);
        real_vector_[0] = op.getReal("", "momentum", 0.9);
        W_vector_.push_back(W.clone());
        break;
    case SOLVER_ADA_DELTA:
    case SOLVER_ADAM:
    case SOLVER_RMS_PROP:
        real_vector_.resize(4);
        real_vector_[0] = op.getReal("solver", "ada_epsilon", 1e-6);
        real_vector_[1] = op.getReal("solver", "ada_rou", 0.95);
        real_vector_[2] = op.getReal("solver", "ada_beta1", 0.95);
        real_vector_[3] = op.getReal("solver", "ada_beta2", 0.95);
        if (solver_type_ == SOLVER_RMS_PROP)
        {
            W_vector_.resize(2);
        }
        else
        {
            W_vector_.resize(3);
        }
        break;
    }

    //学习率相关参数
    lr_adjust_method_ = op.getEnum("solver", "lr_adjust_method", ADJUST_LEARN_RATE_FIXED);

    learn_rate_base_ = op.getReal("solver", "learn_rate_base", 1e-2);
    learn_rate_ = learn_rate_base_;

    lr_weight_scale_ = op.getReal("solver", "lr_weight_scale", 1);
    lr_bias_scale_ = op.getReal("solver", "lr_bias_scale", 2);
    lr_step_ = (std::max)(1, op.getInt("solver", "lr_step", 1));

    auto lr_inter_set = convert::splitString(op.getString("solver", "lr_inter_set", ""), ",");
    for (auto& set_str : lr_inter_set)
    {
        auto sets = convert::splitString(set_str, ": ");
        if (sets.size() == 2)
        {
            lr_inter_epoch_.push_back(atoi(sets[0].c_str()));
            lr_inter_rate_.push_back(atof(sets[1].c_str()));
        }
    }
}

//以下求解器相关
//调整学习率
real Solver::adjustLearnRate(int epoch)
{
    auto method = lr_adjust_method_;
    switch (method)
    {
    case ADJUST_LEARN_RATE_FIXED:
        learn_rate_ = learn_rate_base_;
        break;
    case ADJUST_LEARN_RATE_SCALE_INTER:
    {
        real rate = lr_inter_rate_.back();
        int i = findInter(epoch, lr_inter_epoch_);
        if (i < 0)
        {
            rate = lr_inter_rate_.front();
        }
        else if (i >= 0 && i < lr_inter_epoch_.size() - 1)
        {
            rate = scale_inter(epoch, lr_inter_epoch_[i], lr_inter_epoch_[i + 1], lr_inter_rate_[i], lr_inter_rate_[i + 1]);
        }
        learn_rate_ = learn_rate_base_ * rate;
        break;
    }
    case ADJUST_LEARN_RATE_LINEAR_INTER:
    {
        real rate = lr_inter_rate_.back();
        int i = findInter(epoch, lr_inter_epoch_);
        if (i < 0)
        {
            rate = lr_inter_rate_.front();
        }
        else if (i >= 0 && i < lr_inter_epoch_.size() - 1)
        {
            rate = linear_inter(epoch, lr_inter_epoch_[i], lr_inter_epoch_[i + 1], lr_inter_rate_[i], lr_inter_rate_[i + 1]);
        }
        learn_rate_ = learn_rate_base_ * rate;
        break;
    }
    default:
        break;
    }
    return learn_rate_;
}

void Solver::updateWeightPre()
{
    switch (solver_type_)
    {
    case SOLVER_NAG:
        //依据上一次的参数直接跳一步
        Matrix::add(W_, W_vector_[0], W_, 1 + real_vector_[0], -real_vector_[0]);
        break;
    }
}

//求解器本身
void Solver::updateWeight(int batch)
{
    time_step_ = time_step_ + 1;
    switch (solver_type_)
    {
    case SOLVER_SGD:
    case SOLVER_NAG:
        //DMatrix是包含正则化之后的
        Matrix::add(DW_, W_.DMatrix(), DW_, momentum_, 1);
        Matrix::add(W0_, DW_, W0_, 1, -learn_rate_ / batch);
        if (solver_type_ == SOLVER_NAG)
        {
            Matrix::copyData(W_, W_vector_[0]);
        }
        break;
    }
    if (active_ != ACTIVE_FUNCTION_NONE)
    {
        MatrixExtend::activeForward(W0_, W_, active_);
        MatrixExtend::step(W_, W_);    //测试
    }
}

int Solver::findInter(int x, std::vector<int>& xv)
{
    if (x < xv[0])
    {
        return -1;
    }
    for (int i = 0; i < int(xv.size()) - 1; i++)
    {
        if (x >= xv[i] && x < xv[i + 1])
        {
            return i;
        }
    }
    return xv.size() - 1;
}

real Solver::linear_inter(int x, int x1, int x2, real y1, real y2)
{
    real y = y1 + (y2 - y1) / (x2 - x1) * (x - x1);
    return y;
}

real Solver::scale_inter(int x, int x1, int x2, real y1, real y2)
{
    real p = pow(y2 / y1, 1.0 * (x - x1) / (x2 - x1));
    real y = y1 * p;
    return y;
}

}    // namespace woco