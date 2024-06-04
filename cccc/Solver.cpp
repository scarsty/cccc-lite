#include "Solver.h"
#include "MatrixEx.h"
#include "math_supplement.h"

namespace cccc
{

Solver::Solver()
{
}

Solver::~Solver()
{
    destory();
}

float Solver::getMomentum() const
{
    //if (momentum_clear_epoch_ > 0 && epoch % momentum_clear_epoch_ == 0)
    //{
    //    return 0;
    //}
    return momentum_;
}

//求解器，即如何更新参数的定义
void Solver::init(Option* option, std::string section, Matrix& W)
{
    dW_.resize(W.getDim());
    dW_.fillData(0);
    //batch_ = batch;
    OPTION_GET_REAL(weight_decay_);
    OPTION_GET_REAL(weight_decay_l1_);

    restrict_dweight_ = option->getVector<float>(section, "restrict_dweight", ",", restrict_dweight_);
    restrict_dweight_.resize(2);

    if (weight_decay_l1_ != 0)
    {
        W_sign_.resize(W.getDim());
    }

    //求解器设定
    solver_type_ = option->getEnum(section, "solver", SOLVER_SGD);
    //LOG("Solver type is {}\n", op->getStringFromEnum(solver_type_));

    OPTION_GET_REAL(momentum_);
    OPTION_GET_INT(momentum_clear_epoch_);

    OPTION_GET_REAL(switch_sgd_epoch_);
    OPTION_GET_REAL(switch_sgd_random_);

    destory();

    switch (solver_type_)
    {
    case SOLVER_SGD:
        break;
    case SOLVER_NAG:
        momentum_ = 0;
        real_vector_.resize(1);
        real_vector_[0] = option->getReal(section, "momentum", 0.9);
        W_vector_.push_back(W.clone());
        break;
    case SOLVER_ADA_DELTA:
    case SOLVER_ADAM:
    case SOLVER_RMS_PROP:
        real_vector_.resize(4);
        real_vector_[0] = option->getReal(section, "ada_epsilon", 1e-6);
        real_vector_[1] = option->getReal(section, "ada_rou", 0.95);
        real_vector_[2] = option->getReal(section, "ada_beta1", 0.95);
        real_vector_[3] = option->getReal(section, "ada_beta2", 0.95);
        if (solver_type_ == SOLVER_RMS_PROP)
        {
            W_vector_.resize(1);
        }
        else
        {
            W_vector_.resize(2);
        }
        for (auto& m : W_vector_)
        {
            m.resize(W.getDim());
            m.fillData(0);
        }
        break;
    }

    //学习率相关参数
    int train_epochs = 1;
    OPTION_GET_INT(train_epochs);
    lr_adjust_method_ = option->getEnum(section, "lr_adjust_method", ADJUST_LEARN_RATE_FIXED);

    OPTION_GET_REAL(learn_rate_base_);
    learn_rate_ = learn_rate_base_;
    OPTION_GET_REAL(lr_weight_scale_);
    OPTION_GET_REAL(lr_bias_scale_);
    OPTION_GET_INT(lr_steps_);
    OPTION_GET_REAL(lr_step_decay_);
    OPTION_GET_INT(lr_test_stable_time_);
    OPTION_GET_REAL(lr_test_std_limit_);

    auto lr_inter_set = option->getStringVector(section, "lr_inter_set");
    for (auto& set_str : lr_inter_set)
    {
        auto sets = strfunc::splitString(set_str, ": ");
        if (sets.size() == 2)
        {
            auto e = atof(sets[0].c_str());
            if (e < 1)
            {
                e = e * train_epochs;
            }
            lr_inter_epoch_.push_back(int(e));
            lr_inter_rate_.push_back(atof(sets[1].c_str()));
        }
    }
    if (!lr_inter_epoch_.empty() && lr_inter_epoch_.back() == 1)
    {
        lr_inter_epoch_.back() = train_epochs;
    }
}

//以下求解器相关
//调整学习率
//返回值暂时无特殊定义
int Solver::adjustLearnRate(int epoch, int total_epoch)
{
    auto pre_lr = learn_rate_;
    auto method = lr_adjust_method_;
    if (epoch == switch_sgd_epoch_ && solver_type_ != SOLVER_SGD)
    {
        switch_solver_ = 1;
        solver_type_ = SOLVER_SGD;
        //if (real_vector_.size() >= 4)
        //{
        //    learn_rate_base_ = learn_rate_base_ / (1 - pow(real_vector_[3], epoch));
        //}
        //learn_rate_base_ *= 10;
    }
    switch (method)
    {
    case ADJUST_LEARN_RATE_FIXED:
        learn_rate_ = learn_rate_base_;
        break;
    case ADJUST_LEARN_RATE_SCALE_INTER:
    {
        float rate = lr_inter_rate_.back();
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
        float rate = lr_inter_rate_.back();
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
    case ADJUST_LEARN_RATE_STEPS:
    {
        learn_rate_ = learn_rate_base_ * pow(lr_step_decay_, (epoch - 1) / ((total_epoch + lr_steps_ - 1) / lr_steps_));
        break;
    }
    case ADJUST_LEARN_RATE_STEPS_WARM:
    {
        if (epoch <= total_epoch / 5)
        {
            learn_rate_ = learn_rate_base_ * pow(lr_step_decay_, -(epoch - 1) / ((total_epoch / 5 + lr_steps_ - 1) / lr_steps_) + lr_steps_ - 1);
        }
        else
        {
            learn_rate_ = learn_rate_base_ * pow(lr_step_decay_, (epoch - 1 - total_epoch / 5) / ((total_epoch / 5 * 4 + lr_steps_ - 1) / lr_steps_));
        }
        break;
    }
    default:
        break;
    }
    if (pre_lr != learn_rate_)
    {
        lr_keep_count_ = 0;
        LOG("Learn rate is changed from {} to {}\n", pre_lr, learn_rate_);
    }
    return 0;
}

int Solver::adjustLearnRate2(int epoch, int total_epoch, const std::vector<std::vector<TestInfo>>& test_info)
{
    return 0;
}

void Solver::updateWeightBiasPre(Matrix& W)
{
    //if (W)
    {
        switch (solver_type_)
        {
        case SOLVER_NAG:
            //依据上一次的参数直接跳一步
            Matrix::add(W, W_vector_[0], W, 1 + real_vector_[0], -real_vector_[0]);
            break;
        }
    }
}

//求解器本身（可以独立并行）
void Solver::updateWeights(Matrix& W, int batch)
{
    time_step_ = time_step_ + 1;

    if (restrict_dweight_[0] != 0 || restrict_dweight_[1] != 0)
    {
        auto& dW = W.d();
        auto l1w = W.sumAbs();
        auto l1dw = dW.sumAbs() / batch;
        //LOG("a {}\n", l1dw / l1w);
        if (restrict_dweight_[0] != 0 && l1dw < l1w * restrict_dweight_[0])
        {
            dW.scale(l1w * restrict_dweight_[0] / l1dw);
            restrict_dweight_count_[0]++;
        }
        else if (restrict_dweight_[1] != 0 && l1dw > l1w * restrict_dweight_[1])
        {
            dW.scale(l1w * restrict_dweight_[1] / l1dw);
            restrict_dweight_count_[1]++;
        }

        //LOG("b {}\n", dW.sumAbs() / batch / l1w);
    }

    switch (solver_type_)
    {
    case SOLVER_SGD:
    case SOLVER_NAG:
        Matrix::add(W.d(), dW_, dW_, 1, momentum_, 0);    //动量
        Matrix::add(W, dW_, W, 1 - weight_decay_ * learn_rate_, -learn_rate_ * lr_weight_scale_ / batch);
        //LOG("check dweight = {}, {}, {}\n", W.sumAbs(), dW.sumAbs() / batch, dW.sumAbs() / W.sumAbs() / batch);
        if (weight_decay_l1_ != 0)
        {
            Matrix::sign(W, W_sign_, weight_decay_l1_ * learn_rate_);
            Matrix::add(W, W_sign_, W, 1, -1);
        }
        if (solver_type_ == SOLVER_NAG)
        {
            Matrix::copyData(W, W_vector_[0]);    //what's this?
        }
        break;
    case SOLVER_ADA_DELTA:
        MatrixEx::adaDeltaUpdate(W_vector_[0], W_vector_[1], W.d(), dW_, real_vector_[1], real_vector_[0]);
        Matrix::add(W, dW_, W, 1 - weight_decay_ * learn_rate_, -1.0);
        break;
    case SOLVER_ADAM:
        MatrixEx::adamUpdate(W_vector_[0], W_vector_[1], W.d(), dW_, real_vector_[2], real_vector_[3], real_vector_[0], time_step_);
        Matrix::add(W, dW_, W, 1 - weight_decay_ * learn_rate_, -learn_rate_ * lr_weight_scale_);
        break;
    case SOLVER_RMS_PROP:
        MatrixEx::adaRMSPropUpdate(W_vector_[0], W.d(), dW_, real_vector_[1], real_vector_[0]);
        Matrix::add(W, dW_, W, 1, -learn_rate_ * lr_weight_scale_);
        break;
    }
}

void Solver::reset()
{
    dW_.fillData(0);
    for (auto& W : W_vector_)
    {
        W.fillData(0);
    }
}

void Solver::cancelRestrictDWeight()
{
    restrict_dweight_ = { 0, 0 };
}

void Solver::outputState() const
{
    if (restrict_dweight_count_[0] + restrict_dweight_count_[1] != 0)
    {
        LOG("Restrict gradient count is {}\n", restrict_dweight_count_);
    }
}

void Solver::destory()
{
}

//在数组中查找x所在的区间，返回区间的左边界索引
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

}    // namespace cccc