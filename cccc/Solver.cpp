#include "Solver.h"
#include "MatrixEx.h"

namespace cccc
{

Solver::Solver()
{
}

Solver::~Solver()
{
    destory();
}

real Solver::getMomentum()
{
    //if (momentum_clear_epoch_ > 0 && epoch % momentum_clear_epoch_ == 0)
    //{
    //    return 0;
    //}
    return momentum_;
}

//求解器，即如何更新参数的定义
void Solver::init(Option* op, std::string section, Matrix& W)
{
    //batch_ = batch;
    W_ = &W;
    weight_decay_ = op->getReal2(section, "weight_decay", 0);
    weight_decay_l1_ = op->getReal2(section, "weight_decay_l1", 0);
    auto_weight_decay_ = op->getReal2(section, "auto_weight_decay", 0);

    normalized_dweight_ = op->getInt2(section, "normalized_dweight", 0);

    if (weight_decay_l1_ != 0)
    {
        W_sign_.resize(W);
    }

    if (auto_weight_decay_ == 1)
    {
        weight_decay_ = 0;
    }

    //求解器设定
    solver_type_ = op->getEnum2(section, "solver", SOLVER_SGD);

    LOG("Solver type is {}\n", op->getStringFromEnum(solver_type_));

    momentum_ = op->getReal2(section, "momentum", 0.9);
    momentum_clear_epoch_ = op->getInt2(section, "momentum_clear_epoch", -1);

    switch_sgd_epoch_ = op->getInt2(section, "switch_sgd_epoch", -1);
    switch_sgd_random_ = op->getReal2(section, "switch_sgd_random", 0);

    destory();

    switch (solver_type_)
    {
    case SOLVER_SGD:

        break;
    case SOLVER_NAG:
        momentum_ = 0;
        real_vector_.resize(1);
        real_vector_[0] = op->getReal2(section, "momentum", 0.9);
        W_vector_.push_back(W.clone());
        break;
    case SOLVER_ADA_DELTA:
    case SOLVER_ADAM:
    case SOLVER_RMS_PROP:
        real_vector_.resize(4);
        real_vector_[0] = op->getReal2(section, "ada_epsilon", 1e-6);
        real_vector_[1] = op->getReal2(section, "ada_rou", 0.95);
        real_vector_[2] = op->getReal2(section, "ada_beta1", 0.95);
        real_vector_[3] = op->getReal2(section, "ada_beta2", 0.95);
        if (solver_type_ == SOLVER_RMS_PROP)
        {
            W_vector_.resize(2);
        }
        else
        {
            W_vector_.resize(3);
        }
        for (auto& m : W_vector_)
        {
            m.resize(W);
            m.initData(0);
        }
        break;
    }

    //学习率相关参数
    lr_adjust_method_ = op->getEnum2(section, "lr_adjust_method", ADJUST_LEARN_RATE_FIXED);

    learn_rate_base_ = op->getReal2(section, "learn_rate_base", 1e-2);
    learn_rate_ = learn_rate_base_;

    lr_weight_scale_ = op->getReal2(section, "lr_weight_scale", 1);
    lr_bias_scale_ = op->getReal2(section, "lr_bias_scale", 2);
    lr_step_ = (std::max)(1, op->getInt2(section, "lr_step", 1));

    auto lr_inter_set = strfunc::splitString(op->getString2(section, "lr_inter_set", ""), ",");
    for (auto& set_str : lr_inter_set)
    {
        auto sets = strfunc::splitString(set_str, ": ");
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

void Solver::updateWeightBiasPre()
{
    //if (W)
    {
        switch (solver_type_)
        {
        case SOLVER_NAG:
            //依据上一次的参数直接跳一步
            Matrix::add(*W_, W_vector_[0], *W_, 1 + real_vector_[0], -real_vector_[0]);
            break;
        }
    }
}

//求解器本身（可以独立并行）
void Solver::updateWeights(int batch)
{
    time_step_ = time_step_ + 1;
    //if (W)
    //{    W->message();}
    if (switch_solver_)
    {
        //switch_solver_ = 0;
        //LOG("Switch to SGD!\n");
        ////learn_rate_ = 0.01;
        //real scale = 0;
        //if (W)
        //{
        //    auto W1 = W->clone(DATA_SELF, DeviceType::CPU);
        //    //real min1 = REAL_MAX, max1 = -REAL_MAX;
        //    //for (int i = 0; i < W1->getDataSize(); i++)
        //    //{
        //    //    auto v = W1->getData(i);
        //    //    min1 = std::min(min1, v);
        //    //    max1 = std::max(max1, v);
        //    //}
        //    real a = sqrt(6.0 / W->getWidth() / W->getHeight() / (W->getChannel() + W->getNumber()));
        //    //W->addNumber(-(max1 + min1) / 2);
        //    //scale = a * 2 / (max1 - min1);
        //    scale = sqrt(a * a / 3 / (W->dotSelf() / W->getDataSize()));
        //    W1->scale(scale);
        //    //W->message("W");
        //    //W1->message("W1");
        //    auto W2 = new Matrix(W1, DATA_SELF, DeviceType::CPU);
        //    MatrixFiller::fill(W2, RANDOM_FILL_XAVIER, W->getWidth() * W->getHeight() * W->getChannel(), W->getWidth() * W->getHeight() * W->getNumber());
        //    W2->scale(switch_sgd_random_);
        //    Matrix::add(W1, W2, W1);
        //    Matrix::copyData(W1, W);
        //    delete W1;
        //    delete W2;
        //}
        //if (!W_vector_.empty() && dW)
        //{
        //    //Matrix::copyData(W_vector_[0], dW);
        //    dW->initData(0);
        //}
    }
    //if (W)
    auto& W = *W_;
    auto& dW = W_->d();
    real pre_norm_l2;
    if (auto_weight_decay_ == 1)
    {
        pre_norm_l2 = W.dotSelf();
    }
    switch (solver_type_)
    {
    case SOLVER_SGD:
    case SOLVER_NAG:
        if (normalized_dweight_)
        {
            W.d().scale(sqrt(W.dotSelf() / W.d().dotSelf()));
            Matrix::add(W, dW, W, 1 - weight_decay_ * learn_rate_, -learn_rate_ * lr_weight_scale_);
            if (weight_decay_l1_ != 0)
            {
                Matrix::sign(W, W_sign_, weight_decay_l1_ * learn_rate_);
                Matrix::add(W, W_sign_, W, 1, -1);
            }
        }
        else
        {
            Matrix::add(W, dW, W, 1 - weight_decay_ * learn_rate_, -learn_rate_ * lr_weight_scale_ / batch);
            if (weight_decay_l1_ != 0)
            {
                Matrix::sign(W, W_sign_, weight_decay_l1_ * learn_rate_);
                Matrix::add(W, W_sign_, W, 1, -1);
            }
        }

        if (solver_type_ == SOLVER_NAG)
        {
            Matrix::copyData(W, W_vector_[0]);
        }
        break;
    case SOLVER_ADA_DELTA:
        //LOG("ADADELTA\n");
        //使用第0个矩阵作为真正的更新量，下同
        dW.scale(1.0 / batch);
        MatrixEx::adaDeltaUpdate(W_vector_[1], W_vector_[2], dW, W_vector_[0], real_vector_[1], real_vector_[0]);
        Matrix::add(W, W_vector_[0], W, 1 - weight_decay_ * learn_rate_, -1);
        break;
    case SOLVER_ADAM:
        dW.scale(1.0 / batch);
        MatrixEx::adamUpdate(W_vector_[1], W_vector_[2], dW, W_vector_[0], real_vector_[2], real_vector_[3], real_vector_[0], time_step_);
        Matrix::add(W, W_vector_[0], W, 1 - weight_decay_ * learn_rate_, -learn_rate_ * lr_weight_scale_);
        break;
    case SOLVER_RMS_PROP:
        dW.scale(1.0 / batch);
        MatrixEx::adaRMSPropUpdate(W_vector_[1], dW, W_vector_[0], real_vector_[1], real_vector_[0]);
        Matrix::add(W, W_vector_[0], W, 1, -learn_rate_ * lr_weight_scale_);
        break;
    }
    if (auto_weight_decay_ == 1)
    {
        auto norm_l2 = W.dotSelf();
        if (norm_l2 > pre_norm_l2)    //模只能变小
        {
            W.scale(sqrt(pre_norm_l2 / norm_l2));
        }
    }
}

void Solver::destory()
{
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
    int e = (x / lr_step_) * lr_step_;
    real p = pow(y2 / y1, 1.0 * (e - x1) / (x2 - x1));
    real y = y1 * p;
    return y;
}

}    // namespace cccc