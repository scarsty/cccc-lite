#pragma once
#include "Matrix.h"
#include "Neural.h"
#include "Option.h"

namespace cccc
{

//更新参数的策略
class Solver : public Neural
{
public:
    Solver();
    virtual ~Solver();

protected:
    //int batch_;    //每批个数

    Matrix* W_ = nullptr;

    Matrix W_sign_;    //L1范数调整

    real weight_decay_ = 0, weight_decay_l1_ = 0;    //正则化参数或权重的衰减

    int auto_weight_decay_ = 0;    //自动权重衰减，为1时维持模平方不变

    real momentum_ = 0;                //上次的dWeight保留，即动量
    int momentum_clear_epoch_ = -1;    //每隔数个epoch将动量积累清零

    //学习率调整相关
    AdjustLearnRateType lr_adjust_method_ = ADJUST_LEARN_RATE_FIXED;

    real learn_rate_base_ = 0.01;    //基础学习率
    real learn_rate_;                //学习率

    int lr_step_;
    std::vector<int> lr_inter_epoch_;
    std::vector<real> lr_inter_rate_;

    real lr_weight_scale_ = 1;
    real lr_bias_scale_ = 2;

    //求解器相关
    SolverType solver_type_ = SOLVER_SGD;
    real time_step_ = 0;
    int switch_sgd_epoch_ = -1;
    real switch_sgd_random_ = 0;
    int switch_solver_ = 0;

    int normalized_dweight_ = 0;    //是否需要归一化

    //求解器所需要的特殊参数
    std::vector<real> real_vector_;
    std::vector<int> int_vector_;
    std::vector<Matrix> W_vector_;

public:
    SolverType getSolverType() { return solver_type_; }
    real getMomentum();
    void setMomentum(real m) { momentum_ = m; }
    void setLearnRateBase(real lrb) { learn_rate_base_ = lrb; }
    real getLearnRateBase() { return learn_rate_base_; }
    real getLearnRate() { return learn_rate_; }
    void resetTimeStep() { time_step_ = 0; }

public:
    void init(Option* op, std::string section, Matrix& W);
    real adjustLearnRate(int epoch);
    void updateWeightBiasPre();
    void updateWeights(int batch);
    void actMomentum() { W_->d().scale(momentum_); }

private:
    void destory();

    int findInter(int x, std::vector<int>& xv);
    real linear_inter(int x, int x1, int x2, real y1, real y2);
    real scale_inter(int x, int x1, int x2, real y1, real y2);
};

}    // namespace cccc