#pragma once
#include "Matrix.h"

namespace woco
{

//更新参数的策略
class Solver
{
public:
    Solver() = default;

protected:
    //求解器相关
    SolverType solver_type_ = SOLVER_SGD;
    Matrix W_, DW_, W0_;
    int time_step_ = 0;
    real momentum_ = 0.9;    //上次的dWeight保留，即动量
    //int batch_ = 50;

    //学习率调整相关
    AdjustLearnRateType lr_adjust_method_ = ADJUST_LEARN_RATE_FIXED;
    real learn_rate_base_ = 0.01;    //基础学习率
    real learn_rate_ = 0.01;         //学习率
    int lr_step_;
    std::vector<int> lr_inter_epoch_;
    std::vector<real> lr_inter_rate_;
    real lr_weight_scale_ = 1;
    real lr_bias_scale_ = 2;

    //求解器所需要的特殊参数
    std::vector<real> real_vector_;
    std::vector<int> int_vector_;
    std::vector<Matrix> W_vector_;

    ActiveFunctionType active_ = ACTIVE_FUNCTION_NONE;

public:
    SolverType getSolverType() { return solver_type_; }
    real getMomentum() { return momentum_; }
    void setMomentum(real m) { momentum_ = m; }
    //void setBatch(int b) { batch_ = b; }
    void setLearnRateBase(real lrb) { learn_rate_base_ = lrb; }
    real getLearnRateBase() { return learn_rate_base_; }
    real getLearnRate() { return learn_rate_; }
    void resetTimeStep() { time_step_ = 0; }

public:
    //void init(Option* op, std::string section, int row, int batch, Matrix& W);
    void setWeight(const Matrix& W);
    real adjustLearnRate(int epoch);
    void updateWeightPre();
    void updateWeight(int batch);

private:
    static int findInter(int x, std::vector<int>& xv);
    static real linear_inter(int x, int x1, int x2, real y1, real y2);
    static real scale_inter(int x, int x1, int x2, real y1, real y2);
};

}    // namespace woco