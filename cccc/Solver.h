#pragma once
#include "Matrix.h"
#include "Option.h"

namespace cccc
{

//更新参数的策略
class DLL_EXPORT Solver
{
public:
    Solver();
    virtual ~Solver();
    Solver(const Solver&) = delete;
    Solver& operator=(const Solver&) = delete;

protected:
    //int batch_;    //每批个数

    Matrix W_sign_;    //L1范数调整

    real weight_decay_ = 0, weight_decay_l1_ = 0;    //正则化参数或权重的衰减

    real momentum_ = 0;                //上次的dWeight保留，即动量
    int momentum_clear_epoch_ = -1;    //每隔数个epoch将动量积累清零

    //学习率调整相关
    AdjustLearnRateType lr_adjust_method_ = ADJUST_LEARN_RATE_FIXED;

    //int train_epochs_;

    real learn_rate_base_ = 0.01;    //基础学习率
    real learn_rate_;                //学习率

    int lr_steps_;
    real lr_step_decay_;
    std::vector<int> lr_inter_epoch_;
    std::vector<real> lr_inter_rate_;

    real lr_weight_scale_ = 1;    //权重的学习系数，乘以学习率为实际学习率
    real lr_bias_scale_ = 2;      //偏置的学习系数

    //求解器相关
    SolverType solver_type_ = SOLVER_SGD;
    real time_step_ = 0;
    int switch_sgd_epoch_ = -1;
    real switch_sgd_random_ = 0;
    int switch_solver_ = 0;

    std::vector<real> restrict_dweight_{ 0, 0 };         //是否需要限制梯度，以L1为准，梯度除以batch后，不应超过此值乘以权重
    std::vector<int> restrict_dweight_count_{ 0, 0 };    //上述操作的计数

    //求解器所需要的特殊参数
    std::vector<real> real_vector_;
    std::vector<int> int_vector_;
    std::vector<Matrix> W_vector_;

    Matrix dW_;    //实际用于更新的权重梯度，包含动量项

public:
    SolverType getSolverType() const { return solver_type_; }
    real getMomentum() const;
    void setMomentum(real m) { momentum_ = m; }
    void setLearnRateBase(real lrb) { learn_rate_base_ = lrb; }
    real getLearnRateBase() const { return learn_rate_base_; }
    real getLearnRate() const { return learn_rate_; }
    void resetTimeStep() { time_step_ = 0; }
    std::vector<Matrix>& getWVector() { return W_vector_; }
    //int getTrainEpochs() const { return train_epochs_; }

public:
    void init(Option* op, std::string section, Matrix& W);
    real adjustLearnRate(int epoch, int total_epoch);
    void updateWeightBiasPre(Matrix& W);
    void updateWeights(Matrix& W, int batch);
    void reset();
    void cancelRestrictDWeight();

    void outputState() const;

private:
    void destory();

    int findInter(int x, std::vector<int>& xv);
};

}    // namespace cccc