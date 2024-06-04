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

    float weight_decay_ = 0, weight_decay_l1_ = 0;    //正则化参数或权重的衰减

    float momentum_ = 0.9;              //上次的dWeight保留，即动量
    int momentum_clear_epoch_ = -1;    //每隔数个epoch将动量积累清零

    //学习率调整相关
    AdjustLearnRateType lr_adjust_method_ = ADJUST_LEARN_RATE_FIXED;

    int lr_keep_count_ = 0;      //学习率保持不变的计数
    int lr_change_count_ = 0;    //学习率改变的计数

    float learn_rate_base_ = 0.01;     //基础学习率
    float learn_rate_ = 0.01;          //学习率

    int lr_steps_ = 3;
    float lr_step_decay_ = 0.1;
    int lr_test_stable_time_ = 20;       //数值稳定的判断次数
    double lr_test_std_limit_ = 0.01;    //数值稳定的判断标准差

    std::vector<int> lr_inter_epoch_;
    std::vector<float> lr_inter_rate_;

    float lr_weight_scale_ = 1;    //权重的学习系数，乘以学习率为实际学习率
    float lr_bias_scale_ = 2;      //偏置的学习系数

    //求解器相关
    SolverType solver_type_ = SOLVER_SGD;
    float time_step_ = 0;
    int switch_sgd_epoch_ = -1;
    float switch_sgd_random_ = 0;
    int switch_solver_ = 0;

    std::vector<float> restrict_dweight_{ 0, 0 };         //是否需要限制梯度，以L1为准，梯度除以batch后，不应超过此值乘以权重
    std::vector<int> restrict_dweight_count_{ 0, 0 };    //上述操作的计数

    //求解器所需要的特殊参数
    std::vector<float> real_vector_;
    std::vector<int> int_vector_;
    std::vector<Matrix> W_vector_;

    Matrix dW_;    //实际用于更新的权重梯度

public:
    SolverType getSolverType() const { return solver_type_; }
    float getMomentum() const;
    void setMomentum(float m) { momentum_ = m; }
    void setLearnRateBase(float lrb) { learn_rate_base_ = lrb; }
    float getLearnRateBase() const { return learn_rate_base_; }
    float getLearnRate() const { return learn_rate_; }
    void resetTimeStep() { time_step_ = 0; }
    std::vector<Matrix>& getWVector() { return W_vector_; }
    //int getTrainEpochs() const { return train_epochs_; }

public:
    void init(Option* option, std::string section, Matrix& W);
    int adjustLearnRate(int epoch, int total_epoch);
    int adjustLearnRate2(int epoch, int total_epoch, const std::vector<std::vector<TestInfo>>& test_info);
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