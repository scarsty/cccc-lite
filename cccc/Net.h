#pragma once
#include "MatrixOp.h"
#include "Solver.h"

namespace cccc
{

class DLL_EXPORT Net
{
public:
    Net();
    virtual ~Net();
    Net(const Net&) = delete;
    Net& operator=(const Net&) = delete;

protected:
    Option* option_;

    Matrix all_weights_;

    Solver solver_;

    std::vector<MatrixOp> op_queue_;
    std::vector<MatrixOp> loss_;

    std::vector<Matrix*> weights_;

    MatrixSP X_, A_;
    MatrixSP Y_ = makeMatrixSP();
    MatrixSP loss_weight_ = makeMatrixSP();    //该变量用以强调或忽视某些样本

    GpuControl* gpu_;

    int seperate_update_weight_ = 0;

public:
    void setGpu(GpuControl* gpu) { gpu_ = gpu; }
    GpuControl* getCuda() { return gpu_; }
    void setDeviceSelf() { gpu_->setAsCurrent(); }
    Matrix& getAllWeights() { return all_weights_; }
    void setOption(Option* op) { option_ = op; }
    int getBatch() { return getX().getNumber(); }
    int init();
    virtual int init2() = 0;

public:
    Matrix& getX() { return *X_; }
    Matrix& getY() { return *Y_; }
    Matrix& getA() { return *A_; }
    Matrix& getLossWeight() { return *loss_weight_; }
    void initLossWeight() { loss_weight_->resize(*Y_); }

public:
    void active(Matrix* X, Matrix* Y, Matrix* A, bool back, realc* error);

    void updateWeight();

    virtual int saveWeight(const std::string& filename, const std::string& sign = "");
    int loadWeight(const std::string& str, int load_mode = 0);

    int weightDataSize() const;
    realc weightSumAbs() const;
    realc weightNorm2() const;
    void calNorm(int& n, realc& l1, realc& l2) const;
    void outputNorm() const;

private:
    int resetBatchSize(int n);
    std::vector<int> getTestGroup();

public:
    void setActivePhase(ActivePhaseType ap) {}
    int test(const std::string& info, Matrix* X, Matrix* Y, Matrix* A, int output_group = 0, int test_type = 0, int attack_times = 0, realc* result = nullptr);

protected:
    void combineWeights(std::vector<Matrix*>& weights, Matrix& result);
    void initWeights();
    std::string ir();

public:
    void attack(Matrix* X, Matrix* Y);
    void groupAttack(Matrix* X, Matrix* Y, int attack_times);

public:
    Solver& getSolver() { return solver_; }
    virtual void adjustByEpoch(int epoch, int total_epoch);

public:
    //Net* clone(int clone_data = 0);
};

}    // namespace cccc