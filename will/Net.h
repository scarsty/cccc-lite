#pragma once
#include "MatrixOp.h"
#include "Neural.h"
#include "Solver.h"

namespace cccc
{

class Net : public Neural
{
public:
    Net();
    virtual ~Net();

protected:
    Option* option_;

    int device_id_ = -1;
    //int batch_ = 1;

    Matrix all_weights_;
    Matrix workspace_;

    Solver solver_;

    std::vector<MatrixOp> op_queue_;
    std::vector<MatrixOp> loss_;

    std::vector<Matrix*> weights_;

    MatrixSP X_, A_;
    MatrixSP Y_ = makeMatrixSP();
    Matrix loss_weight_;    //该变量用以强调某些类别

public:
    void setDevice(int dev) { device_id_ = CudaControl::setDevice(dev); }
    int getDevice() { return device_id_; }
    void setDeviceSelf() { CudaControl::setDevice(device_id_); }
    Matrix& getAllWeights() { return all_weights_; }
    Matrix& getWorkspace() { return workspace_; }
    void setOption(Option* op) { option_ = op; }
    //bool hasTrained() { return trained_; }
    int getBatch() { return getX().getNumber(); }
    int init();
    virtual int init2() = 0;

public:
    Matrix& getX() { return *X_; }
    Matrix& getY() { return *Y_; }
    Matrix& getA() { return *A_; }

public:
    void active(Matrix* X, Matrix* Y, Matrix* A, bool learn, realc* error);

    int saveWeight(const std::string& filename);
    int loadWeight(const std::string& str, int load_mode = 0);

    void calNorm(realc& l1, realc& l2);

private:
    int resetBatchSize(int n);
    std::vector<int> getTestGroup();

public:
    void setActivePhase(ActivePhaseType ap) {}
    int test(const std::string& info, Matrix* X, Matrix* Y, Matrix* A, int output_group = 0, int test_max = 0, int attack_times = 0, realc* result = nullptr);

protected:
    void combineWeights();
    void initWeights();

public:
    void attack(Matrix* X, Matrix* Y);
    void groupAttack(Matrix* X, Matrix* Y, int attack_times);

public:
    realc adjustLearnRate(int ec) { return solver_.adjustLearnRate(ec); }
    void setLearnRateBase(real lrb) { solver_.setLearnRateBase(lrb); }

public:
    //Net* clone(int clone_data = 0);
};

}    // namespace cccc