#pragma once
#include "MatrixOperator.h"
#include "Option.h"
#include "Solver.h"

namespace woco
{

class DLL_EXPORT Net
{
public:
    Net();
    virtual ~Net() = default;

protected:
    int device_id_ = -1;
    ActivePhaseType active_phase_ = ACTIVE_PHASE_TRAIN;

    //应注意以下矩阵随网络一同创建，会选择当前的设备

    Matrix X_, Y_, A_;    //规定：X输入，Y标准答案，A计算值

    std::vector<Matrix> weights_;
    Matrix combined_weight_, workspace_weight_;

    MatrixOperator::Queue op_queue_, loss_;
    Matrix workspace_back_;

    std::vector<Solver> solvers_;    //应等于weight的size

    std::string message_;

public:
    void setDevice(int dev) { device_id_ = CudaControl::setDevice(dev); }
    int getDevice() { return device_id_; }
    void setDeviceSelf() { CudaControl::setDevice(device_id_); }
    void setActivePhase(ActivePhaseType ap) { active_phase_ = ap; }

    Matrix& X() { return X_; }
    Matrix& Y() { return Y_; }
    Matrix& A() { return A_; }

    void makeStructure();
    virtual void structure();
    void forward();
    void backward();

public:
    void setXYA(const Matrix& X, const Matrix& Y, const Matrix& A);
    void setX(const Matrix& X) { X_ = X; }
    void setY(const Matrix& Y) { Y_ = Y; }
    void setA(const Matrix& A) { A_ = A; }

    void addWeight(const Matrix& w) { weights_.push_back(w); }
    template <typename... Args>
    void addWeight(const Args&... args)
    {
        (addWeight(args), ...);
    }

    void addLoss(const MatrixOperator::Queue& loss) { loss_ = loss_ + loss; }
    template <typename... Args>
    void addLoss(const Args&... args)
    {
        (addLoss(args), ...);
    }

public:
    void cal(bool learn);

    Matrix& getCombinedWeight() { return combined_weight_; }
    Matrix& getWorkspaceWeight() { return workspace_weight_; }
    int saveWeight(const std::string& filename);
    int loadWeight(const std::string& str, int load_mode = 0);
    void calNorm(realc& l1, realc& l2);

    virtual void save(const std::string& filename) {}

    void setMessage(const std::string& m) { message_ = m; }

private:
    int resetBatchSize(int n);

public:
    realc adjustLearnRate(int ec);
    int test(const std::string& info, Matrix& X, Matrix& Y, Matrix& A, int output_group = 0, int test_max = 0, int attack_times = 0);

public:
    void combineParameters();

public:
    void attack(Matrix& X, Matrix& Y);
    void groupAttack(Matrix& X, Matrix& Y, int attack_times);

public:
    std::vector<Solver>& getSolvers() { return solvers_; }
};

}    // namespace woco