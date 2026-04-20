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

    int id_ = -1;
    int epoch_ = 0;

    Matrix all_weights_;

    //权重是否分开更新
    //若某些层的学习率或者求解器不同，则需要分开更新
    int separate_update_weight_ = 0;
    Solver solver_;
    std::map<Matrix*, std::shared_ptr<Solver>> solvers_;

    std::vector<MatrixOp> op_queue_;
    std::vector<MatrixOp> loss_;

    std::vector<Matrix*> weights_;
    std::vector<Solver> solvers_for_weight_;

    MatrixSP X_, A_;
    MatrixSP Y_ = makeMatrixSP();

    std::map<std::string, MatrixSP> extra_matrixsp_;    //以备不时之需

    Matrix M_error_;

    GpuControl* gpu_;

    TestInfo test_info_;
    std::vector<TestInfo> group_test_info_;

public:
    void setId(int id) { id_ = id; }
    int getId() const { return id_; }
    int init();
    virtual int init2() = 0;
    void setGpu(GpuControl* gpu) { gpu_ = gpu; }
    GpuControl* getGpu() { return gpu_; }
    void setDeviceSelf() { gpu_->setAsCurrent(); }
    Matrix& getAllWeights() { return all_weights_; }
    void setOption(Option* op) { option_ = op; }
    int getBatch() { return getX().getNumber(); }
    void setEpoch(int epoch) { epoch_ = epoch; }
    const TestInfo& getTestInfo() { return test_info_; }
    const std::vector<TestInfo>& getGroupTestInfo() { return group_test_info_; }

public:
    Matrix& getX() { return *X_; }
    Matrix& getY() { return *Y_; }
    Matrix& getA() { return *A_; }
    MatrixSP& getMatrixByName(const std::string& name);
    void addExtraMatrix(const std::string& name, const std::vector<int>& dim);

public:
    void active(Matrix* X, Matrix* Y, Matrix* A, bool back, float* error);
    void updateWeight();

    virtual int saveWeight(const std::string& filename, const std::string& sign = "", int solver_state = 0);
    virtual int loadWeight(const std::string& str, int load_mode = 0, int solver_state = 0);

    int weightDataSize() const;
    float weightSumAbs() const;
    float weightNorm2() const;
    void calNorm(int& n, float& l1, float& l2) const;
    void outputNorm() const;

private:
    int resetBatchSize(int n);
    std::vector<int> getTestGroup();

public:
    virtual int test(Matrix* X, Matrix* Y, Matrix* A);

    virtual void test2(Matrix* X, Matrix* Y, Matrix* A) {}

protected:
    void combineWeights(std::vector<Matrix*>& weights, Matrix& result);
    void initWeights();
    std::string ir();

public:
    void attack(Matrix* X, Matrix* Y);

public:
    Solver& getSolver() { return solver_; }
    int solverAdjustLearnRate(int epoch, int total_epoch)
    {
        int ret = solver_.adjustLearnRate(epoch, total_epoch);
        for (auto& [m, s] : solvers_)
        {
            s->adjustLearnRate(epoch, total_epoch);
        }
        return ret;
    }

    int solverAdjustLearnRate2(int epoch, int total_epoch, const std::vector<std::vector<TestInfo>>& test_info)
    {
        int ret = solver_.adjustLearnRate2(epoch, total_epoch, test_info);
        for (auto& [m, s] : solvers_)
        {
            s->adjustLearnRate2(epoch, total_epoch, test_info);
        }
        return ret;
    }

    void solverReset()
    {
        solver_.reset();
        for (auto& [m, s] : solvers_)
        {
            s->reset();
        }
    }

public:
    virtual void doSomeThing() {}

    void clearTime();
    void outputTime() const;

    //Net* clone(int clone_data = 0);
    void test_only(Matrix& X, Matrix& A);
};

}    // namespace cccc