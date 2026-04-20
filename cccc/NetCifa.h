#pragma once
#include "Cifa.h"
#include "MatrixOp.h"
#include "Net.h"

namespace cccc
{

class DLL_EXPORT NetCifa : public Net
{
public:
    NetCifa();
    ~NetCifa();

private:
    cifa::Cifa cifa_;
    std::unordered_map<std::string, cifa::Object> cifa_parameters_;

    using Loss = std::vector<MatrixOp>;
    using MatrixGroup = std::vector<MatrixSP>;
    static std::vector<cifa::Object> getVector(cifa::ObjectVector& v, int index);
    static std::vector<int> getIntVector(cifa::ObjectVector& v, int index);
    static std::vector<float> getRealVector(cifa::ObjectVector& v, int index);

    std::string message_;

public:
    virtual int init2() override;

    int runScript(const std::string& script);
    int registerFunctions();

    void registerFunction(std::string name, cifa::Cifa::func_type func);

private:
    void setXA(const MatrixSP& X, const MatrixSP& A);

    template <typename... Args>
    MatrixSP makeMatrixSPWithState(Args... args)
    {
        auto m = std::make_shared<Matrix>(args...);
        m->setNeedBack(need_back_state_);
        m->setNeedLoad(need_load_state_);
        return m;
    }

    //创建矩阵时，会使用下面的参数
    bool need_back_state_ = true;    //是否需要反向传播
    bool need_load_state_ = true;    //是否需要加载数据

    SolverType current_solver_type_ = SOLVER_SGD;
};

}    // namespace cccc