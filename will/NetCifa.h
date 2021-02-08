#pragma once
#include "Cifa.h"
#include "MatrixOp.h"
#include "Net.h"

namespace cccc
{

struct CompareObject
{
    bool operator()(const cifa::Object& l, const cifa::Object& r) const
    {
        return l.type + std::to_string(l.value) + l.content < r.type + std::to_string(r.value) + r.content;
    }
};

class NetCifa : public Net
{
private:
    cifa::Cifa cifa_;

    std::map<cifa::Object, MatrixSP, CompareObject> map_matrix_;
    std::map<cifa::Object, std::vector<MatrixOp>, CompareObject> map_loss_;

    cifa::Object registerMatrix(MatrixSP&& m);
    cifa::Object registerLoss(std::vector<MatrixOp> loss);
    static std::vector<cifa::Object> getVector(cifa::ObjectVector& v, int index);
    static std::vector<int> getIntVector(cifa::ObjectVector& v, int index);
    static std::vector<real> getRealVector(cifa::ObjectVector& v, int index);

    std::string message_;

    int count_;    //没有初始化，爱咋咋地

public:
    NetCifa();
    ~NetCifa();
    virtual int init2() override;

    int runScript(const std::string& script);
    int registerFunctions();

private:
    void setXA(MatrixSP& X, MatrixSP& A)
    {
        X_ = X;
        A_ = A;
    }
};

}    // namespace cccc