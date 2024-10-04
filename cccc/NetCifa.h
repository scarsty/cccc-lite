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

private:
    void setXA(const MatrixSP& X, const MatrixSP& A)
    {
        X_ = X;
        A_ = A;
    }
};

}    // namespace cccc