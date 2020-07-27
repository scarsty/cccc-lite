#pragma once
#include "Cifa.h"
#include "Net.h"

class lua_State;

namespace woco
{

struct CompareObject
{
    bool operator()(const cifa::Object& l, const cifa::Object& r) const
    {
        return l.type + std::to_string(l.value) + l.content < r.type + std::to_string(r.value) + r.content;
    }
};

class DLL_EXPORT NetCifa : public Net
{
private:
    cifa::Cifa cifa_;

    std::map<cifa::Object, Matrix, CompareObject> map_matrix_;
    std::map<cifa::Object, MatrixOperator::Queue, CompareObject> map_loss_;

    cifa::Object registerMatrix(Matrix m);
    cifa::Object registerLoss(MatrixOperator::Queue loss);
    static std::vector<int> getVector(cifa::ObjectVector& v, int index);

public:
    NetCifa();
    virtual void structure() override;

    int runScript(const std::string& script);
    int registerFunctions();
};

}    // namespace woco