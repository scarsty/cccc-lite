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
    std::string script_;

    static NetCifa& getThis(cifa::Cifa& c);

    static cifa::Object registerMatrix(cifa::Cifa& c, Matrix& m);
    static cifa::Object registerLoss(cifa::Cifa& c, MatrixOperator::Queue& loss);
    static std::vector<int> getVector(cifa::ObjectVector& v, int index);

    static Matrix& toMatrix(cifa::Cifa& c, const cifa::Object& o);
    static MatrixOperator::Queue& toLoss(cifa::Cifa& c, const cifa::Object& o);

    static bool isMatrix(cifa::Cifa& c, const cifa::Object& o);
    static bool isLoss(cifa::Cifa& c, const cifa::Object& o);

public:
    NetCifa();
    virtual void structure() override;
    void setScript(const std::string& script) { script_ = script; }

    int runScript(const std::string& script);
    int registerFunctions();
};

}    // namespace woco