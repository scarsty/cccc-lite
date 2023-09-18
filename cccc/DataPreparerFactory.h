#pragma once
#include "DataPreparer.h"

namespace cccc
{

struct DataPreparerFactory : public DataPreparer
{
private:
    static DataPreparer* create(Option* op, const std::string& section, const std::vector<int>& dim0, const std::vector<int>& dim1);

public:
    using UniquePtr = std::unique_ptr<DataPreparer>;
    inline static UniquePtr makeUniquePtr(Option* op, const std::string& section, const std::vector<int>& dim0, const std::vector<int>& dim1)
    {
        UniquePtr p(create(op, section, dim0, dim1));
        return p;
    }
};

}    // namespace cccc