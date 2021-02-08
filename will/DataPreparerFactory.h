#pragma once
#include "DataPreparer.h"

namespace cccc
{

class DataPreparerFactory : DataPreparer
{
private:
    DataPreparerFactory() {}
    virtual ~DataPreparerFactory() {}

public:
    static DataPreparer* create(Option* op, const std::string& section, const std::vector<int>& dim0, const std::vector<int>& dim1);
    static void destroy(DataPreparer*& dp);
};

}    // namespace cccc