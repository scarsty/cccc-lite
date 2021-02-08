#pragma once
#include "DataPreparer.h"
#include <string>

namespace cccc
{

class DataPreparerTxt : public DataPreparer
{
private:
    int input_;
    int output_;
    std::string content_;

public:
    DataPreparerTxt();
    virtual ~DataPreparerTxt();

    void init2() override;
    void fillData(Matrix& X, Matrix& Y) override;

private:
    float getContent(int i);
};

}    // namespace cccc