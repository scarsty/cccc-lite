#pragma once
#include "DataPreparerImage.h"
#include "Random.h"

namespace cccc
{

class DataPreparerMnist : public DataPreparerImage
{
private:
    int fill_times_ = 0;
    int type_ = 1;
    int random_diff_ = 0;
    int remove59_ = 0;
    int reverse_color_ = 0;
    Random<double> rand_;

public:
    DataPreparerMnist();
    virtual ~DataPreparerMnist();

    void init2() override;
    void fillData(Matrix& X, Matrix& Y) override;
    void transOne(Matrix& X1, Matrix& Y1) override;
};

}    // namespace cccc