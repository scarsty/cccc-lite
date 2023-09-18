#pragma once
#include "DataPreparer.h"

namespace cccc
{

class DLL_EXPORT DataPreparerImage : public DataPreparer
{
protected:
    int flip_ = 0;                      //翻转
    int transpose_ = 0;                 //转置
    std::vector<double> d_contrast_;      //变换对比度
    std::vector<double> d_brightness_;    //变换亮度
    int d_channel_ = 0;                 //是否通道分别变换
    double d_noise_ = 0;                  //增加随机噪声

public:
    DataPreparerImage();
    virtual ~DataPreparerImage();

    void init2() override;
    void transOne(Matrix& X1, Matrix& Y1) override;

private:
    Random<double> rand_;
};

}    // namespace cccc