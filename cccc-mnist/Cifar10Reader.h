#pragma once
#include "Matrix.h"

namespace cccc
{

class Cifar10Reader
{
public:
    Cifar10Reader();
    const int label_ = 10;

private:
    //void getDataSize(const std::string& file_image, int* w, int* h, int* n);
    //void readLabelFile(const std::string& filename, real* y_data);
    //void readImageFile(const std::string& filename, real* x_data);
    //void readData(const std::string& file_label, const std::string& file_image, Matrix& X, Matrix& Y);

public:
    void load(Matrix& X, Matrix& Y, std::string path = "cifa10", int data_type = 1);    //data_type：1是训练集，2是测试集
};

}    // namespace cccc