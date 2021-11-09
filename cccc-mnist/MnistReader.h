#pragma once
#include "Matrix.h"

namespace cccc
{

class MnistReader
{
public:
    MnistReader();
    const int label_ = 10;

private:
    void getDataSize(const std::string& file_image, int* w, int* h, int* n);
    void readLabelFile(const std::string& filename, real* y_data);
    void readImageFile(const std::string& filename, real* x_data);
    void readData(const std::string& file_label, const std::string& file_image, Matrix& X, Matrix& Y);

public:
    void load(Matrix& X, Matrix& Y, std::string path = "mnist", int data_type = 1);    //data_type��1��ѵ������2�ǲ��Լ�
};

}    // namespace cccc