#pragma once
#include "Log.h"
#include "Matrix.h"
#include "Random.h"

namespace woco
{

class DLL_EXPORT DataPreparer
{
public:
    friend class Factory;

public:
    DataPreparer();
    virtual ~DataPreparer() = default;

protected:
    void init();
    virtual void init2() {}
    int getFillGroup();
    //virtual void destroy() {}

public:
    virtual void fillData(Matrix& X, Matrix& Y) {}
    void transData(const Matrix& X0, const Matrix& Y0, Matrix& X1, Matrix& Y1, const std::vector<int>& fill_queue);
    virtual void transOne(Matrix& X1, Matrix& Y1) {}
    //virtual void showData(DataGroup& data, int number) {}

protected:
    std::string getSection() { return section_; }

private:
    std::string create_by_dll_;

public:
    void reload();
    void shuffleQueue(std::vector<int>& train_queue);
    void prepareData(int epoch, const std::string& info, Matrix& X, Matrix& Y, Matrix& X_cpu, Matrix& Y_cpu);

    void initData(Matrix& X, Matrix& Y);
    void initData(Matrix& X, Matrix& Y, Matrix& X_cpu, Matrix& Y_cpu);
    static void readTxt(const std::string& filename, Matrix& X, Matrix& Y);
    static void readBin(const std::string& file_bin_x, const std::string& file_bin_y, Matrix& X, Matrix& Y);
    static void readBin(const std::string& file_bin, Matrix& data);
    static void writeBin(const std::string& file_bin, const Matrix& data);

    void resetDataDim(Matrix& X, Matrix& Y);

protected:
    int shuffle_ = 1;       //是否乱序训练集
    int trans_ = 0;         //是否变换
    int fill_ = 0;          //是否填充
    int fill_group_ = 0;    //填充的组数

    std::string section_ = "data_preparer";

    //图的尺寸
    std::vector<int> dimx_, dimy_;

    std::vector<int> train_queue_origin_;    //填充的顺序
    int trained_in_origin_ = 0;              //已经处理的部分

protected:
    std::vector<std::string> message_;

public:
    std::string getMessage(int i);
    int isFill() { return fill_; }

};

}    // namespace woco