#pragma once
#include "Matrix.h"
#include "Option.h"
#include "Random.h"

namespace cccc
{

class DLL_EXPORT DataPreparer
{
public:
    Matrix X;    //数据变换后的形态，例如随机顺序，亮度对比度，增加噪声等
    Matrix Y;

    Matrix X0;    //数据的原始状态，一般是直接从文件中读取的样子
    Matrix Y0;

    std::unordered_map<std::string, Matrix> extra_data_, extra_data0_;    //额外的数据，例如位置等

    bool is_new_data_ = false;    //是否有新数据，供外部查询

public:
    friend class DataPreparerFactory;

public:
    DataPreparer();
    virtual ~DataPreparer();
    DataPreparer(const DataPreparer&) = delete;
    DataPreparer& operator=(const DataPreparer&) = delete;

protected:
    void init();
    virtual void init2() {}
    int getFillGroup();
    //virtual void destroy() {}

public:
    virtual void initData();
    virtual void fillData0() {}
    virtual void transOne(Matrix& X1, Matrix& Y1) {}
    virtual void showOneData(int number) {}
    virtual void checkData(Matrix& A) {}    //对训练集进行一些检查，统计等，由子类决定
    virtual void summary() {}
    void addMatrix(const std::string& name, const std::vector<int>& dim);
    const std::unordered_map<std::string, Matrix>& getExtraData() { return extra_data_; }

protected:
    std::string getSection() { return section_; }

private:
    std::string create_by_dll_;

public:
    void shuffleQueue(std::vector<int>& train_queue);
    void prepareData(int epoch, const std::string& info);

    void resetDataDim();

protected:
    int shuffle_ = 1;        //是否乱序训练集
    int trans_ = 0;          //是否变换
    int fill_ = 0;           //是否填充
    int fill_group_ = 0;     //填充的组数
    int fill_period_ = 1;    //填充的周期

    std::string section_ = "data_preparer";
    Option* option_;

    //for macro in Option.h
    std::string& section = section_;
    Option*& option = option_;

    //图的尺寸
    std::vector<int> dim0_, dim1_;

    std::vector<int> queue_origin_;    //填充的顺序

private:
    std::vector<std::string> message_;
    Random<double> rand_;

public:
    std::string getMessage(int i);
    int isFill() { return fill_; }
};

//template <typename T>
//void parallel_for_each(std::function<void(int)> fn, int concurrency)
//{
//    std::vector<std::thread> threads(concurrency);
//    int count_th = (queue_origin_.size() + concurrency - 1) / concurrency;
//    int i_th = 0;
//    for (auto& t : threads)
//    {
//        t = std::thread{[](){fn();}};
//        i_th++;
//    }
//    for (auto& t : threads)
//    {
//        t.join();
//    }
//}

}    // namespace cccc