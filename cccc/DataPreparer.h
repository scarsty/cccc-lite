#pragma once
#include "Matrix.h"
#include "Neural.h"
#include "Option.h"
#include "Random.h"

namespace cccc
{

class DataPreparer : public Neural
{
public:
    friend class DataPreparerFactory;

public:
    DataPreparer();
    virtual ~DataPreparer();

protected:
    void init();
    virtual void init2() {}
    int getFillGroup();
    //virtual void destroy() {}

public:
    virtual void fillData(Matrix& X, Matrix& Y) {}
    void transData(const Matrix& X0, const Matrix& Y0, Matrix& X1, Matrix& Y1, const std::vector<int>& fill_queue);
    virtual void transOne(Matrix& X1, Matrix& Y1) {}
    virtual void showData(Matrix& X, Matrix& Y, int number) {}

protected:
    std::string getSection() { return section_; }

private:
    std::string create_by_dll_;

public:
    void reload();
    void shuffleQueue(std::vector<int>& train_queue);
    void prepareData(int epoch, const std::string& info, Matrix& X, Matrix& Y, Matrix& X_cpu, Matrix& Y_cpu);

    void initData(Matrix& X, Matrix& Y);
    static void readTxt(const std::string& filename, Matrix& X, Matrix& Y);
    static void readBin(const std::string& file_bin_x, const std::string& file_bin_y, Matrix& X, Matrix& Y);
    static void readOneBin(const std::string& file_bin, Matrix& data);
    static void writeBin(const std::string& file_bin, const Matrix& data);

    void resetDataDim(Matrix& X, Matrix& Y);

protected:
    int shuffle_ = 1;       //是否乱序训练集
    int trans_ = 0;         //是否变换
    int fill_ = 0;          //是否填充
    int fill_group_ = 0;    //填充的组数

    std::string section_ = "data_preparer";
    Option* option_;

    Random<double> rand_;

    //图的尺寸
    std::vector<int> dim0_, dim1_;

    std::vector<int> train_queue_origin_;    //填充的顺序
    int trained_in_origin_ = 0;              //已经处理的部分

protected:
    std::vector<std::string> message_;

public:
    std::string getMessage(int i);
    int isFill() { return fill_; }

protected:
    //以下函数仅用于简化读取
    template <typename T>
    static void fillNumVector(std::vector<T>& v, double value, int size)
    {
        for (int i = v.size(); i < size; i++)
        {
            v.push_back(T(value));
        }
    }
    //去掉下划线，使输出略为美观
    static std::string removeEndUnderline(const std::string& str)
    {
        if (!str.empty() && str.back() == '_')
        {
            return str.substr(0, str.size() - 1);
        }
        return str;
    }
};

//以下宏仅用于简化准备器的参数的读取，不可用于其他
#define NAME_STR(a) (removeEndUnderline(#a).c_str())
#define OPTION_GET_INT(a) \
    do { \
        a = option_->getInt(section_, #a, a); \
        LOG("  {} = {}\n", NAME_STR(a), a); \
    } while (0)
#define OPTION_GET_INT2(a, v) \
    do { \
        a = option_->getInt(section_, #a, v); \
        LOG("  {} = {}\n", NAME_STR(a), a); \
    } while (0)
#define OPTION_GET_REAL(a) \
    do { \
        a = option_->getReal(section_, #a, a); \
        LOG("  {} = {}\n", NAME_STR(a), a); \
    } while (0)
#define OPTION_GET_REAL2(a, v) \
    do { \
        a = option_->getReal(section_, #a, v); \
        LOG("  {} = {}\n", NAME_STR(a), a); \
    } while (0)
#define OPTION_GET_STRING(a) \
    do { \
        a = option_->getString(section_, #a, ""); \
        LOG("  {} = {}\n", NAME_STR(a), a); \
    } while (0)
#define OPTION_GET_NUMVECTOR(a, v, n, d) \
    do { \
        a.clear(); \
        strfunc::findNumbers(option_->getString(section_, #a, v), a); \
        a.resize(n, d); \
        LOG("  {} = {}\n", NAME_STR(a), a); \
    } while (0)

}    // namespace cccc