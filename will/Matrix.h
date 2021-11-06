#pragma once
#include "CudaControl.h"
#include "MatrixData.h"
#include "blas_types.h"
#include "cblas_real.h"
#include "types.h"
#include <algorithm>
#include <cfloat>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#define MATRIX_OPERATOR

namespace cccc
{

//矩阵类

//该类的赋值和构造（浅复制），对于矩阵本身的数据来说，是一个完全的替身，但是可以拥有不同的维度
//对赋值后的矩阵的数据进行任何修改，包含修改指针位置，均会影响原矩阵！
//如有特殊需求，请考虑clone

class Matrix
{
public:
    friend class MatrixEx;

private:
    //这个结构实际上是直接用cudnn的desc，属实验性质
    struct TensorDesc
    {
        int64_t unknown_ = 0;
        int64_t dim_size_ = 0;
        int64_t data_size_ = 0;
        int64_t data_size1_ = 0;
        int number_ = 0, channel_ = 0, height_ = 0, width_ = 0;
        int c3[4] = { 0 };
        int row_ = 0;
        int c4[15] = { 0 };
    };

    //std::shared_ptr<DataWarpper> makeSharedData() { return std::make_shared<DataWarpper>(); }

protected:
    //矩阵所使用的常数
    static constexpr realc const_real_1{ 1.0 };
    static constexpr realc const_real_0{ 0.0 };

    //一列的数据作为一个或一组图像，矩阵本身是列优先
    //但是在图片处理，包含卷积核默认是行优先（遵从cudnn），也就是说图片和卷积核可以认为是转置保存的！！

    int64_t data_size_ = 0;
    int width_ = 0, height_ = 0, channel_ = 0, number_ = 0;
    int row_ = 0;

    std::vector<int> dim_;

    TensorDesc tensor_desc_;    //这里太麻烦了，干脆这么糊弄吧

    //数据，被迫使用双重指针是因为矩阵计算返回对象，和生成计算流的需要！
    real* data_ = nullptr;
    //共享指针的意义是自动析构不被引用的数据，不可以直接使用
    std::shared_ptr<MatrixData> shared_data_ = std::make_shared<MatrixData>();

    //反向数据，实际上除了数据指针，其他必然跟原矩阵相同，矩阵赋值或复制之后仍然维持关联。除操作符和求解器外，禁止其他部分使用
    mutable std::shared_ptr<Matrix> d_this_;

    bool need_reverse_ = true;    //仅反向使用此参数，决定是否需要更新D数据，禁止本体使用
    realc keep_weight_ = 0;

    void* user_data_ = nullptr;

public:
    //Matrix& operator=(const Matrix& m);
    //Matrix(const Matrix& src);

    Matrix(const std::vector<int>& dim, DeviceType device_type = DeviceType::GPU, bool create_d = true);
    Matrix(int w, int h, int c, int n, DeviceType device_type = DeviceType::GPU);
    Matrix(int m, int n, DeviceType device_type = DeviceType::GPU);
    Matrix(const std::vector<int>& dim, real* data, DeviceType device_type = DeviceType::GPU);
    Matrix(DeviceType device_type = DeviceType::GPU);
    //~Matrix();
    Matrix clone(DeviceType device_type = DeviceType::GPU) const;
    Matrix cloneShared() const;
    Matrix cloneSharedCol(int col = 0) const;
    Matrix autoShareClone(DeviceType device_type) const;

private:
    inline real* data() const { return data_; }
    //bool haveD() const { return d_this_ != nullptr; }
    const CudaControl* cuda() const { return shared_data_->cuda_; }
    cudnnTensorDescriptor_t tensor_desc() const { return (cudnnTensorDescriptor_t)&tensor_desc_; }

public:
    DeviceType getDeviceType() const { return shared_data_->cuda_ == nullptr ? DeviceType::CPU : DeviceType::GPU; }
    bool inGPU() const { return cuda() != nullptr; }

public:
    Matrix& d() const;
    void setNeedReverse(bool b) { need_reverse_ = b; }    //会生成反向矩阵，需慎用
    bool needReverse() const { return need_reverse_; }
    void setKeepWeight(real kw) { keep_weight_ = kw; }
    realc keepWeight() const { return keep_weight_; }
    void*& user_data() { return user_data_; }

private:
    void setDim(const std::vector<int>& dim);

public:
    //注意有一些在矩阵模式下才能正确返回结果，有一些在4阶张量模式下使用，而实际上4阶张量模式的作用范围有限
    int mn2i(int m, int n) const { return m + n * row_; }
    int whcn2i(int w, int h, int c, int n) const { return w + h * width_ + c * width_ * height_ + n * channel_ * width_ * height_; }
    int coord2i(const std::vector<int>& c);
    bool haveData(int w, int h, int c, int n) const
    {
        int i = whcn2i(w, h, c, n);
        return (i >= 0 && i < data_size_);
    }

public:
    int getDimSize() const { return dim_.size(); }
    const std::vector<int>& getDim() const;    //维度
    int getWidth() const { return width_; }
    int getHeight() const { return height_; }
    int getChannel() const { return channel_; }
    int getNumber() const { return number_; }
    int getRow() const { return row_; }

    int64_t getDataSize() const { return data_size_; }
    int64_t getDataSizeInByte() const { return getDataSize() * sizeof(real); }

    //以下3个函数，注意如果数据在显存中，一般x来说是无法赋值和输出的
    //有胡搞嫌疑，但是看起来没有违反语义
    //这里似乎应该是调用两个版本重载 const real getData() const / real& getData()
    real& getData(int i) const { return data()[i]; }
    real& getData(int m, int n) const { return data()[mn2i(m, n)]; }
    real& getData(int w, int h, int c, int n) const { return data()[whcn2i(w, h, c, n)]; }

    real* getDataPointer() const { return data(); }
    real* getDataPointer(int i) const { return &data()[i]; }
    real* getDataPointer(int m, int n) const { return &data()[mn2i(m, n)]; }
    real* getDataPointer(int w, int h, int c, int n) const { return &data()[whcn2i(w, h, c, n)]; }
    //real& operator[](int i) const { return data_[i]; }

public:
    //改变矩阵维度，同时矩阵数据尺寸可能会变化，如果尺寸变大会备份数据
    //返回值：-1空矩阵，未重新分配内存，1重新分配内存
    int resize(int m, int n, bool reserve_data = true, bool force = false);
    int resize(int w, int h, int c, int n, bool reserve_data = true, bool force = false);
    int resize(const Matrix& X, bool reserve_data = true, bool force = false);
    int resize(const std::vector<int>& dim, bool reserve_data = true, bool force = false);
    int resizeNumber(int n, bool reserve_data = true, bool force = false);
    int resizeAndNumber(const std::vector<int>& dim, int n, bool reserve_data = true, bool force = false);
    int resizeKeepNumber(const std::vector<int>& dim, bool reserve_data = true, bool force = false);    //注意此处参数dim的最后一个值是无用的

    void print() const;
    void printAsVector() const;
    void printAsMatrix() const;

    //int save(SaveBuffer& buffer) const;
    //int load(SaveBuffer& buffer);
    int64_t save(void* buffer, int64_t size) const;
    int64_t load(const void* buffer, int64_t size);

private:
    void copyDataInFromHost(real* src, int64_t size);
    void copyDataOutToHost(real* dst, int64_t size);

public:
    static int64_t copyDataPointer(const Matrix& A, const real* A_pointer, Matrix& R, real* R_pointer, int64_t size = -1);

public:
    static void copyData(const Matrix& A, Matrix& R, int64_t size = -1);
    static void copyRows(const Matrix& A, int ra, Matrix& R, int rr, int64_t rows);
    static void copyDataAcrossDevice(const Matrix& A, Matrix& R, int64_t size = -1);

    void shareData(const Matrix& A);
    void shareData(const Matrix& A, int m, int n);
    void shareData(const Matrix& A, int w, int h, int c, int n);
    void shareData(real* data);

    //初始化数据后返回自身的引用，可以在一行代码内初始化
    Matrix& initData(real v, int inc = 0);    //非常慢
    Matrix& initData(real* data, int64_t size);
    Matrix& initRandom(int seed = 0);    //调试用

public:
    //这两个会修改共享数据，影响所有相关的矩阵
    void toGPU();
    void toCPU(bool reserve_data = true);

public:
    void flip(int flip_flag);
    void transpose();

private:
    //这两个函数主要用于内部操作数据，这个写法是避免数据已经在cpu时再复制一次
    std::shared_ptr<MatrixData> dataMirrorCPU(bool reserve_data = true) const;

public:
    //静态运算函数在结果矩阵使用显存时就调用cuda函数计算，但是调用者应保证所有矩阵一致
    //在使用cuda的时候也有可能存在在内存中的矩阵
    //运算函数
    void repeat(int c = 1);
    int indexColMaxAbs(int c) const;
    realc sumAbs() const;
    realc sumAbsCol(int c) const;
    realc sum() const;

    void sectionLimit(real v0, real v1);
    void scale(real v);
    static void scale(const Matrix& A, Matrix& R, real v);
    void scaleCol(real v, int c);

    //为使功能完整，所有正向运算均应有这个返回矩阵的形式
    static void mul(const Matrix& A, const Matrix& B, Matrix& R, real a = 1, real r = 0, MatrixTransType ta = MATRIX_NO_TRANS, MatrixTransType tb = MATRIX_NO_TRANS);
    static void mulVector(Matrix& A, Matrix& B, Matrix& R, real a = 1, real r = 0, MatrixTransType ta = MATRIX_NO_TRANS);
    static void mulVector2(Matrix& A, Matrix& B, Matrix& R, real a = 1, real r = 0, MatrixTransType ta = MATRIX_NO_TRANS);
    static void elementMul(const Matrix& A, const Matrix& B, Matrix& R, real a = 1, real r = 0);
    static void add(const Matrix& A, const Matrix& B, Matrix& R, realc a = 1, realc b = 1, realc r = 0);
    static realc dot(const Matrix& A, const Matrix& B);
    static realc dotCol(const Matrix& A, int cA, const Matrix& B, int cB);
    static realc dotPart(int size, const Matrix& A, real* a, int cA, real* b, int cB);
    realc dotSelf() const;
    static void sign(Matrix& A, Matrix& R, real v = 1, real section = 0);

public:
    //PYTHON only ----------------------------------------------------------------------------------------------------
    //取值和赋值，通常不推荐在c++中使用，仅用于python接口，故安全保护较多
    real getDataValue(int i)
    {
        if (getDeviceType() == DeviceType::CPU && i >= 0 && i < data_size_)
        {
            return getData(i);
        }
        return 0;
    }
    real getDataValue(int m, int n) { return getDataValue(mn2i(m, n)); }
    real getDataValue(int w, int h, int c, int n) { return getDataValue(whcn2i(w, h, c, n)); }

    void setDataValue(float v, int i)
    {
        if (getDeviceType() == DeviceType::CPU && i >= 0 && i < data_size_)
        {
            getData(i) = v;
        }
    }
    void setDataValue(float v, int m, int n) { setDataValue(v, mn2i(m, n)); }
    void setDataValue(float v, int w, int h, int c, int n) { setDataValue(v, whcn2i(w, h, c, n)); }

    void importData(real* v, int64_t n);
    void exportData(real* v, int64_t n);

    //PYTHON only ----------------------------------------------------------------------------------------------------

public:
    //以下函数都是自己写cuda部分
    void reciprocal(real scale = 1);
    void addNumber(real v, real scale = 1);
    static void addNumber(const Matrix& A, Matrix& R, real v, real scale = 1);
    void addNumberCol(real v, real scale, int c);
    static void elementPow(const Matrix& A, Matrix& R, real e, real bias = 0);
    static void elementDiv(const Matrix& A, const Matrix& B, Matrix& R, real a = 0, real b = 0, real scale = 1);
    static void crossEntropy(const Matrix& A, const Matrix& Y, Matrix& R, real a = 0, real scale = 1);
    static void crossEntropy2(const Matrix& A, const Matrix& Y, Matrix& R, real a = 0, real scale = 1);    //二分类时仅用一个结果时的交叉熵

public:
    //void setPrevWeight(real r) { prev_weight_ = r; }
    //void clearWeight() { prev_weight_ = 0; }
    //real getPrevWeight() { return prev_weight_; }

public:
    static bool checkMatrixDevice(const std::vector<const Matrix*>& v);    //此处用对象会有效率问题
    void message(const std::string& info = "") const;
};

using MatrixSP = std::shared_ptr<Matrix>;
template <typename... Args>
inline MatrixSP makeMatrixSP(Args... args) { return std::make_shared<Matrix>(args...); }

//运算符重载：+-*数乘
inline Matrix operator+(const Matrix& A, const Matrix& B)
{
    Matrix R(A.getDim(), A.getDeviceType());
    Matrix::add(A, B, R);
    return R;
}

inline Matrix operator-(const Matrix& A, const Matrix& B)
{
    Matrix R(A.getDim(), A.getDeviceType());
    Matrix::add(A, B, R, 1, -1);
    return R;
}

inline Matrix operator*(const Matrix& A, const Matrix& B)
{
    Matrix R(A.getRow(), B.getNumber(), A.getDeviceType());
    Matrix::mul(A, B, R);
    return R;
}

inline Matrix operator*(real r, const Matrix& A)
{
    Matrix R(A.getDim());
    Matrix::scale(A, R, r);
    return R;
}

inline Matrix operator*(const Matrix& A, real r)
{
    Matrix R(A.getDim());
    Matrix::scale(A, R, r);
    return R;
}

}    // namespace cccc