#pragma once
#include "GpuControl.h"
#include "MatrixData.h"
#include "TensorDesc.h"
#include "blas_types.h"
#include "types.h"
#include <algorithm>
#include <memory>
#include <string>
#include <vector>

#define MATRIX_OPERATOR

struct cudnnTensorStruct;

namespace cccc
{

//矩阵类

//该类的赋值和构造（浅复制），对于矩阵本身的数据来说，是一个完全的替身，但是可以拥有不同的维度
//对赋值后的矩阵的数据进行任何修改，包含修改指针位置，均会影响原矩阵！
//如有特殊需求，请考虑clone
//NCHW

class DLL_EXPORT Matrix
{
public:
    friend class MatrixEx;

protected:
    //矩阵所使用的常数
    static constexpr float const_real_1{ 1.0 };
    static constexpr float const_real_0{ 0.0 };
    static constexpr double const_double_1{ 1.0 };
    static constexpr double const_double_0{ 0.0 };
    //static constexpr half const_half_1{ 1.0 };
    //static constexpr half const_half_0{ 0.0 };

    //默认的矩阵数据类型，注意是一个全局变量，影响所有线程
    //应在程序开始时设置，不要在多线程中修改
    static DataType& current_data_type()
    {
        static DataType cdt;
        return cdt;
    }

    //一列的数据作为一个或一组图像，矩阵本身是列优先
    //但是在图片处理，包含卷积核默认是行优先（遵从cudnn），也就是说图片和卷积核可以认为是转置保存的！！

    //总尺寸为int64_t，分开的坐标为int，适应目前其他库的情况
    int64_t data_size_ = 0;
    int width_ = 0, height_ = 0, channel_ = 0, number_ = 0;
    int row_ = 0;

    std::vector<int> dim_;

    std::shared_ptr<TensorDesc> tensor_desc_;    //这里太麻烦了，干脆这么糊弄吧

    //数据，被迫使用双重指针是因为矩阵计算返回对象，和生成计算流的需要！
    void* data_ = nullptr;
    //共享指针的意义是自动析构不被引用的数据，不可以直接使用
    std::shared_ptr<MatrixData> shared_data_ = std::make_shared<MatrixData>();

    //反向数据，实际上除了数据指针，其他必然跟原矩阵相同，矩阵赋值或复制之后仍然维持关联。除操作符和求解器外，禁止其他部分使用
    mutable std::shared_ptr<Matrix> d_this_;

    bool need_back_ = true;    //仅反向使用此参数，决定是否需要更新D数据，本体的参数决定是否更新D数据
    float keep_weight_ = 0;

    void* user_data_ = nullptr;

public:
    //Matrix& operator=(const Matrix& m);
    //Matrix(const Matrix& src);

    Matrix(const std::vector<int>& dim, DataType data_type = DataType::CURRENT, UnitType device_type = UnitType::GPU, bool create_d = true);
    Matrix(int w, int h, int c, int n, DataType data_type = DataType::CURRENT, UnitType device_type = UnitType::GPU);
    Matrix(int m, int n, DataType data_type = DataType::CURRENT, UnitType device_type = UnitType::GPU);
    //Matrix(size_t size, DataType data_type = DataType::CURRENT, UnitType device_type = UnitType::GPU);
    Matrix(const std::vector<int>& dim, void* data, DataType data_type = DataType::CURRENT, UnitType device_type = UnitType::GPU);
    Matrix(DataType data_type = DataType::CURRENT, UnitType device_type = UnitType::GPU);
    //~Matrix();
    Matrix clone(UnitType device_type = UnitType::GPU) const;
    Matrix createShared() const;
    Matrix createSharedCol(int col = 0) const;
    Matrix autoShareClone(UnitType device_type) const;
    Matrix transDataType(DataType data_type) const;
    void release();    //会释放所有共享的数据，除非特别需要，一般不要使用


private:
    void* data() const { return data_; }
    float* dataf() const { return (float*)data_; }
    double* datad() const { return (double*)data_; }
    half* datah() const { return (half*)data_; }
    template <typename T>
    T* data() const { return (T*)data_; }
    //bool haveD() const { return d_this_ != nullptr; }
    GpuControl* gpu() const { return shared_data_->gpu_; }
    cudnnTensorStruct* cudnn_desc() const { return tensor_desc_->cudnnDesc(); }
    miopenTensorDescriptor* miopen_desc() const { return tensor_desc_->miopenDesc(); }

public:
    UnitType getDeviceType() const { return shared_data_->gpu_ == nullptr ? UnitType::CPU : UnitType::GPU; }
    bool inGpu() const { return gpu(); }
    bool isCuda() const { return gpu() && gpu()->getApiType() == API_CUDA; }
    bool isHip() const { return gpu() && gpu()->getApiType() == API_HIP; }
    ApiType getApiType() const { return shared_data_->getApiType(); }
    DataType getDataType() const { return shared_data_->getDataType(); }
    int getDataTypeByInt() const { return int(shared_data_->getDataType()); }
    size_t getDataTypeSize() const { return shared_data_->getDataTypeSize(); }
    static void setCurrentDataType(DataType dt) { current_data_type() = dt; }

public:
    Matrix& d() const;
    void setNeedBack(bool b) { need_back_ = b; }    //会生成反向矩阵，需慎用
    bool needBack() const { return need_back_; }
    void setKeepWeight(float kw) { keep_weight_ = kw; }
    float keepWeight() const { return keep_weight_; }
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
    int getOneChannelSize() const { return row_ / channel_; }

    int64_t getDataSize() const { return data_size_; }
    int64_t getDataSizeInByte() const { return getDataSize() * getDataTypeSize(); }

    void* getDataPtr() const { return data(); }
    void* getDataPtr(int i) const { return (char*)data() + i * getDataTypeSize(); }
    void* getDataPtr(int m, int n) const { return getDataPtr(mn2i(m, n)); }
    void* getDataPtr(int w, int h, int c, int n) const { return getDataPtr(whcn2i(w, h, c, n)); }

    //以下3个函数，注意如果数据在显存中，一般x来说是无法赋值和输出的
    //注意只返回float
    float getData(int i) const { return shared_data_->getData(i); }
    float getData(int m, int n) const { return getData(mn2i(m, n)); }
    float getData(int w, int h, int c, int n) const { return getData(whcn2i(w, h, c, n)); }

    template <typename T>
    void setData(int i, T v) { shared_data_->setData(i, v); }
    template <typename T>
    void setData(int m, int n, T v) { setData(mn2i(m, n), v); }
    template <typename T>
    void setData(int w, int h, int c, int n, T v) { setData(whcn2i(w, h, c, n), v); }
    void setData(int i, void* p, DataType data_type) { shared_data_->setData(i, p, data_type); }

public:
    //改变矩阵维度，同时矩阵数据尺寸可能会变化，如果尺寸变大会备份数据
    //返回值：-1空矩阵，未重新分配内存，1重新分配内存
    int resize(int m, int n, bool reserve_data = true, bool force = false);
    int resize(int w, int h, int c, int n, bool reserve_data = true, bool force = false);
    //int resize(const Matrix& X, bool reserve_data = true, bool force = false);
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
    void copyDataInFromHost(float* src, int64_t size);
    void copyDataOutToHost(float* dst, int64_t size);

public:
    static int64_t copyDataPtr(const Matrix& A, const void* A_ptr, Matrix& R, void* R_ptr, int64_t size = -1);

public:
    static void copyData(const Matrix& A, Matrix& R, int64_t size = -1);
    static void copyRows(const Matrix& A, int ra, Matrix& R, int rr, int64_t rows);
    static void copyDataAcrossDevice(const Matrix& A, Matrix& R, int64_t size = -1);

    void shareData(const Matrix& A);
    void shareData(const Matrix& A, int m, int n);
    void shareData(const Matrix& A, int w, int h, int c, int n);
    void shareData(float* data);

    //初始化数据后返回自身的引用，可以在一行代码内初始化
    Matrix& fillData(float v, float inc = 0);    //非常慢
    Matrix& loadDataPtr(void* data, int64_t size);
    Matrix& fillRandom(int seed = 0);    //调试用

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
    float sumAbs() const;
    float sumAbsCol(int c) const;
    float sum() const;

    void sectionLimit(float v0, float v1);
    void scale(float v);
    static void scale(const Matrix& A, Matrix& R, float v);
    void scaleCol(float v, int c);

    //为使功能完整，所有正向运算均应有这个返回矩阵的形式
    static void mul(const Matrix& A, const Matrix& B, Matrix& R, float a = 1, float r = 0, MatrixTransType ta = MATRIX_NO_TRANS, MatrixTransType tb = MATRIX_NO_TRANS);
    static void mulVector(Matrix& A, Matrix& B, Matrix& R, float a = 1, float r = 0, MatrixTransType ta = MATRIX_NO_TRANS);
    static void mulVector2(Matrix& A, Matrix& B, Matrix& R, float a = 1, float r = 0, MatrixTransType ta = MATRIX_NO_TRANS);
    static void elementMul(const Matrix& A, const Matrix& B, Matrix& R, float a = 1, float r = 0);
    static void add(const Matrix& A, const Matrix& B, Matrix& R, float a = 1, float b = 1, float r = 0);
    static float dot(const Matrix& A, const Matrix& B);
    static float dotCol(const Matrix& A, int cA, const Matrix& B, int cB);
    static float dotPart(int size, const Matrix& A, void* a, int cA, void* b, int cB);
    float dotSelf() const;
    static void sign(Matrix& A, Matrix& R, float v = 1, float section = 0);

public:
    //PYTHON only ----------------------------------------------------------------------------------------------------
    //取值和赋值，通常不推荐在c++中使用，仅用于python接口，故安全保护较多
    float getDataValue(int i)
    {
        if (getDeviceType() == UnitType::CPU && i >= 0 && i < data_size_)
        {
            return getData(i);
        }
        return 0;
    }
    float getDataValue(int m, int n) { return getDataValue(mn2i(m, n)); }
    float getDataValue(int w, int h, int c, int n) { return getDataValue(whcn2i(w, h, c, n)); }

    void setDataValue(float v, int i)
    {
        if (getDeviceType() == UnitType::CPU && i >= 0 && i < data_size_)
        {
            setData(i, v);
        }
    }
    void setDataValue(float v, int m, int n) { setDataValue(v, mn2i(m, n)); }
    void setDataValue(float v, int w, int h, int c, int n) { setDataValue(v, whcn2i(w, h, c, n)); }

    void importData(float* v, int64_t n);
    void exportData(float* v, int64_t n);

    //PYTHON only ----------------------------------------------------------------------------------------------------

public:
    //以下函数都是自己写cuda部分
    void reciprocal(float scale = 1);
    void addNumber(float v, float scale = 1);
    static void addNumber(const Matrix& A, Matrix& R, float v, float scale = 1);
    void addNumberCol(float v, float scale, int c);
    static void elementPow(const Matrix& A, Matrix& R, float e, float bias = 0);
    static void elementDiv(const Matrix& A, const Matrix& B, Matrix& R, float a = 0, float b = 0, float scale = 1);
    static void crossEntropy(const Matrix& A, const Matrix& Y, Matrix& R, float a = 0, float scale = 1);
    static void crossEntropy2(const Matrix& A, const Matrix& Y, Matrix& R, float a = 0, float scale = 1);    //二分类时仅用一个结果时的交叉熵

public:
    static bool checkMatrixDevice(const std::vector<const Matrix*>& v);    //此处用对象会有效率问题
    void message(const std::string& info = "") const;
    std::string sizeMessage(int include_batch = 1) const;
};

using MatrixSP = std::shared_ptr<Matrix>;
template <typename... Args>
inline MatrixSP makeMatrixSP(Args... args) { return std::make_shared<Matrix>(args...); }

//运算符重载：+-*数乘
inline Matrix operator+(const Matrix& A, const Matrix& B)
{
    Matrix R(A.getDim(), A.getDataType(), A.getDeviceType());
    Matrix::add(A, B, R);
    return R;
}

inline Matrix operator-(const Matrix& A, const Matrix& B)
{
    Matrix R(A.getDim(), A.getDataType(), A.getDeviceType());
    Matrix::add(A, B, R, 1, -1);
    return R;
}

inline Matrix operator*(const Matrix& A, const Matrix& B)
{
    Matrix R({A.getRow(), B.getNumber()}, A.getDataType(), A.getDeviceType());
    Matrix::mul(A, B, R);
    return R;
}

inline Matrix operator*(float r, const Matrix& A)
{
    Matrix R(A.getDim(), A.getDataType());
    Matrix::scale(A, R, r);
    return R;
}

inline Matrix operator*(const Matrix& A, float r)
{
    Matrix R(A.getDim(), A.getDataType());
    Matrix::scale(A, R, r);
    return R;
}

}    // namespace cccc

#define RUN_BY_DATA_TYPE(dt, func, p1, p2) \
    switch (dt) \
    { \
    case DataType::FLOAT: \
        func((float*)p1); \
        break; \
    case DataType::DOUBLE: \
        func((double*)p1); \
        break; \
    case DataType::HALF: \
        func((half*)p1); \
        break; \
    }
//cudnn中的scale数值，在half和float中均是half，但是在double中是double，此处暂时不处理double，其他同