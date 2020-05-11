#pragma once
#include "CudaControl.h"
#include "SaveBuffer.h"
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
#include <vector>

#define MATRIX_OPERATOR

namespace woco
{

//矩阵类

//该类的赋值和构造（浅复制），对于矩阵本身的数据来说，是一个完全的替身，但是可以拥有不同的维度
//对赋值后的矩阵的数据进行任何修改，包含修改指针位置，均会影响原矩阵！
//如有特殊需求，请考虑clone

class DLL_EXPORT Matrix
{
public:
    friend class MatrixExtend;
    using Size = std::vector<int>;

private:
    //因存在多设备可能，数据须同时保存其设备指针，需以共享指针使用此类
    struct DataWarpper
    {
        CudaControl* cuda_ = nullptr;    //为空表示在cpu中，也即默认情况
        real* data_ = nullptr;
        int64_t occupy_data_size_ = 0;    //实际占用的数据长度，当重设置尺寸不大于此值的时候不会重新分配内存
    public:
        DataWarpper() = default;
        ~DataWarpper() { free(); }
        DataWarpper(const DataWarpper&) = delete;
        DataWarpper& operator=(const DataWarpper) = delete;
        void setCuda(CudaControl* cuda) { cuda_ = cuda; }
        void setCudaAsCurrent() { cuda_ = CudaControl::getCurrentCuda(); }
        real* resize(int64_t size, bool reserve_data = true, bool force = false);    //resize前应setCuda
        std::shared_ptr<real*> make_shared_data() { return std::make_shared<real*>(data_); }
        void free();
    };
    //std::shared_ptr<DataWarpper> makeSharedData() { return std::make_shared<DataWarpper>(); }

protected:
    //矩阵所使用的常数
    static const realc const_real_1;
    static const realc const_real_0;

    //DeviceType getDeviceType() = DeviceType::CPU;

    int64_t data_size_ = 0;

    //一列的数据作为一个或一组图像，矩阵本身是列优先
    //但是在图片处理，包含卷积核默认是行优先（遵从cudnn），也就是说图片和卷积核可以认为是转置保存的！！
    int width_ = 0, height_ = 0, channel_ = 0, number_ = 0;

    int row_ = 0;    //作为矩阵时的行数，列数等于number_

    Size dim_;    //维度

    TensorDescWrapper tensor_desc_;

    //数据，被迫使用双重指针是因为矩阵计算返回对象，和生成计算流的需要！
    std::shared_ptr<real*> data_ = std::make_shared<real*>();
    //共享指针的意义是自动析构不被引用的数据，不可以直接使用
    std::shared_ptr<DataWarpper> shared_data_ = std::make_shared<DataWarpper>();

    //反向数据，实际上除了数据指针，其他必然跟原矩阵相同，矩阵赋值或复制之后仍然维持关联。除操作符和求解器外，禁止其他部分使用
    mutable std::shared_ptr<std::unique_ptr<Matrix>> d_this_ = std::make_shared<std::unique_ptr<Matrix>>(nullptr);

    bool need_updata_ = true;    //仅反向使用此参数，决定是否需要更新D数据，禁止本体使用

public:
    //Matrix& operator=(const Matrix& m);
    //Matrix(const Matrix& src);

    Matrix(const Size& dim, DeviceType device_type = DeviceType::GPU);
    Matrix(int w, int h, int c, int n, DeviceType device_type = DeviceType::GPU);
    Matrix(int m, int n, DeviceType device_type = DeviceType::GPU);
    Matrix(const Size& dim, real* data, DeviceType device_type = DeviceType::GPU);
    Matrix(DeviceType device_type = DeviceType::GPU);
    //~Matrix();
    Matrix clone(DeviceType device_type = DeviceType::GPU) const;
    Matrix cloneShared() const;
    Matrix cloneSharedCol(int col = 0) const;

private:
    inline real* data() const { return *data_; }
    bool haveD() const { return d_this_->get() != nullptr; }
    const CudaControl* cuda() const { return shared_data_->cuda_; }
    cudnnTensorDescriptor_t getCudnnTensorDesc() const { return tensor_desc_.get(); }

public:
    DeviceType getDeviceType() const { return shared_data_->cuda_ == nullptr ? DeviceType::CPU : DeviceType::GPU; }
    bool inGPU() const { return cuda() != nullptr; }

public:
    Matrix& DMatrix() const;
    void setNeedReverse(bool b) { DMatrix().need_updata_ = b; }    //会生成反向矩阵，需慎用
    bool needReverse() const { return haveD() && DMatrix().need_updata_; }

private:
    void mallocDData() const;      //分配反向的矩阵空间，如果已经分配了就不再重新分配，经复制产生的矩阵共享反向
    void reMallocDData() const;    //重新创建一个反向，此矩阵若来自复制，则不再与被复制的对象共享空间

private:
    void setDim(const Size& dim);

public:
    //注意有一些在矩阵模式下才能正确返回结果，有一些在4阶张量模式下使用，而实际上4阶张量模式的作用范围有限
    int mn2i(int m, int n) const { return m + n * row_; }
    int whcn2i(int w, int h, int c, int n) const { return w + h * width_ + c * width_ * height_ + n * channel_ * width_ * height_; }
    int coord2i(const Size& c);
    bool haveData(int w, int h, int c, int n) const
    {
        int i = whcn2i(w, h, c, n);
        return (i >= 0 && i < data_size_);
    }

public:
    int getDimSize() const { return dim_.size(); }
    const std::vector<int>& getDim() const { return dim_; }
    int getRow() const { return row_; }
    int getCol() const { return number_; }
    int getWidth() const { return width_; }
    int getHeight() const { return height_; }
    int getChannel() const { return channel_; }
    int getNumber() const { return number_; }
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
    int resize(int m, int n, bool force = false);
    int resize(int w, int h, int c, int n, bool force = false);
    int resize(const Matrix& X, bool force = false);
    int resize(const Size& dim, bool force = false);
    int resizeNumber(int n, bool force = false);
    int resizeAndNumber(const Size& dim, int n, bool force = false);
    int resizeKeepNumber(const Size& dim, bool force = false);    //注意此处参数dim的最后一个值是无用的

    //重设数据位置类型，慎用！！
    //void resetDataType();

    //重设数据指针，这个函数可能不安全，慎用！！
    //void resetDataPointer(real* d) { data() = d; }

    //使用这个函数，主要是为了析构时同时删除数据指针，最好你清楚你在干啥！
    //void setInsideData(MatrixDataType id) { matrix_data_type_ = id; }

    void print(FILE* fout = stdout) const;
    void printAsVector(FILE* fout = stdout) const;
    void printAsMatrix(FILE* fout = stdout) const;

    int save(SaveBuffer& buffer) const;
    int load(SaveBuffer& buffer);

private:
    void copyDataInFromHost(real* src, int64_t size);
    void copyDataOutToHost(real* dst, int64_t size);

public:
    static void copyDataPointer(const Matrix& A, const real* A_pointer, Matrix& R, real* R_pointer, int64_t size = -1);
    static void copyDataPointer(DeviceType dt_src, const real* src, DeviceType dt_dst, real* dst, int64_t size);

public:
    static void copyData(const Matrix& A, Matrix& R, int64_t size = -1);
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
    void transpose(int transpose_flag);

private:
    //这两个函数主要用于内部操作数据，这个写法是避免数据已经在cpu时再复制一次
    std::shared_ptr<DataWarpper> dataMirrorCPU(bool reserve_data = true) const;

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
    realc dotSelf();
    static void sign(Matrix& A, Matrix& R, real v = 1, real section = 1e-4);

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

    void importData(real* v, int n);
    void exportData(real* v, int n);

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
    static bool checkMatrixDevice(const std::vector<const Matrix*>& v);    //此处用对象会有效率问题
    void message(const std::string& info = "");
};

}    // namespace woco