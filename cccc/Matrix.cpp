#include "Matrix.h"
#include "Log.h"
#include "Random.h"
#include "VectorMath.h"
#include "cblas_real.h"
#include "gpu_lib.h"
#include <cassert>

namespace cccc
{

//const realc Matrix::const_real_1 = 1;
//const realc Matrix::const_real_0 = 0;

//DataType Matrix::current_data_type_ = DataType::FLOAT;

//任意阶张量
//描述符最低为4维
Matrix::Matrix(const std::vector<int>& dim, DataType data_type, UnitType device_type, bool create_d)
{
    if (GpuControl::getGlobalCudaType() == UnitType::CPU) { device_type = UnitType::CPU; }
    if (data_type == DataType::CURRENT) { data_type = current_data_type(); }
    shared_data_->setDataType(data_type);
    auto size = VectorMath::multiply(dim);
    assert(device_type == UnitType::CPU || device_type == UnitType::GPU && GpuControl::getGlobalCudaType() == UnitType::GPU);
    if (device_type == UnitType::GPU && GpuControl::getGlobalCudaType() == UnitType::GPU)
    {
        shared_data_->setCudaAsCurrent();
    }
    resize(dim);
    if (create_d)
    {
        d_this_ = std::make_shared<Matrix>(std::vector<int>{ 0, 0 }, data_type, getDeviceType(), false);
    }
}

Matrix::Matrix(std::initializer_list<int> dim, DataType data_type, UnitType device_type, bool create_d) :
    Matrix(std::vector<int>(dim), data_type, device_type, create_d)
{
}

//4阶张量形式
Matrix::Matrix(int w, int h, int c, int n, DataType data_type, UnitType device_type) :
    Matrix(std::vector<int>{ w, h, c, n }, data_type, device_type)
{
}

//普通二维矩阵构造函数
//这样生成的矩阵在以张量形式处理时认为是4维张量，但是所有张量的row都是前面项的积
Matrix::Matrix(int m, int n, DataType data_type, UnitType device_type) :
    Matrix(std::vector<int>{ m, n }, data_type, device_type)
{
}

//Matrix::Matrix(size_t size, DataType data_type, UnitType device_type)
//{
//}

Matrix::Matrix(const std::vector<int>& dim, void* data, DataType data_type, UnitType device_type) :
    Matrix(std::vector<int>{ 0, 0 }, data_type, device_type)
{
    data_ = data;
    resize(dim);
}

//空矩阵，后期再调整
Matrix::Matrix(DataType data_type, UnitType device_type) :
    Matrix(std::vector<int>{ 0, 0 }, data_type, device_type)
{
}

Matrix Matrix::clone(UnitType device_type) const
{
    Matrix M(dim_, getDataType(), device_type);
    copyData(*this, M);
    return M;
}

Matrix Matrix::createShared() const
{
    return *this;
}

Matrix Matrix::createSharedCol(int col) const
{
    auto dim = dim_;
    dim.back() = 1;
    Matrix M(dim, getDataType(), getDeviceType());
    M.shareData(*this, 0, col);
    return M;
}

Matrix Matrix::autoShareClone(UnitType device_type) const
{
    if (device_type == getDeviceType())
    {
        return *this;
    }
    else
    {
        return clone(device_type);
    }
}

Matrix Matrix::transDataType(DataType data_type) const
{
    if (data_type == getDataType())
    {
        return *this;
    }
    Matrix M(dim_, data_type);
    if (isCuda())
    {
        //unfinished
    }
    else if (isHip())
    {
        //unfinished
    }
    else
    {
        for (int i = 0; i < data_size_; i++)
        {
            M.setData(i, getData(i));
        }
    }
    return M;
}

void Matrix::release()
{
    shared_data_->release();
}

Matrix& Matrix::d() const
{
    if (d_this_->data_size_ != data_size_)
    {
        d_this_->resize(dim_);
    }
    return *d_this_;
}

//依据dim设置尺寸相关的参数，内部使用
//若dim中仅有一个元素，则相当于{1,1,1,n}
void Matrix::setDim(const std::vector<int>& dim)
{
    dim_ = dim;
    int dim_size = dim.size();

    number_ = dim.back();
    width_ = 1;
    height_ = 1;
    channel_ = 1;
    data_size_ = 1;
    row_ = 1;
    for (int i = 0; i < dim_size; i++)
    {
        data_size_ *= int64_t(dim[i]);
        if (i < dim_size - 1) { row_ *= dim[i]; }
        if (i < dim_size - 3) { width_ *= dim[i]; }
        if (i == dim_size - 3) { height_ = dim[i]; }
        if (i == dim_size - 2) { channel_ = dim[i]; }
    }
    tensor_desc_ = std::make_shared<TensorDesc>();
    if (dim.size() <= 4)
    {
        tensor_desc_->setDesc4D(getDataType(), width_, height_, channel_, number_);
    }
    else
    {
        tensor_desc_->setDescND(getDataType(), dim);
    }
}

int Matrix::coord2i(const std::vector<int>& c)
{
    int r = 0;
    int s = 1;
    for (int i = 0; i < c.size(); i++)
    {
        r += c[i] * s;
        s = s * dim_[i];
    }
    return r;
}

const std::vector<int>& Matrix::getDim() const
{
    return dim_;
}

int Matrix::resize(int m, int n, bool reserve_data, bool force)
{
    return resize(std::vector<int>{ m, n }, reserve_data, force);
}

int Matrix::resize(int w, int h, int c, int n, bool reserve_data, bool force)
{
    return resize(std::vector<int>{ w, h, c, n }, reserve_data, force);
}

//int Matrix::resize(const Matrix& X, bool reserve_data, bool force)
//{
//    return resize(X.dim_, reserve_data, force);
//}

int Matrix::resize(const std::vector<int>& dim, bool reserve_data, bool force)
{
    setDim(dim);
    //指针与共享指针不同时，则认为是数据不由自己管理，不分配空间，此时可能会有隐患
    if (data() != shared_data_->data_)
    {
        return 2;
    }
    //空间不够或者强制则重新分配
    if (shared_data_->data_ == nullptr || data_size_ > shared_data_->occupy_data_size_ || force)
    {
        //重新申请空间
        shared_data_->resize(data_size_, getDataType(), reserve_data, force);
    }
    data_ = shared_data_->data_;
    return 0;
}

int Matrix::resizeNumber(int n, bool reserve_data, bool force)
{
    dim_.back() = n;
    return resize(dim_, reserve_data, force);
}

int Matrix::resizeAndNumber(const std::vector<int>& dim, int n, bool reserve_data, bool force)
{
    auto dim1 = dim;
    dim1.back() = n;
    return resize(dim1, reserve_data, force);
}

int Matrix::resizeKeepNumber(const std::vector<int>& dim, bool reserve_data, bool force)
{
    auto dim1 = dim;
    dim1.back() = number_;
    return resize(dim1, reserve_data, force);
}

//输出矩阵内容
//注意这个实际上是按照内存顺序
void Matrix::print() const
{
    auto temp = dataMirrorCPU();
    for (int p = 0; p < channel_ * number_; p++)
    {
        for (int h = 0; h < height_; h++)
        {
            for (int w = 0; w < width_; w++)
            {
                auto v = temp->getData(whcn2i(w, h, p, 0));
                LOG("{} ", float(v));
            }
            LOG("\n");
        }
        LOG("\n");
    }
}

//将矩阵当做向量，按照内存中的顺序依次输出
void Matrix::printAsVector() const
{
    auto temp = dataMirrorCPU();
    for (int i = 0; i < data_size_; i++)
    {
        LOG("{} ", temp->getData(i));
    }
    LOG("\n");
}

//按照矩阵输出，因为是列优先，故不是内存顺序
void Matrix::printAsMatrix() const
{
    auto temp = dataMirrorCPU();
    for (int r = 0; r < row_; r++)
    {
        for (int c = 0; c < number_; c++)
        {
            auto v = temp->getData(mn2i(r, c));
            LOG("{} ", float(v));
        }
        LOG("\n");
    }
    LOG("\n");
}

int64_t Matrix::save(void* buffer, int64_t size) const
{
    return MatrixData::copy(getApiType(), getDataPtr(), API_UNKNOWN, (float*)buffer, std::min(data_size_, size), getDataType());
}

int64_t Matrix::load(const void* buffer, int64_t size)
{
    return MatrixData::copy(API_UNKNOWN, (float*)buffer, getApiType(), getDataPtr(), std::min(data_size_, size), getDataType());
}

//将外界的值复制到矩阵，参数指针必须指向Host内存！
void Matrix::copyDataInFromHost(float* src, int64_t size)
{
    if (isCuda())
    {
        cudaMemcpy(data(), src, int(sizeof(float) * std::min(size, data_size_)), cudaMemcpyHostToDevice);
    }
    else
    {
        memcpy(data(), src, int(sizeof(float) * std::min(size, data_size_)));
    }
}

//将矩阵的值复制到外界，参数指针必须指向Host内存！
void Matrix::copyDataOutToHost(float* dst, int64_t size)
{
    if (isCuda())
    {
        cudaMemcpy(dst, data(), int(sizeof(float) * std::min(size, data_size_)), cudaMemcpyDeviceToHost);
    }
    else
    {
        memcpy(dst, data(), int(sizeof(float) * std::min(size, data_size_)));
    }
}

//警告：乱用模式会受惩罚！
int64_t Matrix::copyDataPtr(const Matrix& A, const void* A_ptr, Matrix& R, void* R_ptr, int64_t size)
{
    if (size < 0)
    {
        size = std::min(A.getDataSize(), R.getDataSize());
    }
    return MatrixData::copy(A.getApiType(), A_ptr, R.getApiType(), R_ptr, size, A.getDataType());
}

//复制数据，只处理较少的
void Matrix::copyData(const Matrix& A, Matrix& R, int64_t size)
{
    if (&A == &R)
    {
        return;
    }
    copyDataPtr(A, A.data(), R, R.data(), size);
}

void Matrix::copyRows(const Matrix& A, int ra, Matrix& R, int rr, int64_t rows)
{
    if (A.row_ != R.row_ || A.getDataSize() == 0)
    {
        return;
    }
    MatrixData::copy(A.getApiType(), A.getDataPtr(0, ra), R.getApiType(), R.getDataPtr(0, rr), A.row_ * rows, A.getDataType());
}

void Matrix::copyDataAcrossDevice(const Matrix& A, Matrix& R, int64_t size)
{
    if (size < 0)
    {
        size = std::min(A.getDataSize(), R.getDataSize());
    }
    int64_t size_in_byte = size * sizeof(float);
    if (R.isCuda() && A.isCuda())
    {
        cudaError state = cudaMemcpyPeer(R.data(), R.gpu()->getDeviceID(), A.data(), A.gpu()->getDeviceID(), size_in_byte);
        if (state != cudaSuccess)
        {
            LOG("cudaMemcpyPeer error: {}, size in byte {:g}\n", cudaGetErrorString(state), 1.0 * size_in_byte);
        }
    }
    else
    {
        copyDataPtr(A, A.data(), R, R.data(), size);
    }
}

void Matrix::shareData(const Matrix& A)
{
    shareData(A, 0, 0, 0, 0);
}

//将一个外部数据矩阵的指针指向其他位置
void Matrix::shareData(const Matrix& A, int m, int n)
{
    shareData(A, 0, 0, m, n);
}

void Matrix::shareData(const Matrix& A, int w, int h, int c, int n)
{
    assert(checkMatrixDevice({ this, &A }));
    if (gpu() != A.gpu())
    {
        LOG_ERR("Error: share data are on different device ({}, {})!\n", gpu()->getDeviceID(), A.gpu()->getDeviceID());
    }
    if (getDeviceType() == A.getDeviceType())
    {
        data_ = A.getDataPtr(w, h, c, n);
        if (shared_data_->data_ != A.shared_data_->data_)
        {
            shared_data_ = A.shared_data_;
        }
    }
}

//指针来自外部，故此时不宜将指针交由自身管理
void Matrix::shareData(float* data)
{
    data_ = data;
    auto gpu = shared_data_->gpu_;
    shared_data_ = std::make_shared<MatrixData>();
    shared_data_->gpu_ = gpu;
}

//以同一个值初始化矩阵
//inc不为零时仅用于测试，不要用于实际计算！
Matrix& Matrix::fillData(float v, float inc /*=0*/)
{
    if (!data())
    {
        return *this;
    }
    if (isCuda() && inc == 0)
    {
        if (v == 0)
        {
            cudaMemset(data(), 0, getDataSizeInByte());
        }
        else
        {
            cudnnSetTensor(gpu()->cudnn_handle_, cudnn_desc(), data(), &v);
        }
    }
    else if (isHip() && inc == 0 && v == 0)
    {
        if (v == 0)
        {
            hipMemset(data(), 0, getDataSizeInByte());
        }
    }
    else
    {
        auto temp = dataMirrorCPU(false);
        //#pragma loop(hint_parallel(8))
        if (v == 0 && inc == 0)
        {
            memset(data(), 0, getDataSizeInByte());
        }
        else
        {
            for (int i = 0; i < data_size_; i++)
            {
                temp->setData(i, i * inc + v);
            }
        }
        MatrixData::copy(API_UNKNOWN, temp->data_, getApiType(), getDataPtr(), data_size_, getDataType());
    }
    return *this;
}

Matrix& Matrix::loadDataPtr(void* data, int64_t size)
{
    MatrixData::copy(API_UNKNOWN, data, getApiType(), getDataPtr(), std::min(size, data_size_), getDataType());
    return *this;
}

//随机数初始化矩阵，注意这个函数调用次数很少
Matrix& Matrix::fillRandom(int seed /*= 0*/)
{
    if (!data())
    {
        return *this;
    }
    Random<float> r;
    r.set_seed(seed);
    std::vector<float> temp(data_size_);
    r.rand_data(temp.data(), temp.size());
    MatrixData::copy(API_UNKNOWN, temp.data(), getApiType(), getDataPtr(), data_size_, getDataType());
    return *this;
}

//内存中的数据转移到显存
void Matrix::toGPU()
{
    if (!isCuda() && GpuControl::getGlobalCudaType() == UnitType::GPU)
    {
        auto temp = shared_data_;
        //std::swap(temp, shared_data_->data_);
        shared_data_ = std::make_shared<MatrixData>();
        shared_data_->occupy_data_size_ = 0;
        shared_data_->setCudaAsCurrent();
        shared_data_->resize(data_size_, getDataType());
        MatrixData::copy(API_UNKNOWN, temp->data_, getApiType(), shared_data_->data_, data_size_, getDataType());
        //delete[] temp;
        data_ = shared_data_->data_;
    }
}

//显存中的数据转移到内存
void Matrix::toCPU(bool reserve_data)
{
    if (isCuda())
    {
        void* temp = new char[getDataSizeInByte()];
        MatrixData::copy(getApiType(), shared_data_->data_, API_UNKNOWN, temp, data_size_, getDataType());
        //shared_data_->free();
        shared_data_ = std::make_shared<MatrixData>();
        shared_data_->setCuda(nullptr);
        shared_data_->occupy_data_size_ = data_size_;
        std::swap(temp, shared_data_->data_);
        data_ = shared_data_->data_;
    }
}

//flip和transpose暂时仅用于cpu
void Matrix::flip(int flip_flag)
{
    if (isCuda())
    {
        return;
    }
    Matrix temp({ width_, height_ }, getDataType(), UnitType::CPU);
    for (int c = 0; c < channel_; c++)
    {
        for (int n = 0; n < number_; n++)
        {
            Matrix::copyDataPtr(*this, getDataPtr(0, 0, c, n), temp, temp.getDataPtr());
            switch (flip_flag)
            {
            case 1:
                for (int i = 0; i < width_; i++)
                {
                    for (int j = 0; j < height_; j++)
                    {
                        setData(i, j, c, n, temp.getData(width_ - 1 - i, j));
                    }
                }
                break;
            case 0:
                for (int i = 0; i < width_; i++)
                {
                    for (int j = 0; j < height_; j++)
                    {
                        setData(i, j, c, n, temp.getData(i, height_ - 1 - j));
                    }
                }
                break;
            case -1:
                for (int i = 0; i < width_; i++)
                {
                    for (int j = 0; j < height_; j++)
                    {
                        setData(i, j, c, n, temp.getData(width_ - 1 - i, height_ - 1 - j));
                    }
                }
                break;
            default:
                break;
            }
        }
    }
}

void Matrix::transpose()
{
    if (isCuda())
    {
        return;
    }
    auto temp = clone(UnitType::CPU);
    if (row_ != channel_)
    {
        resize({ height_, width_, channel_, number_ });
        for (int c = 0; c < channel_; c++)
        {
            for (int n = 0; n < number_; n++)
            {
                for (int i = 0; i < width_; i++)
                {
                    for (int j = 0; j < height_; j++)
                    {
                        setData(i, j, c, n, temp.getData(j, i, c, n));
                    }
                }
            }
        }
    }
    else
    {
        resize({ number_, row_ });
        for (int c = 0; c < channel_; c++)
        {
            for (int n = 0; n < number_; n++)
            {
                setData(c, n, temp.getData(n, c));
            }
        }
    }
}

//生成一个数据在cpu中的镜像
std::shared_ptr<MatrixData> Matrix::dataMirrorCPU(bool reserve_data) const
{
    if (inGpu())
    {
        auto new_shared_data = std::make_shared<MatrixData>();
        new_shared_data->resize(data_size_, getDataType());
        if (reserve_data)
        {
            MatrixData::copy(getApiType(), shared_data_->data_, API_UNKNOWN, new_shared_data->data_, data_size_, getDataType());
        }
        return new_shared_data;
    }
    else
    {
        return shared_data_;
    }
}

//将前面几列复制到整个矩阵
void Matrix::repeat(int c)
{
    if (isCuda())
    {
        for (int i = c; i < number_; i *= 2)
        {
            cudaMemcpy(getDataPtr(0, i), getDataPtr(0, 0), sizeof(float) * row_ * std::min(i, number_ - i), cudaMemcpyDeviceToDevice);
        }
    }
    else if (isHip())
    {
    }
    else
    {
        //#pragma loop(hint_parallel(8))
        for (int i = c; i < number_; i *= 2)
        {
            memcpy(getDataPtr(0, i), getDataPtr(0, 0), sizeof(float) * row_ * std::min(i, number_ - i));
        }
    }
}

//一列中最大值的序号
int Matrix::indexColMaxAbs(int c) const
{
    if (isCuda())
    {
        switch (getDataType())
        {
        case DataType::FLOAT:
            return gpu()->cublas_->iamax(row_, (float*)getDataPtr(0, c), 1);
            break;
        case DataType::DOUBLE:
            return gpu()->cublas_->iamax(row_, (double*)getDataPtr(0, c), 1);
            break;
        case DataType::HALF:
            return gpu()->cublas_->iamax(row_, (half*)getDataPtr(0, c), 1);
            break;
        }
    }
    else if (isHip())
    {
        return gpu()->rocblas_->iamax(row_, (float*)getDataPtr(0, c), 1);
    }
    else
    {
        return Cblas::iamax(row_, (float*)getDataPtr(0, c), 1);
    }
}

//绝对值求和（直接调用的blas，注意这里实际上需要的功能只是求和）
float Matrix::sumAbs() const
{
    if (isCuda())
    {
        switch (getDataType())
        {
        case DataType::FLOAT:
            return gpu()->cublas_->asum(data_size_, dataf(), 1);
            break;
        case DataType::DOUBLE:
            return gpu()->cublas_->asum(data_size_, datad(), 1);
            break;
        case DataType::HALF:
            return gpu()->cublas_->asum(data_size_, datah(), 1);
            break;
        }
    }
    else if (isHip())
    {
        return gpu()->rocblas_->asum(data_size_, dataf(), 1);
    }
    else
    {
        switch (getDataType())
        {
        case DataType::FLOAT:
            return Cblas::asum(data_size_, dataf(), 1);
            break;
        case DataType::DOUBLE:
            return Cblas::asum(data_size_, datad(), 1);
            break;
        case DataType::HALF:
        {
            float r = 0;
            for (int i = 0; i < data_size_; i++)
            {
                r += abs(getData(i));
            }
            return r;
            break;
        }
        }
    }
}

//一列的绝对值和
float Matrix::sumAbsCol(int c) const
{
    if (isCuda())
    {
        switch (getDataType())
        {
        case DataType::FLOAT:
            return gpu()->cublas_->asum(row_, (float*)getDataPtr(0, c), 1);
            break;
        case DataType::DOUBLE:
            return gpu()->cublas_->asum(row_, (double*)getDataPtr(0, c), 1);
            break;
        case DataType::HALF:
            return gpu()->cublas_->asum(row_, (half*)getDataPtr(0, c), 1);
            break;
        }
    }
    else if (isHip())
    {
        return gpu()->rocblas_->asum(row_, (float*)getDataPtr(0, c), 1);
    }
    else
    {
        return Cblas::asum(row_, (float*)getDataPtr(0, c), 1);
    }
}

float Matrix::sum() const
{
    Matrix temp1(dim_);
    temp1.fillData(1);
    float r = dot(*this, temp1);
    return r;
}

void Matrix::sectionLimit(float v0, float v1)
{
    if (isCuda())
    {
        cuda_sectionlimit(getDataTypeByInt(), data(), nullptr, data(), data_size_, v0, v1);
    }
    else if (isHip())
    {
    }
    else
    {
        for (int i = 0; i < data_size_; i++)
        {
            setData(i, std::min(getData(i), v1));
            setData(i, std::max(getData(i), v0));
        }
    }
}

//数乘
void Matrix::scale(float v)
{
    if (v == 1)
    {
        return;
    }
    if (v == 0)
    {
        fillData(0);
        return;
    }
    if (isCuda())
    {
        switch (getDataType())
        {
        case DataType::FLOAT:
            gpu()->cublas_->scal(data_size_, v, dataf(), 1);
            break;
        case DataType::DOUBLE:
            gpu()->cublas_->scal(data_size_, v, datad(), 1);
            break;
        case DataType::HALF:
            gpu()->cublas_->scal(data_size_, (half)v, datah(), 1);
            break;
        }
    }
    else if (isHip())
    {
        gpu()->rocblas_->scal(data_size_, v, dataf(), 1);
    }
    else
    {
        Cblas::scal(data_size_, v, dataf(), 1);
    }
}

//好像没有直接的功能
void Matrix::scale(const Matrix& A, Matrix& R, float v)
{
    copyData(A, R);
    R.scale(v);
}

//选择一列数乘
void Matrix::scaleCol(float v, int c)
{
    if (v == 1)
    {
        return;
    }
    if (isCuda())
    {
        switch (getDataType())
        {
        case DataType::FLOAT:
            gpu()->cublas_->scal(row_, v, (float*)getDataPtr(0, c), 1);
            break;
        case DataType::DOUBLE:
            gpu()->cublas_->scal(row_, v, (double*)getDataPtr(0, c), 1);
            break;
        case DataType::HALF:
            gpu()->cublas_->scal(row_, (half)v, (half*)getDataPtr(0, c), 1);
            break;
        }
    }
    else if (isHip())
    {
        gpu()->rocblas_->scal(row_, v, (float*)getDataPtr(0, c), 1);
    }
    else
    {
        Cblas::scal(row_, v, (float*)getDataPtr(0, c), 1);
    }
}

//矩阵乘，R = aAB+rR
void Matrix::mul(const Matrix& A, const Matrix& B, Matrix& R, float a, float r, MatrixTransType ta, MatrixTransType tb)
{
    assert(checkMatrixDevice({ &A, &B, &R }));
    int m = R.row_;
    int n = R.number_;
    int lda = A.row_;
    int k = A.number_;
    int ldb = B.row_;
    if (ta == MATRIX_TRANS)
    {
        k = A.row_;
    }
    if (R.isCuda())
    {
        switch (R.getDataType())
        {
        case DataType::FLOAT:
            R.gpu()->cublas_->gemm(ta, tb, m, n, k, a, A.dataf(), lda, B.dataf(), ldb, r, R.dataf(), m);
            break;
        case DataType::DOUBLE:
            R.gpu()->cublas_->gemm(ta, tb, m, n, k, a, A.datad(), lda, B.datad(), ldb, r, R.datad(), m);
            break;
        case DataType::HALF:
            R.gpu()->cublas_->gemm(ta, tb, m, n, k, (half)a, A.datah(), lda, B.datah(), ldb, (half)r, R.datah(), m);
            break;
        }
    }
    else if (R.isHip())
    {
        R.gpu()->rocblas_->gemm(ta, tb, m, n, k, a, A.dataf(), lda, B.dataf(), ldb, r, R.dataf(), m);
    }
    else
    {
        Cblas::gemm(ta, tb, m, n, k, a, A.dataf(), lda, B.dataf(), ldb, r, R.dataf(), m);
    }
}

//矩阵乘以向量，R = aAB+rR
//B和R的维度会被无视
void Matrix::mulVector(Matrix& A, Matrix& B, Matrix& R, float a, float r, MatrixTransType ta)
{
    assert(checkMatrixDevice({ &A, &B, &R }));
    int m = A.row_, n = A.number_;

    if (R.isCuda())
    {
        switch (R.getDataType())
        {
        case DataType::FLOAT:
            R.gpu()->cublas_->gemv(ta, m, n, a, A.dataf(), A.row_, B.dataf(), 1, r, R.dataf(), 1);
            break;
        case DataType::DOUBLE:
            R.gpu()->cublas_->gemv(ta, m, n, a, A.datad(), A.row_, B.datad(), 1, r, R.datad(), 1);
            break;
        case DataType::HALF:
            R.gpu()->cublas_->gemv(ta, m, n, (half)a, A.datah(), A.row_, B.datah(), 1, (half)r, R.datah(), 1);
            break;
        }
    }
    else if (R.isHip())
    {
        R.gpu()->rocblas_->gemv(ta, m, n, a, A.dataf(), A.row_, B.dataf(), 1, r, R.dataf(), 1);
    }
    else
    {
        Cblas::gemv(ta, m, n, a, A.dataf(), A.row_, B.dataf(), 1, r, R.dataf(), 1);
    }
}

//没什么用，废弃
void Matrix::mulVector2(Matrix& A, Matrix& B, Matrix& R, float a, float r, MatrixTransType ta)
{
    assert(checkMatrixDevice({ &A, &B, &R }));
    int m = A.row_, n = A.number_;
    if (ta == MATRIX_TRANS)
    {
        std::swap(m, n);
    };

    if (R.isCuda())
    {
        for (int i = 0; i <= R.number_; i++)
        {
            switch (R.getDataType())
            {
            case DataType::FLOAT:
                R.gpu()->cublas_->gemv(ta, m, n, a, A.dataf(), A.row_, B.dataf(), 1, r, (float*)R.getDataPtr(0, i), 1);
                break;
            case DataType::DOUBLE:
                R.gpu()->cublas_->gemv(ta, m, n, a, A.datad(), A.row_, B.datad(), 1, r, (double*)R.getDataPtr(0, i), 1);
                break;
            case DataType::HALF:
                R.gpu()->cublas_->gemv(ta, m, n, (half)a, A.datah(), A.row_, B.datah(), 1, (half)r, (half*)R.getDataPtr(0, i), 1);
                break;
            }
        }
    }
    else if (R.isHip())
    {
    }
    else
    {
        for (int i = 0; i <= R.number_; i++)
        {
            Cblas::gemv(ta, m, n, a, A.dataf(), A.row_, B.dataf(), 1, r, (float*)R.getDataPtr(0, i), 1);
        }
    }
}

//矩阵元素乘，B和R数据不能指向同一区域
void Matrix::elementMul(const Matrix& A, const Matrix& B, Matrix& R, float a, float r)
{
    assert(checkMatrixDevice({ &A, &B, &R }));
    if (R.isCuda())
    {
        int op_desc[64] = { 0 };
        cudnnSetOpTensorDescriptor((cudnnOpTensorDescriptor_t)op_desc, CUDNN_OP_TENSOR_MUL, toCudnnDataType(R.getDataType()), CUDNN_NOT_PROPAGATE_NAN);
        //好像B不能与R相同
        if (R.data() != B.data())
        {
            cudnnOpTensor(R.gpu()->cudnn_handle_, (cudnnOpTensorDescriptor_t)op_desc,
                &a, A.cudnn_desc(), A.data(), &const_real_1, B.cudnn_desc(), B.data(), &r, R.cudnn_desc(), R.data());
        }
        else
        {
            cudnnOpTensor(R.gpu()->cudnn_handle_, (cudnnOpTensorDescriptor_t)op_desc,
                &a, B.cudnn_desc(), B.data(), &const_real_1, A.cudnn_desc(), A.data(), &r, R.cudnn_desc(), R.data());
        }
    }
    else if (R.isHip())
    {
    }
    else
    {
        for (int i = 0; i < A.data_size_; i++)
        {
            R.setData(i, A.getData(i) * B.getData(i) * a + R.getData(i) * r);
        }
    }
}

//矩阵加，系数为负时可以为减
void Matrix::add(const Matrix& A, const Matrix& B, Matrix& R, float a, float b, float r)
{
    assert(checkMatrixDevice({ &A, &B, &R }));
    if (A.data() == B.data() && B.data() == R.data() && a + b + r == 1)
    {
        return;
    }
    //注意只使用R的tensor_desc
    if (R.isCuda())
    {
        if (A.data() == R.data())
        {
            r = r + a;
            if (b != 0 || r != 1)
            {
                cudnnAddTensor(R.gpu()->cudnn_handle_, &b, R.cudnn_desc(), B.data(), &r, R.cudnn_desc(), A.data());
            }
        }
        else if (B.data() == R.data())
        {
            r = r + b;
            if (a != 0 || r != 1)
            {
                cudnnAddTensor(R.gpu()->cudnn_handle_, &a, R.cudnn_desc(), A.data(), &r, R.cudnn_desc(), B.data());
            }
        }
        else
        {
            cudnnOpTensorDesc op_desc;
            //geam非Blas标准
            //R.cuda()->cublas_->geam(MATRIX_NO_TRANS, MATRIX_NO_TRANS, A.row_, A.number_, a, A.data(), A.row_, b, B.data(), B.row_, R.data(), R.row_);
            cudnnSetOpTensorDescriptor(op_desc(), CUDNN_OP_TENSOR_ADD, CUDNN_DATA_FLOAT, CUDNN_NOT_PROPAGATE_NAN);
            //这个函数要求R不可等于A或B
            auto ret = cudnnOpTensor(R.gpu()->cudnn_handle_, op_desc(), &a, R.cudnn_desc(), A.data(), &b, R.cudnn_desc(), B.data(), &r, R.cudnn_desc(), R.data());
            if (ret)
            {
                LOG_ERR("{}\n", R.gpu()->lastCudnnErrorString());
            }
        }
    }
    else if (R.isHip())
    {
        R.gpu()->rocblas_->geam(MATRIX_NO_TRANS, MATRIX_NO_TRANS, A.row_, A.number_, a, A.dataf(), A.row_, b, B.dataf(), B.row_, R.dataf(), R.row_);
    }
    else
    {
        for (int i = 0; i < R.data_size_; i++)
        {
            R.setData(i, a * A.getData(i) + b * B.getData(i) + r * R.getData(i));
        }
    }
}

//整个矩阵点乘
float Matrix::dot(const Matrix& A, const Matrix& B)
{
    assert(checkMatrixDevice({ &A, &B }));
    if (A.isCuda())
    {
        switch (A.getDataType())
        {
        case DataType::FLOAT:
            return A.gpu()->cublas_->dot(A.data_size_, A.dataf(), 1, B.dataf(), 1);
            break;
        case DataType::DOUBLE:
            return A.gpu()->cublas_->dot(A.data_size_, A.datad(), 1, B.datad(), 1);
            break;
        case DataType::HALF:
            return A.gpu()->cublas_->dot(A.data_size_, A.datah(), 1, B.datah(), 1);
            break;
        }
    }
    else if (A.isHip())
    {
        return A.gpu()->rocblas_->dot(A.data_size_, (float*)A.getDataPtr(), 1, (float*)B.getDataPtr(), 1);
    }
    else
    {
        return Cblas::dot(A.data_size_, (float*)A.getDataPtr(), 1, (float*)B.getDataPtr(), 1);
    }
}

//选择矩阵的某列点乘
float Matrix::dotCol(const Matrix& A, int cA, const Matrix& B, int cB)
{
    assert(checkMatrixDevice({ &A, &B }));
    if (A.isCuda())
    {
        switch (A.getDataType())
        {
        case DataType::FLOAT:
            return A.gpu()->cublas_->dot(A.row_, (float*)A.getDataPtr(0, cA), 1, (float*)B.getDataPtr(0, cA), 1);
            break;
        case DataType::DOUBLE:
            return A.gpu()->cublas_->dot(A.row_, (double*)A.getDataPtr(0, cA), 1, (double*)B.getDataPtr(0, cA), 1);
            break;
        case DataType::HALF:
            return A.gpu()->cublas_->dot(A.row_, (half*)A.getDataPtr(0, cA), 1, (half*)B.getDataPtr(0, cA), 1);
            break;
        }
    }
    else if (A.isHip())
    {
        return A.gpu()->rocblas_->dot(A.row_, (float*)A.getDataPtr(0, cA), 1, (float*)B.getDataPtr(0, cA), 1);
    }
    else
    {
        return Cblas::dot(A.row_, (float*)A.getDataPtr(0, cA), 1, (float*)B.getDataPtr(0, cA), 1);
    }
}

//选择部分点乘
float Matrix::dotPart(int size, const Matrix& A, void* a, int cA, void* b, int cB)
{
    if (A.isCuda())
    {
        switch (A.getDataType())
        {
        case DataType::FLOAT:
            return A.gpu()->cublas_->dot(size, (float*)a, cA, (float*)b, cB);
            break;
        case DataType::DOUBLE:
            return A.gpu()->cublas_->dot(size, (double*)a, cA, (double*)b, cB);
            break;
        case DataType::HALF:
            return A.gpu()->cublas_->dot(size, (half*)a, cA, (half*)b, cB);
            break;
        }
    }
    else if (A.isHip())
    {
        return A.gpu()->rocblas_->dot(size, (float*)a, cA, (float*)b, cB);
    }
    else
    {
        return Cblas::dot(size, (float*)a, cA, (float*)b, cB);
    }
}

//点乘，即所有元素平方和
float Matrix::dotSelf() const
{
    if (isCuda())
    {
        switch (getDataType())
        {
        case DataType::FLOAT:
            return gpu()->cublas_->dot(data_size_, dataf(), 1, dataf(), 1);
            break;
        case DataType::DOUBLE:
            return gpu()->cublas_->dot(data_size_, datad(), 1, datad(), 1);
            break;
        case DataType::HALF:
            return gpu()->cublas_->dot(data_size_, datah(), 1, datah(), 1);
            break;
        }
    }
    else if (isHip())
    {
        return gpu()->rocblas_->dot(data_size_, dataf(), 1, dataf(), 1);
    }
    else
    {
        switch (getDataType())
        {
        case DataType::FLOAT:
            return Cblas::dot(data_size_, dataf(), 1, dataf(), 1);
            break;
        case DataType::DOUBLE:
            return Cblas::dot(data_size_, datad(), 1, datad(), 1);
            break;
        case DataType::HALF:
        {
            float r = 0;
            for (int i = 0; i < data_size_; i++)
            {
                r += getData(i) * getData(i);
            }
            return r;
            break;
        }
        }
    }
}

//取符号
void Matrix::sign(Matrix& A, Matrix& R, float v, float section)
{
    if (A.isCuda())
    {
        cuda_sign(A.getDataTypeByInt(), A.data(), R.data(), A.data_size_, v, section);
    }
    else if (A.isHip())
    {
        hip_sign(A.getDataTypeByInt(), A.data(), R.data(), A.data_size_, v, section);
    }
    else
    {
        for (int i = 0; i < A.data_size_; i++)
        {
            if (A.getData(i) > section)
            {
                R.setData(i, 1);
                continue;
            }
            if (A.getData(i) < -section)
            {
                R.setData(i, -1);
                continue;
            }
            R.setData(i, 0);
        }
    }
}

void Matrix::importData(float* v, int64_t n)
{
    MatrixData::copy(API_UNKNOWN, v, getApiType(), data(), std::min(n, data_size_), getDataType());
    //for (int i = 0; i < n; i++)
    //{
    //    LOG("{}, ", v[i]);
    //}
}

void Matrix::exportData(float* v, int64_t n)
{
    MatrixData::copy(getApiType(), data(), API_UNKNOWN, v, std::min(n, data_size_), getDataType());
}

//求倒数，a = scale ./ a
void Matrix::reciprocal(float scale)
{
    if (isCuda())
    {
        cuda_reciprocal(getDataTypeByInt(), data(), data(), data_size_, scale, 0.0);
    }
    else if (isHip())
    {
        hip_reciprocal(getDataTypeByInt(), data(), data(), data_size_, scale, 0.0);
    }
    else
    {
        for (int i = 0; i < data_size_; i++)
        {
            setData(i, scale / getData(i));
        }
    }
}

//加上一个数字，a = v + scale .* a;
void Matrix::addNumber(float v, float scale)
{
    addNumber(*this, *this, v, scale);
}

void Matrix::addNumber(const Matrix& A, Matrix& R, float v, float scale)
{
    assert(checkMatrixDevice({ &A, &R }));
    if (A.isCuda())
    {
        cuda_addnumber(A.getDataTypeByInt(), A.data(), R.data(), A.data_size_, v, scale);
    }
    else if (A.isHip())
    {
        hip_addnumber(A.getDataTypeByInt(), A.data(), R.data(), A.data_size_, v, scale);
    }
    else
    {
        for (int i = 0; i < A.data_size_; i++)
        {
            R.setData(i, v + scale * A.getData(i));
        }
    }
}

void Matrix::addNumberCol(float v, float scale, int c)
{
    if (isCuda())
    {
        cuda_addnumber(getDataTypeByInt(), getDataPtr(0, c), getDataPtr(0, c), row_, v, scale);
    }
    else if (isHip())
    {
        hip_addnumber(getDataTypeByInt(), getDataPtr(0, c), getDataPtr(0, c), row_, v, scale);
    }
    else
    {
        for (int i = 0; i < row_; i++)
        {
            setData(i, c, v + scale * getData(i, c));
        }
    }
}

void Matrix::elementPow(const Matrix& A, Matrix& R, float e, float bias)
{
    assert(checkMatrixDevice({ &A, &R }));
    if (A.isCuda())
    {
        cuda_pow(A.getDataTypeByInt(), A.data(), R.data(), A.data_size_, e, bias);
    }
    else if (A.isHip())
    {
        hip_pow(A.getDataTypeByInt(), A.data(), R.data(), A.data_size_, e, bias);
    }
    else
    {
        for (int i = 0; i < A.data_size_; i++)
        {
            R.setData(i, pow(bias + A.getData(i), e));
        }
    }
}

void Matrix::elementDiv(const Matrix& A, const Matrix& B, Matrix& R, float a, float b, float scale)
{
    assert(checkMatrixDevice({ &A, &B, &R }));
    if (A.isCuda())
    {
        cuda_div(A.getDataTypeByInt(), A.data(), B.data(), R.data(), A.data_size_, a, b, scale);
    }
    else if (A.isHip())
    {
        hip_div(A.getDataTypeByInt(), A.data(), B.data(), R.data(), A.data_size_, a, b, scale);
    }
    else
    {
        for (int i = 0; i < A.data_size_; i++)
        {
            R.setData(i, scale * (A.getData(i) + a) / (B.getData(i) + b));
        }
    }
}

void Matrix::crossEntropy(const Matrix& A, const Matrix& Y, Matrix& R, float a, float scale)
{
    assert(checkMatrixDevice({ &A, &Y, &R }));
    if (A.isCuda())
    {
        cuda_cross_entropy(A.getDataTypeByInt(), A.data(), Y.data(), R.data(), A.data_size_, a, scale);
    }
    else if (A.isHip())
    {
        hip_cross_entropy(A.getDataTypeByInt(), A.data(), Y.data(), R.data(), A.data_size_, a, scale);
    }
    else
    {
        for (int i = 0; i < A.data_size_; i++)
        {
            R.setData(i, Y.getData(i) * log(A.getData(i) + a));
            R.setData(i, -R.getData(i) * scale);
        }
    }
}

void Matrix::crossEntropy2(const Matrix& A, const Matrix& Y, Matrix& R, float a, float scale)
{
    assert(checkMatrixDevice({ &A, &Y, &R }));
    if (A.isCuda())
    {
        cuda_cross_entropy2(A.getDataTypeByInt(), A.data(), Y.data(), R.data(), A.data_size_, a, scale);
    }
    else if (A.isHip())
    {
        hip_cross_entropy2(A.getDataTypeByInt(), A.data(), Y.data(), R.data(), A.data_size_, a, scale);
    }
    else
    {
        for (int i = 0; i < A.data_size_; i++)
        {
            R.setData(i, Y.getData(i) * log(A.getData(i) + a) + (1 - Y.getData(i)) * log(1 - A.getData(i) + a));
            R.setData(i, -R.getData(i) * scale);
        }
    }
}

bool Matrix::checkMatrixDevice(const std::vector<const Matrix*>& v)
{
    if (v.size() <= 1)
    {
        return true;
    }
    else
    {
        const GpuControl* gpu = nullptr;
        for (int i = 0; i < v.size(); i++)
        {
            if (v[i]->data_size_ > 0)
            {
                gpu = v[i]->gpu();
                break;
            }
        }
        for (int i = 0; i < v.size(); i++)
        {
            if (v[i]->data_size_ > 0 && gpu != v[i]->gpu())
            {
                LOG_ERR("Matrices are not in the same device!\n");
                return false;
                break;
            }
        }
        return true;
    }
}

void Matrix::message(const std::string& info) const
{
    LOG("{}:", info);
    if (isCuda())
    {
        LOG(" GPU({}, CUDA),", gpu()->getDeviceID());
    }
    else if (isHip())
    {
        LOG(" GPU({}, HIP),", gpu()->getDeviceID());
        //int w, h, c, n, t;
        //miopenDataType_t tt;
        //miopenGet4dTensorDescriptor(miopen_desc(), &tt, &n, &c, &h, &w, &t, nullptr, nullptr, nullptr);
        //LOG(stdout, " dim = {} {} {} {} {} {}\n", w, h, c, n, t, int(tt));
    }
    else
    {
        LOG(" CPU,");
    }
    LOG(" Dtype({}),", int(getDataType()));
    LOG(" Dim{}({})\n", dim_, getDataSize());
    LOG("  L1 = {}, L2 = {}\n", sumAbs() / getDataSize(), dotSelf() / getDataSize());
    return;
    cudnnDataType_t t;
    int n;
    int d1[8];
    int s1[8];
    cudnnGetTensorNdDescriptor(cudnn_desc(), 8, &t, &n, d1, s1);
    LOG(" cudnn desc: {} {}\n", int(t), n);
    for (int i = 0; i < 8; i++)
    {
        LOG(" {} {}\n", d1[i], s1[i]);
    }
    //cudnnSetTensorNdDescriptor(tensor_desc_, t, n, d1, s1);
}

std::string Matrix::sizeMessage(int include_batch) const
{
    if (include_batch)
    {
        return fmt1::format("Matrix({}, {}, {}, {})", width_, height_, channel_, number_, getDataPtr());
    }
    else
    {
        return fmt1::format("Matrix({}, {}, {}, batch)", width_, height_, channel_);
    }
}
}    // namespace cccc