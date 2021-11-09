#include "MatrixEx.h"
#include "Random.h"
#include "VectorMath.h"
#include "Log.h"
#include <cassert>

#if !defined(M_PI)
#define M_PI 3.1415926535897932
#endif

namespace cccc
{

//若bias某一维度为1，则视为对X的对应维度加上同样的值，cudnn的文档中指出最高支持到5维
//此处仅用于两种情况：1 - bias仅有channel不为1，2 - bias的number为1，注意1是2的一个特例，其他情况不一定能得到正确结果
void MatrixEx::addBias(const Matrix& X, const Matrix& bias, Matrix& Y, realc a /*= 1*/, realc b /*= 1*/)
{
    assert(checkMatrixDevice({ &X, &bias, &Y }));
    assert(bias.getDataSize() == bias.channel_ || bias.number_ == 1);
    //if (X != &Y)
    {
        copyData(X, Y);
    }
    if (X.inGPU())
    {
        auto cuda = X.cuda();
        if (bias.getDimSize() <= 5)
        {
            cudnnAddTensor(cuda->cudnn_handle_, &a, bias.tensor_desc(), bias.data(), &b, Y.tensor_desc(), Y.data());
        }
        else
        {
            CudaControl::setTensorDesc4D(cuda->tensor_desc_, bias.width_, bias.height_, bias.channel_, bias.number_);
            CudaControl::setTensorDesc4D(cuda->tensor_desc2_, Y.width_, Y.height_, Y.channel_, Y.number_);
            cudnnAddTensor(cuda->cudnn_handle_, &a, cuda->tensor_desc_, bias.data(), &b, cuda->tensor_desc2_, Y.data());
        }
    }
    else
    {
        if (bias.getDataSize() == bias.channel_)
        {
            for (int i = 0; i < Y.data_size_; i++)
            {
                int c = i % Y.row_ / (Y.row_ / Y.channel_);
                Y.data()[i] += bias.data()[c];
            }
        }
        else
        {
            for (int i = 0; i < Y.data_size_; i++)
            {
                int c = i % bias.getDataSize();
                Y.data()[i] += bias.data()[c];
            }
        }
    }
}

void MatrixEx::addBiasBackward(Matrix& X, Matrix& bias, const Matrix& Y, realc a /*= 1*/, realc b /*= 1*/)
{
    assert(checkMatrixDevice({ &X, &bias, &Y }));
    if (X.needReverse())
    {
        Matrix::add(X.d(), Y.d(), X.d(), a, 1, 0);
    }
    if (bias.needReverse())
    {
        if (X.inGPU())
        {
            //用卷积反向代替一般反向，此处待验证
            auto cuda = Y.cuda();
            cudnnConvolutionBackwardBias(cuda->cudnn_handle_, &const_real_1, Y.tensor_desc(), Y.d().data(), &b, bias.tensor_desc(), bias.d().data());
        }
        else
        {
            if (bias.d().data())
            {
                //bias.DMatrix().scale(r);
                //这个就是对对应的dR求和
                for (int n = 0; n < Y.number_; n++)
                {
                    for (int c = 0; c < Y.channel_; c++)
                    {
                        bias.d().getData(0, 0, c, 0) += b * VectorMath::sum(Y.d().getDataPointer(0, 0, c, n), Y.width_ * Y.height_);
                    }
                }
            }
        }
    }
}

void MatrixEx::concatByChannel(const std::vector<MatrixSP>& X_vector, Matrix& Y)
{
    for (int n = 0; n < Y.number_; n++)
    {
        int c_off = 0;
        for (int i = 0; i < X_vector.size(); i++)
        {
            auto& tmp = *X_vector[i];
            copyDataPointer(tmp, tmp.getDataPointer(0, 0, 0, n), Y, Y.getDataPointer(0, 0, c_off, n), tmp.row_);
            c_off += tmp.channel_;
        }
    }
}

void MatrixEx::concatByChannelBackward(std::vector<MatrixSP>& X_vector, const Matrix& Y)
{
    //此处可能应该是考虑求和
    Matrix dy(Y.getDeviceType());
    Matrix dx(Y.getDeviceType());
    for (int n = 0; n < Y.number_; n++)
    {
        int c_off = 0;
        for (int i = 0; i < X_vector.size(); i++)
        {
            if (X_vector[i]->needReverse())
            {
                dy.shareData(Y.d(), 0, 0, c_off, n);
                dy.resize(X_vector[i]->row_, 1);
                dx.shareData(X_vector[i]->d(), 0, 0, 0, n);
                dx.resize(X_vector[i]->row_, 1);
                Matrix::add(dy, dx, dx, 1, X_vector[i]->keepWeight(), 0);    //这里写错了
                //copyDataPointer(Y, Y.d().getDataPointer(0, 0, c_off, n), tmp, tmp.getDataPointer(0, 0, 0, n), tmp.row_);
            }
            c_off += X_vector[i]->channel_;
        }
    }
}

void MatrixEx::splitByChannel(const Matrix& X, std::vector<Matrix>& Y_vector)
{
    for (int n = 0; n < X.number_; n++)
    {
        int c_off = 0;
        for (int i = 0; i < Y_vector.size(); i++)
        {
            auto& tmp = Y_vector[i];
            copyDataPointer(X, X.getDataPointer(0, 0, c_off, n), tmp, tmp.getDataPointer(0, 0, 0, n), tmp.row_);
            c_off += tmp.channel_;
        }
    }
}

//初始化激活需要的缓冲区
void MatrixEx::activeBufferInit(const Matrix& X, ActiveFunctionType af, std::vector<int>& int_vector, std::vector<real>& real_vector, std::vector<Matrix>& matrix_vector)
{
}

//正向激活，依据X计算A
//此处我们定义激活操作为输入和输出矩阵（或张量）的维度完全相同
//传附加参数的时候使用了C++11的初始化列表，因此效率可能较低，实际上如不考虑效率可以代替基本激活函数
//调用时请自己保证参数数量的正确性！
//vector参数的含义请参考Activer.cpp中的注释，以及上面函数中matrix向量的含义
void MatrixEx::activeForward(const Matrix& X, Matrix& Y, ActiveFunctionType af,
    std::vector<int>& int_vector, std::vector<real>& real_vector, std::vector<Matrix>& matrix_vector, real a /*= 1*/, real r /*= 0*/)
{
    assert(checkMatrixDevice({ &X, &Y }));
    auto nan = CUDNN_NOT_PROPAGATE_NAN;
    auto cuda = X.cuda();
    switch (af)
    {
    case ACTIVE_FUNCTION_NONE:
        Matrix::copyData(X, Y);
        break;
    case ACTIVE_FUNCTION_SIGMOID:
    case ACTIVE_FUNCTION_SIGMOID_CE:
        if (X.inGPU())
        {
            CudaControl::setActivationDesc(cuda->activation_desc_, CUDNN_ACTIVATION_SIGMOID, 1);
            cudnnActivationForward(cuda->cudnn_handle_, cuda->activation_desc_,
                &a, X.tensor_desc(), X.data(), &r, Y.tensor_desc(), Y.data());
        }
        else
        {
            VectorMath::sigmoid_v(X.data(), Y.data(), Y.data_size_, a, r);
        }
        break;
    case ACTIVE_FUNCTION_RELU:
        if (X.inGPU())
        {
            CudaControl::setActivationDesc(cuda->activation_desc_, CUDNN_ACTIVATION_RELU, 1);
            cudnnActivationForward(cuda->cudnn_handle_, cuda->activation_desc_,
                &a, X.tensor_desc(), X.data(), &r, Y.tensor_desc(), Y.data());
        }
        else
        {
            VectorMath::relu_v(X.data(), Y.data(), Y.data_size_, a, r);
        }
        break;
    case ACTIVE_FUNCTION_TANH:
        if (X.inGPU())
        {
            CudaControl::setActivationDesc(cuda->activation_desc_, CUDNN_ACTIVATION_TANH, 1);
            cudnnActivationForward(cuda->cudnn_handle_, cuda->activation_desc_,
                &a, X.tensor_desc(), X.data(), &r, Y.tensor_desc(), Y.data());
        }
        else
        {
            VectorMath::tanh_v(X.data(), Y.data(), Y.data_size_, a, r);
        }
        break;
    case ACTIVE_FUNCTION_SOFTMAX:
    case ACTIVE_FUNCTION_SOFTMAX_CE:
        if (X.inGPU())
        {
            CudaControl::setTensorDesc4D(cuda->tensor_desc_, 1, 1, X.row_, X.number_);
            cudnnSoftmaxForward(cuda->cudnn_handle_, CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_INSTANCE,
                &a, cuda->tensor_desc_, X.data(), &r, cuda->tensor_desc_, Y.data());
        }
        else
        {
            //因为数值问题，可能需要减去每列最大值
            MatrixEx::copyData(X, Y);
            for (int i = 0; i < Y.number_; i++)
            {
                VectorMath::minus_max(Y.getDataPointer(0, i), Y.row_);
            }
            VectorMath::exp_v(Y.data(), Y.data(), Y.data_size_);
            for (int i = 0; i < Y.number_; i++)
            {
                real sum = Y.sumAbsCol(i);
                if (sum == 0)
                {
                    continue;
                }
                Y.scaleCol(1 / sum, i);
            }
        }
        break;
    case ACTIVE_FUNCTION_SOFTMAX_FAST:
    case ACTIVE_FUNCTION_SOFTMAX_FAST_CE:
        if (X.inGPU())
        {
            CudaControl::setTensorDesc4D(cuda->tensor_desc_, 1, 1, X.row_, X.number_);
            cudnnSoftmaxForward(cuda->cudnn_handle_, CUDNN_SOFTMAX_FAST, CUDNN_SOFTMAX_MODE_INSTANCE,
                &a, cuda->tensor_desc_, X.data(), &r, cuda->tensor_desc_, Y.data());
        }
        else
        {
            VectorMath::exp_v(X.data(), Y.data(), Y.data_size_, a, r);
            for (int i = 0; i < Y.number_; i++)
            {
                real sum = Y.sumAbsCol(i);
                if (sum == 0)
                {
                    continue;
                }
                Y.scaleCol(1 / sum, i);
            }
        }
        break;
    case ACTIVE_FUNCTION_SOFTMAX_LOG:
        if (X.inGPU())
        {
            CudaControl::setTensorDesc4D(cuda->tensor_desc_, 1, 1, X.row_, X.number_);
            cudnnSoftmaxForward(cuda->cudnn_handle_, CUDNN_SOFTMAX_LOG, CUDNN_SOFTMAX_MODE_INSTANCE,
                &a, cuda->tensor_desc_, X.data(), &r, cuda->tensor_desc_, Y.data());
        }
        else
        {
            activeForward(X, Y, ACTIVE_FUNCTION_SOFTMAX, int_vector, real_vector, matrix_vector);
            VectorMath::log_v(Y.data(), Y.data(), Y.data_size_, a, r);
        }
        break;
    case ACTIVE_FUNCTION_ABSMAX:
        //计算时尽量不要使用，只用在验证时
        if (Y.data_size_ <= 0)
        {
            return;
        }
        if (X.inGPU())
        {
            Matrix T(Y.row_, Y.number_, DeviceType::CPU);
            T.scale(0);
            for (int i_group = 0; i_group < Y.number_; i_group++)
            {
                int index = X.indexColMaxAbs(i_group);
                T.getData(index, i_group) = 1;
            }
            Matrix::copyData(T, Y);
        }
        else
        {
            Y.scale(0);
            for (int i_group = 0; i_group < Y.number_; i_group++)
            {
                int index = X.indexColMaxAbs(i_group);
                Y.getData(index, i_group) = 1;
            }
        }
        break;
    default:
        LOG(stderr, "ACTIVE forward not right {}!\n", af);
        break;
    }
}

//参考activeForward2
//反向激活，依据X，A，dA计算dX
//softmax的cpu部分貌似不支持a，b
//这里的系数应该是不对,unfinished
void MatrixEx::activeBackward(Matrix& X, const Matrix& Y, ActiveFunctionType af,
    std::vector<int>& int_vector, std::vector<real>& real_vector, std::vector<Matrix>& matrix_vector, real a /*= 1*/, real r /*= 0*/)
{
    assert(checkMatrixDevice({ &X, &Y }));
    auto nan = CUDNN_NOT_PROPAGATE_NAN;
    auto cuda = Y.cuda();
    switch (af)
    {
    case ACTIVE_FUNCTION_NONE:
    case ACTIVE_FUNCTION_SIGMOID_CE:
    case ACTIVE_FUNCTION_SOFTMAX_CE:
    case ACTIVE_FUNCTION_SOFTMAX_FAST_CE:
        Matrix::add(Y.d(), X.d(), X.d(), a, r, 0);
        break;
    case ACTIVE_FUNCTION_SIGMOID:
        if (X.inGPU())
        {
            CudaControl::setActivationDesc(cuda->activation_desc_, CUDNN_ACTIVATION_SIGMOID, 1);
            cudnnActivationBackward(cuda->cudnn_handle_, cuda->activation_desc_, &a, Y.tensor_desc(), Y.data(),
                Y.tensor_desc(), Y.d().data(), X.tensor_desc(), X.data(), &r, X.tensor_desc(), X.d().data());
        }
        else
        {
            VectorMath::sigmoid_vb(Y.data(), Y.d().data(), X.data(), X.d().data(), X.data_size_, a, r);
        }
        break;
    case ACTIVE_FUNCTION_RELU:
        if (X.inGPU())
        {
            CudaControl::setActivationDesc(cuda->activation_desc_, CUDNN_ACTIVATION_RELU, 1);
            cudnnActivationBackward(cuda->cudnn_handle_, cuda->activation_desc_, &a, Y.tensor_desc(), Y.data(),
                Y.tensor_desc(), Y.d().data(), X.tensor_desc(), X.data(), &r, X.tensor_desc(), X.d().data());
        }
        else
        {
            VectorMath::relu_vb(Y.data(), Y.d().data(), X.data(), X.d().data(), X.data_size_, a, r);
        }
        break;
    case ACTIVE_FUNCTION_TANH:
        //两者结果在1e-10的精度有区别
        if (X.inGPU())
        {
            CudaControl::setActivationDesc(cuda->activation_desc_, CUDNN_ACTIVATION_TANH, 1);
            cudnnActivationBackward(cuda->cudnn_handle_, cuda->activation_desc_, &a, Y.tensor_desc(), Y.data(),
                Y.tensor_desc(), Y.d().data(), X.tensor_desc(), X.data(), &r, X.tensor_desc(), X.d().data());
        }
        else
        {
            VectorMath::tanh_vb(Y.data(), Y.d().data(), X.data(), X.d().data(), X.data_size_, a, r);
        }
        break;
    case ACTIVE_FUNCTION_SOFTMAX:
    case ACTIVE_FUNCTION_SOFTMAX_FAST:
        if (X.inGPU())
        {
            auto softmax_flag = CUDNN_SOFTMAX_ACCURATE;
            if (af == ACTIVE_FUNCTION_SOFTMAX_FAST)
            {
                softmax_flag = CUDNN_SOFTMAX_FAST;
            }
            CudaControl::setTensorDesc4D(cuda->tensor_desc_, 1, 1, X.row_, X.number_);
            auto s = cudnnSoftmaxBackward(cuda->cudnn_handle_, softmax_flag, CUDNN_SOFTMAX_MODE_INSTANCE,
                &a, cuda->tensor_desc_, Y.data(), cuda->tensor_desc_, Y.d().data(), &r, cuda->tensor_desc_, X.d().data());
        }
        else
        {
            for (int i = 0; i < X.number_; i++)
            {
                real v = MatrixEx::dotCol(Y, i, Y.d(), i);
                VectorMath::softmax_vb_sub(Y.getDataPointer(0, i), Y.d().getDataPointer(0, i), v, X.d().getDataPointer(0, i), X.row_);
            }
        }
        break;
    case ACTIVE_FUNCTION_SOFTMAX_LOG:
        if (X.inGPU())
        {
            CudaControl::setTensorDesc4D(cuda->tensor_desc_, 1, 1, X.row_, X.number_);
            auto s = cudnnSoftmaxBackward(cuda->cudnn_handle_, CUDNN_SOFTMAX_LOG, CUDNN_SOFTMAX_MODE_INSTANCE,
                &a, cuda->tensor_desc_, Y.data(), cuda->tensor_desc_, Y.d().data(), &r, cuda->tensor_desc_, X.d().data());
        }
        else
        {
            for (int i = 0; i < X.number_; i++)
            {
                real v = 0;
                for (int j = 0; j < X.row_; j++)
                {
                    v += Y.d().getData(i, j);
                }
                VectorMath::softmaxlog_vb_sub(Y.getDataPointer(0, i), Y.d().getDataPointer(0, i), v, X.d().getDataPointer(0, i), X.row_);
            }
        }
        break;
    case ACTIVE_FUNCTION_ABSMAX:
        //似乎应该是返回一个常数矩阵，若考虑效率应当留空此处在外部处理
        //dX.initData(1);
        LOG(stderr, "Unsupported backward of FINDMAX!\n");
        break;
    default:
        LOG(stderr, "ACTIVE backward not right {}!\n", af);
        break;
    }
}

void MatrixEx::activeForwardSimple(const Matrix& X, Matrix& Y, ActiveFunctionType af, real a /*= 1*/, real r /*= 0*/)
{
    std::vector<int> int_vector;
    std::vector<real> real_vector;
    std::vector<Matrix> matrix_vector;
    activeForward(X, Y, af, int_vector, real_vector, matrix_vector, a, r);
}

void MatrixEx::activeBackwardSimple(Matrix& X, const Matrix& Y, ActiveFunctionType af, real a /*= 1*/, real r /*= 0*/)
{
    std::vector<int> int_vector;
    std::vector<real> real_vector;
    std::vector<Matrix> matrix_vector;
    activeBackward(X, Y, af, int_vector, real_vector, matrix_vector, a, r);
}

//池化
//gpu部分，平均模式下对padding的支持目前还有问题
void MatrixEx::poolingForward(const Matrix& X, Matrix& Y, PoolingType pooling_type, int reverse,
    const std::vector<int>& window, const std::vector<int>& stride, const std::vector<int>& padding,
    realc a /*= 1*/, realc r /*= 0*/)
{
    assert(checkMatrixDevice({ &X, &Y }));
    assert(window.size() >= 2 && stride.size() >= 2 && padding.size() >= 2);
    auto cuda = X.cuda();
    if (Y.inGPU())
    {
        if (stride.size() == 2)
        {
            cudnnSetPooling2dDescriptor(cuda->pooling_desc_, cudnnPoolingMode_t(pooling_type), CUDNN_NOT_PROPAGATE_NAN,
                window[1], window[0], padding[1], padding[0], stride[1], stride[0]);
        }
        else
        {
            auto wr = window;
            auto pr = padding;
            auto sr = stride;
            std::reverse(wr.begin(), wr.end());
            std::reverse(pr.begin(), pr.end());
            std::reverse(sr.begin(), sr.end());
            cudnnSetPoolingNdDescriptor(cuda->pooling_desc_, cudnnPoolingMode_t(pooling_type), CUDNN_NOT_PROPAGATE_NAN,
                window.size(), wr.data(), pr.data(), sr.data());
        }
        cudnnStatus_t s;
        if (reverse == 0)
        {
            s = cudnnPoolingForward(cuda->cudnn_handle_, cuda->pooling_desc_, &a, X.tensor_desc(), X.data(), &r, Y.tensor_desc(), Y.data());
        }
        else
        {
            s = cudnnPoolingBackward(cuda->cudnn_handle_, cuda->pooling_desc_,
                &a, X.tensor_desc(), X.data(), X.tensor_desc(), X.data(), Y.tensor_desc(), Y.data(), &r, Y.tensor_desc(), Y.data());
            Y.scale(VectorMath::multiply(window));
        }
        if (s)
        {
            LOG(stderr, "POOL forward error {}\n", s);
        }
    }
    else
    {
        Y.scale(r);
        for (int p = 0; p < Y.number_ * Y.channel_; p++)
        {
            for (int wA = 0; wA < Y.width_; wA++)
            {
                for (int hA = 0; hA < Y.height_; hA++)
                {
                    real v = 0;
                    if (pooling_type == POOLING_MAX)
                    {
                        v = -REAL_MAX;
                    }
                    int n = 0;
                    int wX0 = wA * stride[0] - padding[0];
                    int hX0 = hA * stride[1] - padding[1];
                    for (int wX = wX0; wX < std::min(X.width_, wX0 + window[0]); wX++)
                    {
                        for (int hX = hX0; hX < std::min(X.height_, hX0 + window[1]); hX++)
                        {
                            if (pooling_type == POOLING_AVERAGE_PADDING || pooling_type == POOLING_AVERAGE_NOPADDING)
                            {
                                if (X.haveData(wX, hX, p, 0))
                                {
                                    v += X.getData(wX, hX, p, 0);
                                }
                                n++;
                            }
                            else if (pooling_type == POOLING_MAX)
                            {
                                if (X.haveData(wX, hX, p, 0))
                                {
                                    auto x = X.getData(wX, hX, p, 0);
                                    if (x > v)
                                    {
                                        v = x;
                                    }
                                }
                            }
                        }
                    }
                    if (pooling_type == POOLING_AVERAGE_PADDING)
                    {
                        v /= window[0] * window[1];
                    }
                    else if (pooling_type == POOLING_AVERAGE_NOPADDING)
                    {
                        v /= n;
                    }
                    Y.getData(wA, hA, p, 0) += a * v;    // +r * Y.getData(wA, hA, p, 0);
                }
            }
        }
    }
}

//使用cpu时利用了record -- 取消，直接计算
void MatrixEx::poolingBackward(Matrix& X, const Matrix& Y, PoolingType pooling_type, int reverse,
    const std::vector<int>& window, const std::vector<int>& stride, const std::vector<int>& padding,
    realc a /*= 1*/, realc r /*= 0*/)
{
    assert(checkMatrixDevice({ &X, &Y }));
    assert(window.size() >= 2 && stride.size() >= 2 && padding.size() >= 2);
    auto cuda = Y.cuda();
    if (X.inGPU())
    {
        if (window.size() == 2)
        {
            //这个怎么看都快不了
            cudnnSetPooling2dDescriptor(cuda->pooling_desc_, cudnnPoolingMode_t(pooling_type), CUDNN_NOT_PROPAGATE_NAN,
                window[1], window[0], padding[1], padding[0], stride[1], stride[0]);
        }
        else
        {
            auto wr = window;
            auto pr = padding;
            auto sr = stride;
            std::reverse(wr.begin(), wr.end());
            std::reverse(pr.begin(), pr.end());
            std::reverse(sr.begin(), sr.end());
            cudnnSetPoolingNdDescriptor(cuda->pooling_desc_, cudnnPoolingMode_t(pooling_type), CUDNN_NOT_PROPAGATE_NAN,
                window.size(), wr.data(), pr.data(), sr.data());
        }
        cudnnStatus_t s;
        if (reverse == 0)
        {
            s = cudnnPoolingBackward(cuda->cudnn_handle_, cuda->pooling_desc_,
                &a, Y.tensor_desc(), Y.data(), Y.tensor_desc(), Y.d().data(), X.tensor_desc(), X.data(), &r, X.tensor_desc(), X.d().data());
        }
        else
        {
            s = cudnnPoolingForward(cuda->cudnn_handle_, cuda->pooling_desc_, &a, Y.tensor_desc(), Y.d().data(), &r, X.tensor_desc(), X.d().data());
            X.d().scale(1.0 / VectorMath::multiply(window));
        }
        if (s)
        {
            LOG(stderr, "POOL backward error {}\n", s);
        }
    }
    else
    {
        X.d().scale(r);
        for (int p = 0; p < Y.number_ * Y.channel_; p++)
        {
            for (int wdA = 0; wdA < Y.width_; wdA++)
            {
                for (int hdA = 0; hdA < Y.height_; hdA++)
                {
                    int wdX0 = wdA * stride[0] - padding[0];
                    int hdX0 = hdA * stride[1] - padding[1];
                    if (pooling_type == POOLING_MAX)
                    {
                        real max_v = -REAL_MAX;
                        real* max_p = nullptr;
                        for (int wdX = wdX0; wdX < std::min(X.width_, wdX0 + window[0]); wdX++)
                        {
                            for (int hdX = hdX0; hdX < std::min(X.height_, hdX0 + window[1]); hdX++)
                            {
                                if (X.haveData(wdX, hdX, p, 0))
                                {
                                    real v = X.getData(wdX, hdX, p, 0);
                                    if (v > max_v)
                                    {
                                        max_v = v;
                                        max_p = &X.d().getData(wdX, hdX, p, 0);
                                    }
                                }
                            }
                        }
                        if (max_p)
                        {
                            *max_p += a * Y.d().getData(wdA, hdA, p, 0);
                        }
                    }
                    else
                    {
                        int n;
                        if (pooling_type == POOLING_AVERAGE_NOPADDING)
                        {
                            n = std::min(window[0], X.width_ - wdA * stride[0]) * std::min(window[1], X.height_ - hdA * stride[1]);
                        }
                        else
                        {
                            n = window[0] * window[1];
                        }
                        real v = Y.d().getData(wdA, hdA, p, 0) / n;
                        for (int wdX = wdX0; wdX < std::min(X.width_, wdX0 + window[0]); wdX++)
                        {
                            for (int hdX = hdX0; hdX < std::min(X.height_, hdX0 + window[1]); hdX++)
                            {
                                if (X.haveData(wdX, hdX, p, 0))
                                {
                                    X.d().getData(wdX, hdX, p, 0) += a * v;    // +r * X.d().getData(wdX, hdX, p, 0);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

//卷积就是有连接就算
//这个循环顺序次数最少
#define CONV_OPERATION1(X, W, Y, pw, ph, sw, sh, DO_SOMETHING) \
    do { \
        for (int wY = 0; wY < Y.width_; wY++) \
            for (int hY = 0; hY < Y.height_; hY++) \
                for (int wW = 0; wW < W.width_; wW++) \
                    for (int hW = 0; hW < W.height_; hW++) \
                    { \
                        int wX = wY + wW - pw; \
                        int hX = hY + hW - ph; \
                        if (wX >= 0 && wX < X.width_ && hX >= 0 && hX < X.height_) \
                        { \
                            DO_SOMETHING \
                        } \
                    } \
    } while (0)

//前向卷积
//从外部引入辅助空间目的是降低初始化的次数
//当使用CUDA计算时，不需要辅助转换的整数数组，这时该数组会被初始化为两个元素，分别为算法和所需的工作空间大小，在首次计算的时候完成
//CPU模式仅支持stride为1，padding为0的二维卷积
//methods须至少3个: 算法，类型，组数，若考虑反向则应9个
void MatrixEx::convolutionForward(const Matrix& X, const Matrix& W, Matrix& Y, std::vector<int>& methods, std::vector<Matrix>& workspaces,
    const std::vector<int>& stride, const std::vector<int>& padding, realc a /*= 1*/, realc r /*= 0*/)
{
    assert(checkMatrixDevice({ &X, &W, &Y }));
    assert(stride.size() >= 2 && padding.size() >= 2);
    if (methods.size() < 3)
    {
        methods.resize(3, -1);
    }
    if (workspaces.size() < 2)
    {
        workspaces.resize(2);
    }
    auto cuda = X.cuda();
    if (Y.inGPU())
    {
        int scd = -1, sfd = -1;
        if (stride.size() == 2)
        {
            scd = cudnnSetConvolution2dDescriptor(cuda->convolution_desc_, padding[1], padding[0], stride[1], stride[0], 1, 1, CUDNN_CROSS_CORRELATION, MYCUDNN_DATA_REAL);
            sfd = cudnnSetFilter4dDescriptor(cuda->filter_desc_, MYCUDNN_DATA_REAL, CUDNN_TENSOR_NCHW, W.number_, W.channel_, W.height_, W.width_);
        }
        else
        {
            //这里有可能需要反序vector，待测试
            std::vector<int> dilation(padding.size(), 1);
            auto pr = padding;
            auto sr = stride;
            std::reverse(pr.begin(), pr.end());
            std::reverse(sr.begin(), sr.end());
            scd = cudnnSetConvolutionNdDescriptor(cuda->convolution_desc_, padding.size(), pr.data(), sr.data(), dilation.data(), CUDNN_CROSS_CORRELATION, MYCUDNN_DATA_REAL);
            auto w_dim = W.getDim();
            std::reverse(w_dim.begin(), w_dim.end());
            sfd = cudnnSetFilterNdDescriptor(cuda->filter_desc_, MYCUDNN_DATA_REAL, CUDNN_TENSOR_NCHW, W.getDimSize(), w_dim.data());
        }
        cudnnSetConvolutionMathType(cuda->convolution_desc_, CUDNN_TENSOR_OP_MATH);
        //寻找最快的算法
        if (methods[2] < X.number_)
        {
            int n;
            cudnnConvolutionFwdAlgoPerf_t cfap[conv_method_count];
            auto t = cudnnFindConvolutionForwardAlgorithm(cuda->cudnn_handle_, X.tensor_desc(),
                cuda->filter_desc_, cuda->convolution_desc_, Y.tensor_desc(), conv_method_count, &n, cfap);
            if (t)
            {
                //cudnnDataType_t tt;
                //int n;
                //int nn[8];
                //int ss[8];
                //cudnnGetTensorNdDescriptor(A.getCudnnTensorDesc(), 8, &tt, &n, nn, ss);
                //cudnnGetTensorNdDescriptor(X.getCudnnTensorDesc(), 8, &tt, &n, nn, ss);
                LOG(stderr, "Find CONV forward algorithm failed! {}, {}, {}\n", scd, sfd, t);
            }
            else
            {
                int c = 0;
                size_t memory = 0;
                for (int i = 0; i < n; i++)
                {
                    if (cfap[c].algo == cfap[i].algo)
                    {
                        memory = std::max(memory, cfap[i].memory);
                    }
                }
                methods[0] = cfap[c].algo;
                methods[1] = cfap[c].mathType;
                methods[2] = X.number_;
                //size_t memory = cfap[c].memory;
                if (memory > workspaces[0].getDataSizeInByte())
                {
                    workspaces[0].resize(1, 1, 1, memory / sizeof(real) + 2);
                    //LOG("resize work space {}\n", memory);
                }
#ifdef _DEBUG
                LOG("conv forward choose {},{},{}\n", cfap[c].algo, cfap[c].mathType, cfap[c].memory);
#endif
            }
            //workspace->message();
        }
        auto cfa = cudnnConvolutionFwdAlgo_t(methods[0]);
        auto tensor = cudnnSetConvolutionMathType(cuda->convolution_desc_, cudnnMathType_t(methods[1]));
        auto scf = cudnnConvolutionForward(cuda->cudnn_handle_, &a, X.tensor_desc(), X.data(), cuda->filter_desc_, W.data(),
            cuda->convolution_desc_, cfa, workspaces[0].data(), workspaces[0].getDataSizeInByte(), &r, Y.tensor_desc(), Y.data());
        if (scf)
        {
            LOG(stderr, "CONV forward error {}, {}, {}, {}\n", scd, sfd, scf, cfa);
        }
    }
    else
    {
        Y.initData(0);
        //辅助矩阵的尺寸
        int row = Y.width_ * Y.height_;
        int col = X.channel_ * W.width_ * W.height_;
        Matrix X_ex(row, col, DeviceType::CPU);
        if (methods[0] < 1)
        {
            methods[0] <= 1;
            workspaces[0].resize(1, 1, row * col, 1);
            workspaces[0].initData(-1);
            //ex_pos记录展开的位置，预先记录节省时间
            for (int cX = 0; cX < X.channel_; cX++)
            {
                CONV_OPERATION1(X, W, Y, padding[0], padding[1], stride[0], stride[1],
                    {
                        //记录X_ex中每个位置对应X中的元素，为了效率将二者都拍扁
                        int pY = Y.whcn2i(wY, hY, 0, 0);     //X_ex中对应的行，即在A中的位置
                        int pW = W.whcn2i(wW, hW, cX, 0);    //X_ex中对应的列，即在W中的位置
                        //拍扁
                        int pX = X.whcn2i(wX, hX, cX, 0);    //X其中一组特征对应的位置
                        int pX_ex = X_ex.mn2i(pY, pW);
                        workspaces[0].getData(pX_ex, 0) = pX;    //记录展开的位置
                    });
            }
        }
        Matrix X_sub({ X.row_, 1 }, X.data_, DeviceType::CPU);
        Matrix Y_sub({ Y.width_ * Y.height_, Y.channel_ }, Y.data_, DeviceType::CPU);    //这两个是专用于share
        for (int i = 0; i < X.number_; i++)
        {
            X_sub.shareData(X, 0, i);
            Y_sub.shareData(Y, 0, i);
            X_ex.initData(0);
            for (int j = 0; j < X_ex.getDataSize(); j++)
            {
                int p = workspaces[0].getData(j, 0);
                if (p >= 0 && p < X_sub.getDataSize())
                {
                    X_ex.getData(j) = X_sub.getData(p);
                }
            }
            MatrixEx::mul(X_ex, W, Y_sub, a, r);
        }

#ifdef DIRECT_COMPUTE_CONVOLUTION
        //这是原始计算方法，速度很慢，不要打开这段代码
        //LOG(stderr, "Please supply buffer vector and use the faster convolution method.\n");
        Y.initData(0);
        for (int n = 0; n < Y.number_; n++)
        {
            for (int cX = 0; cX < X.channel_; cX++)
            {
                for (int cY = 0; cY < Y.channel_; cY++)
                {
                    CONV_OPERATION1(X, W, Y, padding[0], padding[1], stride[0], stride[1],
                        {
                            Y.getData(wY, hY, cY, n) += X.getData(wX, hX, cX, n) * W.getData(wW, hW, cX, cY);
                        });
                }
            }
        }
#endif
    }
}

//计算dX只需要W，dA；计算dW只需要X，dA；计算dB只需要dA
void MatrixEx::convolutionBackward(Matrix& X, Matrix& W, const Matrix& Y, std::vector<int>& methods, std::vector<Matrix>& workspaces,
    const std::vector<int>& stride, const std::vector<int>& padding, realc a /*= 1*/, realc rx /*= 0*/, realc rw /*= 0*/)
{
    assert(checkMatrixDevice({ &X, &W, &Y }));
    assert(stride.size() >= 2 && padding.size() >= 2);
    assert(methods.size() >= 9 && workspaces.size() >= 3);
    auto methods_x = &methods[3];
    auto methods_w = &methods[6];
    auto cuda = Y.cuda();
    //这里不用dX判断是因为dX可能是空
    if (Y.inGPU())
    {
        if (stride.size() == 2)
        {
            cudnnSetConvolution2dDescriptor(cuda->convolution_desc_, padding[1], padding[0], stride[1], stride[0], 1, 1, CUDNN_CROSS_CORRELATION, MYCUDNN_DATA_REAL);
            cudnnSetFilter4dDescriptor(cuda->filter_desc_, MYCUDNN_DATA_REAL, CUDNN_TENSOR_NCHW, W.number_, W.channel_, W.height_, W.width_);
        }
        else
        {
            std::vector<int> dilation(padding.size(), 1);
            auto pr = padding;
            auto sr = stride;
            std::reverse(pr.begin(), pr.end());
            std::reverse(sr.begin(), sr.end());
            cudnnSetConvolutionNdDescriptor(cuda->convolution_desc_, padding.size(), pr.data(), sr.data(), dilation.data(), CUDNN_CROSS_CORRELATION, MYCUDNN_DATA_REAL);
            auto w_dim = W.getDim();
            std::reverse(w_dim.begin(), w_dim.end());
            cudnnSetFilterNdDescriptor(cuda->filter_desc_, MYCUDNN_DATA_REAL, CUDNN_TENSOR_NCHW, W.getDimSize(), w_dim.data());
        }
        cudnnSetConvolutionMathType(cuda->convolution_desc_, CUDNN_TENSOR_OP_MATH);
        if (X.needReverse())
        {
            //寻找最快的算法
            if (methods_x[2] < X.number_)
            {
                int n;
                cudnnConvolutionBwdDataAlgoPerf_t cbdap[conv_method_count];
                auto t = cudnnFindConvolutionBackwardDataAlgorithm(cuda->cudnn_handle_, cuda->filter_desc_,
                    Y.tensor_desc(), cuda->convolution_desc_, X.tensor_desc(), conv_method_count, &n, cbdap);
                int c = 0;
                size_t memory = 0;
                for (int i = 0; i < n; i++)
                {
                    if (cbdap[c].algo == cbdap[i].algo)
                    {
                        memory = std::max(memory, cbdap[i].memory);
                    }
                }
                methods_x[0] = cbdap[c].algo;
                methods_x[1] = cbdap[c].mathType;
                methods_x[2] = X.number_;
                //size_t memory = cbdap[c].memory;
                if (memory > workspaces[0].getDataSizeInByte())
                {
                    workspaces[0].resize(1, 1, 1, memory / sizeof(real) + 2);
                    //LOG("resize work space {}\n", memory);
                }
#ifdef _DEBUG
                LOG("conv backward X choose dx {},{},{}\n", cbdap[c].algo, cbdap[c].mathType, cbdap[c].memory);
#endif
                //workspace->message();
            }
            auto cbda = cudnnConvolutionBwdDataAlgo_t(methods_x[0]);
            auto tensor = cudnnSetConvolutionMathType(cuda->convolution_desc_, cudnnMathType_t(methods_x[1]));
            auto scbx = cudnnConvolutionBackwardData(cuda->cudnn_handle_, &a, cuda->filter_desc_, W.data(), Y.tensor_desc(), Y.d().data(),
                cuda->convolution_desc_, cbda, workspaces[0].data(), workspaces[0].getDataSizeInByte(), &rx, X.tensor_desc(), X.d().data());
            if (scbx)
            {
                LOG(stderr, "CONV backward data error {}\n", scbx);
            }
        }
        if (W.needReverse())
        {
            //寻找最快的算法
            if (methods_w[2] < X.number_)
            {
                int n;
                cudnnConvolutionBwdFilterAlgoPerf_t cbfap[conv_method_count];
                cudnnFindConvolutionBackwardFilterAlgorithm(cuda->cudnn_handle_, X.tensor_desc(), Y.tensor_desc(),
                    cuda->convolution_desc_, cuda->filter_desc_, conv_method_count, &n, cbfap);
                int c = 0;
                size_t memory = 0;
                for (int i = 0; i < n; i++)
                {
                    if (cbfap[c].algo == cbfap[i].algo)
                    {
                        memory = std::max(memory, cbfap[i].memory);
                    }
                }
                methods_w[0] = cbfap[c].algo;
                methods_w[1] = cbfap[c].mathType;
                methods_w[2] = X.number_;
                //size_t memory = cbfap[c].memory;
                if (memory > workspaces[0].getDataSizeInByte())
                {
                    workspaces[0].resize(1, 1, 1, memory / sizeof(real) + 2);
                    //LOG("resize work space {}\n", memory);
                }
#ifdef _DEBUG
                LOG("conv backward W choose dw {},{},{}\n", cbfap[c].algo, cbfap[c].mathType, cbfap[c].memory);
#endif
                //workspace->message();
            }
            auto cbfa = cudnnConvolutionBwdFilterAlgo_t(methods_w[0]);
            auto tensor = cudnnSetConvolutionMathType(cuda->convolution_desc_, cudnnMathType_t(methods_w[1]));
            auto scbw = cudnnConvolutionBackwardFilter(cuda->cudnn_handle_, &a, X.tensor_desc(), X.data(), Y.tensor_desc(), Y.d().data(),
                cuda->convolution_desc_, cbfa, workspaces[0].data(), workspaces[0].getDataSizeInByte(), &rw, cuda->filter_desc_, W.d().data());
            if (scbw)
            {
                LOG(stderr, "CONV backward weight error {}\n", scbw);
            }
        }
        //if (B.needUpdate())
        //{
        //    //这个似乎也可以用于一般情况下bias的反向，待查，unfinished
        //    auto scbb = cudnnConvolutionBackwardBias(cuda->cudnn_handle_, &a, R.getCudnnTensorDesc(), R.DMatrix().data(), &r, B.getCudnnTensorDesc(), B.DMatrix().data());
        //}
    }
    else
    {
        if (X.needReverse())
        {
            //计算dX从数学上来看可以反向来求展开后的dX，再压缩，但是看起来加法次数较多
            //转置W的输入和输出
            Matrix W2(W.width_, W.height_, W.number_, W.channel_, DeviceType::CPU);
            for (int i = 0; i < W.channel_; i++)
            {
                for (int j = 0; j < W.number_; j++)
                {
                    copyDataPointer(W, W.getDataPointer(0, 0, i, j), W2, W2.getDataPointer(0, 0, j, i), W.width_ * W.height_);
                }
            }
            //辅助矩阵的尺寸
            int row = X.width_ * X.height_;
            int col = Y.channel_ * W.width_ * W.height_;
            Matrix dY_ex(row, col, DeviceType::CPU);
            dY_ex.initData(0);
            if (methods_x[0] < 1)
            {
                methods_x[0] = 1;
                workspaces[1].resize(1, 1, row * col, 1);
                workspaces[1].initData(-1);
                for (int cY = 0; cY < Y.channel_; cY++)
                {
                    CONV_OPERATION1(X, W, Y, padding[0], padding[1], stride[0], stride[1],
                        {
                            int pX = X.whcn2i(wX, hX, 0, 0);
                            //这里用W或者W2没有区别
                            int pW = W2.whcn2i(wW, hW, cY, 0);
                            //拍扁
                            int pY = Y.whcn2i(wY, hY, cY, 0);
                            int p_ex = dY_ex.mn2i(pX, pW);
                            workspaces[1].getData(p_ex, 0) = pY;
                        });
                }
            }
            Matrix dY_sub(Y.row_, 1, DeviceType::CPU);
            Matrix dX_sub(X.width_ * X.height_, X.channel_, DeviceType::CPU);
            for (int i = 0; i < Y.number_; i++)
            {
                dY_sub.shareData(Y.d(), 0, i);
                dX_sub.shareData(X.d(), 0, i);
                for (int j = 0; j < dY_ex.getDataSize(); j++)
                {
                    if (workspaces[1].getData(j, 0) >= 0)
                    {
                        dY_ex.getData(j) = dY_sub.getData(workspaces[1].getData(j, 0));
                    }
                }
                MatrixEx::mul(dY_ex, W2, dX_sub, a, rx);
            }
        }
        //暂时如此写，看情况能否跟上面合并
        if (W.needReverse())
        {
            //W.d().scale(rw);
            //辅助矩阵的尺寸
            int row = W.width_ * W.height_ * W.channel_;
            int col = Y.width_ * Y.height_;
            Matrix X_ex(row, col, DeviceType::CPU);
            X_ex.initData(0);
            if (methods_w[0] < 1)
            {
                methods_w[0] = 1;
                workspaces[2].resize(1, 1, row * col, 1);
                workspaces[2].initData(-1);
                //cW==cX, nW=cA
                for (int cW = 0; cW < W.channel_; cW++)
                {
                    CONV_OPERATION1(X, W, Y, padding[0], padding[1], stride[0], stride[1],
                        {
                            int pW = W.whcn2i(wW, hW, cW, 0);
                            int pY = Y.whcn2i(wY, hY, 0, 0);
                            //拍扁
                            int pX = X.whcn2i(wX, hX, cW, 0);
                            int p_ex = X_ex.mn2i(pW, pY);
                            workspaces[2].getData(p_ex, 0) = pX;
                        });
                }
            }
            Matrix dY_sub(Y.width_ * Y.height_, Y.channel_, DeviceType::CPU);
            Matrix X_sub(X.row_, 1, DeviceType::CPU);
            for (int i = 0; i < Y.number_; i++)
            {
                dY_sub.shareData(Y.d(), 0, i);
                X_sub.shareData(X, 0, i);
                for (int j = 0; j < X_ex.getDataSize(); j++)
                {
                    //if ((*ex2)[j] >= 0) //因为是满的不需要
                    X_ex.getData(j) = X_sub.getData(workspaces[2].getData(j, 0));
                }
                MatrixEx::mul(X_ex, dY_sub, W.d(), a, rw);    //有点麻烦，暂时不管
            }
        }
        //if (B.needUpdate())
        //{
        //    B.DMatrix().scale(r);
        //    //这个就是对对应的dR求和
        //    for (int n = 0; n < R.number_; n++)
        //    {
        //        for (int c = 0; c < R.channel_; c++)
        //        {
        //            B.DMatrix().getData(0, 0, c, 0) += a * VectorMath::sum(R.DMatrix().getDataPointer(0, 0, c, n), R.width_ * R.height_);
        //        }
        //    }
        //}

#ifdef DIRECT_COMPUTE_CONVOLUTION
        //这是原始计算方法，速度很慢，不要打开这段代码
        //LOG(stderr, "Please supply buffer vector and use the faster convolution method.\n");
        if (W.needReverse())
        {
            W.d().scale(rw);
        }
        if (X.needReverse())
        {
            X.d().scale(rx);
        }
        for (int n = 0; n < Y.number_; n++)
        {
            for (int cX = 0; cX < X.channel_; cX++)
            {
                for (int cY = 0; cY < Y.channel_; cY++)
                {
                    CONV_OPERATION1(X, W, Y, wX, hX, wW, hW, wY, hY,
                        {
                            if (X.needReverse())
                            {
                                X.d().getData(wX, hX, cX, n) += a * Y.d().getData(wY, hY, cY, n) * W.getData(wW, hW, cX, cY);
                            }
                            if (W.needReverse())
                            {
                                W.d().getData(wW, hW, cX, cY) += a * X.getData(wX, hX, cX, n) * Y.d().getData(wY, hY, cY, n);
                            }
                        });
                }
            }
        }
#endif
    }
}

//ada_d表示ada方法计算得到的参数的改变量，应在下一步加到原参数上，下同
void MatrixEx::adaDeltaUpdate(Matrix& mean_d2, Matrix& mean_ada_d2, Matrix& d, Matrix& ada_d, real rou, real epsilon)
{
    assert(checkMatrixDevice({ &mean_d2, &mean_ada_d2, &d, &ada_d }));
    if (d.inGPU())
    {
        cuda_ada_delta_update(mean_d2.data(), mean_ada_d2.data(), d.data(), ada_d.data(), d.data_size_, rou, epsilon);
    }
    else
    {
        auto p1 = mean_d2.data();
        auto p2 = mean_ada_d2.data();
        auto p3 = d.data();
        auto p4 = ada_d.data();
        for (int i = 0; i < d.data_size_; i++)
        {
            p1[i] = p1[i] * rou + p3[i] * p3[i] * (1 - rou);
            p4[i] = p3[i] * sqrt((p2[i] + epsilon) / (p1[i] + epsilon));
            p2[i] = p2[i] * rou + p4[i] * p4[i] * (1 - rou);
        }
    }
}

void MatrixEx::adamUpdate(Matrix& mean_d, Matrix& mean_d2, Matrix& d, Matrix& ada_d, real beta1, real beta2, real epsilon, real t)
{
    assert(checkMatrixDevice({ &mean_d, &mean_d2, &d, &ada_d }));
    if (d.inGPU())
    {
        cuda_adam_update(mean_d.data(), mean_d2.data(), d.data(), ada_d.data(), d.data_size_, beta1, beta2, epsilon, t);
    }
    else
    {
        auto p1 = mean_d.data();
        auto p2 = mean_d2.data();
        auto p3 = d.data();
        auto p4 = ada_d.data();
        for (int i = 0; i < d.data_size_; i++)
        {
            p1[i] = p1[i] * beta1 + p3[i] * (1 - beta1);
            p2[i] = p2[i] * beta2 + p3[i] * p3[i] * (1 - beta2);
            p4[i] = p3[i] / (1 - pow(beta1, t)) / (sqrt(p2[i] / (1 - pow(beta2, t))) + epsilon);
        }
    }
}

void MatrixEx::adaRMSPropUpdate(Matrix& mean_d2, Matrix& d, Matrix& ada_d, real rou, real epsilon)
{
    assert(checkMatrixDevice({ &mean_d2, &d, &ada_d }));
    if (d.inGPU())
    {
        cuda_rms_prop_update(mean_d2.data(), d.data(), ada_d.data(), d.data_size_, rou, epsilon);
    }
    else
    {
        auto p1 = mean_d2.data();
        auto p2 = d.data();
        auto p3 = ada_d.data();
        for (int i = 0; i < d.data_size_; i++)
        {
            p1[i] = p1[i] * rou + p2[i] * p2[i] * (1 - rou);
            p3[i] = p2[i] * sqrt(1.0 / (p1[i] + epsilon));
        }
    }
}

//R = ((1-rou)/(1-rou_hat)-rou/rou_hat)*beta
void MatrixEx::sparse(Matrix& rou_hat, Matrix& R, real rou, real beta)
{
    if (rou_hat.inGPU())
    {
        cuda_sparse(rou_hat.data(), R.data(), R.data_size_, rou, beta);
    }
    else
    {
        for (int i = 0; i < R.data_size_; i++)
        {
            R.data()[i] = ((1 - rou) / (1 - rou_hat.data()[i]) - rou / rou_hat.data()[i]) * beta;
        }
    }
}

void MatrixEx::fill(Matrix& m, RandomFillType random_type, int in, int out)
{
    if (m.getDataSize() <= 0)
    {
        return;
    }
    Random<real> random_generator;
    random_generator.set_seed();
    real a = 0, b = 0;

    switch (random_type)
    {
    case RANDOM_FILL_CONSTANT:
        m.initData(0);
        return;
        break;
    case RANDOM_FILL_XAVIER:
        random_generator.set_random_type(RANDOM_UNIFORM);
        a = sqrt(6.0 / (in + out));
        random_generator.set_parameter(-a, a);
        //LOG("Xavier, %d, %d, %f\n", prev_layer->out_total, out_total, a);
        break;
    case RANDOM_FILL_GAUSSIAN:
        random_generator.set_random_type(RANDOM_NORMAL);
        //LOG("Gaussian distribution\n");
        break;
    case RANDOM_FILL_MSRA:
        random_generator.set_random_type(RANDOM_NORMAL);
        a = sqrt(2.0 / in);
        random_generator.set_parameter(0, a);
        //LOG("Gaussian distribution\n");
        break;
    case RANDOM_FILL_LECUN:
        random_generator.set_random_type(RANDOM_NORMAL);
        a = 1.0 / in;
        random_generator.set_parameter(0, a);
        //LOG("Gaussian distribution\n");
        break;
    default:
        break;
    }
    std::vector<real> temp(m.getDataSize());
    random_generator.rand_data(temp.data(), temp.size());
    m.initData(temp.data(), temp.size());
}

}    // namespace cccc