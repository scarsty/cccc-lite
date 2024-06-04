#include "MatrixEx.h"
#include "Log.h"
#include "Random.h"
#include "Timer.h"
#include "VectorMath.h"
#include "gpu_lib.h"
#include <cassert>

#if !defined(M_PI)
#define M_PI 3.1415926535897932
#endif

namespace cccc
{

static void setDesc4D(cudnnTensorDescriptor_t desc, DataType data_type, int w, int h, int c, int n)
{
    cudnnSetTensor4dDescriptor(desc, CUDNN_TENSOR_NCHW, toCudnnDataType(data_type), n, c, h, w);
}

//若bias某一维度为1，则视为对X的对应维度加上同样的值，cudnn的文档中指出最高支持到5维
//此处仅用于两种情况：1 - bias仅有channel不为1，2 - bias的number为1，注意1是2的一个特例，其他情况不一定能得到正确结果
void MatrixEx::addBias(const Matrix& X, const Matrix& bias, Matrix& Y, float a /*= 1*/, float b /*= 1*/)
{
    assert(checkMatrixDevice({ &X, &bias, &Y }));
    assert(bias.getDataSize() == bias.channel_ || bias.number_ == 1);
    //if (&X != &Y)
    {
        copyData(X, Y);
    }
    if (X.isCuda())
    {
        auto gpu = X.gpu();
        if (bias.getDimSize() <= 5)
        {
            cudnnAddTensor(gpu->cudnn_handle_, &a, bias.cudnn_desc(), bias.data(), &b, Y.cudnn_desc(), Y.data());
        }
        else
        {
            LOG_ERR("Dim of bias cannot > 5!\n");
            //int op_desc[64] = { 0 };
            //int op_desc2[64] = { 0 };
            //GpuControl::setTensorDesc4D(op_desc_t, bias.width_, bias.height_, bias.channel_, bias.number_);
            //GpuControl::setTensorDesc4D((cudnnTensorDescriptor_t)op_desc2, Y.width_, Y.height_, Y.channel_, Y.number_);
            //cudnnAddTensor(gpu->cudnn_handle_, &a, op_desc_t, bias.data(), &b, op_desc_t2, Y.data());
        }
    }
    else if (X.isHip())
    {
        //hip_addbias(X.data(), bias.data(), Y.data(), X.getDataSize(), Y.row_ / Y.channel_, bias.getDataSize(), a, b);
        auto gpu = Y.gpu();
        b = b - a;    //only for miopen
        miopenConvolutionForwardBias(gpu->miopen_handle_, &a, bias.miopen_desc(), bias.data(), &b, Y.miopen_desc(), Y.data());
        //miopenOpTensor(gpu->miopen_handle_, miopenTensorOpAdd, &const_real_1, X.miopen_desc(), X.data(), &const_real_1, bias.miopen_desc(), bias.data(), &b, Y.miopen_desc(), Y.data());
    }
    else
    {
        if (bias.getDataSize() == bias.channel_)
        {
            for (int i = 0; i < Y.data_size_; i++)
            {
                int c = i % Y.row_ / (Y.row_ / Y.channel_);
                Y.setData(i, Y.getData(i) + bias.getData(c));
            }
        }
        else
        {
            for (int i = 0; i < Y.data_size_; i++)
            {
                int c = i % bias.getDataSize();
                Y.setData(i, Y.getData(i) + bias.getData(c));
            }
        }
    }
}

void MatrixEx::addBiasBackward(Matrix& X, Matrix& bias, const Matrix& Y, float a /*= 1*/, float b /*= 1*/)
{
    assert(checkMatrixDevice({ &X, &bias, &Y }));
    if (X.needBack())
    {
        Matrix::add(X.d(), Y.d(), X.d(), a, 1, 0);
    }
    if (bias.needBack())
    {
        if (X.isCuda())
        {
            //用卷积反向代替一般反向，此处待验证
            auto gpu = Y.gpu();
            cudnnConvolutionBackwardBias(gpu->cudnn_handle_, &const_real_1, Y.cudnn_desc(), Y.d().data(), &b, bias.cudnn_desc(), bias.d().data());
        }
        else if (X.isHip())
        {
            auto gpu = Y.gpu();
            //b = 0;    //todo: miopen unfinished
            miopenConvolutionBackwardBias(gpu->miopen_handle_, &const_real_1, Y.miopen_desc(), Y.d().data(), &b, bias.miopen_desc(), bias.d().data());
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
                        bias.d().setData(0, 0, c, 0, bias.d().getData(0, 0, c, 0) + b * VectorMath::sum((float*)Y.d().getDataPtr(0, 0, c, n), Y.width_ * Y.height_));
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
            copyDataPtr(tmp, tmp.getDataPtr(0, 0, 0, n), Y, Y.getDataPtr(0, 0, c_off, n), tmp.row_);
            c_off += tmp.channel_;
        }
    }
}

void MatrixEx::concatByChannelBackward(std::vector<MatrixSP>& X_vector, const Matrix& Y)
{
    //此处可能应该是考虑求和
    Matrix dy(Y.getDataType(), Y.getDeviceType());
    Matrix dx(Y.getDataType(), Y.getDeviceType());
    for (int n = 0; n < Y.number_; n++)
    {
        int c_off = 0;
        for (int i = 0; i < X_vector.size(); i++)
        {
            if (X_vector[i]->needBack())
            {
                dy.shareData(Y.d(), 0, 0, c_off, n);
                dy.resize(X_vector[i]->row_, 1);
                dx.shareData(X_vector[i]->d(), 0, 0, 0, n);
                dx.resize(X_vector[i]->row_, 1);
                Matrix::add(dy, dx, dx, 1, X_vector[i]->keepWeight(), 0);    //这里写错了
                //copyDataPtr(Y, Y.d().getDataPtr(0, 0, c_off, n), tmp, tmp.getDataPtr(0, 0, 0, n), tmp.row_);
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
            copyDataPtr(X, X.getDataPtr(0, 0, c_off, n), tmp, tmp.getDataPtr(0, 0, 0, n), tmp.row_);
            c_off += tmp.channel_;
        }
    }
}

//初始化激活需要的缓冲区
void MatrixEx::activeBufferInit(const Matrix& X, ActiveFunctionType af, std::vector<int>& int_vector, std::vector<float>& real_vector, std::vector<Matrix>& matrix_vector)
{
    auto gpu = X.gpu();
    switch (af)
    {
    case ACTIVE_FUNCTION_DROPOUT:
        if (X.isCuda())
        {
            size_t size1, size2;
            cudnnDropoutGetStatesSize(gpu->cudnn_handle_, &size1);
            Matrix rg_stat({ int(size1 / sizeof(float) + 1), 1 });    //int64->int32
            cudnnDropoutGetReserveSpaceSize(X.cudnn_desc(), &size2);
            Matrix reverse_space({ int(size2 / sizeof(float) + 1), 1 });    //int64->int32
            //LOG(stderr, "dropout size {},{}\n", size, size2);
            matrix_vector = { rg_stat, reverse_space };
        }
        break;
    case ACTIVE_FUNCTION_LOCAL_CONSTRAST_NORMALIZATION:
    case ACTIVE_FUNCTION_DIVISIVE_NORMALIZATION:
        if (X.isCuda())
        {
            matrix_vector = { Matrix(X.getDim()), Matrix(X.getDim()), Matrix({ X.getRow(), X.getNumber() * 2 }) };
        }
        break;
    case ACTIVE_FUNCTION_BATCH_NORMALIZATION:
        if (X.isCuda())
        {
            cudnnTensorDesc op_desc;
            cudnnDeriveBNTensorDescriptor(op_desc(), X.cudnn_desc(), cudnnBatchNormMode_t(int_vector[1]));
            int w, h, c, n, p1, p2, p3, p4;
            cudnnDataType_t dt;
            cudnnGetTensor4dDescriptor(op_desc(), &dt, &n, &c, &h, &w, &p1, &p2, &p3, &p4);
            std::vector<int> size = { w, h, c, n };
            matrix_vector = { Matrix(size).fillData(0.5), Matrix(size), Matrix(size), Matrix(size), Matrix(size), Matrix(size), Matrix(size), Matrix(size) };
        }
        break;
    case ACTIVE_FUNCTION_SPATIAL_TRANSFORMER:
        if (X.isCuda())
        {
            matrix_vector = { Matrix({ 3, 2, 1, X.number_ }), Matrix(X).getDim(), Matrix({ 3, 2, 1, X.number_ }), Matrix(X).getDim() };
        }
        break;
    case ACTIVE_FUNCTION_RECURRENT:
        break;
    case ACTIVE_FUNCTION_ZERO_CHANNEL:
    {
        //1 matrix: factor for mul
        Matrix m(X.getDim(), X.getDataType(), UnitType::CPU);
        m.fillData(1);
        for (int w = 0; w < X.width_; w++)
        {
            for (int h = 0; h < X.height_; h++)
            {
                for (int c : int_vector)
                {
                    for (int n = 0; n < X.number_; n++)
                    {
                        m.setData(w, h, c, n, 0);
                    }
                }
            }
            m.toGPU();
            matrix_vector = { m };
        }
        break;
    }
    case ACTIVE_FUNCTION_LEAKY_RELU:
        if (real_vector.size() == 0)
        {
            real_vector.push_back(0.02);
        }
        break;
    case ACTIVE_FUNCTION_ELU:
        if (real_vector.size() == 0)
        {
            real_vector.push_back(1.6732632423543772848170429916717);
        }
        break;
    case ACTIVE_FUNCTION_CLIPPED_RELU:
        if (real_vector.size() == 0)
        {
            real_vector.push_back(6);
        }
        break;
    case ACTIVE_FUNCTION_SILU:
        matrix_vector = { Matrix(X.getDim()) };
        break;
    default:
        break;
    }
}

//正向激活，依据X计算A
//此处我们定义激活操作为输入和输出矩阵（或张量）的维度完全相同
//传附加参数的时候使用了C++11的初始化列表，因此效率可能较低，实际上如不考虑效率可以代替基本激活函数
//调用时请自己保证参数数量的正确性！
//vector参数的含义请参考Activer.cpp中的注释，以及上面函数中matrix向量的含义
void MatrixEx::activeForward(const Matrix& X, Matrix& Y, ActiveFunctionType af,
    std::vector<int>& int_vector, std::vector<float>& real_vector, std::vector<Matrix>& matrix_vector, float a /*= 1*/, float r /*= 0*/)
{
    assert(checkMatrixDevice({ &X, &Y }));
    auto nan = CUDNN_NOT_PROPAGATE_NAN;
    auto gpu = X.gpu();
    //cudnnActivationDesc activation_desc;
    //cudnnTensorDesc tensor_desc;
    switch (af)
    {
    case ACTIVE_FUNCTION_NONE:
        Matrix::copyData(X, Y);
        break;
    case ACTIVE_FUNCTION_SIGMOID:
    case ACTIVE_FUNCTION_SIGMOID_CE:
    case ACTIVE_FUNCTION_SIGMOID3:
        if (X.isCuda())
        {
            cudnnActivationDesc activation_desc;
            GpuControl::setActivationDesc(activation_desc(), CUDNN_ACTIVATION_SIGMOID, 1);
            cudnnActivationForward(gpu->cudnn_handle_, activation_desc(),
                &a, X.cudnn_desc(), X.data(), &r, Y.cudnn_desc(), Y.data());
        }
        else if (X.isHip())
        {
            auto act_desc = gpu->getDesc<miopenActivationDescriptor_t>();
            miopenSetActivationDescriptor(act_desc, miopenActivationLOGISTIC, 1, 1, 1);
            miopenActivationForward(gpu->miopen_handle_, act_desc, &a, X.miopen_desc(), X.data(), &r, Y.miopen_desc(), Y.data());
        }
        else
        {
            VectorMath::sigmoid_v(X.data(), Y.data(), Y.data_size_, a, r);
        }
        break;
    case ACTIVE_FUNCTION_RELU:
        if (X.isCuda())
        {
            cudnnActivationDesc activation_desc;
            GpuControl::setActivationDesc(activation_desc(), CUDNN_ACTIVATION_RELU, 1);
            cudnnActivationForward(gpu->cudnn_handle_, activation_desc(),
                &a, X.cudnn_desc(), X.data(), &r, Y.cudnn_desc(), Y.data());
        }
        else if (X.isHip())
        {
            auto act_desc = gpu->getDesc<miopenActivationDescriptor_t>();
            miopenSetActivationDescriptor(act_desc, miopenActivationRELU, 1, 1, 1);
            miopenActivationForward(gpu->miopen_handle_, act_desc, &a, X.miopen_desc(), X.data(), &r, Y.miopen_desc(), Y.data());
        }
        else
        {
            VectorMath::relu_v(X.data(), Y.data(), Y.data_size_, a, r);
        }
        break;
    case ACTIVE_FUNCTION_TANH:
        if (X.isCuda())
        {
            cudnnActivationDesc activation_desc;
            GpuControl::setActivationDesc(activation_desc(), CUDNN_ACTIVATION_TANH, 1);
            cudnnActivationForward(gpu->cudnn_handle_, activation_desc(),
                &a, X.cudnn_desc(), X.data(), &r, Y.cudnn_desc(), Y.data());
        }
        else if (X.isHip())
        {
            auto act_desc = gpu->getDesc<miopenActivationDescriptor_t>();
            miopenSetActivationDescriptor(act_desc, miopenActivationTANH, 1, 1, 1);
            miopenActivationForward(gpu->miopen_handle_, act_desc, &a, X.miopen_desc(), X.data(), &r, Y.miopen_desc(), Y.data());
        }
        else
        {
            VectorMath::tanh_v(X.data(), Y.data(), Y.data_size_, a, r);
        }
        break;
    case ACTIVE_FUNCTION_SOFTMAX:
    case ACTIVE_FUNCTION_SOFTMAX_CE:
    case ACTIVE_FUNCTION_SOFTMAX3:
        if (X.isCuda())
        {
            cudnnTensorDesc tensor_desc;
            setDesc4D(tensor_desc(), X.getDataType(), 1, 1, X.row_, X.number_);
            cudnnSoftmaxForward(gpu->cudnn_handle_, CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_INSTANCE,
                &a, tensor_desc(), X.data(), &r, tensor_desc(), Y.data());
        }
        else if (X.isHip())
        {
            miopenSoftmaxForward_V2(gpu->miopen_handle_, &a, X.miopen_desc(), X.data(), &r, Y.miopen_desc(), Y.data(), MIOPEN_SOFTMAX_ACCURATE, MIOPEN_SOFTMAX_MODE_INSTANCE);
        }
        else
        {
            //因为数值问题，可能需要减去每列最大值
            MatrixEx::copyData(X, Y);
            for (int i = 0; i < Y.number_; i++)
            {
                VectorMath::minus_max(Y.getDataPtr(0, i), Y.row_);
            }
            VectorMath::exp_v(Y.data(), Y.data(), Y.data_size_);
            for (int i = 0; i < Y.number_; i++)
            {
                float sum = Y.sumAbsCol(i);
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
        if (X.isCuda())
        {
            cudnnTensorDesc tensor_desc;
            setDesc4D(tensor_desc(), X.getDataType(), 1, 1, X.row_, X.number_);
            cudnnSoftmaxForward(gpu->cudnn_handle_, CUDNN_SOFTMAX_FAST, CUDNN_SOFTMAX_MODE_INSTANCE,
                &a, tensor_desc(), X.data(), &r, tensor_desc(), Y.data());
        }
        else if (X.isHip())
        {
            miopenSoftmaxForward_V2(gpu->miopen_handle_, &a, X.miopen_desc(), X.data(), &r, Y.miopen_desc(), Y.data(), MIOPEN_SOFTMAX_FAST, MIOPEN_SOFTMAX_MODE_INSTANCE);
        }
        else
        {
            VectorMath::exp_v(X.data(), Y.data(), Y.data_size_, a, r);
            for (int i = 0; i < Y.number_; i++)
            {
                float sum = Y.sumAbsCol(i);
                if (sum == 0)
                {
                    continue;
                }
                Y.scaleCol(1 / sum, i);
            }
        }
        break;
    case ACTIVE_FUNCTION_SOFTMAX_LOG:
        if (X.isCuda())
        {
            cudnnTensorDesc tensor_desc;
            setDesc4D(tensor_desc(), X.getDataType(), 1, 1, X.row_, X.number_);
            cudnnSoftmaxForward(gpu->cudnn_handle_, CUDNN_SOFTMAX_LOG, CUDNN_SOFTMAX_MODE_INSTANCE,
                &a, tensor_desc(), X.data(), &r, tensor_desc(), Y.data());
        }
        else if (X.isHip())
        {
            miopenSoftmaxForward_V2(gpu->miopen_handle_, &a, X.miopen_desc(), X.data(), &r, Y.miopen_desc(), Y.data(), MIOPEN_SOFTMAX_LOG, MIOPEN_SOFTMAX_MODE_INSTANCE);
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
        if (X.inGpu())
        {
            Matrix T({ Y.row_, Y.number_ }, Y.getDataType(), UnitType::CPU);
            T.scale(0);
            for (int i_group = 0; i_group < Y.number_; i_group++)
            {
                int index = X.indexColMaxAbs(i_group);
                T.setData(index, i_group, 1);
            }
            Matrix::copyData(T, Y);
        }
        else
        {
            Y.scale(0);
            for (int i_group = 0; i_group < Y.number_; i_group++)
            {
                int index = X.indexColMaxAbs(i_group);
                Y.setData(index, i_group, 1);
            }
        }
        break;
    default:
        LOG_ERR("ACTIVE forward not right {}!\n", int(af));
        break;
    }
}

//参考activeForward2
//反向激活，依据X，A，dA计算dX
//softmax的cpu部分貌似不支持a，b
//这里的系数应该是不对,unfinished
void MatrixEx::activeBackward(Matrix& X, const Matrix& Y, ActiveFunctionType af,
    std::vector<int>& int_vector, std::vector<float>& real_vector, std::vector<Matrix>& matrix_vector, float a /*= 1*/, float r /*= 0*/)
{
    assert(checkMatrixDevice({ &X, &Y }));
    auto nan = CUDNN_NOT_PROPAGATE_NAN;
    auto gpu = Y.gpu();
    switch (af)
    {
    case ACTIVE_FUNCTION_NONE:
    case ACTIVE_FUNCTION_SIGMOID_CE:
    case ACTIVE_FUNCTION_SOFTMAX_CE:
    case ACTIVE_FUNCTION_SOFTMAX_FAST_CE:
        Matrix::add(Y.d(), X.d(), X.d(), a, r, 0);
        break;
    case ACTIVE_FUNCTION_SIGMOID:
        if (X.isCuda())
        {
            cudnnActivationDesc activation_desc;
            GpuControl::setActivationDesc(activation_desc(), CUDNN_ACTIVATION_SIGMOID, 1);
            cudnnActivationBackward(gpu->cudnn_handle_, activation_desc(), &a, Y.cudnn_desc(), Y.data(),
                Y.cudnn_desc(), Y.d().data(), X.cudnn_desc(), X.data(), &r, X.cudnn_desc(), X.d().data());
        }
        else if (X.isHip())
        {
            auto act_desc = gpu->getDesc<miopenActivationDescriptor_t>();
            miopenSetActivationDescriptor(act_desc, miopenActivationLOGISTIC, 1, 1, 1);
            miopenActivationBackward(gpu->miopen_handle_, act_desc, &a, Y.miopen_desc(), Y.data(), Y.miopen_desc(), Y.d().data(), X.miopen_desc(), X.data(), &r, X.miopen_desc(), X.d().data());
        }
        else
        {
            VectorMath::sigmoid_vb(Y.data(), Y.d().data(), X.data(), X.d().data(), X.data_size_, a, r);
        }
        break;
    case ACTIVE_FUNCTION_RELU:
        if (X.isCuda())
        {
            cudnnActivationDesc activation_desc;
            GpuControl::setActivationDesc(activation_desc(), CUDNN_ACTIVATION_RELU, 1);
            cudnnActivationBackward(gpu->cudnn_handle_, activation_desc(), &a, Y.cudnn_desc(), Y.data(),
                Y.cudnn_desc(), Y.d().data(), X.cudnn_desc(), X.data(), &r, X.cudnn_desc(), X.d().data());
        }
        else if (X.isHip())
        {
            auto act_desc = gpu->getDesc<miopenActivationDescriptor_t>();
            miopenSetActivationDescriptor(act_desc, miopenActivationRELU, 1, 1, 1);
            miopenActivationBackward(gpu->miopen_handle_, act_desc, &a, Y.miopen_desc(), Y.data(), Y.miopen_desc(), Y.d().data(), X.miopen_desc(), X.data(), &r, X.miopen_desc(), X.d().data());
        }
        else
        {
            VectorMath::relu_vb(Y.data(), Y.d().data(), X.data(), X.d().data(), X.data_size_, a, r);
        }
        break;
    case ACTIVE_FUNCTION_TANH:
        //两者结果在1e-10的精度有区别
        if (X.isCuda())
        {
            cudnnActivationDesc activation_desc;
            GpuControl::setActivationDesc(activation_desc(), CUDNN_ACTIVATION_TANH, 1);
            cudnnActivationBackward(gpu->cudnn_handle_, activation_desc(), &a, Y.cudnn_desc(), Y.data(),
                Y.cudnn_desc(), Y.d().data(), X.cudnn_desc(), X.data(), &r, X.cudnn_desc(), X.d().data());
        }
        else if (X.isHip())
        {
            auto act_desc = gpu->getDesc<miopenActivationDescriptor_t>();
            miopenSetActivationDescriptor(act_desc, miopenActivationTANH, 1, 1, 1);
            miopenActivationBackward(gpu->miopen_handle_, act_desc, &a, Y.miopen_desc(), Y.data(), Y.miopen_desc(), Y.d().data(), X.miopen_desc(), X.data(), &r, X.miopen_desc(), X.d().data());
        }
        else
        {
            VectorMath::tanh_vb(Y.data(), Y.d().data(), X.data(), X.d().data(), X.data_size_, a, r);
        }
        break;
    case ACTIVE_FUNCTION_SOFTMAX:
    case ACTIVE_FUNCTION_SOFTMAX_FAST:
        if (X.isCuda())
        {
            cudnnTensorDesc tensor_desc;
            auto softmax_flag = CUDNN_SOFTMAX_ACCURATE;
            if (af == ACTIVE_FUNCTION_SOFTMAX_FAST)
            {
                softmax_flag = CUDNN_SOFTMAX_FAST;
            }
            setDesc4D(tensor_desc(), X.getDataType(), 1, 1, X.row_, X.number_);
            auto s = cudnnSoftmaxBackward(gpu->cudnn_handle_, softmax_flag, CUDNN_SOFTMAX_MODE_INSTANCE,
                &a, tensor_desc(), Y.data(), tensor_desc(), Y.d().data(), &r, tensor_desc(), X.d().data());
        }
        else if (X.isHip())
        {
            auto softmax_flag = MIOPEN_SOFTMAX_ACCURATE;
            if (af == ACTIVE_FUNCTION_SOFTMAX_FAST)
            {
                softmax_flag = MIOPEN_SOFTMAX_FAST;
            }
            miopenSoftmaxBackward_V2(gpu->miopen_handle_, &a, Y.miopen_desc(), Y.data(), Y.miopen_desc(), Y.d().data(), &r, X.miopen_desc(), X.d().data(), softmax_flag, MIOPEN_SOFTMAX_MODE_INSTANCE);
        }
        else
        {
            for (int i = 0; i < X.number_; i++)
            {
                float v = MatrixEx::dotCol(Y, i, Y.d(), i);
                VectorMath::softmax_vb_sub((float*)Y.getDataPtr(0, i), (float*)Y.d().getDataPtr(0, i), v, (float*)X.d().getDataPtr(0, i), X.row_);
            }
        }
        break;
    case ACTIVE_FUNCTION_SOFTMAX_LOG:
        if (X.isCuda())
        {
            cudnnTensorDesc tensor_desc;
            setDesc4D(tensor_desc(), X.getDataType(), 1, 1, X.row_, X.number_);
            auto s = cudnnSoftmaxBackward(gpu->cudnn_handle_, CUDNN_SOFTMAX_LOG, CUDNN_SOFTMAX_MODE_INSTANCE,
                &a, tensor_desc(), Y.data(), tensor_desc(), Y.d().data(), &r, tensor_desc(), X.d().data());
        }
        else if (X.isHip())
        {
            miopenSoftmaxBackward_V2(gpu->miopen_handle_, &a, Y.miopen_desc(), Y.data(), Y.miopen_desc(), Y.d().data(), &r, X.miopen_desc(), X.d().data(), MIOPEN_SOFTMAX_LOG, MIOPEN_SOFTMAX_MODE_INSTANCE);
        }
        else
        {
            for (int i = 0; i < X.number_; i++)
            {
                float v = 0;
                for (int j = 0; j < X.row_; j++)
                {
                    v += Y.d().getData(i, j);
                }
                VectorMath::softmaxlog_vb_sub((float*)Y.getDataPtr(0, i), (float*)Y.d().getDataPtr(0, i), v, (float*)X.d().getDataPtr(0, i), X.row_);
            }
        }
        break;
    case ACTIVE_FUNCTION_ABSMAX:
        //似乎应该是返回一个常数矩阵，若考虑效率应当留空此处在外部处理
        //dX.fillData(1);
        LOG_ERR("Unsupported backward of FINDMAX!\n");
        break;
        //case ACTIVE_FUNCTION_SUMMAX:
        //    Matrix::copyDataPtr(A,A.DMatrix().data(), X,X.DMatrix().data());
        //    for (int i = 0; i < X.getCol(); i++)
        //    {
        //        dX.addNumberCol(-Matrix::dotCol(A, i, dA, i), 1, i);
        //        dX.scaleCol(1.0 / X.sumAbsCol(i), i);
        //    }
        //    break;
    default:
        LOG_ERR("ACTIVE backward not right {}!\n", int(af));
        break;
    }
}

void MatrixEx::activeForwardSimple(const Matrix& X, Matrix& Y, ActiveFunctionType af, float a /*= 1*/, float r /*= 0*/)
{
    std::vector<int> int_vector;
    std::vector<float> real_vector;
    std::vector<Matrix> matrix_vector;
    activeForward(X, Y, af, int_vector, real_vector, matrix_vector, a, r);
}

void MatrixEx::activeBackwardSimple(Matrix& X, const Matrix& Y, ActiveFunctionType af, float a /*= 1*/, float r /*= 0*/)
{
    std::vector<int> int_vector;
    std::vector<float> real_vector;
    std::vector<Matrix> matrix_vector;
    activeBackward(X, Y, af, int_vector, real_vector, matrix_vector, a, r);
}

//池化
//gpu部分，平均模式下对padding的支持目前还有问题
void MatrixEx::poolingForward(const Matrix& X, Matrix& Y, PoolingType pooling_type, int reverse, std::vector<Matrix>& workspaces,
    const std::vector<int>& window, const std::vector<int>& stride,
    const std::vector<int>& padding, float a /*= 1*/
    ,
    float r /*= 0*/)
{
    assert(checkMatrixDevice({ &X, &Y }));
    assert(window.size() >= 2 && stride.size() >= 2 && padding.size() >= 2);
    auto gpu = X.gpu();
    if (Y.isCuda())
    {
        cudnnPoolingMode_t pooling_type_c = CUDNN_POOLING_MAX;
        if (pooling_type == POOLING_AVERAGE_NOPADDING) { pooling_type_c = CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING; }
        if (pooling_type == POOLING_AVERAGE_PADDING) { pooling_type_c = CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING; }
        cudnnPoolingDesc pooling_desc;
        if (stride.size() == 2)
        {
            cudnnSetPooling2dDescriptor(pooling_desc(), pooling_type_c, CUDNN_NOT_PROPAGATE_NAN,
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
            cudnnSetPoolingNdDescriptor(pooling_desc(), pooling_type_c, CUDNN_NOT_PROPAGATE_NAN,
                window.size(), wr.data(), pr.data(), sr.data());
        }
        cudnnStatus_t s;
        if (reverse == 0)
        {
            s = cudnnPoolingForward(gpu->cudnn_handle_, pooling_desc(), &a, X.cudnn_desc(), X.data(), &r, Y.cudnn_desc(), Y.data());
        }
        else
        {
            s = cudnnPoolingBackward(gpu->cudnn_handle_, pooling_desc(),
                &a, X.cudnn_desc(), X.data(), X.cudnn_desc(), X.data(), Y.cudnn_desc(), Y.data(), &r, Y.cudnn_desc(), Y.data());
            Y.scale(VectorMath::multiply(window));
        }
        if (s)
        {
            LOG_ERR("POOL forward error {}, {}\n", cudnnGetErrorString(s), X.gpu()->lastCudnnErrorString());
        }
    }
    else if (Y.isHip())
    {
        miopenPoolingMode_t pooling_type_c = miopenPoolingMax;
        if (pooling_type == POOLING_AVERAGE_NOPADDING) { pooling_type_c = miopenPoolingAverage; }
        if (pooling_type == POOLING_AVERAGE_PADDING) { pooling_type_c = miopenPoolingAverageInclusive; }
        //auto op_desc = gpu->getDesc<miopenPoolingDescriptor_t>();
        miopenPoolingDesc op_desc;    //todo: 未测试
        if (stride.size() == 2)
        {
            miopenSet2dPoolingDescriptor(op_desc(), pooling_type_c, window[1], window[0], padding[1], padding[0], stride[1], stride[0]);
        }
        if (workspaces.size() == 0)
        {
            workspaces.resize(1);
        }
        if (workspaces[0].getDataSizeInByte() == 0)
        {
            size_t ws;
            miopenPoolingGetWorkSpaceSize(Y.miopen_desc(), &ws);
            workspaces[0].resize(1, 1, 1, ws);
        }
        miopenStatus_t s;
        bool back = gpu->active_phase_ == ACTIVE_PHASE_TRAIN;
        if (reverse == 0)
        {
            s = miopenPoolingForward(gpu->miopen_handle_, op_desc(), &a, X.miopen_desc(), X.data(), &r, Y.miopen_desc(), Y.data(), back, workspaces[0].data(), workspaces[0].getDataSizeInByte());
        }
        else
        {
            s = miopenPoolingBackward(gpu->miopen_handle_, op_desc(), &a, X.miopen_desc(), X.data(), X.miopen_desc(), X.data(), Y.miopen_desc(), Y.data(), &r, Y.miopen_desc(), Y.data(), workspaces[0].data());
            Y.scale(VectorMath::multiply(window));
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
                    float v = 0;
                    if (pooling_type == POOLING_MAX)
                    {
                        v = -FLT_MAX;
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
                    Y.setData(wA, hA, p, 0, Y.getData(wA, hA, p, 0) + a * v);    // +r * Y.getData(wA, hA, p, 0);
                }
            }
        }
    }
}

//使用cpu时利用了record -- 取消，直接计算
void MatrixEx::poolingBackward(Matrix& X, const Matrix& Y, PoolingType pooling_type, int reverse,
    std::vector<Matrix>& workspaces, const std::vector<int>& window, const std::vector<int>& stride,
    const std::vector<int>& padding, float a /*= 1*/
    ,
    float r /*= 0*/)
{
    assert(checkMatrixDevice({ &X, &Y }));
    assert(window.size() >= 2 && stride.size() >= 2 && padding.size() >= 2);
    auto gpu = Y.gpu();
    if (X.isCuda())
    {
        cudnnPoolingDesc pooling_desc;
        cudnnPoolingMode_t pooling_type_c = CUDNN_POOLING_MAX;
        if (pooling_type == POOLING_AVERAGE_NOPADDING) { pooling_type_c = CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING; }
        if (pooling_type == POOLING_AVERAGE_PADDING) { pooling_type_c = CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING; }
        if (window.size() == 2)
        {
            //这个怎么看都快不了
            cudnnSetPooling2dDescriptor(pooling_desc(), pooling_type_c, CUDNN_NOT_PROPAGATE_NAN,
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
            cudnnSetPoolingNdDescriptor(pooling_desc(), pooling_type_c, CUDNN_NOT_PROPAGATE_NAN,
                window.size(), wr.data(), pr.data(), sr.data());
        }
        cudnnStatus_t s;
        if (reverse == 0)
        {
            s = cudnnPoolingBackward(gpu->cudnn_handle_, pooling_desc(),
                &a, Y.cudnn_desc(), Y.data(), Y.cudnn_desc(), Y.d().data(), X.cudnn_desc(), X.data(), &r, X.cudnn_desc(), X.d().data());
        }
        else
        {
            s = cudnnPoolingForward(gpu->cudnn_handle_, pooling_desc(), &a, Y.cudnn_desc(), Y.d().data(), &r, X.cudnn_desc(), X.d().data());
            X.d().scale(1.0 / VectorMath::multiply(window));
        }
        if (s)
        {
            LOG_ERR("POOL backward error {}, {}\n", cudnnGetErrorString(s), X.gpu()->lastCudnnErrorString());
        }
    }
    else if (X.isHip())
    {
        miopenPoolingMode_t pooling_type_c = miopenPoolingMax;
        if (pooling_type == POOLING_AVERAGE_NOPADDING) { pooling_type_c = miopenPoolingAverage; }
        if (pooling_type == POOLING_AVERAGE_PADDING) { pooling_type_c = miopenPoolingAverageInclusive; }
        auto op_desc = gpu->getDesc<miopenPoolingDescriptor_t>();
        if (stride.size() == 2)
        {
            miopenSet2dPoolingDescriptor(op_desc, pooling_type_c, window[1], window[0], padding[1], padding[0], stride[1], stride[0]);
        }
        miopenStatus_t s;
        bool back = gpu->active_phase_ == ACTIVE_PHASE_TRAIN;
        if (reverse == 0)
        {
            s = miopenPoolingBackward(gpu->miopen_handle_, op_desc, &a, Y.miopen_desc(), Y.data(), Y.miopen_desc(), Y.d().data(), X.miopen_desc(), X.data(), &r, X.miopen_desc(), X.d().data(), workspaces[0].data());
        }
        else
        {
            s = miopenPoolingForward(gpu->miopen_handle_, op_desc, &a, Y.miopen_desc(), Y.d().data(), &r, X.miopen_desc(), X.d().data(), back, workspaces[0].data(), workspaces[0].getDataSizeInByte());
            X.d().scale(1.0 / VectorMath::multiply(window));
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
                        float max_v = -FLT_MAX;
                        int64_t max_p = -1;
                        for (int wdX = wdX0; wdX < std::min(X.width_, wdX0 + window[0]); wdX++)
                        {
                            for (int hdX = hdX0; hdX < std::min(X.height_, hdX0 + window[1]); hdX++)
                            {
                                if (X.haveData(wdX, hdX, p, 0))
                                {
                                    float v = X.getData(wdX, hdX, p, 0);
                                    if (v > max_v)
                                    {
                                        max_v = v;
                                        max_p = X.d().whcn2i(wdX, hdX, p, 0);
                                        //max_p = X.d().getDataPtr(wdX, hdX, p, 0);
                                    }
                                }
                            }
                        }
                        if (max_p >= 0)
                        {
                            X.d().setData(max_p, 0, X.d().getData(max_p, 0) + a * Y.d().getData(wdA, hdA, p, 0));
                            //*max_p += a * Y.d().getData(wdA, hdA, p, 0);
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
                        float v = Y.d().getData(wdA, hdA, p, 0) / n;
                        for (int wdX = wdX0; wdX < std::min(X.width_, wdX0 + window[0]); wdX++)
                        {
                            for (int hdX = hdX0; hdX < std::min(X.height_, hdX0 + window[1]); hdX++)
                            {
                                if (X.haveData(wdX, hdX, p, 0))
                                {
                                    X.d().setData(wdX, hdX, p, 0, X.d().getData(wdX, hdX, p, 0) + a * v);    // +r * X.d().getData(wdX, hdX, p, 0);
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
//methods保存信息为3个: 算法，类型，组数。若考虑反向则应为9个，后面对应反向数据和反向权重
void MatrixEx::convolutionForward(const Matrix& X, const Matrix& W, Matrix& Y, std::vector<int>& methods, std::vector<Matrix>& workspaces,
    const std::vector<int>& stride, const std::vector<int>& padding, float a /*= 1*/, float r /*= 0*/)
{
    assert(checkMatrixDevice({ &X, &W, &Y }));
    assert(stride.size() >= 2 && padding.size() >= 2);
    assert(X.getChannel() == W.getChannel() && Y.getChannel() == W.getNumber());

    if (methods.size() < 3)
    {
        methods.resize(3, 0);
    }
    auto gpu = X.gpu();
    if (Y.isCuda())
    {
        cudnnConvolutionDesc conv_desc;
        cudnnFilterDesc filter_desc;
        if (workspaces.size() < 2)
        {
            workspaces.resize(2);
        }
        cudnnStatus_t scd, sfd;
        if (stride.size() == 2)
        {
            scd = cudnnSetConvolution2dDescriptor(conv_desc(), padding[1], padding[0], stride[1], stride[0], 1, 1, CUDNN_CROSS_CORRELATION, toCudnnDataType(Y.getDataType()));
            sfd = cudnnSetFilter4dDescriptor(filter_desc(), toCudnnDataType(W.getDataType()), CUDNN_TENSOR_NCHW, W.number_, W.channel_, W.height_, W.width_);
        }
        else
        {
            //这里有可能需要反序vector，待测试
            std::vector<int> dilation(padding.size(), 1);
            auto pr = padding;
            auto sr = stride;
            std::reverse(pr.begin(), pr.end());
            std::reverse(sr.begin(), sr.end());
            scd = cudnnSetConvolutionNdDescriptor(conv_desc(), padding.size(), pr.data(), sr.data(), dilation.data(), CUDNN_CROSS_CORRELATION, toCudnnDataType(Y.getDataType()));
            auto w_dim = W.getDim();
            std::reverse(w_dim.begin(), w_dim.end());
            sfd = cudnnSetFilterNdDescriptor(filter_desc(), toCudnnDataType(Y.getDataType()), CUDNN_TENSOR_NCHW, W.getDimSize(), w_dim.data());
        }
        cudnnSetConvolutionMathType(conv_desc(), CUDNN_TENSOR_OP_MATH);
        //寻找最快的算法
        if (methods[2] <= 0)
        {
            size_t ws_size;
            if (0)
            {
                cudnnGetConvolutionForwardWorkspaceSize(gpu->cudnn_handle_, X.cudnn_desc(), filter_desc(), conv_desc(), Y.cudnn_desc(), (cudnnConvolutionFwdAlgo_t)0, &ws_size);
                methods[0] = 0;
                methods[1] = 0;
            }
            else
            {
                int n;
                cudnnConvolutionFwdAlgoPerf_t cfap[conv_method_count];
                auto t = cudnnFindConvolutionForwardAlgorithm(gpu->cudnn_handle_, X.cudnn_desc(),
                    filter_desc(), conv_desc(), Y.cudnn_desc(), conv_method_count, &n, cfap);

                //for (int i = 0; i < 8; i++)
                //{
                //    //该算法小组数可能需要更大缓冲区，此处是测试代码
                //    if (cfap[i].algo == CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM)
                //    {
                //        std::swap(cfap[0], cfap[i]);
                //    }
                //}
                methods[0] = cfap[0].algo;
                methods[1] = cfap[0].mathType;
                ws_size = cfap[0].memory;
            }
            if (ws_size > workspaces[0].getDataSizeInByte())
            {
                workspaces[0].resize(1, 1, 1, ws_size / workspaces[0].getDataTypeSize() + 2);
            }
#ifdef _DEBUG
            LOG("conv forward choose {}({}), workspace {}\n", methods[0], methods[1], ws_size);
#endif
        }
        else if (methods[2] != X.number_)    //有可能存在组数小反而需要缓冲区更大的情况
        {
            size_t ws_size;
            cudnnGetConvolutionForwardWorkspaceSize(gpu->cudnn_handle_, X.cudnn_desc(),
                filter_desc(), conv_desc(), Y.cudnn_desc(), cudnnConvolutionFwdAlgo_t(methods[0]), &ws_size);
            if (ws_size > workspaces[0].getDataSizeInByte())
            {
                workspaces[0].resize(1, 1, 1, ws_size / workspaces[0].getDataTypeSize() + 2);
#ifdef _DEBUG
                LOG("resize conv workspace {}\n", ws_size);
#endif
            }
        }
        methods[2] = X.number_;
        auto cfa = cudnnConvolutionFwdAlgo_t(methods[0]);
        auto tensor = cudnnSetConvolutionMathType(conv_desc(), cudnnMathType_t(methods[1]));
        auto scf = cudnnConvolutionForward(gpu->cudnn_handle_, &a, X.cudnn_desc(), X.data(), filter_desc(), W.data(),
            conv_desc(), cfa, workspaces[0].data(), workspaces[0].getDataSizeInByte(), &r, Y.cudnn_desc(), Y.data());
        if (scf)
        {
            LOG_ERR("CONV forward error: status {}, algo {}, {}\n", cudnnGetErrorString(scf), int(cfa), X.gpu()->lastCudnnErrorString());
        }
    }
    else if (Y.isHip())
    {
        if (workspaces.size() < 2)
        {
            workspaces.resize(2);
        }
        miopenStatus_t scd, sfd;
        auto op_desc = gpu->getDesc<miopenConvolutionDescriptor_t>();
        if (stride.size() == 2)
        {
            scd = miopenInitConvolutionDescriptor(op_desc, miopenConvolution, padding[1], padding[0], stride[1], stride[0], 1, 1);
        }
        else
        {
            //这里有可能需要反序vector，待测试
            std::vector<int> dilation(padding.size(), 1);
            auto pr = padding;
            auto sr = stride;
            std::reverse(pr.begin(), pr.end());
            std::reverse(sr.begin(), sr.end());
            scd = miopenInitConvolutionNdDescriptor(op_desc, padding.size(), pr.data(), sr.data(), dilation.data(), miopenConvolution);
            auto w_dim = W.getDim();
            std::reverse(w_dim.begin(), w_dim.end());
        }
        //暂无类型设定
        if (methods[2] <= 0)
        {
            int n;
            size_t ws_size;
            miopenConvAlgoPerf_t cfap[conv_method_count];
            miopenTensorDescriptor_t d1, d2;
            miopenCreateTensorDescriptor(&d1);
            miopenCreateTensorDescriptor(&d2);
            for (int i = 1; i <= X.number_; i++)
            {
                miopenSet4dTensorDescriptor(d1, miopenFloat, i, X.channel_, X.height_, X.width_);
                miopenSet4dTensorDescriptor(d2, miopenFloat, i, Y.channel_, Y.height_, Y.width_);
                miopenFindConvolutionForwardAlgorithm(gpu->miopen_handle_, d1, X.data(), W.miopen_desc(), W.data(),
                    op_desc, d2, Y.data(), conv_method_count, &n, cfap, workspaces[0].data(), workspaces[0].getDataSizeInByte(), true);
            }
            ws_size = cfap[0].memory;
            if (ws_size > workspaces[0].getDataSizeInByte())
            {
                workspaces[0].resize(1, 1, cfap[0].memory / workspaces[0].getDataTypeSize() + 2, 1);
            }
            methods[0] = cfap[0].fwd_algo;
        }
        else if (methods[2] != X.number_)
        {
            size_t ws_size;
            miopenConvolutionForwardGetWorkSpaceSize(gpu->miopen_handle_, W.miopen_desc(), X.miopen_desc(), op_desc, Y.miopen_desc(), &ws_size);
            if (ws_size > workspaces[0].getDataSizeInByte())
            {
                workspaces[0].resize(1, 1, ws_size / workspaces[0].getDataTypeSize() + 2, 1);
            }
        }
        methods[2] = X.number_;
        auto scf = miopenConvolutionForward(gpu->miopen_handle_, &a, X.miopen_desc(), X.data(), W.miopen_desc(), W.data(),
            op_desc, miopenConvFwdAlgorithm_t(methods[0]), &r, Y.miopen_desc(), Y.data(), workspaces[0].data(), workspaces[0].getDataSizeInByte());
        if (scf)
        {
            LOG_ERR("CONV forward error: status {}, methods {}\n", miopenGetErrorString(scf), methods);
        }
    }
    else
    {
        Y.fillData(0);
        //辅助矩阵的尺寸
        int row = Y.width_ * Y.height_;
        int col = X.channel_ * W.width_ * W.height_;
        Matrix X_ex({ row, col }, X.getDataType(), UnitType::CPU);
        if (methods[0] < 1)
        {
            methods[0] = 1;
            workspaces[0].resize(1, 1, row * col, 1);
            workspaces[0].fillData(-1);
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
                        workspaces[0].setData(pX_ex, 0, pX);    //记录展开的位置
                    });
            }
        }
        Matrix X_sub({ X.row_, 1 }, X.data_, X.getDataType(), UnitType::CPU);
        Matrix Y_sub({ Y.width_ * Y.height_, Y.channel_ }, Y.data_, Y.getDataType(), UnitType::CPU);    //这两个是专用于share
        for (int i = 0; i < X.number_; i++)
        {
            X_sub.shareData(X, 0, i);
            Y_sub.shareData(Y, 0, i);
            X_ex.fillData(0);
            for (int j = 0; j < X_ex.getDataSize(); j++)
            {
                int p = workspaces[0].getData(j, 0);
                if (p >= 0 && p < X_sub.getDataSize())
                {
                    X_ex.setData(j, X_sub.getData(p));
                }
            }
            MatrixEx::mul(X_ex, W, Y_sub, a, r);
        }

#ifdef DIRECT_COMPUTE_CONVOLUTION
        //这是原始计算方法，速度很慢，不要打开这段代码
        //LOG(stderr, "Please supply buffer vector and use the faster convolution method.\n");
        Y.fillData(0);
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
    const std::vector<int>& stride, const std::vector<int>& padding, float a /*= 1*/, float rx /*= 0*/, float rw /*= 0*/)
{
    assert(checkMatrixDevice({ &X, &W, &Y }));
    assert(stride.size() >= 2 && padding.size() >= 2);
    assert(methods.size() >= 9 && workspaces.size() >= 3);
    assert(X.getChannel() == W.getChannel() && Y.getChannel() == W.getNumber());

    if (methods.size() < 9)
    {
        methods.resize(9, 0);
    }
    auto methods_x = &methods[3];
    auto methods_w = &methods[6];
    auto gpu = Y.gpu();
    //这里不用dX判断是因为dX可能是空
    if (Y.isCuda())
    {
        cudnnConvolutionDesc conv_desc;
        cudnnFilterDesc filter_desc;
        if (stride.size() == 2)
        {
            cudnnSetConvolution2dDescriptor(conv_desc(), padding[1], padding[0], stride[1], stride[0], 1, 1, CUDNN_CROSS_CORRELATION, toCudnnDataType(Y.getDataType()));
            cudnnSetFilter4dDescriptor(filter_desc(), toCudnnDataType(Y.getDataType()), CUDNN_TENSOR_NCHW, W.number_, W.channel_, W.height_, W.width_);
        }
        else
        {
            std::vector<int> dilation(padding.size(), 1);
            auto pr = padding;
            auto sr = stride;
            std::reverse(pr.begin(), pr.end());
            std::reverse(sr.begin(), sr.end());
            cudnnSetConvolutionNdDescriptor(conv_desc(), padding.size(), pr.data(), sr.data(), dilation.data(), CUDNN_CROSS_CORRELATION, toCudnnDataType(Y.getDataType()));
            auto w_dim = W.getDim();
            std::reverse(w_dim.begin(), w_dim.end());
            cudnnSetFilterNdDescriptor(filter_desc(), toCudnnDataType(Y.getDataType()), CUDNN_TENSOR_NCHW, W.getDimSize(), w_dim.data());
        }
        cudnnSetConvolutionMathType(conv_desc(), CUDNN_TENSOR_OP_MATH);
        if (X.needBack())
        {
            //寻找最快的算法
            if (methods_x[2] <= 0)
            {
                int n;
                cudnnConvolutionBwdDataAlgoPerf_t cbdap[conv_method_count];
                auto t = cudnnFindConvolutionBackwardDataAlgorithm(gpu->cudnn_handle_, filter_desc(),
                    Y.cudnn_desc(), conv_desc(), X.cudnn_desc(), conv_method_count, &n, cbdap);
                methods_x[0] = cbdap[0].algo;
                methods_x[1] = cbdap[0].mathType;
                size_t ws_size = cbdap[0].memory;
                if (ws_size > workspaces[0].getDataSizeInByte())
                {
                    workspaces[0].resize(1, 1, 1, ws_size / workspaces[0].getDataTypeSize() + 2);
                    //LOG("resize work space {}\n", memory);
                }
#ifdef _DEBUG
                LOG("conv backward X choose {}({}), workspace {}\n", methods_x[0], methods_x[1], cbdap[0].memory);
#endif
                //workspace->message();
            }
            else if (methods_x[2] != X.number_)
            {
                size_t ws_size;
                cudnnGetConvolutionBackwardDataWorkspaceSize(gpu->cudnn_handle_, filter_desc(),
                    Y.cudnn_desc(), conv_desc(), X.cudnn_desc(), cudnnConvolutionBwdDataAlgo_t(methods_x[0]), &ws_size);
                if (ws_size > workspaces[0].getDataSizeInByte())
                {
                    workspaces[0].resize(1, 1, 1, ws_size / workspaces[0].getDataTypeSize() + 2);
#ifdef _DEBUG
                    LOG("resize conv workspace {}\n", ws_size);
#endif
                }
            }
            methods_x[2] = X.number_;
            auto cbda = cudnnConvolutionBwdDataAlgo_t(methods_x[0]);
            auto tensor = cudnnSetConvolutionMathType(conv_desc(), cudnnMathType_t(methods_x[1]));
            auto scbx = cudnnConvolutionBackwardData(gpu->cudnn_handle_, &a, filter_desc(), W.data(), Y.cudnn_desc(), Y.d().data(),
                conv_desc(), cbda, workspaces[0].data(), workspaces[0].getDataSizeInByte(), &rx, X.cudnn_desc(), X.d().data());
            if (scbx)
            {
                LOG_ERR("CONV backward data error: status {}, {}\n", cudnnGetErrorString(scbx), X.gpu()->lastCudnnErrorString());
            }
        }
        if (W.needBack())
        {
            //寻找最快的算法
            if (methods_w[2] <= 0)
            {
                int n;
                cudnnConvolutionBwdFilterAlgoPerf_t cbfap[conv_method_count];
                cudnnFindConvolutionBackwardFilterAlgorithm(gpu->cudnn_handle_, X.cudnn_desc(), Y.cudnn_desc(),
                    conv_desc(), filter_desc(), conv_method_count, &n, cbfap);
                methods_w[0] = cbfap[0].algo;
                methods_w[1] = cbfap[0].mathType;
                size_t memory = cbfap[0].memory;
                if (memory > workspaces[0].getDataSizeInByte())
                {
                    workspaces[0].resize(1, 1, 1, memory / workspaces[0].getDataTypeSize() + 2);
                    //LOG("resize work space {}\n", memory);
                }
#ifdef _DEBUG
                LOG("conv backward W choose {}({}), workspace {}\n", methods_w[0], methods_w[1], cbfap[0].memory);
#endif
                //workspace->message();
            }
            else if (methods_w[2] != X.number_)
            {
                size_t memory;
                cudnnGetConvolutionBackwardFilterWorkspaceSize(gpu->cudnn_handle_, X.cudnn_desc(), Y.cudnn_desc(),
                    conv_desc(), filter_desc(), cudnnConvolutionBwdFilterAlgo_t(methods_w[0]), &memory);
                if (memory > workspaces[0].getDataSizeInByte())
                {
                    workspaces[0].resize(1, 1, 1, memory / workspaces[0].getDataTypeSize() + 2);
#ifdef _DEBUG
                    LOG("resize conv workspace {}\n", memory);
#endif
                }
            }
            methods_w[2] = X.number_;
            auto cbfa = cudnnConvolutionBwdFilterAlgo_t(methods_w[0]);
            auto tensor = cudnnSetConvolutionMathType(conv_desc(), cudnnMathType_t(methods_w[1]));
            auto scbw = cudnnConvolutionBackwardFilter(gpu->cudnn_handle_, &a, X.cudnn_desc(), X.data(), Y.cudnn_desc(), Y.d().data(),
                conv_desc(), cbfa, workspaces[0].data(), workspaces[0].getDataSizeInByte(), &rw, filter_desc(), W.d().data());
            if (scbw)
            {
                LOG_ERR("CONV backward weight error: status {}, {}\n", cudnnGetErrorString(scbw), X.gpu()->lastCudnnErrorString());
            }
        }
        //if (B.needUpdate())
        //{
        //    //这个似乎也可以用于一般情况下bias的反向，待查，unfinished
        //    auto scbb = cudnnConvolutionBackwardBias(cuda->cudnn_handle_, &a, R.getCudnnTensorDesc(), R.DMatrix().data(), &r, B.getCudnnTensorDesc(), B.DMatrix().data());
        //}
    }
    else if (Y.isHip())
    {
        auto op_desc = gpu->getDesc<miopenConvolutionDescriptor_t>();
        if (stride.size() == 2)
        {
            miopenInitConvolutionDescriptor(op_desc, miopenConvolution, padding[1], padding[0], stride[1], stride[0], 1, 1);
        }
        else
        {
            std::vector<int> dilation(padding.size(), 1);
            auto pr = padding;
            auto sr = stride;
            std::reverse(pr.begin(), pr.end());
            std::reverse(sr.begin(), sr.end());
            miopenInitConvolutionNdDescriptor(op_desc, padding.size(), pr.data(), sr.data(), dilation.data(), miopenConvolution);
            auto w_dim = W.getDim();
            std::reverse(w_dim.begin(), w_dim.end());
        }
        if (X.needBack())
        {
            if (methods_x[2] <= 0)
            {
                int n;
                size_t ws_size;
                miopenConvAlgoPerf_t cfap[conv_method_count];
                miopenFindConvolutionBackwardDataAlgorithm(gpu->miopen_handle_, Y.miopen_desc(), Y.d().data(), W.miopen_desc(), W.data(),
                    op_desc, X.miopen_desc(), X.d().data(), conv_method_count, &n, cfap, workspaces[0].data(), workspaces[0].getDataSizeInByte(), true);
                ws_size = cfap[0].memory;
                if (ws_size > workspaces[0].getDataSizeInByte())
                {
                    workspaces[0].resize(1, 1, cfap[0].memory / workspaces[0].getDataTypeSize() + 2, 1);
                }
                methods_x[0] = cfap[0].bwd_data_algo;
            }
            else if (methods_x[2] != X.number_)
            {
                size_t ws_size;
                miopenConvolutionBackwardDataGetWorkSpaceSize(gpu->miopen_handle_, Y.miopen_desc(), W.miopen_desc(), op_desc, X.miopen_desc(), &ws_size);
                if (ws_size > workspaces[0].getDataSizeInByte())
                {
                    workspaces[0].resize(1, 1, ws_size / workspaces[0].getDataTypeSize() + 2, 1);
                }
            }
            methods_x[2] = X.number_;
            auto scf = miopenConvolutionBackwardData(gpu->miopen_handle_, &a, Y.miopen_desc(), Y.d().data(), W.miopen_desc(), W.data(),
                op_desc, miopenConvBwdDataAlgorithm_t(methods_x[0]), &rx, X.miopen_desc(), X.d().data(), workspaces[0].data(), workspaces[0].getDataSizeInByte());
        }
        if (W.needBack())
        {
            if (methods_w[2] <= 0)
            {
                int n;
                size_t ws_size;
                miopenConvAlgoPerf_t cfap[conv_method_count];
                miopenFindConvolutionBackwardWeightsAlgorithm(gpu->miopen_handle_, Y.miopen_desc(), Y.d().data(), X.miopen_desc(), X.data(),
                    op_desc, W.miopen_desc(), W.d().data(), conv_method_count, &n, cfap, workspaces[0].data(), workspaces[0].getDataSizeInByte(), true);
                ws_size = cfap[0].memory;
                if (ws_size > workspaces[0].getDataSizeInByte())
                {
                    workspaces[0].resize(1, 1, cfap[0].memory / workspaces[0].getDataTypeSize() + 2, 1);
                }
                methods_w[0] = cfap[0].bwd_weights_algo;
            }
            else if (methods_w[2] != X.number_)
            {
                size_t ws_size;
                miopenConvolutionBackwardWeightsGetWorkSpaceSize(gpu->miopen_handle_, Y.miopen_desc(), X.miopen_desc(), op_desc, W.miopen_desc(), &ws_size);
                if (ws_size > workspaces[0].getDataSizeInByte())
                {
                    workspaces[0].resize(1, 1, ws_size / workspaces[0].getDataTypeSize() + 2, 1);
                }
            }
            methods_w[2] = X.number_;
            auto scf = miopenConvolutionBackwardWeights(gpu->miopen_handle_, &a, Y.miopen_desc(), Y.d().data(), X.miopen_desc(), X.data(),
                op_desc, miopenConvBwdWeightsAlgorithm_t(methods_w[0]), &rw, W.miopen_desc(), W.d().data(), workspaces[0].data(), workspaces[0].getDataSizeInByte());
        }
    }
    else
    {
        if (X.needBack())
        {
            //计算dX从数学上来看可以反向来求展开后的dX，再压缩，但是看起来加法次数较多
            //转置W的输入和输出
            Matrix W2({ W.width_, W.height_, W.number_, W.channel_ }, W.getDataType(), UnitType::CPU);
            for (int i = 0; i < W.channel_; i++)
            {
                for (int j = 0; j < W.number_; j++)
                {
                    copyDataPtr(W, W.getDataPtr(0, 0, i, j), W2, W2.getDataPtr(0, 0, j, i), W.width_ * W.height_);
                }
            }
            //辅助矩阵的尺寸
            int row = X.width_ * X.height_;
            int col = Y.channel_ * W.width_ * W.height_;
            Matrix dY_ex({ row, col }, DataType::CURRENT, UnitType::CPU);
            dY_ex.fillData(0);
            if (methods_x[0] < 1)
            {
                methods_x[0] = 1;
                workspaces[1].resize(1, 1, row * col, 1);
                workspaces[1].fillData(-1);
                for (int cY = 0; cY < Y.channel_; cY++)
                {
                    CONV_OPERATION1(X, W, Y, padding[0], padding[1], stride[0], stride[1],
                        {
                            int pX = X.whcn2i(wX, hX, 0, 0);
                            int pW = W2.whcn2i(wW, hW, cY, 0);    //这里用W或者W2没有区别
                            int pY = Y.whcn2i(wY, hY, cY, 0);     //拍扁
                            int p_ex = dY_ex.mn2i(pX, pW);
                            workspaces[1].setData(p_ex, 0, pY);
                        });
                }
            }
            Matrix dY_sub({ Y.row_, 1 }, Y.getDataType(), UnitType::CPU);
            Matrix dX_sub({ X.width_ * X.height_, X.channel_ }, X.getDataType(), UnitType::CPU);
            for (int i = 0; i < Y.number_; i++)
            {
                dY_sub.shareData(Y.d(), 0, i);
                dX_sub.shareData(X.d(), 0, i);
                for (int j = 0; j < dY_ex.getDataSize(); j++)
                {
                    if (workspaces[1].getData(j, 0) >= 0)
                    {
                        dY_ex.setData(j, dY_sub.getData(workspaces[1].getData(j, 0)));
                    }
                }
                MatrixEx::mul(dY_ex, W2, dX_sub, a, rx);
            }
        }
        //暂时如此写，看情况能否跟上面合并
        if (W.needBack())
        {
            //W.d().scale(rw);
            //辅助矩阵的尺寸
            int row = W.width_ * W.height_ * W.channel_;
            int col = Y.width_ * Y.height_;
            Matrix X_ex({ row, col }, X.getDataType(), UnitType::CPU);
            X_ex.fillData(0);
            if (methods_w[0] < 1)
            {
                methods_w[0] = 1;
                workspaces[2].resize(1, 1, row * col, 1);
                workspaces[2].fillData(-1);
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
                            workspaces[2].setData(p_ex, 0, pX);
                        });
                }
            }
            Matrix dY_sub({ Y.width_ * Y.height_, Y.channel_ }, Y.getDataType(), UnitType::CPU);
            Matrix X_sub({ X.row_, 1 }, X.getDataType(), UnitType::CPU);
            for (int i = 0; i < Y.number_; i++)
            {
                dY_sub.shareData(Y.d(), 0, i);
                X_sub.shareData(X, 0, i);
                for (int j = 0; j < X_ex.getDataSize(); j++)
                {
                    //if ((*ex2)[j] >= 0) //因为是满的不需要
                    X_ex.setData(j, X_sub.getData(workspaces[2].getData(j, 0)));
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
        //            B.DMatrix().getData(0, 0, c, 0) += a * VectorMath::sum(R.DMatrix().getDataPtr(0, 0, c, n), R.width_ * R.height_);
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

//随机让一些点不参与计算
void MatrixEx::dropoutForward(const Matrix& X, Matrix& Y, ActivePhaseType work_phase, float v, int seed, Matrix& rg_stat, Matrix& reverse_space)
{
    assert(checkMatrixDevice({ &X, &Y }));
    if (work_phase == ACTIVE_PHASE_TEST)
    {
        Matrix::copyData(X, Y);
        Y.scale(v);
        return;
    }
    auto gpu = X.gpu();
    int op_desc_buf[64] = { 0 };
    auto op_desc = (cudnnDropoutDescriptor_t)op_desc_buf;
    if (Y.isCuda())
    {
        cudnnSetDropoutDescriptor(op_desc, gpu->cudnn_handle_, v, rg_stat.data(), rg_stat.getDataSizeInByte(), seed);
        cudnnDropoutForward(gpu->cudnn_handle_, op_desc, X.cudnn_desc(), X.data(),
            Y.cudnn_desc(), Y.data(), reverse_space.data(), reverse_space.getDataSizeInByte());
    }
    else
    {
        Random<float> r;
        r.set_seed(seed);
        for (int i = 0; i < Y.data_size_; i++)
        {
            if (r.rand() < v)
            {
                Y.setData(i, 0);
            }
            else
            {
                Y.setData(i, X.getData(i) / (1 - v));
            }
        }
    }
}

void MatrixEx::dropoutBackward(Matrix& X, const Matrix& Y, float v, int seed, Matrix& rg_stat, Matrix& reverse_space)
{
    assert(checkMatrixDevice({ &X, &Y }));
    auto gpu = Y.gpu();
    int op_desc_buf[64] = { 0 };
    auto op_desc = (cudnnDropoutDescriptor_t)op_desc_buf;
    if (Y.isCuda())
    {
        //这里的seed应该是没关系，待查
        cudnnSetDropoutDescriptor(op_desc, gpu->cudnn_handle_, v, rg_stat.data(), rg_stat.getDataSizeInByte(), seed);
        cudnnDropoutBackward(gpu->cudnn_handle_, op_desc, Y.cudnn_desc(), Y.d().data(),
            X.cudnn_desc(), X.d().data(), reverse_space.data(), reverse_space.getDataSizeInByte());
    }
    else
    {
        for (int i = 0; i < X.data_size_; i++)
        {
            if (Y.getData(i) == 0)
            {
                X.d().setData(i, 0);
            }
            else
            {
                X.d().setData(i, Y.d().getData(i) / (1 - v));
            }
        }
    }
}

//批归一化
void MatrixEx::batchNormalizationForward(const Matrix& X, Matrix& Y,
    ActivePhaseType work_phase, BatchNormalizationType bn_type, float& exp_aver_factor, float epsilon, Matrix& scale, Matrix& bias,
    Matrix& result_running_mean, Matrix& result_running_variance, Matrix& result_save_mean, Matrix& result_save_inv_variance)
{
    assert(checkMatrixDevice({ &X, &Y }));
    auto gpu = X.gpu();
    if (Y.isCuda())
    {
        if (work_phase == ACTIVE_PHASE_TRAIN)
        {
            cudnnBatchNormalizationForwardTraining(gpu->cudnn_handle_, cudnnBatchNormMode_t(bn_type),
                &const_real_1, &const_real_0, X.cudnn_desc(), X.data(), Y.cudnn_desc(), Y.data(),
                scale.cudnn_desc(), scale.data(), bias.data(), exp_aver_factor, result_running_mean.data(), result_running_variance.data(),
                std::max(epsilon * 1.0, CUDNN_BN_MIN_EPSILON), result_save_mean.data(), result_save_inv_variance.data());
            exp_aver_factor = 1 / (1 / exp_aver_factor + 1);
        }
        if (work_phase == ACTIVE_PHASE_TEST)
        {
            cudnnBatchNormalizationForwardInference(gpu->cudnn_handle_, cudnnBatchNormMode_t(bn_type),
                &const_real_1, &const_real_0, X.cudnn_desc(), X.data(), Y.cudnn_desc(), Y.data(),
                scale.cudnn_desc(), scale.data(), bias.data(), result_running_mean.data(), result_running_variance.data(),
                std::max(epsilon * 1.0, CUDNN_BN_MIN_EPSILON));
        }
    }
}

void MatrixEx::batchNormalizationBackward(Matrix& X, const Matrix& Y,
    ActivePhaseType work_phase, BatchNormalizationType bn_type, float epsilon, float rate, Matrix& scale, Matrix& bias,
    Matrix& saved_mean, Matrix& saved_inv_variance, Matrix& result_dscale, Matrix& result_dbias)
{
    assert(checkMatrixDevice({ &X, &Y }));
    auto gpu = Y.gpu();
    if (X.isCuda())
    {
        cudnnBatchNormalizationBackward(gpu->cudnn_handle_, cudnnBatchNormMode_t(bn_type),
            &const_real_1, &const_real_0, &const_real_1, &const_real_0, X.cudnn_desc(), X.data(), Y.cudnn_desc(), Y.d().data(), X.cudnn_desc(), X.d().data(),
            scale.cudnn_desc(), scale.data(), result_dscale.data(), result_dbias.data(),
            std::max(epsilon * 1.0, CUDNN_BN_MIN_EPSILON), saved_mean.data(), saved_inv_variance.data());
        MatrixEx::add(scale, result_dscale, scale, 1, -rate);
        MatrixEx::add(bias, result_dbias, bias, 1, -rate);
    }
}

//ada_d表示ada方法计算得到的参数的改变量，应在下一步加到原参数上，下同
void MatrixEx::adaDeltaUpdate(Matrix& mean_d2, Matrix& mean_ada_d2, Matrix& d, Matrix& ada_d, float rou, float epsilon)
{
    assert(checkMatrixDevice({ &mean_d2, &mean_ada_d2, &d, &ada_d }));
    if (d.isCuda())
    {
        cuda_ada_delta_update(mean_d2.getDataTypeByInt(), mean_d2.data(), mean_ada_d2.data(), d.data(), ada_d.data(), d.data_size_, rou, epsilon);
    }
    else if (d.isHip())
    {
        hip_ada_delta_update(mean_d2.data(), mean_ada_d2.data(), d.data(), ada_d.data(), d.data_size_, rou, epsilon);
    }
    else
    {
        auto& p1 = mean_d2;
        auto& p2 = mean_ada_d2;
        auto& p3 = d;
        auto& p4 = ada_d;
        for (int i = 0; i < d.data_size_; i++)
        {
            p1.setData(i, p1.getData(i) * rou + p3.getData(i) * p3.getData(i) * (1 - rou));
            p4.setData(i, p3.getData(i) * sqrt((p2.getData(i) + epsilon) / (p1.getData(i) + epsilon)));
            p2.setData(i, p2.getData(i) * rou + p4.getData(i) * p4.getData(i) * (1 - rou));
        }
    }
}

void MatrixEx::adamUpdate(Matrix& mean_d, Matrix& mean_d2, Matrix& d, Matrix& ada_d, float beta1, float beta2, float epsilon, float t)
{
    assert(checkMatrixDevice({ &mean_d, &mean_d2, &d, &ada_d }));
    if (d.isCuda())
    {
        cuda_adam_update(mean_d.getDataTypeByInt(), mean_d.data(), mean_d2.data(), d.data(), ada_d.data(), d.data_size_, beta1, beta2, epsilon, t);
    }
    else if (d.isHip())
    {
        hip_adam_update(mean_d.data(), mean_d2.data(), d.data(), ada_d.data(), d.data_size_, beta1, beta2, epsilon, t);
    }
    else
    {
        auto& p1 = mean_d;
        auto& p2 = mean_d2;
        auto& p3 = d;
        auto& p4 = ada_d;
        for (int i = 0; i < d.data_size_; i++)
        {
            p1.setData(i, p1.getData(i) * beta1 + p3.getData(i) * (1 - beta1));
            p2.setData(i, p2.getData(i) * beta2 + p3.getData(i) * p3.getData(i) * (1 - beta2));
            p4.setData(i, p1.getData(i) / (1 - pow(beta1, t)) / (sqrt(p2.getData(i) / (1 - pow(beta2, t))) + epsilon));
        }
    }
}

void MatrixEx::adaRMSPropUpdate(Matrix& mean_d2, Matrix& d, Matrix& ada_d, float rou, float epsilon)
{
    assert(checkMatrixDevice({ &mean_d2, &d, &ada_d }));
    if (d.isCuda())
    {
        cuda_rms_prop_update(mean_d2.getDataTypeByInt(), mean_d2.data(), d.data(), ada_d.data(), d.data_size_, rou, epsilon);
    }
    else
    {
        auto& p1 = mean_d2;
        auto& p2 = d;
        auto& p3 = ada_d;
        for (int i = 0; i < d.data_size_; i++)
        {
            p1.setData(i, p1.getData(i) * rou + p2.getData(i) * p2.getData(i) * (1 - rou));
            p3.setData(i, p2.getData(i) * sqrt(1.0 / (p1.getData(i) + epsilon)));
        }
    }
}

//R = ((1-rou)/(1-rou_hat)-rou/rou_hat)*beta
void MatrixEx::sparse(Matrix& rou_hat, Matrix& R, float rou, float beta)
{
    if (rou_hat.isCuda())
    {
        cuda_sparse(R.getDataTypeByInt(), rou_hat.data(), R.data(), R.data_size_, rou, beta);
    }
    else
    {
        for (int i = 0; i < R.data_size_; i++)
        {
            R.setData(i, ((1 - rou) / (1 - rou_hat.getData(i)) - rou / rou_hat.getData(i)) * beta);
        }
    }
}

void MatrixEx::fill(Matrix& m, RandomFillType random_type, int in, int out)
{
    if (m.getDataSize() <= 0)
    {
        return;
    }
    Random<float> random_generator;
    random_generator.set_seed();
    float a = 0, b = 0;

    switch (random_type)
    {
    case RANDOM_FILL_CONSTANT:
        m.fillData(0);
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
        break;
    case RANDOM_FILL_MSRA:
        random_generator.set_random_type(RANDOM_NORMAL);
        a = sqrt(2.0 / in);
        random_generator.set_parameter(0, a);
        break;
    case RANDOM_FILL_LECUN:
        random_generator.set_random_type(RANDOM_NORMAL);
        a = 1.0 / in;
        random_generator.set_parameter(0, a);
        break;
    default:
        break;
    }
    std::vector<float> temp(m.getDataSize());
    random_generator.rand_data(temp.data(), temp.size());

    switch (m.getDataType())
    {
    case DataType::FLOAT:
        m.loadDataPtr(temp.data(), temp.size());
        break;
    case DataType::DOUBLE:
    {
        std::vector<double> temp1(m.getDataSize());
        for (int i = 0; i < m.getDataSize(); i++)
        {
            temp1[i] = temp[i];
        }
        m.loadDataPtr(temp1.data(), temp1.size());
    }
    break;
    case DataType::HALF:
    {
        std::vector<half> temp1(m.getDataSize());
        for (int i = 0; i < m.getDataSize(); i++)
        {
            temp1[i] = temp[i];
        }
        m.loadDataPtr(temp1.data(), temp1.size());
    }
    break;
    }
    //m.message();
}

void MatrixEx::sin(const Matrix& X, Matrix& Y, float a /*= 1*/)
{
    assert(checkMatrixDevice({ &X, &Y }));
    if (X.isCuda())
    {
        cuda_sin(X.getDataTypeByInt(), X.data(), Y.data(), X.getDataSize(), a, 0);
    }
    else
    {
        for (int i = 0; i < Y.data_size_; i++)
        {
            Y.setData(i, ::sin(a * X.getData(i)));
        }
    }
}

void MatrixEx::cos(const Matrix& X, Matrix& Y, float a /*= 1*/)
{
    assert(checkMatrixDevice({ &X, &Y }));
    if (X.isCuda())
    {
        cuda_cos(X.getDataTypeByInt(), X.data(), Y.data(), X.getDataSize(), a, 0);
    }
    else
    {
        for (int i = 0; i < Y.data_size_; i++)
        {
            Y.setData(i, ::cos(a * X.getData(i)));
        }
    }
}

void MatrixEx::zigzag(const Matrix& X, Matrix& Y)
{
    assert(checkMatrixDevice({ &X, &Y }));
    if (X.isCuda())
    {
        cuda_zigzag(X.getDataTypeByInt(), X.data(), Y.data(), X.getDataSize(), 1, 0);
    }
    else
    {
        for (int i = 0; i < Y.data_size_; i++)
        {
            auto x = X.getData(i);
            Y.setData(i, x - 2 * floor((x - 1) / 2) - 2);
        }
    }
}

//实际上这个激活函数在奇异点不连续，无法训练
void MatrixEx::zigzagb(Matrix& X, const Matrix& Y)
{
    assert(checkMatrixDevice({ &X, &Y }));
    if (X.isCuda())
    {
        cuda_zigzagb(X.getDataTypeByInt(), X.data(), X.d().data(), Y.data(), Y.d().data(), X.getDataSize(), 1, 0);
    }
    else
    {
        auto& p1 = Y;
        auto& p2 = Y.d();
        auto& p3 = X.d();
        for (int i = 0; i < Y.data_size_; i++)
        {
            if (abs(p1.getData(i)) > 1 - 1e-2)
            {
                p3.setData(i, -p2.getData(i) * 100);
                continue;
            }
            p3.setData(i, p2.getData(i));
        }
    }
}

void MatrixEx::step(const Matrix& X, Matrix& Y)
{
    assert(checkMatrixDevice({ &X, &Y }));
    if (X.isCuda())
    {
        cuda_step(X.getDataTypeByInt(), X.data(), Y.data(), X.getDataSize(), 0, 0);
    }
    else
    {
        //未完成
    }
}

void MatrixEx::leaky_relu(const Matrix& X, Matrix& Y, float l, float a /*= 1*/, float b /*= 0*/)
{
    assert(checkMatrixDevice({ &X, &Y }));
    if (X.isCuda())
    {
        cuda_leaky_relu(X.getDataTypeByInt(), X.data(), Y.data(), X.getDataSize(), l, 1, 0);
    }
    else
    {
        //未完成
    }
}

void MatrixEx::leaky_relub(Matrix& X, const Matrix& Y, float l, float a /*= 1*/, float b /*= 0*/)
{
    assert(checkMatrixDevice({ &X, &Y }));
    if (X.isCuda())
    {
        cuda_leaky_relub(X.getDataTypeByInt(), X.data(), X.d().data(), Y.data(), Y.d().data(), X.getDataSize(), l, 1, 0);
    }
    else
    {
        //未完成
    }
}

void MatrixEx::correlationForward(const Matrix& X, const Matrix& W, Matrix& Y, std::vector<int>& methods, std::vector<Matrix>& workspaces, const std::vector<int>& stride, const std::vector<int>& padding, float a /*= 1*/, float r /*= 0*/)
{
    float epsilon = 1e-4;

    int need_resize = 1;
    if (workspaces.size() < 16)
    {
        workspaces.resize(16);
        need_resize = 0;
    }
    if (X.getNumber() != workspaces[3].getNumber())
    {
        need_resize = 0;
    }

    auto& Y_den_square = workspaces[3];
    auto& Y_den = workspaces[4];
    auto& Y_aver = workspaces[5];
    auto& Y_conv_aver = workspaces[6];

    auto& W_den_square = workspaces[7];
    auto& W_den = workspaces[8];
    auto& W_aver = workspaces[9];
    auto& W_minus_aver = workspaces[10];
    auto& W_norm = workspaces[11];

    auto& W_1 = workspaces[12];    //与W同维，但是所有值为1
    auto& as_W_aver = workspaces[13];

    auto& X1 = workspaces[14];    //与X同维，但是所有值为1
    auto& X_square = workspaces[15];

    if (need_resize == 0)
    {
        //以下为初始化

        for (auto& m : { &Y_den_square, &Y_den, &Y_aver, &Y_conv_aver })
        {
            (*m).resize(Y.getDim());
            (*m).fillData(0);
        }

        for (auto& m : { &W_den_square, &W_den, &W_aver, &W_minus_aver, &W_norm })
        {
            (*m).resize(W.getDim());
            (*m).fillData(0);
        }

        as_W_aver.resize(W.getRow(), W.getRow());
        as_W_aver.fillData(1);

        W_1.resize(W.getDim());
        W_1.fillData(1);
        X1.resize(X.getDim());
        X1.fillData(1);
        X_square.resize(X.getDim());
    }

    //计算W相关
    Matrix::mul(as_W_aver, W, W_aver, 1.0 / W.getRow());    //计算平均值
    Matrix::add(W, W_aver, W_minus_aver, 1, -1);            //原值减去平均值
    Matrix::elementMul(W_minus_aver, W_minus_aver, W_den);
    Matrix::mul(as_W_aver, W_den, W_den_square);
    W_den_square.addNumber(epsilon);    //以上计算分母上的平方和，加上一个小量避免为0，利用W_den保存中间值

    Matrix::elementPow(W_den_square, W_den, 0.5);    //计算分母上的平方根
    Matrix::elementDiv(W, W_den, W_norm);            //计算归一化之后的核

    //计算X相关
    Matrix::elementMul(X, X, X_square);                                                                      //X_square平方
    MatrixEx::convolutionForward(X_square, W_1, Y_den_square, methods, workspaces, stride, padding, 1);      //Y_den_square临时保存平方和
    MatrixEx::convolutionForward(X, W_1, Y_aver, methods, workspaces, stride, padding, 1.0 / W.getRow());    //Y_aver保存平均值
    Matrix::elementMul(Y_aver, Y_aver, Y_den);                                                               //Y_den临时保存平均值平方
    Matrix::add(Y_den_square, Y_den, Y_den_square, 1, -W.getRow());
    Y_den_square.addNumber(epsilon);
    Matrix::elementPow(Y_den_square, Y_den, 0.5);    //Y_den_square，Y_den计算完毕

    //计算原图与归一化核的卷积，并除以分母
    MatrixEx::convolutionForward(X, W_norm, Y, methods, workspaces, stride, padding);               //X与归一化后W的卷积
    MatrixEx::convolutionForward(X1, W_norm, Y_conv_aver, methods, workspaces, stride, padding);    //X的元素全为1时与W的卷积
    Matrix::elementMul(Y_conv_aver, Y_aver, Y_conv_aver);                                           //元素乘平均值
    Matrix::add(Y, Y_conv_aver, Y, 1, -1);                                                          //相减计算出分子的前半部分

    Matrix::elementDiv(Y, Y_den, Y);
}

void MatrixEx::correlationBackward(Matrix& X, Matrix& W, const Matrix& Y, std::vector<int>& methods, std::vector<Matrix>& workspaces, const std::vector<int>& stride, const std::vector<int>& padding, float a /*= 1*/, float rx /*= 0*/, float rw /*= 0*/)
{
    //注意仅有W的反向，即只能为首层
    auto& Y_den_square = workspaces[3];
    auto& Y_den = workspaces[4];
    auto& Y_aver = workspaces[5];
    auto& Y_conv_aver = workspaces[6];

    auto& W_den_square = workspaces[7];
    auto& W_den = workspaces[8];
    auto& W_aver = workspaces[9];
    auto& W_minus_aver = workspaces[10];
    auto& W_norm = workspaces[11];

    auto& W_1 = workspaces[12];    //与W同维，但是所有值为1
    auto& as_W_aver = workspaces[13];

    auto& X1 = workspaces[14];    //与X同维，但是所有值为1
    auto& X_square = workspaces[15];

    if (W.needBack())
    {
        Matrix::elementDiv(Y.d(), Y_den, Y_den.d());
        X.setNeedBack(false);
        MatrixEx::convolutionBackward(X, W, Y_den, methods, workspaces, stride, padding, 1 - 1.0 / W.getRow(), 0, 0);
        Matrix::elementMul(Y_den.d(), Y_aver, Y_aver.d());
        X1.setNeedBack(false);
        MatrixEx::convolutionBackward(X1, W, Y_aver, methods, workspaces, stride, padding, -(1 - 1.0 / W.getRow()), 0, 1);
        Matrix::elementDiv(W.d(), W_den, W.d());

        Matrix::elementMul(Y.d(), Y, Y_den.d());
        //应是反向卷积再元素乘
        MatrixEx::convolutionBackward(X1, W_norm, Y_den, methods, workspaces, stride, padding);
        Matrix::elementMul(W_norm.d(), W_minus_aver, W_norm.d());
        Matrix::elementDiv(W_norm.d(), W_den_square, W_norm.d());
        Matrix::add(W.d(), W_norm.d(), W.d(), a, -a);
    }
}

void MatrixEx::matrix_max(const Matrix& X1, const Matrix& X2, Matrix& Y)
{
    assert(checkMatrixDevice({ &X1, &X2, &Y }));
    if (X1.isCuda())
    {
        cuda_max(X1.getDataTypeByInt(), X1.data(), X2.data(), Y.data(), X1.getDataSize(), 1, 1, 0);
    }
    else
    {
        //未完成
    }
}

void MatrixEx::matrix_maxb(Matrix& X1, Matrix& X2, const Matrix& Y, float a1, float a2, float r)
{
    assert(checkMatrixDevice({ &X1, &X2, &Y }));
    if (X1.isCuda())
    {
        cuda_maxb(X1.getDataTypeByInt(), X1.data(), X1.d().data(), X2.data(), X2.d().data(), Y.data(), Y.d().data(), X1.getDataSize(), a1, a2, 1);
    }
    else
    {
        //未完成
    }
}
}    // namespace cccc