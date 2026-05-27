#include "MatrixEx.h"
#include "Log.h"
#include "MatrixData.h"
#include "Random.h"
#include "Timer.h"
#include "VectorMath.h"
#include "gpu_lib.h"
#include <cassert>
#include <cstdlib>

#if !defined(M_PI)
#define M_PI 3.1415926535897932
#endif

namespace cccc
{

static void setDesc4D(cudnnTensorDescriptor_t desc, DataType data_type, int w, int h, int c, int n)
{
    cudnnSetTensor4dDescriptor(desc, CUDNN_TENSOR_NCHW, toCudnnDataType(data_type), n, c, h, w);
}

//R仅有一个通道，将A和B对应元素相乘后按channel求和，结果存入R
void MatrixEx::elementMulSum(const Matrix& A, const Matrix& B, Matrix& R, float a, float r)
{
    assert(checkMatrixDevice({ &A, &B, &R }));
    auto& workspaces = R.workspace_;
    if (workspaces.empty())
    {
        workspaces.resize(3);
        auto& R0 = workspaces[0];
        auto& W = workspaces[1];
        R0.resize(A.getDim());
        W.resize(1, 1, A.getChannel(), 1);
        W.fillData(1);
    }
    auto& R0 = workspaces[0];
    auto& W = workspaces[1];

    Matrix::elementMul(A, B, R0);
    //按channel求和
    MatrixEx::convolutionForward(R0, W, R, { 1, 1 }, { 0, 0 }, a, r);
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
#if ENABLE_CUDA
        if (X.getDataType() == DataType::FP8_E4M3 || X.getDataType() == DataType::FP8_E5M2)
        {
            bool is_cnn_bias = ((unsigned int)bias.channel_ == bias.getDataSize());
            unsigned int size_mc = is_cnn_bias ? (Y.row_ / Y.channel_) : 1u;
            unsigned int size_b = bias.getDataSize();
            if (bias.getDataType() == DataType::BFLOAT16)
            {
                cuda_addbias_fp8e4m3_bf16bias(X.data(), bias.data(), Y.data(), (unsigned int)X.getDataSize(), size_mc, size_b, a, b);
            }
            else
            {
                cuda_addbias_fp8e4m3(X.data(), bias.data(), Y.data(), (unsigned int)X.getDataSize(), size_mc, size_b, a, b);
            }
            Y.setQuantScale(1.0f);
            return;
        }
#endif
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
        auto gpu = Y.gpu();
        // CNN bias: bias.channel_ == bias.data_size_ (per-channel), use W*H stride
        // FC bias: bias.channel_ != bias.data_size_, use stride=1 so index = i % size_b
        bool is_cnn_bias = ((unsigned int)bias.channel_ == bias.getDataSize());
        unsigned int size_mc = is_cnn_bias ? (Y.row_ / Y.channel_) : 1u;
        if (X.getDataType() == DataType::BFLOAT16)
        {
            unsigned int size_b = bias.getDataSize();
            hip_addbias_bf16(Y.databf(), bias.databf(), X.getDataSize(), size_mc, size_b, a, b);
        }
        else
        {
            hip_addbias((float*)X.data(), (float*)bias.data(), (float*)Y.data(), X.getDataSize(), size_mc, bias.getDataSize(), a, b);
        }
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

void MatrixEx::addBiasBackward(const Matrix& X, Matrix& bias, const Matrix& Y, float a /*= 1*/, float b /*= 1*/)
{
    assert(checkMatrixDevice({ &X, &bias, &Y }));
    if (X.isCuda())
    {
        //用卷积反向代替一般反向，此处待验证
        auto gpu = Y.gpu();
        cudnnConvolutionBackwardBias(gpu->cudnn_handle_, &a, Y.cudnn_desc(), Y.d().data(), &b, bias.cudnn_desc(), bias.d().data());
    }
    else if (X.isHip())
    {
        auto gpu = Y.gpu();
        //b = 0;
        miopenConvolutionBackwardBias(gpu->miopen_handle_, &a, Y.miopen_desc(), Y.d().data(), &b, bias.miopen_desc(), bias.d().data());
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
                    float s = 0;
                    for (int i = 0; i < Y.width_ * Y.height_; i++)
                    {
                        s += Y.d().getData(i, 0, c, n);
                    }
                    bias.d().setData(0, 0, c, 0, bias.d().getData(0, 0, c, 0) + a * s);
                }
            }
        }
    }
}

void MatrixEx::concatByChannel(const std::vector<MatrixSP>& X_vector, Matrix& Y)
{
    //当前使用矩阵乘法实现concat，经测试比逐sample内存复制更快（GPU上矩阵乘可以充分利用并行度）
    //下面注释部分为内存复制实现，在某些场景下可能更高效，保留备用
#if 0
    //内存复制实现：逐sample将各输入的channel拼接到Y中
    //优点：无需额外的稀疏矩阵，内存占用少
    //缺点：GPU上逐次memcpy启动开销大，实测比矩阵乘慢
    for (int n = 0; n < Y.number_; n++)
    {
        int c_off = 0;
        for (int i = 0; i < X_vector.size(); i++)
        {
            auto& x = X_vector[i];
            copyDataPtr(*x, x->getDataPtr(0, 0, 0, n), Y, Y.getDataPtr(0, 0, c_off, n), x->row_);
            c_off += x->channel_;
        }
    }
#else
    auto& v = Y.workspace_;
    if (v.empty())
    {
        int index = 0;
        for (int i = 0; i < (int)X_vector.size(); i++)
        {
            auto& x = X_vector[i];
            Matrix temp({ Y.getRow(), x->getRow() }, x->getDataType(), UnitType::CPU);
            temp.fillData(0);
            for (int i_x = 0; i_x < x->getRow(); i_x++)
            {
                temp.setData(0, 0, i_x + index, i_x, 1);
            }
            temp.toGPU();
            v.push_back(std::move(temp));
            index += x->getRow();
        }
    }

    for (int i = 0; i < X_vector.size(); i++)
    {
        float r = 0;
        if (i > 0)
        {
            r = 1;
        }
        auto& x = X_vector[i];
        auto& tmp = v[i];
        if (x->number_ == Y.number_)
        {
            Matrix::mul(tmp, *x, Y, 1, r);
        }
    }
#endif
}

void MatrixEx::concatByChannelBackward(std::vector<MatrixSP>& X_vector, Matrix& Y)
{}

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

// Chunk forward: 沿 width(axis=0) 拷贝 X 的 [start_w, start_w+size_w) 列到 Y
// X: (W, H, C, N); Y: (size_w, H, C, N)
// 对每个 (h, c, n) 切片做一次连续内存拷贝
void MatrixEx::chunkForward(const Matrix& X, Matrix& Y, int start_w, int size_w)
{
    int H = X.getHeight();
    int C = X.getChannel();
    int N = X.getNumber();
    int W = X.getWidth();
    for (int n = 0; n < N; n++)
    {
        for (int c = 0; c < C; c++)
        {
            for (int h = 0; h < H; h++)
            {
                copyDataPtr(X, X.getDataPtr(start_w, h, c, n),
                    Y, Y.getDataPtr(0, h, c, n), size_w);
            }
        }
    }
}

// Chunk backward: 将 dY 累加到 dX 的 [start_w, start_w+size_w) 处
// CPU 路径: 直接循环累加; GPU 路径: 临时矩阵 + Matrix::add
void MatrixEx::chunkBackward(Matrix& X, const Matrix& Y, int start_w, int size_w)
{}

//初始化激活需要的缓冲区
//都是几个冷门的激活，因此不再建议使用
void MatrixEx::activeBufferInit(const Matrix& X, Matrix& Y, ActiveFunctionType af, std::vector<int>& int_vector, std::vector<float>& real_vector)
{
    auto gpu = X.gpu();
    auto& matrix_vector = Y.workspace_;
    switch (af)
    {
    case ACTIVE_FUNCTION_LOCAL_CONSTRAST_NORMALIZATION:
    case ACTIVE_FUNCTION_DIVISIVE_NORMALIZATION:
        if (X.isCuda())
        {
            matrix_vector = { Matrix(X.getDim()), Matrix(X.getDim()), Matrix({ X.getRow(), X.getNumber() * 2 }) };
        }
        break;
    //case ACTIVE_FUNCTION_BATCH_NORMALIZATION_DE:
    //    if (X.isCuda())
    //    {
    //        cudnnTensorDesc op_desc;
    //        cudnnDeriveBNTensorDescriptor(op_desc(), X.cudnn_desc(), cudnnBatchNormMode_t(int_vector[1]));
    //        int w, h, c, n, p1, p2, p3, p4;
    //        cudnnDataType_t dt;
    //        cudnnGetTensor4dDescriptor(op_desc(), &dt, &n, &c, &h, &w, &p1, &p2, &p3, &p4);
    //        std::vector<int> size = { w, h, c, n };
    //        matrix_vector = { Matrix(size).fillData(0.5), Matrix(size), Matrix(size), Matrix(size), Matrix(size), Matrix(size), Matrix(size), Matrix(size) };
    //    }
    //    break;
    case ACTIVE_FUNCTION_SPATIAL_TRANSFORMER:
        if (X.isCuda())
        {
            matrix_vector = { Matrix({ 3, 2, 1, X.number_ }), Matrix(X).getDim(), Matrix({ 3, 2, 1, X.number_ }), Matrix(X).getDim() };
        }
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
        }
        m.toGPU();
        matrix_vector = { m };
        break;
    }
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
    std::vector<int>& int_vector, std::vector<float>& real_vector, float a /*= 1*/, float r /*= 0*/)
{
    assert(checkMatrixDevice({ &X, &Y }));
    auto nan = CUDNN_NOT_PROPAGATE_NAN;
    auto gpu = X.gpu();
    auto& matrix_vector = Y.workspace_;
    for (auto& m : matrix_vector)
    {
        m.resizeNumber(X.getNumber());
    }
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
            if (X.getDataType() == DataType::BFLOAT16)
            {
                // miopenActivationForward does not support BF16; use custom kernel
                hip_sigmoid_bf16(Y.data(), X.data(), (unsigned int)X.getDataSize(), a, r);
            }
            else
            {
                auto act_desc = gpu->getDesc<miopenActivationDescriptor_t>();
                miopenSetActivationDescriptor(act_desc, miopenActivationLOGISTIC, 1, 1, 1);
                miopenActivationForward(gpu->miopen_handle_, act_desc, &a, X.miopen_desc(), X.data(), &r, Y.miopen_desc(), Y.data());
            }
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
            if (X.getDataType() == DataType::BFLOAT16)
            {
                hip_relu_bf16(Y.data(), X.data(), (unsigned int)X.getDataSize(), a, r);
            }
            else
            {
                auto act_desc = gpu->getDesc<miopenActivationDescriptor_t>();
                miopenSetActivationDescriptor(act_desc, miopenActivationRELU, 1, 1, 1);
                miopenActivationForward(gpu->miopen_handle_, act_desc, &a, X.miopen_desc(), X.data(), &r, Y.miopen_desc(), Y.data());
            }
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
            if (X.getDataType() == DataType::BFLOAT16)
            {
                hip_tanh_bf16(Y.data(), X.data(), (unsigned int)X.getDataSize(), a, r);
            }
            else
            {
                auto act_desc = gpu->getDesc<miopenActivationDescriptor_t>();
                miopenSetActivationDescriptor(act_desc, miopenActivationTANH, 1, 1, 1);
                miopenActivationForward(gpu->miopen_handle_, act_desc, &a, X.miopen_desc(), X.data(), &r, Y.miopen_desc(), Y.data());
            }
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
            if (X.getDataType() == DataType::BFLOAT16)
            {
                hip_softmax_bf16(Y.data(), X.data(), (unsigned int)X.row_, (unsigned int)X.number_, a, r);
            }
            else
            {
                miopenSoftmaxForward_V2(gpu->miopen_handle_, &a, X.miopen_desc(), X.data(), &r, Y.miopen_desc(), Y.data(), MIOPEN_SOFTMAX_ACCURATE, MIOPEN_SOFTMAX_MODE_INSTANCE);
            }
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
            if (X.getDataType() == DataType::BFLOAT16)
            {
                hip_softmax_bf16(Y.data(), X.data(), (unsigned int)X.row_, (unsigned int)X.number_, a, r);
            }
            else
            {
                miopenSoftmaxForward_V2(gpu->miopen_handle_, &a, X.miopen_desc(), X.data(), &r, Y.miopen_desc(), Y.data(), MIOPEN_SOFTMAX_FAST, MIOPEN_SOFTMAX_MODE_INSTANCE);
            }
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
            if (X.getDataType() == DataType::BFLOAT16)
            {
                hip_log_softmax_bf16(Y.data(), X.data(), (unsigned int)X.row_, (unsigned int)X.number_, a, r);
            }
            else
            {
                miopenSoftmaxForward_V2(gpu->miopen_handle_, &a, X.miopen_desc(), X.data(), &r, Y.miopen_desc(), Y.data(), MIOPEN_SOFTMAX_LOG, MIOPEN_SOFTMAX_MODE_INSTANCE);
            }
        }
        else
        {
            activeForward(X, Y, ACTIVE_FUNCTION_SOFTMAX, int_vector, real_vector);
            VectorMath::log_v(Y.data(), Y.data(), Y.data_size_, a, r);
        }
        break;
    case ACTIVE_FUNCTION_SOFTMAX_CHANNEL_CE:
    case ACTIVE_FUNCTION_SOFTMAX_CHANNEL:
        //沿 inner = X.width_ 维度做 softmax, outer = X.height_*X.channel_*X.number_
        //attention 中 scores 形状 (W=T_k, H=T_q, C=1, N=B*head): 每行 (固定 H,N) 在 W 维度上归一化
        if (X.isCuda())
        {
            int inner = X.width_;
            int outer = X.row_ / X.width_ * X.number_;
            cudnnTensorDesc tensor_desc;
            //把 inner 放到 C 维, MODE_CHANNEL 即对 C 做 softmax
            setDesc4D(tensor_desc(), X.getDataType(), 1, 1, inner, outer);
            cudnnSoftmaxForward(gpu->cudnn_handle_, CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_CHANNEL,
                &a, tensor_desc(), X.data(), &r, tensor_desc(), Y.data());
        }
        else if (X.isHip())
        {
            int inner = X.width_;
            int outer = X.row_ / X.width_ * X.number_;
            if (X.getDataType() == DataType::BFLOAT16)
            {
                hip_softmax_bf16(Y.data(), X.data(), (unsigned int)inner, (unsigned int)outer, a, r);
            }
            else
            {
                TensorDesc td(2);
                td.setDesc4D(X.getDataType(), 1, 1, inner, outer);
                miopenSoftmaxForward_V2(gpu->miopen_handle_, &a, td.miopenDesc(), X.data(), &r, td.miopenDesc(), Y.data(), MIOPEN_SOFTMAX_ACCURATE, MIOPEN_SOFTMAX_MODE_CHANNEL);
            }
        }
        else
        {
            //CPU 实现: 逐行 softmax
            int inner = X.width_;
            int outer = X.data_size_ / inner;
            MatrixEx::copyData(X, Y);
            for (int g = 0; g < outer; g++)
            {
                float* p = (float*)Y.getDataPtr(g * inner);
                float maxv = p[0];
                for (int i = 1; i < inner; i++)
                {
                    if (p[i] > maxv) { maxv = p[i]; }
                }
                float sum = 0;
                for (int i = 0; i < inner; i++)
                {
                    p[i] = std::exp(p[i] - maxv);
                    sum += p[i];
                }
                if (sum > 0)
                {
                    for (int i = 0; i < inner; i++) { p[i] /= sum; }
                }
            }
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
    case ACTIVE_FUNCTION_SQUARE:
        Matrix::elementMul(X, X, Y);
        break;
        //case ACTIVE_FUNCTION_SUMMAX:
        //    //这个和对应的反向计算是对的但是效率低，通常不要使用
        //    Matrix::copyData(X, A);
        //    for (int i = 0; i < X.getCol(); i++)
        //    {
        //        A.scaleCol(1.0 / X.sumAbsCol(i), i);
        //    }
        //    break;
    case ACTIVE_FUNCTION_SIN:
        MatrixEx::sin(X, Y);
        break;
    case ACTIVE_FUNCTION_COS:
        MatrixEx::cos(X, Y);
        break;
    case ACTIVE_FUNCTION_ZIGZAG:
        MatrixEx::zigzag(X, Y);
        //Matrix::copyData(A, R);
        break;
    case ACTIVE_FUNCTION_SIN_STEP:
    {
        //auto temp = A.clone(DeviceType::CPU);
        //int sum = 0;
        //for (int i = 0; i < A.getDataSize(); i++)
        //{
        //    if (abs(temp.getData(i)) > 1) { sum++; }
        //}
        //if (sum > 0)
        //{
        //    LOG("fajdogairfjgaiorg {} / {}\n", sum, temp.getDataSize());
        //}
    }
        MatrixEx::sin(X, Y, M_PI / 2 * 128);
        MatrixEx::step(Y, Y);
        break;
    case ACTIVE_FUNCTION_SELU:
        if (X.isCuda())
        {
            cudnnActivationDesc activation_desc;
            GpuControl::setActivationDesc(activation_desc(), CUDNN_ACTIVATION_ELU, 1.6732632423543772848170429916717);
            float a1 = a * 1.0507009873554804934193349852946;
            cudnnActivationForward(gpu->cudnn_handle_, activation_desc(),
                &a1, X.cudnn_desc(), X.data(), &r, Y.cudnn_desc(), Y.data());
        }
        break;
    case ACTIVE_FUNCTION_SIN_PLUS:
        MatrixEx::sin(X, Y);
        MatrixEx::add(X, Y, Y, 0.5, 0.5);
        break;
    case ACTIVE_FUNCTION_CLIPPED_RELU:
        if (real_vector.size() == 0)
        {
            real_vector.push_back(6);
        }
        if (X.isCuda())
        {
            cudnnActivationDesc activation_desc;
            GpuControl::setActivationDesc(activation_desc(), CUDNN_ACTIVATION_CLIPPED_RELU, real_vector[0]);
            cudnnActivationForward(gpu->cudnn_handle_, activation_desc(),
                &a, X.cudnn_desc(), X.data(), &r, Y.cudnn_desc(), Y.data());
        }
        else if (X.isHip())
        {
            auto act_desc = gpu->getDesc<miopenActivationDescriptor_t>();
            miopenSetActivationDescriptor(act_desc, miopenActivationCLIPPEDRELU, real_vector[0], 1, 1);
            miopenActivationForward(gpu->miopen_handle_, act_desc, &a, X.miopen_desc(), X.data(), &r, Y.miopen_desc(), Y.data());
        }
        else
        {
            VectorMath::clipped_relu_v(X.data(), Y.data(), real_vector[0], Y.data_size_, a, r);
        }
        break;
    case ACTIVE_FUNCTION_ELU:
        if (real_vector.size() == 0)
        {
            real_vector.push_back(1.6732632423543772848170429916717);
        }
        if (X.isCuda())
        {
            cudnnActivationDesc activation_desc;
            GpuControl::setActivationDesc(activation_desc(), CUDNN_ACTIVATION_ELU, real_vector[0]);
            cudnnActivationForward(gpu->cudnn_handle_, activation_desc(),
                &a, X.cudnn_desc(), X.data(), &r, Y.cudnn_desc(), Y.data());
        }
        else if (X.isHip())
        {
            auto act_desc = gpu->getDesc<miopenActivationDescriptor_t>();
            miopenSetActivationDescriptor(act_desc, miopenActivationELU, real_vector[0], 1, 1);
            miopenActivationForward(gpu->miopen_handle_, act_desc, &a, X.miopen_desc(), X.data(), &r, Y.miopen_desc(), Y.data());
        }
        break;
    case ACTIVE_FUNCTION_DROPOUT:
        int_vector[0] = rand();
        MatrixEx::dropoutForward(X, Y, real_vector[0], int_vector[0]);
        break;
        //以下无CPU支持
    case ACTIVE_FUNCTION_LOCAL_RESPONSE_NORMALIZATION:
    {
        if (X.isCuda())
        {
            cudnnLRNDesc lrn_desc;
            cudnnSetLRNDescriptor(lrn_desc(), int_vector[0], real_vector[0], real_vector[1], real_vector[2]);
            cudnnLRNCrossChannelForward(gpu->cudnn_handle_, lrn_desc(), CUDNN_LRN_CROSS_CHANNEL_DIM1,
                &a, X.cudnn_desc(), X.data(), &r, Y.cudnn_desc(), Y.data());
        }
        //HIP/CPU: LRN前向未实现
        break;
    }
    case ACTIVE_FUNCTION_LOCAL_CONSTRAST_NORMALIZATION:
        //MatrixExtend::lcnForward(X, A, matrix_vector[0], matrix_vector[2], int_vector[0], real_vector[0], real_vector[1], real_vector[2], true);
        break;
    case ACTIVE_FUNCTION_DIVISIVE_NORMALIZATION:
        //MatrixExtend::lcnForward(X, A, matrix_vector[0], matrix_vector[2], int_vector[0], real_vector[0], real_vector[1], real_vector[2], false);
        break;
    //case ACTIVE_FUNCTION_BATCH_NORMALIZATION_DE:
    //    MatrixEx::batchNormalizationForward(X, Y, ActivePhaseType(int_vector[0]), BatchNormalizationType(int_vector[1]), real_vector[1], real_vector[2],
    //        matrix_vector[0], matrix_vector[1], matrix_vector[2], matrix_vector[3], matrix_vector[4], matrix_vector[5]);
    //    break;
    case ACTIVE_FUNCTION_SPATIAL_TRANSFORMER:
        //MatrixExtend::spatialTfSamplerForward(X, A, matrix_vector[0], matrix_vector[1]);
        break;
    case ACTIVE_FUNCTION_RECURRENT:
        //unfinished
        break;
    case ACTIVE_FUNCTION_LEAKY_RELU:
        if (real_vector.size() == 0)
        {
            real_vector.push_back(0.02);
        }
        MatrixEx::leaky_relu(X, Y, real_vector[0]);
        break;
    case ACTIVE_FUNCTION_SILU:
#if ENABLE_CUDA
        // cuDNN ops treat FP8 as float (4× OOB); use native CUDA SiLU kernel
        if (X.isCuda() && (X.getDataType() == DataType::FP8_E4M3 || X.getDataType() == DataType::FP8_E5M2))
        {
            if (cuda_silu_fp8e4m3)
            {
                cuda_silu_fp8e4m3(X.data(), Y.data(), (unsigned int)X.getDataSize());
            }
            break;
        }
#endif
        if (matrix_vector.empty() || matrix_vector[0].getDim() != X.getDim())
        {
            matrix_vector.resize(1);
            matrix_vector[0] = Matrix(X.getDim(), DataType::CURRENT, UnitType::GPU);
        }
        MatrixEx::activeForward(X, matrix_vector[0], ACTIVE_FUNCTION_SIGMOID, int_vector, real_vector, a, r);
        Matrix::elementMul(X, matrix_vector[0], Y);
        break;
    case ACTIVE_FUNCTION_GELU:
        if (X.isCuda())
        {
            cuda_gelu(X.getDataTypeByInt(), X.data(), Y.data(), (unsigned int)X.getDataSize(), a, r);
        }
        else
        {
            // CPU precise GELU using erf: gelu(x) = 0.5*x*(1+erf(x/sqrt(2)))
            int n = (int)X.getDataSize();
            auto* xi = (const float*)X.data();
            auto* yi = (float*)Y.data();
            for (int i = 0; i < n; i++)
            {
                float v = xi[i];
                yi[i] = 0.5f * v * (1.0f + erff(v * 0.70710678f));
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
    std::vector<int>& int_vector, std::vector<float>& real_vector, float a /*= 1*/, float r /*= 0*/)
{
    assert(checkMatrixDevice({ &X, &Y }));
    auto nan = CUDNN_NOT_PROPAGATE_NAN;
    auto gpu = Y.gpu();
    auto& matrix_vector = Y.workspace_;
    switch (af)
    {
    case ACTIVE_FUNCTION_NONE:
    case ACTIVE_FUNCTION_SIGMOID_CE:
    case ACTIVE_FUNCTION_SOFTMAX_CE:
    case ACTIVE_FUNCTION_SOFTMAX_FAST_CE:
    case ACTIVE_FUNCTION_SOFTMAX_CHANNEL_CE:    //per-position CE: 梯度直接透传, 不展开 softmax Jacobian
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
    case ACTIVE_FUNCTION_SOFTMAX_CHANNEL:
        if (X.isCuda())
        {
            int inner = X.width_;
            int outer = X.row_ / X.width_ * X.number_;
            cudnnTensorDesc tensor_desc;
            setDesc4D(tensor_desc(), X.getDataType(), 1, 1, inner, outer);
            cudnnSoftmaxBackward(gpu->cudnn_handle_, CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_CHANNEL,
                &a, tensor_desc(), Y.data(), tensor_desc(), Y.d().data(), &r, tensor_desc(), X.d().data());
        }
        else if (X.isHip())
        {
            int inner = X.width_;
            int outer = X.row_ / X.width_ * X.number_;
            TensorDesc td(2);
            td.setDesc4D(X.getDataType(), 1, 1, inner, outer);
            miopenSoftmaxBackward_V2(gpu->miopen_handle_, &a, td.miopenDesc(), Y.data(), td.miopenDesc(), Y.d().data(), &r, td.miopenDesc(), X.d().data(), MIOPEN_SOFTMAX_ACCURATE, MIOPEN_SOFTMAX_MODE_CHANNEL);
        }
        else
        {
            //CPU 反向: 每行 dx_i = y_i * (dy_i - sum_j y_j * dy_j)
            int inner = X.width_;
            int outer = X.data_size_ / inner;
            for (int g = 0; g < outer; g++)
            {
                const float* py = (const float*)Y.getDataPtr(g * inner);
                const float* pdy = (const float*)Y.d().getDataPtr(g * inner);
                float* pdx = (float*)X.d().getDataPtr(g * inner);
                float dot = 0;
                for (int i = 0; i < inner; i++) { dot += py[i] * pdy[i]; }
                for (int i = 0; i < inner; i++) { pdx[i] = a * py[i] * (pdy[i] - dot) + r * pdx[i]; }
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
    case ACTIVE_FUNCTION_SQUARE:
        Matrix::elementMul(Y, X, X.d(), 2, 1);
        break;
    case ACTIVE_FUNCTION_SIN:
    case ACTIVE_FUNCTION_SIN_STEP:
        MatrixEx::cos(X, X.d());
        MatrixEx::elementMul(X.d(), Y.d(), X.d(), 1);    //不严格，需改为加法
        break;
    case ACTIVE_FUNCTION_COS:
        MatrixEx::sin(X, X.d());
        MatrixEx::elementMul(X.d(), Y.d(), X.d(), -1);
        break;
    case ACTIVE_FUNCTION_ZIGZAG:
        MatrixEx::zigzagb(X, Y);
        break;
    case ACTIVE_FUNCTION_SELU:
        if (X.isCuda())
        {
            cudnnActivationDesc activation_desc;
            GpuControl::setActivationDesc(activation_desc(), CUDNN_ACTIVATION_ELU, 1.6732632423543772848170429916717);
            float a1 = a * 1.0507009873554804934193349852946;
            cudnnActivationBackward(gpu->cudnn_handle_, activation_desc(), &a1, Y.cudnn_desc(), Y.data(),
                Y.cudnn_desc(), Y.d().data(), X.cudnn_desc(), X.data(), &r, X.cudnn_desc(), X.d().data());
        }
        else
        {
            //CPU/HIP fallback: dSELU/dx = lambda if y > 0, (y + lambda*alpha) if y <= 0
            const float lambda = 1.0507009873554804934193349852946f;
            const float alpha = 1.6732632423543772848170429916717f;
            auto* y_data = (const float*)Y.data();
            auto* yd_data = (const float*)Y.d().data();
            auto* xd_data = (float*)X.d().data();
            for (int i = 0; i < X.data_size_; i++)
            {
                float grad = (y_data[i] > 0) ? lambda : (y_data[i] + lambda * alpha);
                xd_data[i] = a * yd_data[i] * grad + r * xd_data[i];
            }
        }
        break;
    case ACTIVE_FUNCTION_SIN_PLUS:
        MatrixEx::cos(X, X.d());
        MatrixEx::addNumber(X.d(), X.d(), 0.5, 0.5);
        MatrixEx::elementMul(X.d(), Y.d(), X.d());
        break;
    case ACTIVE_FUNCTION_CLIPPED_RELU:
        if (X.isCuda())
        {
            cudnnActivationDesc activation_desc;
            GpuControl::setActivationDesc(activation_desc(), CUDNN_ACTIVATION_CLIPPED_RELU, real_vector[0]);
            cudnnActivationBackward(gpu->cudnn_handle_, activation_desc(), &a, Y.cudnn_desc(), Y.data(),
                Y.cudnn_desc(), Y.d().data(), X.cudnn_desc(), X.data(), &r, X.cudnn_desc(), X.d().data());
        }
        else if (X.isHip())
        {
            auto act_desc = gpu->getDesc<miopenActivationDescriptor_t>();
            miopenSetActivationDescriptor(act_desc, miopenActivationCLIPPEDRELU, real_vector[0], 1, 1);
            miopenActivationBackward(gpu->miopen_handle_, act_desc, &a, Y.miopen_desc(), Y.data(), Y.miopen_desc(), Y.d().data(), X.miopen_desc(), X.data(), &r, X.miopen_desc(), X.d().data());
        }
        else
        {
            VectorMath::clipped_relu_vb(Y.data(), Y.d().data(), X.data(), X.d().data(), real_vector[0], X.data_size_, a, r);
        }
        break;
    case ACTIVE_FUNCTION_ELU:
        if (X.isCuda())
        {
            cudnnActivationDesc activation_desc;
            GpuControl::setActivationDesc(activation_desc(), CUDNN_ACTIVATION_ELU, real_vector[0]);
            cudnnActivationBackward(gpu->cudnn_handle_, activation_desc(), &a, Y.cudnn_desc(), Y.data(),
                Y.cudnn_desc(), Y.d().data(), X.cudnn_desc(), X.data(), &r, X.cudnn_desc(), X.d().data());
        }
        else if (X.isHip())
        {
            auto act_desc = gpu->getDesc<miopenActivationDescriptor_t>();
            miopenSetActivationDescriptor(act_desc, miopenActivationELU, real_vector[0], 1, 1);
            miopenActivationBackward(gpu->miopen_handle_, act_desc, &a, Y.miopen_desc(), Y.data(), Y.miopen_desc(), Y.d().data(), X.miopen_desc(), X.data(), &r, X.miopen_desc(), X.d().data());
        }
        else
        {
            //CPU/HIP fallback: dELU/dx = 1 if y > 0, (y + alpha) if y <= 0
            float alpha = real_vector[0];
            auto* y_data = (const float*)Y.data();
            auto* yd_data = (const float*)Y.d().data();
            auto* xd_data = (float*)X.d().data();
            for (int i = 0; i < X.data_size_; i++)
            {
                float grad = (y_data[i] > 0) ? 1.0f : (y_data[i] + alpha);
                xd_data[i] = a * yd_data[i] * grad + r * xd_data[i];
            }
        }
        break;
    case ACTIVE_FUNCTION_DROPOUT:
        MatrixEx::dropoutBackward(X, Y, real_vector[0], int_vector[0]);
        break;
    case ACTIVE_FUNCTION_LOCAL_RESPONSE_NORMALIZATION:
    {
        if (X.isCuda())
        {
            cudnnLRNDesc lrn_desc;
            cudnnSetLRNDescriptor(lrn_desc(), int_vector[0], real_vector[0], real_vector[1], real_vector[2]);
            cudnnLRNCrossChannelBackward(gpu->cudnn_handle_, lrn_desc(), CUDNN_LRN_CROSS_CHANNEL_DIM1,
                &a, Y.cudnn_desc(), Y.data(), Y.cudnn_desc(), Y.d().data(), X.cudnn_desc(), X.data(), &r, X.cudnn_desc(), X.d().data());
        }
        //HIP/CPU: LRN反向未实现
        break;
    }
    case ACTIVE_FUNCTION_LOCAL_CONSTRAST_NORMALIZATION:
        //MatrixExtend::lcnBackward(A, dA, X, dX, matrix_vector[0], matrix_vector[1], matrix_vector[2], int_vector[0], real_vector[0], real_vector[1], real_vector[2], true, 0);
        break;
    case ACTIVE_FUNCTION_DIVISIVE_NORMALIZATION:
        //MatrixExtend::lcnBackward(A, dA, X, dX, matrix_vector[0], matrix_vector[1], matrix_vector[2], int_vector[0], real_vector[0], real_vector[1], real_vector[2], false, real_vector[3]);
        break;
    //case ACTIVE_FUNCTION_BATCH_NORMALIZATION_DE:
    //    MatrixEx::batchNormalizationBackward(X, Y, ActivePhaseType(int_vector[0]), BatchNormalizationType(int_vector[1]), real_vector[2], real_vector[0],
    //        matrix_vector[0], matrix_vector[1], matrix_vector[4], matrix_vector[5], matrix_vector[6], matrix_vector[7]);
    //    break;
    case ACTIVE_FUNCTION_SPATIAL_TRANSFORMER:
        //MatrixExtend::spatialTfSamplerBackward(A, dA, X, dX, real_vector[0], matrix_vector[0], matrix_vector[1], matrix_vector[2], matrix_vector[3]);
        break;
    case ACTIVE_FUNCTION_RECURRENT:
        break;
    case ACTIVE_FUNCTION_LEAKY_RELU:
        MatrixEx::leaky_relub(X, Y, real_vector[0]);
        break;
    case ACTIVE_FUNCTION_SILU:
        Matrix::elementMul(matrix_vector[0], Y, matrix_vector[0], -1, 1);
        Matrix::add(Y, matrix_vector[0], matrix_vector[0]);
        Matrix::elementMul(matrix_vector[0], Y.d(), X.d());
        break;
    case ACTIVE_FUNCTION_GELU:
        if (X.isCuda())
        {
            cuda_gelub(X.getDataTypeByInt(), X.data(), X.d().data(), Y.data(), Y.d().data(),
                (unsigned int)X.getDataSize(), a, X.keepWeight());
        }
        else
        {
            // CPU: dX = a * gelu'(x) * dY + keepWeight * dX
            int n = (int)X.getDataSize();
            const auto* xi = (const float*)X.data();
            const auto* dyi = (const float*)Y.d().data();
            auto* dxi = (float*)X.d().data();
            float kw = X.keepWeight();
            for (int i = 0; i < n; i++)
            {
                float v = xi[i];
                float z = 0.7978845608f * (v + 0.044715f * v * v * v);
                float t = tanhf(z);
                float sech2 = 1.0f - t * t;
                float grad = 0.5f * (1.0f + t) + 0.5f * v * sech2 * 0.7978845608f * (1.0f + 3.0f * 0.044715f * v * v);
                dxi[i] = a * grad * dyi[i] + kw * dxi[i];
            }
        }
        break;
    case ACTIVE_FUNCTION_SIGMOID3:
        if (X.isCuda())
        {
            cudnnActivationDesc activation_desc;
            auto a1 = a * 0.1;
            GpuControl::setActivationDesc(activation_desc(), CUDNN_ACTIVATION_SIGMOID, 1);
            cudnnActivationBackward(gpu->cudnn_handle_, activation_desc(), &a, Y.cudnn_desc(), Y.data(),
                Y.cudnn_desc(), Y.d().data(), X.cudnn_desc(), X.data(), &r, X.cudnn_desc(), X.d().data());
            Matrix::add(Y.d(), X.d(), X.d(), a1, r, 0);
        }
        else
        {
            VectorMath::sigmoid_vb(Y.data(), Y.d().data(), X.data(), X.d().data(), X.data_size_, a, r);
        }
        break;
    case ACTIVE_FUNCTION_SOFTMAX3:
        if (X.isCuda())
        {
            cudnnTensorDesc tensor_desc;
            auto a1 = a * 0.1;
            auto softmax_flag = CUDNN_SOFTMAX_ACCURATE;
            if (af == ACTIVE_FUNCTION_SOFTMAX_FAST)
            {
                softmax_flag = CUDNN_SOFTMAX_FAST;
            }
            setDesc4D(tensor_desc(), X.getDataType(), 1, 1, X.row_, X.number_);
            auto s = cudnnSoftmaxBackward(gpu->cudnn_handle_, softmax_flag, CUDNN_SOFTMAX_MODE_INSTANCE,
                &a, tensor_desc(), Y.data(), tensor_desc(), Y.d().data(), &r, tensor_desc(), X.d().data());
            Matrix::add(Y.d(), X.d(), X.d(), a1, r, 0);
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
    activeForward(X, Y, af, int_vector, real_vector, a, r);
}

void MatrixEx::activeBackwardSimple(Matrix& X, const Matrix& Y, ActiveFunctionType af, float a /*= 1*/, float r /*= 0*/)
{
    std::vector<int> int_vector;
    std::vector<float> real_vector;
    std::vector<Matrix> matrix_vector;
    activeBackward(X, Y, af, int_vector, real_vector, a, r);
}

//池化
//gpu部分，平均模式下对padding的支持目前还有问题
void MatrixEx::poolingForward(const Matrix& X, Matrix& Y, PoolingType pooling_type, PoolingReverseType reverse_type, const std::vector<int>& window, const std::vector<int>& stride, const std::vector<int>& padding, float a, float r)
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
            auto s = cudnnSetPooling2dDescriptor(pooling_desc(), pooling_type_c, CUDNN_NOT_PROPAGATE_NAN,
                window[1], window[0], padding[1], padding[0], stride[1], stride[0]);
            if (s)
            {
                LOG_ERR("POOL forward error {}, {}\n", cudnnGetErrorString(s), X.gpu()->lastCudnnErrorString());
            }
        }
        else
        {
            auto wr = window;
            auto pr = padding;
            auto sr = stride;
            std::reverse(wr.begin(), wr.end());
            std::reverse(pr.begin(), pr.end());
            std::reverse(sr.begin(), sr.end());
            auto s = cudnnSetPoolingNdDescriptor(pooling_desc(), pooling_type_c, CUDNN_NOT_PROPAGATE_NAN,
                window.size(), wr.data(), pr.data(), sr.data());
            if (s)
            {
                LOG_ERR("POOL forward error {}, {}\n", cudnnGetErrorString(s), X.gpu()->lastCudnnErrorString());
            }
        }
        cudnnStatus_t s;
        if (reverse_type == POOLING_NOT_REVERSE)
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
        if (Y.workspace_.empty())
        {
            Y.workspace_.resize(1);
        }
        if (Y.workspace_[0].getDataSizeInByte() == 0)
        {
            size_t ws;
            miopenPoolingGetWorkSpaceSize(Y.miopen_desc(), &ws);
            Y.workspace_[0].resize(1, 1, 1, ws);
        }
        miopenStatus_t s;
        bool back = gpu->active_phase_ == ACTIVE_PHASE_TRAIN;
        if (reverse_type == POOLING_NOT_REVERSE)
        {
            s = miopenPoolingForward(gpu->miopen_handle_, op_desc(), &a, X.miopen_desc(), X.data(), &r, Y.miopen_desc(), Y.data(), back, Y.workspace_[0].data(), Y.workspace_[0].getDataSizeInByte());
        }
        else
        {
            s = miopenPoolingBackward(gpu->miopen_handle_, op_desc(), &a, X.miopen_desc(), X.data(), X.miopen_desc(), X.data(), Y.miopen_desc(), Y.data(), &r, Y.miopen_desc(), Y.data(), Y.workspace_[0].data());
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

void MatrixEx::poolingBackward(Matrix& X, const Matrix& Y, PoolingType pooling_type, PoolingReverseType reverse_type, const std::vector<int>& window, const std::vector<int>& stride, const std::vector<int>& padding, float a, float r)
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
        if (reverse_type == POOLING_NOT_REVERSE)
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

        if (Y.workspace_.empty())
        {
            Y.workspace_.resize(1);
        }
        auto& workspace = Y.workspace_[0];
        if (reverse_type == POOLING_NOT_REVERSE)
        {
            s = miopenPoolingBackward(gpu->miopen_handle_, op_desc, &a, Y.miopen_desc(), Y.data(), Y.miopen_desc(), Y.d().data(), X.miopen_desc(), X.data(), &r, X.miopen_desc(), X.d().data(), workspace.data());
        }
        else
        {
            s = miopenPoolingForward(gpu->miopen_handle_, op_desc, &a, Y.miopen_desc(), Y.d().data(), &r, X.miopen_desc(), X.d().data(), back, workspace.data(), workspace.getDataSizeInByte());
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
                                    X.d().setData(wdX, hdX, p, 0, X.d().getData(wdX, hdX, p, 0) + a * v);
                                    // +r * X.d().getData(wdX, hdX, p, 0);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

void MatrixEx::poolingChannelForward(Matrix& X, Matrix& Y, PoolingType pooling_type, PoolingReverseType reverse_type, float a, float r)
{
    auto dimx0 = X.getDim();
    auto dimy0 = Y.getDim();
    auto channel = X.getChannel();
    auto dimx = { X.getWidth() * X.getHeight(), X.getChannel(), 1, X.getNumber() };
    auto dimy = { Y.getWidth() * Y.getHeight(), 1, 1, Y.getNumber() };

    X.resize(dimx);
    Y.resize(dimy);
    poolingForward(X, Y, pooling_type, reverse_type, { 1, channel }, { 1, channel }, { 0, 0 }, a, r);
    X.resize(dimx0);
    Y.resize(dimy0);
}

void MatrixEx::poolingChannelBackward(Matrix& X, Matrix& Y, PoolingType pooling_type, PoolingReverseType reverse_type, float a, float r)
{
    auto channel = X.getChannel();
    auto dimx0 = X.getDim();
    auto dimy0 = Y.getDim();
    auto dimx = { X.getWidth() * X.getHeight(), X.getChannel(), 1, X.getNumber() };
    auto dimy = { Y.getWidth() * Y.getHeight(), 1, 1, Y.getNumber() };
    X.resize(dimx);
    X.d().resize(dimx);
    Y.resize(dimy);
    Y.d().resize(dimy);
    poolingBackward(X, Y, pooling_type, reverse_type, { 1, channel }, { 1, channel }, { 0, 0 }, a, r);
    X.resize(dimx0);
    X.d().resize(dimx0);
    Y.resize(dimy0);
    Y.d().resize(dimy0);
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
void MatrixEx::convolutionForward(const Matrix& X, const Matrix& W, Matrix& Y, const std::vector<int>& stride, const std::vector<int>& padding, float a, float r)
{
    assert(checkMatrixDevice({ &X, &W, &Y }));
    assert(stride.size() >= 2 && padding.size() >= 2);
    assert(X.getChannel() == W.getChannel() && Y.getChannel() == W.getNumber());

    //若工作空间为空则创建一个（HIP/CPU 路径仍需使用）
    if (Y.workspace_.empty())
    {
        Y.workspace_.resize(1);
    }

    auto& method = Y.user_data<ConvMethod>();
    auto gpu = X.gpu();

    if (!gpu->user_data_.has_value())
    {
        gpu->user_data_ = Matrix();
    }
    auto& workspace = gpu->getUserData<Matrix>();

    if (Y.isCuda())
    {
        cudnnConvolutionDesc conv_desc;
        cudnnFilterDesc filter_desc;
        cudnnStatus_t scd, sfd;
        if (stride.size() == 2)
        {
            scd = cudnnSetConvolution2dDescriptor(conv_desc(), padding[1], padding[0], stride[1], stride[0], 1, 1, CUDNN_CROSS_CORRELATION, toConvComputeType(Y.getDataType()));
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
            scd = cudnnSetConvolutionNdDescriptor(conv_desc(), padding.size(), pr.data(), sr.data(), dilation.data(), CUDNN_CROSS_CORRELATION, toConvComputeType(Y.getDataType()));
            auto w_dim = W.getDim();
            std::reverse(w_dim.begin(), w_dim.end());
            sfd = cudnnSetFilterNdDescriptor(filter_desc(), toCudnnDataType(Y.getDataType()), CUDNN_TENSOR_NCHW, W.getDimSize(), w_dim.data());
        }
        //寻找最快的算法（启发式，不做 GPU 基准测试，避免 cuDNN 内部 benchmark pool 无限积累显存）
        if (method.algo < 0)
        {
            int n;
            cudnnConvolutionFwdAlgoPerf_t cfap[conv_method_count];
            cudnnGetConvolutionForwardAlgorithm_v7(gpu->cudnn_handle_, X.cudnn_desc(),
                filter_desc(), conv_desc(), Y.cudnn_desc(), conv_method_count, &n, cfap);
            if (n == 0 || cfap[0].status != CUDNN_STATUS_SUCCESS)
            {
                LOG_ERR("CONV forward: no valid algorithm found\n");
                return;
            }
            method.algo = int(cfap[0].algo);
            method.math_type = int(cfap[0].mathType);
            size_t ws_size = cfap[0].memory;
            if (ws_size > workspace.getDataSizeInByte())
            {
                workspace.resize(1, 1, 1, ws_size / workspace.getDataTypeSize() + 2);
            }
#ifdef _DEBUG
            LOG("conv forward choose {}({}), workspace {}\n", method.algo, method.math_type,
                (n > 0) ? cfap[0].memory : 0);
#endif
        }
        else if (method.group_number != X.number_)
        {
            size_t ws_size;
            auto s = cudnnGetConvolutionForwardWorkspaceSize(gpu->cudnn_handle_, X.cudnn_desc(),
                filter_desc(), conv_desc(), Y.cudnn_desc(), cudnnConvolutionFwdAlgo_t(method.algo), &ws_size);
            if (s == CUDNN_STATUS_SUCCESS && ws_size > workspace.getDataSizeInByte())
            {
                workspace.resize(1, 1, 1, ws_size / workspace.getDataTypeSize() + 2);
            }
        }
        method.group_number = X.number_;
        cudnnSetConvolutionMathType(conv_desc(), cudnnMathType_t(method.math_type));
        auto cfa = cudnnConvolutionFwdAlgo_t(method.algo);
        auto scf = cudnnConvolutionForward(gpu->cudnn_handle_, &a, X.cudnn_desc(), X.data(),
            filter_desc(), W.data(), conv_desc(), cfa, workspace.data(), workspace.getDataSizeInByte(),
            &r, Y.cudnn_desc(), Y.data());
        if (scf)
        {
            LOG_ERR("CONV forward error: {}, algo {}\n", cudnnGetErrorString(scf), int(cfa));
            LOG_ERR("  X: {}x{}x{}x{}  W: {}x{}x{}x{}  Y: {}x{}x{}x{}\n",
                X.width_, X.height_, X.channel_, X.number_,
                W.width_, W.height_, W.channel_, W.number_,
                Y.width_, Y.height_, Y.channel_, Y.number_);
        }
    }
    else if (Y.isHip())
    {
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
        if (method.algo < 0)
        {
            int n;
            size_t ws_size;
            miopenConvAlgoPerf_t cfap[conv_method_count];
            miopenConvolutionForwardGetWorkSpaceSize(gpu->miopen_handle_, W.miopen_desc(), X.miopen_desc(), op_desc, Y.miopen_desc(), &ws_size);
            if (ws_size > workspace.getDataSizeInByte())
            {
                workspace.resize(1, 1, ws_size / workspace.getDataTypeSize() + 2, 1);
            }
            miopenFindConvolutionForwardAlgorithm(gpu->miopen_handle_, X.miopen_desc(), X.data(), W.miopen_desc(), W.data(),
                op_desc, Y.miopen_desc(), Y.data(), conv_method_count, &n, cfap, workspace.data(), workspace.getDataSizeInByte(), true);
            method.algo = (n > 0) ? cfap[0].fwd_algo : 0;
        }
        else if (method.group_number != X.number_)
        {
            size_t ws_size;
            auto s = miopenConvolutionForwardGetWorkSpaceSize(gpu->miopen_handle_, W.miopen_desc(), X.miopen_desc(), op_desc, Y.miopen_desc(), &ws_size);
            if (s == 0 & ws_size > workspace.getDataSizeInByte())
            {
                workspace.resize(1, 1, ws_size / workspace.getDataTypeSize() + 2, 1);
            }
        }
        method.group_number = X.number_;
        auto scf = miopenConvolutionForward(gpu->miopen_handle_, &a, X.miopen_desc(), X.data(), W.miopen_desc(), W.data(),
            op_desc, miopenConvFwdAlgorithm_t(method.algo), &r, Y.miopen_desc(), Y.data(), workspace.data(), workspace.getDataSizeInByte());
        if (scf)
        {
            LOG_ERR("CONV forward error: status {}, methods {}\n", miopenGetErrorString(scf), method.algo);
        }
    }
    else
    {
        Y.fillData(0);
        //辅助矩阵的尺寸
        int row = Y.width_ * Y.height_;
        int col = X.channel_ * W.width_ * W.height_;
        Matrix X_ex({ row, col }, X.getDataType(), UnitType::CPU);
        if (method.algo < 0)
        {
            method.algo = 0;
            workspace.resize(1, 1, row * col, 1);
            workspace.fillData(-1);
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
                        workspace.setData(pX_ex, 0, pX);    //记录展开的位置
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
                int p = workspace.getData(j, 0);
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
//此函数暂时不起作用，请使用convolutionBackwardDX和convolutionBackwardDW
void MatrixEx::convolutionBackward(Matrix& X, Matrix& W, const Matrix& Y, const std::vector<int>& stride, const std::vector<int>& padding, float ax, float rx, float aw, float rw)
{
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

void MatrixEx::convolutionBackwardDX(Matrix& X, const Matrix& W, const Matrix& Y, const std::vector<int>& stride, const std::vector<int>& padding, float a, float r)
{
    assert(checkMatrixDevice({ &X, &W, &Y }));
    assert(stride.size() >= 2 && padding.size() >= 2);
    assert(X.getChannel() == W.getChannel() && Y.getChannel() == W.getNumber());

    if (X.workspace_.empty())
    {
        X.workspace_.resize(1);
    }
    auto& method_dx = X.user_data<ConvMethod>();
    //auto& workspace_dx = X.workspace_[0];
    auto gpu = Y.gpu();
    auto& workspace_dx = std::any_cast<Matrix&>(gpu->user_data_);
    //这里不用dX判断是因为dX可能是空
    if (Y.isCuda())
    {
        cudnnConvolutionDesc conv_desc;
        cudnnFilterDesc filter_desc;
        if (stride.size() == 2)
        {
            cudnnSetConvolution2dDescriptor(conv_desc(), padding[1], padding[0], stride[1], stride[0], 1, 1, CUDNN_CROSS_CORRELATION, toConvComputeType(Y.getDataType()));
            cudnnSetFilter4dDescriptor(filter_desc(), toCudnnDataType(Y.getDataType()), CUDNN_TENSOR_NCHW, W.number_, W.channel_, W.height_, W.width_);
        }
        else
        {
            std::vector<int> dilation(padding.size(), 1);
            auto pr = padding;
            auto sr = stride;
            std::reverse(pr.begin(), pr.end());
            std::reverse(sr.begin(), sr.end());
            cudnnSetConvolutionNdDescriptor(conv_desc(), padding.size(), pr.data(), sr.data(), dilation.data(), CUDNN_CROSS_CORRELATION, toConvComputeType(Y.getDataType()));
            auto w_dim = W.getDim();
            std::reverse(w_dim.begin(), w_dim.end());
            cudnnSetFilterNdDescriptor(filter_desc(), toCudnnDataType(Y.getDataType()), CUDNN_TENSOR_NCHW, W.getDimSize(), w_dim.data());
        }
        cudnnSetConvolutionMathType(conv_desc(), CUDNN_TENSOR_OP_MATH);

        //寻找最快的算法（启发式，不做 GPU 基准测试，避免 cuDNN 内部 benchmark pool 无限积累显存）
        if (method_dx.algo < 0)
        {
            int n;
            cudnnConvolutionBwdDataAlgoPerf_t cbdap[conv_method_count];
            cudnnGetConvolutionBackwardDataAlgorithm_v7(gpu->cudnn_handle_, filter_desc(),
                Y.cudnn_desc(), conv_desc(), X.cudnn_desc(), conv_method_count, &n, cbdap);
            method_dx.algo = cbdap[0].algo;
            method_dx.math_type = cbdap[0].mathType;
            size_t ws_size = cbdap[0].memory;
            if (ws_size > workspace_dx.getDataSizeInByte())
            {
                workspace_dx.resize(1, 1, 1, ws_size / workspace_dx.getDataTypeSize() + 2);
            }
#ifdef _DEBUG
            LOG("conv backward X choose {}({}), workspace {}\n", method_dx.algo, method_dx.math_type, cbdap[0].memory);
#endif
        }
        else if (method_dx.group_number != X.number_)
        {
            size_t ws_size;
            auto s = cudnnGetConvolutionBackwardDataWorkspaceSize(gpu->cudnn_handle_, filter_desc(),
                Y.cudnn_desc(), conv_desc(), X.cudnn_desc(), cudnnConvolutionBwdDataAlgo_t(method_dx.algo), &ws_size);
            if (s == CUDNN_STATUS_SUCCESS && ws_size > workspace_dx.getDataSizeInByte())
            {
                workspace_dx.resize(1, 1, 1, ws_size / workspace_dx.getDataTypeSize() + 2);
#ifdef _DEBUG
                LOG("resize conv workspace for back X{}\n", ws_size);
#endif
            }
        }
        method_dx.group_number = X.number_;
        auto cbda = cudnnConvolutionBwdDataAlgo_t(method_dx.algo);
        auto tensor = cudnnSetConvolutionMathType(conv_desc(), cudnnMathType_t(method_dx.math_type));
        auto scbx = cudnnConvolutionBackwardData(gpu->cudnn_handle_, &a, filter_desc(), W.data(), Y.cudnn_desc(), Y.d().data(),
            conv_desc(), cbda, workspace_dx.data(), workspace_dx.getDataSizeInByte(), &r, X.cudnn_desc(), X.d().data());
        if (scbx)
        {
            LOG_ERR("CONV backward X error: status {}, {}\n", cudnnGetErrorString(scbx), X.gpu()->lastCudnnErrorString());
        }
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

        if (method_dx.algo < 0)
        {
            int n;
            size_t ws_size;
            miopenConvAlgoPerf_t cfap[conv_method_count];
            miopenConvolutionBackwardDataGetWorkSpaceSize(gpu->miopen_handle_, Y.miopen_desc(), W.miopen_desc(), op_desc, X.miopen_desc(), &ws_size);
            if (ws_size > workspace_dx.getDataSizeInByte())
            {
                workspace_dx.resize(1, 1, ws_size / workspace_dx.getDataTypeSize() + 2, 1);
            }
            miopenFindConvolutionBackwardDataAlgorithm(gpu->miopen_handle_, Y.miopen_desc(), Y.d().data(), W.miopen_desc(), W.data(),
                op_desc, X.miopen_desc(), X.d().data(), conv_method_count, &n, cfap, workspace_dx.data(), workspace_dx.getDataSizeInByte(), true);
            method_dx.algo = cfap[0].bwd_data_algo;
        }
        else if (method_dx.group_number != X.number_)
        {
            size_t ws_size;
            auto s = miopenConvolutionBackwardDataGetWorkSpaceSize(gpu->miopen_handle_, Y.miopen_desc(), W.miopen_desc(), op_desc, X.miopen_desc(), &ws_size);
            if (s == 0 && ws_size > workspace_dx.getDataSizeInByte())
            {
                workspace_dx.resize(1, 1, ws_size / workspace_dx.getDataTypeSize() + 2, 1);
            }
        }
        method_dx.group_number = X.number_;
        auto scf = miopenConvolutionBackwardData(gpu->miopen_handle_, &a, Y.miopen_desc(), Y.d().data(), W.miopen_desc(), W.data(),
            op_desc, miopenConvBwdDataAlgorithm_t(method_dx.algo), &r, X.miopen_desc(), X.d().data(), workspace_dx.data(), workspace_dx.getDataSizeInByte());
    }
    else
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
        if (method_dx.algo < 0)
        {
            method_dx.algo = 0;
            workspace_dx.resize(1, 1, row * col, 1);
            workspace_dx.fillData(-1);
            for (int cY = 0; cY < Y.channel_; cY++)
            {
                CONV_OPERATION1(X, W, Y, padding[0], padding[1], stride[0], stride[1],
                    {
                        int pX = X.whcn2i(wX, hX, 0, 0);
                        int pW = W2.whcn2i(wW, hW, cY, 0);    //这里用W或者W2没有区别
                        int pY = Y.whcn2i(wY, hY, cY, 0);     //拍扁
                        int p_ex = dY_ex.mn2i(pX, pW);
                        workspace_dx.setData(p_ex, 0, pY);
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
                if (workspace_dx.getData(j, 0) >= 0)
                {
                    dY_ex.setData(j, dY_sub.getData(workspace_dx.getData(j, 0)));
                }
            }
            MatrixEx::mul(dY_ex, W2, dX_sub, a, r);
        }
    }
}

void MatrixEx::convolutionBackwardDW(const Matrix& X, Matrix& W, const Matrix& Y, const std::vector<int>& stride, const std::vector<int>& padding, float a, float r)
{
    assert(checkMatrixDevice({ &X, &W, &Y }));
    assert(stride.size() >= 2 && padding.size() >= 2);
    assert(X.getChannel() == W.getChannel() && Y.getChannel() == W.getNumber());

    //if (W.workspace_.empty())
    //{
    //    W.workspace_.resize(1);
    //}
    auto& method_dw = W.user_data<ConvMethod>();
    //auto& workspace_dw = W.workspace_[0];
    auto gpu = Y.gpu();
    auto& workspace_dw = std::any_cast<Matrix&>(gpu->user_data_);
    if (Y.isCuda())
    {
        cudnnConvolutionDesc conv_desc;
        cudnnFilterDesc filter_desc;
        if (stride.size() == 2)
        {
            cudnnSetConvolution2dDescriptor(conv_desc(), padding[1], padding[0], stride[1], stride[0], 1, 1, CUDNN_CROSS_CORRELATION, toConvComputeType(Y.getDataType()));
            cudnnSetFilter4dDescriptor(filter_desc(), toCudnnDataType(Y.getDataType()), CUDNN_TENSOR_NCHW, W.number_, W.channel_, W.height_, W.width_);
        }
        else
        {
            std::vector<int> dilation(padding.size(), 1);
            auto pr = padding;
            auto sr = stride;
            std::reverse(pr.begin(), pr.end());
            std::reverse(sr.begin(), sr.end());
            cudnnSetConvolutionNdDescriptor(conv_desc(), padding.size(), pr.data(), sr.data(), dilation.data(), CUDNN_CROSS_CORRELATION, toConvComputeType(Y.getDataType()));
            auto w_dim = W.getDim();
            std::reverse(w_dim.begin(), w_dim.end());
            cudnnSetFilterNdDescriptor(filter_desc(), toCudnnDataType(Y.getDataType()), CUDNN_TENSOR_NCHW, W.getDimSize(), w_dim.data());
        }
        cudnnSetConvolutionMathType(conv_desc(), CUDNN_TENSOR_OP_MATH);

        //寻找最快的算法（启发式，不做 GPU 基准测试，避免 cuDNN 内部 benchmark pool 无限积累显存）
        if (method_dw.algo < 0)
        {
            int n;
            cudnnConvolutionBwdFilterAlgoPerf_t cbfap[conv_method_count];
            cudnnGetConvolutionBackwardFilterAlgorithm_v7(gpu->cudnn_handle_, X.cudnn_desc(), Y.cudnn_desc(),
                conv_desc(), filter_desc(), conv_method_count, &n, cbfap);
            method_dw.algo = cbfap[0].algo;
            method_dw.math_type = cbfap[0].mathType;
            size_t memory = cbfap[0].memory;
            if (memory > workspace_dw.getDataSizeInByte())
            {
                workspace_dw.resize(1, 1, 1, memory / workspace_dw.getDataTypeSize() + 2);
            }
#ifdef _DEBUG
            LOG("conv backward W choose {}({}), workspace {}\n", method_dw.algo, method_dw.math_type, cbfap[0].memory);
#endif
        }
        else if (method_dw.group_number != X.number_)
        {
            size_t memory;
            auto s = cudnnGetConvolutionBackwardFilterWorkspaceSize(gpu->cudnn_handle_, X.cudnn_desc(), Y.cudnn_desc(),
                conv_desc(), filter_desc(), cudnnConvolutionBwdFilterAlgo_t(method_dw.algo), &memory);
            if (s == CUDNN_STATUS_SUCCESS && memory > workspace_dw.getDataSizeInByte())
            {
                workspace_dw.resize(1, 1, 1, memory / workspace_dw.getDataTypeSize() + 2);
#ifdef _DEBUG
                LOG("resize conv workspace for back W {}\n", memory);
#endif
            }
        }
        method_dw.group_number = X.number_;
        auto cbfa = cudnnConvolutionBwdFilterAlgo_t(method_dw.algo);
        auto tensor = cudnnSetConvolutionMathType(conv_desc(), cudnnMathType_t(method_dw.math_type));
        auto scbw = cudnnConvolutionBackwardFilter(gpu->cudnn_handle_, &a, X.cudnn_desc(), X.data(), Y.cudnn_desc(), Y.d().data(),
            conv_desc(), cbfa, workspace_dw.data(), workspace_dw.getDataSizeInByte(), &r, filter_desc(), W.d().data());
        if (scbw)
        {
            LOG_ERR("CONV backward W error: status {}, {}\n", cudnnGetErrorString(scbw), X.gpu()->lastCudnnErrorString());
        }
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

        if (method_dw.algo < 0)
        {
            int n;
            size_t ws_size;
            miopenConvAlgoPerf_t cfap[conv_method_count];
            miopenConvolutionBackwardWeightsGetWorkSpaceSize(gpu->miopen_handle_, Y.miopen_desc(), X.miopen_desc(), op_desc, W.miopen_desc(), &ws_size);
            if (ws_size > workspace_dw.getDataSizeInByte())
            {
                workspace_dw.resize(1, 1, ws_size / workspace_dw.getDataTypeSize() + 2, 1);
            }
            miopenFindConvolutionBackwardWeightsAlgorithm(gpu->miopen_handle_, Y.miopen_desc(), Y.d().data(), X.miopen_desc(), X.data(),
                op_desc, W.miopen_desc(), W.d().data(), conv_method_count, &n, cfap, workspace_dw.data(), workspace_dw.getDataSizeInByte(), true);
            method_dw.algo = cfap[0].bwd_weights_algo;
        }
        else if (method_dw.group_number != X.number_)
        {
            size_t ws_size;
            auto s = miopenConvolutionBackwardWeightsGetWorkSpaceSize(gpu->miopen_handle_, Y.miopen_desc(), X.miopen_desc(), op_desc, W.miopen_desc(), &ws_size);
            if (s == 0 && ws_size > workspace_dw.getDataSizeInByte())
            {
                workspace_dw.resize(1, 1, ws_size / workspace_dw.getDataTypeSize() + 2, 1);
            }
        }
        method_dw.group_number = X.number_;
        auto scf = miopenConvolutionBackwardWeights(gpu->miopen_handle_, &a, Y.miopen_desc(), Y.d().data(), X.miopen_desc(), X.data(),
            op_desc, miopenConvBwdWeightsAlgorithm_t(method_dw.algo), &r, W.miopen_desc(), W.d().data(), workspace_dw.data(), workspace_dw.getDataSizeInByte());
    }
    else
    {
        //W.d().scale(rw);
        //辅助矩阵的尺寸
        int row = W.width_ * W.height_ * W.channel_;
        int col = Y.width_ * Y.height_;
        Matrix X_ex({ row, col }, X.getDataType(), UnitType::CPU);
        X_ex.fillData(0);
        if (method_dw.algo < 0)
        {
            method_dw.algo = 0;
            workspace_dw.resize(1, 1, row * col, 1);
            workspace_dw.fillData(-1);
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
                        workspace_dw.setData(p_ex, 0, pX);
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
                X_ex.setData(j, X_sub.getData(workspace_dw.getData(j, 0)));
            }
            MatrixEx::mul(X_ex, dY_sub, W.d(), a, r);    //有点麻烦，暂时不管
        }
    }
}

//随机让一些点不参与计算
void MatrixEx::dropoutForward(const Matrix& X, Matrix& Y, float v, int seed)
{
    assert(checkMatrixDevice({ &X, &Y }));
    auto gpu = X.gpu();
    auto work_phase = gpu->active_phase_;
    if (work_phase == ACTIVE_PHASE_TEST)
    {
        Matrix::copyData(X, Y);
        Y.scale(v);
        return;
    }
    //int op_desc_buf[64] = { 0 };
    //auto op_desc = (cudnnDropoutDescriptor_t)op_desc_buf;
    cudnnDropoutDesc op_desc;
    if (Y.isCuda())
    {
        if (Y.workspace_.empty())
        {
            size_t size1, size2;
            cudnnDropoutGetStatesSize(gpu->cudnn_handle_, &size1);
            Matrix rg_stat({ int(size1 / sizeof(float) + 1), 1 });    //int64->int32
            cudnnDropoutGetReserveSpaceSize(X.cudnn_desc(), &size2);
            Matrix reverse_space({ int(size2 / sizeof(float) + 1), 1 });    //int64->int32
            //LOG(stderr, "dropout size {},{}\n", size, size2);
            Y.workspace_ = { rg_stat, reverse_space };
        }
        auto& rg_stat = Y.workspace_[0];
        auto& reverse_space = Y.workspace_[1];

        cudnnSetDropoutDescriptor(op_desc(), gpu->cudnn_handle_, v, rg_stat.data(), rg_stat.getDataSizeInByte(), seed);
        cudnnDropoutForward(gpu->cudnn_handle_, op_desc(), X.cudnn_desc(), X.data(),
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

void MatrixEx::dropoutBackward(Matrix& X, const Matrix& Y, float v, int seed)
{}

//批归一化
void MatrixEx::batchNormalizationForward(const Matrix& X, Matrix& Y, BatchNormalizationType bn_type,
    float& exp_aver_factor, float epsilon, Matrix& scale, Matrix& bias)
{
    assert(checkMatrixDevice({ &X, &Y }));
    auto gpu = X.gpu();
    auto work_phase = gpu ? gpu->active_phase_ : ACTIVE_PHASE_TRAIN;
    if (Y.workspace_.empty())
    {
        auto dim = scale.getDim();
        auto dt = X.getDataType();
        auto dev = X.getDeviceType();
        Y.workspace_.resize(6);
        Y.workspace_[0] = Matrix(dim, dt, dev);    //running_mean
        Y.workspace_[0].fillData(0);
        Y.workspace_[1] = Matrix(dim, dt, dev);    //running_variance
        Y.workspace_[1].fillData(1);
        Y.workspace_[2] = Matrix(dim, dt, dev);    //save_mean
        Y.workspace_[2].fillData(0);
        Y.workspace_[3] = Matrix(dim, dt, dev);    //save_inv_variance
        Y.workspace_[3].fillData(0);
        Y.workspace_[4] = Matrix(dim, dt, dev);    //dscale
        Y.workspace_[4].fillData(0);
        Y.workspace_[5] = Matrix(dim, dt, dev);    //dbias
        Y.workspace_[5].fillData(0);
    }
    auto& ws = Y.workspace_;
    if (Y.isCuda())
    {
        if (work_phase == ACTIVE_PHASE_TRAIN)
        {
            cudnnBatchNormalizationForwardTraining(gpu->cudnn_handle_, cudnnBatchNormMode_t(bn_type),
                &const_real_1, &const_real_0, X.cudnn_desc(), X.data(), Y.cudnn_desc(), Y.data(),
                scale.cudnn_desc(), scale.data(), bias.data(), exp_aver_factor, ws[0].data(), ws[1].data(),
                std::max(epsilon * 1.0, CUDNN_BN_MIN_EPSILON), ws[2].data(), ws[3].data());
            exp_aver_factor = 1 / (1 / exp_aver_factor + 1);
        }
        if (work_phase == ACTIVE_PHASE_TEST)
        {
            cudnnBatchNormalizationForwardInference(gpu->cudnn_handle_, cudnnBatchNormMode_t(bn_type),
                &const_real_1, &const_real_0, X.cudnn_desc(), X.data(), Y.cudnn_desc(), Y.data(),
                scale.cudnn_desc(), scale.data(), bias.data(), ws[0].data(), ws[1].data(),
                std::max(epsilon * 1.0, CUDNN_BN_MIN_EPSILON));
        }
    }
    else if (Y.isHip())
    {
        if (work_phase == ACTIVE_PHASE_TRAIN)
        {
            miopenBatchNormalizationForwardTraining(gpu->miopen_handle_, miopenBatchNormMode_t(bn_type),
                (void*)&const_real_1, (void*)&const_real_0, X.miopen_desc(), X.data(), Y.miopen_desc(), Y.data(),
                scale.miopen_desc(), scale.data(), bias.data(), exp_aver_factor, ws[0].data(), ws[1].data(),
                std::max(epsilon * 1.0, CUDNN_BN_MIN_EPSILON), ws[2].data(), ws[3].data());
            exp_aver_factor = 1 / (1 / exp_aver_factor + 1);
        }
        if (work_phase == ACTIVE_PHASE_TEST)
        {
            miopenBatchNormalizationForwardInference(gpu->miopen_handle_, miopenBatchNormMode_t(bn_type),
                (void*)&const_real_1, (void*)&const_real_0, X.miopen_desc(), X.data(), Y.miopen_desc(), Y.data(),
                scale.miopen_desc(), scale.data(), bias.data(), ws[0].data(), ws[1].data(),
                std::max(epsilon * 1.0, CUDNN_BN_MIN_EPSILON));
        }
    }
}

void MatrixEx::batchNormalizationBackward(Matrix& X, const Matrix& Y, BatchNormalizationType bn_type,
    float epsilon, Matrix& scale, Matrix& bias)
{}

//Layer Normalization 实现
//inner = X.width_, outer = X.row_/X.width_ * X.number_
//workspace 布局: [0]=mean (outer), [1]=invstd (outer), [2]=dscale (inner), [3]=dbias (inner)
void MatrixEx::layerNormalizationForward(const Matrix& X, Matrix& Y, Matrix& scale, Matrix& bias, float epsilon)
{
    assert(checkMatrixDevice({ &X, &Y, &scale, &bias }));
    int inner = X.width_;
    int outer = (X.row_ / X.width_) * X.number_;
    auto dt = X.getDataType();
    auto dev = X.getDeviceType();
    if (Y.workspace_.empty())
    {
        Y.workspace_.resize(4);
        // mean/invstd are always written as float32 by GPU kernels regardless of input type
        Y.workspace_[0] = Matrix({ outer }, DataType::FLOAT, dev);    //mean
        Y.workspace_[1] = Matrix({ outer }, DataType::FLOAT, dev);    //invstd
        Y.workspace_[2] = Matrix({ inner }, dt, dev);                 //dscale
        Y.workspace_[3] = Matrix({ inner }, dt, dev);                 //dbias
    }
    auto& ws = Y.workspace_;
    if (X.isCuda())
    {
        cuda_layer_norm_fwd(X.getDataTypeByInt(), X.data(), Y.data(), scale.data(), bias.data(),
            ws[0].data(), ws[1].data(), outer, inner, epsilon);
    }
    else if (X.isHip())
    {
        hip_layer_norm_fwd(X.getDataTypeByInt(), X.data(), Y.data(), scale.data(), bias.data(),
            ws[0].data(), ws[1].data(), outer, inner, epsilon);
    }
    else
    {
        //CPU 路径
        for (int g = 0; g < outer; g++)
        {
            float mean = 0;
            for (int i = 0; i < inner; i++) { mean += X.getData(g * inner + i); }
            mean /= inner;
            float var = 0;
            for (int i = 0; i < inner; i++)
            {
                float d = X.getData(g * inner + i) - mean;
                var += d * d;
            }
            var /= inner;
            float invstd = 1.0f / std::sqrt(var + epsilon);
            for (int i = 0; i < inner; i++)
            {
                float xhat = (X.getData(g * inner + i) - mean) * invstd;
                float s = scale.getData(i);
                float b = bias.getData(i);
                Y.setData(g * inner + i, xhat * s + b);
            }
            ws[0].setData(g, mean);
            ws[1].setData(g, invstd);
        }
    }
}

void MatrixEx::layerNormalizationBackward(Matrix& X, const Matrix& Y, Matrix& scale, Matrix& bias, float epsilon)
{}

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
        hip_ada_delta_update(mean_d2.getDataTypeByInt(), mean_d2.data(), mean_ada_d2.data(), d.data(), ada_d.data(), d.data_size_, rou, epsilon);
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
        hip_adam_update(mean_d.getDataTypeByInt(), mean_d.data(), mean_d2.data(), d.data(), ada_d.data(), d.data_size_, beta1, beta2, epsilon, t);
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
    else if (d.isHip())
    {
        hip_rms_prop_update(mean_d2.getDataTypeByInt(), mean_d2.data(), d.data(), ada_d.data(), d.data_size_, rou, epsilon);
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
    else if (rou_hat.isHip())
    {
        hip_sparse(R.getDataTypeByInt(), rou_hat.data(), R.data(), R.data_size_, rou, beta);
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
    case DataType::BFLOAT16:
    {
        std::vector<bfloat16> temp1(m.getDataSize());
        for (int i = 0; i < m.getDataSize(); i++)
        {
            temp1[i] = bfloat16(temp[i]);
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
    else if (X.isHip())
    {
        hip_sin(X.getDataTypeByInt(), X.data(), Y.data(), X.getDataSize(), a, 0);
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
    else if (X.isHip())
    {
        hip_cos(X.getDataTypeByInt(), X.data(), Y.data(), X.getDataSize(), a, 0);
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
    else if (X.isHip())
    {
        hip_zigzag(X.getDataTypeByInt(), X.data(), Y.data(), X.getDataSize(), 1, 0);
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
    else if (X.isHip())
    {
        hip_zigzagb(X.getDataTypeByInt(), X.data(), X.d().data(), Y.data(), Y.d().data(), X.getDataSize(), 1, 0);
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
    else if (X.isHip())
    {
        hip_step(X.getDataTypeByInt(), X.data(), Y.data(), X.getDataSize(), 0, 0);
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
    else if (X.isHip())
    {
        hip_leaky_relu(X.getDataTypeByInt(), X.data(), Y.data(), X.getDataSize(), l, 1, 0);
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
    else if (X.isHip())
    {
        hip_leaky_relub(X.getDataTypeByInt(), X.data(), X.d().data(), Y.data(), Y.d().data(), X.getDataSize(), l, 1, 0);
    }
    else
    {
        //未完成
    }
}

//此功能过于复杂且不实用，暂时废弃
void MatrixEx::correlationForward(const Matrix& X, const Matrix& W, Matrix& Y, const std::vector<int>& stride, const std::vector<int>& padding, float a, float r)
{
    float epsilon = 1e-4;

    int need_resize = 1;
    auto& workspaces = Y.workspace_;
    if (workspaces.size() < 17)
    {
        workspaces.resize(17);
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

    auto& Y_copy = workspaces[16];

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
    Matrix::elementMul(X, X, X_square);                                                    //X_square平方
    MatrixEx::convolutionForward(X_square, W_1, Y_den_square, stride, padding, 1, 0);      //Y_den_square临时保存平方和
    MatrixEx::convolutionForward(X, W_1, Y_aver, stride, padding, 1.0 / W.getRow(), 0);    //Y_aver保存平均值
    Matrix::elementMul(Y_aver, Y_aver, Y_den);                                             //Y_den临时保存平均值平方
    Matrix::add(Y_den_square, Y_den, Y_den_square, 1, -W.getRow());
    Y_den_square.addNumber(epsilon);
    Matrix::elementPow(Y_den_square, Y_den, 0.5);    //Y_den_square，Y_den计算完毕

    //计算原图与归一化核的卷积，并除以分母
    MatrixEx::convolutionForward(X, W_norm, Y, stride, padding, 1, 0);    //X与归一化后W的卷积
    Matrix::copyData(Y, Y_copy);
    MatrixEx::convolutionForward(X1, W_norm, Y_conv_aver, stride, padding, 1, 0);    //X的元素全为1时与W的卷积
    Matrix::elementMul(Y_conv_aver, Y_aver, Y_conv_aver);                            //元素乘平均值
    Matrix::add(Y, Y_conv_aver, Y, 1, -1);                                           //相减计算出分子的前半部分

    Matrix::elementDiv(Y, Y_den, Y);
    Matrix::add(Y, Y_copy, Y);
}

void MatrixEx::correlationBackward(Matrix& X, Matrix& W, const Matrix& Y, std::vector<int>& methods, const std::vector<int>& stride, const std::vector<int>& padding, float a /*= 1*/, float rx /*= 0*/, float rw /*= 0*/)
{}

void MatrixEx::matrix_max(const Matrix& X1, const Matrix& X2, Matrix& Y)
{
    assert(checkMatrixDevice({ &X1, &X2, &Y }));
    if (X1.isCuda())
    {
        cuda_max(X1.getDataTypeByInt(), X1.data(), X2.data(), Y.data(), X1.getDataSize(), 1, 1, 0);
    }
    else if (X1.isHip())
    {
        hip_max(X1.getDataTypeByInt(), X1.data(), X2.data(), Y.data(), X1.getDataSize(), 1, 1, 0);
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
    else if (X1.isHip())
    {
        hip_maxb(X1.getDataTypeByInt(), X1.data(), X1.d().data(), X2.data(), X2.d().data(), Y.data(), Y.d().data(), X1.getDataSize(), a1, a2, 1);
    }
    else
    {
        //未完成
    }
}

void MatrixEx::zero_limit(const Matrix& A, const Matrix& B, Matrix& R, float beta_a, float beta_b)
{
    assert(checkMatrixDevice({ &A, &B, &R }));
    if (A.isCuda())
    {
        cuda_zero_limit(A.getDataTypeByInt(), A.data(), B.data(), R.data(), A.getDataSize(), beta_a, beta_b);
    }
    else if (A.isHip())
    {
        hip_zero_limit(A.getDataTypeByInt(), A.data(), B.data(), R.data(), A.getDataSize(), beta_a, beta_b);
    }
    else
    {
    }
}

//RMS Normalization
//inner = X.width_, outer = X.row_/X.width_ * X.number_
//workspace 布局: [0]=invstd (outer), [1]=dscale (inner)
void MatrixEx::rmsNormForward(const Matrix& X, Matrix& Y, Matrix& scale, float epsilon)
{
    assert(checkMatrixDevice({ &X, &Y, &scale }));
    int inner = X.width_;
    int outer = (X.row_ / X.width_) * X.number_;
    auto dt = X.getDataType();
    auto dev = X.getDeviceType();
    if (Y.workspace_.empty())
    {
        Y.workspace_.resize(2);
        Y.workspace_[0] = Matrix({ outer }, DataType::FLOAT, dev);    //invstd: always float for numerical stability
        Y.workspace_[1] = Matrix({ inner }, dt, dev);                 //dscale
    }
    auto& ws = Y.workspace_;
    if (X.isCuda())
    {
        cuda_rms_norm_fwd(X.getDataTypeByInt(), X.data(), Y.data(), scale.data(), scale.getDataTypeByInt(),
            ws[0].data(), outer, inner, epsilon,
            (X.getDataType() == DataType::FP8_E4M3 || X.getDataType() == DataType::FP8_E5M2) ? X.getQuantScale() : 1.0f);
    }
    else if (X.isHip())
    {
        hip_rms_norm_fwd(X.getDataTypeByInt(), X.data(), Y.data(), scale.data(),
            ws[0].data(), outer, inner, epsilon);
    }
    else
    {
        for (int g = 0; g < outer; g++)
        {
            float sqsum = 0;
            for (int i = 0; i < inner; i++)
            {
                float v = X.getData(g * inner + i);
                sqsum += v * v;
            }
            float invstd = 1.0f / std::sqrt(sqsum / inner + epsilon);
            for (int i = 0; i < inner; i++)
            {
                Y.setData(g * inner + i, X.getData(g * inner + i) * invstd * scale.getData(i));
            }
            ws[0].setData(g, invstd);
        }
    }
}

void MatrixEx::rmsNormBackward(Matrix& X, const Matrix& Y, Matrix& scale, float epsilon)
{}

//4D 任意轴置换
//维度数组取自 X.getDim() (恒为长度 4: {W, H, C, N})
//输出已由调用者 resize 为正确形状
static inline void permute4d_cpu(const Matrix& X, Matrix& Y, const std::vector<int>& perm)
{
    int in_d[4] = { X.getWidth(), X.getHeight(), X.getChannel(), X.getNumber() };
    int out_d[4] = { in_d[perm[0]], in_d[perm[1]], in_d[perm[2]], in_d[perm[3]] };
    int64_t total = (int64_t)out_d[0] * out_d[1] * out_d[2] * out_d[3];
    for (int64_t idx = 0; idx < total; idx++)
    {
        int o[4];
        int64_t t = idx;
        o[0] = t % out_d[0];
        t /= out_d[0];
        o[1] = t % out_d[1];
        t /= out_d[1];
        o[2] = t % out_d[2];
        t /= out_d[2];
        o[3] = (int)t;
        int in_coord[4] = { 0, 0, 0, 0 };
        in_coord[perm[0]] = o[0];
        in_coord[perm[1]] = o[1];
        in_coord[perm[2]] = o[2];
        in_coord[perm[3]] = o[3];
        int64_t in_lin = in_coord[0]
            + (int64_t)in_coord[1] * in_d[0]
            + (int64_t)in_coord[2] * in_d[0] * in_d[1]
            + (int64_t)in_coord[3] * in_d[0] * in_d[1] * in_d[2];
        Y.setData(idx, X.getData(in_lin));
    }
}

void MatrixEx::permute4dForward(const Matrix& X, Matrix& Y, const std::vector<int>& perm)
{
    assert(checkMatrixDevice({ &X, &Y }));
    if (X.isCuda())
    {
        if (!X.data() || !Y.data())
        {
            fprintf(stderr, "permute4dForward: NULL ptr X=%p Y=%p type=%d dim=(%d,%d,%d,%d)\n",
                X.data(), Y.data(), X.getDataTypeByInt(),
                X.getWidth(), X.getHeight(), X.getChannel(), X.getNumber());
            return;
        }
        cuda_permute4d(X.getDataTypeByInt(), X.data(), Y.data(),
            X.getWidth(), X.getHeight(), X.getChannel(), X.getNumber(),
            perm[0], perm[1], perm[2], perm[3]);
    }
    else if (X.isHip())
    {
        hip_permute4d(X.getDataTypeByInt(), X.data(), Y.data(),
            X.getWidth(), X.getHeight(), X.getChannel(), X.getNumber(),
            perm[0], perm[1], perm[2], perm[3]);
    }
    else
    {
        permute4d_cpu(X, Y, perm);
    }
}

//反向: dX_axis_perm[i] = dY_axis_i, 等价于对 dY 用 inverse perm
void MatrixEx::permute4dBackward(Matrix& X, const Matrix& Y, const std::vector<int>& perm)
{}

//RoPE half-rotate
void MatrixEx::ropeForward(const Matrix& X, Matrix& Y, const Matrix& cos_tab, const Matrix& sin_tab, int pos_offset /*= 0*/)
{
    assert(checkMatrixDevice({ &X, &Y, &cos_tab, &sin_tab }));
    int D = X.getWidth();
    int T = X.getHeight();
    int B = X.getNumber();
    if (X.isCuda())
    {
        cuda_rope_fwd(X.getDataTypeByInt(), X.data(), Y.data(),
            cos_tab.data(), sin_tab.data(), D, T, B, (unsigned int)pos_offset);
    }
    else if (X.isHip())
    {
        hip_rope_fwd(X.getDataTypeByInt(), X.data(), Y.data(),
            cos_tab.data(), sin_tab.data(), D, T, B, (unsigned int)pos_offset);
    }
    else
    {
        int half = D / 2;
        for (int b = 0; b < B; b++)
        {
            for (int t = 0; t < T; t++)
            {
                int base = (b * T + t) * D;
                for (int i = 0; i < half; i++)
                {
                    float c = cos_tab.getData((t + pos_offset) * half + i);
                    float s = sin_tab.getData((t + pos_offset) * half + i);
                    float xl = X.getData(base + i);
                    float xr = X.getData(base + half + i);
                    Y.setData(base + i, xl * c - xr * s);
                    Y.setData(base + half + i, xr * c + xl * s);
                }
            }
        }
    }
}

void MatrixEx::ropeBackward(Matrix& X, const Matrix& Y, const Matrix& cos_tab, const Matrix& sin_tab)
{}

//RoPE interleaved (ncnn mode=1): y[2i]=x[2i]*c-x[2i+1]*s, y[2i+1]=x[2i+1]*c+x[2i]*s
void MatrixEx::ropeInterleavedForward(const Matrix& X, Matrix& Y, const Matrix& cos_tab, const Matrix& sin_tab, int pos_offset)
{
    assert(checkMatrixDevice({ &X, &Y, &cos_tab, &sin_tab }));
    int D = X.getWidth();
    int T = X.getHeight();
    int B = X.getNumber();
    if (X.isCuda())
    {
        cuda_rope_interleaved_fwd(X.getDataTypeByInt(), X.data(), Y.data(),
            cos_tab.data(), sin_tab.data(), D, T, B, (unsigned int)pos_offset);
    }
    else if (X.isHip())
    {
        hip_rope_interleaved_fwd(X.getDataTypeByInt(), X.data(), Y.data(),
            cos_tab.data(), sin_tab.data(), D, T, B, (unsigned int)pos_offset);
    }
    else
    {
        int half = D / 2;
        for (int b = 0; b < B; b++)
        {
            for (int t = 0; t < T; t++)
            {
                int base = (b * T + t) * D;
                for (int i = 0; i < half; i++)
                {
                    float c = cos_tab.getData((t + pos_offset) * half + i);
                    float s = sin_tab.getData((t + pos_offset) * half + i);
                    float xl = X.getData(base + 2 * i);
                    float xr = X.getData(base + 2 * i + 1);
                    Y.setData(base + 2 * i, xl * c - xr * s);
                    Y.setData(base + 2 * i + 1, xr * c + xl * s);
                }
            }
        }
    }
}

//pixel_shuffle: X (W, H, C_out*r*r, N) -> Y (W*r, H*r, C_out, N)
void MatrixEx::pixelShuffleForward(const Matrix& X, Matrix& Y, int r)
{
    assert(checkMatrixDevice({ &X, &Y }));
    int W = X.getWidth(), H = X.getHeight(), C_in = X.getChannel(), N = X.getNumber();
    int C_out = C_in / (r * r);
    if (X.isCuda())
    {
        cuda_pixel_shuffle_fwd(X.getDataTypeByInt(), X.data(), Y.data(),
            (unsigned)W, (unsigned)H, (unsigned)C_out, (unsigned)r, (unsigned)N);
    }
    else if (X.isHip())
    {
        hip_pixel_shuffle_fwd(X.getDataTypeByInt(), X.data(), Y.data(),
            (unsigned)W, (unsigned)H, (unsigned)C_out, (unsigned)r, (unsigned)N);
    }
    else
    {
        int W_out = W * r, H_out = H * r;
        for (int n = 0; n < N; n++)
        {
            for (int c_out = 0; c_out < C_out; c_out++)
            {
                for (int y_out = 0; y_out < H_out; y_out++)
                {
                    for (int x_out = 0; x_out < W_out; x_out++)
                    {
                        int x_in = x_out / r, y_in = y_out / r;
                        int dx = x_out % r, dy = y_out % r;
                        int c_in = c_out * r * r + dy * r + dx;
                        float v = X.getData(x_in + y_in * W + c_in * W * H + n * W * H * C_in);
                        Y.setData(x_out + y_out * W_out + c_out * W_out * H_out + n * W_out * H_out * C_out, v);
                    }
                }
            }
        }
    }
}

void MatrixEx::pixelShuffleBackward(Matrix& X, const Matrix& Y, int r)
{}

//Scaled Dot-Product Attention: Y = softmax_channel(K^T @ Q * scale) @ V
//Q/K/V/Y: (D, T, 1, B); scale = 1/sqrt(dk)
//workspace 布局: [0]=attn (T,T,1,B) 正向 softmax 输出(反向用), [1]=dAttn (T,T,1,B) 临时梯度, [2]=dScores (T,T,1,B) 临时梯度
void MatrixEx::attentionForward(const Matrix& Q, const Matrix& K, const Matrix& V, Matrix& Y,
    float dk, int causal, int pos_offset /*= 0*/)
{
    attentionForwardImpl(Q, K, V, Y, nullptr, dk, causal, pos_offset);
}

void MatrixEx::attentionForward(const Matrix& Q, const Matrix& K, const Matrix& V, Matrix& Y,
    const Matrix& bias, float dk, int causal, int pos_offset /*= 0*/)
{
    attentionForwardImpl(Q, K, V, Y, &bias, dk, causal, pos_offset);
}

void MatrixEx::attentionForwardImpl(const Matrix& Q, const Matrix& K, const Matrix& V,
    Matrix& Y, const Matrix* bias, float dk, int causal, int pos_offset)
{
    assert(checkMatrixDevice({ &Q, &K, &V, &Y }));
    int D = Q.getWidth(), T_q = Q.getHeight(), B = Q.getNumber();
    int T_k = K.getHeight();    // may differ from T_q when using KV-cache
    float scale = 1.0f / std::sqrt(dk);
    auto dt = Q.getDataType();
    auto dev = Q.getDeviceType();
    // In FP8 mode, use BF16 for attention score workspace so cuDNN softmax works correctly.
    // (cuDNN maps FP8 → FLOAT internally, causing 4× OOB access on FP8 score buffers.)
    bool fp8_mode = (dt == DataType::FP8_E4M3 || dt == DataType::FP8_E5M2);
    DataType ws_dt = fp8_mode ? DataType::BFLOAT16 : dt;
    // Re-allocate workspace if T_q or T_k changed, or if FP8 mode needs extra Y_bf16 slot
    bool need_alloc = Y.workspace_.empty()
        || Y.workspace_[0].getWidth() != T_k
        || Y.workspace_[0].getHeight() != T_q
        || Y.workspace_[0].getNumber() != B
        || (fp8_mode && (int)Y.workspace_.size() < 4);
    if (need_alloc)
    {
        Y.workspace_.resize(fp8_mode ? 4 : 3);
        Y.workspace_[0] = Matrix({ T_k, T_q, 1, B }, ws_dt, dev);    // attn: softmax 输出 (BF16 in FP8 mode)
        Y.workspace_[1] = Matrix({ T_k, T_q, 1, B }, ws_dt, dev);    // dAttn: 反向临时缓冲
        Y.workspace_[2] = Matrix({ T_k, T_q, 1, B }, ws_dt, dev);    // dScores: 反向临时缓冲
        if (fp8_mode)
        {
            Y.workspace_[3] = Matrix({ D, T_q, 1, B }, DataType::BFLOAT16, dev);    // BF16 output buffer before FP8 quant
        }
    }
    auto& attn = Y.workspace_[0];
    // scores = K^T @ Q * (1/sqrt(dk)) → attn: shape (T_k, T_q, 1, B)
    Matrix::mulBatched(K, Q, attn, MATRIX_TRANS, MATRIX_NO_TRANS, scale);
    // float16 overflow guard: clamp ±inf/nan to ±60000 before softmax
    // (CUBLAS stores fp32 scores as fp16; values >65504 become inf → NaN in softmax)
    if (dt == DataType::HALF && attn.isCuda())
    {
        cuda_clamp_scores_half(attn.data(), (unsigned int)attn.getDataSize(), 60000.0f);
    }
    // optional additive bias (e.g. Box RPB): added to scores before softmax
    if (bias != nullptr)
    {
        Matrix::add(attn, *bias, attn);
    }
    // 因果掩码: 屏蔽未来位置 k > q + pos_offset
    if (causal)
    {
        if (attn.isCuda())
        {
            cuda_causal_mask(attn.getDataTypeByInt(), attn.data(), T_q, T_k, B, pos_offset);
        }
        else if (attn.isHip())
        {
            hip_causal_mask(attn.getDataTypeByInt(), attn.data(), T_q, T_k, B, pos_offset);
        }
        else
        {
            // CPU: attn 形状 (T_k, T_q, 1, B), 线性索引 k + q*T_k + b*T_q*T_k
            for (int b = 0; b < B; b++)
            {
                for (int q = 0; q < T_q; q++)
                {
                    for (int k = q + pos_offset + 1; k < T_k; k++)
                    {
                        attn.setData(k + q * T_k + (long long)b * T_k * T_q, -1e9f);
                    }
                }
            }
        }
    }
    // attn = softmax_channel(scores), in-place
    std::vector<int> iv;
    std::vector<float> rv;
    MatrixEx::activeForward(attn, attn, ACTIVE_FUNCTION_SOFTMAX_CHANNEL, iv, rv);
    // Y = V @ attn: V(D, T_k, 1, B) x attn(T_k, T_q, 1, B) = Y(D, T_q, 1, B)
    if (fp8_mode && Y.isCuda())
    {
        // V(FP8) × attn(BF16) → Y_bf16 via W8A16 path, then quantize to FP8 with scale=1.0
        auto& Y_bf16 = Y.workspace_[3];
        Matrix::mulBatched(V, attn, Y_bf16);
        cuda_convert(Y_bf16.data(), BFLOAT16, Y.data(), FP8_E4M3, (unsigned int)Y.getDataSize(), 1.0f);
    }
    else
    {
        Matrix::mulBatched(V, attn, Y);
    }
}

void MatrixEx::attentionBackward(Matrix& Q, Matrix& K, Matrix& V, const Matrix& Y, float dk, int causal)
{}

// ============================================================
// ROI Align forward
// feat: (W,H,C,B) WHCN; boxes: (4,N,1,B) WHCN [x1,y1,x2,y2];
// Y: (roi_size,roi_size,C,N*B)  aligned=True convention.
// ============================================================
void MatrixEx::roiAlignForward(const Matrix& feat, const Matrix& boxes, Matrix& Y,
    int roi_size, float spatial_scale)
{
    int W = feat.getWidth(), H = feat.getHeight(), C = feat.getChannel(), B = feat.getNumber();
    int N_boxes = boxes.getHeight();
    Y.resize({ roi_size, roi_size, C, N_boxes * B });
    if (feat.isCuda())
    {
        cuda_roi_align_fwd(feat.data(), W, H, C, B,
            boxes.data(), N_boxes, Y.data(), roi_size, spatial_scale);
    }
    else
    {
        // CPU bilinear interpolation fallback
        int total = roi_size * roi_size * C * N_boxes * B;
        for (int idx = 0; idx < total; idx++)
        {
            int RSRSC = roi_size * roi_size * C;
            int roi_idx = idx / RSRSC, rem = idx % RSRSC;
            int c = rem / (roi_size * roi_size), sp = rem % (roi_size * roi_size);
            int oy = sp / roi_size, ox = sp % roi_size;
            int box_n = roi_idx % N_boxes, batch_b = roi_idx / N_boxes;
            int box_off = box_n * 4 + batch_b * 4 * N_boxes;
            float x1 = boxes.getData(box_off + 0) * spatial_scale;
            float y1 = boxes.getData(box_off + 1) * spatial_scale;
            float x2 = boxes.getData(box_off + 2) * spatial_scale;
            float y2 = boxes.getData(box_off + 3) * spatial_scale;
            float bin_w = (x2 - x1) / roi_size, bin_h = (y2 - y1) / roi_size;
            float xs = x1 + (ox + 0.5f) * bin_w - 0.5f;
            float ys = y1 + (oy + 0.5f) * bin_h - 0.5f;
            if (xs < -1.f || xs > W || ys < -1.f || ys > H)
            {
                Y.setData(idx, 0.f);
                continue;
            }
            xs = std::max(xs, 0.f);
            ys = std::max(ys, 0.f);
            int ix0 = (int)xs, ix1 = std::min(ix0 + 1, W - 1);
            int iy0 = (int)ys, iy1 = std::min(iy0 + 1, H - 1);
            ix0 = std::min(ix0, W - 1);
            iy0 = std::min(iy0, H - 1);
            float dx = xs - ix0, dy = ys - iy0;
            int fb = c * W * H + batch_b * W * H * C;
            float v = (1 - dx) * (1 - dy) * feat.getData(ix0 + iy0 * W + fb) + dx * (1 - dy) * feat.getData(ix1 + iy0 * W + fb)
                + (1 - dx) * dy * feat.getData(ix0 + iy1 * W + fb) + dx * dy * feat.getData(ix1 + iy1 * W + fb);
            Y.setData(idx, v);
        }
    }
}

// ============================================================
// ROI Align backward
// Scatter-add grad_out to feat.d() via bilinear weights.
// grad_feat must be zeroed (or accumulated with keepWeight) before call.
// ============================================================
void MatrixEx::roiAlignBackward(const Matrix& feat, const Matrix& boxes, Matrix& Y,
    int roi_size, float spatial_scale)
{}

// ============================================================
// Embedding lookup
// ids: (T,1,1,B)  float-as-int  W: (D,1,1,V)  Y: (D,T,1,B)
// ============================================================
void MatrixEx::embedForward(const Matrix& ids, const Matrix& W, Matrix& Y)
{
    assert(checkMatrixDevice({ &ids, &W, &Y }));
    int D = W.getWidth();
    int T = ids.getWidth();    // ids shape (T,1,1,B)
    int B = ids.getNumber();
    if (ids.isCuda())
    {
        // FP8 W + FP8 Y: byte-copy encoded FP8 rows from the weight cache and preserve W's
        // quantization scale on Y. The first consumer is RMSNorm, which explicitly dequantizes
        // by Y.quant_scale_ before re-encoding to scale=1.0, so embedding precision is retained.
        if ((W.getDataType() == DataType::FP8_E4M3 || W.getDataType() == DataType::FP8_E5M2)
            && (Y.getDataType() == DataType::FP8_E4M3 || Y.getDataType() == DataType::FP8_E5M2))
        {
            cuda_embed_fwd_fp8_to_fp8(ids.data(), W.data(), Y.data(), D, T, B, 1.0f);
            Y.setQuantScale(W.getQuantScale());
        }
        // BF16 W + FP8 Y: 专用转换 kernel，避免 BF16(2字节) 写入 FP8(1字节) 缓冲区溢出
        else if (W.getDataType() == DataType::BFLOAT16
            && (Y.getDataType() == DataType::FP8_E4M3 || Y.getDataType() == DataType::FP8_E5M2))
        {
            if (Y.workspace_.size() < 3
                || Y.workspace_[0].getDim() != Y.getDim()
                || Y.workspace_[0].getDataType() != DataType::BFLOAT16)
            {
                Y.workspace_.resize(3);
                Y.workspace_[0] = Matrix(Y.getDim(), DataType::BFLOAT16, Y.getDeviceType());
                Y.workspace_[1] = Matrix({ 1 }, DataType::FLOAT, Y.getDeviceType());
                Y.workspace_[2] = Matrix({ 1 }, DataType::FLOAT, Y.getDeviceType());
            }
            auto& y_bf16 = Y.workspace_[0];
            auto& absmax_tmp = Y.workspace_[1];
            auto& inv_scale_dev = Y.workspace_[2];
            cuda_embed_fwd(W.getDataTypeByInt(), ids.data(), W.data(), y_bf16.data(), D, T, B, 1.0f);
            cuda_bf16_to_fp8e4m3_dynamic(y_bf16.data(), Y.data(), (unsigned int)Y.getDataSize(), absmax_tmp.data(), inv_scale_dev.data());
            float inv_scale = 1.0f;
            cudaMemcpy(&inv_scale, inv_scale_dev.data(), sizeof(float), cudaMemcpyDeviceToHost);
            Y.setQuantScale(inv_scale > 0.f ? inv_scale : 1.0f);
        }
        else
        {
            cuda_embed_fwd(W.getDataTypeByInt(), ids.data(), W.data(), Y.data(), D, T, B,
                W.getQuantScale());
        }
    }
    else if (ids.isHip())
    {
        hip_embed_fwd(W.getDataTypeByInt(), ids.data(), W.data(), Y.data(), D, T, B);
    }
    else
    {
        const float* id_ptr = ids.dataf();
        const float* w_ptr = W.dataf();
        float* y_ptr = Y.dataf();
        for (int b = 0; b < B; b++)
        {
            for (int t = 0; t < T; t++)
            {
                int id = (int)id_ptr[t + (int64_t)b * T];
                const float* row = w_ptr + (int64_t)id * D;
                float* dst = y_ptr + (int64_t)(b * T + t) * D;
                for (int d = 0; d < D; d++)
                {
                    dst[d] = row[d];
                }
            }
        }
    }
}

void MatrixEx::embedBackward(const Matrix& ids, Matrix& W, const Matrix& Y)
{}

// ============================================================
// Tile: X (W_in, H_in, C_in, N_in) -> Y (W_out, H_out, C_out, N_out)
// repeats[4]: r0=W, r1=H, r2=C, r3=N
// ============================================================
void MatrixEx::tileForward(const Matrix& X, Matrix& Y, const std::vector<int>& repeats)
{
    assert(checkMatrixDevice({ &X, &Y }));
    int W_in = X.getWidth(), H_in = X.getHeight(), C_in = X.getChannel(), N_in = X.getNumber();
    int W_out = W_in * repeats[0], H_out = H_in * repeats[1],
        C_out = C_in * repeats[2], N_out = N_in * repeats[3];
    if (X.isCuda())
    {
        cuda_tile_fwd(X.getDataTypeByInt(), X.data(), Y.data(),
            W_in, H_in, C_in, N_in, W_out, H_out, C_out, N_out);
    }
    else if (X.isHip())
    {
        hip_tile_fwd(X.getDataTypeByInt(), X.data(), Y.data(),
            W_in, H_in, C_in, N_in, W_out, H_out, C_out, N_out);
    }
    else
    {
        const float* x_ptr = X.dataf();
        float* y_ptr = Y.dataf();
        int64_t total = (int64_t)W_out * H_out * C_out * N_out;
        for (int64_t idx = 0; idx < total; idx++)
        {
            int64_t t = idx;
            int w = (int)(t % W_out);
            t /= W_out;
            int h = (int)(t % H_out);
            t /= H_out;
            int c = (int)(t % C_out);
            t /= C_out;
            int n = (int)t;
            int64_t x_idx = (w % W_in)
                + (int64_t)(h % H_in) * W_in
                + (int64_t)(c % C_in) * W_in * H_in
                + (int64_t)(n % N_in) * W_in * H_in * C_in;
            y_ptr[idx] = x_ptr[x_idx];
        }
    }
}

void MatrixEx::tileBackward(Matrix& X, const Matrix& Y, const std::vector<int>& repeats)
{}

// ===========================================================================
// 转置卷积 (Deconvolution / Transposed Convolution)
// A: [W_in, H_in, C_in, N], C_in = W.getNumber()
// W: [kW, kH, C_out, C_in], C_out = W.getChannel()
// Y: [W_out, H_out, C_out, N]
// ===========================================================================

void MatrixEx::deconvolutionForward(const Matrix& A, const Matrix& W, Matrix& Y,
    const std::vector<int>& stride, const std::vector<int>& padding, float a, float r)
{
    assert(checkMatrixDevice({ &A, &W, &Y }));
    assert(stride.size() >= 2 && padding.size() >= 2);
    // A.channel = W.number (deconv input channels), Y.channel = W.channel (deconv output channels)
    assert(A.getChannel() == W.getNumber() && Y.getChannel() == W.getChannel());

    //若工作空间为空则创建一个（HIP/CPU 路径仍需使用）
    if (Y.workspace_.empty())
    {
        Y.workspace_.resize(1);
    }

    auto& method = Y.user_data<ConvMethod>();
    auto gpu = A.gpu();
    if (!gpu->user_data_.has_value())
    {
        gpu->user_data_ = Matrix();
    }
    auto& workspace = gpu->getUserData<Matrix>();

    if (Y.isCuda())
    {
        cudnnConvolutionDesc conv_desc;
        cudnnFilterDesc filter_desc;
        // 滤波器描述与正向卷积相同: (W.number_, W.channel_, kH, kW)
        cudnnSetConvolution2dDescriptor(conv_desc(), padding[1], padding[0], stride[1], stride[0], 1, 1,
            CUDNN_CROSS_CORRELATION, toCudnnDataType(Y.getDataType()));
        cudnnSetFilter4dDescriptor(filter_desc(), toCudnnDataType(W.getDataType()), CUDNN_TENSOR_NCHW,
            W.number_, W.channel_, W.height_, W.width_);
        cudnnSetConvolutionMathType(conv_desc(), CUDNN_DEFAULT_MATH);

        //寻找最快的算法
        if (method.algo < 0)
        {
            int n;
            cudnnConvolutionBwdDataAlgoPerf_t cbdap[conv_method_count];
            cudnnGetConvolutionBackwardDataAlgorithm_v7(gpu->cudnn_handle_, filter_desc(),
                A.cudnn_desc(), conv_desc(), Y.cudnn_desc(), conv_method_count, &n, cbdap);
            method.algo = (n > 0 && cbdap[0].status == CUDNN_STATUS_SUCCESS) ? int(cbdap[0].algo) : int(CUDNN_CONVOLUTION_BWD_DATA_ALGO_0);
            method.math_type = (n > 0 && cbdap[0].status == CUDNN_STATUS_SUCCESS) ? int(cbdap[0].mathType) : int(CUDNN_DEFAULT_MATH);
            size_t ws_size = (n > 0 && cbdap[0].status == CUDNN_STATUS_SUCCESS) ? cbdap[0].memory : 0;
            if (ws_size > workspace.getDataSizeInByte())
            {
                workspace.resize(1, 1, 1, ws_size / workspace.getDataTypeSize() + 2);
            }
        }
        else if (method.group_number != A.number_)
        {
            size_t ws_size;
            auto s = cudnnGetConvolutionBackwardDataWorkspaceSize(gpu->cudnn_handle_, filter_desc(),
                A.cudnn_desc(), conv_desc(), Y.cudnn_desc(), cudnnConvolutionBwdDataAlgo_t(method.algo), &ws_size);
            if (s == CUDNN_STATUS_SUCCESS && ws_size > workspace.getDataSizeInByte())
            {
                workspace.resize(1, 1, 1, ws_size / workspace.getDataTypeSize() + 2);
            }
        }
        method.group_number = A.number_;
        cudnnSetConvolutionMathType(conv_desc(), cudnnMathType_t(method.math_type));
        auto cbda = cudnnConvolutionBwdDataAlgo_t(method.algo);
        auto s = cudnnConvolutionBackwardData(gpu->cudnn_handle_, &a,
            filter_desc(), W.data(),
            A.cudnn_desc(), A.data(),
            conv_desc(), cbda,
            workspace.data(), workspace.getDataSizeInByte(),
            &r, Y.cudnn_desc(), Y.data());
        if (s)
        {
            LOG_ERR("DECONV forward error: {}\n", cudnnGetErrorString(s));
        }
    }
    else
    {
        // CPU fallback: 直接散射累加 (scatter-add)
        if (r == 0)
        {
            Y.fillData(0);
        }
        else
        {
            for (int i = 0; i < Y.getDataSize(); i++)
            {
                Y.setData(i, Y.getData(i) * r);
            }
        }
        int W_in = A.width_, H_in = A.height_, C_in = A.channel_, N = A.number_;
        int kW = W.width_, kH = W.height_, C_out = W.channel_;
        int W_out = Y.width_, H_out = Y.height_;
        int pad_w = padding[0], pad_h = padding[1];
        int str_w = stride[0], str_h = stride[1];
        for (int n = 0; n < N; n++)
        {
            for (int ci = 0; ci < C_in; ci++)
            {
                for (int co = 0; co < C_out; co++)
                {
                    for (int hi = 0; hi < H_in; hi++)
                    {
                        for (int wi = 0; wi < W_in; wi++)
                        {
                            float av = a * A.getData(wi + hi * W_in + ci * W_in * H_in + n * W_in * H_in * C_in);
                            for (int kh = 0; kh < kH; kh++)
                            {
                                for (int kw = 0; kw < kW; kw++)
                                {
                                    int wo = wi * str_w + kw - pad_w;
                                    int ho = hi * str_h + kh - pad_h;
                                    if (wo >= 0 && wo < W_out && ho >= 0 && ho < H_out)
                                    {
                                        int y_idx = wo + ho * W_out + co * W_out * H_out + n * W_out * H_out * C_out;
                                        int w_idx = kw + kh * kW + co * kW * kH + ci * kW * kH * C_out;
                                        Y.setData(y_idx, Y.getData(y_idx) + av * W.getData(w_idx));
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

void MatrixEx::deconvolutionBackwardDA(Matrix& A, const Matrix& W, const Matrix& Y,
    const std::vector<int>& stride, const std::vector<int>& padding, float a, float r)
{}

void MatrixEx::deconvolutionBackwardDW(const Matrix& A, Matrix& W, const Matrix& Y,
    const std::vector<int>& stride, const std::vector<int>& padding, float a, float r)
{}

// ===========================================================================
// Group Normalization
// X: [W,H,C,N], scale/bias: [C], Y: [W,H,C,N]
// 归一化: 每组 inner=W*H*CperG 元素; outer=G*N 组
// workspace: [0]=mean (outer), [1]=invstd (outer), [2]=X_hat (同形 X), [3]=dscale [C], [4]=dbias [C]
// ===========================================================================

void MatrixEx::groupNormForward(const Matrix& X, Matrix& Y,
    const Matrix& scale, const Matrix& bias, int G, float epsilon)
{
    assert(checkMatrixDevice({ &X, &Y, &scale, &bias }));
    int W = X.width_, H = X.height_, C = X.channel_, N = X.number_;
    int CperG = C / G;
    int inner = W * H * CperG;    // elements per group per sample
    int outer = G * N;            // total groups
    auto dt = X.getDataType();
    auto dev = X.getDeviceType();

    if (Y.workspace_.empty())
    {
        Y.workspace_.resize(6);
        // mean/invstd 必须是 float，因为 layer_norm_fwd_half_kernel 写的是 float*，
        // 不管输入 X 是 half 还是 float
        auto stat_dt = DataType::FLOAT;
        Y.workspace_[0] = Matrix({ outer }, stat_dt, dev);    // mean
        Y.workspace_[1] = Matrix({ outer }, stat_dt, dev);    // invstd
        Y.workspace_[2] = Matrix(X.getDim(), dt, dev);        // X_hat (pre-affine normalized)
        Y.workspace_[3] = Matrix({ C }, dt, dev);             // dscale
        Y.workspace_[4] = Matrix({ C }, dt, dev);             // dbias
        Y.workspace_[5] = Matrix(X.getDim(), dt, dev);        // dX_hat (backward buffer)
    }
    auto& ws = Y.workspace_;

    if (X.isCuda())
    {
        {
            // Step 1: 归一化 (单位 scale/bias), 结果写入 ws[2] (X_hat)
            cuda_layer_norm_fwd(X.getDataTypeByInt(), X.data(), ws[2].data(),
                nullptr, nullptr,
                ws[0].data(), ws[1].data(), outer, inner, epsilon);
            // Step 2: 逐通道仿射变换 Y = scale[c] * X_hat + bias[c]
            cuda_group_norm_affine_fwd(X.getDataTypeByInt(),
                ws[2].data(), Y.data(), scale.data(), bias.data(),
                outer, inner, G, CperG, W * H);
        }
    }
    else if (X.isHip())
    {
        hip_layer_norm_fwd(X.getDataTypeByInt(), X.data(), ws[2].data(),
            nullptr, nullptr,
            ws[0].data(), ws[1].data(), outer, inner, epsilon);
        hip_group_norm_affine_fwd(X.getDataTypeByInt(),
            ws[2].data(), Y.data(), scale.data(), bias.data(),
            outer, inner, G, CperG, W * H);
    }
    else
    {
        // CPU 路径
        const float* xp = X.dataf();
        float* yp = Y.dataf();
        float* xhat_p = ws[2].dataf();
        float* mean_p = ws[0].dataf();
        float* invstd_p = ws[1].dataf();
        for (int idx = 0; idx < outer; idx++)
        {
            int n = idx / G;
            int g = idx % G;
            const float* xg = xp + (n * C + g * CperG) * W * H;
            float* xhat_g = xhat_p + idx * inner;
            float mean = 0.f;
            for (int i = 0; i < inner; i++)
            {
                mean += xg[i];
            }
            mean /= inner;
            float var = 0.f;
            for (int i = 0; i < inner; i++)
            {
                float d = xg[i] - mean;
                var += d * d;
            }
            var /= inner;
            float invstd = 1.f / std::sqrt(var + epsilon);
            mean_p[idx] = mean;
            invstd_p[idx] = invstd;
            for (int i = 0; i < inner; i++)
            {
                float xhat = (xg[i] - mean) * invstd;
                xhat_g[i] = xhat;
                // 通道索引: c = g*CperG + i/(W*H)
                int c = g * CperG + i / (W * H);
                yp[idx * inner + i] = scale.getData(c) * xhat + bias.getData(c);
            }
        }
    }
}

void MatrixEx::groupNormBackward(Matrix& X, const Matrix& Y,
    const Matrix& scale, const Matrix& bias, int G, float epsilon)
{}

// ===========================================================================
// VAE 重参数化 (Reparameterization Trick)
// mu, log_var: [D,1,1,B]; z: [D,1,1,B]
// workspace: z.workspace_[0] = eps (同形 mu), 供反向使用
// ===========================================================================

void MatrixEx::reparamForward(const Matrix& mu, const Matrix& log_var, Matrix& z)
{
    assert(checkMatrixDevice({ &mu, &log_var, &z }));
    auto gpu = mu.gpu();
    auto work_phase = gpu ? gpu->active_phase_ : ACTIVE_PHASE_TRAIN;

    if (work_phase == ACTIVE_PHASE_TEST)
    {
        // 推理阶段: z = mu (确定性)
        Matrix::copyData(mu, z);
        return;
    }

    // 训练阶段: 生成 eps ~ N(0,1) 并存入 workspace (与 mu 同设备)
    if (z.workspace_.empty())
    {
        z.workspace_.resize(1);
        z.workspace_[0] = Matrix(mu.getDim(), mu.getDataType(), mu.getDeviceType());
    }
    auto& eps = z.workspace_[0];
    if (eps.getDataSize() != mu.getDataSize())
    {
        eps = Matrix(mu.getDim(), mu.getDataType(), mu.getDeviceType());
    }

    int sz = (int)mu.getDataSize();

    // Box-Muller 在 CPU 生成标准正态噪声
    Random<float> rng;
    std::vector<float> eps_host(sz);
    for (int i = 0; i + 1 < sz; i += 2)
    {
        float u1 = rng.rand() + 1e-7f;
        float u2 = rng.rand();
        float r_ = std::sqrt(-2.f * std::log(u1));
        float theta = 2.f * (float)M_PI * u2;
        eps_host[i] = r_ * std::cos(theta);
        eps_host[i + 1] = r_ * std::sin(theta);
    }
    if (sz % 2 == 1)
    {
        float u1 = rng.rand() + 1e-7f;
        float u2 = rng.rand();
        eps_host[sz - 1] = std::sqrt(-2.f * std::log(u1)) * std::cos(2.f * (float)M_PI * u2);
    }

    if (mu.isCuda())
    {
        // 将 CPU 生成的 eps 上传到 GPU eps 矩阵, 再调用 CUDA kernel
        Matrix eps_cpu(mu.getDim(), mu.getDataType(), UnitType::CPU);
        for (int i = 0; i < sz; i++)
        {
            eps_cpu.setData(i, eps_host[i]);
        }
        Matrix::copyData(eps_cpu, eps);    // CPU → GPU
        cuda_reparam_fwd(mu.getDataTypeByInt(),
            mu.data(), log_var.data(), eps.data(), z.data(), (unsigned)sz);
    }
    else if (mu.isHip())
    {
        Matrix eps_cpu(mu.getDim(), mu.getDataType(), UnitType::CPU);
        for (int i = 0; i < sz; i++)
        {
            eps_cpu.setData(i, eps_host[i]);
        }
        Matrix::copyData(eps_cpu, eps);
        hip_reparam_fwd(mu.getDataTypeByInt(),
            mu.data(), log_var.data(), eps.data(), z.data(), (unsigned)sz);
    }
    else
    {
        // CPU 路径 (eps 本身就在 CPU)
        for (int i = 0; i < sz; i++)
        {
            eps.setData(i, eps_host[i]);
        }
        for (int i = 0; i < sz; i++)
        {
            float std_i = std::exp(log_var.getData(i) * 0.5f);
            z.setData(i, mu.getData(i) + std_i * eps.getData(i));
        }
    }
}

void MatrixEx::reparamBackward(Matrix& mu, Matrix& log_var, const Matrix& z)
{}

// ============================================================
// L1 loss backward: dA[i] = beta*dA[i] + alpha*sign(A[i]-Y[i])
// ============================================================
void MatrixEx::l1LossBackward(Matrix& A, const Matrix& Y, float alpha, float beta)
{}

// ============================================================
// KL log_var backward: dlv[i] = beta*dlv[i] + alpha*0.5*(exp(lv[i])-1)
// ============================================================
void MatrixEx::klLvBackward(Matrix& log_var, float alpha, float beta)
{}

// ============================================================
// Upsample forward: X(W,H,C,N) -> Y(W*sw, H*sh, C, N)
// ============================================================
void MatrixEx::upsampleForward(const Matrix& X, Matrix& Y, int sh, int sw, bool bilinear)
{
    assert(checkMatrixDevice({ &X, &Y }));
    int W = X.getWidth(), H = X.getHeight(), C = X.getChannel(), N = X.getNumber();
    if (X.isCuda())
    {
        if (bilinear)
        {
            int W_out = Y.getWidth(), H_out = Y.getHeight();
            cuda_upsample_bilinear_fwd(X.getDataTypeByInt(), X.data(), Y.data(),
                (unsigned)W, (unsigned)H, (unsigned)W_out, (unsigned)H_out,
                (unsigned)C, (unsigned)N);
        }
        else
        {
            cuda_upsample_nearest_fwd(X.getDataTypeByInt(), X.data(), Y.data(),
                (unsigned)W, (unsigned)H, (unsigned)C, (unsigned)N, sh, sw);
        }
    }
    else if (X.isHip())
    {
        if (bilinear)
        {
            int W_out = Y.getWidth(), H_out = Y.getHeight();
            hip_upsample_bilinear_fwd(X.getDataTypeByInt(), X.data(), Y.data(),
                (unsigned)W, (unsigned)H, (unsigned)W_out, (unsigned)H_out,
                (unsigned)C, (unsigned)N);
        }
        else
        {
            hip_upsample_nearest_fwd(X.getDataTypeByInt(), X.data(), Y.data(),
                (unsigned)W, (unsigned)H, (unsigned)C, (unsigned)N, sh, sw);
        }
    }
    else
    {
        int W_out = W * sw, H_out = H * sh;
        const float* x = X.dataf();
        float* y = Y.dataf();
        if (!bilinear)
        {
            for (int n = 0; n < N; n++)
            {
                for (int c = 0; c < C; c++)
                {
                    for (int h = 0; h < H; h++)
                    {
                        for (int w = 0; w < W; w++)
                        {
                            float val = x[w + h * W + c * W * H + n * W * H * C];
                            for (int sy = 0; sy < sh; sy++)
                            {
                                for (int sx = 0; sx < sw; sx++)
                                {
                                    y[(w * sw + sx) + (h * sh + sy) * W_out + c * W_out * H_out + n * W_out * H_out * C] = val;
                                }
                            }
                        }
                    }
                }
            }
        }
        else
        {
            for (int n = 0; n < N; n++)
            {
                for (int c = 0; c < C; c++)
                {
                    for (int yo = 0; yo < H_out; yo++)
                    {
                        for (int xo = 0; xo < W_out; xo++)
                        {
                            float src_w = (xo + 0.5f) * W / (float)W_out - 0.5f;
                            float src_h = (yo + 0.5f) * H / (float)H_out - 0.5f;
                            src_w = std::max(0.0f, std::min(src_w, (float)(W - 1)));
                            src_h = std::max(0.0f, std::min(src_h, (float)(H - 1)));
                            int x0 = (int)src_w, x1 = std::min(x0 + 1, W - 1);
                            int y0 = (int)src_h, y1 = std::min(y0 + 1, H - 1);
                            float dx = src_w - x0, dy = src_h - y0;
                            int base = c * W * H + n * W * H * C;
                            float v = (1 - dx) * (1 - dy) * x[base + y0 * W + x0]
                                + dx * (1 - dy) * x[base + y0 * W + x1]
                                + (1 - dx) * dy * x[base + y1 * W + x0]
                                + dx * dy * x[base + y1 * W + x1];
                            y[xo + yo * W_out + c * W_out * H_out + n * W_out * H_out * C] = v;
                        }
                    }
                }
            }
        }
    }
}

// ============================================================
// Upsample backward: dX from dY
// ============================================================
void MatrixEx::upsampleBackward(Matrix& X, const Matrix& Y, int sh, int sw, bool bilinear)
{}

}    // namespace cccc