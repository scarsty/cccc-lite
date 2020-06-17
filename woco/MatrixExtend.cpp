#include "MatrixExtend.h"
#include "Random.h"
#include "VectorMath.h"
#include <cassert>

namespace woco
{

//若bias某一维度为1，则视为对X的对应维度加上同样的值，cudnn的文档中指出最高支持到5维
//此处仅用于两种情况：1 - bias仅有channel不为1，2 - bias的number为1，注意1是2的一个特例，其他情况不一定能得到正确结果
void MatrixExtend::addBias(const Matrix& A, const Matrix& bias, Matrix& R, realc a, realc b)
{
    assert(checkMatrixDevice({ &A, &bias, &R }));
    assert(bias.getDataSize() == bias.getChannel() || bias.getNumber() == 1);
    copyData(A, R);
    if (A.inGPU())
    {
        auto cuda = A.cuda();
        if (bias.dim_.size() <= 5)
        {
            cudnnAddTensor(cuda->cudnn_handle_, &a, bias.getCudnnTensorDesc(), bias.data(), &b, R.getCudnnTensorDesc(), R.data());
        }
        else
        {
            CudaControl::setTensorDesc4D(cuda->tensor_desc_, bias.width_, bias.height_, bias.channel_, bias.number_);
            CudaControl::setTensorDesc4D(cuda->tensor_desc2_, R.width_, R.height_, R.channel_, R.number_);
            cudnnAddTensor(cuda->cudnn_handle_, &a, cuda->tensor_desc_, bias.data(), &b, cuda->tensor_desc2_, R.data());
        }
    }
    else
    {
        if (bias.getDataSize() == bias.getChannel())
        {
            for (int i = 0; i < R.data_size_; i++)
            {
                int c = i % R.getRow() / (R.getRow() / R.getChannel());
                R.data()[i] += bias.data()[c];
            }
        }
        else
        {
            for (int i = 0; i < R.data_size_; i++)
            {
                int c = i % bias.getDataSize();
                R.data()[i] += bias.data()[c];
            }
        }
    }
}

void MatrixExtend::addBiasBackward(Matrix& A, Matrix& bias, const Matrix& R, realc a, realc b)
{
    assert(checkMatrixDevice({ &A, &bias, &R }));
    if (A.needReverse())
    {
        Matrix::add(A.DMatrix(), R.DMatrix(), A.DMatrix(), 1, a);
    }
    if (bias.needReverse())
    {
        if (A.inGPU())
        {
            //用卷积反向代替一般反向，此处待验证
            auto cuda = R.cuda();
            cudnnConvolutionBackwardBias(cuda->cudnn_handle_, &b, R.getCudnnTensorDesc(), R.DMatrix().data(), &const_real_1, bias.getCudnnTensorDesc(), bias.DMatrix().data());
        }
        else
        {
            if (bias.DMatrix().data())
            {
                //bias.DMatrix().scale(r);
                //这个就是对对应的dR求和
                for (int n = 0; n < R.number_; n++)
                {
                    for (int c = 0; c < R.channel_; c++)
                    {
                        bias.DMatrix().getData(0, 0, c, 0) += a * VectorMath::sum(R.DMatrix().getDataPointer(0, 0, c, n), R.width_ * R.height_);
                    }
                }
            }
        }
    }
}

void MatrixExtend::concatByChannel(const std::vector<Matrix>& A_vector, Matrix& R)
{
    for (int n = 0; n < R.getCol(); n++)
    {
        int c_off = 0;
        for (int i = 0; i < A_vector.size(); i++)
        {
            auto& tmp = A_vector[i];
            copyDataPointer(tmp, tmp.getDataPointer(0, 0, 0, n), R, R.getDataPointer(0, 0, c_off, n), tmp.getRow());
            c_off += tmp.getChannel();
        }
    }
}

void MatrixExtend::concatByChannelBackward(std::vector<Matrix>& A_vector, const Matrix& R)
{
    for (int n = 0; n < R.getCol(); n++)
    {
        int c_off = 0;
        for (int i = 0; i < A_vector.size(); i++)
        {

            auto& tmp = A_vector[i].DMatrix();
            if (A_vector[i].needReverse())
            {
                copyDataPointer(R, R.DMatrix().getDataPointer(0, 0, c_off, n), tmp, tmp.getDataPointer(0, 0, 0, n), tmp.getRow());
            }
            c_off += tmp.getChannel();
        }
    }
}

void MatrixExtend::splitByChannel(const Matrix& A, std::vector<Matrix>& R_vector)
{
    for (int n = 0; n < A.getCol(); n++)
    {
        int c_off = 0;
        for (int i = 0; i < R_vector.size(); i++)
        {
            auto& tmp = R_vector[i];
            copyDataPointer(A, A.getDataPointer(0, 0, c_off, n), tmp, tmp.getDataPointer(0, 0, 0, n), tmp.getRow());
            c_off += tmp.getChannel();
        }
    }
}

//初始化激活需要的缓冲区
void MatrixExtend::activeBufferInit(ActiveFunctionType af, Matrix& A, std::vector<int>& int_vector, std::vector<Matrix>& matrix_vector)
{
    auto cuda = A.cuda();
    switch (af)
    {
    case ACTIVE_FUNCTION_DROPOUT:
        if (A.inGPU())
        {
            size_t size1, size2;
            cudnnDropoutGetStatesSize(cuda->cudnn_handle_, &size1);
            Matrix rg_stat(size1 / sizeof(real) + 1, 1);
            cudnnDropoutGetReserveSpaceSize(A.getCudnnTensorDesc(), &size2);
            Matrix reverse_space(size2 / sizeof(real) + 1, 1);
            //fprintf(stderr, "dropout size %d,%d\n", size, size2);
            matrix_vector = { rg_stat, reverse_space };
        }
        break;
    case ACTIVE_FUNCTION_LOCAL_CONSTRAST_NORMALIZATION:
    case ACTIVE_FUNCTION_DIVISIVE_NORMALIZATION:
        if (A.inGPU())
        {
            matrix_vector = { Matrix(A.dim_), Matrix(A.dim_), Matrix(1, 1, 1, A.getDataSizeInByte() * 2) };
        }
        break;
    case ACTIVE_FUNCTION_BATCH_NORMALIZATION:
        if (A.inGPU())
        {
            cudnnDeriveBNTensorDescriptor(cuda->tensor_desc_, A.getCudnnTensorDesc(), cudnnBatchNormMode_t(int_vector[1]));
            int w, h, c, n, p1, p2, p3, p4;
            cudnnDataType_t dt;
            cudnnGetTensor4dDescriptor(cuda->tensor_desc_, &dt, &n, &c, &h, &w, &p1, &p2, &p3, &p4);
            Size size = { w, h, c, n };
            matrix_vector = { Matrix(size).initData(0.5), Matrix(size), Matrix(size), Matrix(size), Matrix(size), Matrix(size), Matrix(size), Matrix(size) };
        }
        break;
    case ACTIVE_FUNCTION_SPATIAL_TRANSFORMER:
        if (A.inGPU())
        {
            matrix_vector = { Matrix(3, 2, 1, A.number_), Matrix(A).dim_, Matrix(3, 2, 1, A.number_), Matrix(A).dim_ };
        }
        break;
    case ACTIVE_FUNCTION_RECURRENT:
        break;
    case ACTIVE_FUNCTION_ZERO_CHANNEL:
    {
        //1 matrix: factor for mul
        Matrix m(A.dim_, DeviceType::CPU);
        m.initData(1);
        for (int w = 0; w < A.getWidth(); w++)
        {
            for (int h = 0; h < A.getHeight(); h++)
            {
                for (int c : int_vector)
                {
                    for (int n = 0; n < A.getNumber(); n++)
                    {
                        m.getData(w, h, c, n) = 0;
                    }
                }
            }
            m.toGPU();
            matrix_vector = { m };
        }
        break;
    }
    default:
        break;
    }
}

//正向激活，依据X计算A
//此处我们定义激活操作为输入和输出矩阵（或张量）的维度完全相同
void MatrixExtend::activeForward(const Matrix& A, Matrix& R, ActiveFunctionType af, real a, real r)
{
    assert(checkMatrixDevice({ &A, &R }));
    auto nan = CUDNN_NOT_PROPAGATE_NAN;
    auto cuda = A.cuda();
    switch (af)
    {
    case ACTIVE_FUNCTION_NONE:
        Matrix::copyData(A, R);
        break;
    case ACTIVE_FUNCTION_SIGMOID:
    case ACTIVE_FUNCTION_SIGMOID_CE:
        if (A.inGPU())
        {
            CudaControl::setActivationDesc(cuda->activation_desc_, CUDNN_ACTIVATION_SIGMOID, 1);
            cudnnActivationForward(cuda->cudnn_handle_, cuda->activation_desc_,
                &a, A.getCudnnTensorDesc(), A.data(), &r, R.getCudnnTensorDesc(), R.data());
        }
        else
        {
            VectorMath::sigmoid_v(A.data(), R.data(), R.data_size_, a, r);
        }
        break;
    case ACTIVE_FUNCTION_RELU:
        if (A.inGPU())
        {
            CudaControl::setActivationDesc(cuda->activation_desc_, CUDNN_ACTIVATION_RELU, 1);
            cudnnActivationForward(cuda->cudnn_handle_, cuda->activation_desc_,
                &a, A.getCudnnTensorDesc(), A.data(), &r, R.getCudnnTensorDesc(), R.data());
        }
        else
        {
            VectorMath::relu_v(A.data(), R.data(), R.data_size_, a, r);
        }
        break;
    case ACTIVE_FUNCTION_TANH:
        if (A.inGPU())
        {
            CudaControl::setActivationDesc(cuda->activation_desc_, CUDNN_ACTIVATION_TANH, 1);
            cudnnActivationForward(cuda->cudnn_handle_, cuda->activation_desc_,
                &a, A.getCudnnTensorDesc(), A.data(), &r, R.getCudnnTensorDesc(), R.data());
        }
        else
        {
            VectorMath::tanh_v(A.data(), R.data(), R.data_size_, a, r);
        }
        break;
    case ACTIVE_FUNCTION_SOFTMAX:
    case ACTIVE_FUNCTION_SOFTMAX_CE:
        if (A.inGPU())
        {
            CudaControl::setTensorDesc4D(cuda->tensor_desc_, 1, 1, A.row_, A.number_);
            cudnnSoftmaxForward(cuda->cudnn_handle_, CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_INSTANCE,
                &a, cuda->tensor_desc_, A.data(), &r, cuda->tensor_desc_, R.data());
        }
        else
        {
            //因为数值问题，可能需要减去每列最大值
            MatrixExtend::copyData(A, R);
            for (int i = 0; i < R.number_; i++)
            {
                VectorMath::minus_max(R.getDataPointer(0, i), R.row_);
            }
            VectorMath::exp_v(R.data(), R.data(), R.data_size_);
            for (int i = 0; i < R.number_; i++)
            {
                real sum = R.sumAbsCol(i);
                if (sum == 0)
                {
                    continue;
                }
                R.scaleCol(1 / sum, i);
            }
        }
        break;
    case ACTIVE_FUNCTION_SOFTMAX_FAST:
    case ACTIVE_FUNCTION_SOFTMAX_FAST_CE:
        if (A.inGPU())
        {
            CudaControl::setTensorDesc4D(cuda->tensor_desc_, 1, 1, A.row_, A.number_);
            cudnnSoftmaxForward(cuda->cudnn_handle_, CUDNN_SOFTMAX_FAST, CUDNN_SOFTMAX_MODE_INSTANCE,
                &a, cuda->tensor_desc_, A.data(), &r, cuda->tensor_desc_, R.data());
        }
        else
        {
            VectorMath::exp_v(A.data(), R.data(), R.data_size_, a, r);
            for (int i = 0; i < R.number_; i++)
            {
                real sum = R.sumAbsCol(i);
                if (sum == 0)
                {
                    continue;
                }
                R.scaleCol(1 / sum, i);
            }
        }
        break;
    case ACTIVE_FUNCTION_SOFTMAX_LOG:
        if (A.inGPU())
        {
            CudaControl::setTensorDesc4D(cuda->tensor_desc_, 1, 1, A.row_, A.number_);
            cudnnSoftmaxForward(cuda->cudnn_handle_, CUDNN_SOFTMAX_LOG, CUDNN_SOFTMAX_MODE_INSTANCE,
                &a, cuda->tensor_desc_, A.data(), &r, cuda->tensor_desc_, R.data());
        }
        else
        {
            activeForward(A, R, ACTIVE_FUNCTION_SOFTMAX);
            VectorMath::log_v(R.data(), R.data(), R.data_size_, a, r);
        }
        break;
    case ACTIVE_FUNCTION_ABSMAX:
        //计算时尽量不要使用，只用在验证时
        if (R.data_size_ <= 0)
        {
            return;
        }
        if (A.inGPU())
        {
            Matrix T(R.row_, R.number_, DeviceType::CPU);
            T.scale(0);
            for (int i_group = 0; i_group < R.number_; i_group++)
            {
                int index = A.indexColMaxAbs(i_group);
                T.getData(index, i_group) = 1;
            }
            Matrix::copyData(T, R);
        }
        else
        {
            R.scale(0);
            for (int i_group = 0; i_group < R.number_; i_group++)
            {
                int index = A.indexColMaxAbs(i_group);
                R.getData(index, i_group) = 1;
            }
        }
        break;
    case ACTIVE_FUNCTION_SQUARE:
        Matrix::elementMul(A, A, R);
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
        MatrixExtend::sin(A, R, M_PI / 2);
        break;
    case ACTIVE_FUNCTION_ZIGZAG:
        MatrixExtend::zigzag(A, R);
        //Matrix::copyData(A, R);
        break;
    case ACTIVE_FUNCTION_SIN_STEP:
        MatrixExtend::sin(A, R, M_PI / 2);
        MatrixExtend::step(R, R);
        break;
    default:
        fprintf(stderr, "Parameters not enough!\n");
        break;
    }
}

//反向激活，依据X，A，dA计算dX
//softmax的cpu部分貌似不支持a，b
//这里的系数应该是不对,unfinished
void MatrixExtend::activeBackward(Matrix& A, const Matrix& R, ActiveFunctionType af, real a, real r)
{
    assert(checkMatrixDevice({ &A, &R }));
    auto nan = CUDNN_NOT_PROPAGATE_NAN;
    auto cuda = R.cuda();
    switch (af)
    {
    case ACTIVE_FUNCTION_NONE:
    case ACTIVE_FUNCTION_SIGMOID_CE:
    case ACTIVE_FUNCTION_SOFTMAX_CE:
    case ACTIVE_FUNCTION_SOFTMAX_FAST_CE:
        Matrix::add(R.DMatrix(), A.DMatrix(), A.DMatrix(), a, 1);
        break;
    case ACTIVE_FUNCTION_SIGMOID:
        if (A.inGPU())
        {
            CudaControl::setActivationDesc(cuda->activation_desc_, CUDNN_ACTIVATION_SIGMOID, 1);
            cudnnActivationBackward(cuda->cudnn_handle_, cuda->activation_desc_, &a, R.getCudnnTensorDesc(), R.data(),
                R.getCudnnTensorDesc(), R.DMatrix().data(), A.getCudnnTensorDesc(), A.data(), &r, A.getCudnnTensorDesc(), A.DMatrix().data());
        }
        else
        {
            VectorMath::sigmoid_vb(R.data(), R.DMatrix().data(), A.data(), A.DMatrix().data(), A.data_size_, a, r);
        }
        break;
    case ACTIVE_FUNCTION_RELU:
        if (A.inGPU())
        {
            CudaControl::setActivationDesc(cuda->activation_desc_, CUDNN_ACTIVATION_RELU, 1);
            cudnnActivationBackward(cuda->cudnn_handle_, cuda->activation_desc_, &a, R.getCudnnTensorDesc(), R.data(),
                R.getCudnnTensorDesc(), R.DMatrix().data(), A.getCudnnTensorDesc(), A.data(), &r, A.getCudnnTensorDesc(), A.DMatrix().data());
        }
        else
        {
            VectorMath::relu_vb(R.data(), R.DMatrix().data(), A.data(), A.DMatrix().data(), A.data_size_, a, r);
        }
        break;
    case ACTIVE_FUNCTION_TANH:
        //两者结果在1e-10的精度有区别
        if (A.inGPU())
        {
            CudaControl::setActivationDesc(cuda->activation_desc_, CUDNN_ACTIVATION_TANH, 1);
            cudnnActivationBackward(cuda->cudnn_handle_, cuda->activation_desc_, &a, R.getCudnnTensorDesc(), R.data(),
                R.getCudnnTensorDesc(), R.DMatrix().data(), A.getCudnnTensorDesc(), A.data(), &r, A.getCudnnTensorDesc(), A.DMatrix().data());
        }
        else
        {
            VectorMath::tanh_vb(R.data(), R.DMatrix().data(), A.data(), A.DMatrix().data(), A.data_size_, a, r);
        }
        break;
    case ACTIVE_FUNCTION_SOFTMAX:
    case ACTIVE_FUNCTION_SOFTMAX_FAST:
        if (A.inGPU())
        {
            auto softmax_flag = CUDNN_SOFTMAX_ACCURATE;
            if (af == ACTIVE_FUNCTION_SOFTMAX_FAST)
            {
                softmax_flag = CUDNN_SOFTMAX_FAST;
            }
            CudaControl::setTensorDesc4D(cuda->tensor_desc_, 1, 1, A.row_, A.number_);
            auto s = cudnnSoftmaxBackward(cuda->cudnn_handle_, softmax_flag, CUDNN_SOFTMAX_MODE_INSTANCE,
                &a, cuda->tensor_desc_, R.data(), cuda->tensor_desc_, R.DMatrix().data(), &r, cuda->tensor_desc_, A.DMatrix().data());
        }
        else
        {
            for (int i = 0; i < A.number_; i++)
            {
                real v = MatrixExtend::dotCol(R, i, R.DMatrix(), i);
                VectorMath::softmax_vb_sub(R.getDataPointer(0, i), R.DMatrix().getDataPointer(0, i), v, A.DMatrix().getDataPointer(0, i), A.row_);
            }
        }
        break;
    case ACTIVE_FUNCTION_SOFTMAX_LOG:
        if (A.inGPU())
        {
            CudaControl::setTensorDesc4D(cuda->tensor_desc_, 1, 1, A.row_, A.number_);
            auto s = cudnnSoftmaxBackward(cuda->cudnn_handle_, CUDNN_SOFTMAX_LOG, CUDNN_SOFTMAX_MODE_INSTANCE,
                &a, cuda->tensor_desc_, R.data(), cuda->tensor_desc_, R.DMatrix().data(), &r, cuda->tensor_desc_, A.DMatrix().data());
        }
        else
        {
            for (int i = 0; i < A.number_; i++)
            {
                real v = 0;
                for (int j = 0; j < A.row_; j++)
                {
                    v += R.DMatrix().getData(i, j);
                }
                VectorMath::softmaxlog_vb_sub(R.getDataPointer(0, i), R.DMatrix().getDataPointer(0, i), v, A.DMatrix().getDataPointer(0, i), A.row_);
            }
        }
        break;
    case ACTIVE_FUNCTION_ABSMAX:
        //似乎应该是返回一个常数矩阵，若考虑效率应当留空此处在外部处理
        //dX.initData(1);
        fprintf(stderr, "Unsupported backward of FINDMAX!\n");
        break;
    //case ACTIVE_FUNCTION_SUMMAX:
    //    Matrix::copyDataPointer(A,A.DMatrix().data(), X,X.DMatrix().data());
    //    for (int i = 0; i < X.getCol(); i++)
    //    {
    //        dX.addNumberCol(-Matrix::dotCol(A, i, dA, i), 1, i);
    //        dX.scaleCol(1.0 / X.sumAbsCol(i), i);
    //    }
    //    break;
    case ACTIVE_FUNCTION_SIN:
    case ACTIVE_FUNCTION_SIN_STEP:
        MatrixExtend::cos(A, A.DMatrix(), M_PI / 2);
        MatrixExtend::elementMul(A.DMatrix(), R.DMatrix(), A.DMatrix(), 1);    //不严格，需改为加法
        break;
    case ACTIVE_FUNCTION_ZIGZAG:
        MatrixExtend::zigzagb(A, R);
        break;
    default:
        fprintf(stderr, "Parameters not enough!\n");
        break;
    }
}

//参数更多的的激活函数，包含了前面的功能，
//传附加参数的时候使用了C++11的初始化列表，因此效率可能较低，实际上如不考虑效率可以代替基本激活函数
//调用时请自己保证参数数量的正确性！
//vector参数的含义请参考Activer.cpp中的注释，以及上面函数中matrix向量的含义
void MatrixExtend::activeForward2(const Matrix& A, Matrix& R, ActiveFunctionType af,
    std::vector<int>& int_vector, std::vector<real>& real_vector, std::vector<Matrix>& matrix_vector, real a, real r)
{
    assert(checkMatrixDevice({ &A, &R }));
    auto nan = CUDNN_NOT_PROPAGATE_NAN;
    auto cuda = A.cuda();
    switch (af)
    {
    case ACTIVE_FUNCTION_CLIPPED_RELU:
        if (A.inGPU())
        {
            CudaControl::setActivationDesc(cuda->activation_desc_, CUDNN_ACTIVATION_CLIPPED_RELU, real_vector[0]);
            cudnnActivationForward(cuda->cudnn_handle_, cuda->activation_desc_,
                &a, A.getCudnnTensorDesc(), A.data(), &r, R.getCudnnTensorDesc(), R.data());
        }
        else
        {
            VectorMath::clipped_relu_v(A.data(), R.data(), real_vector[0], R.data_size_, a, r);
        }
        break;
    case ACTIVE_FUNCTION_DROPOUT:
        MatrixExtend::dropoutForward(A, R, ActivePhaseType(int_vector[0]), real_vector[0], int_vector[1], matrix_vector[0], matrix_vector[1]);
        break;
    //以下无CPU支持
    case ACTIVE_FUNCTION_LOCAL_RESPONSE_NORMALIZATION:
        cudnnSetLRNDescriptor(cuda->lrn_desc_, int_vector[0], real_vector[0], real_vector[1], real_vector[2]);
        cudnnLRNCrossChannelForward(cuda->cudnn_handle_, cuda->lrn_desc_, CUDNN_LRN_CROSS_CHANNEL_DIM1,
            &a, A.getCudnnTensorDesc(), A.data(), &r, R.getCudnnTensorDesc(), R.data());
        break;
    case ACTIVE_FUNCTION_LOCAL_CONSTRAST_NORMALIZATION:
        //MatrixExtend::lcnForward(X, A, matrix_vector[0], matrix_vector[2], int_vector[0], real_vector[0], real_vector[1], real_vector[2], true);
        break;
    case ACTIVE_FUNCTION_DIVISIVE_NORMALIZATION:
        //MatrixExtend::lcnForward(X, A, matrix_vector[0], matrix_vector[2], int_vector[0], real_vector[0], real_vector[1], real_vector[2], false);
        break;
    case ACTIVE_FUNCTION_BATCH_NORMALIZATION:
        MatrixExtend::batchNormalizationForward(A, R, ActivePhaseType(int_vector[0]), BatchNormalizationType(int_vector[1]), real_vector[1], real_vector[2],
            matrix_vector[0], matrix_vector[1], matrix_vector[2], matrix_vector[3], matrix_vector[4], matrix_vector[5]);
        break;
    case ACTIVE_FUNCTION_SPATIAL_TRANSFORMER:
        //MatrixExtend::spatialTfSamplerForward(X, A, matrix_vector[0], matrix_vector[1]);
        break;
    case ACTIVE_FUNCTION_RECURRENT:
        //unfinished
        break;
    default:
        activeForward(A, R, af, a, r);
        break;
    }
}

//参考activeForward2
void MatrixExtend::activeBackward2(Matrix& A, const Matrix& R, ActiveFunctionType af,
    std::vector<int>& int_vector, std::vector<real>& real_vector, std::vector<Matrix>& matrix_vector, real a, real r)
{
    assert(checkMatrixDevice({ &A, &R }));
    auto nan = CUDNN_NOT_PROPAGATE_NAN;
    auto cuda = R.cuda();
    switch (af)
    {
    case ACTIVE_FUNCTION_CLIPPED_RELU:
        if (A.inGPU())
        {
            CudaControl::setActivationDesc(cuda->activation_desc_, CUDNN_ACTIVATION_CLIPPED_RELU, real_vector[0]);
            cudnnActivationBackward(cuda->cudnn_handle_, cuda->activation_desc_, &a, R.getCudnnTensorDesc(), R.data(),
                R.getCudnnTensorDesc(), R.DMatrix().data(), A.getCudnnTensorDesc(), A.data(), &r, A.getCudnnTensorDesc(), A.DMatrix().data());
        }
        else
        {
            VectorMath::clipped_relu_vb(R.data(), R.DMatrix().data(), A.data(), A.DMatrix().data(), real_vector[0], A.data_size_, a, r);
        }
        break;
    case ACTIVE_FUNCTION_DROPOUT:
        MatrixExtend::dropoutBackward(A, R, real_vector[0], int_vector[0], matrix_vector[0], matrix_vector[1]);
        break;
    case ACTIVE_FUNCTION_LOCAL_RESPONSE_NORMALIZATION:
        cudnnSetLRNDescriptor(cuda->lrn_desc_, int_vector[0], real_vector[0], real_vector[1], real_vector[2]);
        cudnnLRNCrossChannelBackward(cuda->cudnn_handle_, cuda->lrn_desc_, CUDNN_LRN_CROSS_CHANNEL_DIM1,
            &a, R.getCudnnTensorDesc(), R.data(), R.getCudnnTensorDesc(), R.DMatrix().data(), A.getCudnnTensorDesc(), A.data(), &r, A.getCudnnTensorDesc(), A.DMatrix().data());
        break;
    case ACTIVE_FUNCTION_LOCAL_CONSTRAST_NORMALIZATION:
        //MatrixExtend::lcnBackward(A, dA, X, dX, matrix_vector[0], matrix_vector[1], matrix_vector[2], int_vector[0], real_vector[0], real_vector[1], real_vector[2], true, 0);
        break;
    case ACTIVE_FUNCTION_DIVISIVE_NORMALIZATION:
        //MatrixExtend::lcnBackward(A, dA, X, dX, matrix_vector[0], matrix_vector[1], matrix_vector[2], int_vector[0], real_vector[0], real_vector[1], real_vector[2], false, real_vector[3]);
        break;
    case ACTIVE_FUNCTION_BATCH_NORMALIZATION:
        MatrixExtend::batchNormalizationBackward(A, R, ActivePhaseType(int_vector[0]), BatchNormalizationType(int_vector[1]), real_vector[2], real_vector[0],
            matrix_vector[0], matrix_vector[1], matrix_vector[4], matrix_vector[5], matrix_vector[6], matrix_vector[7]);
        break;
    case ACTIVE_FUNCTION_SPATIAL_TRANSFORMER:
        //MatrixExtend::spatialTfSamplerBackward(A, dA, X, dX, real_vector[0], matrix_vector[0], matrix_vector[1], matrix_vector[2], matrix_vector[3]);
        break;
    case ACTIVE_FUNCTION_RECURRENT:
        break;
    default:
        activeBackward(A, R, af, a, r);
        break;
    }
}

//池化
//gpu部分，平均模式下对padding的支持目前还有问题
void MatrixExtend::poolingForward(const Matrix& A, Matrix& R, PoolingType pooling_type,
    const std::vector<int>& window, const std::vector<int>& stride, const std::vector<int>& padding,
    realc a /*= 1*/, realc r /*= 0*/)
{
    assert(checkMatrixDevice({ &A, &R }));
    assert(window.size() >= 2 && stride.size() >= 2 && padding.size() >= 2);
    auto cuda = A.cuda();
    if (R.inGPU())
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
        auto t = cudnnPoolingForward(cuda->cudnn_handle_, cuda->pooling_desc_, &a, A.getCudnnTensorDesc(), A.data(), &r, R.getCudnnTensorDesc(), R.data());
        if (t)
        {
            fprintf(stderr, "POOL forward error %d\n", t);
        }
    }
    else
    {
        R.initData(0);
        for (int p = 0; p < R.number_ * R.channel_; p++)
        {
            for (int wA = 0; wA < R.width_; wA++)
            {
                for (int hA = 0; hA < R.height_; hA++)
                {
                    real v = 0;
                    if (pooling_type == POOLING_MAX)
                    {
                        v = -REAL_MAX;
                    }
                    int n = 0;
                    int wX0 = wA * stride[0] - padding[0];
                    int hX0 = hA * stride[1] - padding[1];
                    for (int wX = wX0; wX < std::min(A.width_, wX0 + window[0]); wX++)
                    {
                        for (int hX = hX0; hX < std::min(A.height_, hX0 + window[1]); hX++)
                        {
                            if (pooling_type == POOLING_AVERAGE_PADDING || pooling_type == POOLING_AVERAGE_NOPADDING)
                            {
                                if (A.haveData(wX, hX, p, 0))
                                {
                                    v += A.getData(wX, hX, p, 0);
                                }
                                n++;
                            }
                            else if (pooling_type == POOLING_MAX)
                            {
                                if (A.haveData(wX, hX, p, 0))
                                {
                                    auto x = A.getData(wX, hX, p, 0);
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
                    R.getData(wA, hA, p, 0) = v;
                }
            }
        }
    }
}

//使用cpu时利用了record -- 取消，直接计算
void MatrixExtend::poolingBackward(Matrix& A, const Matrix& R, PoolingType pooling_type,
    const std::vector<int>& window, const std::vector<int>& stride, const std::vector<int>& padding,
    realc a /*= 1*/, realc r /*= 0*/)
{
    assert(checkMatrixDevice({ &A, &R }));
    assert(window.size() >= 2 && stride.size() >= 2 && padding.size() >= 2);
    auto cuda = R.cuda();
    if (A.inGPU())
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
        cudnnPoolingBackward(cuda->cudnn_handle_, cuda->pooling_desc_,
            &a, R.getCudnnTensorDesc(), R.data(), R.getCudnnTensorDesc(), R.DMatrix().data(), A.getCudnnTensorDesc(), A.data(), &r, A.getCudnnTensorDesc(), A.DMatrix().data());
    }
    else
    {
        //dX->initData(0);
        for (int p = 0; p < R.number_ * R.channel_; p++)
        {
            for (int wdA = 0; wdA < R.width_; wdA++)
            {
                for (int hdA = 0; hdA < R.height_; hdA++)
                {
                    int wdX0 = wdA * stride[0] - padding[0];
                    int hdX0 = hdA * stride[1] - padding[1];
                    if (pooling_type == POOLING_MAX)
                    {
                        real max_v = -REAL_MAX;
                        real* max_p = nullptr;
                        for (int wdX = wdX0; wdX < std::min(A.width_, wdX0 + window[0]); wdX++)
                        {
                            for (int hdX = hdX0; hdX < std::min(A.height_, hdX0 + window[1]); hdX++)
                            {
                                if (A.haveData(wdX, hdX, p, 0))
                                {
                                    real v = A.getData(wdX, hdX, p, 0);
                                    if (v > max_v)
                                    {
                                        max_v = v;
                                        max_p = &A.DMatrix().getData(wdX, hdX, p, 0);
                                    }
                                }
                            }
                        }
                        if (max_p)
                        {
                            *max_p = R.DMatrix().getData(wdA, hdA, p, 0);
                        }
                    }
                    else
                    {
                        int n;
                        if (pooling_type == POOLING_AVERAGE_NOPADDING)
                        {
                            n = std::min(window[0], A.width_ - wdA * stride[0]) * std::min(window[1], A.height_ - hdA * stride[1]);
                        }
                        else
                        {
                            n = window[0] * window[1];
                        }
                        real v = R.DMatrix().getData(wdA, hdA, p, 0) / n;
                        for (int wdX = wdX0; wdX < std::min(A.width_, wdX0 + window[0]); wdX++)
                        {
                            for (int hdX = hdX0; hdX < std::min(A.height_, hdX0 + window[1]); hdX++)
                            {
                                if (A.haveData(wdX, hdX, p, 0))
                                {
                                    A.DMatrix().getData(wdX, hdX, p, 0) = a * v + r * A.DMatrix().getData(wdX, hdX, p, 0);
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
#define CONV_OPERATION1(A, W, R, pw, ph, sw, sh, DO_SOMETHING) \
    do { \
        for (int wR = 0; wR < R.width_; wR++) \
            for (int hR = 0; hR < R.height_; hR++) \
                for (int wW = 0; wW < W.width_; wW++) \
                    for (int hW = 0; hW < W.height_; hW++) \
                    { \
                        int wA = wR + wW - pw; \
                        int hA = hR + hW - ph; \
                        if (wA >= 0 && wA < A.width_ && hA >= 0 && hA < A.height_) \
                        { \
                            DO_SOMETHING \
                        } \
                    } \
    } while (0)

//前向卷积
//从外部引入辅助空间目的是降低初始化的次数
//当使用CUDA计算时，不需要辅助转换的整数数组，这时该数组会被初始化为两个元素，分别为算法和所需的工作空间大小，在首次计算的时候完成
//CPU模式仅支持stride为1，padding为0的二维卷积
//method保存方法为目前计算过的最大组数*8+算法，先前的组数如不足，则应重新选择算法
void MatrixExtend::convolutionForward(const Matrix& A, const Matrix& W, Matrix& R, Matrix& workspace,
    int& method, const std::vector<int>& stride, const std::vector<int>& padding, realc a, realc r)
{
    assert(checkMatrixDevice({ &A, &W, &R }));
    assert(stride.size() >= 2 && padding.size() >= 2);
    auto cuda = A.cuda();
    if (R.inGPU())
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
        if (method < A.getNumber() * conv_method_count * conv_method_count)
        {
            int n;
            cudnnConvolutionFwdAlgoPerf_t cfap[conv_method_count];
            auto t = cudnnFindConvolutionForwardAlgorithm(cuda->cudnn_handle_, A.getCudnnTensorDesc(),
                cuda->filter_desc_, cuda->convolution_desc_, R.getCudnnTensorDesc(), conv_method_count, &n, cfap);
            if (t)
            {
                //cudnnDataType_t tt;
                //int n;
                //int nn[8];
                //int ss[8];
                //cudnnGetTensorNdDescriptor(A.getCudnnTensorDesc(), 8, &tt, &n, nn, ss);
                //cudnnGetTensorNdDescriptor(X.getCudnnTensorDesc(), 8, &tt, &n, nn, ss);
                fprintf(stderr, "Find convolution forward algorithm failed! %d, %d, %d\n", scd, sfd, t);
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
                method = cfap[c].algo + cfap[c].mathType * conv_method_count + A.getNumber() * conv_method_count * conv_method_count;
                //size_t memory = cfap[c].memory;
                if (memory > workspace.getDataSizeInByte())
                {
                    workspace.resize(1, 1, 1, memory / sizeof(real) + 2);
                    //fprintf(stdout, "resize work space %d\n", memory);
                }
                //fprintf(stdout, "choose %d,%d,%d\n", cfap[c].algo, cfap[c].memory, cfap[c].mathType);
            }
            //workspace->message();
        }
        auto cfa = cudnnConvolutionFwdAlgo_t(method % conv_method_count);
        //auto tensor = cudnnSetConvolutionMathType(cuda->convolution_desc_, cudnnMathType_t(method / conv_method_count % conv_method_count));
        auto scf = cudnnConvolutionForward(cuda->cudnn_handle_, &a, A.getCudnnTensorDesc(), A.data(), cuda->filter_desc_, W.data(),
            cuda->convolution_desc_, cfa, workspace.data(), workspace.getDataSizeInByte(), &r, R.getCudnnTensorDesc(), R.data());
        if (scf)
        {
            fprintf(stderr, "CONV forward error %d, %d, %d, %d\n", scd, sfd, scf, cfa);
        }
    }
    else
    {
        R.initData(0);
        //辅助矩阵的尺寸
        int row = R.width_ * R.height_;
        int col = A.channel_ * W.width_ * W.height_;
        Matrix A_ex(row, col, DeviceType::CPU);
        if (method < 0)
        {
            method = 0;
            workspace.resize(1, 1, row * col, 3);
            workspace.initData(-1);
            //ex_pos记录展开的位置，预先记录节省时间
            for (int cA = 0; cA < A.channel_; cA++)
            {
                CONV_OPERATION1(A, W, R, padding[0], padding[1], stride[0], stride[1],
                    {
                        //记录X_ex中每个位置对应X中的元素，为了效率将二者都拍扁
                        int pR = R.whcn2i(wR, hR, 0, 0);     //X_ex中对应的行，即在A中的位置
                        int pW = W.whcn2i(wW, hW, cA, 0);    //X_ex中对应的列，即在W中的位置
                        //拍扁
                        int pA = A.whcn2i(wA, hA, cA, 0);    //X其中一组特征对应的位置
                        int pA_ex = A_ex.mn2i(pA, pW);
                        workspace.getData(pA_ex, 0) = pA;    //记录展开的位置
                    });
            }
        }
        Matrix A_sub(A.row_, 1, DeviceType::CPU);
        Matrix R_sub(R.width_ * R.height_, R.channel_, DeviceType::CPU);    //todo这两个应该是专用于share, 这样写会导致内存多释放一次,待处理,下同
        for (int i = 0; i < A.number_; i++)
        {
            A_sub.shareData(A, 0, i);
            R_sub.shareData(R, 0, i);
            A_ex.initData(0);
            for (int j = 0; j < A_ex.getDataSize(); j++)
            {
                int p = workspace.getData(j, 0);
                if (p >= 0 && p < A_sub.getDataSize())
                {
                    A_ex.getData(j) = A_sub.getData(p);
                }
            }
            MatrixExtend::mul(A_ex, W, R_sub, a, r);
        }

#ifdef DIRECT_COMPUTE_CONVOLUTION
        //这是原始计算方法，速度很慢，不要打开这段代码
        //fprintf(stderr, "Please supply buffer vector and use the faster convolution method.\n");
        for (int n = 0; n < R.number_; n++)
        {
            for (int cX = 0; cX < A.channel_; cX++)
            {
                for (int cA = 0; cA < R.channel_; cA++)
                {
                    CONV_OPERATION1(A, W, R,
                        {
                            R.getData(wA, hA, cA, n) += A.getData(wX, hX, cX, n) * W.getData(wW, hW, cX, cA);
                        });
                }
            }
        }
#endif
    }
}

//计算dX只需要W，dA；计算dW只需要X，dA；计算dB只需要dA
void MatrixExtend::convolutionBackward(Matrix& A, Matrix& W, const Matrix& R, Matrix& workspace,
    int& method_dx, int& method_dw, const std::vector<int>& stride, const std::vector<int>& padding, realc a, realc r)
{
    assert(checkMatrixDevice({ &A, &W, &R }));
    assert(stride.size() >= 2 && padding.size() >= 2);
    auto cuda = R.cuda();
    //这里不用dX判断是因为dX可能是空
    if (R.inGPU())
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
        if (A.needReverse())
        {
            //寻找最快的算法
            if (method_dx < A.getNumber() * conv_method_count * conv_method_count)
            {
                int n;
                cudnnConvolutionBwdDataAlgoPerf_t cbdap[conv_method_count];
                auto t = cudnnFindConvolutionBackwardDataAlgorithm(cuda->cudnn_handle_, cuda->filter_desc_,
                    R.getCudnnTensorDesc(), cuda->convolution_desc_, A.getCudnnTensorDesc(), conv_method_count, &n, cbdap);
                int c = 0;
                size_t memory = 0;
                for (int i = 0; i < n; i++)
                {
                    if (cbdap[c].algo == cbdap[i].algo)
                    {
                        memory = std::max(memory, cbdap[i].memory);
                    }
                }
                method_dx = cbdap[c].algo + cbdap[c].mathType * conv_method_count + A.getNumber() * conv_method_count * conv_method_count;
                //size_t memory = cbdap[c].memory;
                if (memory > workspace.getDataSizeInByte())
                {
                    workspace.resize(1, 1, 1, memory / sizeof(real) + 2);
                    //fprintf(stdout, "resize work space %d\n", memory);
                }
                //fprintf(stdout, "choose dx %d,%d,%d\n", cbdap[c].algo, cbdap[c].memory, cbdap[c].mathType);
                //workspace->message();
            }
            auto cbda = cudnnConvolutionBwdDataAlgo_t(method_dx % conv_method_count);
            //auto tensor = cudnnSetConvolutionMathType(cuda->convolution_desc_, cudnnMathType_t(method_dx / conv_method_count % conv_method_count));
            auto scbx = cudnnConvolutionBackwardData(cuda->cudnn_handle_, &a, cuda->filter_desc_, W.data(), R.getCudnnTensorDesc(), R.DMatrix().data(),
                cuda->convolution_desc_, cbda, workspace.data(), workspace.getDataSizeInByte(), &r, A.getCudnnTensorDesc(), A.DMatrix().data());
            if (scbx)
            {
                fprintf(stderr, "CONV backward data error %d\n", scbx);
            }
        }
        if (W.needReverse())
        {
            //寻找最快的算法
            if (method_dw < A.getNumber() * conv_method_count * conv_method_count)
            {
                int n;
                cudnnConvolutionBwdFilterAlgoPerf_t cbfap[conv_method_count];
                cudnnFindConvolutionBackwardFilterAlgorithm(cuda->cudnn_handle_, A.getCudnnTensorDesc(), R.getCudnnTensorDesc(),
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
                method_dw = cbfap[c].algo + cbfap[c].mathType * conv_method_count + A.getNumber() * conv_method_count * conv_method_count;
                //size_t memory = cbfap[c].memory;
                if (memory > workspace.getDataSizeInByte())
                {
                    workspace.resize(1, 1, 1, memory / sizeof(real) + 2);
                    //fprintf(stdout, "resize work space %d\n", memory);
                }
                //fprintf(stdout, "choose dw %d,%d,%d\n", cbfap[c].algo, cbfap[c].memory, cbfap[c].mathType);
                //workspace->message();
            }
            auto cbfa = cudnnConvolutionBwdFilterAlgo_t(method_dw % conv_method_count);
            //auto tensor = cudnnSetConvolutionMathType(cuda->convolution_desc_, cudnnMathType_t(method_dw / conv_method_count % conv_method_count));
            auto scbw = cudnnConvolutionBackwardFilter(cuda->cudnn_handle_, &a, A.getCudnnTensorDesc(), A.data(), R.getCudnnTensorDesc(), R.DMatrix().data(),
                cuda->convolution_desc_, cbfa, workspace.data(), workspace.getDataSizeInByte(), &r, cuda->filter_desc_, W.DMatrix().data());
            if (scbw)
            {
                fprintf(stderr, "CONV backward weight error %d\n", scbw);
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
        if (A.needReverse())
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
            int row = A.width_ * A.height_;
            int col = R.channel_ * W.width_ * W.height_;
            Matrix dR_ex(row, col, DeviceType::CPU);
            dR_ex.initData(0);
            if (method_dx < 0)
            {
                method_dx = 0;
                //workspace->resize(1, 1, 3, row * col);
                //workspace->initData(-1);
                for (int cR = 0; cR < R.channel_; cR++)
                {
                    CONV_OPERATION1(A, W, R, padding[0], padding[1], stride[0], stride[1],
                        {
                            int pA = A.whcn2i(wA, hA, 0, 0);
                            //这里用W或者W2没有区别
                            int pW = W2.whcn2i(wW, hW, cR, 0);
                            //拍扁
                            int pR = R.whcn2i(wR, hR, cR, 0);
                            int p_ex = dR_ex.mn2i(pA, pW);
                            workspace.getData(p_ex, 1) = pR;
                        });
                }
            }
            Matrix dR_sub(R.row_, 1, DeviceType::CPU);
            Matrix dA_sub(A.width_ * A.height_, A.channel_, DeviceType::CPU);
            for (int i = 0; i < R.number_; i++)
            {
                dR_sub.shareData(R, 0, i);
                dA_sub.shareData(A, 0, i);
                for (int j = 0; j < dR_ex.getDataSize(); j++)
                {
                    if (workspace.getData(j, 1) >= 0)
                    {
                        dR_ex.DMatrix().getData(j) = dR_sub.DMatrix().getData(workspace.getData(j, 1));
                    }
                }
                MatrixExtend::mul(dR_ex, W2, dA_sub, a, r);
            }
        }
        //暂时如此写，看情况能否跟上面合并
        if (W.needReverse())
        {
            W.DMatrix().scale(r);
            //辅助矩阵的尺寸
            int row = W.width_ * W.height_ * W.channel_;
            int col = R.width_ * R.height_;
            Matrix A_ex(row, col, DeviceType::CPU);
            A_ex.initData(0);
            if (method_dw < 0)
            {
                method_dw = 0;
                //workspace->resize(1, 1, 3, row * col);
                //workspace->initData(-1);
                //cW==cX, nW=cA
                for (int cW = 0; cW < W.channel_; cW++)
                {
                    CONV_OPERATION1(A, W, R, padding[0], padding[1], stride[0], stride[1],
                        {
                            int pW = W.whcn2i(wW, hW, cW, 0);
                            int pR = R.whcn2i(wR, hR, 0, 0);
                            //拍扁
                            int pA = A.whcn2i(wA, hA, cW, 0);
                            int p_ex = A_ex.mn2i(pW, pR);
                            workspace.getData(p_ex, 2) = pA;
                        });
                }
            }
            Matrix dR_sub(R.width_ * R.height_, R.channel_, DeviceType::CPU);
            Matrix A_sub(A.row_, 1, DeviceType::CPU);
            for (int i = 0; i < R.number_; i++)
            {
                dR_sub.shareData(R, 0, i);
                A_sub.shareData(A, 0, i);
                for (int j = 0; j < A_ex.getDataSize(); j++)
                {
                    //if ((*ex2)[j] >= 0) //因为是满的不需要
                    A_ex.getData(j) = A_sub.getData(workspace.getData(j, 2));
                }
                MatrixExtend::mul(A_ex, dR_sub, W.DMatrix(), a, 1);    //有点麻烦，暂时不管
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
        //fprintf(stderr, "Please supply buffer vector and use the faster convolution method.\n");
        if (dW)
        {
            dW.scale(r);
        }
        if (dX)
        {
            dX.scale(r);
        }
        for (int n = 0; n < R.number_; n++)
        {
            for (int cX = 0; cX < A.channel_; cX++)
            {
                for (int cA = 0; cA < R.channel_; cA++)
                {
                    CONV_OPERATION1(A, W, R, wX, hX, wW, hW, wA, hA,
                        {
                            if (dX)
                            {
                                dX.getData(wX, hX, cX, n) += a * dA.getData(wA, hA, cA, n) * W.getData(wW, hW, cX, cA);
                            }
                            if (dW)
                            {
                                dW.getData(wW, hW, cX, cA) += a * A.getData(wX, hX, cX, n) * dA.getData(wA, hA, cA, n);
                            }
                        });
                }
            }
        }
#endif
    }
}

//随机让一些点不参与计算
void MatrixExtend::dropoutForward(const Matrix& A, Matrix& R, ActivePhaseType work_phase, real v, int seed, Matrix& rg_stat, Matrix& reverse_space)
{
    assert(checkMatrixDevice({ &A, &R }));
    if (work_phase == ACTIVE_PHASE_TEST)
    {
        Matrix::copyData(A, R);
        R.scale(v);
        return;
    }
    auto cuda = A.cuda();
    if (R.inGPU())
    {
        cudnnSetDropoutDescriptor(cuda->dropout_desc_, cuda->cudnn_handle_, v, rg_stat.data(), rg_stat.getDataSizeInByte(), seed);
        cudnnDropoutForward(cuda->cudnn_handle_, cuda->dropout_desc_, A.getCudnnTensorDesc(), A.data(),
            R.getCudnnTensorDesc(), R.data(), reverse_space.data(), reverse_space.getDataSizeInByte());
    }
    else
    {
        Random<real> r;
        r.set_seed(seed);
        for (int i = 0; i < R.data_size_; i++)
        {
            if (r.rand() < v)
            {
                R.data()[i] = 0;
            }
            else
            {
                R.data()[i] = A.data()[i] / (1 - v);
            }
        }
    }
}

void MatrixExtend::dropoutBackward(Matrix& A, const Matrix& R, real v, int seed, Matrix& rg_stat, Matrix& reverse_space)
{
    assert(checkMatrixDevice({ &A, &R }));
    auto cuda = R.cuda();
    if (R.inGPU())
    {
        //这里的seed应该是没关系，待查
        cudnnSetDropoutDescriptor(cuda->dropout_desc_, cuda->cudnn_handle_, v, rg_stat.data(), rg_stat.getDataSizeInByte(), seed);
        cudnnDropoutBackward(cuda->cudnn_handle_, cuda->dropout_desc_, R.getCudnnTensorDesc(), R.DMatrix().data(),
            A.getCudnnTensorDesc(), A.DMatrix().data(), reverse_space.data(), reverse_space.getDataSizeInByte());
    }
    else
    {
        for (int i = 0; i < A.data_size_; i++)
        {
            if (R.data()[i] == 0)
            {
                A.DMatrix().data()[i] = 0;
            }
            else
            {
                A.DMatrix().data()[i] = R.DMatrix().data()[i] / (1 - v);
            }
        }
    }
}

//批归一化
void MatrixExtend::batchNormalizationForward(const Matrix& A, Matrix& R,
    ActivePhaseType work_phase, BatchNormalizationType bn_type, real& exp_aver_factor, real epsilon, Matrix& scale, Matrix& bias,
    Matrix& result_running_mean, Matrix& result_running_variance, Matrix& result_save_mean, Matrix& result_save_inv_variance)
{
    assert(checkMatrixDevice({ &A, &R }));
    auto cuda = A.cuda();
    if (R.inGPU())
    {
        if (work_phase == ACTIVE_PHASE_TRAIN)
        {
            cudnnBatchNormalizationForwardTraining(cuda->cudnn_handle_, cudnnBatchNormMode_t(bn_type),
                &const_real_1, &const_real_0, A.getCudnnTensorDesc(), A.data(), R.getCudnnTensorDesc(), R.data(),
                scale.getCudnnTensorDesc(), scale.data(), bias.data(), exp_aver_factor, result_running_mean.data(), result_running_variance.data(),
                std::max(epsilon * 1.0, CUDNN_BN_MIN_EPSILON), result_save_mean.data(), result_save_inv_variance.data());
            exp_aver_factor = 1 / (1 / exp_aver_factor + 1);
        }
        if (work_phase == ACTIVE_PHASE_TEST)
        {
            cudnnBatchNormalizationForwardInference(cuda->cudnn_handle_, cudnnBatchNormMode_t(bn_type),
                &const_real_1, &const_real_0, A.getCudnnTensorDesc(), A.data(), R.getCudnnTensorDesc(), R.data(),
                scale.getCudnnTensorDesc(), scale.data(), bias.data(), result_running_mean.data(), result_running_variance.data(),
                std::max(epsilon * 1.0, CUDNN_BN_MIN_EPSILON));
        }
    }
}

void MatrixExtend::batchNormalizationBackward(Matrix& A, const Matrix& R,
    ActivePhaseType work_phase, BatchNormalizationType bn_type, real epsilon, real rate, Matrix& scale, Matrix& bias,
    Matrix& saved_mean, Matrix& saved_inv_variance, Matrix& result_dscale, Matrix& result_dbias)
{
    assert(checkMatrixDevice({ &A, &R }));
    auto cuda = R.cuda();
    if (A.inGPU())
    {
        cudnnBatchNormalizationBackward(cuda->cudnn_handle_, cudnnBatchNormMode_t(bn_type),
            &const_real_1, &const_real_0, &const_real_1, &const_real_0, A.getCudnnTensorDesc(), A.data(), R.getCudnnTensorDesc(), R.DMatrix().data(), A.getCudnnTensorDesc(), A.DMatrix().data(),
            scale.getCudnnTensorDesc(), scale.data(), result_dscale.data(), result_dbias.data(),
            std::max(epsilon * 1.0, CUDNN_BN_MIN_EPSILON), saved_mean.data(), saved_inv_variance.data());
        MatrixExtend::add(scale, result_dscale, scale, 1, -rate);
        MatrixExtend::add(bias, result_dbias, bias, 1, -rate);
    }
}

//ada_d表示ada方法计算得到的参数的改变量，应在下一步加到原参数上，下同
void MatrixExtend::adaDeltaUpdate(Matrix& mean_d2, Matrix& mean_ada_d2, Matrix& d, Matrix& ada_d, real rou, real epsilon)
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

void MatrixExtend::adamUpdate(Matrix& mean_d, Matrix& mean_d2, Matrix& d, Matrix& ada_d, real beta1, real beta2, real epsilon, real t)
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

void MatrixExtend::adaRMSPropUpdate(Matrix& mean_d2, Matrix& d, Matrix& ada_d, real rou, real epsilon)
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
void MatrixExtend::sparse(Matrix& rou_hat, Matrix& R, real rou, real beta)
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

void MatrixExtend::fill(Matrix& m, RandomFillType random_type, int in, int out)
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
    default:
        break;
    }
    std::vector<real> temp(m.getDataSize());
    random_generator.rand_data(temp.data(), temp.size());
    m.initData(temp.data(), temp.size());
}

void MatrixExtend::sin(const Matrix& A, Matrix& R, real a)
{
    assert(checkMatrixDevice({ &A, &R }));
    if (A.inGPU())
    {
        cuda_sin(A.data(), R.data(), A.getDataSize(), a, 0);
    }
    else
    {
        for (int i = 0; i < R.data_size_; i++)
        {
            R.getData(i) = ::sin(a * A.getData(i));
        }
    }
}

void MatrixExtend::cos(const Matrix& A, Matrix& R, real a)
{
    assert(checkMatrixDevice({ &A, &R }));
    if (A.inGPU())
    {
        cuda_cos(A.data(), R.data(), A.getDataSize(), a, 0);
    }
    else
    {
        for (int i = 0; i < R.data_size_; i++)
        {
            R.getData(i) = ::cos(a * A.getData(i));
        }
    }
}

void MatrixExtend::zigzag(const Matrix& A, Matrix& R)
{
    assert(checkMatrixDevice({ &A, &R }));
    if (A.inGPU())
    {
        cuda_zigzag(A.data(), R.data(), A.getDataSize(), 1, 0);
    }
    else
    {
        for (int i = 0; i < R.data_size_; i++)
        {
            auto& x = A.getData(i);
            R.getData(i) = x - 2 * floor((x - 1) / 2) - 2;
        }
    }
}

//实际上这个激活函数在奇异点不连续，无法训练
void MatrixExtend::zigzagb(Matrix& A, const Matrix& R)
{
    assert(checkMatrixDevice({ &A, &R }));
    if (A.inGPU())
    {
        cuda_zigzagb(R.data(), R.DMatrix().data(), A.DMatrix().data(), A.getDataSize(), 1, 0);
    }
    else
    {
        auto p1 = R.data();
        auto p2 = R.DMatrix().data();
        auto p3 = A.DMatrix().data();
        for (int i = 0; i < R.data_size_; i++)
        {
            if (abs(p1[i]) > 1 - 1e-2)
            {
                p3[i] = -p2[i] * 100;
                continue;
            }
            p3[i] = p2[i];
        }
    }
}

void MatrixExtend::step(const Matrix& A, Matrix& R)
{
    assert(checkMatrixDevice({ &A, &R }));
    if (A.inGPU())
    {
        cuda_step(A.data(), R.data(), A.getDataSize(), 0, 0);
    }
    else
    {
        //未完成
    }
}

}    // namespace woco