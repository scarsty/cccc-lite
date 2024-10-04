#include "MatrixOp.h"
#include "MatrixEx.h"
#include "strfunc.h"
#include <functional>

#ifdef FMT1_USE_STD_FORMAT
template <>
struct std::formatter<cccc::MatrixSP>
{
    constexpr auto parse(std::format_parse_context& context)
    {
        return context.begin();
    }

    auto format(const cccc::MatrixSP& v, std::format_context& format_context) const
    {
        return std::format_to(format_context.out(), "M{}", (uint64_t)v.get());
    }
};
#endif

namespace cccc
{
#ifndef FMT1_USE_STD_FORMAT
inline std::string to_string(const std::string& fmt, const cccc::MatrixSP& m)
{
    return fmt1::sprintf2(fmt, "M%p", m.get());
}
#endif

void MatrixOp::forward(std::vector<MatrixOp>& op_queue)
{
    //Timer t;
    for (auto& op : op_queue)
    {
        op.forwardData();
        //op.in_[0]->message("in");
        //if (op.wb_.size() > 0) { op.wb_[0]->message("w"); }
        //op.out_[0]->message("out");
    }
    //LOG("forward time: {} s\n", t.getElapsedTime());
}

void MatrixOp::backward(std::vector<MatrixOp>& op_queue, std::vector<MatrixOp>& loss, bool clear_d)
{
    if (clear_d)
    {
        for (auto& op : op_queue)
        {
            for (auto& m : op.in_)
            {
                m->setKeepWeight(0);
            }
        }
        for (auto& op : loss)
        {
            for (auto& m : op.in_)
            {
                m->setKeepWeight(0);
            }
        }
    }
    for (auto& op : loss)
    {
        op.backwardLoss();
    }
    for (auto it = op_queue.rbegin(); it != op_queue.rend(); ++it)
    {
        //if (!it->wb_.empty()) { it->wb_[0]->message("W" + std::to_string(int(it->type_))); }
        it->backwardDataWeight();
        //it->out_[0]->d().message("dY" + std::to_string(int(it->type_)));
        //it->in_[0]->d().message("dX" + std::to_string(int(it->type_)));
        //if (!it->wb_.empty()) { it->wb_[0]->d().message("dW" + std::to_string(int(it->type_))); }
        //if (!it->wb_.empty()) { it->wb_[0]->message("W" + std::to_string(int(it->type_))); }
    }
}

void MatrixOp::forwardData()
{
    auto& X = *in_[0];
    auto& Y = *out_[0];
    switch (type_)
    {
    case MatrixOpType::ADD:
        Matrix::add(X, *in_[1], Y, a_[0], b_[0]);
        for (int i = 2; i < in_.size(); i++)
        {
            Matrix::add(Y, *in_[i], Y, 1, a_[i], 1);
        }
        break;
    case MatrixOpType::MUL:
        Matrix::mul(*wb_[0], X, Y, a_[0]);
        break;
    case MatrixOpType::ELE_MUL:
        Matrix::elementMul(X, *in_[1], Y, a_[0]);
        break;
    case MatrixOpType::ADD_BIAS:
        MatrixEx::addBias(X, *wb_[0], Y, a_[0], b_[0]);
        break;
    case MatrixOpType::CONCAT:
        MatrixEx::concatByChannel(in_, Y);
        break;
    case MatrixOpType::ACTIVE:
        MatrixEx::activeForward(X, Y, anys_[0].to<ActiveFunctionType>(), anys_[1].to<std::vector<int>>(), anys_[2].to<std::vector<float>>(), workspace_);
        break;
    case MatrixOpType::POOL:
        MatrixEx::poolingForward(X, Y, anys_[0].to<PoolingType>(), anys_[1].to<PoolingReverseType>(), window_, stride_, padding_, a_[0], b_[0], &workspace_[0]);
        break;
    case MatrixOpType::CONV:
        MatrixEx::convolutionForward(X, *wb_[0], Y, stride_, padding_, a_[0], b_[0], &anys_[0].to<MatrixEx::ConvMethod>(), &workspace_[0]);
        break;
    case MatrixOpType::CORR:
        //MatrixEx::correlationForward(X, *wb_[0], Y, para_int_, para_matrix_, para_int_v_[0], para_int_v_[1], para_real_[0]);
        break;
    case MatrixOpType::MAX:
        MatrixEx::matrix_max(*in_[0], *in_[1], Y);
        break;
    }
}

void MatrixOp::backwardDataWeight()
{
    auto& Y = *out_[0];
    float data_weight = 0;
    //若反向过程需更新多个矩阵，则在函数内部判断needUpdate
    //if (Y.isHip()) { data_weight = 0; }    //miopen只支持反向时beta为0
    switch (type_)
    {
    case MatrixOpType::ADD:
        for (int i = 0; i < in_.size(); i++)
        {
            if (in_[i]->needBack())
            {
                Matrix::add(in_[i]->d(), Y.d(), in_[i]->d(), in_[i]->keepWeight(), b_[i]);
            }
        }
        break;
    case MatrixOpType::MUL:
        if (in_[0]->needBack())
        {
            Matrix::mul(*wb_[0], Y.d(), in_[0]->d(), a_[0], in_[0]->keepWeight(), MATRIX_TRANS, MATRIX_NO_TRANS);
        }
        if (wb_[0]->needBack())
        {
            Matrix::mul(Y.d(), *in_[0], wb_[0]->d(), a_[0], data_weight, MATRIX_NO_TRANS, MATRIX_TRANS);
        }
        break;
    case MatrixOpType::ELE_MUL:
        if (in_[0]->needBack())
        {
            Matrix::elementMul(Y.d(), *in_[1], in_[0]->d(), a_[0], in_[0]->keepWeight());
        }
        if (in_[1]->needBack())
        {
            Matrix::elementMul(Y.d(), *in_[0], in_[1]->d(), a_[0], in_[1]->keepWeight());
        }
        break;
    case MatrixOpType::ADD_BIAS:
        if (in_[0]->needBack())
        {
            Matrix::add(in_[0]->d(), Y.d(), in_[0]->d(), 0, 1);
        }
        if (wb_[0]->needBack())
        {
            MatrixEx::addBiasBackward(*in_[0], *wb_[0], Y, 1, data_weight);
        }
        break;
    case MatrixOpType::CONCAT:
        MatrixEx::concatByChannelBackward(in_, Y);
        break;
    case MatrixOpType::ACTIVE:
        if (in_[0]->needBack())
        {
            MatrixEx::activeBackward(*in_[0], Y, anys_[0].to<ActiveFunctionType>(), anys_[1].to<std::vector<int>>(), anys_[2].to<std::vector<float>>(), workspace_, 1, in_[0]->keepWeight());
        }
        break;
    case MatrixOpType::POOL:
        if (in_[0]->needBack())
        {
            MatrixEx::poolingBackward(*in_[0], Y, anys_[0].to<PoolingType>(), anys_[1].to<PoolingReverseType>(), window_, stride_, padding_, a_[0], in_[0]->keepWeight(), &workspace_[0]);
        }
        break;
    case MatrixOpType::CONV:
        if (in_[0]->needBack())
        {
            MatrixEx::convolutionBackwardDX(*in_[0], *wb_[0], Y, stride_, padding_, a_[0], in_[0]->keepWeight(), &anys_[1].to<MatrixEx::ConvMethod>(), &workspace_[1]);
        }
        if (wb_[0]->needBack())
        {
            MatrixEx::convolutionBackwardDW(*in_[0], *wb_[0], Y, stride_, padding_, a_[0], data_weight, &anys_[2].to<MatrixEx::ConvMethod>(), &workspace_[2]);
        }
        //MatrixEx::convolutionBackward(*in_[0], *wb_[0], Y, stride_, padding_, a_[0], in_[0]->keepWeight(), a[0], data_weight, &anys_[1].to<MatrixEx::ConvMethod>(), &anys_[2].to<MatrixEx::ConvMethod>(), &workspace_[1], &workspace_[2]);
        break;
    case MatrixOpType::CORR:
        //MatrixEx::correlationBackward(*in_[0], *wb_[0], Y, para_int_, para_matrix_, para_int_v_[0], para_int_v_[1], para_real_[0], in_[0]->keepWeight(), data_weight);
        break;
    case MatrixOpType::RESHAPE:
        Matrix::copyData(Y.d(), in_[0]->d());
        break;
    case MatrixOpType::MAX:
        MatrixEx::matrix_maxb(*in_[0], *in_[1], Y, in_[0]->keepWeight(), in_[1]->keepWeight(), 1);
    }
    for (auto& m : in_)
    {
        m->setKeepWeight(1);
    }
}

void MatrixOp::backwardLoss()
{
    //若反向过程需更新多个矩阵，则在函数内部判断needUpdate
    switch (type_)
    {
    case MatrixOpType::LOSS:
        if (scale_ != 0)
        {
            if (in_.size() >= 3 && in_[2]->getDataSize() == in_[0]->getDataSize())
            {
                //有损失权重的情况
                //注意这里使用in_[2]->d()作为了中间变量
                Matrix::add(*in_[0], *in_[1], in_[2]->d(), scale_, -scale_, 0);
                Matrix::elementMul(in_[2]->d(), *in_[2], in_[0]->d(), 1, in_[0]->keepWeight());
            }
            else
            {
                //没有损失权重的情况，也是一般的情况
                //此处直接相减，表示欧氏距离平方，若配合前一层的softmax_ce或sigmoid_ce则表示交叉熵
                Matrix::add(*in_[0], *in_[1], in_[0]->d(), scale_, -scale_, in_[0]->keepWeight());
            }
        }
        break;
    case MatrixOpType::L2:
        if (scale_ != 0)
        {
            //Matrix::add(X->d(), X, X->d(), data_weight_, scale_);
        }
        break;
    }
    in_[0]->setKeepWeight(1);
}

std::string MatrixOp::ir(const std::vector<MatrixOp>& op_queue)
{
    std::string content;
    content += fmt1::format("{} = {};", op_queue.front().in_[0], op_queue.front().in_[0]->sizeMessage(0));
    for (const auto& op : op_queue)
    {
        content += op.print();
    }
    content += fmt1::format("setXY({}, {});", op_queue.front().in_[0], op_queue.back().out_[0]);
    return content;
}

std::string MatrixOp::print() const
{
    std::string line;
    switch (type_)
    {
    case MatrixOpType::ADD:
        line = fmt1::format("{} = add({}, {});", out_[0], in_[0], in_[1]);
        break;
    case MatrixOpType::MUL:
        line = fmt1::format("{} = mul({}, {}, [{}, {}, {}, batch]);", out_[0], wb_[0], in_[0],
            out_[0]->getWidth(), out_[0]->getHeight(), out_[0]->getChannel());
        break;
    case MatrixOpType::ELE_MUL:
        break;
    case MatrixOpType::ADD_BIAS:
        line = fmt1::format("{} = addBias({}, {});", out_[0], in_[0], wb_[0]);
        break;
    case MatrixOpType::CONCAT:
        line = fmt1::format("{} = concat({});", out_[0], in_);
        break;
    case MatrixOpType::ACTIVE:
        line = fmt1::format("{} = active({}, {}, {}, {});", out_[0], in_[0], int(anys_[0].to<ActiveFunctionType>()), anys_[1].to<std::vector<int>>(), anys_[2].to<std::vector<float>>());
        break;
    case MatrixOpType::POOL:
        if (anys_[1].to<PoolingReverseType>() == POOLING_NOT_REVERSE)
        {
            line = fmt1::format("{} = pool({}, {}, {}, {}, {});", out_[0], in_[0], int(anys_[0].to<PoolingType>()), window_, stride_, padding_);
        }
        else
        {
            line = fmt1::format("{} = reversepool({}, {}, {}, {});", out_[0], in_[0], window_, stride_, padding_);
        }
        break;
    case MatrixOpType::CONV:
        line = fmt1::format("{} = conv({}, {}, {}, {});", out_[0], in_[0], wb_[0], stride_, padding_);
        break;
    case MatrixOpType::CORR:
        line = fmt1::format("{} = corr({}, {}, {}, {});", out_[0], in_[0], wb_[0], stride_, padding_);
        break;
    case MatrixOpType::MAX:
        line = fmt1::format("{} = max({}, {});", out_[0], in_[0], in_[1]);
        break;
    }
    strfunc::replaceAllSubStringRef(line, "[", "{");
    strfunc::replaceAllSubStringRef(line, "]", "}");
    line += fmt1::format("/*out {}*/;", out_[0]->sizeMessage());
    return line;
}

void MatrixOp::simpleQueue(std::vector<MatrixOp>& op_queue, Matrix& X, Matrix& A)
{
    std::vector<int> connect_X(op_queue.size(), 0), connect_A(op_queue.size(), 0);    //1-有连接

    std::function<void(Matrix&, int, std::vector<int>&)> check_connect = [&op_queue, &check_connect](Matrix& M, int direct, std::vector<int>& connect)
    {
        for (int i = 0; i < op_queue.size(); i++)
        {
            if (connect[i] != 0)
            {
                continue;
            }
            auto& op = op_queue[i];
            std::vector<MatrixSP>*v1, *v2;
            if (direct > 0)
            {
                v1 = &op.in_;
                v2 = &op.out_;
            }
            else
            {
                v1 = &op.out_;
                v2 = &op.in_;
            }
            for (auto& m : *v1)
            {
                if (m->getDataPtr() == M.getDataPtr())
                {
                    connect[i]++;
                    for (auto& m : *v2)
                    {
                        check_connect(*m, direct, connect);
                    }
                    break;
                }
            }
        }
    };

    check_connect(X, 1, connect_X);
    check_connect(A, -1, connect_A);

    int i = 0;
    for (auto it = op_queue.begin(); it != op_queue.end();)
    {
        if (connect_X[i] == 0 || connect_A[i] == 0)
        {
            it = op_queue.erase(it);
        }
        else
        {
            ++it;
        }
        i++;
    }
    //repair sigmoid or softmax at the last layer but no cross entropy
    if (op_queue.back().type_ == MatrixOpType::ACTIVE)
    {
        auto active_type = op_queue.back().anys_[0].to<ActiveFunctionType>();
        if (active_type == ACTIVE_FUNCTION_SIGMOID) { active_type = ACTIVE_FUNCTION_SIGMOID_CE; }
        if (active_type == ACTIVE_FUNCTION_SOFTMAX) { active_type = ACTIVE_FUNCTION_SOFTMAX_CE; }
        op_queue.back().anys_[0].to<ActiveFunctionType>() = active_type;
    }
}

void MatrixOp::getDefaultStridePadding(MatrixOpType type, const std::vector<int>& dim, std::vector<int>& stride, std::vector<int>& padding)
{
    if (type == MatrixOpType::CONV)
    {
        if (stride.size() == 0) { stride.resize(dim.size() - 2, 1); }
        if (padding.size() == 0) { padding.resize(dim.size() - 2, 0); }
    }
    if (type == MatrixOpType::POOL)
    {
        if (stride.size() == 0) { stride = dim; }
        if (padding.size() == 0) { padding.resize(dim.size(), 0); }
    }
}

void MatrixOp::as_scale(const MatrixSP& X, const MatrixSP& Y, float r)
{
    Y->resize(X->getDim());
    set(MatrixOpType::MUL, { X }, {}, { Y }, {}, { r });
}

void MatrixOp::as_mul(const MatrixSP& X1, const MatrixSP& X2, const MatrixSP& Y, float a, std::vector<int> dim)
{
    //此处可强制reshape，返回可以直接卷积的维度
    if (dim.empty())
    {
        dim = X1->getDim();
        dim.back() = X2->getNumber();
    }
    Y->resize(dim);
    set(MatrixOpType::MUL, { X2 }, { X1 }, { Y }, { a });    //这里注意顺序
    if (X1->getNumber() != X2->getRow())
    {
        LOG_ERR("Error: cannot product!\n");
    }
}

void MatrixOp::as_elementMul(const MatrixSP& X1, const MatrixSP& X2, const MatrixSP& Y, float a)
{
    Y->resize(X1->getDim());
    set(MatrixOpType::ELE_MUL, { X1, X2 }, {}, { Y }, { a });
}

void MatrixOp::as_add(const MatrixSP& X1, const MatrixSP& X2, const MatrixSP& Y, float a, float b)
{
    Y->resize(X1->getDim());
    set(MatrixOpType::ADD, { X1, X2 }, {}, { Y }, { a }, { b });
}

void MatrixOp::as_add(const std::vector<MatrixSP>& X_vector, const MatrixSP& Y)
{
    set(MatrixOpType::ADD, X_vector, {}, { Y }, {}, std::vector<float>(X_vector.size(), 1.0));
}

void MatrixOp::as_addBias(const MatrixSP& X, const MatrixSP& bias, const MatrixSP& Y, float a, float b)
{
    //Matrix as_1(A.getNumber(), 1);
    //as_1.fillData(1);
    //需要注意cudnn自带的只支持到5维，若需更多维可以在这里修改写入op_queue的矩阵的维度
    Y->shareData(*X);    //需注意偏移操作是特殊处理的
    Y->resize(X->getDim());
    set(MatrixOpType::ADD_BIAS, { X }, { bias }, { Y }, { a }, { b });
}

void MatrixOp::as_concat(const std::vector<MatrixSP>& X_vector, const MatrixSP& Y)
{
    if (X_vector.size() > 0)
    {
        int sum_channel = 0;
        for (auto& X : X_vector)
        {
            sum_channel += X->getChannel();
        }
        auto dim = X_vector[0]->getDim();
        dim[dim.size() - 2] = sum_channel;
        Y->resize(dim);
    }
    set(MatrixOpType::CONCAT, X_vector, {}, { Y });
}

void MatrixOp::as_active(const MatrixSP& X, const MatrixSP& Y, ActiveFunctionType af)
{
    Y->resize(X->getDim());
    set(MatrixOpType::ACTIVE, { X }, {}, { Y }, {}, {}, { af, std::vector<int>(), std::vector<float>() }, {});
}

void MatrixOp::as_active(const MatrixSP& X, const MatrixSP& Y, ActiveFunctionType af, std::vector<int>&& int_vector, std::vector<float>&& real_vector, std::vector<Matrix>&& matrix_vector)
{
    Y->resize(X->getDim());
    MatrixEx::activeBufferInit(*X, af, int_vector, real_vector, matrix_vector);
    set(MatrixOpType::ACTIVE, { X }, {}, { Y }, {}, {}, { af, int_vector, real_vector }, std::move(matrix_vector));
}

void MatrixOp::as_pool(const MatrixSP& X, const MatrixSP& Y, PoolingType pooling_type, PoolingReverseType reverse_type, std::vector<int> window, std::vector<int> stride, std::vector<int> padding, float a)
{
    auto dim = X->getDim();
    getDefaultStridePadding(MatrixOpType::POOL, window, stride, padding);
    if (reverse_type == POOLING_NOT_REVERSE)
    {
        for (int i = 0; i < dim.size() - 2; i++)
        {
            dim[i] = (dim[i] + 2 * padding[i] - window[i]) / stride[i] + 1;
        }
    }
    else
    {
        pooling_type = POOLING_AVERAGE_NOPADDING;
        for (int i = 0; i < dim.size() - 2; i++)
        {
            dim[i] = stride[i] * (dim[i] - 1) + window[i] - 2 * padding[i];
        }
    }
    VectorAny pv = { pooling_type, reverse_type, int(window.size()) };
    Y->resize(dim);
    set(MatrixOpType::POOL, { X }, {}, { Y }, { a }, {}, std::move(pv), { Matrix() }, std::move(window), std::move(stride), std::move(padding));
}

void MatrixOp::as_conv(const MatrixSP& X, const MatrixSP& W, const MatrixSP& Y, std::vector<int> stride, std::vector<int> padding, int conv_algo, float a /*= 1*/)
{
    auto dim = X->getDim();
    getDefaultStridePadding(MatrixOpType::CONV, W->getDim(), stride, padding);
    std::vector<int> v(9);
    for (int i = 0; i < dim.size() - 2; i++)
    {
        dim[i] = (dim[i] + 2 * padding[i] - W->getDim()[i]) / stride[i] + 1;
    }
    dim[dim.size() - 2] = W->getDim().back();
    Y->resize(dim);
    auto t = MatrixOpType::CONV;
    MatrixEx::ConvMethod method;
    if (conv_algo >= 0)
    {
        //仅推导时固定算法
        method.algo = conv_algo;
        method.math_type = 0;
    }
    //在卷积计算开始之前，会查找最快的算法，v中依次保存了前向、反向数据、反向权重的算法编号、张量参数（mathtype）、组数
    //组数不同会导致工作空间尺寸不同，实际上此处只三个工作空间只用了一个，另两个未使用
    set(t, { X }, { W }, { Y }, { a }, {}, { method, MatrixEx::ConvMethod(), MatrixEx::ConvMethod() }, { Matrix(), Matrix(), Matrix() }, {}, std::move(stride), std::move(padding));
}

void MatrixOp::as_reshape(const MatrixSP& X, const MatrixSP& Y, std::vector<int>& dim)
{
    Y->shareData(*X);
    Y->resize(dim);
    set(MatrixOpType::RESHAPE, { X }, {}, { Y });
}

void MatrixOp::as_max(const MatrixSP& X1, const MatrixSP& X2, const MatrixSP& Y)
{
    Y->resize(X1->getDim());
    set(MatrixOpType::MAX, { X1, X2 }, {}, { Y });
}

std::vector<MatrixOp> operator+(const std::vector<MatrixOp>& A, const std::vector<MatrixOp>& B)
{
    auto R = A;
    R.insert(R.end(), B.begin(), B.end());
    return R;
}

std::vector<MatrixOp>& operator+=(std::vector<MatrixOp>& A, const std::vector<MatrixOp>& B)
{
    A.insert(A.end(), B.begin(), B.end());
    return A;
}

std::vector<MatrixOp> operator*(const std::vector<MatrixOp>& A, double v)
{
    auto R = A;
    for (auto& R1 : R)
    {
        R1.scale_ *= v;
    }
    return R;
}

std::vector<MatrixOp> operator*(double v, const std::vector<MatrixOp>& A)
{
    auto R = A;
    for (auto& R1 : R)
    {
        R1.scale_ *= v;
    }
    return R;
}

std::vector<MatrixOp> crossEntropy(const MatrixSP& A, const MatrixSP& Y)
{
    MatrixOp op;
    op.set(MatrixOpType::LOSS, { A, Y }, {}, {});
    return { op };
}

std::vector<MatrixOp> L2(const MatrixSP& A)
{
    MatrixOp op;
    op.set(MatrixOpType::L2, { A }, {}, {});
    return { op };
}

std::vector<MatrixOp> L2(const std::vector<MatrixSP>& v)
{
    std::vector<MatrixOp> q;
    for (auto& m : v)
    {
        MatrixOp op;
        op.set(MatrixOpType::L2, { m }, {}, {});
        q.push_back(op);
    }
    return q;
}

}    // namespace cccc