#include "MatrixOp.h"
#include "Timer.h"
#include "VectorMath.h"
#include "strfunc.h"
#include <functional>
#include <map>

namespace cccc
{
inline std::string to_string(const std::string& fmt, const cccc::MatrixSP& m)
{
    return fmt1::sprintf2(fmt, "M%p", m.get());
}

void MatrixOp::forward(std::vector<MatrixOp>& op_queue)
{
    //Timer t;
    for (auto& op : op_queue)
    {
        //op.in_[0]->message("in");
        op.forwardData();
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
        it->backwardDataWeight();
        //it->out_[0]->d().message("dY" + std::to_string(int(it->type_)));
        //it->in_[0]->d().message("dX" + std::to_string(int(it->type_)));
        //if (!it->wb_.empty())
        //{
        //    it->wb_[0]->d().message("dW" + std::to_string(int(it->type_)));
        //}
    }
}

void MatrixOp::forwardData()
{
    auto& X = *in_[0];
    auto& Y = *out_[0];
    switch (type_)
    {
    case MatrixOpType::ADD:
        Matrix::add(X, *in_[1], Y, para_real_[0], para_real_[1]);
        for (int i = 2; i < in_.size(); i++)
        {
            Matrix::add(Y, *in_[i], Y, 1, para_real_[i], 1);
        }
        break;
    case MatrixOpType::MUL:
        Matrix::mul(*wb_[0], X, Y, para_real_[0]);
        break;
    case MatrixOpType::ELE_MUL:
        Matrix::elementMul(X, *in_[1], Y, para_real_[0]);
        break;
    case MatrixOpType::ADD_BIAS:
        MatrixEx::addBias(X, *wb_[0], Y, para_real_[0], para_real_[1]);
        break;
    case MatrixOpType::CONCAT:
        MatrixEx::concatByChannel(in_, Y);
        break;
    case MatrixOpType::ACTIVE:
        MatrixEx::activeForward(X, Y, ActiveFunctionType(para_int_.back()), para_int_, para_real_, para_matrix_);
        break;
    case MatrixOpType::POOL:
        MatrixEx::poolingForward(X, Y, PoolingType(para_int_[1]), para_int_[2], para_int_v_[0], para_int_v_[1], para_int_v_[2], para_real_[0]);
        break;
    case MatrixOpType::CONV:
        MatrixEx::convolutionForward(X, *wb_[0], Y, para_int_, para_matrix_, para_int_v_[0], para_int_v_[1], para_real_[0]);
        break;
    case MatrixOpType::CORR:
        MatrixEx::correlationForward(X, *wb_[0], Y, para_int_, para_matrix_, para_int_v_[0], para_int_v_[1], para_real_[0]);
        break;
    case MatrixOpType::MAX:
        MatrixEx::matrix_max(*in_[0], *in_[1], Y);
        break;
    }
}

void MatrixOp::backwardDataWeight()
{
    auto& Y = *out_[0];
    real data_weight = 0;
    //若反向过程需更新多个矩阵，则在函数内部判断needUpdate
    switch (type_)
    {
    case MatrixOpType::ADD:
        for (int i = 0; i < in_.size(); i++)
        {
            if (in_[i]->needReverse())
            {
                Matrix::add(in_[i]->d(), Y.d(), in_[i]->d(), in_[i]->keepWeight(), para_real_[i]);
            }
        }
        break;
    case MatrixOpType::MUL:
        if (in_[0]->needReverse())
        {
            Matrix::mul(*wb_[0], Y.d(), in_[0]->d(), para_real_[0], in_[0]->keepWeight(), MATRIX_TRANS, MATRIX_NO_TRANS);
        }
        if (wb_[0]->needReverse())
        {
            Matrix::mul(Y.d(), *in_[0], wb_[0]->d(), para_real_[0], data_weight, MATRIX_NO_TRANS, MATRIX_TRANS);
        }
        break;
    case MatrixOpType::ELE_MUL:
        if (in_[0]->needReverse())
        {
            Matrix::elementMul(Y.d(), *in_[1], in_[0]->d(), para_real_[0], in_[0]->keepWeight());
        }
        if (in_[1]->needReverse())
        {
            Matrix::elementMul(Y.d(), *in_[0], in_[1]->d(), para_real_[0], in_[1]->keepWeight());
        }
        break;
    case MatrixOpType::ADD_BIAS:
        MatrixEx::addBiasBackward(*in_[0], *wb_[0], Y, 0, data_weight);
        break;
    case MatrixOpType::CONCAT:
        MatrixEx::concatByChannelBackward(in_, Y);
        break;
    case MatrixOpType::ACTIVE:
        if (in_[0]->needReverse())
        {
            MatrixEx::activeBackward(*in_[0], Y, ActiveFunctionType(para_int_.back()), para_int_, para_real_, para_matrix_, 1, in_[0]->keepWeight());
        }
        break;
    case MatrixOpType::POOL:
        if (in_[0]->needReverse())
        {
            MatrixEx::poolingBackward(*in_[0], Y, PoolingType(para_int_[1]), para_int_[2], para_int_v_[0], para_int_v_[1], para_int_v_[2], para_real_[0], in_[0]->keepWeight());
        }
        break;
    case MatrixOpType::CONV:
        MatrixEx::convolutionBackward(*in_[0], *wb_[0], Y, para_int_, para_matrix_, para_int_v_[0], para_int_v_[1], para_real_[0], in_[0]->keepWeight(), data_weight);
        break;
    case MatrixOpType::CORR:
        MatrixEx::correlationBackward(*in_[0], *wb_[0], Y, para_int_, para_matrix_, para_int_v_[0], para_int_v_[1], para_real_[0], in_[0]->keepWeight(), data_weight);
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
        line = fmt1::format("{} = active({}, {}, {}, {});", out_[0], in_[0], para_int_.back(), para_int_v_);
        break;
    case MatrixOpType::POOL:
        if (para_int_[2] == 0)
        {
            line = fmt1::format("{} = pool({}, {}, {}, {}, {});", out_[0], in_[0], para_int_[1], para_int_v_[0], para_int_v_[1], para_int_v_[2]);
        }
        else
        {
            line = fmt1::format("{} = reversepool({}, {}, {}, {});", out_[0], in_[0], para_int_v_[0], para_int_v_[1], para_int_v_[2]);
        }
        break;
    case MatrixOpType::CONV:
        line = fmt1::format("{} = conv({}, {}, {}, {});", out_[0], in_[0], wb_[0], para_int_v_[0], para_int_v_[1]);
        break;
    case MatrixOpType::CORR:
        line = fmt1::format("{} = corr({}, {}, {}, {});", out_[0], in_[0], wb_[0], para_int_v_[0], para_int_v_[1]);
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

ActiveFunctionType MatrixOp::getActiveType() const
{
    if (type_ == MatrixOpType::ACTIVE)
    {
        return (ActiveFunctionType)para_int_.back();
    }
    else
    {
        return ACTIVE_FUNCTION_NONE;
    }
}

int MatrixOp::setActiveType(ActiveFunctionType af)
{
    if (type_ == MatrixOpType::ACTIVE)
    {
        para_int_.back() = (int)af;
        return 0;
    }
    else
    {
        return -1;
    }
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
        auto active_type = ActiveFunctionType(op_queue.back().para_int_.back());
        if (active_type == ACTIVE_FUNCTION_SIGMOID) { active_type = ACTIVE_FUNCTION_SIGMOID_CE; }
        if (active_type == ACTIVE_FUNCTION_SOFTMAX) { active_type = ACTIVE_FUNCTION_SOFTMAX_CE; }
        op_queue.back().para_int_.back() = int(active_type);
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

void MatrixOp::as_scale(MatrixSP& X, MatrixSP& Y, real r)
{
    Y->resize(*X);
    set(MatrixOpType::MUL, { X }, {}, { Y }, {}, { r });
}

void MatrixOp::as_mul(MatrixSP& X1, MatrixSP& X2, MatrixSP& Y, real a, std::vector<int> dim)
{
    //此处可强制reshape，返回可以直接卷积的维度
    if (dim.empty())
    {
        dim = X1->getDim();
        dim.back() = X2->getNumber();
    }
    Y->resize(dim);
    set(MatrixOpType::MUL, { X2 }, { X1 }, { Y }, {}, { a });    //这里注意顺序
    if (X1->getNumber() != X2->getRow())
    {
        LOG(stderr, "Error: cannot product!\n");
    }
}

void MatrixOp::as_elementMul(MatrixSP& X1, MatrixSP& X2, MatrixSP& Y, real a)
{
    Y->resize(*X1);
    set(MatrixOpType::ELE_MUL, { X1, X2 }, {}, { Y }, {}, { a });
}

void MatrixOp::as_add(MatrixSP& X1, MatrixSP& X2, MatrixSP& Y, realc a, realc b)
{
    Y->resize(*X1);
    set(MatrixOpType::ADD, { X1, X2 }, {}, { Y }, {}, { a, b });
}

void MatrixOp::as_add(std::vector<MatrixSP>& X_vector, MatrixSP& Y)
{
    set(MatrixOpType::ADD, X_vector, {}, { Y }, {}, std::vector<real>(X_vector.size(), 1.0));
}

void MatrixOp::as_addBias(MatrixSP& X, MatrixSP& bias, MatrixSP& Y, realc a, realc b)
{
    //Matrix as_1(A.getNumber(), 1);
    //as_1.fillData(1);
    //需要注意cudnn自带的只支持到5维，若需更多维可以在这里修改写入op_queue的矩阵的维度
    Y->shareData(*X);    //需注意偏移操作是特殊处理的
    Y->resize(*X);
    set(MatrixOpType::ADD_BIAS, { X }, { bias }, { Y }, {}, { a, b });
}

void MatrixOp::as_concat(std::vector<MatrixSP>& X_vector, MatrixSP& Y)
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

void MatrixOp::as_active(MatrixSP& X, MatrixSP& Y, ActiveFunctionType af)
{
    std::vector<int> int_vector;
    std::vector<real> real_vector;
    std::vector<Matrix> matrix_vector;
    //todo
    //MatrixExtend::activeBufferInit(X, af, int_vector, matrix_vector);
    int_vector.push_back(af);
    if (af != 1)
    {
        af = af;
    }
    Y->resize(*X);
    set(MatrixOpType::ACTIVE, { X }, {}, { Y }, int_vector, real_vector, matrix_vector);
}

void MatrixOp::as_active(MatrixSP& X, MatrixSP& Y, ActiveFunctionType af, std::vector<int>&& int_vector, std::vector<real>&& real_vector, std::vector<Matrix>&& matrix_vector)
{
    auto v = int_vector;
    //MatrixExtend::activeBufferInit(*X, af, int_vector, matrix_vector);
    v.push_back(af);
    Y->resize(*X);
    MatrixEx::activeBufferInit(*X, af, int_vector, real_vector, matrix_vector);
    set(MatrixOpType::ACTIVE, { X }, {}, { Y }, v, real_vector, matrix_vector);
}

void MatrixOp::as_pool(MatrixSP& X, MatrixSP& Y, PoolingType pooling_type, int reverse, std::vector<int> window, std::vector<int> stride, std::vector<int> padding, realc a)
{
    auto dim = X->getDim();
    getDefaultStridePadding(MatrixOpType::POOL, window, stride, padding);
    if (reverse == 0)
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
    std::vector<int> v = { int(window.size()), int(pooling_type), reverse };
    Y->resize(dim);
    set(MatrixOpType::POOL, { X }, {}, { Y }, v, { a }, {}, { window, stride, padding });
}

void MatrixOp::as_conv(MatrixSP& X, MatrixSP& W, MatrixSP& Y, std::vector<int> stride, std::vector<int> padding, int conv_type, realc a /*= 1*/)
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
    if (conv_type == 1)
    {
        t = MatrixOpType::CORR;
    }
    //在卷积计算开始之前，会查找最快的算法，v中依次保存了前向、反向数据、反向权重的算法编号、张量参数（mathtype）、组数
    //组数不同会导致工作空间尺寸不同，实际上此处只三个工作空间只用了一个，另两个未使用
    set(t, { X }, { W }, { Y }, v, { a }, { Matrix(), Matrix(), Matrix() }, { stride, padding });
}

void MatrixOp::as_reshape(MatrixSP& X, MatrixSP& Y, std::vector<int>& dim)
{
    Y->shareData(*X);
    Y->resize(dim);
    set(MatrixOpType::RESHAPE, { X }, {}, { Y });
}

void MatrixOp::as_max(MatrixSP& X1, MatrixSP& X2, MatrixSP& Y)
{
    Y->resize(*X1);
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

std::vector<MatrixOp> crossEntropy(MatrixSP& A, MatrixSP& Y)
{
    MatrixOp op;
    op.set(MatrixOpType::LOSS, { A, Y }, {}, {});
    return { op };
}

std::vector<MatrixOp> L2(MatrixSP& A)
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