#include "MatrixOp.h"
#include "VectorMath.h"
#include <map>

namespace cccc
{

void MatrixOp::forward(std::vector<MatrixOp>& op_queue)
{
    for (auto& op : op_queue)
    {
        //op.in_[0]->message("in");
        op.forwardData();
        //op.out_[0]->message("out");
    }
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
        MatrixEx::poolingForward(X, Y, PoolingType(para_int_[1]), para_int_[2],
            para_int_v_[0], para_int_v_[1], para_int_v_[2], para_real_[0]);
        break;
    case MatrixOpType::CONV:
        MatrixEx::convolutionForward(X, *wb_[0], Y, para_int_, para_matrix_, para_int_v_[0], para_int_v_[1], para_real_[0]);
        break;
    case MatrixOpType::CORR:
        MatrixEx::correlationForward(X, *wb_[0], Y, para_int_, para_matrix_, para_int_v_[0], para_int_v_[1], para_real_[0]);
        break;
    }
}

void MatrixOp::backwardDataWeight()
{
    auto& Y = *out_[0];
    real data_weight = 1;
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
            //Y.d().message();
            //(*in_[0]).message();
            //wb_[0]->d().message();
            Matrix::mul(Y.d(), *in_[0], wb_[0]->d(), para_real_[0], data_weight, MATRIX_NO_TRANS, MATRIX_TRANS);
            //wb_[0]->d().message();
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
        MatrixEx::addBiasBackward(*in_[0], *wb_[0], Y, 0, 1);
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
            MatrixEx::poolingBackward(*in_[0], Y, PoolingType(para_int_[1]), para_int_[2],
                para_int_v_[0], para_int_v_[1], para_int_v_[2], para_real_[0], in_[0]->keepWeight());
        }
        break;
    case MatrixOpType::CONV:
        MatrixEx::convolutionBackward(*in_[0], *wb_[0], Y, para_int_, para_matrix_, para_int_v_[0], para_int_v_[1], para_real_[0], in_[0]->keepWeight(), 1);
        break;
    case MatrixOpType::CORR:
        MatrixEx::correlationBackward(*in_[0], *wb_[0], Y, para_int_, para_matrix_, para_int_v_[0], para_int_v_[1], para_real_[0], in_[0]->keepWeight(), 1);
        break;
    case MatrixOpType::RESHAPE:
        Matrix::copyData(Y.d(), in_[0]->d());
        break;
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
            //此处直接相减，表示欧氏距离平方，若配合前一层的softmax_ce或sigmoid_ce则表示交叉熵
            //in_[0]->message();
            //in_[1]->message("answer");
            //in_[0]->d().message("0");
            Matrix::add(*in_[0], *in_[1], in_[0]->d(), scale_, -scale_, in_[0]->keepWeight());
            //in_[0]->d().message("loss1");
            //printf("%f\n", in_[0].keepWeight());
            if (para_matrix_.size() >= 1 && para_matrix_[0].getDataSize() == in_[0]->getDataSize())    //调整loss的强调权重
            {
                Matrix::elementMul(in_[0]->d(), para_matrix_[0], in_[0]->d());
                //in_[1]->printAsMatrix();
                //in_[0]->d().printAsMatrix();
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

void MatrixOp::print(const std::vector<MatrixOp>& op_queue)
{
    LOG("begin->");
    for (const auto& op : op_queue)
    {
        op.print();
    }
    LOG("end\n");
}

void MatrixOp::print() const
{
    std::vector<std::string> strs = {
        "none",
        "add",
        "mul",
        "ele_mul",
        "add_bias",
        "concat",
        "active",
        "pool",
        "conv",
        "reshape",
        "loss",
        "l2",
    };

    LOG("{}->", strs[int(type_)]);
#ifdef _DEBUG
    //LOG( "\n");
    //for (const auto& m : matrix_in_)
    //{
    //    m.message();
    //}
    //for (const auto& m : matrix_out_)
    //{
    //    m.message();
    //}
    //LOG("\n");
#endif
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
                if (m->getDataPointer() == M.getDataPointer())
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
    //as_1.initData(1);
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
    set(t, { X }, { W }, { Y }, v, { a }, { Matrix(), Matrix(), Matrix() }, { stride, padding });
}

void MatrixOp::as_reshape(MatrixSP& X, MatrixSP& Y, std::vector<int>& dim)
{
    Y->shareData(*X);
    Y->resize(dim);
    set(MatrixOpType::RESHAPE, { X }, {}, { Y });
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