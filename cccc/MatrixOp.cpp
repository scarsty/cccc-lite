#include "MatrixOp.h"
#include "MatrixEx.h"
#include "Timer.h"
#include "strfunc.h"
#include <functional>

namespace cccc
{

void MatrixOp::forward(std::vector<MatrixOp>& op_queue)
{
    //Timer t;
    for (auto& op : op_queue)
    {
        if (op.connect_x_)
        {
            //for (auto& in : op.in_)
            //{
            //    in->message("in");
            //}
            Timer t;
            op.forwardData();
            op.forward_time_ += t.getElapsedTime();
            op.forward_count_++;
            //op.out_[0]->message("out");
            //LOG("{}, {} forward time: {}\n", getOpName(op.type_), op.index_, t.getElapsedTime());
        }
    }
    //LOG("forward time: {} s\n", t.getElapsedTime());
}

void MatrixOp::backward(std::vector<MatrixOp>& op_queue, std::vector<MatrixOp>& loss, bool clear_d)
{
    if (clear_d)
    {
        //Timer t;
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
        //LOG("clear d time: {} s\n", t.getElapsedTime());
    }
    for (auto& op : loss)
    {
        Timer t;
        op.backwardLoss();
        op.backwark_time_ += t.getElapsedTime();
        op.backward_count_++;
        //LOG("{}, {} loss time: {} s\n", getOpName(op.type_), op.index_, t.getElapsedTime());
    }
    for (auto it = op_queue.rbegin(); it != op_queue.rend(); ++it)
    {
        if (it->connect_loss_)
        {
            Timer t;
            it->backwardDataWeight();
            it->backwark_time_ += t.getElapsedTime();
            it->backward_count_++;
            //LOG("{}, {} backward time: {}\n", getOpName(it->type_), it->index_, t.getElapsedTime());
            //it->out_[0]->d().message("dY" + getOpName(it->type_));
            //for (auto& m : it->in_)
            //{
            //    if (m->needBack())
            //    {
            //        m->message("X");
            //        m->d().message("dX" +getOpName(it->type_));
            //    }
            //}
        }
    }
}

void MatrixOp::forwardData()
{
    auto& X = *in_[0];
    auto& Y = *out_[0];
    switch (type_)
    {
    case MatrixOpType::SCALE:
        Matrix::scale(X, Y, a_[0]);
        break;
    case MatrixOpType::ADD:
        Matrix::add(X, *in_[1], Y, a_[0], a_[1], b_[0]);
        for (int i = 2; i < in_.size(); i++)
        {
            Matrix::add(Y, *in_[i], Y, 1, a_[i], 1);
        }
        break;
    case MatrixOpType::MUL:
        Matrix::mul(X, *in_[1], Y, a_[0]);
        break;
    case MatrixOpType::ELE_MUL:
        //参数in_[1]可以为单通道
        Matrix::elementMul(X, *in_[1], Y, a_[0], b_[0]);
        break;
    case MatrixOpType::ADD_BIAS:
        MatrixEx::addBias(X, *in_[1], Y, a_[0], b_[0]);
        break;
    case MatrixOpType::CONCAT:
        MatrixEx::concatByChannel(in_, Y);
        break;
    case MatrixOpType::ACTIVE:
        MatrixEx::activeForward(X, Y, anys_[0].to<ActiveFunctionType>(), anys_[1].to<std::vector<int>>(), anys_[2].to<std::vector<float>>());
        break;
    case MatrixOpType::POOL:
        MatrixEx::poolingForward(X, Y, anys_[0].to<PoolingType>(), anys_[1].to<PoolingReverseType>(), window_, stride_, padding_, a_[0], b_[0]);
        break;
    case MatrixOpType::CONV:
        MatrixEx::convolutionForward(X, *in_[1], Y, stride_, padding_, a_[0], b_[0]);
        break;
    case MatrixOpType::CORR:
        MatrixEx::correlationForward(X, *in_[1], Y, stride_, padding_, a_[0], b_[0]);
        break;
    case MatrixOpType::RESHAPE:
        Matrix::copyData(X, Y);
        break;
    case MatrixOpType::MAX:
        MatrixEx::matrix_max(*in_[0], *in_[1], Y);
        break;
    case MatrixOpType::BATCH_NORM:
        break;
    case MatrixOpType::POOL_CHANNEL:
        MatrixEx::poolingChannelForward(X, Y, anys_[0].to<PoolingType>(), anys_[1].to<PoolingReverseType>(), a_[0], b_[0]);
        break;
    }
}

void MatrixOp::backwardDataWeight()
{
    auto& Y = *out_[0];
    //float data_weight = 0;
    //若反向过程需更新多个矩阵，则在函数内部判断needUpdate
    //if (Y.isHip()) { data_weight = 0; }    //miopen只支持反向时beta为0
    switch (type_)
    {
    case MatrixOpType::SCALE:
        for (int i = 0; i < in_.size(); i++)
        {
            if (in_[i]->needBack())
            {
                Matrix::scale(Y.d(), in_[i]->d(), a_[i], in_[i]->keepWeight());
            }
        }
        break;
    case MatrixOpType::ADD:
        for (int i = 0; i < in_.size(); i++)
        {
            if (in_[i]->needBack())
            {
                Matrix::add(in_[i]->d(), Y.d(), in_[i]->d(), in_[i]->keepWeight(), a_[i]);
            }
        }
        break;
    case MatrixOpType::MUL:
        if (in_[1]->needBack())
        {
            Matrix::mul(*in_[0], Y.d(), in_[1]->d(), a_[0], in_[1]->keepWeight(), MATRIX_TRANS, MATRIX_NO_TRANS);
        }
        if (in_[0]->needBack())
        {
            Matrix::mul(Y.d(), *in_[1], in_[0]->d(), a_[0], in_[0]->keepWeight(), MATRIX_NO_TRANS, MATRIX_TRANS);
        }
        break;
    case MatrixOpType::ELE_MUL:
        if (in_[0]->needBack())
        {
            Matrix::elementMul(Y.d(), *in_[1], in_[0]->d(), a_[0], in_[0]->keepWeight());
        }
        if (in_[1]->needBack())
        {
            if (in_[1]->getChannel() == Y.getChannel())
            {
                Matrix::elementMul(Y.d(), *in_[0], in_[1]->d(), a_[0], in_[1]->keepWeight());
            }
            else if (in_[1]->getChannel() == 1)
            {
                MatrixEx::elementMulSum(Y.d(), *in_[0], in_[1]->d(), a_[0], in_[1]->keepWeight());
            }
        }
        break;
    case MatrixOpType::ADD_BIAS:
        if (in_[0]->needBack())
        {
            Matrix::add(in_[0]->d(), Y.d(), in_[0]->d(), 0, 1);
        }
        if (in_[1]->needBack())
        {
            MatrixEx::addBiasBackward(*in_[0], *in_[1], Y, 1, in_[1]->keepWeight());
        }
        break;
    case MatrixOpType::CONCAT:
        MatrixEx::concatByChannelBackward(in_, Y);
        break;
    case MatrixOpType::ACTIVE:
        if (in_[0]->needBack())
        {
            MatrixEx::activeBackward(*in_[0], Y, anys_[0].to<ActiveFunctionType>(), anys_[1].to<std::vector<int>>(), anys_[2].to<std::vector<float>>(), 1, in_[0]->keepWeight());
        }
        break;
    case MatrixOpType::POOL:
        if (in_[0]->needBack())
        {
            MatrixEx::poolingBackward(*in_[0], Y, anys_[0].to<PoolingType>(), anys_[1].to<PoolingReverseType>(), window_, stride_, padding_, a_[0], in_[0]->keepWeight());
        }
        break;
    case MatrixOpType::CONV:
        if (in_[0]->needBack())
        {
            MatrixEx::convolutionBackwardDX(*in_[0], *in_[1], Y, stride_, padding_, a_[0], in_[0]->keepWeight());
        }
        if (in_[1]->needBack())
        {
            MatrixEx::convolutionBackwardDW(*in_[0], *in_[1], Y, stride_, padding_, a_[0], in_[1]->keepWeight());
        }
        //MatrixEx::convolutionBackward(*in_[0], *wb_[0], Y, stride_, padding_, a_[0], in_[0]->keepWeight(), a[0], data_weight, &anys_[1].to<MatrixEx::ConvMethod>(), &anys_[2].to<MatrixEx::ConvMethod>(), &workspace_[1], &workspace_[2]);
        break;
    case MatrixOpType::CORR:
        //MatrixEx::correlationBackward(*in_[0], *wb_[0], Y, para_int_, para_matrix_, para_int_v_[0], para_int_v_[1], para_real_[0], in_[0]->keepWeight(), data_weight);
        break;
    case MatrixOpType::RESHAPE:
        Matrix::add(Y.d(), in_[0]->d(), in_[0]->d(), 1, in_[0]->keepWeight());
        break;
    case MatrixOpType::MAX:
        MatrixEx::matrix_maxb(*in_[0], *in_[1], Y, in_[0]->keepWeight(), in_[1]->keepWeight(), 1);
        break;
    case MatrixOpType::BATCH_NORM:
        break;
    case MatrixOpType::POOL_CHANNEL:
        if (in_[0]->needBack())
        {
            MatrixEx::poolingChannelBackward(*in_[0], Y, anys_[0].to<PoolingType>(), anys_[1].to<PoolingReverseType>(), a_[0], in_[0]->keepWeight());
        }
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
    //scale不合理，待计算
    switch (type_)
    {
    case MatrixOpType::LOSS:
        if (scale_ != 0)
        {
            if (in_.size() >= 3 && in_[2]->getDataSize() == in_[0]->getDataSize())
            {
                //有损失权重的情况
                //注意这里使用in_[2]->d()作为了中间变量
                Matrix::add(*in_[0], *in_[1], in_[2]->d(), a_[0] * scale_, -a_[1] * scale_, 0);
                Matrix::elementMul(in_[2]->d(), *in_[2], in_[0]->d(), 1, in_[0]->keepWeight());
                //in_[0]->message("X");
                //in_[2]->message("Y1");
                //in_[0]->d().message("Xd");
            }
            else
            {
                //没有损失权重的情况，也是一般的情况
                //此处直接相减，表示欧氏距离平方，若配合前一层的softmax_ce或sigmoid_ce则表示交叉熵
                Matrix::add(*in_[0], *in_[1], in_[0]->d(), a_[0] * scale_, -a_[1] * scale_, in_[0]->keepWeight());
                //in_[0]->message("X");
                //in_[1]->message("Y1");
                //in_[0]->d().message("Xd");
            }
        }
        break;
    case MatrixOpType::FOCAL:
        if (scale_ != 0)
        {
            if (in_.size() >= 3 && in_[2]->getDataSize() == in_[0]->getDataSize())
            {
                Matrix::add(*in_[0], *in_[1], in_[1]->d(), scale_, -scale_, 0);
                MatrixEx::elementPow(in_[1]->d(), in_[2]->d(), 0.2);
                Matrix::elementMul(in_[2]->d(), *in_[2], in_[0]->d(), 1, in_[0]->keepWeight());
            }
            else
            {
                Matrix::add(*in_[0], *in_[1], in_[1]->d(), scale_, -scale_, in_[0]->keepWeight());
                MatrixEx::elementPow(in_[1]->d(), in_[0]->d(), 0.2);
            }
            //in_[0]->d().message("Xdpow");
        }
        break;
    case MatrixOpType::ZERO_LIMIT:
        if (scale_ != 0)
        {
            MatrixEx::zero_limit(*in_[0], *in_[1], in_[0]->d(), 0.5, 0);
        }
        break;
    case MatrixOpType::L2:
        if (scale_ != 0)
        {
            //Matrix::add(in_[0]->d(), *in_[0], in_[0]->d(), data_weight_, scale_);
        }
        break;
    }
    in_[0]->setKeepWeight(1);
}

std::string MatrixOp::inference_ir(const std::vector<MatrixOp>& op_queue)
{
    std::string content;
    for (const auto& op : op_queue)
    {
        if (op.connect_a_)
        {
            content += op.print();    //仅用于推理，故有些连接loss，但不连接A的可以不计算
        }
    }
    return content;
}

std::string MatrixOp::print() const
{
    Option op;
    std::string line;
    switch (type_)
    {
    case MatrixOpType::SCALE:
        line = std::format("{} = scale({}, {});", out_[0], in_[0], a_[0]);
        break;
    case MatrixOpType::ADD:
        line = std::format("{} = add({}, {});", out_[0], in_, a_);
        break;
    case MatrixOpType::MUL:
        line = std::format("{} = mul({}, {}, [{}, {}, {}, batch]);", out_[0], in_[0], in_[1],
            out_[0]->getWidth(), out_[0]->getHeight(), out_[0]->getChannel());
        break;
    case MatrixOpType::ELE_MUL:
        line = std::format("{} = elementmul({}, {}, {});", out_[0], in_[0], in_[1], a_[0]);
        break;
    case MatrixOpType::ADD_BIAS:
        line = std::format("{} = addBias({}, {});", out_[0], in_[0], in_[1]);
        break;
    case MatrixOpType::CONCAT:
        line = std::format("{} = concat({});", out_[0], in_);
        break;
    case MatrixOpType::ACTIVE:
        line = std::format("{} = active({}, active_{}, {}, {});", out_[0], in_[0], op.getStringFromEnum(anys_[0].to<ActiveFunctionType>()), anys_[1].to<std::vector<int>>(), anys_[2].to<std::vector<float>>());
        break;
    case MatrixOpType::POOL:
        if (anys_[1].to<PoolingReverseType>() == POOLING_NOT_REVERSE)
        {
            line = std::format("{} = pool({}, pool_{}, {}, {}, {});", out_[0], in_[0], op.getStringFromEnum(anys_[0].to<PoolingType>()), window_, stride_, padding_);
        }
        else
        {
            line = std::format("{} = reversepool({}, {}, {}, {});", out_[0], in_[0], window_, stride_, padding_);
        }
        break;
    case MatrixOpType::CONV:
        line = std::format("{} = conv({}, {}, {}, {});", out_[0], in_[0], in_[1], stride_, padding_);
        break;
    case MatrixOpType::CORR:
        line = std::format("{} = corr({}, {}, {}, {});", out_[0], in_[0], in_[1], stride_, padding_);
        break;
    case MatrixOpType::RESHAPE:
        line = std::format("{} = reshape({}, {});", out_[0], in_[0], anys_[0].to<std::vector<int>>());
        break;
    case MatrixOpType::MAX:
        line = std::format("{} = max({}, {});", out_[0], in_[0], in_[1]);
        break;
    case MatrixOpType::POOL_CHANNEL:
        line = std::format("{} = poolchannel({}, pool_{});", out_[0], in_[0], op.getStringFromEnum(anys_[0].to<PoolingType>()));
        break;
    case MatrixOpType::LOSS:
        line = std::format("addloss(commonloss({}, {}, {}));", int(type_), in_, a_);
        break;
    default:
        line = std::format("addloss(commonloss({}, {}, {}));", int(type_), in_, a_);
    }
    strfunc::replaceAllSubStringRef(line, "[", "{");
    strfunc::replaceAllSubStringRef(line, "]", "}");
    for (auto& out : out_)
    {
        line += std::format("/*{}: out {}*/;", index_, out->sizeMessage(0));
    }
    return line;
}

ActiveFunctionType MatrixOp::getActiveType() const
{
    if (type_ == MatrixOpType::ACTIVE)
    {
        return anys_[0].to<ActiveFunctionType>();
    }
    return ACTIVE_FUNCTION_NONE;
}

//检查连接，并判断哪些是权重
void MatrixOp::checkConnect(std::vector<MatrixOp>& op_queue, Matrix& X, Matrix& A, std::vector<MatrixOp>& losses)
{
    enum ConnectType
    {
        CONNECT_X = 1,
        CONNECT_LOSS = -1,
        CONNECT_A = -2,
    };

    std::unordered_map<Matrix*, int> linkX, linkLoss, linkA;
    std::function<void(Matrix&, ConnectType, std::unordered_map<Matrix*, int>&)> check_connect = [&op_queue, &check_connect](Matrix& M, ConnectType direct, std::unordered_map<Matrix*, int>& link_record)
    {
        if (direct == 0)
        {
            return;
        }
        for (int i = 0; i < op_queue.size(); i++)
        {
            bool* connect = nullptr;
            if (direct == CONNECT_X)
            {
                connect = &op_queue[i].connect_x_;
            }
            else if (direct == CONNECT_A)
            {
                connect = &op_queue[i].connect_a_;
            }
            else if (direct == CONNECT_LOSS)
            {
                connect = &op_queue[i].connect_loss_;
            }

            if (*connect)
            {
                continue;
            }
            auto& op = op_queue[i];
            std::vector<MatrixSP>*v1, *v2;
            if (direct == CONNECT_X)
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
                    *connect = true;
                    for (auto& m : *v2)
                    {
                        link_record[m.get()] = 1;
                        check_connect(*m, direct, link_record);
                    }
                    break;
                }
            }
        }
    };

    linkX[&X] = 1;
    check_connect(X, CONNECT_X, linkX);
    linkA[&A] = 1;
    check_connect(A, CONNECT_A, linkA);

    for (auto& loss : losses)
    {
        for (auto& m : loss.in_)
        {
            linkLoss[m.get()] = 1;
            check_connect(*m, CONNECT_LOSS, linkLoss);
        }
    }

    int i = 0;
    for (auto it = op_queue.begin(); it != op_queue.end();)
    {
        if (!it->connect_x_ && !it->connect_loss_)
        {
            it = op_queue.erase(it);
        }
        else
        {
            it->index_ = i++;
            for (auto& in : it->in_)
            {
                if (!linkX.contains(in.get()) && linkLoss.contains(in.get()))
                {
                    in->setIsWeight(true);
                }
            }
            for (auto& out : it->out_)
            {
                out->setIsInput(false);    //非输入矩阵
            }
            ++it;
        }
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
    set(MatrixOpType::SCALE, { X }, { Y }, { r }, { 0 });
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
    set(MatrixOpType::MUL, { X1, X2 }, { Y }, { a });    //这里注意顺序
    if (X1->getNumber() != X2->getRow())
    {
        LOG_ERR("Error: cannot product!\n");
    }
}

void MatrixOp::as_elementMul(const MatrixSP& X1, const MatrixSP& X2, const MatrixSP& Y, float a)
{
    Y->resize(X1->getDim());
    set(MatrixOpType::ELE_MUL, { X1, X2 }, { Y }, { a }, { 0 });
}

void MatrixOp::as_add(const MatrixSP& X1, const MatrixSP& X2, const MatrixSP& Y, float a, float b)
{
    Y->resize(X1->getDim());
    set(MatrixOpType::ADD, { X1, X2 }, { Y }, { a, b }, { 0 });
}

void MatrixOp::as_add(const std::vector<MatrixSP>& X_vector, const MatrixSP& Y, std::vector<float> a)
{
    Y->resize(X_vector[0]->getDim());
    set(MatrixOpType::ADD, X_vector, { Y }, std::move(a), { 0 });
}

void MatrixOp::as_addBias(const MatrixSP& X, const MatrixSP& bias, const MatrixSP& Y, float a, float b)
{
    //Matrix as_1(A.getNumber(), 1);
    //as_1.fillData(1);
    //需要注意cudnn自带的只支持到5维，若需更多维可以在这里修改写入op_queue的矩阵的维度
    Y->shareData(*X);    //需注意偏移操作是特殊处理的
    Y->resize(X->getDim());
    set(MatrixOpType::ADD_BIAS, { X, bias }, { Y }, { a }, { b });
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
    set(MatrixOpType::CONCAT, X_vector, { Y });
}

void MatrixOp::as_active(const MatrixSP& X, const MatrixSP& Y, ActiveFunctionType af)
{
    Y->resize(X->getDim());
    set(MatrixOpType::ACTIVE, { X }, { Y }, {}, {}, { af, std::vector<int>(), std::vector<float>() });
}

void MatrixOp::as_active(const MatrixSP& X, const MatrixSP& Y, ActiveFunctionType af, std::vector<int>&& int_vector, std::vector<float>&& real_vector, std::vector<Matrix>&& matrix_vector)
{
    Y->resize(X->getDim());
    set(MatrixOpType::ACTIVE, { X }, { Y }, {}, {}, { af, int_vector, real_vector });
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
    set(MatrixOpType::POOL, { X }, { Y }, { a }, {}, std::move(pv), std::move(window), std::move(stride), std::move(padding));
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
    Y->user_data<MatrixEx::ConvMethod>() = method;
    //在卷积计算开始之前，会查找最快的算法，v中依次保存了前向、反向数据、反向权重的算法编号、张量参数（mathtype）、组数
    set(t, { X, W }, { Y }, { a }, {}, {}, {}, std::move(stride), std::move(padding));
}

void MatrixOp::as_corr(const MatrixSP& X, const MatrixSP& W, const MatrixSP& Y, std::vector<int> stride, std::vector<int> padding, int conv_algo, float a)
{
    as_conv(X, W, Y, stride, padding, conv_algo, a);
    type_ = MatrixOpType::CORR;
}

void MatrixOp::as_reshape(const MatrixSP& X, const MatrixSP& Y, std::vector<int>& dim)
{
    Y->shareData(*X);
    if (!dim.empty())
    {
        dim.back() = X->getNumber();
    }
    //不处理组数
    Y->resize(dim);
    set(MatrixOpType::RESHAPE, { X }, { Y }, {}, {}, { dim });
}

void MatrixOp::as_max(const MatrixSP& X1, const MatrixSP& X2, const MatrixSP& Y)
{
    Y->resize(X1->getDim());
    set(MatrixOpType::MAX, { X1, X2 }, { Y });
}

//未完成
void MatrixOp::as_batchNorm(const MatrixSP& X1, const MatrixSP& X2, const MatrixSP& Y) {}

void MatrixOp::as_poolChannel(const MatrixSP& X, const MatrixSP& Y, PoolingType pooling_type, PoolingReverseType reverse_type, float a)
{
    auto dim = X->getDim();
    dim[dim.size() - 2] = 1;
    Y->resize(dim);
    set(MatrixOpType::POOL_CHANNEL, { X }, { Y }, { a }, {}, { pooling_type, reverse_type });
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

std::vector<MatrixOp> commonLoss(MatrixOpType type, const std::vector<MatrixSP>& M, const std::vector<float>& a)
{
    MatrixOp op;
    auto a1 = a;
    op.set(type, M, {}, std::move(a1), {});
    return { op };
}

std::vector<MatrixOp> crossEntropy(const MatrixSP& A, const MatrixSP& Y)
{
    MatrixOp op;
    op.set(MatrixOpType::LOSS, { A, Y }, {}, {});
    return { op };
}

std::vector<MatrixOp> crossEntropy(const MatrixSP& A, const MatrixSP& Y, const MatrixSP& LW)
{
    MatrixOp op;
    op.set(MatrixOpType::LOSS, { A, Y, LW }, {}, {});
    return { op };
}

std::vector<MatrixOp> focal(const MatrixSP& A, const MatrixSP& Y)
{
    MatrixOp op;
    op.set(MatrixOpType::FOCAL, { A, Y }, {}, {});
    return { op };
}

std::vector<MatrixOp> focal(const MatrixSP& A, const MatrixSP& Y, const MatrixSP& LW)
{
    MatrixOp op;
    op.set(MatrixOpType::FOCAL, { A, Y, LW }, {}, {});
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