#include "Layer.h"
#include "MatrixEx.h"
#include "Random.h"
#include "VectorMath.h"

namespace cccc
{

Layer::Layer()
{
}

Layer::~Layer()
{
}

//设置名字
void Layer::setName(const std::string& name)
{
    layer_name_ = name;
}

//这个会检查前一层
LayerConnectionType Layer::getConnection2()
{
    auto ct = connetion_type_;
    if (connetion_type_ == LAYER_CONNECTION_DIRECT)
    {
        ct = prev_layer_->connetion_type_;
    }
    return ct;
}

//必须先设置option和layer_name
void Layer::message()
{
    //激活函数相关
    auto string_vector_int = [](const std::vector<int>& v)
    {
        std::string str;
        for (auto i : v)
        {
            str += std::to_string(i) + ", ";
        }
        if (str.size() >= 2)
        {
            str.resize(str.size() - 2);
        }
        return str;
    };

    //输出本层信息
    LOG("  name: {}\n", layer_name_);
    LOG("  type: {}\n", option_->getStringFromEnum(connetion_type_));
    LOG("  active: {}\n", option_->getStringFromEnum(option_->getEnum(layer_name_, "active", ACTIVE_FUNCTION_NONE)));
    //LOG("  solver: {}\n", option_->getStringFromEnum(solver_->getSolverType()));
    auto dim = A_->getDim();
    dim.pop_back();
    LOG("  out nodes: {} {}\n", A_->getRow(), dim);
    if (W_->getDataSize() > 0)
    {
        LOG("  weight size: {} {}\n", W_->getDataSize(), W_->getDim());
    }
    if (b_->getDataSize() > 0)
    {
        LOG("  bias size: {}\n", b_->getDataSize());
    }
    LOG("  x data size: {} [{}, batch = {}]\n", A_->getDataSize(), A_->getRow(), A_->getNumber());
    //LOG("  have bias: %d\n", need_bias_);

    auto string_layer_names = [](std::vector<Layer*>& ls)
    {
        std::string str;
        for (auto l : ls)
        {
            str += l->layer_name_ + ", ";
        }
        if (str.size() >= 2)
        {
            str.resize(str.size() - 2);
        }
        return str;
    };

    if (!prev_layers_.empty())
    {
        LOG("  prev layer(s): {}\n", string_layer_names(prev_layers_));
        if (connetion_type_ != LAYER_CONNECTION_COMBINE && prev_layers_.size() > 1)
        {
            LOG("Warning: only {} is effective!", prev_layer_->getName().c_str());
        }
    }
    if (!next_layers_.empty())
    {
        LOG("  next layer(s): {}\n", string_layer_names(next_layers_));
    }
}

void Layer::makeMatrixOp(std::vector<MatrixOp>& op_queue)
{
    int batch = option_->getInt("train", "batch", 1);
    switch (connetion_type_)
    {
    case LAYER_CONNECTION_FULLCONNECT:
    {
        MatrixOp op;
        auto dim = option_->getVector<int>(layer_name_, "node");
        int out_channel = option_->getInt(layer_name_, "channel", 0);
        if (out_channel > 0)
        {
            dim.push_back(out_channel);
        }
        W_->resize(VectorMath::multiply(dim), prev_layer_->A_->getRow());
        dim.push_back(prev_layer_->A_->getNumber());
        op.as_mul(W_, prev_layer_->A_, A_, 1, dim);
        op_queue.push_back(op);
        if (option_->getInt(layer_name_, "need_bias", 1))
        {
            MatrixOp op;
            b_->resize(A_->getChannel(), 1);
            op.as_addBias(A_, b_, A_);
            op_queue.push_back(op);
        }
        break;
    }
    case LAYER_CONNECTION_CONVOLUTION:
    case LAYER_CONNECTION_CORRELATION:
    {
        MatrixOp op;
        int out_channel = option_->getInt(layer_name_, "channel", 1);
        auto prev_dim = prev_layer_->A_->getDim();
        auto prev_dim_window = prev_dim;
        int window_dim_size = prev_dim.size() - 2;
        auto window = option_->getVector<int>(layer_name_, "window");
        auto stride = option_->getVector<int>(layer_name_, "stride");
        auto padding = option_->getVector<int>(layer_name_, "padding");
        VectorMath::force_resize(window, window_dim_size, 1);
        VectorMath::force_resize(stride, window_dim_size, 1);
        VectorMath::force_resize(padding, window_dim_size, 0);
        auto weight_dim = window;
        weight_dim.push_back(prev_layer_->A_->getChannel());
        weight_dim.push_back(out_channel);
        W_->resize(weight_dim);
        int t = 0;
        if (connetion_type_ == LAYER_CONNECTION_CONVOLUTION)
        {
            t = 0;
        }
        else if (connetion_type_ == LAYER_CONNECTION_CORRELATION)
        {
            t = 1;
        }
        op.as_conv(prev_layer_->A_, W_, A_, stride, padding, t);
        op_queue.push_back(op);
        if (option_->getInt(layer_name_, "need_bias", 1))
        {
            MatrixOp op;
            b_->resize(A_->getChannel(), 1);
            op.as_addBias(A_, b_, A_);
            op_queue.push_back(op);
        }
        break;
    }
    case LAYER_CONNECTION_POOLING:
    {
        MatrixOp op;
        auto prev_dim = prev_layer_->A_->getDim();
        auto prev_dim_window = prev_dim;
        int window_dim_size = prev_dim.size() - 2;
        auto window = option_->getVector<int>(layer_name_, "window");
        auto stride = option_->getVector<int>(layer_name_, "stride");
        auto padding = option_->getVector<int>(layer_name_, "padding");
        VectorMath::force_resize(window, window_dim_size, 1);
        VectorMath::force_resize(stride, window_dim_size, -1);
        for (int i = 0; i < window_dim_size; i++)
        {
            if (stride[i] < 0)
            {
                stride[i] = window[i];
            }
        }
        VectorMath::force_resize(padding, window_dim_size, 0);
        int reverse = option_->getInt(layer_name_, "reverse", 0);
        op.as_pool(prev_layer_->A_, A_, POOLING_MAX, reverse, window, stride, padding);
        op_queue.push_back(op);
        break;
    }
    case LAYER_CONNECTION_DIRECT:
    {
        //layer = new LayerDirect();
        break;
    }
    case LAYER_CONNECTION_COMBINE:
    {
        MatrixOp op;
        std::vector<MatrixSP> in;
        for (auto l : prev_layers_)
        {
            in.push_back(l->A_);
        }
        if (option_->getEnum<CombineType>(layer_name_, "combine_type") == COMBINE_CONCAT)
        {
            op.as_concat(in, A_);
        }
        else
        {
            op.as_add(in, A_);
        }
        op_queue.push_back(op);
        break;
    }
    case LAYER_CONNECTION_NONE:
    {
        auto dim = option_->getVector<int>(layer_name_, "data");
        int out_channel = option_->getInt(layer_name_, "channel", 0);
        if (out_channel > 0)
        {
            dim.push_back(out_channel);
        }
        dim.push_back(batch);
        A_->resize(dim);
        break;
    }
    }
    auto active = option_->getEnum(layer_name_, "active", ACTIVE_FUNCTION_NONE);
    if (active != ACTIVE_FUNCTION_NONE)
    {
        MatrixOp op;
        auto Ap = A_;
        A_ = makeMatrixSP();
        real coef = option_->getReal(layer_name_, "coef", 0.2);
        op.as_active(Ap, A_, active, {}, { coef }, {});
        op_queue.push_back(op);
    }
}

}    // namespace cccc