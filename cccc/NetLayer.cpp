#include "NetLayer.h"
#include "ConsoleControl.h"
#include "filefunc.h"
#include "Timer.h"
#include "VectorMath.h"
#include <algorithm>

namespace cccc
{

NetLayer::NetLayer()
{
}

NetLayer::~NetLayer()
{
}

//注意：输入和输出层的名字可以自定义，因此一个ini中可以写多个网络
int NetLayer::init2()
{
    //batch在一开始会用来设置网络的初始大小
    LOG::setLevel(option_->getInt("train", "output_net", 1));
    int state = createAndConnectLayers();
    LOG::restoreLevel();
    if (state)
    {
        LOG("Net structure is wrong!\n");
        return 1;
    }
    Y_->resize(*A_);
    return 0;
}

int NetLayer::createAndConnectLayers()
{
    Layer *layer_in = nullptr, *layer_out = nullptr;
    std::string layer_in_name = "layer_in";
    std::string layer_out_name = "layer_out";
    std::vector<Layer*> all_layer_vector;    //全部查找到的层，包含一些多余的
    std::vector<Layer*> layer_vector;        //此向量的顺序即计算顺序

    //lambda函数：计算上层和下层
    auto connect_layers = [this]()
    {
        for (auto& name_layer : all_layer_map_)
        {
            auto& l = name_layer.second;
            auto nexts = strfunc::splitString(option_->getString(l->getName(), "next"), ",");
            for (auto& next : nexts)
            {
                strfunc::replaceAllSubStringRef(next, " ", "");
                if (next != "" && all_layer_map_.count(next) > 0)
                {
                    l->addNextLayers(all_layer_map_[next].get());
                    all_layer_map_[next]->addPrevLayers(l.get());
                }
            }
        }
    };
    //lambda函数：层是否已经在向量中
    auto contains = [&](std::vector<Layer*>& v, Layer* l) -> bool
    { return std::find(v.begin(), v.end(), l) != v.end(); };
    //lambda函数：递归将层压入向量
    //最后一个参数为假，仅计算是否存在连接，为真则是严格计算传导顺序
    std::function<void(Layer*, int, std::vector<Layer*>&, bool)> push_cal_stack = [&](Layer* layer, int direct, std::vector<Layer*>& stack, bool turn)
    {
        //层连接不能回环
        if (layer == nullptr || contains(stack, layer))
        {
            return;
        }
        std::vector<Layer*> connect0, connect1;
        connect1 = layer->getNextLayers();
        connect0 = layer->getPrevLayers();

        if (direct < 0)
        {
            std::swap(connect0, connect1);
        }
        //前面的层都被压入，才压入本层
        bool contain_all0 = true;
        for (auto& l : connect0)
        {
            if (!contains(stack, l))
            {
                contain_all0 = false;
                break;
            }
        }
        if (!turn || (!contains(stack, layer) && contain_all0))
        {
            stack.push_back(layer);
        }
        else
        {
            return;
        }
        for (auto& l : connect1)
        {
            push_cal_stack(l, direct, stack, turn);
        }
    };

    //查找所有存在定义的层
    auto sections = option_->getAllSections();

    //先把层都创建起来
    for (auto& section : sections)
    {
        if (strfunc::toLowerCase(section).find("layer") == 0)
        {
            LOG("Found layer {}\n", section);
            auto ct = option_->getEnum(section, "type", LAYER_CONNECTION_NONE);
            auto l = std::make_shared<Layer>();
            l->setConnection(ct);
            l->setOption(option_);
            l->setName(section);
            l->setNet(this);

            //此处检查是否某些层是否实际上无输出，注意并不严格，网络的正确性应由用户验证
            auto dim = option_->getVector<int>(section, "node");
            dim.push_back(option_->getInt(section, "channel", 1));
            if (VectorMath::multiply(dim) <= 0)
            {
                option_->setKey(section, "next", "");
            }
            all_layer_vector.push_back(l.get());
            if (l->getName() == layer_in_name)
            {
                layer_in = l.get();
                l->setVisible(LAYER_VISIBLE_IN);
            }
            if (l->getName() == layer_out_name)
            {
                layer_out = l.get();
                l->setVisible(LAYER_VISIBLE_OUT);
            }
            all_layer_map_[l->getName()] = l;
        }
    }
    //连接，计算双向的连接，清除无用项，重新连接
    connect_layers();
    std::vector<Layer*> forward, backward;
    //层的计算顺序
    push_cal_stack(layer_in, 1, forward, false);
    push_cal_stack(layer_out, -1, backward, false);

    //不在双向层中的都废弃
    std::vector<std::pair<std::string, std::shared_ptr<Layer>>> temp;
    for (auto& name_layer : all_layer_map_)
    {
        auto l = name_layer.second.get();
        if (!(contains(forward, l) && contains(backward, l)))
        {
            temp.push_back(name_layer);
        }
    }
    for (auto& name_layer : temp)
    {
        all_layer_map_.erase(name_layer.first);
        //safe_delete(name_layer.second);
        LOG("Remove bad layer {}\n", name_layer.first);
    }
    for (auto& l : all_layer_vector)
    {
        l->clearConnect();
    }
    connect_layers();
    forward.clear();
    backward.clear();
    push_cal_stack(layer_in, 1, layer_vector, true);
    for (int i = 0; i < layer_vector.size(); i++)
    {
        layer_vector[i]->setID(i);
    }
    //push_cal_stack(layer_out, -1, backward, true);

    //若不包含out项则有误
    if (!contains(layer_vector, layer_out))
    {
        return 1;
    }

    //初始化
    int index = 0;
    for (auto& layer : layer_vector)
    {
        LOG("---------- Layer {:3} ----------\n", index);
        layer->makeMatrixOp(op_queue_);
        layer->message();
        index++;
    }

    X_ = layer_vector.front()->A_;
    A_ = layer_vector.back()->A_;

    all_layer_map_.clear();

    auto loss_weight_values = strfunc::findNumbers<real>(option_->getString(layer_out_name, "loss_weight"));
    if (loss_weight_values.size() > 0)
    {
        LOG("Loss weight {}: {}\n", loss_weight_values.size(), loss_weight_values);
    }
    loss_weight_.resize(loss_weight_values.size(), 1);
    loss_weight_.importData(loss_weight_values.data(), loss_weight_values.size());
    return 0;
}

}    // namespace cccc