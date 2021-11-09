#pragma once
#include "Layer.h"
#include "Net.h"

namespace cccc
{

class NetLayer : public Net
{
public:
    NetLayer();
    virtual ~NetLayer();

private:
    //网络结构

    std::map<std::string, std::shared_ptr<Layer>> all_layer_map_;    //按名字保存的层

public:
    virtual int init2() override;

private:
    int createAndConnectLayers();
};

}    // namespace cccc