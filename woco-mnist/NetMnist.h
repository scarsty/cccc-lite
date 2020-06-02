#pragma once
#include "Net.h"

namespace woco
{

class NetMnist : public Net
{
public:
    void structureExample();
    virtual void structure() override;
};

}    // namespace woco