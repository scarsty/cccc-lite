#pragma once
#include "Net.h"

namespace woco
{

class NetOnnx : public Net
{
public:    
    virtual void structure() override;
};

}    // namespace woco
