#pragma once
#include "NetCifa.h"

namespace woco
{

class NetOnnx : public NetCifa
{
public:
    virtual void structure() override;
    virtual void save(const std::string& filename) override;
};

}    // namespace woco
