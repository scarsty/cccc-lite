#pragma once
#include "Log.h"
#include "Util.h"

namespace will
{

//神经基类，其子类的用于构造神经网
//因safe_delete和log非常常用，故在基类直接提供

class Neural : public Util, public Log
{
public:
    Neural();
    virtual ~Neural();

private:
    Neural(const Neural&) = delete;
    Neural& operator=(const Neural&) = delete;

};

}    // namespace will