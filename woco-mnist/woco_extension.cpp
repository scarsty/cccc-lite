#include "woco_extension.h"
#include "DataPreparerMnist.h"
#include "NetMnist.h"

void* dp_ext()
{
    static int i = 0;
    static std::map<int, woco::DataPreparerMnist> m;
    return &m[i++];
}

void* net_ext()
{
    static int i = 0;
    static std::map<int, woco::NetMnist> n;
    return &n[i++];
}