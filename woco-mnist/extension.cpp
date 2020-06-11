#include "DataPreparerMnist.h"
#include "NetMnist.h"
#include "dll_export.h"

extern "C" DLL_EXPORT void* dp_ext()
{
    static int i = 0;
    static std::map<int, woco::DataPreparerMnist> m;
    return &m[i++];
}

extern "C" DLL_EXPORT void* net_ext()
{
    static int i = 0;
    static std::map<int, woco::NetMnist> n;
    return &n[i++];
}