#include "NetOnnx.h"
#include "dll_export.h"

extern "C" DLL_EXPORT void* net_ext()
{
    static int i = 0;
    static std::map<int, woco::NetOnnx> n;
    return &n[i++];
}