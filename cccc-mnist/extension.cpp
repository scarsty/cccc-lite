#include "DataPreparerMnist.h"
#include "cccc_extension.h"

DLL_EXPORT void* dp_ext()
{
    static int i = 0;
    static std::map<int, cccc::DataPreparerMnist> m;
    return &m[i++];
}
