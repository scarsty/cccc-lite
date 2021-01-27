#include "DataPreparerMnist.h"
#include "will_extension.h"

DLL_EXPORT void* dp_ext()
{
    static int i = 0;
    static std::map<int, will::DataPreparerMnist> m;
    return &m[i++];
}
