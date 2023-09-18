#include "DataPreparerMnist.h"
#include "cccc_extension.h"

DLL_EXPORT void* dp_ext()
{
    return new cccc::DataPreparerMnist();
}
