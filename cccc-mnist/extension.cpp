#include "DataPreparerMnist.h"

extern "C" CCCC_EXPORT void* dp_ext()
{
    return new cccc::DataPreparerMnist();
}
