#include "cuda_lib.h"
#include "DynamicLibrary.h"
#include <string>
#include <vector>

#if defined(_WIN32) && defined(AUTO_CUDA_VERSION) && !defined(NO_CUDA)

namespace cccc
{

#define IMPORT(func) func##_t func = nullptr;
#include "cuda_lib.inc"
#undef IMPORT

class cuda_assist_class_t
{
public:
    cuda_assist_class_t()
    {
#define IMPORT(func) \
    for (auto& lib : libs) \
    { \
        func = (func##_t)DynamicLibrary::getFunction(lib, #func); \
        if (func) \
        { \
            /*fprintf(stdout, "Found %s in %s\n", #func, lib.c_str());*/ \
            break; \
        } \
    } \
    if (func == nullptr) { fprintf(stderr, "Found %s failed!\n", #func); }

#include "cuda_lib.inc"
#undef IMPORT
    }

private:
    std::vector<std::string> libs = { "cublas64_11", "cudnn64_8", "cublas64_100", "cudnn64_7" };
};
static cuda_assist_class_t cuda_assist_class;

};    // namespace cccc

#endif
