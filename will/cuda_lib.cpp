#include "cuda_lib.h"
#include "DynamicLibrary.h"
#include "Log.h"
#include <string>
#include <vector>

#ifndef NO_CUDA
#if defined(_WIN32) && defined(AUTO_CUDA_VERSION)

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
            /*LOG("Found {} in {}\n", #func, lib);*/ \
            libs_used[lib]++; \
            break; \
        } \
    } \
    if (func == nullptr) { LOG(stderr, "Found {} failed!\n", #func); }

#include "cuda_lib.inc"
#undef IMPORT
        for (auto& lu : libs_used)
        {
            LOG("Loaded dynamic library {}\n", lu.first);
        }
    }

private:
    std::vector<std::string> libs = { "cublas64_11", "cudnn64_8", "cublas64_100", "cudnn64_7" };
    std::map<std::string, int> libs_used;
};
static cuda_assist_class_t cuda_assist_class;

};    // namespace cccc

#endif
#endif
