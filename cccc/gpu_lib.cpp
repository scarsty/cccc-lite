#include "gpu_lib.h"
#include "DynamicLibrary.h"
#include "Log.h"
#include <string>
#include <vector>

#if defined(_WIN32) && defined(AUTO_LOAD_GPU_FUNCTIONS)

namespace cccc
{

#define IMPORT2(func)

#define IMPORT(func) func##_t func = nullptr;
#if ENABLE_CUDA
#include "cuda_libs.inc"
#endif
#if ENABLE_HIP
#include "hip_libs.inc"
#endif
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
            LOG("Found {} in {}\n", #func, lib); \
            libs_used[lib]++; \
            break; \
        } \
    } \
    if (func == nullptr) \
    { \
        LOG(stderr, "Failed to found {}!\n", #func); \
    }
#if ENABLE_CUDA
#include "cuda_libs.inc"
#endif
#if ENABLE_HIP
#include "hip_libs.inc"
#endif
#undef IMPORT
        for (auto& lu : libs_used)
        {
            //LOG("Loaded dynamic library {}\n", lu.first);
        }
    }

private:
    std::vector<std::string> libs = {
        "cudart64_12",
        "cudart64_110",
        "cudart64_100",
        "cublas64_12",
        "cublas64_11",
        "cublas64_100",
        "cudnn_adv_infer64_8",
        "cudnn_adv_train64_8",
        "cudnn_cnn_infer64_8",
        "cudnn_cnn_train64_8",
        "cudnn_ops_infer64_8",
        "cudnn_ops_train64_8",
        "cudnn64_7",
        "nvml",
        "amdhip64",
        "rocblas",
        "cccc-hip",
    };
    std::map<std::string, int> libs_used;
};
static cuda_assist_class_t cuda_assist_class;

};    // namespace cccc

#endif
