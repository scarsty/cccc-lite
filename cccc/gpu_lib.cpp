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
        // Look for the name with "_v2" first (NVIDIA's habit), then without it
        // Note that sometimes func is a macro to define to ignore the suffix, the searching string will be expanded without it
        // If NVIDIA updates the library, the suffix may be changed to "_v3", so we can update the list here
#define IMPORT(func) \
    for (auto& lib : libs) \
    { \
        auto str = #func; \
        func = (func##_t)DynamicLibrary::getFunction(lib, str); \
        if (func) \
        { \
            /*LOG("Found {} in {}\n", str, lib);*/ \
            libs_used[lib].push_back(str); \
            break; \
        } \
    } \
    if (func == nullptr) \
    { \
        /*LOG(stderr, "Failed to found {}!\n", #func);*/ \
        func_failed.push_back(#func); \
    }
#if ENABLE_CUDA
#include "cuda_libs.inc"
#endif
#if ENABLE_HIP
#include "hip_libs.inc"
#endif
#undef IMPORT
        int sum = 0;
        for (auto& lu : libs_used)
        {
            //LOG("Loaded dynamic library {} for {} functions\n", lu.first, lu.second.size());
            sum += lu.second.size();
        }
        LOG("Found {} functions, {} failed\n", sum, func_failed.size());
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
    std::vector<std::string> suffix = { "", "_v2" };
    std::map<std::string, std::vector<std::string>> libs_used;
    std::vector<std::string> func_failed;
};
static cuda_assist_class_t cuda_assist_class;

};    // namespace cccc

#endif
