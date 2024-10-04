#include "gpu_lib.h"
#include "DynamicLibrary.h"
#include "Log.h"
#include <string>
#include <vector>

namespace cccc
{
#if defined(_WIN32) && defined(AUTO_LOAD_GPU_FUNCTIONS)

#define UNIMPORT_OR_BLANKTEMP(func)

#define IMPORT(func, ...) func##_t func = nullptr;
#if ENABLE_CUDA
#include "cuda_libs.inc"
#endif
#if ENABLE_HIP
#include "hip_libs.inc"
#endif
#undef IMPORT

class gpu_assist_class_t
{
public:
    gpu_assist_class_t()
    {
        std::vector<std::string> strs;
#define IMPORT(func, ...) \
    strs = \
        { \
            #func, \
            __VA_ARGS__ \
        }; \
    for (auto& lib : libs) \
    { \
        for (auto str : strs) \
        { \
            func = (func##_t)DynamicLibrary::getFunction(lib, str); \
            if (func) \
            { \
                if (std::string(#func) == str) \
                { /*LOG(stderr, "Found {} in {}\n", #func, lib);*/ \
                    libs_used[lib].push_back(#func); \
                } \
                else \
                { \
                } \
                break; \
            } \
        } \
        if (func) { break; } \
    } \
    IMPORT_2(func)
#define IMPORT_2(func) \
    if (func) { func_c.push_back(#func); } \
    else { func_c_failed.push_back(#func); }
#if ENABLE_CUDA
#include "cuda_libs.inc"
#endif
#undef IMPORT_2
#define IMPORT_2(func) \
    if (func) { func_h.push_back(#func); } \
    else { func_h_failed.push_back(#func); }
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
        LOG("Found {} functions, {} failed\n", sum, func_c_failed.size() + func_h_failed.size());
#if ENABLE_CUDA
        if (func_c.size() > 0 && func_c_failed.size() > 0)
        {
            LOG("Some CUDA functions are lost: {}\n", func_c_failed);
        }
        if (func_c.size() == 0)
        {
            LOG("No CUDA libraries!\n");
        }
#endif
#if ENABLE_HIP
        if (func_h.size() > 0 && func_h_failed.size() > 0)
        {
            LOG("Some HIP functions are lost: {}\n", func_h_failed);
        }
        if (func_h.size() == 0)
        {
            LOG("No HIP libraries!\n");
        }
#endif
    }

private:
    std::vector<std::string> libs = {
        "cudart64_12",
        "cudart64_110",
        "cudart64_100",
        "cublas64_12",
        "cublas64_11",
        "cublas64_100",
        "cudnn_graph64_9",
        "cudnn_adv64_9",
        "cudnn_cnn64_9",
        "cudnn_ops64_9",
        "cudnn_adv_infer64_8",
        "cudnn_adv_train64_8",
        "cudnn_cnn_infer64_8",
        "cudnn_cnn_train64_8",
        "cudnn_ops_infer64_8",
        "cudnn_ops_train64_8",
        "cudnn64_7",
        "amdhip64",
        "rocblas",
        "miopen",
        "cccc-hip",
    };
    std::map<std::string, std::vector<std::string>> libs_used;
    std::vector<std::string> func_c, func_h, func_c_failed, func_h_failed;
};

void find_gpu_functions()
{
    static gpu_assist_class_t gpu_assist_class;
}

#else
void find_gpu_functions()
{
}
#endif
}    //namespace cccc
