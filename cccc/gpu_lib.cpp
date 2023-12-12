#include "gpu_lib.h"
#include "DynamicLibrary.h"
#include "Log.h"
#include <string>
#include <vector>

#if defined(_WIN32) && defined(AUTO_LOAD_GPU_FUNCTIONS)

namespace cccc
{

#define UNIMPORT_OR_BLANKTEMP(func)

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
        std::vector<std::string> strs;
#define IMPORT(func, ...) \
    strs = { #func, __VA_ARGS__ }; \
    for (auto& lib : libs) \
    { \
        for (auto str : strs) \
        { \
            func = (func##_t)DynamicLibrary::getFunction(lib, str); \
            if (func) \
            { \
                if (std::string(#func) == str) \
                { /*LOG(stderr, "Found {} in {}\n", #func, lib);*/ \
                } \
                else \
                { /*LOG(stderr, "Found {}() in {}\n", #func, str, lib);*/ \
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
        //LOG("Found {} functions, {} failed\n", sum, func_c_failed.size() + func_h_failed.size());
        if (func_c.size() > 0 && func_c_failed.size() > 0)
        {
            LOG("Some CUDA functions are lost: {}\n", func_c_failed);
        }
        if (func_h.size() > 0 && func_h_failed.size() > 0)
        {
            LOG("Some HIP functions are lost: {}\n", func_h_failed);
        }
        if (func_c.size() == 0)
        {
            LOG("No CUDA libraries!\n");
        }
        if (func_h.size() == 0)
        {
            LOG("No HIP libraries!\n");
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
        "amdhip64",
        "rocblas",
        "miopen",
        "cccc-hip",
    };
    std::map<std::string, std::vector<std::string>> libs_used;
    std::vector<std::string> func_c, func_h, func_c_failed, func_h_failed;
};
static cuda_assist_class_t cuda_assist_class;

};    // namespace cccc

#endif
