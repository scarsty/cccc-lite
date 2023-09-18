#pragma once
#include "types.h"
#include <atomic>
#include <memory>
#include <mutex>
#include <vector>

struct cudnnTensorStruct;
struct cudnnContext;

namespace cccc
{
class Rocblas;
class Cublas;

//使用设备的类型，主要决定数据位置
enum class UnitType
{
    CPU = 0,
    GPU,
};

enum ApiType
{
    API_UNKNOWN = 0,
    API_CUDA,    //NVIDIA
    API_HIP,     //AMD
};

//GPU架构
enum CudaArch
{
    ARCH_UNKNOWN = 0,
    ARCH_FERMI = 2,
    ARCH_KEPLER = 3,
    ARCH_MAXWELL = 5,
    ARCH_PASCAL = 6,
    //where is ARCH_VOLTA??
    ARCH_TURING = 7,
    ARCH_AMPERE = 8,
};

enum memcpyKind
{
    HostToHost = 0,     /**< Host   -> Host */
    HostToDevice = 1,   /**< Host   -> Device */
    DeviceToHost = 2,   /**< Device -> Host */
    DeviceToDevice = 3, /**< Device -> Device */
    Default = 4         /**< Direction of the transfer is inferred from the pointer values. Requires unified virtual addressing */
};

//该类包含一些cuda的基本参数，例如cublas和cudnn的handle
//此类型不能被随机创建，而仅能从已知的对象中选择一个
class DLL_EXPORT GpuControl
{
public:
    GpuControl();
    ~GpuControl();

    GpuControl(GpuControl&) = delete;
    GpuControl& operator=(GpuControl&) = delete;
    GpuControl(GpuControl&&) = default;
    GpuControl& operator=(GpuControl&&) = default;

private:
    static int device_count_;
    static int device_count_c_;
    static int device_count_h_;
    static std::vector<int> cuda_devices_turn_;
    static std::atomic<int> auto_choose_turn_;

public:
    static void checkDevices();
    static int getDeviceCount();
    static GpuControl* getCurrentCuda();
    static void setCurrentCuda(GpuControl* gpu);
    static void setUseCPU();

    static UnitType getGlobalCudaType();
    static void evaluateDevices();
    static std::vector<int> cudaDevicesTurn();

    void setAsCurrent();
    int getDeviceID() const { return gpu_id_; }
    void getFreeMemory(size_t& free, size_t& total) const;
    int init(int dev_id = -1);
    ApiType getApiType() { return api_type_; }
    void destroy();

    int malloc(void** p, size_t size);
    int free(void* p);
    int memcpy(void* dst, const void* src, size_t count, memcpyKind kind);

private:
    static int autoChooseId();

private:
    bool inited_ = false;
    int gpu_id_ = -1;    //从0开始，为总计数
    ApiType api_type_{ API_UNKNOWN };

    int api_id_ = -1;    //cuda或hip的id，从0开始

    CudaArch micro_arch_{ ARCH_UNKNOWN };

public:
    Cublas* cublas_ = nullptr;
    cudnnContext* cudnn_handle_ = nullptr;

    Rocblas* rocblas_ = nullptr;

    size_t memory_used_ = 0;

public:
    //设置张量数据，用于简化代码
    static void setTensorDesc4D(cudnnTensorStruct* tensor, int w, int h, int c, int n);
    static void setTensorDescND(cudnnTensorStruct* tensor, std::vector<int> dim);

    //设置激活函数，用于简化代码
    static void setActivationDesc(void* activation, int mode, double v);

    //private:
    //copy and modify from helper_cuda.h
    //Blas* selectBlas(CudaType mc) { return mc == mc_NoCuda ? (Blas*)(cblas) : (Blas*)(cublas); }

    //public:
    //    std::mutex& getMutex() { return *mutex_; }
    //
    //private:
    //    std::unique_ptr<std::mutex> mutex_ = std::make_unique<std::mutex>();
};

}    // namespace cccc