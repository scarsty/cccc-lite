#include "GpuControl.h"
#include "Log.h"
#include "cblas_real.h"
#include "gpu_lib.h"
#include "types.h"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <vector>

namespace cccc
{

thread_local GpuControl* current_gpu_ = nullptr;    //当前使用的cuda设备

//-1表示没有初始化，该值正常的值应为非负
int GpuControl::device_count_ = -1;
int GpuControl::device_count_c_ = 0;
int GpuControl::device_count_h_ = 0;
std::vector<int> GpuControl::cuda_devices_turn_;        //cuda设备按优劣的排序
std::atomic<int> GpuControl::auto_choose_turn_{ 0 };    //若自动选择，当前使用哪个设备

GpuControl::GpuControl()
{
}

GpuControl::~GpuControl()
{
    destroy();
}

//返回0正常，其他情况都有问题
int GpuControl::init(int dev_id /*= -1*/)
{
    if (inited_)
    {
        return 0;
    }
    if (device_count_ <= 0)
    {
        return 0;
    }

    if (dev_id < 0)
    {
        dev_id = autoChooseId();
    }
    //依据dev_id是否大于等于device_count_c_，来判断是cuda还是hip

    if (dev_id < device_count_c_)
    {
        api_id_ = dev_id;
        api_type_ = API_CUDA;
    }
    else
    {
        api_id_ = dev_id - device_count_c_;
        api_type_ = API_HIP;
    }
    gpu_id_ = dev_id;

#ifndef NO_CUDA
    if (api_type_ == API_CUDA)
    {
        cudaSetDevice(api_id_);
        cublas_ = new Cublas();
        if (cublas_->init() != CUBLAS_STATUS_SUCCESS)
        {
            LOG(stderr, "CUBLAS initialization error on device {}!\n", dev_id);
            return 1;
        }
        if (cudnnCreate(&cudnn_handle_) != CUDNN_STATUS_SUCCESS)
        {
            LOG(stderr, "CUDNN initialization error on device {}!\n", dev_id);
            return 2;
        }
        LOG("CUDA initialization on device {} succeed\n", gpu_id_);
        //LOG("CUBLAS Version = {}, CUDNN Version = {}\n", cublas_->get_version(), cudnnGetVersion());
        LOG("Float precision is {}\n", sizeof(real) * 8);
        inited_ = true;

        cudaDeviceProp device_prop;
        cudaGetDeviceProperties(&device_prop, api_id_);
        micro_arch_ = CudaArch(device_prop.major);
    }
    else if (api_type_ == API_HIP)
    {
        hipSetDevice(api_id_);
        rocblas_ = new Rocblas();
        if (rocblas_->init())
        {
            LOG(stderr, "ROCBLAS initialization error on device {}!\n", dev_id);
            return 1;
        }
    }
#endif
    setAsCurrent();
    return 0;
}

//虽然此处的设计是可以单独关闭其中一个设备，但是不要做这样的事！
//请清楚你在做什么！
void GpuControl::destroy()
{
#ifndef NO_CUDA
    delete cublas_;
    if (cudnn_handle_)
    {
        cudnnDestroy(cudnn_handle_);
    }
#endif
#ifndef NO_HIP
    delete rocblas_;
#endif
    inited_ = false;
}

int GpuControl::malloc(void** p, size_t size)
{
    if (api_type_ == API_CUDA)
    {
        return cudaMalloc(p, size);
    }
    else if (api_type_ == API_HIP)
    {
        return hipMalloc(p, size);
    }
    return 1;
}

int GpuControl::free(void* p)
{
    if (api_type_ == API_CUDA)
    {
        return cudaFree(p);
    }
    else if (api_type_ == API_HIP)
    {
        return hipFree(p);
    }
    return 1;
}

int GpuControl::memcpy(void* dst, const void* src, size_t count, memcpyKind kind)
{
    if (api_type_ == API_CUDA)
    {
        return cudaMemcpy(dst, src, count, cudaMemcpyKind(kind));
    }
    else if (api_type_ == API_HIP)
    {
        return hipMemcpy(dst, src, count, hipMemcpyKind(kind));
    }
    return 1;
}

void GpuControl::checkDevices()
{
#ifdef NO_CUDA
    return;
#else
    if (device_count_ >= 0)
    {
        return;
    }

    int ver_r = 0, ver_d = 0;
    device_count_c_ = 0;
    if (cudaGetDeviceCount)
    {
        cudaGetDeviceCount(&device_count_c_);
        if (cudaRuntimeGetVersion(&ver_r) != cudaSuccess)
        {
            LOG("Get runtime state failed, please check hardware or driver!\n");
        }
        if (cudaDriverGetVersion(&ver_d) != cudaSuccess)
        {
            LOG("Get driver state failed, please check hardware or driver!\n");
        }
        LOG("CUDA version: Runtime {}, Driver {} (R must <= D)\n", ver_r, ver_d);
    }

    ver_r = 0;
    ver_d = 0;
    device_count_h_ = 0;
    if (hipGetDeviceCount)
    {
        hipGetDeviceCount(&device_count_h_);
        if (hipRuntimeGetVersion(&ver_r) != hipSuccess)
        {
            LOG("Get runtime state failed, please check hardware or driver!\n");
        }
        if (hipDriverGetVersion(&ver_d) != hipSuccess)
        {
            LOG("Get driver state failed, please check hardware or driver!\n");
        }
        LOG("HIP version: Runtime {}, Driver {} (R must <= D)\n", ver_r, ver_d);
    }

    device_count_ = device_count_c_ + device_count_h_;
    if (device_count_ > 0)
    {
        LOG("Found {} GPU device(s), {} CUDA, {} HIP\n", device_count_, device_count_c_, device_count_h_);
    }
    else
    {
        LOG("Error: No GPU devices!!\n");
    }
    evaluateDevices();

#endif
}

int GpuControl::getDeviceCount()
{
    return device_count_;
}

GpuControl* GpuControl::getCurrentCuda()
{
    return current_gpu_;
}

static void setCurrentCuda(GpuControl* gpu)
{
    if (gpu)
    {
        gpu->setAsCurrent();
    }
    current_gpu_ = nullptr;
}

void GpuControl::setUseCPU()
{
    current_gpu_ = nullptr;
}

void GpuControl::setAsCurrent()
{
    if (device_count_ <= 0)
    {
        current_gpu_ = nullptr;
        return;
    }
    current_gpu_ = this;
    if (api_type_ == API_CUDA)
    {
        cudaSetDevice(api_id_);
    }
    else if (api_type_ == API_HIP)
    {
        hipSetDevice(api_id_);
    }
}

cccc::UnitType GpuControl::getGlobalCudaType()
{
    if (current_gpu_)
    {
        return UnitType::GPU;
    }
    return UnitType::CPU;
}

std::vector<int> GpuControl::cudaDevicesTurn()
{
    return cuda_devices_turn_;
}

void GpuControl::getFreeMemory(size_t& free, size_t& total) const
{
    if (api_type_ == API_CUDA)
    {
#ifdef NVML_API_VERSION
        nvmlDevice_t nvml_device;
        nvmlDeviceGetHandleByIndex(api_id_, &nvml_device);
        nvmlMemory_t nvml_memory;
        nvmlDeviceGetMemoryInfo(nvml_device, &nvml_memory);
        free = nvml_memory.free;
        total = nvml_memory.total;
#else
        cudaSetDevice(api_id_);
        cudaMemGetInfo(&free, &total);
#endif
    }
    else if (api_type_ == API_HIP)
    {
        hipSetDevice(api_id_);
        hipMemGetInfo(&free, &total);
    }
}

int GpuControl::autoChooseId()
{
    int turn = auto_choose_turn_;
    turn++;
    if (turn >= device_count_)
    {
        turn = 0;
    }
    auto_choose_turn_ = turn;
    return cuda_devices_turn_[turn];
}

// 为每个设备评分
void GpuControl::evaluateDevices()
{
    auto getSPcores = [](cudaDeviceProp& devive_prop) -> int
    {
        int mp = devive_prop.multiProcessorCount;
        int major = devive_prop.major;
        int minor = devive_prop.minor;

        typedef struct
        {
            int SM;    // 0xMm (hexadecimal notation), M = SM Major version,
            // and m = SM minor version
            int Cores;
        } sSMtoCores;

        sSMtoCores nGpuArchCoresPerSM[] = {
            { 0x30, 192 },
            { 0x32, 192 },
            { 0x35, 192 },
            { 0x37, 192 },
            { 0x50, 128 },
            { 0x52, 128 },
            { 0x53, 128 },
            { 0x60, 64 },
            { 0x61, 128 },
            { 0x62, 128 },
            { 0x70, 64 },
            { 0x72, 64 },
            { 0x75, 64 },
            { 0x80, 64 },
            { 0x86, 128 },
            { -1, -1 }
        };

        int index = 0;

        while (nGpuArchCoresPerSM[index].SM != -1)
        {
            if (nGpuArchCoresPerSM[index].SM == ((major << 4) + minor))
            {
                return mp * nGpuArchCoresPerSM[index].Cores;
            }
            index++;
        }

        // If we don't find the values, we default use the previous one
        // to run properly
        LOG(stderr,
            "MapSMtoCores for SM {}.{} is undefined, default to use {} Cores/SM\n",
            major, minor, nGpuArchCoresPerSM[index - 1].Cores);

        return mp * nGpuArchCoresPerSM[index - 1].Cores;
    };

    if (device_count_ <= 0)
    {
        return;
    }

    struct Info
    {
        int id;
        double state_score = 0;
        int pci;
    };

    double best_state = -1e8;
    std::vector<Info> infos(device_count_);
    int best_i = 0;
    int nvml_state = 1;
#ifdef NVML_API_VERSION
    nvml_state = nvmlInit();
#endif
    for (int i = 0; i < device_count_c_; i++)
    {
        infos[i].id = i;
        auto& state = infos[i].state_score;
        state = 0;
        infos[i].pci = i;

        cudaDeviceProp device_prop;
        cudaGetDeviceProperties(&device_prop, i);
        cudaSetDevice(i);
        if (device_prop.computeMode != cudaComputeModeProhibited)
        {
            //通常情况系数是2，800M系列是1，但是不管了
            double flops = 2.0e3 * device_prop.clockRate * getSPcores(device_prop);
            char pci_info[128];
            cudaDeviceGetPCIBusId(pci_info, 128, i);
            size_t free, total;
#ifdef NVML_API_VERSION
            nvmlDevice_t nvml_device;
            nvmlDeviceGetHandleByPciBusId(pci_info, &nvml_device);
            nvmlMemory_t nvml_memory;
            nvmlDeviceGetMemoryInfo(nvml_device, &nvml_memory);
            free = nvml_memory.free;
            total = nvml_memory.total;
            unsigned int temperature;
            nvmlDeviceGetTemperature(nvml_device, NVML_TEMPERATURE_GPU, &temperature);
#endif
            if (nvml_state)
            {
                cudaMemGetInfo(&free, &total);
            }
            LOG("Device {} ({}): {}, {:7.5f} TFLOPS, {:7.2f}/{:7.2f} MB free/total\n",
                i, pci_info, device_prop.name, flops / 1e12, free / 1048576.0, total / 1048576.0);
            state = flops / 1e12 + free / 1e9;
            infos[i].pci = device_prop.pciBusID;
            if (state > best_state)
            {
                best_state = state;
                best_i = i;
            }
        }
    }

    for (int i = 0; i < device_count_h_; i++)
    {
        infos[i].id = i;
        auto& state = infos[i].state_score;
        state = 0;
        infos[i].pci = i;

        hipDeviceProp_t device_prop;
        hipGetDeviceProperties(&device_prop, i);
        hipSetDevice(i);
        double flops = 4.0e3 * device_prop.clockRate * device_prop.multiProcessorCount * 128;
        char pci_info[128];
        hipDeviceGetPCIBusId(pci_info, 128, i);
        size_t free, total;

        hipMemGetInfo(&free, &total);

        LOG("Device {} ({}): {}, {:7.5f} TFLOPS, {:7.2f}/{:7.2f} MB free/total\n",
            i + device_count_c_, pci_info, device_prop.name, flops / 1e12, free / 1048576.0, total / 1048576.0);
        state = flops / 1e12 + free / 1e9;
        infos[i].pci = device_prop.pciBusID;
    }
    //#ifdef NVML_API_VERSION
    //    nvmlShutdown();
    //#endif
    //这里需要重新计算，考虑与最好设备的距离
    for (int i = 0; i < device_count_; i++)
    {
        infos[i].state_score -= std::abs((infos[i].pci - infos[best_i].pci)) / 50.0;
    }

    std::sort(infos.begin(), infos.end(), [](const Info& l, const Info& r)
        {
            return l.state_score > r.state_score;
        });
    auto_choose_turn_ = best_i;
    cuda_devices_turn_.resize(device_count_);
    for (int i = 0; i < device_count_; i++)
    {
        cuda_devices_turn_[i] = infos[i].id;
    }
}

void GpuControl::setTensorDesc4D(cudnnTensorStruct* tensor, int w, int h, int c, int n)
{
    if (tensor && n * c * h * w > 0)
    {
        auto r = cudnnSetTensor4dDescriptor(tensor, CUDNN_TENSOR_NCHW, MYCUDNN_DATA_REAL, n, c, h, w);
        if (r)
        {
            LOG(stderr, "Set tensor failed!\n");
        }
    }
}

void GpuControl::setTensorDescND(cudnnTensorStruct* tensor, std::vector<int> dim)
{
    if (tensor)
    {
        std::vector<int> dim1, stride;
        int size = dim.size();
        if (size > CUDNN_DIM_MAX)
        {
            LOG(stderr, "Error: wrong dimensions of tensor!\n");
            return;
        }
        dim1 = dim;
        stride.resize(size);
        std::reverse(dim1.begin(), dim1.end());
        int s = 1;
        for (int i = 0; i < size; i++)
        {
            stride[size - 1 - i] = s;
            s *= dim1[size - 1 - i];
        }
        cudnnSetTensorNdDescriptor(tensor, MYCUDNN_DATA_REAL, size, dim1.data(), stride.data());
        //cudnnSetTensorNdDescriptorEx(tensor, CUDNN_TENSOR_NCHW, MYCUDNN_DATA_REAL, size, dim1.data());

        //以下测试效果用
        //cudnnDataType_t t;
        //int n;
        //int d1[8];
        //int s1[8];
        //cudnnGetTensorNdDescriptor(tensor, 8, &t, &n, d1, s1);
    }
}

void GpuControl::setActivationDesc(void* activation, int mode, double v)
{
    if (activation)
    {
        cudnnSetActivationDescriptor((cudnnActivationDescriptor_t)activation, (cudnnActivationMode_t)mode, CUDNN_NOT_PROPAGATE_NAN, v);
    }
}

}    // namespace cccc