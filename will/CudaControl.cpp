#include "CudaControl.h"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <vector>

#if defined(_WIN32) && !defined(NO_CUDA)
#include <nvml.h>
#endif

#include "Log.h"

namespace cccc
{

int CudaControl::device_count_ = -1;    //-1表示没有初始化，该值正常的值应为非负
std::vector<CudaControl> CudaControl::cuda_toolkit_vector_;
DeviceType CudaControl::global_device_type_ = DeviceType::CPU;

CudaControl::CudaControl()
{
}

CudaControl::~CudaControl()
{
    //destroy();
}

CudaControl* CudaControl::select(int dev_id)
{
    if (global_device_type_ == DeviceType::CPU)
    {
        if (device_count_ < 0)
        {
            device_count_ = 0;
        }
        return nullptr;
    }
    else
    {
        //未初始化则尝试之
        if (device_count_ < 0)
        {
            checkDevices();
            if (device_count_ == 0)
            {
                return nullptr;
            }
        }

        if (dev_id < 0 || dev_id >= device_count_)
        {
            auto old_dev_id = dev_id;
            dev_id = getBestDevice();
            cudaDeviceProp device_prop;
            cudaGetDeviceProperties(&device_prop, cuda_toolkit_vector_[dev_id].cuda_id_);
            LOG("Device {} does not exist, automatically choose device {}\n",
                old_dev_id, dev_id);
        }

        auto& cuda_tk = cuda_toolkit_vector_[dev_id];

        if (!cuda_tk.inited_)
        {
            cuda_tk.init(1, dev_id);
        }
        cudaSetDevice(cuda_tk.cuda_id_);
        return &cuda_tk;
    }
}

//返回值为设备数
int CudaControl::checkDevices()
{
#ifdef NO_CUDA
    return 0;
#else
    if (device_count_ >= 0)
    {
        return device_count_;
    }
    int ver_r, ver_d;
    cudaRuntimeGetVersion(&ver_r);
    cudaDriverGetVersion(&ver_d);
    LOG("CUDA version: Runtime {}, Driver {} (R must <= D)\n", ver_r, ver_d);
    device_count_ = 0;
    if (cudaGetDeviceCount(&device_count_) != cudaSuccess || device_count_ <= 0)
    {
        return 0;
    }
    cuda_toolkit_vector_.clear();
    //for (int i = 0; i < device_count_; i++)
    //{
    //    cuda_toolkit_vector_.push_back(new CudaControl());
    //}
    cuda_toolkit_vector_.resize(device_count_);

    for (int i = 0; i < device_count_; i++)
    {
        cuda_toolkit_vector_[i].cuda_id_ = i;
    }

    evaluateDevices();
    return device_count_;
#endif
}

//返回0正常，其他情况都有问题
int CudaControl::init(int use_cuda, int dev_id /*= -1*/)
{
#ifndef NO_CUDA
    cudnnCreateTensorDescriptor(&tensor_desc_);
    cudnnCreateTensorDescriptor(&tensor_desc2_);
    cudnnCreateActivationDescriptor(&activation_desc_);
    cudnnCreateOpTensorDescriptor(&op_tensor_desc_);
    cudnnCreatePoolingDescriptor(&pooling_desc_);
    cudnnCreateConvolutionDescriptor(&convolution_desc_);
    cudnnCreateFilterDescriptor(&filter_desc_);
    cudnnCreateRNNDescriptor(&rnn_desc_);
    cudnnCreateDropoutDescriptor(&dropout_desc_);
    cudnnCreateSpatialTransformerDescriptor(&spatial_transformer_desc_);
    cudnnCreateLRNDescriptor(&lrn_desc_);

    cudaSetDevice(cuda_toolkit_vector_[dev_id].cuda_id_);
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
    //if (curandCreateGenerator(&toolkit_.curand_generator_, CURAND_RNG_PSEUDO_DEFAULT) != CURAND_STATUS_SUCCESS)
    //{
    //    LOG(stderr, "CURAND initialization error!\n");
    //    return 3;
    //}
    //else
    //{
    //    curandSetPseudoRandomGeneratorSeed(toolkit_.curand_generator_, 1234ULL);
    //}
    LOG("CUDA initialization on device {} succeed\n", cuda_id_);
    //LOG("CUBLAS Version = {}, CUDNN Version = {}\n", cublas_->get_version(), cudnnGetVersion());
    LOG("Float precision is {}\n", sizeof(real) * 8);
    inited_ = true;
    global_device_type_ = DeviceType::GPU;

    cudaDeviceProp device_prop;
    cudaGetDeviceProperties(&device_prop, cuda_toolkit_vector_[dev_id].cuda_id_);
    micro_arch_ = MicroArchitectureType(device_prop.major);
#endif

    return 0;
}

//虽然此处的设计是可以单独关闭其中一个设备，但是不要做这样的事！
//请清楚你在做什么！
void CudaControl::destroy()
{
#ifndef NO_CUDA
    cudnnDestroyTensorDescriptor(tensor_desc_);
    cudnnDestroyTensorDescriptor(tensor_desc2_);
    cudnnDestroyActivationDescriptor(activation_desc_);
    cudnnDestroyOpTensorDescriptor(op_tensor_desc_);
    cudnnDestroyPoolingDescriptor(pooling_desc_);
    cudnnDestroyConvolutionDescriptor(convolution_desc_);
    cudnnDestroyFilterDescriptor(filter_desc_);
    cudnnDestroyRNNDescriptor(rnn_desc_);
    cudnnDestroyDropoutDescriptor(dropout_desc_);
    cudnnDestroySpatialTransformerDescriptor(spatial_transformer_desc_);
    cudnnDestroyLRNDescriptor(lrn_desc_);
    if (cublas_)
    {
        cublas_->destroy();
    }
    if (cudnn_handle_)
    {
        cudnnDestroy(cudnn_handle_);
    }
    //if (cblas_) { delete cblas_; }
    inited_ = false;
#endif
}

//不建议主动调用这个函数，而由系统回收全局变量
void CudaControl::destroyAll()
{
    global_device_type_ = DeviceType::CPU;
    cuda_toolkit_vector_.clear();
    device_count_ = -1;
}

cccc::DeviceType CudaControl::getGlobalCudaType()
{
    return global_device_type_;
}

void CudaControl::setGlobalCudaType(DeviceType ct)
{
    global_device_type_ = ct;
    checkDevices();
    if (device_count_ <= 0)
    {
        global_device_type_ = DeviceType::CPU;
    }
}

int CudaControl::setDevice(int dev_id)
{
    if (dev_id >= 0 && dev_id < device_count_)
    {
        cudaSetDevice(cuda_toolkit_vector_[dev_id].cuda_id_);
        return dev_id;
    }
    return -1;
}

int CudaControl::getCurrentDevice()
{
    int device = 0;
    cudaGetDevice(&device);
    for (int i = 0; i < device_count_; i++)
    {
        if (device == cuda_toolkit_vector_[i].cuda_id_)
        {
            return i;
        }
    }
    return 0;
}

int CudaControl::getBestDevice(int i /*= 0*/)
{
    if (device_count_ <= 1 || i < 0 || i >= device_count_)
    {
        return 0;
    }
    struct DeviceState
    {
        int dev_id;
        double score;
    };
    std::vector<DeviceState> best_device(device_count_);
    for (int i = 0; i < device_count_; i++)
    {
        best_device[i] = { i, cuda_toolkit_vector_[i].state_score_ };
    }

    std::sort(best_device.begin(), best_device.end(), [](DeviceState& l, DeviceState& r)
        { return l.score > r.score; });

    //冒泡排序，因为数量较少，不考虑效率了
    //for (int i = 0; i < device_count_ - 1; i++)
    //{
    //    for (int j = 0; j < device_count_ - 1 - i; j++)
    //    {
    //        if (state[j] < state[j + 1])
    //        {
    //            std::swap(best_device[j], best_device[j + 1]);
    //        }
    //    }
    //}
    return best_device[i].dev_id;
}

void CudaControl::setTensorDesc4D(cudnnTensorDescriptor_t tensor, int w, int h, int c, int n)
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

void CudaControl::setTensorDescND(cudnnTensorDescriptor_t tensor, std::vector<int> dim)
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

void CudaControl::setActivationDesc(cudnnActivationDescriptor_t activation, cudnnActivationMode_t mode, double v)
{
    if (activation)
    {
        cudnnSetActivationDescriptor(activation, mode, CUDNN_NOT_PROPAGATE_NAN, v);
    }
}

// 为每个设备评分
void CudaControl::evaluateDevices()
{
#ifndef NO_CUDA
    auto getSPcores = [](cudaDeviceProp& devive_prop) -> int
    {
        int mp = devive_prop.multiProcessorCount;
        int major = devive_prop.major;
        int minor = devive_prop.minor;

        typedef struct
        {
            int SM;    // 0xMm (hexidecimal notation), M = SM Major version,
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
            "MapSMtoCores for SM {}.{} is undefined."
            "  Default to use %d Cores/SM\n",
            major, minor, nGpuArchCoresPerSM[index - 1].Cores);

        return mp * nGpuArchCoresPerSM[index - 1].Cores;
    };

    if (device_count_ <= 0)
    {
        return;
    }

    double best_state = -1e8;
    std::vector<int> pci(device_count_);
    int best_i = 0;
    int nvml_state = 1;
#ifdef NVML_API_VERSION
    nvml_state = nvmlInit();
#endif
    for (int i = 0; i < device_count_; i++)
    {
        auto& state = cuda_toolkit_vector_[i].state_score_;
        state = 0;
        pci[i] = i;

        auto& ct = cuda_toolkit_vector_[i];
        cudaDeviceProp device_prop;
        cudaGetDeviceProperties(&device_prop, ct.cuda_id_);
        setDevice(i);
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
            LOG("Device {} ({}): {}, {:7.2f} GFLOPS, {:7.2f} MB ({:g}%) free\n",
                ct.cuda_id_, pci_info, device_prop.name, flops / 1e9, free / 1048576.0, 100.0 * free / total);
            state = flops / 1e12 + free / 1e9;
            pci[i] = device_prop.pciBusID;
            if (state > best_state)
            {
                best_state = state;
                best_i = i;
            }
        }
    }
#ifdef NVML_API_VERSION
    nvmlShutdown();
#endif
    //这里需要重新计算，考虑与最好设备的距离
    for (int i = 0; i < device_count_; i++)
    {
        cuda_toolkit_vector_[i].state_score_ -= std::abs((pci[i] - pci[best_i])) / 50.0;
    }
#endif
}

}    // namespace cccc