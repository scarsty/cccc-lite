#include "CudaControl.h"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>

#ifdef _WIN32
#include <nvml.h>
#endif

namespace woco
{

//static int tensor_count = 0;
TensorDescWrapper::TensorDescWrapper()
{
    cudnnCreateTensorDescriptor(&tensor_desc_);
    //tensor_count++;
}

TensorDescWrapper::TensorDescWrapper(const TensorDescWrapper& o) : TensorDescWrapper()
{
    cudnnDataType_t t;
    int n;
    int d1[8];
    int s1[8];
    cudnnGetTensorNdDescriptor(o.tensor_desc_, 8, &t, &n, d1, s1);
    cudnnSetTensorNdDescriptor(tensor_desc_, t, n, d1, s1);
}

TensorDescWrapper::TensorDescWrapper(TensorDescWrapper&& o)
{
    tensor_desc_ = o.tensor_desc_;
    o.tensor_desc_ = nullptr;
}

TensorDescWrapper& TensorDescWrapper::operator=(const TensorDescWrapper& o)
{
    if (this != &o)
    {
        cudnnDataType_t t;
        int n;
        int d1[8];
        int s1[8];
        cudnnGetTensorNdDescriptor(o.tensor_desc_, 8, &t, &n, d1, s1);
        cudnnSetTensorNdDescriptor(tensor_desc_, t, n, d1, s1);
    }
    return *this;
}

//TensorDescWrapper& TensorDescWrapper::operator=(TensorDescWrapper o)
//{
//    std::swap(tensor_desc_, o.tensor_desc_);
//    return *this;
//}

TensorDescWrapper& TensorDescWrapper::operator=(TensorDescWrapper&& o)
{
    if (this != &o)
    {
        std::swap(tensor_desc_, o.tensor_desc_);
    }
    return *this;
}

TensorDescWrapper::~TensorDescWrapper()
{
    if (tensor_desc_)
    {
        cudnnDestroyTensorDescriptor(tensor_desc_);
        //tensor_count--;
    }
    //printf("%d tensors\n", int(tensor_count));
}

int CudaControl::device_count_ = -1;    //-1表示没有初始化，该值正常的值应为非负
std::vector<CudaControl*> CudaControl::cuda_toolkit_vector_;
DeviceType CudaControl::global_device_type_ = DeviceType::CPU;

CudaControl::CudaControl()
{
}

CudaControl::~CudaControl()
{
    destroy();
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
            dev_id = getBestDevice();
            cudaDeviceProp device_prop;
            cudaGetDeviceProperties(&device_prop, cuda_toolkit_vector_[dev_id]->cuda_id_);
            fprintf(stdout, "Auto choose the best device %d: \"%s\" with compute capability %d.%d\n",
                dev_id, device_prop.name, device_prop.major, device_prop.minor);
        }

        auto cuda_tk = cuda_toolkit_vector_[dev_id];

        if (!cuda_tk->inited_)
        {
            cuda_tk->init(1, dev_id);
        }

        return cuda_tk;
    }
}

//返回值为设备数
int CudaControl::checkDevices()
{
    if (device_count_ >= 0)
    {
        return device_count_;
    }
    device_count_ = 0;
    if (cudaGetDeviceCount(&device_count_) != cudaSuccess || device_count_ <= 0)
    {
        return 0;
    }
    cuda_toolkit_vector_.clear();
    for (int i = 0; i < device_count_; i++)
    {
        cuda_toolkit_vector_.push_back(new CudaControl());
    }

    for (int i = 0; i < device_count_; i++)
    {
        cuda_toolkit_vector_[i]->cuda_id_ = i;
    }

    findBestDevice();
    return device_count_;
}

//返回0正常，其他情况都有问题
int CudaControl::init(int use_cuda, int dev_id /*= -1*/)
{
    //cblas_ = new Cblas();
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

    cudaSetDevice(cuda_toolkit_vector_[dev_id]->cuda_id_);
    cublas_ = new Cublas();
    if (cublas_->init() != CUBLAS_STATUS_SUCCESS)
    {
        fprintf(stderr, "CUBLAS initialization error on device %d!\n", dev_id);
        return 1;
    }
    if (cudnnCreate(&cudnn_handle_) != CUDNN_STATUS_SUCCESS)
    {
        fprintf(stderr, "CUDNN initialization error on device %d!\n", dev_id);
        return 2;
    }
    //if (curandCreateGenerator(&toolkit_.curand_generator_, CURAND_RNG_PSEUDO_DEFAULT) != CURAND_STATUS_SUCCESS)
    //{
    //    fprintf(stderr, "CURAND initialization error!\n");
    //    return 3;
    //}
    //else
    //{
    //    curandSetPseudoRandomGeneratorSeed(toolkit_.curand_generator_, 1234ULL);
    //}
    fprintf(stdout, "CUDA initialization on device %d succeed\n", cuda_id_);
    fprintf(stdout, "Float precision is %ld\n", sizeof(real) * 8);
    inited_ = true;
    global_device_type_ = DeviceType::GPU;

    cudaDeviceProp device_prop;
    cudaGetDeviceProperties(&device_prop, cuda_toolkit_vector_[dev_id]->cuda_id_);
    micro_arch_ = MicroArchitectureType(device_prop.major);

    return 0;
}

//虽然此处的设计是可以单独关闭其中一个设备，但是不要做这样的事！
//请清楚你在做什么！
void CudaControl::destroy()
{
#ifndef _NO_CUDA
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
#endif
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
}

void CudaControl::destroyAll()
{
    auto vec = cuda_toolkit_vector_;
    for (auto c : vec)
    {
        delete c;
    }
    global_device_type_ = DeviceType::CPU;
    cuda_toolkit_vector_.clear();
    device_count_ = -1;
}

woco::DeviceType CudaControl::getGlobalCudaType()
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
        cudaSetDevice(cuda_toolkit_vector_[dev_id]->cuda_id_);
        return dev_id;
    }
    return -1;
}

void CudaControl::setDevice()
{
    setDevice(cuda_id_);
}

int CudaControl::getCurrentDevice()
{
    int device = 0;
    cudaGetDevice(&device);
    for (int i = 0; i < device_count_; i++)
    {
        if (device == cuda_toolkit_vector_[i]->cuda_id_)
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
        best_device[i] = { i, cuda_toolkit_vector_[i]->state_score_ };
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
    if (tensor)
    {
        cudnnSetTensor4dDescriptor(tensor, CUDNN_TENSOR_NCHW, MYCUDNN_DATA_REAL, n, c, h, w);
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
            fprintf(stderr, "Error: wrong dimensions of tensor!\n");
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

// This function returns the best GPU (with maximum GFLOPS)
void CudaControl::findBestDevice()
{
    auto getSPcores = [](cudaDeviceProp& devive_prop) -> int
    {
        int cores = 0;
        int mp = devive_prop.multiProcessorCount;
        switch (devive_prop.major)
        {
        case 2:    // Fermi
            if (devive_prop.minor == 1)
            {
                cores = mp * 48;
            }
            else
            {
                cores = mp * 32;
            }
            break;
        case 3:    // Kepler
            cores = mp * 192;
            break;
        case 5:    // Maxwell
            cores = mp * 128;
            break;
        case 6:    // Pascal
            if (devive_prop.minor == 1)
            {
                cores = mp * 128;
            }
            else if (devive_prop.minor == 0)
            {
                cores = mp * 64;
            }
            else
            {
                fprintf(stderr, "Unknown device type\n");
            }
            break;
        case 7:    // Turing
            cores = mp * 64;
            break;
        default:
            fprintf(stderr, "Unknown device type\n");
            break;
        }
        return cores;
    };

    if (device_count_ <= 0)
    {
        return;
    }

    double best_state = -1e8;
    std::vector<int> pci(device_count_);
    int best_i = 0;
#ifdef NVML_API_VERSION
    nvmlInit();
#endif
    for (int i = 0; i < device_count_; i++)
    {
        auto& state = cuda_toolkit_vector_[i]->state_score_;
        state = 0;
        pci[i] = i;

        auto ct = cuda_toolkit_vector_[i];
        cudaDeviceProp device_prop;
        cudaGetDeviceProperties(&device_prop, ct->cuda_id_);
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
#else
            cudaMemGetInfo(&free, &total);
#endif
            fprintf(stdout, "Device %d (%s): %s, %7.2f GFLOPS (single), free memory %7.2f MB (%g%%)\n",
                ct->cuda_id_, pci_info, device_prop.name, flops / 1e9, free / 1048576.0, 100.0 * free / total);
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
        cuda_toolkit_vector_[i]->state_score_ -= std::abs((pci[i] - pci[best_i])) / 50.0;
    }
}

}    // namespace woco