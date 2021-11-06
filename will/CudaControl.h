#pragma once
#include "cblas_real.h"
#include "cublas_real.h"
#include "types.h"
//#include "curand.h"
#include <vector>

namespace cccc
{

//数据是否存储于CUDA设备
enum class DeviceType
{
    CPU = 0,
    GPU,
};

//GPU架构
enum MicroArchitectureType
{
    ARCH_FERMI = 2,
    ARCH_KEPLER = 3,
    ARCH_MAXWELL = 5,
    ARCH_PASCAL = 6,
    //where is ARCH_VOLTA??
    ARCH_TURING = 7,
};

//该类包含一些cuda的基本参数，例如cublas和cudnn的handle
//此类型不能被随机创建，而仅能从已知的对象中选择一个
class CudaControl
{
public:
    CudaControl();
    ~CudaControl();

    //CudaControl(CudaControl&) = delete;
    //CudaControl& operator=(CudaControl&) = delete;

public:
    static CudaControl* select(int dev_id);
    static int checkDevices();
    static CudaControl* getCurrentCuda() { return select(getCurrentDevice()); }

private:
    bool inited_ = false;
    int cuda_id_;    //与vector中顺序相同
    MicroArchitectureType micro_arch_;

public:
    //cublasHandle_t cublas_handle_ = nullptr;
    Cublas* cublas_ = nullptr;
    //Cblas* cblas_ = nullptr;
    cudnnHandle_t cudnn_handle_ = nullptr;
    //curandGenerator_t curand_generator_ = nullptr;

    //这些是公用的，matrix类自带的是私用的
    cudnnTensorDescriptor_t tensor_desc_ = nullptr, tensor_desc2_ = nullptr;
    cudnnPoolingDescriptor_t pooling_desc_ = nullptr;
    cudnnConvolutionDescriptor_t convolution_desc_ = nullptr;
    cudnnFilterDescriptor_t filter_desc_ = nullptr;

    cudnnActivationDescriptor_t activation_desc_ = nullptr;
    cudnnOpTensorDescriptor_t op_tensor_desc_ = nullptr;

    cudnnRNNDescriptor_t rnn_desc_ = nullptr;
    cudnnDropoutDescriptor_t dropout_desc_ = nullptr;
    cudnnSpatialTransformerDescriptor_t spatial_transformer_desc_ = nullptr;
    cudnnLRNDescriptor_t lrn_desc_ = nullptr;

    size_t memory_used_ = 0;

private:
    static int device_count_;
    static std::vector<CudaControl> cuda_toolkit_vector_;    //排序按照nvml_id_
    static DeviceType global_device_type_;

    double state_score_ = 0;

private:
    int init(int use_cuda, int dev_id = -1);
    void destroy();

public:
    static void destroyAll();
    static DeviceType getGlobalCudaType();
    static void setGlobalCudaType(DeviceType ct);
    static int setDevice(int dev_id);    //仅为快速切换，无错误检查！
    static int getCurrentDevice();
    static int getBestDevice(int i = 0);

    //设置张量数据，用于简化代码
    static void setTensorDesc4D(cudnnTensorDescriptor_t tensor, int w, int h, int c, int n);
    static void setTensorDescND(cudnnTensorDescriptor_t tensor, std::vector<int> dim);

    //设置激活函数，用于简化代码
    static void setActivationDesc(cudnnActivationDescriptor_t activation, cudnnActivationMode_t mode, double v);

private:
    //copy and modify from helper_cuda.h
    static void evaluateDevices();
    //Blas* selectBlas(CudaType mc) { return mc == mc_NoCuda ? (Blas*)(cblas) : (Blas*)(cublas); }

public:
    const int getDeviceID() const { return cuda_id_; }
    void setThisDevice() const { setDevice(cuda_id_); }
};

}    // namespace cccc