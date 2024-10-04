#include "TensorDesc.h"
#include "Log.h"
#include "gpu_lib.h"
#include "types.h"
#include <algorithm>

namespace cccc
{

TensorDesc::TensorDesc(uint32_t flag)
{
#if ENABLE_CUDA
    if ((flag & 1) && cudnnCreateTensorDescriptor)
    {
        cudnnCreateTensorDescriptor(&cudnn_tensor_desc_);
    }
#endif
#if ENABLE_HIP
    if ((flag & 2) && miopenCreateTensorDescriptor)
    {
        miopenCreateTensorDescriptor(&miopen_tensor_desc_);
    }
#endif
}

TensorDesc::~TensorDesc()
{
#if ENABLE_CUDA
    if (cudnn_tensor_desc_)
    {
        cudnnDestroyTensorDescriptor(cudnn_tensor_desc_);
    }
#endif
#if ENABLE_HIP
    if (miopen_tensor_desc_)
    {
        miopenDestroyTensorDescriptor(miopen_tensor_desc_);
    }
#endif
}

void TensorDesc::setDesc4D(DataType data_type, int w, int h, int c, int n)
{
    if (n * c * h * w > 0)
    {
        if (cudnn_tensor_desc_)
        {

            auto r = cudnnSetTensor4dDescriptor(cudnn_tensor_desc_, CUDNN_TENSOR_NCHW, toCudnnDataType(data_type), n, c, h, w);
            if (r)
            {
                cccc::LOG_ERR("Set tensor failed!\n");
            }
        }
        if (miopen_tensor_desc_)
        {
            auto r = miopenSet4dTensorDescriptor(miopen_tensor_desc_, toMiopenDataType(data_type), n, c, h, w);
            if (r)
            {
                cccc::LOG_ERR("Set tensor failed!\n");
            }
        }
    }
}

void TensorDesc::setDescND(DataType data_type, std::vector<int> dim)
{
    std::vector<int> dim1, stride;
    int size = dim.size();

    dim1 = dim;
    stride.resize(size);
    std::reverse(dim1.begin(), dim1.end());
    int s = 1;
    for (int i = 0; i < size; i++)
    {
        stride[size - 1 - i] = s;
        s *= dim1[size - 1 - i];
    }
    if (cudnn_tensor_desc_)
    {
        if (size > CUDNN_DIM_MAX)
        {
            LOG_ERR("Error: wrong dimensions of tensor!\n");
            return;
        }
        cudnnSetTensorNdDescriptor(cudnn_tensor_desc_, toCudnnDataType(data_type), size, dim1.data(), stride.data());
    }
    if (miopen_tensor_desc_)
    {
        miopenSetTensorDescriptor(miopen_tensor_desc_, toMiopenDataType(data_type), size, dim1.data(), stride.data());
    }
}
#if ENABLE_CUDA
#define CREATE_CUDNN_DESC(type) \
    template <> \
    void OtherDesc::create<cudnn##type##_t>(void** p) \
    { \
        cudnnCreate##type((cudnn##type##_t*)p); \
    }
template <>
void OtherDesc::create<void*>(void** p)
{
}
CREATE_CUDNN_DESC(PoolingDescriptor)
#endif
#if ENABLE_HIP
#define CREATE_MIOPEN_DESC(type) \
    template <> \
    void OtherDesc::create<miopen##type##_t>(void** p) \
    { \
        miopenCreate##type((miopen##type##_t*)p); \
    }

CREATE_MIOPEN_DESC(ConvolutionDescriptor)
CREATE_MIOPEN_DESC(ActivationDescriptor)
CREATE_MIOPEN_DESC(PoolingDescriptor)
#endif
#if !ENABLE_CUDA && !ENABLE_HIP
template <>
void OtherDesc::create<void*>(void** p)
{
}
#endif
}    //namespace cccc