#include "cuda_hack.h"

#if CUDNN_VERSION == 2000

cudnnStatus_t cudnnCreateOpTensorDescriptor(cudnnOpTensorDescriptor_t* des)
{
    *des = new cudnnOpTensorOp_t();
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnDestroyOpTensorDescriptor(cudnnOpTensorDescriptor_t des)
{
    delete des;
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnSetOpTensorDescriptor(cudnnOpTensorDescriptor_t des, cudnnOpTensorOp_t mode, cudnnDataType_t, int)
{
    *des = mode;
    return CUDNN_STATUS_SUCCESS;
}

//这个函数并没有完全复制cudnnOpTensor的功能，有一些忽略的参数
cudnnStatus_t cudnnOpTensor(cudnnHandle_t handle, const cudnnOpTensorDescriptor_t opTensorDesc,
    const void* alpha1, const cudnnTensorDescriptor_t aDesc, const void* A,
    const void* alpha2, const cudnnTensorDescriptor_t bDesc, const void* B,
    const void* beta, const cudnnTensorDescriptor_t cDesc, void* C)
{
    int n, c, h, w, n1, c1, h1, w1;
    cudnnDataType_t dt;
    cudnnGetTensor4dDescriptor(aDesc, &dt, &n, &c, &h, &w, &n1, &c1, &h1, &w1);
    int size = n * c * h * w;
    switch (*opTensorDesc)
    {
    case CUDNN_OP_TENSOR_ADD:
        cuda_add((real*)A, (real*)B, (real*)C, size, *(real*)alpha1, *(real*)alpha2);
        break;
    case CUDNN_OP_TENSOR_MUL:
        cuda_mul((real*)A, (real*)B, (real*)C, size, *(real*)alpha1, *(real*)beta);
        break;
    }
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnCreateActivationDescriptor(cudnnActivationDescriptor_t* des)
{
    *des = new cudnnActivationMode_t();
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnDestroyActivationDescriptor(cudnnActivationDescriptor_t des)
{
    delete des;
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnSetActivationDescriptor(cudnnActivationDescriptor_t des, cudnnActivationMode_t mode, int, int)
{
    *des = mode;
    return CUDNN_STATUS_SUCCESS;
}

#endif
