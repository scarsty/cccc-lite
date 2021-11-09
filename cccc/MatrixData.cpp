#include "MatrixData.h"
#include "Matrix.h"
#include "Log.h"

namespace cccc
{
//此处有问题，目前来看只能在同设备中resize

real* MatrixData::resize(int64_t size, bool reserve_data, bool force)
{
    if (size <= occupy_data_size_ && !force)
    {
        return data_;
    }
    real* new_data = nullptr;
    auto device_type = DeviceType::GPU;
    if (!reserve_data)    //数据不需保留可以先行释放，可以节省显存的使用，否则峰值占用是新旧尺寸之和
    {
        free();
    }
    if (cuda_)
    {
        cuda_->setThisDevice();
        if (cudaMalloc((void**)&new_data, size * sizeof(real)) == cudaSuccess)
        {
            cuda_->memory_used_ += size * sizeof(real);
            LOG(2, "Device {} MALLOC {:g}, memory used {:g}!\n", cuda_->getDeviceID(), 1.0 * size * sizeof(real), 1.0 * cuda_->memory_used_);
            //if (size * sizeof(real) > 3e8)
            //{
            //    LOG(2, "Very big!\n");
            //}
        }
        else
        {
            LOG(2, "Device {} FAIL TO MALLOC {:g}!\n", cuda_->getDeviceID(), 1.0 * size * sizeof(real));
        }
    }
    else
    {
        device_type = DeviceType::CPU;
        new_data = new real[size];
    }
    if (reserve_data)
    {
        copy(device_type, data_, device_type, new_data, std::min(size, occupy_data_size_));
        free();
    }
    occupy_data_size_ = size;
    return data_ = new_data;
}

void MatrixData::free()
{
    if (data_ == nullptr)
    {
        occupy_data_size_ = 0;
        return;
    }
    if (cuda_)
    {
        auto current_gpu = CudaControl::getCurrentDevice();
        if (current_gpu != cuda_->getDeviceID())
        {
            cuda_->setThisDevice();
        }
        auto status = cudaFree(data_);
        if (status == cudaSuccess)
        {
            cuda_->memory_used_ -= occupy_data_size_ * sizeof(real);
            LOG(2, "Device {} FREE {:g}, memory used {:g}!\n", cuda_->getDeviceID(), 1.0 * occupy_data_size_ * sizeof(real), 1.0 * cuda_->memory_used_);
        }
        else
        {
            LOG(2, "Device {} FAIL TO FREE {:g}!\n", cuda_->getDeviceID(), 1.0 * occupy_data_size_ * sizeof(real));
        }
        if (current_gpu != cuda_->getDeviceID())
        {
            CudaControl::setDevice(current_gpu);
        }
    }
    else
    {
        delete[] data_;
    }
    occupy_data_size_ = 0;
    data_ = nullptr;
}

int64_t MatrixData::copy(DeviceType dt_src, const real* src, DeviceType dt_dst, real* dst, int64_t size)
{
    if (src == nullptr || dst == nullptr || src == dst)
    {
        return 0;
    }
    int64_t size_in_byte = size * sizeof(real);
    cudaError state = cudaSuccess;
    if (dt_dst == DeviceType::GPU && dt_src == DeviceType::GPU)
    {
        state = cudaMemcpy(dst, src, size_in_byte, cudaMemcpyDeviceToDevice);
    }
    else if (dt_dst == DeviceType::GPU && dt_src == DeviceType::CPU)
    {
        state = cudaMemcpy(dst, src, size_in_byte, cudaMemcpyHostToDevice);
    }
    else if (dt_dst == DeviceType::CPU && dt_src == DeviceType::GPU)
    {
        state = cudaMemcpy(dst, src, size_in_byte, cudaMemcpyDeviceToHost);
    }
    else
    {
        memcpy(dst, src, size_in_byte);
    }
    if (state != cudaSuccess)
    {
        LOG(stderr, "Error: cudaMemcpy failed with error code is {}, size in byte is {:g}!\n", state, 1.0 * size_in_byte);
    }
    return size;
}

}    // namespace cccc