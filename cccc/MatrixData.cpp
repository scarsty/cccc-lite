#include "MatrixData.h"
#include "Log.h"
#include "Matrix.h"
#include "gpu_lib.h"

namespace cccc
{
//此处有问题，目前来看只能在同设备中resize

void* MatrixData::resize(int64_t size, DataType data_type, bool reserve_data, bool force)
{
    data_type_ = data_type;
    if (size <= occupy_data_size_ && !force)
    {
        return data_;
    }
    float* new_data = nullptr;
    auto device_type = UnitType::GPU;
    if (!reserve_data)    //数据不需保留可以先行释放，可以节省显存的使用，否则峰值占用是新旧尺寸之和
    {
        release();
    }
    if (gpu_)
    {
        gpu_->setAsCurrent();
        if (gpu_->malloc((void**)&new_data, size * getDataTypeSize(data_type_)) == 0)
        {
            gpu_->memory_used_ += size * getDataTypeSize(data_type_);
            //LOG("Device {} MALLOC {:g}, memory used {:g}!\n", cuda_->getDeviceID(), 1.0 * size * sizeof(real), 1.0 * cuda_->memory_used_);
            //if (size * sizeof(real) > 3e8)
            //{
            //    LOG(2, "Very big!\n");
            //}
        }
        else
        {
            LOG("Device {} FAIL TO MALLOC {:g}!\n", gpu_->getDeviceID(), 1.0 * size * getDataTypeSize(data_type_));
        }
    }
    else
    {
        device_type = UnitType::CPU;
        new_data = new float[size];
    }
    if (reserve_data)
    {
        copy(getApiType(), data_, getApiType(), new_data, std::min(size, occupy_data_size_), data_type_);
        release();
    }
    occupy_data_size_ = size;
    return data_ = new_data;
}

void MatrixData::release()
{
    if (data_ == nullptr)
    {
        occupy_data_size_ = 0;
        return;
    }
    if (gpu_)
    {
        auto current_gpu = GpuControl::getCurrentCuda();
        if (current_gpu != gpu_)
        {
            gpu_->setAsCurrent();
        }
        auto status = gpu_->free(data_);
        if (status == 0)
        {
            gpu_->memory_used_ -= occupy_data_size_ * getDataTypeSize(data_type_);
            //LOG("Device {} FREE {:g}, memory used {:g}!\n", gpu_->getDeviceID(), 1.0 * occupy_data_size_ * getDataTypeSize(data_type_), 1.0 * gpu_->memory_used_);
        }
        else
        {
            //LOG("Device {} FAIL TO FREE {:g}!\n", gpu_->getDeviceID(), 1.0 * occupy_data_size_ * getDataTypeSize(data_type_));
        }
        if (current_gpu != gpu_)
        {
            current_gpu->setAsCurrent();
        }
    }
    else
    {
        delete[] data_;
    }
    occupy_data_size_ = 0;
    data_ = nullptr;
}

int64_t MatrixData::copy(ApiType dt_src, const void* src, ApiType dt_dst, void* dst, int64_t size, DataType dt)
{
    return copyByByte(dt_src, src, dt_dst, dst, size * getDataTypeSize(dt));
}

int64_t MatrixData::copyByByte(ApiType dt_src, const void* src, ApiType dt_dst, void* dst, int64_t size_in_byte)
{
    if (src == nullptr || dst == nullptr || src == dst)
    {
        return 0;
    }
    int state = 0;
    //cuda
    if (dt_dst == API_CUDA && dt_src == API_CUDA)
    {
        state = cudaMemcpy(dst, src, size_in_byte, cudaMemcpyDeviceToDevice);
    }
    else if (dt_dst == API_CUDA && dt_src == API_UNKNOWN)
    {
        state = cudaMemcpy(dst, src, size_in_byte, cudaMemcpyHostToDevice);
    }
    else if (dt_dst == API_UNKNOWN && dt_src == API_CUDA)
    {
        state = cudaMemcpy(dst, src, size_in_byte, cudaMemcpyDeviceToHost);
    }
    //hip
    else if (dt_dst == API_HIP && dt_src == API_HIP)
    {
        state = hipMemcpy(dst, src, size_in_byte, hipMemcpyDeviceToDevice);
    }
    else if (dt_dst == API_HIP && dt_src == API_UNKNOWN)
    {
        state = hipMemcpy(dst, src, size_in_byte, hipMemcpyHostToDevice);
    }
    else if (dt_dst == API_UNKNOWN && dt_src == API_HIP)
    {
        state = hipMemcpy(dst, src, size_in_byte, hipMemcpyDeviceToHost);
    }
    //cuda&hip
    else if (dt_dst == API_CUDA && dt_src == API_HIP)
    {
        auto temp = new char[size_in_byte];
        state += hipMemcpy(temp, src, size_in_byte, hipMemcpyDeviceToHost);
        state += cudaMemcpy(dst, temp, size_in_byte, cudaMemcpyHostToDevice);
        delete[] temp;
    }
    else if (dt_dst == API_HIP && dt_src == API_CUDA)
    {
        auto temp = new char[size_in_byte];
        state += cudaMemcpy(temp, src, size_in_byte, cudaMemcpyDeviceToHost);
        state += hipMemcpy(dst, temp, size_in_byte, hipMemcpyHostToDevice);
        delete[] temp;
    }
    //other
    else
    {
        memcpy(dst, src, size_in_byte);
    }
    if (state != 0)
    {
        LOG_ERR("Memcpy error: {}, size in byte {:g}\n", state, 1.0 * size_in_byte);
    }
    return size_in_byte;
}
}    // namespace cccc