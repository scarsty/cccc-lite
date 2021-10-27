#include "MatrixData.h"
#include "Matrix.h"

namespace cccc
{
//�˴������⣬Ŀǰ����ֻ����ͬ�豸��resize
real* MatrixData::resize(int64_t size, bool reserve_data, bool force)
{
    if (size <= occupy_data_size_ && !force)
    {
        return data_;
    }
    real* new_data = nullptr;
    auto device_type = DeviceType::GPU;
    if (!reserve_data)    //���ݲ��豣�����������ͷţ����Խ�ʡ�Դ��ʹ�ã������ֵռ�����¾ɳߴ�֮��
    {
        free();
    }
    if (cuda_)
    {
        cuda_->setThisDevice();
        if (cudaMalloc((void**)&new_data, size * sizeof(real)) == cudaSuccess)
        {
            //fmt::print(stderr, "Success malloc size in byte is %lld (%g)!\n", size * sizeof(real), 1.0 * size * sizeof(real));
            //if (size * sizeof(real) > 3e8)
            //{
            //    fmt::print(stderr, "Very big!\n");
            //}
        }
        else
        {
            fmt::print(stderr, "Error: matrix malloc data failed! size in byte is {} ({:g})!\n", size * sizeof(real), 1.0 * size * sizeof(real));
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
    //if (occupy_data_size_ * sizeof(real) > 1e9)
    //{
    //    fmt::print(stderr, "Free %lld bytes!\n", occupy_data_size_ * sizeof(real));
    //}

    if (data_ == nullptr)
    {
        occupy_data_size_ = 0;
        return;
    }
    if (cuda_)
    {
        auto current_gpu = CudaControl::getCurrentDevice();
        if (current_gpu == cuda_->getDeviceID())
        {
            auto status = cudaFree(data_);
            //if (status != cudaSuccess)
            //{
            //    fmt::print(stderr, "Failed to free %lld bytes!\n", occupy_data_size_ * sizeof(real));
            //}
        }
        else
        {
            cuda_->setThisDevice();
            cudaFree(data_);
            CudaControl::setDevice(current_gpu);
            //cuda_ = CudaControl::getCurrentCuda();
        }
    }
    else
    {
        delete[] data_;
    }
    occupy_data_size_ = 0;
    data_ = nullptr;
}

}    // namespace cccc