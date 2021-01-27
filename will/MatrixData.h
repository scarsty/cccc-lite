#pragma once
#include "CudaControl.h"
#include "types.h"
#include <memory>

namespace will
{

//因存在多设备可能，数据须同时保存其设备指针，需以共享指针使用此类
struct MatrixData
{
    CudaControl* cuda_ = nullptr;    //为空表示在cpu中，也即默认情况
    real* data_ = nullptr;
    int64_t occupy_data_size_ = 0;    //实际占用的数据长度，当重设置尺寸不大于此值的时候不会重新分配内存
public:
    MatrixData() = default;
    ~MatrixData() { free(); }
    MatrixData(const MatrixData&) = delete;
    MatrixData& operator=(const MatrixData) = delete;
    void setCuda(CudaControl* cuda) { cuda_ = cuda; }
    void setCudaAsCurrent() { cuda_ = CudaControl::getCurrentCuda(); }
    real* resize(int64_t size, bool reserve_data = true, bool force = false);    //resize前应setCuda
    std::shared_ptr<real*> make_shared_data() { return std::make_shared<real*>(data_); }
    void free();

public:
    static int64_t copy(DeviceType dt_src, const real* src, DeviceType dt_dst, real* dst, int64_t size)
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
            fmt::print(stderr, "Error: cudaMemcpy failed with error code is {}, size in byte is {} ({:g})!\n", state, size_in_byte, 1.0 * size_in_byte);
        }
        return size;
    }
};

}    // namespace will