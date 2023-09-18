#pragma once
#include "GpuControl.h"
#include <memory>

namespace cccc
{

//因存在多设备可能，数据须同时保存其设备指针，需以共享指针使用此类
struct MatrixData
{
    GpuControl* gpu_ = nullptr;    //为空表示在cpu中，也即默认情况
    real* data_ = nullptr;
    int64_t occupy_data_size_ = 0;    //实际占用的数据长度，当重设置尺寸不大于此值的时候不会重新分配内存
public:
    MatrixData() = default;
    ~MatrixData() { release(); }
    MatrixData(const MatrixData&) = delete;
    MatrixData& operator=(const MatrixData) = delete;
    void setCuda(GpuControl* gpu) { gpu_ = gpu; }
    void setCudaAsCurrent() { gpu_ = GpuControl::getCurrentCuda(); }
    real* resize(int64_t size, bool reserve_data = true, bool force = false);    //resize前应setCuda
    std::shared_ptr<real*> make_shared_data() { return std::make_shared<real*>(data_); }
    void release();
    ApiType getApiType()
    {
        if (gpu_) { return gpu_->getApiType(); }
        return API_UNKNOWN;
    }

public:
    static int64_t copy(ApiType dt_src, const real* src, ApiType dt_dst, real* dst, int64_t size);
};

}    // namespace cccc