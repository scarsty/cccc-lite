#pragma once
#include "GpuControl.h"
#include <memory>

namespace cccc
{

//因存在多设备可能，数据须同时保存其设备指针，需以共享指针使用此类
struct MatrixData
{
    GpuControl* gpu_ = nullptr;    //为空表示在cpu中，也即默认情况
    void* data_ = nullptr;
    int64_t occupy_data_size_ = 0;    //实际占用的数据长度，当重设置尺寸不大于此值的时候不会重新分配内存
    DataType data_type_ = DataType::FLOAT;

    //记录数据所在设备类型和id，实际仅用于析构
    //因为析构时有可能gpu_已经被析构，所以需要不依赖gpu_来记录
    //ApiType api_type_ = API_UNKNOWN;
    //int api_id_ = -1;

public:
    // Global lazy-allocation switch: when enabled, MatrixData::resize only records
    // requested size/type and does not allocate CPU/GPU memory.
    static inline bool lazy_mode = false;

    MatrixData() = default;
    ~MatrixData() { release(); }
    MatrixData(const MatrixData&) = delete;
    MatrixData& operator=(const MatrixData) = delete;
    void setGpu(GpuControl* gpu) { gpu_ = gpu; }
    void setCurrentGpuToHere() { gpu_ = GpuControl::getCurrentGpu(); }
    void setDataType(DataType dt) { data_type_ = dt; }
    DataType getDataType() const { return data_type_; }
    size_t getDataTypeSize() const { return getDataTypeSize(data_type_); }
    size_t size() const { return occupy_data_size_; }
    size_t sizeInByte() const { return occupy_data_size_ * getDataTypeSize(); }
    static bool isLazyMode() { return lazy_mode; }
    static void setLazyMode(bool enabled) { lazy_mode = enabled; }
    void* resize(int64_t size, DataType data_type, bool reserve_data = true, bool force = false);    //resize前应setCuda
    std::shared_ptr<void*> make_shared_data() { return std::make_shared<void*>(data_); }
    void release();
    ApiType getApiType() const
    {
        if (gpu_) { return gpu_->getApiType(); }
        return API_UNKNOWN;
    }
    void* getDataPtr(int i) const { return (char*)data_ + i * getDataTypeSize(); }
    float getData(int i) const
    {
        auto p = getDataPtr(i);
        switch (getDataType())
        {
        case DataType::FLOAT:
            return *(float*)p;
        case DataType::DOUBLE:
            return *(double*)p;
        case DataType::HALF:
            return *(half*)p;
        case DataType::BFLOAT16:
            return (float)(*(bfloat16*)p);
        default:
            return 0;
        }
    }
    static float getData(void* data, int i, DataType dt)
    {
        switch (dt)
        {
        case DataType::FLOAT:
            return *(float*)((char*)data + i * getDataTypeSize(dt));
        case DataType::DOUBLE:
            return *(double*)((char*)data + i * getDataTypeSize(dt));
        case DataType::HALF:
            return *(half*)((char*)data + i * getDataTypeSize(dt));
        case DataType::BFLOAT16:
            return (float)(*(bfloat16*)((char*)data + i * getDataTypeSize(dt)));
        default:
            return 0;
        }
    }
    template <typename T>
    static void setData(void* p, DataType data_type, T v)
    {
        switch (data_type)
        {
        case DataType::FLOAT:
            *(float*)p = v;
            break;
        case DataType::DOUBLE:
            *(double*)p = v;
            break;
        case DataType::HALF:
            *(half*)p = v;
            break;
        case DataType::BFLOAT16:
            *(bfloat16*)p = bfloat16(static_cast<float>(v));
            break;
        }
    }
    template <typename T>
    void setData(int i, T v) { setData(getDataPtr(i), getDataType(), v); }
    void setData(int i, void* p, DataType data_type)    //data_type为p的数据类型
    {
        switch (data_type)
        {
        case DataType::FLOAT:
            setData(i, *(float*)p);
            break;
        case DataType::DOUBLE:
            setData(i, *(double*)p);
            break;
        case DataType::HALF:
            setData(i, *(half*)p);
            break;
        case DataType::BFLOAT16:
            setData(i, *(bfloat16*)p);
            break;
        }
    }

public:
    static int64_t copy(ApiType dt_src, const void* src, ApiType dt_dst, void* dst, int64_t size, DataType dt);
    static int64_t copyByByte(ApiType dt_src, const void* src, ApiType dt_dst, void* dst, int64_t size_in_byte);
    // 异步重载：仅支持 CUDA H2D/D2H/D2D，stream 为非阻塞 cudaStream_t；其他路径退化为同步
    static int64_t copyByByteAsync(ApiType dt_src, const void* src, ApiType dt_dst, void* dst, int64_t size_in_byte, void* stream);
    static size_t getDataTypeSize(DataType dt)
    {
        switch (dt)
        {
        case DataType::FLOAT:
            return sizeof(float);
        case DataType::DOUBLE:
            return sizeof(double);
        case DataType::HALF:
            return sizeof(half);
        case DataType::BFLOAT16:
            return sizeof(bfloat16);
        default:
            return 0;
        }
    }
};

}    // namespace cccc