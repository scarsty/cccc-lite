#include "MatrixData.h"
#include "Log.h"
#include "gpu_lib.h"

namespace cccc
{
//此处有问题，目前来看只能在同设备中resize

static int64_t memory_used_ = 0;

void* MatrixData::resize(int64_t size, DataType data_type, bool reserve_data, bool force)
{
    data_type_ = data_type;
    if (isLazyMode() && !force)
    {
        // Lazy mode: only record requested capacity, defer real allocation.
        // Keep existing buffer if any, but do not allocate a new one.
        if (size > occupy_data_size_ || force)
        {
            occupy_data_size_ = size;
        }
        return data_;
    }
    // Only skip reallocation if the buffer is already allocated.
    // After lazy mode, occupy_data_size_ may be set but data_ is still null —
    // must fall through to actually allocate in that case.
    if (size <= occupy_data_size_ && !force && data_ != nullptr)
    {
        return data_;
    }
    // When data_ is null (first real allocation after lazy planning),
    // allocate the FULL planned capacity (occupy_data_size_), not just the
    // requested size.  Multiple tensors may share this MatrixData and the pool
    // owner recorded the max size during lazy planning.  If a smaller aliased
    // tensor triggers allocation first, we must still reserve the full buffer so
    // that the larger pool owner can use it without reallocation (which would
    // leave the aliased tensor with a stale/freed pointer).
    int64_t alloc_size = (data_ == nullptr) ? std::max(size, occupy_data_size_) : size;
    void* new_data = nullptr;
    auto device_type = UnitType::GPU;
    if (!reserve_data)    //数据不需保留可以先行释放，可以节省显存的使用，否则峰值占用是新旧尺寸之和
    {
        release();
    }
    if (gpu_)
    {
        gpu_->setAsCurrent();
        if (gpu_->malloc(new_data, alloc_size * getDataTypeSize(data_type_)) == 0)
        {
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
        //new_data = new char[size * getDataTypeSize(data_type_)];
        new_data = malloc(alloc_size * getDataTypeSize(data_type_));
        //cudaMallocHost(&new_data, alloc_size * getDataTypeSize(data_type_));
    }
    if (reserve_data)
    {
        copy(getApiType(), data_, getApiType(), new_data, std::min(size, occupy_data_size_), data_type_);
        release();
    }
    occupy_data_size_ = alloc_size;
    memory_used_ += occupy_data_size_ * getDataTypeSize(data_type_);
    return data_ = new_data;
}

void MatrixData::release()
{
    if (data_ == nullptr)
    {
        occupy_data_size_ = 0;
        return;
    }

    auto status = GpuControl::free(data_);
    if (status == 0)
    {
        //LOG("Device {} FREE {:g}, memory used {:g}!\n", gpu_->getDeviceID(), 1.0 * occupy_data_size_ * getDataTypeSize(data_type_), 1.0 * gpu_->getMemoryUsed());
    }
    else if (status < 0)
    {
        LOG("Device {} FAIL TO FREE {:g}!\n", gpu_->getDeviceID(), 1.0 * occupy_data_size_ * getDataTypeSize(data_type_));
    }
    else
    {
        //delete[] (char*)data_;
        free(data_);
        //cudaFreeHost(data_);
    }
    memory_used_ -= occupy_data_size_ * getDataTypeSize(data_type_);
    occupy_data_size_ = 0;
    //LOG("memory used {:g}!\n", 1.0 * memory_used_);
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
        std::vector<char> temp(size_in_byte);
        state += hipMemcpy(temp.data(), src, size_in_byte, hipMemcpyDeviceToHost);
        state += cudaMemcpy(dst, temp.data(), size_in_byte, cudaMemcpyHostToDevice);
    }
    else if (dt_dst == API_HIP && dt_src == API_CUDA)
    {
        std::vector<char> temp(size_in_byte);
        state += cudaMemcpy(temp.data(), src, size_in_byte, cudaMemcpyDeviceToHost);
        state += hipMemcpy(dst, temp.data(), size_in_byte, hipMemcpyHostToDevice);
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

// 异步拷贝重载，stream 为 cudaStream_t（以 void* 传入避免头文件引入 cuda_runtime.h）。
// 仅对 CUDA H2D/D2H/D2D 生效，其余路径退化为同步 copyByByte。
// 调用前须确保：1) 使用 pinned memory（cudaMallocHost）；2) stream 以 cudaStreamNonBlocking 创建。
int64_t MatrixData::copyByByteAsync(ApiType dt_src, const void* src, ApiType dt_dst, void* dst, int64_t size_in_byte, void* stream)
{
    if (src == nullptr || dst == nullptr || src == dst || size_in_byte <= 0)
    {
        return 0;
    }
#ifdef ENABLE_CUDA
    cudaStream_t cs = static_cast<cudaStream_t>(stream);
    int state = 0;
    if (dt_dst == API_CUDA && dt_src == API_CUDA)
    {
        state = cudaMemcpyAsync(dst, src, size_in_byte, cudaMemcpyDeviceToDevice, cs);
    }
    else if (dt_dst == API_CUDA && dt_src == API_UNKNOWN)
    {
        state = cudaMemcpyAsync(dst, src, size_in_byte, cudaMemcpyHostToDevice, cs);
    }
    else if (dt_dst == API_UNKNOWN && dt_src == API_CUDA)
    {
        state = cudaMemcpyAsync(dst, src, size_in_byte, cudaMemcpyDeviceToHost, cs);
    }
    else
    {
        // 非 CUDA 路径（HIP、CPU-CPU 等）退化为同步
        return copyByByte(dt_src, src, dt_dst, dst, size_in_byte);
    }
    if (state != 0)
    {
        LOG_ERR("cudaMemcpyAsync error: {}, size in byte {:g}\n", state, 1.0 * size_in_byte);
    }
    return size_in_byte;
#else
    // 未编译 CUDA 时退化为同步
    return copyByByte(dt_src, src, dt_dst, dst, size_in_byte);
#endif
}
}    // namespace cccc