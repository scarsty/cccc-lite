#pragma once
#include "CudaControl.h"
#include "types.h"
#include <memory>

namespace cccc
{

//����ڶ��豸���ܣ�������ͬʱ�������豸ָ�룬���Թ���ָ��ʹ�ô���
struct MatrixData
{
    CudaControl* cuda_ = nullptr;    //Ϊ�ձ�ʾ��cpu�У�Ҳ��Ĭ�����
    real* data_ = nullptr;
    int64_t occupy_data_size_ = 0;    //ʵ��ռ�õ����ݳ��ȣ��������óߴ粻���ڴ�ֵ��ʱ�򲻻����·����ڴ�
public:
    MatrixData() = default;
    ~MatrixData() { free(); }
    MatrixData(const MatrixData&) = delete;
    MatrixData& operator=(const MatrixData) = delete;
    void setCuda(CudaControl* cuda) { cuda_ = cuda; }
    void setCudaAsCurrent() { cuda_ = CudaControl::getCurrentCuda(); }
    real* resize(int64_t size, bool reserve_data = true, bool force = false);    //resizeǰӦsetCuda
    std::shared_ptr<real*> make_shared_data() { return std::make_shared<real*>(data_); }
    void free();

public:
    static int64_t copy(DeviceType dt_src, const real* src, DeviceType dt_dst, real* dst, int64_t size);
};

}    // namespace cccc