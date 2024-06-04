#pragma once
#include "types.h"
#include <string>
#include <typeinfo>
#include <unordered_map>
#include <vector>

struct cudnnTensorStruct;
struct miopenTensorDescriptor;

namespace cccc
{

class TensorDesc
{
private:
    cudnnTensorStruct* cudnn_tensor_desc_ = nullptr;
    miopenTensorDescriptor* miopen_tensor_desc_ = nullptr;

public:
    TensorDesc(uint32_t flag = 3);    //1: cudnn, 2: miopen, 3: both
    ~TensorDesc();

    cudnnTensorStruct* cudnnDesc() { return cudnn_tensor_desc_; }
    miopenTensorDescriptor* miopenDesc() { return miopen_tensor_desc_; }
    void setDesc4D(DataType data_type, int w, int h, int c, int n);
    void setDescND(DataType data_type, std::vector<int> dim);
};

class OtherDesc
{
    std::unordered_map<size_t, void*> desc_;
    template <typename T>
    static void create(void** p);

public:
    template <typename T>
    T getDesc()
    {
        auto it = desc_.find(typeid(T).hash_code());
        if (it != desc_.end())
        {
            return T(it->second);
        }
        else
        {
            auto& p = desc_[typeid(T).hash_code()];
            create<T>(&p);
            return T(p);
        }
    }
};

}    //namespace cccc