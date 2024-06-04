#pragma once
#include <cmath>
#include <cstring>

namespace cccc
{

namespace VectorMath
{
//向量数学类，全部是模板函数
//cpu计算只使用float
#define VECTOR(fv, f) \
    inline void fv(const void* Av, void* Rv, int size, float a = 1, float r = 0) \
    { \
        auto A = (const float*)Av; \
        auto R = (float*)Rv; \
        for (int i = 0; i < size; i++) { R[i] = a * f(A[i]) + r * R[i]; } \
    }
#define VECTOR_B(fv, content) \
    inline void fv(const void* Av, const void* DAv, const void* Rv, void* DRv, int size, float a = 1, float r = 0) \
    { \
        auto A = (const float*)Av; \
        auto DA = (const float*)DAv; \
        auto R = (const float*)Rv; \
        auto DR = (float*)DRv; \
        for (int i = 0; i < size; i++) { DR[i] = a * (content) + r * DR[i]; } \
    }

template <typename T>
inline T sigmoid(T x)
{
    return 1 / (1 + exp(-x));
}
template <typename T>
inline T softplus(T x) { return log(1 + exp(x)); }
template <typename T>
inline T relu(T x) { return x > T(0) ? x : T(0); }

VECTOR(log_v, log);
VECTOR(exp_v, exp);

VECTOR(sigmoid_v, sigmoid);
VECTOR(relu_v, relu);
VECTOR(tanh_v, tanh);
VECTOR(softplus_v, softplus);
template <typename T>
void linear_v(T* x, T* a, int size) { memcpy(a, x, sizeof(T) * size); }

inline void clipped_relu_v(const void* Av, void* Rv, float v, int size, float a = 1, float r = 0)
{
    auto A = (const float*)Av;
    auto R = (float*)Rv;
    for (int i = 0; i < size; i++)
    {
        if (A[i] > v)
        {
            R[i] = a * v + r * R[i];
        }
        else if (A[i] < 0)
        {
            R[i] = r * R[i];
        }
        else
        {
            R[i] = a * A[i] + r * R[i];
        }
    }
}

VECTOR_B(exp_vb, A[i]);
VECTOR_B(sigmoid_vb, A[i] * (1 - A[i]) * DA[i]);    //sigmoid导数直接使用a计算
VECTOR_B(relu_vb, R[i] > 0 ? DA[i] : 0);
VECTOR_B(tanh_vb, (1 - A[i] * A[i]) * DA[i]);
VECTOR_B(softplus_vb, sigmoid(R[i]));
VECTOR_B(linear_vb, 1);

inline void clipped_relu_vb(const void* Av, const void* DAv, const void* Rv, void* DRv, float v, int size, float a = 1, float r = 0)
{
    auto A = (const float*)Av;
    auto DA = (const float*)DAv;
    auto R = (const float*)Rv;
    auto DR = (float*)DRv;
    for (int i = 0; i < size; i++)
    {
        DR[i] = a * ((R[i] > 0) && (R[i] < v) ? DA[i] : 0) + r * DR[i];
    }
}

//下面3个都是softmax用的
inline void minus_max(void* xv, int size)
{
    auto x = (float*)xv;
    auto m = x[0];
    for (int i = 1; i < size; i++)
    {
        m = std::max(x[i], m);
    }
    for (int i = 0; i < size; i++)
    {
        x[i] -= m;
    }
}

inline void softmax_vb_sub(const void* av, const void* dav, float v, void* dxv, int size)
{
    auto a = (const float*)av;
    auto da = (const float*)dav;
    auto dx = (float*)dxv;
    for (int i = 0; i < size; i++)
    {
        dx[i] = a[i] * (da[i] - v);
    }
}

inline void softmaxlog_vb_sub(const void* av, const void* dav, float v, void* dxv, int size)
{
    auto a = (const float*)av;
    auto da = (const float*)dav;
    auto dx = (float*)dxv;
    for (int i = 0; i < size; i++)
    {
        dx[i] = da[i] - v * exp(a[i]);
    }
}

template <typename T>
bool inbox(T _x, T _y, T x, T y, T w, T h)
{
    return _x >= x && _y >= y && _x < x + h && _y < y + h;
}

template <typename T>
T sum(T* x, int size)
{
    T sum = 0;
    for (int i = 0; i < size; i++)
    {
        sum += x[i];
    }
    return sum;
}

//please use this in "do while" loop
inline int get_next_coord(std::vector<int>& r, const std::vector<int>& dim)
{
    int ret = 0;
    for (int i = 0; i < r.size(); i++)
    {
        if (r[i] < dim[i] - 1)
        {
            ret = 1;
            r[i]++;
            break;
        }
        else
        {
            r[i] = 0;
        }
    }
    return ret;
}

//inline std::vector<std::vector<int>> get_all_coords(const std::vector<int>& dim)
//{
//    std::vector<std::vector<int>> r;
//    std::vector<int> a(dim.size(), 0);
//    r.push_back(a);
//    while (get_next_coord(a, dim))
//    {
//        r.push_back(a);
//    }
//    return r;
//}

template <typename T>
void force_resize(std::vector<T>& vec, int size, T v)
{
    if (vec.size() == 0)
    {
        vec.resize(size, v);
    }
    else
    {
        vec.resize(size, vec.back());
    }
}

template <typename T>
T multiply(const std::vector<T>& vec, int size = -1)
{
    T ret = 1;
    if (size < 0)
    {
        size = vec.size();
    }
    else
    {
        size = (std::min)(int(vec.size()), size);
    }
    for (int i = 0; i < size; i++)
    {
        ret *= vec[i];
    }
    return ret;
}

#undef VECTOR
#undef VECTOR_B

template <typename T>
bool vector_have(const std::vector<T>& ops, const T& op)
{
    for (auto& o : ops)
    {
        if (op == o)
        {
            return true;
        }
    }
    return false;
}
}    // namespace VectorMath

//极端情况使用vector可能过于臃肿
//通常没必要
/*
template<typename T>
class SimpleVector
{
private:
    T* data_ = nullptr;
    int size_ = 0;
public:
    SimpleVector() {}
    SimpleVector(int n)
    {
        data_ = new T[n];
        size_ = n;
    }
    ~SimpleVector() { delete[] data_; }
    int size() { return size_; }
    void resize(int n)
    {
        if (data_) { delete[] data_; }
        data_ = new T[n];
    }
    T& operator [](int i) { return data_[i]; }
    T& getData(int i) { return data_[i]; }
    void init(T t) { for (int i=0; i < size_; i++) { data_[i]=t; } }
};
*/

}    // namespace cccc