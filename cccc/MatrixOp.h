#pragma once
#include <any>

#include "Log.h"
#include "Matrix.h"

namespace cccc
{

//矩阵操作，规定网络前向只准使用此文件中的计算

enum class MatrixOpType
{
    NONE,
    ADD,
    MUL,
    ELE_MUL,
    ADD_BIAS,
    CONCAT,
    ACTIVE,
    POOL,
    CONV,
    CORR,
    RESHAPE,
    MAX,
    LOSS,
    L2,
};

struct Any;
template <typename T>
concept notParaAny = !std::is_same_v<std::decay_t<T>, Any>;

struct Any : std::any
{
    template <notParaAny T>
    const T& to() const { return std::any_cast<const T&>(*this); }

    template <notParaAny T>
    T& to() { return std::any_cast<T&>(*this); }

    template <notParaAny T>
    Any(T&& t) :
        std::any(std::forward<T>(t))
    {
    }
};

using VectorAny = std::vector<Any>;

class DLL_EXPORT MatrixOp
{
private:
    MatrixOpType type_ = MatrixOpType::NONE;
    std::vector<MatrixSP> in_;     //输入数据
    std::vector<MatrixSP> wb_;     //参数，通常是权重和偏置
    std::vector<MatrixSP> out_;    //输出数据
    //以下参数一般外界不可见
    //常用的类型
    std::vector<float> a_, b_;
    std::vector<int> window_, stride_, padding_;
    std::vector<Matrix> workspace_;
    //未知或多变的类型
    VectorAny anys_;

    //float dw_scale_ = 1;

public:
    MatrixOp() = default;

    void set(MatrixOpType t, const std::vector<MatrixSP>& m_in, const std::vector<MatrixSP>& m_wb, const std::vector<MatrixSP>& m_out, std::vector<float>&& a = {}, std::vector<float>&& b = {}, VectorAny&& pv = {}, std::vector<Matrix>&& workspace = {},
        std::vector<int>&& window = {}, std::vector<int>&& stride = {}, std::vector<int>&& padding = {})
    {
        type_ = t;
        in_ = m_in;
        wb_ = m_wb;
        out_ = m_out;

        a_ = a;
        b_ = b;
        a_.resize(in_.size(), 1);
        b_.resize(out_.size(), 0);

        anys_ = pv;

        window_ = window;
        stride_ = stride;
        padding_ = padding;
        workspace_ = workspace;

        if (out_.size() > 0 && out_[0]->getDataSize() == 0)
        {
            LOG_ERR("Error: output is empty!\n");
        }
    }

    static void forward(std::vector<MatrixOp>& op_queue);
    static void backward(std::vector<MatrixOp>& op_queue, std::vector<MatrixOp>& loss, bool clear_d);
    void forwardData();
    void backwardDataWeight();
    void backwardLoss();

    static std::string ir(const std::vector<MatrixOp>& op_queue);
    std::string print() const;

    MatrixOpType getType() { return type_; }

    std::vector<MatrixSP>& getMatrixIn() { return in_; }

    std::vector<MatrixSP>& getMatrixWb() { return wb_; }

    std::vector<MatrixSP>& getMatrixOut() { return out_; }

    ActiveFunctionType getActiveType() const;
    int setActiveType(ActiveFunctionType af);

    static void simpleQueue(std::vector<MatrixOp>& op_queue, Matrix& X, Matrix& A);    //仅保留计算图中与X和A有关联的部分

public:
    static void getDefaultStridePadding(MatrixOpType type, const std::vector<int>& dim, std::vector<int>& stride, std::vector<int>& padding);

    void setNeedReverse(bool r)
    {
        for (auto& m : in_)
        {
            m->setNeedBack(r);
        }
        for (auto& m : wb_)
        {
            m->setNeedBack(r);
        }
    }

    //void setDWScale(float s) { dw_scale_ = s; }

public:
    //下面这些函数会设置这个op的参数，并自动计算Y的尺寸返回
    void as_scale(const MatrixSP& X, const MatrixSP& Y, float r);
    void as_mul(const MatrixSP& X1, const MatrixSP& X2, const MatrixSP& Y, float a = 1, std::vector<int> dim = {});
    void as_elementMul(const MatrixSP& X1, const MatrixSP& X2, const MatrixSP& Y, float a = 1);
    void as_add(const MatrixSP& X1, const MatrixSP& X2, const MatrixSP& Y, float a = 1, float b = 1);
    void as_add(const std::vector<MatrixSP>& X_vector, const MatrixSP& Y);
    void as_addBias(const MatrixSP& X, const MatrixSP& bias, const MatrixSP& Y, float a = 1, float b = 1);
    void as_concat(const std::vector<MatrixSP>& X_vector, const MatrixSP& Y);
    void as_active(const MatrixSP& X, const MatrixSP& Y, ActiveFunctionType af);
    void as_active(const MatrixSP& X, const MatrixSP& Y, ActiveFunctionType af, std::vector<int>&& int_vector, std::vector<float>&& real_vector, std::vector<Matrix>&& matrix_vector);
    void as_pool(const MatrixSP& X, const MatrixSP& Y, PoolingType pooling_type, PoolingReverseType reverse_type, std::vector<int> window, std::vector<int> stride, std::vector<int> padding, float a = 1);
    void as_conv(const MatrixSP& X, const MatrixSP& W, const MatrixSP& Y, std::vector<int> stride, std::vector<int> padding, int conv_algo, float a = 1);
    void as_reshape(const MatrixSP& X, const MatrixSP& Y, std::vector<int>& dim);
    void as_max(const MatrixSP& X1, const MatrixSP& X2, const MatrixSP& Y);

    //以下专为处理损失函数
private:
    double value_ = 0;
    double scale_ = 1;

    friend std::vector<MatrixOp> operator+(const std::vector<MatrixOp>& A, const std::vector<MatrixOp>& B);
    friend std::vector<MatrixOp> operator*(const std::vector<MatrixOp>& A, double v);
    friend std::vector<MatrixOp> operator*(double v, const std::vector<MatrixOp>& A);
    friend std::vector<MatrixOp> crossEntropy(MatrixSP& A, MatrixSP& Y);
    friend std::vector<MatrixOp> L2(MatrixSP& A);

public:
    double calc(const MatrixOp& op) { return op.value_ * op.scale_; }

    double calc(const std::vector<MatrixOp>& ops)
    {
        double sum = 0;
        for (auto& op : ops)
        {
            sum += calc(op);
        }
        return sum;
    }
};

//基础运算结束

//以下为处理损失函数
std::vector<MatrixOp> operator+(const std::vector<MatrixOp>& A, const std::vector<MatrixOp>& B);
std::vector<MatrixOp>& operator+=(std::vector<MatrixOp>& A, const std::vector<MatrixOp>& B);
std::vector<MatrixOp> operator*(const std::vector<MatrixOp>& A, double v);
std::vector<MatrixOp> operator*(double v, const std::vector<MatrixOp>& A);

std::vector<MatrixOp> crossEntropy(const MatrixSP& A, const MatrixSP& Y);
std::vector<MatrixOp> L2(const MatrixSP& A);
std::vector<MatrixOp> L2(const std::vector<MatrixSP>& v);

}    // namespace cccc