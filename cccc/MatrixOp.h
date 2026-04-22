#pragma once
#include <any>

#include "Log.h"
#include "Matrix.h"
#include "Solver.h"

#include <map>

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

namespace cccc
{

//矩阵操作，规定网络前向只准使用此文件中的计算

enum class MatrixOpType
{
    NONE,
    SCALE,
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
    BATCH_NORM,
    POOL_CHANNEL,
    LOSS,
    FOCAL,
    ZERO_LIMIT,
    L2,
};

class CCCC_EXPORT MatrixOp
{
public:
    static std::string getOpName(MatrixOpType type)
    {
        std::map<MatrixOpType, std::string> m = {
            { MatrixOpType::NONE, "none" },
            { MatrixOpType::SCALE, "scale" },
            { MatrixOpType::ADD, "add" },
            { MatrixOpType::MUL, "mul" },
            { MatrixOpType::ELE_MUL, "ele_mul" },
            { MatrixOpType::ADD_BIAS, "add_bias" },
            { MatrixOpType::CONCAT, "concat" },
            { MatrixOpType::ACTIVE, "active" },
            { MatrixOpType::POOL, "pool" },
            { MatrixOpType::CONV, "conv" },
            { MatrixOpType::CORR, "corr" },
            { MatrixOpType::RESHAPE, "reshape" },
            { MatrixOpType::MAX, "max" },
            { MatrixOpType::BATCH_NORM, "batch_norm" },
            { MatrixOpType::POOL_CHANNEL, "pool_channel" },
            { MatrixOpType::LOSS, "loss" },
            { MatrixOpType::FOCAL, "focal" },
            { MatrixOpType::ZERO_LIMIT, "zero_limit" },
            { MatrixOpType::L2, "l2" },
        };
        return m[type];
    }

private:
    int index_ = 0;
    MatrixOpType type_ = MatrixOpType::NONE;
    std::vector<MatrixSP> in_;     //输入数据
    std::vector<MatrixSP> out_;    //输出数据
    //以下参数一般外界不可见
    //常用的类型
    std::vector<float> a_, b_;
    std::vector<int> window_, stride_, padding_;
    //未知或多变的类型
    VectorAny anys_;
    bool connect_x_ = false;       //是否与X有关联，用于简化计算图
    bool connect_a_ = false;       //是否与a有关联，用于简化计算图
    bool connect_loss_ = false;    //是否与loss有关联，用于简化计算图

    double forward_time_ = 0, backward_time_;    //秒，主要用于调试和性能分析
    int forward_count_ = 0, backward_count_ = 0;

public:
    SolverType solver_type_ = SOLVER_SGD;

    //float dw_scale_ = 1;

public:
    MatrixOp() = default;

    void set(MatrixOpType t, const std::vector<MatrixSP>& m_in, const std::vector<MatrixSP>& m_out, std::vector<float>&& a = {}, std::vector<float>&& b = {}, VectorAny&& pv = {}, std::vector<int>&& window = {}, std::vector<int>&& stride = {}, std::vector<int>&& padding = {})
    {
        type_ = t;
        in_ = m_in;
        out_ = m_out;

        a_ = a;
        b_ = b;
        a_.resize(in_.size(), 1);
        b_.resize(out_.size(), 0);

        anys_ = pv;

        window_ = window;
        stride_ = stride;
        padding_ = padding;

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

    static std::string inference_ir(const std::vector<MatrixOp>& op_queue);
    std::string print() const;

    MatrixOpType getType() const { return type_; }

    int getIndex() const { return index_; }

    std::vector<MatrixSP>& getMatrixIn() { return in_; }

    std::vector<MatrixSP>& getMatrixOut() { return out_; }

    ActiveFunctionType getActiveType() const;
    int setActiveType(ActiveFunctionType af);

    static void checkConnect(std::vector<MatrixOp>& op_queue, Matrix& X, Matrix& A, std::vector<MatrixOp>& losses);    //仅保留计算图中与X和loss有关联的部分

public:
    static void getDefaultStridePadding(MatrixOpType type, const std::vector<int>& dim, std::vector<int>& stride, std::vector<int>& padding);

    void setNeedReverse(bool r)
    {
        for (auto& m : in_)
        {
            m->setNeedBack(r);
        }
    }

    void clearTime()
    {
        forward_time_ = backward_time_ = 0;
        forward_count_ = backward_count_ = 0;
    }
    double getForwardTime() const { return forward_time_; }
    double getBackwardTime() const { return backward_time_; }

    //void setDWScale(float s) { dw_scale_ = s; }

public:
    //下面这些函数会设置这个op的参数，并自动计算Y的尺寸返回
    void as_scale(const MatrixSP& X, const MatrixSP& Y, float r);
    void as_mul(const MatrixSP& X1, const MatrixSP& X2, const MatrixSP& Y, float a = 1, std::vector<int> dim = {});
    void as_elementMul(const MatrixSP& X1, const MatrixSP& X2, const MatrixSP& Y, float a = 1);
    void as_add(const MatrixSP& X1, const MatrixSP& X2, const MatrixSP& Y, float a = 1, float b = 1);
    void as_add(const std::vector<MatrixSP>& X_vector, const MatrixSP& Y, std::vector<float> a = {});
    void as_addBias(const MatrixSP& X, const MatrixSP& bias, const MatrixSP& Y, float a = 1, float b = 1);
    void as_concat(const std::vector<MatrixSP>& X_vector, const MatrixSP& Y);
    void as_active(const MatrixSP& X, const MatrixSP& Y, ActiveFunctionType af);
    void as_active(const MatrixSP& X, const MatrixSP& Y, ActiveFunctionType af, std::vector<int>&& int_vector, std::vector<float>&& real_vector, std::vector<Matrix>&& matrix_vector);
    void as_pool(const MatrixSP& X, const MatrixSP& Y, PoolingType pooling_type, PoolingReverseType reverse_type, std::vector<int> window, std::vector<int> stride, std::vector<int> padding, float a = 1);
    void as_conv(const MatrixSP& X, const MatrixSP& W, const MatrixSP& Y, std::vector<int> stride, std::vector<int> padding, int conv_algo, float a = 1);
    void as_corr(const MatrixSP& X, const MatrixSP& W, const MatrixSP& Y, std::vector<int> stride, std::vector<int> padding, int conv_algo, float a = 1);
    void as_reshape(const MatrixSP& X, const MatrixSP& Y, std::vector<int>& dim);
    void as_max(const MatrixSP& X1, const MatrixSP& X2, const MatrixSP& Y);
    void as_batchNorm(const MatrixSP& X, const MatrixSP& scale, const MatrixSP& Y, BatchNormalizationType bn_type = BATCH_NORMALIZATION_SPATIAL, float epsilon = 1e-5f);
    void as_poolChannel(const MatrixSP& X, const MatrixSP& Y, PoolingType pooling_type, PoolingReverseType reverse_type, float a = 1);

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

std::vector<MatrixOp> commonLoss(MatrixOpType type, const std::vector<MatrixSP>& B, const std::vector<float>& a);
std::vector<MatrixOp> crossEntropy(const MatrixSP& A, const MatrixSP& Y);
std::vector<MatrixOp> crossEntropy(const MatrixSP& A, const MatrixSP& Y, const MatrixSP& LW);
std::vector<MatrixOp> focal(const MatrixSP& A, const MatrixSP& Y);
std::vector<MatrixOp> focal(const MatrixSP& A, const MatrixSP& Y, const MatrixSP& LW);
std::vector<MatrixOp> L2(const MatrixSP& A);
std::vector<MatrixOp> L2(const std::vector<MatrixSP>& v);

}    // namespace cccc

template <typename CharT>
struct std::formatter<cccc::MatrixSP, CharT>
{
    constexpr auto parse(std::basic_format_parse_context<CharT>& pc)
    {
        return pc.begin();
    }
    template <typename FormatContext>
    auto format(const cccc::MatrixSP& v, FormatContext& fc) const
    {
        if (v->isInput())
        {
            if (v->isWeight())
            {
                return std::format_to(fc.out(), "{}", v->sizeMessage());
            }
            return std::format_to(fc.out(), "{}", v->sizeMessage(0));
        }
        return std::format_to(fc.out(), "M{}", (uint64_t)v.get());
    }
};