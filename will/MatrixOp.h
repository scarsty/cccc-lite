#pragma once
#include "Log.h"
#include "Matrix.h"
#include "MatrixEx.h"

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
    LOSS,
    L2,
};

class MatrixOp
{
private:
    MatrixOpType type_ = MatrixOpType::NONE;
    std::vector<MatrixSP> in_;     //输入数据
    std::vector<MatrixSP> wb_;     //参数，通常是权重和偏置
    std::vector<MatrixSP> out_;    //输出数据
    std::vector<int> para_int_;
    std::vector<real> para_real_;
    std::vector<Matrix> para_matrix_;    //某些计算需要的额外工作空间
    std::vector<std::vector<int>> para_int_v_;

public:
    MatrixOp() = default;
    void set(MatrixOpType t, const std::vector<MatrixSP>& m_in, const std::vector<MatrixSP>& m_wb, const std::vector<MatrixSP>& m_out,
        const std::vector<int>& i = {}, const std::vector<real>& r = {}, const std::vector<Matrix>& m2 = {}, const std::vector<std::vector<int>> iv = {})
    {
        type_ = t;
        in_ = m_in;
        wb_ = m_wb;
        out_ = m_out;
        para_int_ = i;
        para_real_ = r;
        para_matrix_ = m2;
        para_int_v_ = iv;
        if (out_.size() > 0 && out_[0]->getDataSize() == 0)
        {
            LOG(stderr, "Error: output is empty!\n");
        }
    }

    static void forward(std::vector<MatrixOp>& op_queue);
    static void backward(std::vector<MatrixOp>& op_queue, std::vector<MatrixOp>& loss, bool clear_d);
    void forwardData();
    void backwardDataWeight();
    void backwardLoss();

    static void print(const std::vector<MatrixOp>& op_queue);
    void print() const;

    MatrixOpType getType() { return type_; }
    std::vector<MatrixSP>& getMatrixIn() { return in_; }
    std::vector<MatrixSP>& getMatrixWb() { return wb_; }
    std::vector<MatrixSP>& getMatrixOut() { return out_; }
    const std::vector<int>& getPataInt() { return para_int_; }
    const std::vector<std::vector<int>>& getPataInt2() { return para_int_v_; }

    static void simpleQueue(std::vector<MatrixOp>& op_queue, Matrix& X, Matrix& A);    //仅保留计算图中与X和A有关联的部分

public:
    static void getDefaultStridePadding(MatrixOpType type, const std::vector<int>& dim, std::vector<int>& stride, std::vector<int>& padding);

public:
    //下面这些函数会设置这个op的参数，并自动计算Y的尺寸返回
    void as_scale(MatrixSP& X, MatrixSP& Y, real r);
    void as_mul(MatrixSP& X1, MatrixSP& X2, MatrixSP& Y, real a = 1, std::vector<int> dim = {});
    void as_elementMul(MatrixSP& X1, MatrixSP& X2, MatrixSP& Y, real a = 1);
    void as_add(MatrixSP& X1, MatrixSP& X2, MatrixSP& Y, realc a = 1, realc b = 1);
    void as_add(std::vector<MatrixSP>& X_vector, MatrixSP& Y);
    void as_addBias(MatrixSP& X, MatrixSP& bias, MatrixSP& Y, realc a = 1, realc b = 1);
    void as_concat(std::vector<MatrixSP>& X_vector, MatrixSP& Y);
    void as_active(MatrixSP& X, MatrixSP& Y, ActiveFunctionType af);
    void as_active(MatrixSP& X, MatrixSP& Y, ActiveFunctionType af, std::vector<int>&& int_vector, std::vector<real>&& real_vector, std::vector<Matrix>&& matrix_vector);
    void as_pool(MatrixSP& X, MatrixSP& Y, PoolingType pooling_type, int reverse, std::vector<int> window, std::vector<int> stride, std::vector<int> padding, realc a = 1);
    void as_conv(MatrixSP& X, MatrixSP& W, MatrixSP& Y, std::vector<int> stride, std::vector<int> padding, int conv_type, realc a = 1);
    void as_reshape(MatrixSP& X, MatrixSP& Y, std::vector<int>& dim);

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

std::vector<MatrixOp> crossEntropy(MatrixSP& A, MatrixSP& Y);
std::vector<MatrixOp> L2(MatrixSP& A);
std::vector<MatrixOp> L2(const std::vector<MatrixSP>& v);

}    // namespace cccc