#pragma once
#include "Matrix.h"
#include "MatrixExtend.h"

namespace woco
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
    LOSS,
    L2,
};

class DLL_EXPORT MatrixOperator
{
public:
    using Queue = std::vector<MatrixOperator>;

private:
    static Queue op_queue_;
    static int calc_;
    static int making_;

    //只有最底层的函数才有必要友元，可据此判断是否位于最底层！

    friend DLL_EXPORT void scale(const Matrix& A, Matrix& R, real r);
    friend DLL_EXPORT void mul(const Matrix& A, const Matrix& B, Matrix& R, real a);
    friend DLL_EXPORT void elementMul(const Matrix& A, const Matrix& B, Matrix& R, real a);
    friend DLL_EXPORT void add(const Matrix& A, const Matrix& B, Matrix& R, realc a, realc b);
    friend DLL_EXPORT void addBias(const Matrix& A, const Matrix& bias, Matrix& R, realc a, realc b);
    friend DLL_EXPORT void concat(const std::vector<Matrix>& A_vector, Matrix& R);
    friend DLL_EXPORT void active(const Matrix& A, Matrix& R, ActiveFunctionType af);
    friend DLL_EXPORT void active(const Matrix& A, Matrix& R, ActiveFunctionType af, std::vector<int>& int_vector, std::vector<real>& real_vector, std::vector<Matrix>& matrix_vector);
    friend DLL_EXPORT void pool(const Matrix& A, Matrix& R, PoolingType pooling_type, const std::vector<int>& window, const std::vector<int>& stride, const std::vector<int>& padding, realc a);
    friend DLL_EXPORT void conv(const Matrix& A, const Matrix& W, Matrix& R, const std::vector<int>& stride, const std::vector<int>& padding, realc a);

private:
    MatrixOpType type_ = MatrixOpType::NONE;
    std::vector<Matrix> matrix_in_;
    std::vector<Matrix> matrix_out_;
    std::vector<int> para_int_;
    std::vector<real> para_real_;
    std::vector<Matrix> para_matrix_;
    std::vector<std::vector<int>> para_int2_;

public:
    MatrixOperator() = default;
    MatrixOperator(MatrixOpType t, const std::vector<Matrix>& m_in, const std::vector<Matrix>& m_out,
        const std::vector<int>& i = {}, const std::vector<real>& r = {}, const std::vector<Matrix>& m2 = {}, const std::vector<std::vector<int>> i2 = {})
    {
        type_ = t;
        matrix_in_ = m_in;
        matrix_out_ = m_out;
        para_int_ = i;
        para_real_ = r;
        para_matrix_ = m2;
        para_int2_ = i2;
    }

    static void beginMaking();
    static void endMaking();
    static void setCalc(int c);

    static Queue& getQueue();
    static void forward(Queue& op_queue);
    static void backward(Queue& op_queue, Queue& loss, Matrix& workspace);
    void forward();
    void backward();

    static void print(const Queue& op_queue);
    void print() const;

    MatrixOpType getType() { return type_; }

    static void simpleQueue(MatrixOperator::Queue& op_queue, const Matrix& X, const Matrix& A);    //仅保留计算图中与X和A有关联的部分

    //以下专为处理损失函数
private:
    double value_ = 0;
    double scale_ = 1;

    friend DLL_EXPORT Queue operator+(const Queue& A, const Queue& B);
    friend DLL_EXPORT Queue operator*(const Queue& A, double v);
    friend DLL_EXPORT Queue operator*(double v, const Queue& A);
    friend DLL_EXPORT Queue crossEntropy(const Matrix& A, const Matrix& Y);
    friend DLL_EXPORT Queue L2(const Matrix& A);

public:
    double calc(const MatrixOperator& op) { return op.value_ * op.scale_; }
    double calc(const std::vector<MatrixOperator>& ops)
    {
        double sum = 0;
        for (auto& op : ops)
        {
            sum += calc(op);
        }
        return sum;
    }
};

//基础运算开始
DLL_EXPORT void scale(const Matrix& A, Matrix& R, real r);
DLL_EXPORT void mul(const Matrix& A, const Matrix& B, Matrix& R, real a = 1);
DLL_EXPORT void elementMul(const Matrix& A, const Matrix& B, Matrix& R, real a = 1);
DLL_EXPORT void add(const Matrix& A, const Matrix& B, Matrix& R, realc a = 1, realc b = 1);
DLL_EXPORT void addBias(const Matrix& A, const Matrix& bias, Matrix& R, realc a = 1, realc b = 1);
DLL_EXPORT void concat(const std::vector<Matrix>& A_vector, Matrix& R);
DLL_EXPORT void active(const Matrix& A, Matrix& R, ActiveFunctionType af);
DLL_EXPORT void active(const Matrix& A, Matrix& R, ActiveFunctionType af, std::vector<int>& int_vector, std::vector<real>& real_vector, std::vector<Matrix>& matrix_vector);
DLL_EXPORT void pool(const Matrix& A, Matrix& R, PoolingType pooling_type, const std::vector<int>& window, const std::vector<int>& stride, const std::vector<int>& padding, realc a = 1);
DLL_EXPORT void conv(const Matrix& A, const Matrix& W, Matrix& R, const std::vector<int>& stride, const std::vector<int>& padding, realc a = 1);
//基础运算结束

//带返回的运算
DLL_EXPORT Matrix scale(const Matrix& A, real r);
DLL_EXPORT Matrix mul(const Matrix& A, const Matrix& B, real a = 1);
DLL_EXPORT Matrix elementMul(const Matrix& A, const Matrix& B, real a = 1);
DLL_EXPORT Matrix add(const Matrix& A, const Matrix& B, realc a = 1, realc b = 1);
DLL_EXPORT Matrix addBias(const Matrix& A, const Matrix& bias, realc a = 1, realc b = 1);
DLL_EXPORT Matrix active(const Matrix& A, ActiveFunctionType af);
DLL_EXPORT Matrix active(const Matrix& A, ActiveFunctionType af, std::vector<int>& int_vector, std::vector<real>& real_vector, std::vector<Matrix>& matrix_vector);
DLL_EXPORT Matrix pool(const Matrix& A, PoolingType pooling_type, const std::vector<int>& window, std::vector<int> stride = {}, std::vector<int> padding = {}, realc a = 1);
DLL_EXPORT Matrix conv(const Matrix& A, const Matrix& W, std::vector<int> stride = {}, std::vector<int> padding = {}, realc a = 1);
DLL_EXPORT Matrix maxpool(const Matrix& A, const std::vector<int>& window, std::vector<int> stride = {}, std::vector<int> padding = {}, realc a = 1);
DLL_EXPORT Matrix relu(const Matrix& A);
DLL_EXPORT Matrix sigmoid(const Matrix& A);
DLL_EXPORT Matrix softmax(const Matrix& A);
DLL_EXPORT Matrix softmax_ce(const Matrix& A);

//运算符重载：+-*数乘
DLL_EXPORT Matrix operator+(const Matrix& A, const Matrix& B);
DLL_EXPORT Matrix operator-(const Matrix& A, const Matrix& B);
DLL_EXPORT Matrix operator*(const Matrix& A, const Matrix& B);
DLL_EXPORT Matrix operator*(real r, const Matrix& A);
DLL_EXPORT Matrix operator*(const Matrix& A, real r);

//以下为处理损失函数
DLL_EXPORT MatrixOperator::Queue operator+(const MatrixOperator::Queue& A, const MatrixOperator::Queue& B);
DLL_EXPORT MatrixOperator::Queue& operator+=(MatrixOperator::Queue& A, const MatrixOperator::Queue& B);
DLL_EXPORT MatrixOperator::Queue operator*(const MatrixOperator::Queue& A, double v);
DLL_EXPORT MatrixOperator::Queue operator*(double v, const MatrixOperator::Queue& A);

DLL_EXPORT MatrixOperator::Queue crossEntropy(const Matrix& A, const Matrix& Y);
DLL_EXPORT MatrixOperator::Queue L2(const Matrix& A);
DLL_EXPORT MatrixOperator::Queue L2(const std::vector<Matrix>& v);

}    // namespace woco