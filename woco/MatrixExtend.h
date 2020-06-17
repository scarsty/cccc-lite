#pragma once
#include "Matrix.h"

namespace woco
{

//该类中均不是矩阵基本计算，全部为静态函数
class DLL_EXPORT MatrixExtend : public Matrix
{
private:
    MatrixExtend() = delete;

public:
    //以下函数不属于矩阵基本运算

    //按channel加偏置
    static void addBias(const Matrix& A, const Matrix& bias, Matrix& R, realc a = 1, realc b = 1);
    static void addBiasBackward(Matrix& A, Matrix& bias, const Matrix& R, realc a = 1, realc b = 1);

    // the function is private for concat the data and append the data
    static void concatByChannel(const std::vector<Matrix>& A_vector, Matrix& R);
    static void concatByChannelBackward(std::vector<Matrix>& A_vector, const Matrix& R);
    static void splitByChannel(const Matrix& A, std::vector<Matrix>& R_vector);

    static void activeBufferInit(ActiveFunctionType af, Matrix& A, std::vector<int>& int_vector, std::vector<Matrix>& matrix_vector);

    //激活的实际计算
    //激活和反向激活中，输入和输出矩阵都是同维度
    //请注意反向的情况，常数a和r的含义与正向的对应关系不同
    static void activeForward(const Matrix& A, Matrix& R, ActiveFunctionType af, real a = 1, real r = 0);
    static void activeBackward(Matrix& A, const Matrix& R, ActiveFunctionType af, real a = 1, real r = 0);

    static void activeForward2(const Matrix& A, Matrix& R, ActiveFunctionType af,
        std::vector<int>& int_vector, std::vector<real>& real_vector, std::vector<Matrix>& matrix_vector, real a = 1, real r = 0);
    static void activeBackward2(Matrix& A, const Matrix& R, ActiveFunctionType af,
        std::vector<int>& int_vector, std::vector<real>& real_vector, std::vector<Matrix>& matrix_vector, real a = 1, real r = 0);

    //正向统一由X以及其他参数生成A，前两个参数必定是X，A；反向由A，DA，X生成DX以及其他参数（例如dW），前4个参数必定是A，DA，X，DX
    static void poolingForward(const Matrix& A, Matrix& R, PoolingType pooling_type,
        const std::vector<int>& window, const std::vector<int>& stride, const std::vector<int>& padding, realc a = 1, realc r = 0);
    static void poolingBackward(Matrix& A, const Matrix& R, PoolingType pooling_type,
        const std::vector<int>& window, const std::vector<int>& stride, const std::vector<int>& padding, realc a = 1, realc r = 0);

    enum
    {
        conv_method_count = 8
    };
    static void convolutionForward(const Matrix& A, const Matrix& W, Matrix& R, Matrix& workspace,
        int& method, const std::vector<int>& stride, const std::vector<int>& padding, realc a = 1, realc r = 0);
    static void convolutionBackward(Matrix& A, Matrix& W, const Matrix& R, Matrix& workspace,
        int& method_dx, int& method_dw, const std::vector<int>& stride, const std::vector<int>& padding, realc a = 1, realc r = 0);

    static void dropoutForward(const Matrix& A, Matrix& R, ActivePhaseType work_phase, real v, int seed, Matrix& rg_stat, Matrix& reverse_space);
    static void dropoutBackward(Matrix& A, const Matrix& R, real v, int seed, Matrix& rg_stat, Matrix& reverse_space);

    //GPU only ----------------------------------------------------------------------------------------------------

    //以下带有可以训练调节的参数
    static void batchNormalizationForward(const Matrix& X, Matrix& A, ActivePhaseType work_phase, BatchNormalizationType bn_type,
        real& exp_aver_factor, real epsilon, Matrix& scale, Matrix& bias, Matrix& result_running_mean, Matrix& result_running_variance,
        Matrix& result_save_mean, Matrix& result_save_inv_variance);
    static void batchNormalizationBackward(Matrix& X, const Matrix& A, ActivePhaseType work_phase, BatchNormalizationType bn_type,
        real epsilon, real rate, Matrix& scale, Matrix& bias, Matrix& saved_mean, Matrix& saved_inv_variance, Matrix& result_dscale, Matrix& result_dbias);

    //GPU only ----------------------------------------------------------------------------------------------------

    //此处计算出的ada_d才是实际的更新梯度，之后应将其加在参数上
    static void adaDeltaUpdate(Matrix& mean_d2, Matrix& mean_ada_d2, Matrix& d, Matrix& ada_d, real rou, real epsilon);
    static void adamUpdate(Matrix& mean_d, Matrix& mean_d2, Matrix& d, Matrix& ada_d, real beta1, real beta2, real epsilon, real t);
    static void adaRMSPropUpdate(Matrix& mean_d2, Matrix& d, Matrix& ada_d, real rou, real epsilon);
    static void sparse(Matrix& rou_hat, Matrix& R, real rou, real beta);

    static void fill(Matrix& m, RandomFillType random_type, int in, int out);

    static void sin(const Matrix& A, Matrix& R, real a = 1);
    static void cos(const Matrix& A, Matrix& R, real a = 1);
    static void zigzag(const Matrix& A, Matrix& R);
    static void zigzagb(Matrix& A, const Matrix& R);

    static void step(const Matrix& A, Matrix& R);
};

}    // namespace woco