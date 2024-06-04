﻿#pragma once
#include "Matrix.h"

namespace cccc
{

//该类中均不是矩阵基本计算，全部为静态函数
class DLL_EXPORT MatrixEx : public Matrix
{
private:
    MatrixEx() = delete;

public:
    //以下函数不属于矩阵基本运算

    //按channel加偏置
    static void addBias(const Matrix& X, const Matrix& bias, Matrix& Y, float a = 1, float b = 1);
    static void addBiasBackward(Matrix& X, Matrix& bias, const Matrix& Y, float a = 1, float b = 1);

    // the function is private for concat the data and append the data
    static void concatByChannel(const std::vector<MatrixSP>& X_vector, Matrix& Y);
    static void concatByChannelBackward(std::vector<MatrixSP>& X_vector, const Matrix& Y);
    static void splitByChannel(const Matrix& X, std::vector<Matrix>& Y_vector);

    static void activeBufferInit(const Matrix& X, ActiveFunctionType af, std::vector<int>& int_vector, std::vector<float>& real_vector, std::vector<Matrix>& matrix_vector);

    //激活的实际计算
    //激活和反向激活中，输入和输出矩阵都是同维度
    //请注意反向的情况，常数a和r的含义与正向的对应关系不同
    static void activeForward(const Matrix& X, Matrix& Y, ActiveFunctionType af,
        std::vector<int>& int_vector, std::vector<float>& real_vector, std::vector<Matrix>& matrix_vector, float a = 1, float r = 0);
    static void activeBackward(Matrix& X, const Matrix& Y, ActiveFunctionType af,
        std::vector<int>& int_vector, std::vector<float>& real_vector, std::vector<Matrix>& matrix_vector, float a = 1, float r = 0);

    static void activeForwardSimple(const Matrix& X, Matrix& Y, ActiveFunctionType af, float a = 1, float r = 0);
    static void activeBackwardSimple(Matrix& X, const Matrix& Y, ActiveFunctionType af, float a = 1, float r = 0);

    static void poolingForward(const Matrix& X, Matrix& Y, PoolingType pooling_type, int reverse, std::vector<Matrix>& workspaces,
        const std::vector<int>& window, const std::vector<int>& stride, const std::vector<int>& padding, float a = 1, float r = 0);
    static void poolingBackward(Matrix& X, const Matrix& Y, PoolingType pooling_type, int reverse, std::vector<Matrix>& workspaces,
        const std::vector<int>& window, const std::vector<int>& stride, const std::vector<int>& padding, float a = 1, float r = 0);

    enum
    {
        conv_method_count = 8
    };
    struct ConvMethod
    {
    };
    static void convolutionForward(const Matrix& X, const Matrix& W, Matrix& Y, std::vector<int>& methods, std::vector<Matrix>& workspaces,
        const std::vector<int>& stride, const std::vector<int>& padding, float a = 1, float r = 0);
    static void convolutionBackward(Matrix& X, Matrix& W, const Matrix& Y, std::vector<int>& methods, std::vector<Matrix>& workspaces,
        const std::vector<int>& stride, const std::vector<int>& padding, float a = 1, float rx = 0, float rw = 0);

    static void dropoutForward(const Matrix& X, Matrix& Y, ActivePhaseType work_phase, float v, int seed, Matrix& rg_stat, Matrix& reverse_space);
    static void dropoutBackward(Matrix& X, const Matrix& Y, float v, int seed, Matrix& rg_stat, Matrix& reverse_space);

    //GPU only ----------------------------------------------------------------------------------------------------

    //以下带有可以训练调节的参数
    static void batchNormalizationForward(const Matrix& X, Matrix& Y, ActivePhaseType work_phase, BatchNormalizationType bn_type,
        float& exp_aver_factor, float epsilon, Matrix& scale, Matrix& bias, Matrix& result_running_mean, Matrix& result_running_variance,
        Matrix& result_save_mean, Matrix& result_save_inv_variance);
    static void batchNormalizationBackward(Matrix& X, const Matrix& Y, ActivePhaseType work_phase, BatchNormalizationType bn_type,
        float epsilon, float rate, Matrix& scale, Matrix& bias, Matrix& saved_mean, Matrix& saved_inv_variance, Matrix& result_dscale, Matrix& result_dbias);

    //GPU only ----------------------------------------------------------------------------------------------------

    //此处计算出的ada_d才是实际的更新梯度，之后应将其加在参数上
    static void adaDeltaUpdate(Matrix& mean_d2, Matrix& mean_ada_d2, Matrix& d, Matrix& ada_d, float rou, float epsilon);
    static void adamUpdate(Matrix& mean_d, Matrix& mean_d2, Matrix& d, Matrix& ada_d, float beta1, float beta2, float epsilon, float t);
    static void adaRMSPropUpdate(Matrix& mean_d2, Matrix& d, Matrix& ada_d, float rou, float epsilon);
    static void sparse(Matrix& rou_hat, Matrix& R, float rou, float beta);

    static void fill(Matrix& m, RandomFillType random_type, int in, int out);

    static void sin(const Matrix& X, Matrix& Y, float a = 1);
    static void cos(const Matrix& X, Matrix& Y, float a = 1);
    static void zigzag(const Matrix& X, Matrix& Y);
    static void zigzagb(Matrix& X, const Matrix& Y);

    static void step(const Matrix& X, Matrix& Y);

    static void leaky_relu(const Matrix& X, Matrix& Y, float l, float a = 1, float b = 0);
    static void leaky_relub(Matrix& X, const Matrix& Y, float l, float a = 1, float b = 0);

    static void correlationForward(const Matrix& X, const Matrix& W, Matrix& Y, std::vector<int>& methods, std::vector<Matrix>& workspaces,
        const std::vector<int>& stride, const std::vector<int>& padding, float a = 1, float r = 0);
    static void correlationBackward(Matrix& X, Matrix& W, const Matrix& Y, std::vector<int>& methods, std::vector<Matrix>& workspaces,
        const std::vector<int>& stride, const std::vector<int>& padding, float a = 1, float rx = 0, float rw = 0);
    static void matrix_max(const Matrix& X1, const Matrix& X2, Matrix& Y);
    static void matrix_maxb(Matrix& X1, Matrix& X2, const Matrix& Y, float a1, float a2, float r);
};

}    // namespace cccc