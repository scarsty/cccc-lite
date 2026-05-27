#pragma once
#include "Matrix.h"

namespace cccc
{

//该类中均不是矩阵基本计算，全部为静态函数
class CCCC_EXPORT MatrixEx : public Matrix
{
private:
    MatrixEx() = delete;
    static void attentionForwardImpl(const Matrix& Q, const Matrix& K, const Matrix& V,
        Matrix& Y, const Matrix* bias, float dk, int causal, int pos_offset);

public:
    struct ConvMethod
    {
        int algo = -1, math_type = -1, group_number = 0;
    };

    static const int conv_method_count = 8;

    //以下函数不属于矩阵基本运算

    static void elementMulSum(const Matrix& A, const Matrix& B, Matrix& R, float a = 1, float r = 0);

    //按channel加偏置
    static void addBias(const Matrix& X, const Matrix& bias, Matrix& Y, float a = 1, float b = 1);
    static void addBiasBackward(const Matrix& X, Matrix& bias, const Matrix& Y, float a = 1, float b = 1);

    // the function is private for concat the data and append the data
    static void concatByChannel(const std::vector<MatrixSP>& X_vector, Matrix& Y);
    static void concatByChannelBackward(std::vector<MatrixSP>& X_vector, Matrix& Y);
    static void splitByChannel(const Matrix& X, std::vector<Matrix>& Y_vector);

    // Chunk: 沿 width(axis=0) 取第 chunk_i 块 (共 n_total 块), 不含梯度写回
    // X: (W, H, C, N), Y: (W/n_total, H, C, N), start_w = chunk_i*(W/n_total)
    static void chunkForward(const Matrix& X, Matrix& Y, int start_w, int size_w);
    // Chunk 反向: 将 dY 累加到 dX 的对应偏移处
    static void chunkBackward(Matrix& X, const Matrix& Y, int start_w, int size_w);

    static void activeBufferInit(const Matrix& X, Matrix& Y, ActiveFunctionType af, std::vector<int>& int_vector, std::vector<float>& real_vector);

    //激活的实际计算
    //激活和反向激活中，输入和输出矩阵都是同维度
    //请注意反向的情况，常数a和r的含义与正向的对应关系不同
    static void activeForward(const Matrix& X, Matrix& Y, ActiveFunctionType af,
        std::vector<int>& int_vector, std::vector<float>& real_vector, float a = 1, float r = 0);
    static void activeBackward(Matrix& X, const Matrix& Y, ActiveFunctionType af,
        std::vector<int>& int_vector, std::vector<float>& real_vector, float a = 1, float r = 0);

    static void activeForwardSimple(const Matrix& X, Matrix& Y, ActiveFunctionType af, float a = 1, float r = 0);
    static void activeBackwardSimple(Matrix& X, const Matrix& Y, ActiveFunctionType af, float a = 1, float r = 0);

    static void poolingForward(const Matrix& X, Matrix& Y, PoolingType pooling_type, PoolingReverseType reverse_type,
        const std::vector<int>& window, const std::vector<int>& stride, const std::vector<int>& padding,
        float a = 1, float r = 0);
    static void poolingBackward(Matrix& X, const Matrix& Y, PoolingType pooling_type, PoolingReverseType reverse_type,
        const std::vector<int>& window, const std::vector<int>& stride, const std::vector<int>& padding,
        float a = 1, float r = 0);

    static void poolingChannelForward(Matrix& X, Matrix& Y, PoolingType pooling_type, PoolingReverseType reverse_type, float a = 1, float r = 0);
    static void poolingChannelBackward(Matrix& X, Matrix& Y, PoolingType pooling_type, PoolingReverseType reverse_type, float a = 1, float r = 0);

    static void convolutionForward(const Matrix& X, const Matrix& W, Matrix& Y,
        const std::vector<int>& stride, const std::vector<int>& padding,
        float a = 1, float r = 0);
    static void convolutionBackward(Matrix& X, Matrix& W, const Matrix& Y,
        const std::vector<int>& stride, const std::vector<int>& padding,
        float a = 1, float rx = 0, float aw = 1, float rw = 0);
    static void convolutionBackwardDX(Matrix& X, const Matrix& W, const Matrix& Y,
        const std::vector<int>& stride, const std::vector<int>& padding,
        float a = 1, float r = 0);
    static void convolutionBackwardDW(const Matrix& X, Matrix& W, const Matrix& Y,
        const std::vector<int>& stride, const std::vector<int>& padding,
        float a = 1, float r = 0);

    static void dropoutForward(const Matrix& X, Matrix& Y, float v, int seed);
    static void dropoutBackward(Matrix& X, const Matrix& Y, float v, int seed);

    //GPU only ----------------------------------------------------------------------------------------------------

    //以下带有可以训练调节的参数
    static void batchNormalizationForward(const Matrix& X, Matrix& Y, BatchNormalizationType bn_type,
        float& exp_aver_factor, float epsilon, Matrix& scale, Matrix& bias);
    static void batchNormalizationBackward(Matrix& X, const Matrix& Y, BatchNormalizationType bn_type,
        float epsilon, Matrix& scale, Matrix& bias);

    //Layer Normalization (面向 Transformer)
    //沿 inner = X.width_ 维度做归一化, outer = (X.height_*X.channel_*X.number_)
    //scale / bias 形状: width_ (= inner)
    //内部使用 Y.workspace_ 缓存 mean / invstd / dscale / dbias
    static void layerNormalizationForward(const Matrix& X, Matrix& Y,
        Matrix& scale, Matrix& bias, float epsilon);
    static void layerNormalizationBackward(Matrix& X, const Matrix& Y,
        Matrix& scale, Matrix& bias, float epsilon);

    //RMS Normalization (Transformer/LLM 用, 无均值, 无 bias)
    //沿 inner = X.width_ 维度做归一化
    //scale 形状: width_ (= inner)
    //内部使用 Y.workspace_ 缓存 invstd / dscale
    static void rmsNormForward(const Matrix& X, Matrix& Y,
        Matrix& scale, float epsilon);
    static void rmsNormBackward(Matrix& X, const Matrix& Y,
        Matrix& scale, float epsilon);

    //4 维任意轴置换 (W, H, C, N), perm 长度=4
    //out_dims[i] = in_dims[perm[i]]
    static void permute4dForward(const Matrix& X, Matrix& Y, const std::vector<int>& perm);
    static void permute4dBackward(Matrix& X, const Matrix& Y, const std::vector<int>& perm);

    //RoPE (旋转位置编码, half-rotate 风格)
    //X / Y: (D, T, 1, B), cos/sin: (D/2, T_max) 共享所有 batch
    //pos_offset: absolute position of first token (for KV-cache decode)
    static void ropeForward(const Matrix& X, Matrix& Y, const Matrix& cos_tab, const Matrix& sin_tab, int pos_offset = 0);
    static void ropeInterleavedForward(const Matrix& X, Matrix& Y, const Matrix& cos_tab, const Matrix& sin_tab, int pos_offset = 0);
    static void ropeBackward(Matrix& X, const Matrix& Y, const Matrix& cos_tab, const Matrix& sin_tab);

    //pixel_shuffle: X (W, H, C_out*r*r, N) -> Y (W*r, H*r, C_out, N)
    static void pixelShuffleForward(const Matrix& X, Matrix& Y, int r);
    static void pixelShuffleBackward(Matrix& X, const Matrix& Y, int r);

    //Scaled Dot-Product Attention: Y = softmax_channel(K^T@Q/sqrt(dk)) @ V
    //Q shape (D, T_q, 1, B); K/V shape (D, T_k, 1, B); Y shape (D, T_q, 1, B)
    //pos_offset: absolute position of first Q token (for KV-cache decode causal mask)
    //workspace: Y.workspace_[0] = attn (T_k,T_q,1,B), [1] = dAttn, [2] = dScores
    static void attentionForward(const Matrix& Q, const Matrix& K, const Matrix& V, Matrix& Y, float dk, int causal = 0, int pos_offset = 0);
    static void attentionForward(const Matrix& Q, const Matrix& K, const Matrix& V, Matrix& Y, const Matrix& bias, float dk, int causal = 0, int pos_offset = 0);
    static void attentionBackward(Matrix& Q, Matrix& K, Matrix& V, const Matrix& Y, float dk, int causal = 0);
    // ROI Align forward/backward (float only, aligned=True convention)
    // feat: (W,H,C,B); boxes: (4,N,1,B) [x1,y1,x2,y2]; Y: (roi_size,roi_size,C,N*B)
    static void roiAlignForward(const Matrix& feat, const Matrix& boxes, Matrix& Y,
        int roi_size, float spatial_scale = 1.0f);
    static void roiAlignBackward(const Matrix& feat, const Matrix& boxes, Matrix& Y,
        int roi_size, float spatial_scale = 1.0f);

    //Embedding lookup: ids (T,1,1,B) float-as-int, W (D,1,1,V) -> Y (D,T,1,B)
    //前向为查表 (gather), 反向为 scatter-add 到 W.d()
    static void embedForward(const Matrix& ids, const Matrix& W, Matrix& Y);
    static void embedBackward(const Matrix& ids, Matrix& W, const Matrix& Y);

    //Tile: X (W,H,C,N) 各轴重复 repeats[4] 次 -> Y
    //Y[w,h,c,n] = X[w%W_in, h%H_in, c%C_in, n%N_in]
    //反向为 scatter-add 到 X.d()
    static void tileForward(const Matrix& X, Matrix& Y, const std::vector<int>& repeats);
    static void tileBackward(Matrix& X, const Matrix& Y, const std::vector<int>& repeats);

    //转置卷积 (Deconvolution / Transposed Convolution)
    //A: [W_in, H_in, C_in, N] 其中 C_in = W.getNumber()
    //W: 滤波器 [kW, kH, C_out, C_in]; C_out = W.getChannel()
    //Y: [W_out, H_out, C_out, N] 其中 W_out = (W_in-1)*stride + kW - 2*padding
    //forward 使用 cuDNN BackwardData API; backward dA 用 ForwardConv, backward dW 用 BackwardFilter
    static void deconvolutionForward(const Matrix& A, const Matrix& W, Matrix& Y,
        const std::vector<int>& stride, const std::vector<int>& padding, float a = 1, float r = 0);
    static void deconvolutionBackwardDA(Matrix& A, const Matrix& W, const Matrix& Y,
        const std::vector<int>& stride, const std::vector<int>& padding, float a = 1, float r = 0);
    static void deconvolutionBackwardDW(const Matrix& A, Matrix& W, const Matrix& Y,
        const std::vector<int>& stride, const std::vector<int>& padding, float a = 1, float r = 0);

    //Group Normalization (图像生成模型常用)
    //X: [W,H,C,N], scale/bias: [C], Y: [W,H,C,N]; G 为组数, C 必须整除 G
    //归一化维度: 每组 W*H*C/G 个元素; 使用 Y.workspace_ 缓存 mean/invstd/X_hat/dscale/dbias
    static void groupNormForward(const Matrix& X, Matrix& Y,
        const Matrix& scale, const Matrix& bias, int G, float epsilon = 1e-5f);
    static void groupNormBackward(Matrix& X, const Matrix& Y,
        const Matrix& scale, const Matrix& bias, int G, float epsilon = 1e-5f);

    //VAE 重参数化 (Reparameterization Trick)
    //mu, log_var: [D,1,1,B]; z: [D,1,1,B]
    //前向 (训练): z = mu + exp(log_var*0.5) * eps, eps~N(0,1) 存于 z.workspace_[0]
    //前向 (推理): z = mu (确定性输出)
    //反向: dmu += dz; d(log_var) += dz * exp(log_var*0.5) * 0.5 * eps
    static void reparamForward(const Matrix& mu, const Matrix& log_var, Matrix& z);
    static void reparamBackward(Matrix& mu, Matrix& log_var, const Matrix& z);

    //L1 loss backward: dA[i] = beta*dA[i] + alpha*sign(A[i]-Y[i])
    static void l1LossBackward(Matrix& A, const Matrix& Y, float alpha, float beta);
    //KL log_var backward: dlv[i] = beta*dlv[i] + alpha*0.5*(exp(lv[i])-1)
    static void klLvBackward(Matrix& log_var, float alpha, float beta);

    //Nearest neighbor / bilinear upsample
    //X: [W,H,C,N] -> Y: [W*sw, H*sh, C, N]
    //bilinear=false: 最近邻; bilinear=true: 双线性 (align_corners=False)
    static void upsampleForward(const Matrix& X, Matrix& Y, int sh, int sw, bool bilinear = false);
    //反向: 最近邻累加; 双线性 atomicAdd (caller 须先按 keepWeight 缩放 X.d())
    static void upsampleBackward(Matrix& X, const Matrix& Y, int sh, int sw, bool bilinear = false);

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

    static void correlationForward(const Matrix& X, const Matrix& W, Matrix& Y,
        const std::vector<int>& stride, const std::vector<int>& padding,
        float a = 1, float r = 0);
    static void correlationBackward(Matrix& X, Matrix& W, const Matrix& Y, std::vector<int>& methods,
        const std::vector<int>& stride, const std::vector<int>& padding, float a = 1, float rx = 0, float rw = 0);
    static void matrix_max(const Matrix& X1, const Matrix& X2, Matrix& Y);
    static void matrix_maxb(Matrix& X1, Matrix& X2, const Matrix& Y, float a1, float a2, float r);

    static void zero_limit(const Matrix& A, const Matrix& B, Matrix& R, float beta_a, float beta_b);
};

}    // namespace cccc