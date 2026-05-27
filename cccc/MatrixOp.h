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
    BATCHED_MUL,
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
    LAYER_NORM,
    POOL_CHANNEL,
    LOSS,
    FOCAL,
    ZERO_LIMIT,
    L2,
    PREPEND_TOKEN,
    FIRST_TOKEN,
    RMS_NORM,
    PERMUTE,
    ROPE,
    KV_CACHE,
    PIXEL_SHUFFLE,
    PRINT_RATIO,
    PRINT_MESSAGE,
    ATTENTION,
    EMBED,
    TILE,
    DECONV,
    GROUP_NORM,
    REPARAM,
    MSE_LOSS,
    L1_LOSS,
    KL_LOSS,
    UPSAMPLE,
    CHUNK,               // 沿 width(axis=0) 取第 chunk_i 块: as_chunk(X, Y, chunk_i, n_total)
    SIN_TIME_EMBED,      // 正弦时间步嵌入: sinTimeEmbed(t_scalar, d [, base])
    ROPE_INTERLEAVED,    // interleaved RoPE (ncnn mode=1): y[2i]=x[2i]*c-x[2i+1]*s
    DEBUG_SAVE,          // 调试用: 将矩阵保存为 float32 binary 文件
    ROI_ALIGN,           // ROI Align bilinear sampling: roiAlign(feat, boxes, roi_size, spatial_scale)
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
            { MatrixOpType::BATCHED_MUL, "batched_mul" },
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
            { MatrixOpType::LAYER_NORM, "layer_norm" },
            { MatrixOpType::POOL_CHANNEL, "pool_channel" },
            { MatrixOpType::LOSS, "loss" },
            { MatrixOpType::FOCAL, "focal" },
            { MatrixOpType::ZERO_LIMIT, "zero_limit" },
            { MatrixOpType::L2, "l2" },
            { MatrixOpType::PREPEND_TOKEN, "prepend_token" },
            { MatrixOpType::FIRST_TOKEN, "first_token" },
            { MatrixOpType::RMS_NORM, "rms_norm" },
            { MatrixOpType::PERMUTE, "permute" },
            { MatrixOpType::ROPE, "rope" },
            { MatrixOpType::ROPE_INTERLEAVED, "rope_interleaved" },
            { MatrixOpType::KV_CACHE, "kv_cache" },
            { MatrixOpType::PIXEL_SHUFFLE, "pixel_shuffle" },
            { MatrixOpType::PRINT_RATIO, "print_ratio" },
            { MatrixOpType::PRINT_MESSAGE, "print_message" },
            { MatrixOpType::ATTENTION, "attention" },
            { MatrixOpType::EMBED, "embed" },
            { MatrixOpType::TILE, "tile" },
            { MatrixOpType::DECONV, "deconv" },
            { MatrixOpType::GROUP_NORM, "group_norm" },
            { MatrixOpType::REPARAM, "reparam" },
            { MatrixOpType::MSE_LOSS, "mse_loss" },
            { MatrixOpType::L1_LOSS, "l1_loss" },
            { MatrixOpType::KL_LOSS, "kl_loss" },
            { MatrixOpType::UPSAMPLE, "upsample" },
            { MatrixOpType::CHUNK, "chunk" },
            { MatrixOpType::SIN_TIME_EMBED, "sin_time_embed" },
            { MatrixOpType::DEBUG_SAVE, "debug_save" },
            { MatrixOpType::ROI_ALIGN, "roi_align" },
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

    const std::vector<int>& getStride() const { return stride_; }      // 图模式用：读取卷积/池化 stride
    const std::vector<int>& getPadding() const { return padding_; }    // 图模式用：读取卷积/池化 padding
    const std::vector<int>& getWindow() const { return window_; }      // 图模式用：读取池化 window
    // Set a value in the window_ vector (for KV-cache / RoPE offset control)
    void setWindow(int idx, int val)
    {
        if (idx >= (int)window_.size()) { window_.resize(idx + 1, 0); }
        window_[idx] = val;
    }
    const std::vector<float>& getA() const { return a_; }    // 图模式用：读取算子标量参数 a
    const std::vector<float>& getB() const { return b_; }    // 图模式用：读取算子标量参数 b

    ActiveFunctionType getActiveType() const;
    PoolingType getPoolingType() const;                     // 图模式用：获取 POOL op 的池化类型
    bool isConnectLoss() const { return connect_loss_; }    // 图模式用：该 op 是否处于 loss 的反向路径上
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
    void as_mul(const MatrixSP& X1, const MatrixSP& X2, const MatrixSP& Y, float a = 1, std::vector<int> dim = {}, MatrixTransType ta = MATRIX_NO_TRANS, MatrixTransType tb = MATRIX_NO_TRANS);
    //批量矩阵乘 (用于 self-attention).
    //每个 batch 切片 A: (M,K) col-major, B: (K,N) col-major, Y: (M,N) col-major.
    //M/N/K/batch 自动从矩阵维度推导:
    //  ta=NO_TRANS: M=X1.width_, K=X1.height_; ta=TRANS: M=X1.height_, K=X1.width_
    //  tb=NO_TRANS: N=X2.height_;              tb=TRANS: N=X2.width_
    //  batch = X1.number_
    void as_batchedMul(const MatrixSP& X1, const MatrixSP& X2, const MatrixSP& Y,
        MatrixTransType ta = MATRIX_NO_TRANS, MatrixTransType tb = MATRIX_NO_TRANS, float a = 1);
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
    //允许改 batch 维的 reshape: 不强制 dim.back() = X.number_, 总元素数必须一致
    void as_reshape_batch(const MatrixSP& X, const MatrixSP& Y, std::vector<int>& dim);
    void as_max(const MatrixSP& X1, const MatrixSP& X2, const MatrixSP& Y);
    void as_batchNorm(const MatrixSP& X, const MatrixSP& scale, const MatrixSP& Y, BatchNormalizationType bn_type = BATCH_NORMALIZATION_SPATIAL, float epsilon = 1e-5f);
    //Layer Normalization (Transformer 用)
    //X: 输入, scale/bias: 形状 [width_], Y: 输出
    void as_layerNorm(const MatrixSP& X, const MatrixSP& scale, const MatrixSP& bias, const MatrixSP& Y, float epsilon = 1e-5f);
    void as_poolChannel(const MatrixSP& X, const MatrixSP& Y, PoolingType pooling_type, PoolingReverseType reverse_type, float a = 1);

    //ViT CLS token 辅助: 将学习型 CLS (D,1,1,1) 广播到所有 batch 并拼到 X (D,T,1,B) 序列头部 -> Y (D,T+1,1,B)
    void as_prependToken(const MatrixSP& X, const MatrixSP& cls, const MatrixSP& Y);
    //从序列 X (D,T,1,B) 中取出第 0 个 token -> Y (D,1,1,B)
    void as_firstToken(const MatrixSP& X, const MatrixSP& Y);

    //RMS Normalization (LLM 常用, 无均值, 无 bias)
    //scale: 形状 [width_]; 沿 inner=width_ 轴归一化
    void as_rmsNorm(const MatrixSP& X, const MatrixSP& scale, const MatrixSP& Y, float epsilon = 1e-6f);
    //4 维任意轴置换: dim_out[i] = dim_in[perm[i]], perm 长度=4
    void as_permute(const MatrixSP& X, const MatrixSP& Y, const std::vector<int>& perm);
    //RoPE: X/Y 形状 (D,T,1,B); cos_tab/sin_tab 形状 (D/2, T, 1, 1)
    void as_rope(const MatrixSP& X, const MatrixSP& cos_tab, const MatrixSP& sin_tab, const MatrixSP& Y);
    void as_rope_interleaved(const MatrixSP& X, const MatrixSP& cos_tab, const MatrixSP& sin_tab, const MatrixSP& Y);
    //KV cache (推理用): X_new (D, T_new, H, B) 写入 cache (D, T_max, H, B) 在当前 pos 偏移处.
    //Y 与 cache 共享内存, 形状=cache.dim. 内部用 window_={pos, T_max} 维护当前位置, 每次 forward 自动前移.
    //无反向 (推理算子). 调用 resetKVCache() 重置 pos.
    void as_kvcache(const MatrixSP& X_new, const MatrixSP& cache, const MatrixSP& Y);
    //把当前 op 视为 KV_CACHE, 重置内部 pos=0
    void resetKVCache();
    //批量重置 op_queue 中所有 KV_CACHE 算子的 pos
    static void resetKVCache(std::vector<MatrixOp>& op_queue);

    //pixel_shuffle: X (W, H, C_out*r*r, N) -> Y (W*r, H*r, C_out, N)
    void as_pixelShuffle(const MatrixSP& X, const MatrixSP& Y, int r);

    //print_ratio(A, B, dummy, label): 每隔若干 forward 打印 RMS(A)/RMS(B)，纯诊断
    void as_print_ratio(const MatrixSP& A, const MatrixSP& B, const MatrixSP& dummy, const std::string& label = "attn_ratio");

    //print_message(X [, label]): 每次 forward 打印 X 的 Dim/L1/L2，纯诊断，不影响梯度
    void as_print_message(const MatrixSP& X, const MatrixSP& dummy, const std::string& label = "");

    //save_binary(X, filename): 将矩阵保存为 float32 binary 文件，纯诊断，不影响梯度
    //仅在首次 forward 时保存（由外部 save_count 控制）
    void as_save_binary(const MatrixSP& X, const MatrixSP& dummy, const std::string& filename);

    //Scaled Dot-Product Attention: Y = softmax(K^T @ Q / sqrt(dk)) @ V
    //Q/K/V/Y 形状均为 (D, T, 1, B); dk 通常取 D (Head Dim).
    //causal=1 时启用因果掩码（下三角注意力），用于自回归语言模型.
    //bias (可选): (T_k, T_q, 1, B) additive bias 在 softmax 之前加到 scores 上（用于 Box RPB 等）.
    void as_attention(const MatrixSP& Q, const MatrixSP& K, const MatrixSP& V, const MatrixSP& Y, float dk, int causal = 0, const MatrixSP& bias = nullptr);

    //ROI Align bilinear sampling (float only, aligned=True convention).
    //feat: (W,H,C,B) feature map; boxes: (4,N,1,B) [x1,y1,x2,y2] in pixel coords;
    //roi_size: output spatial size; spatial_scale: multiplier to convert box coords to feat coords.
    //output Y: (roi_size, roi_size, C, N*B)
    void as_roi_align(const MatrixSP& feat, const MatrixSP& boxes, const MatrixSP& Y,
        int roi_size, float spatial_scale = 1.0f);

    //Embedding lookup: ids (T,1,1,B) 整数 token id 以 float 存储, W (D,1,1,V) 词表矩阵 -> Y (D,T,1,B)
    //前向: Y[d,t,b] = W[d, (int)ids[t,b]]
    //反向: scatter-add dY 到 W.d(), ids 无梯度
    void as_embed(const MatrixSP& ids, const MatrixSP& W, const MatrixSP& Y);

    //Tile: X (W,H,C,N) 沿各轴重复 repeats[i] 次 -> Y (W*r0, H*r1, C*r2, N*r3)
    //Y[w,h,c,n] = X[w%W, h%H, c%C, n%N]
    //反向: scatter-add dY 到 X.d()
    void as_tile(const MatrixSP& X, const MatrixSP& Y, const std::vector<int>& repeats);

    //转置卷积 (Deconvolution): A[W_in,H_in,C_in,N] x W[kW,kH,C_out,C_in] -> Y[W_out,H_out,C_out,N]
    //C_in = W.getNumber(), C_out = W.getChannel()
    //Y_W = (A_W-1)*stride - 2*padding + kW
    void as_deconv(const MatrixSP& A, const MatrixSP& W, const MatrixSP& Y,
        std::vector<int> stride, std::vector<int> padding, int conv_algo, float a = 1);

    //Group Normalization: X[W,H,C,N] -> Y[W,H,C,N]; scale/bias: [C]; G: 组数
    //C 必须整除 G; 每组归一化 W*H*C/G 个元素; scale/bias 为可训练参数
    void as_groupNorm(const MatrixSP& X, const MatrixSP& scale, const MatrixSP& bias,
        const MatrixSP& Y, int G, float epsilon = 1e-5f);

    //VAE 重参数化: mu[D,1,1,B] + log_var[D,1,1,B] -> z[D,1,1,B]
    //训练时 z = mu + exp(log_var*0.5)*eps, eps~N(0,1); 推理时 z = mu
    void as_reparam(const MatrixSP& mu, const MatrixSP& log_var, const MatrixSP& z);

    //Nearest/Bilinear upsample: X[W,H,C,N] -> Y[W*sw, H*sh, C, N]
    //bilinear=false: 最近邻; bilinear=true: 双线性 (align_corners=False)
    void as_upsample(const MatrixSP& X, const MatrixSP& Y, int sh, int sw, bool bilinear = false);

    //Chunk: 沿 width(axis=0) 将 X 均分为 n_total 块, 取第 chunk_i 块 (0-based)
    //X: (W, H, C, N), Y: (W/n_total, H, C, N)
    void as_chunk(const MatrixSP& X, const MatrixSP& Y, int chunk_i, int n_total);

    //SliceW: 沿 width(axis=0) 取任意范围 [start_w, start_w+size_w)
    //X: (W, H, C, N), Y: (size_w, H, C, N); 不要求等分, 可用于序列维度切片(配合 permute)
    void as_sliceW(const MatrixSP& X, const MatrixSP& Y, int start_w, int size_w);

    //正弦时间步嵌入: t 为标量 (1,1,1,B 或 1,1,1,1), 输出 (d, 1, 1, B)
    //emb[i] = cos(t*freq_i) for i<d/2; emb[i+d/2] = sin(t*freq_i)
    //freq_i = 1/base^(2*i/d)
    void as_sinTimeEmbed(const MatrixSP& t, const MatrixSP& Y, int d, float base = 10000.0f);

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

//MSE 重建损失: mean((A-Y)^2), 梯度 2*(A-Y)/N
std::vector<MatrixOp> mseLoss(const MatrixSP& A, const MatrixSP& Y);
//L1 重建损失: mean(|A-Y|), 梯度 sign(A-Y)/N
std::vector<MatrixOp> l1Loss(const MatrixSP& A, const MatrixSP& Y);
//KL 散度: -0.5*mean(1+log_var-mu^2-exp(log_var)), 先验 N(0,1)
//可通过 klLoss(mu,lv)*beta 实现 β-VAE 缩放
std::vector<MatrixOp> klLoss(const MatrixSP& mu, const MatrixSP& log_var);

}    // namespace cccc

template <typename CharT>
struct std::formatter<cccc::MatrixSP, CharT>
{
    template <typename FormatParseContext>
    auto parse(FormatParseContext& pc)
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