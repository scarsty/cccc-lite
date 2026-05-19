#pragma once

#include "Matrix.h"
#include "MatrixOp.h"
#include "gpu_lib.h"

#include <cstdint>
#include <initializer_list>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#if ENABLE_CUDA

namespace cccc
{

// =============================================================================
// CudnnBackendDesc
//   对 cudnnBackendDescriptor_t 的 RAII 包装。
//   - 构造时调用 cudnnBackendCreateDescriptor
//   - 析构时调用 cudnnBackendDestroyDescriptor
//   - 提供按类型的 setXxx() / getXxx() 链式接口
//   - finalize() 调用 cudnnBackendFinalize
// 不可拷贝，可移动。
// =============================================================================
class CudnnBackendDesc
{
private:
    cudnnBackendDescriptor_t desc_ = nullptr;
    cudnnBackendDescriptorType_t type_;
    bool finalized_ = false;

public:
    explicit CudnnBackendDesc(cudnnBackendDescriptorType_t type);
    ~CudnnBackendDesc();

    CudnnBackendDesc(const CudnnBackendDesc&) = delete;
    CudnnBackendDesc& operator=(const CudnnBackendDesc&) = delete;
    CudnnBackendDesc(CudnnBackendDesc&& o) noexcept;
    CudnnBackendDesc& operator=(CudnnBackendDesc&& o) noexcept;

    cudnnBackendDescriptor_t get() const { return desc_; }
    cudnnBackendDescriptor_t operator()() const { return desc_; }
    cudnnBackendDescriptorType_t type() const { return type_; }
    bool finalized() const { return finalized_; }

    // 通用 setAttribute，count 是元素数；data 指向 count 个 attributeType 类型元素
    CudnnBackendDesc& setAttribute(cudnnBackendAttributeName_t name,
        cudnnBackendAttributeType_t attr_type,
        int64_t count,
        const void* data);

    // ---- 按类型的链式 setter ----
    CudnnBackendDesc& setI64(cudnnBackendAttributeName_t name, int64_t v);
    CudnnBackendDesc& setI64Array(cudnnBackendAttributeName_t name, const std::vector<int64_t>& v);
    CudnnBackendDesc& setI64Array(cudnnBackendAttributeName_t name, std::initializer_list<int64_t> v);
    CudnnBackendDesc& setI32(cudnnBackendAttributeName_t name, int32_t v);
    CudnnBackendDesc& setF32(cudnnBackendAttributeName_t name, float v);
    CudnnBackendDesc& setF64(cudnnBackendAttributeName_t name, double v);
    CudnnBackendDesc& setBool(cudnnBackendAttributeName_t name, bool v);
    CudnnBackendDesc& setDataType(cudnnBackendAttributeName_t name, cudnnDataType_t dt);
    CudnnBackendDesc& setHandle(cudnnBackendAttributeName_t name, cudnnHandle_t h);

    // 子描述符（如 OPERATION_GRAPH 引用 OPS、ENGINEHEUR 引用 OPERATION_GRAPH 等）
    CudnnBackendDesc& setDesc(cudnnBackendAttributeName_t name, cudnnBackendDescriptor_t d);
    CudnnBackendDesc& setDesc(cudnnBackendAttributeName_t name, const CudnnBackendDesc& d);
    CudnnBackendDesc& setDescArray(cudnnBackendAttributeName_t name, const std::vector<cudnnBackendDescriptor_t>& v);

    // VARIANT_PACK 用
    CudnnBackendDesc& setVoidPtrArray(cudnnBackendAttributeName_t name, const std::vector<void*>& v);

    // 枚举
    CudnnBackendDesc& setPointwiseMode(cudnnBackendAttributeName_t name, cudnnPointwiseMode_t m);
    CudnnBackendDesc& setNormMode(cudnnBackendAttributeName_t name, cudnnBackendNormMode_t m);
    CudnnBackendDesc& setNormFwdPhase(cudnnBackendAttributeName_t name, cudnnBackendNormFwdPhase_t p);
    CudnnBackendDesc& setHeurMode(cudnnBackendAttributeName_t name, cudnnBackendHeurMode_t m);

    // ---- getter ----
    int64_t getI64(cudnnBackendAttributeName_t name) const;
    int64_t getElementCount(cudnnBackendAttributeName_t name, cudnnBackendAttributeType_t attr_type) const;

    // 调用 cudnnBackendFinalize
    CudnnBackendDesc& finalize();
};

// =============================================================================
// CudnnGraphTensor
//   Tensor 描述符工厂。约定：
//   - cccc Matrix 的 dim 顺序为 {W, H, C, N}（W 为最快变化轴，列主序内存）
//   - cuDNN graph 期望 dims 为外→内顺序，stride 也按外→内
//   - 故由 Matrix 生成时，将 dim 反转为 [N, C, H, W]，stride 取 packed NCHW
//   uid 必须全局唯一，是后续 VARIANT_PACK 中数据指针的索引键
// =============================================================================
class CudnnGraphTensor
{
public:
    // 由 Matrix 自动推断（dims/strides/dtype）。alignment 默认 16 字节
    static CudnnBackendDesc fromMatrix(const Matrix& m, int64_t uid,
        bool is_virtual = false, int64_t alignment = 16);

    // 手动指定 dims/strides，全部按外→内顺序
    static CudnnBackendDesc make(int64_t uid, cudnnDataType_t dt,
        const std::vector<int64_t>& dims,
        const std::vector<int64_t>& strides,
        bool is_virtual = false, int64_t alignment = 16);

    // 标量虚张量，常用作 alpha/scale/eps
    static CudnnBackendDesc makeScalar(int64_t uid, cudnnDataType_t dt, bool is_virtual = false);

    // 由 cccc dim {W,H,C,N} 计算出 NCHW packed strides
    static std::vector<int64_t> packedStridesNCHW(const std::vector<int64_t>& dims);

    // bias broadcast: 通常 bias.shape={W:1,H:1,C:c,N:1}（cccc 顺序），
    // 这里直接生成 dims=[1,c,1,1]、strides=[c,1,0,0]（外→内顺序的 strides，
    // 后两维 stride=0 实现 H/W 维度广播）。
    static CudnnBackendDesc fromBiasMatrix(const Matrix& bias, int64_t uid,
        int64_t alignment = 16);
};

// =============================================================================
// CudnnGraphPlan
//   端到端组装：OperationGraph → EngineHeur → EngineCfg → ExecutionPlan
//   构造完成后即可执行；execute() 把 (uid, ptr) 列表打包成 VariantPack 并调用。
// =============================================================================
class CudnnGraphPlan
{
private:
    cudnnHandle_t handle_ = nullptr;
    std::unique_ptr<CudnnBackendDesc> op_graph_;
    std::unique_ptr<CudnnBackendDesc> heur_;
    std::unique_ptr<CudnnBackendDesc> engine_;
    std::unique_ptr<CudnnBackendDesc> engine_cfg_;
    std::unique_ptr<CudnnBackendDesc> plan_;
    int64_t workspace_size_ = 0;
    bool ok_ = false;

public:
    // operations 中元素须是已 finalize 的 OPERATION_*_DESCRIPTOR
    CudnnGraphPlan(cudnnHandle_t handle,
        const std::vector<cudnnBackendDescriptor_t>& operations,
        cudnnBackendHeurMode_t heur_mode = CUDNN_HEUR_MODE_A);

    bool ok() const { return ok_; }
    int64_t workspaceSize() const { return workspace_size_; }

    // 执行：uids 与 ptrs 一一对应，workspace 容量需 >= workspaceSize()
    int execute(const std::vector<int64_t>& uids,
        const std::vector<void*>& ptrs,
        void* workspace);

    // 重载：直接用 Matrix 列表，自动取 getDataPtr()。避免调用方手工拆包。
    // 用 const Matrix* 列表（不持有所有权，零拷贝）。
    int execute(const std::vector<int64_t>& uids,
        const std::vector<const Matrix*>& mats,
        void* workspace);

    // 重载：MatrixSP 列表（共享指针，避免临时拷贝）。
    int execute(const std::vector<int64_t>& uids,
        const std::vector<MatrixSP>& mats,
        void* workspace);
};

// 工具函数：cccc DataType → cudnnDataType_t（在 gpu_lib.h 已有 toCudnnDataType，
// 此处提供对外可见的便捷别名）
inline cudnnDataType_t cudnnTypeOf(DataType dt) { return toCudnnDataType(dt); }

// =============================================================================
// CudnnGraphOps
//   高层算子封装：使用 cuDNN backend graph API 实现常用算子。
//   - 内部按 (dtype, X.dim, W/window, Y.dim, stride, padding, alpha/beta is_zero) 缓存
//     CudnnGraphPlan，避免每次重建。
//   - 工作空间从所属 GpuControl::user_data_ 中取（与 MatrixEx 保持一致）。
//   - enabled() 默认 false；可由环境变量 CCCC_USE_GRAPH_CONV_POOL=1 或
//     setEnabled(true) 打开。
// =============================================================================
class CCCC_EXPORT CudnnGraphOps
{
public:
    // 前向卷积 Y = alpha * conv(X, W) + beta * Y
    // 返回 0 成功，其它为错误码。
    static int convForward(const Matrix& X, const Matrix& W, Matrix& Y,
        const std::vector<int>& stride, const std::vector<int>& padding,
        float alpha, float beta);

    // 反向卷积——数据梯度: dX = alpha * conv_bwd_data(W, dY) + beta * dX
    static int convBackwardData(const Matrix& W, const Matrix& dY, Matrix& dX,
        const std::vector<int>& stride, const std::vector<int>& padding,
        float alpha, float beta);

    // 反向卷积——权重梯度: dW = alpha * conv_bwd_filter(X, dY) + beta * dW
    static int convBackwardFilter(const Matrix& X, const Matrix& dY, Matrix& dW,
        const std::vector<int>& stride, const std::vector<int>& padding,
        float alpha, float beta);

    // 前向池化（max / average）。reverse_type 暂不支持，仅 NOT_REVERSE。
    static int poolForward(const Matrix& X, Matrix& Y, int pooling_type,
        const std::vector<int>& window, const std::vector<int>& stride,
        const std::vector<int>& padding, float alpha, float beta);

    // 前向激活（仅支持 RELU / SIGMOID / TANH，对应 cccc 的
    // ACTIVE_FUNCTION_RELU/SIGMOID/TANH，其它返回 -200 由调用方 fallback）
    static int actForward(const Matrix& X, Matrix& Y, int af, float alpha, float beta);

    // 反向激活: dX = alpha * dY * f'(X) + beta * dX
    static int actBackward(const Matrix& X, const Matrix& Y,
        const Matrix& dY, Matrix& dX, int af, float alpha, float beta);

    // Bias add via POINTWISE_ADD（依赖 broadcast 的 stride=0 通道）：
    //   Y = alpha * X + beta * bias  （bias 通常 shape=[1,C,1,1]）
    // 注：cccc 旧实现先 copyData(X→Y) 再 cudnnAddTensor(bias)。这里直接合成一步，
    //   对调用 addBias 时 X==Y 的情形也安全（POINTWISE_ADD 允许同 ptr）。
    static int biasAdd(const Matrix& X, const Matrix& bias, Matrix& Y,
        float alpha, float beta);

    // 融合：Conv + BiasAdd 一张图（CONV_FWD → POINTWISE_ADD），
    //   Y = conv(X, W) + bias  （bias shape=[1,C,1,1]）
    static int convBiasForward(const Matrix& X, const Matrix& W, const Matrix& bias, Matrix& Y,
        const std::vector<int>& stride, const std::vector<int>& padding);

    // 融合：Conv + BiasAdd + Activation 一张图（CONV_FWD → POINTWISE_ADD → POINTWISE_act），
    //   Y = act(conv(X, W) + bias)
    //   act_mode: 1=RELU, 0=SIGMOID, 2=TANH（对应 ACTIVE_FUNCTION_*）
    static int convBiasActForward(const Matrix& X, const Matrix& W, const Matrix& bias, Matrix& Y,
        const std::vector<int>& stride, const std::vector<int>& padding, int act_mode);

    // 融合：MatMul + BiasAdd[+Activation] 一张图（MATMUL → POINTWISE_ADD →[POINTWISE_act]），
    //   Y = M * X + bias        （act_mode==-1 时省略激活）
    //   Y = act(M * X + bias)
    //   M: weight (row=out, number=in)；X: input (row=in, number=batch)；
    //   bias: shape (out, 1)（dataSize == out）；Y: (out, 1, 1, batch)。
    //   按 cccc::Matrix::mul 的语义这是一次 col-major GEMM，cuDNN matmul 支持
    //   (B,M,K)×(B,K,N)，此处取 B=1，M=out, K=in, N=batch。
    static int matmulBiasForward(const Matrix& M, const Matrix& X, const Matrix& bias, Matrix& Y);
    static int matmulBiasActForward(const Matrix& M, const Matrix& X, const Matrix& bias, Matrix& Y, int act_mode);

    // ---- 反向（option A：activeBackward + ConvBackward 二算子融合） ----
    // 都假设 forward 时 ACTIVE 紧跟在 ADD_BIAS 后面（即 Y_act = act(conv(X_in,W) + bias)）。
    // ADD_BIAS 的反向（dY 透传 + reduce-sum 到 dBias）由调用方另外完成。
    //
    //   actConvBwdData:  dPreAct = act_bwd(dY_act, Y_act);   dX_in = conv_bwd_data(W, dPreAct)
    //   actConvBwdFilter:dPreAct = act_bwd(dY_act, Y_act);   dW    = conv_bwd_filter(X_in, dPreAct)
    static int actConvBwdData(const Matrix& Y_act, const Matrix& dY_act, const Matrix& W, Matrix& dX_in,
        const std::vector<int>& stride, const std::vector<int>& padding, int act_mode);
    static int actConvBwdFilter(const Matrix& Y_act, const Matrix& dY_act, const Matrix& X_in, Matrix& dW,
        const std::vector<int>& stride, const std::vector<int>& padding, int act_mode);

    static bool enabled();
    static void setEnabled(bool v);

    static size_t cacheSize();
    static void clearCache();
};

// =============================================================================
// CudnnOpQueueGraph
//   整网前向一张图：把 op_queue 中所有前向算子合并为单一 cuDNN backend graph。
//   不做任何手动 pattern 检测，由 cuDNN 自动选 fusion engine。
//   若存在不支持的算子，或相同 Matrix* 在不同 op 中需要不同形状描述（如 conv→FC
//   边界的隐式 reshape），则 supported()=false，调用方直接走逐算子 plain 路径。
// =============================================================================
class CCCC_EXPORT CudnnOpQueueGraph
{
public:
    CudnnOpQueueGraph() = default;
    ~CudnnOpQueueGraph() = default;

    // 尝试为 ops 构建整网图。
    // gpu 用于工作空间内存分配；返回 true 表示图构建成功。
    bool build(cudnnHandle_t handle, GpuControl* gpu, std::vector<MatrixOp>& ops);

    bool supported() const { return supported_; }

    // 用当前各 Matrix 的数据指针执行整网 plan（指针在网络生命期内固定）。
    int execute();

private:
    bool supported_ = false;
    std::unique_ptr<CudnnGraphPlan> plan_;
    Matrix workspace_;
    std::vector<CudnnBackendDesc> descs_;    // 保持所有子描述符存活

    // uid 分配与形状一致性跟踪
    struct TensorEntry
    {
        int64_t uid = 0;
        std::vector<int64_t> dims;
        std::vector<int64_t> strides;
    };
    std::unordered_map<Matrix*, TensorEntry> matrix_entry_;    // key: logical Matrix object
    int64_t next_uid_ = 1;

    // variant pack 数据（构建时收集，执行时复用）
    std::vector<int64_t> vp_uids_;
    std::vector<void*> vp_ptrs_;

    // 工具方法
    CudnnBackendDesc& addDesc(CudnnBackendDesc&& d);

    // 为 Matrix m 以给定 dims/strides 获取（或首次分配）uid。
    // dims/strides 不匹配已有记录 → 返回 false（形状冲突，不支持整网图）。
    bool getOrAssignUid(Matrix* m, const std::vector<int64_t>& dims,
        const std::vector<int64_t>& strides, int64_t& uid_out);

    // 根据 op 类型构建对应的 backend operation descriptor，
    // 并把 op handle 追加到 op_handles。返回 false 则整网不支持。
    bool buildOp(MatrixOp& op, std::vector<cudnnBackendDescriptor_t>& op_handles);
};

}    // namespace cccc

#endif    // ENABLE_CUDA
