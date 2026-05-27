#pragma once
#include "MatrixOp.h"
#include "Solver.h"

namespace cccc
{

class CCCC_EXPORT Net
{
public:
    Net();
    virtual ~Net();
    Net(const Net&) = delete;
    Net& operator=(const Net&) = delete;

protected:
    Option* option_;

    int id_ = -1;
    int epoch_ = 0;

    Matrix all_weights_;

    //权重是否分开更新
    //若某些层的学习率或者求解器不同，则需要分开更新
    int separate_update_weight_ = 0;
    Solver solver_;
    std::map<Matrix*, std::shared_ptr<Solver>> solvers_;

    std::vector<MatrixOp> op_queue_;
    std::vector<MatrixOp> loss_;

    // ---- 图模式（整网前向一张图） ----
    // 训练前尝试把整个 op_queue_ 组装成单一 cuDNN backend operation graph，
    // 由 cuDNN 自动决定融合方式，不做任何手动 pattern 检测。
    // 若含不支持算子或存在 conv->FC 隐式 reshape 边界，fwd_graph_ 为 null，
    // 运行时回退到逐算子 plain 路径。ini: [train] use_cudnn_graph=1 开启。
    bool use_graph_ = false;
    std::unique_ptr<CudnnOpQueueGraph> fwd_graph_;    // null = 不支持或未开启
    void buildForwardGraph();
    void runForwardWithGraph();
    void optimizeActivationMemory();    // 推理专用：跨层激活张量显存复用，显著降低显存峰值

    std::vector<Matrix*> weights_;
    std::vector<Solver> solvers_for_weight_;

    MatrixSP X_, A_;
    MatrixSP Y_ = makeMatrixSP();

    std::map<std::string, MatrixSP> extra_matrixsp_;    //以备不时之需
    std::string structure_script_;                      //若非空，覆盖从option读取的structure

    Matrix M_error_;

    GpuControl* gpu_;

    TestInfo test_info_;
    std::vector<TestInfo> group_test_info_;

public:
    void setId(int id) { id_ = id; }
    int getId() const { return id_; }
    int init();
    virtual int init2() = 0;
    void setGpu(GpuControl* gpu) { gpu_ = gpu; }
    GpuControl* getGpu() { return gpu_; }
    void setDeviceSelf() { gpu_->setAsCurrent(); }
    Matrix& getAllWeights() { return all_weights_; }
    void setOption(Option* op) { option_ = op; }
    int getBatch() { return getX().getNumber(); }
    void setEpoch(int epoch) { epoch_ = epoch; }
    const TestInfo& getTestInfo() { return test_info_; }
    const std::vector<TestInfo>& getGroupTestInfo() { return group_test_info_; }
    TensorForm getTensorForm() const { return TensorForm::NCHW; }
    // W8A16: convert all weight matrices in-place from BF16 to a quantized type (FP8_E4M3 / FP8_E5M2 / FP4_E2M1)
    void quantizeWeights(DataType target_dt = DataType::FP8_E4M3);
    // Save all named weights to a binary cache, recording dtype in every key ("weight_<dtype>_<name>").
    // Per-tensor scale ("scale_<name>") is saved for quantized types.  Handles any weight type.
    int saveNamedWeights(const std::string& filename);
    // Load pre-built named weights directly into GPU (avoids on-the-fly conversion at runtime).
    int loadNamedWeights(const std::string& filename);
    // Load activation input_scales for W8A8 from a sidecar bin file (optional).
    // Keys: "input_scale_<cccc_name>" = 4 bytes float32.
    int loadInputScales(const std::string& filename);
    void syncReshapeViews();
    // Set activation data type for GEMM dispatch (controls W8A8 vs W8A16 path).
    void setActDataType(DataType dt);

public:
    Matrix& getX() { return *X_; }
    Matrix& getY() { return *Y_; }
    Matrix& getA() { return *A_; }
    MatrixSP& getMatrixByName(const std::string& name);
    bool haveExtraMatrix(const std::string& name) const { return extra_matrixsp_.count(name) > 0; }
    int getExtraMatrixCount(const std::string& prefix) const
    {
        int n = 0;
        while (extra_matrixsp_.count(prefix + std::to_string(n)) > 0)
        {
            n++;
        }
        return n;
    }
    const std::map<std::string, MatrixSP>& getAllExtraMatrices() const { return extra_matrixsp_; }
    // 从另一组网络预加载已有矩阵（用于多组网络间共享权重/KV cache，不覆盖已有条目）
    void preloadMatrices(const std::map<std::string, MatrixSP>& mats)
    {
        for (auto& [k, v] : mats)
        {
            extra_matrixsp_.emplace(k, v);
        }
    }
    void setStructureScript(const std::string& s) { structure_script_ = s; }
    void addExtraMatrix(const std::string& name, const std::vector<int>& dim);

public:
    void active(Matrix* X, Matrix* Y, Matrix* A, bool back, float* error);
    void updateWeight();

    virtual int saveWeight(const std::string& filename, const std::string& sign = "", int solver_state = 0);
    virtual int loadWeight(const std::string& str, int load_mode = 0, int solver_state = 0);

    int weightDataSize() const;
    float weightSumAbs() const;
    float weightNorm2() const;
    void calNorm(int& n, float& l1, float& l2) const;
    void outputNorm() const;

private:
    int resetBatchSize(int n);
    std::vector<int> getTestGroup();

public:
    virtual int test(Matrix* X, Matrix* Y, Matrix* A);

    virtual void test2(Matrix* X, Matrix* Y, Matrix* A) {}

protected:
    void combineWeights(std::vector<Matrix*>& weights, Matrix& result);
    void initWeights();
    std::string ir();

public:
    void attack(Matrix* X, Matrix* Y);

    //重置网络中所有 KV cache 算子的写入位置, 用于开始新一段自回归推理
    void resetKVCache() { MatrixOp::resetKVCache(op_queue_); }

    // Set rope position offset for all ROPE / ROPE_INTERLEAVED ops (for KV-cache decode)
    void setRopeOffset(int pos)
    {
        for (auto& op : op_queue_)
        {
            if (op.getType() == MatrixOpType::ROPE || op.getType() == MatrixOpType::ROPE_INTERLEAVED)
            {
                op.setWindow(0, pos);
            }
        }
    }

    // Set attention position offset for all ATTENTION ops (causal mask with absolute position)
    void setAttentionOffset(int pos)
    {
        for (auto& op : op_queue_)
        {
            if (op.getType() == MatrixOpType::ATTENTION)
            {
                op.setWindow(0, pos);
            }
        }
    }

    // Override KV cache write position for all KV_CACHE ops (to restart decode after prefill)
    void setKVCachePos(int pos)
    {
        for (auto& op : op_queue_)
        {
            if (op.getType() == MatrixOpType::KV_CACHE)
            {
                op.setWindow(0, pos);
            }
        }
    }

public:
    Solver& getSolver() { return solver_; }
    int solverAdjustLearnRate(int epoch, int total_epoch)
    {
        int ret = solver_.adjustLearnRate(epoch, total_epoch);
        for (auto& [m, s] : solvers_)
        {
            s->adjustLearnRate(epoch, total_epoch);
        }
        return ret;
    }

    int solverAdjustLearnRate2(int epoch, int total_epoch, const std::vector<std::vector<TestInfo>>& test_info)
    {
        int ret = solver_.adjustLearnRate2(epoch, total_epoch, test_info);
        for (auto& [m, s] : solvers_)
        {
            s->adjustLearnRate2(epoch, total_epoch, test_info);
        }
        return ret;
    }

    void solverReset()
    {
        solver_.reset();
        for (auto& [m, s] : solvers_)
        {
            s->reset();
        }
    }

public:
    virtual void doSomeThing() {}

    void clearTime();
    void outputTime() const;

    //Net* clone(int clone_data = 0);
    void inference(Matrix& X, Matrix& A);
};

}    // namespace cccc