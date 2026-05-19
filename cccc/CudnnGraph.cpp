#include "CudnnGraph.h"

#if ENABLE_CUDA

#include "Log.h"

#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <set>
#include <unordered_map>
#include <utility>

namespace cccc
{

// =============================================================================
// CudnnBackendDesc
// =============================================================================
CudnnBackendDesc::CudnnBackendDesc(cudnnBackendDescriptorType_t type) :
    type_(type)
{
    auto s = cudnnBackendCreateDescriptor(type, &desc_);
    if (s != CUDNN_STATUS_SUCCESS)
    {
        LOG_ERR("cudnnBackendCreateDescriptor(type={}) failed: {}\n", int(type), cudnnGetErrorString(s));
        desc_ = nullptr;
    }
}

CudnnBackendDesc::~CudnnBackendDesc()
{
    if (desc_)
    {
        cudnnBackendDestroyDescriptor(desc_);
        desc_ = nullptr;
    }
}

CudnnBackendDesc::CudnnBackendDesc(CudnnBackendDesc&& o) noexcept
    :
    desc_(o.desc_), type_(o.type_), finalized_(o.finalized_)
{
    o.desc_ = nullptr;
    o.finalized_ = false;
}

CudnnBackendDesc& CudnnBackendDesc::operator=(CudnnBackendDesc&& o) noexcept
{
    if (this != &o)
    {
        if (desc_) { cudnnBackendDestroyDescriptor(desc_); }
        desc_ = o.desc_;
        type_ = o.type_;
        finalized_ = o.finalized_;
        o.desc_ = nullptr;
        o.finalized_ = false;
    }
    return *this;
}

CudnnBackendDesc& CudnnBackendDesc::setAttribute(cudnnBackendAttributeName_t name,
    cudnnBackendAttributeType_t attr_type, int64_t count, const void* data)
{
    if (!desc_) { return *this; }
    auto s = cudnnBackendSetAttribute(desc_, name, attr_type, count, data);
    if (s != CUDNN_STATUS_SUCCESS)
    {
        LOG_ERR("cudnnBackendSetAttribute(name={}, type={}) failed: {}\n",
            int(name), int(attr_type), cudnnGetErrorString(s));
    }
    return *this;
}

CudnnBackendDesc& CudnnBackendDesc::setI64(cudnnBackendAttributeName_t name, int64_t v)
{
    return setAttribute(name, CUDNN_TYPE_INT64, 1, &v);
}

CudnnBackendDesc& CudnnBackendDesc::setI64Array(cudnnBackendAttributeName_t name, const std::vector<int64_t>& v)
{
    return setAttribute(name, CUDNN_TYPE_INT64, int64_t(v.size()), v.data());
}

CudnnBackendDesc& CudnnBackendDesc::setI64Array(cudnnBackendAttributeName_t name, std::initializer_list<int64_t> v)
{
    std::vector<int64_t> tmp(v);
    return setI64Array(name, tmp);
}

CudnnBackendDesc& CudnnBackendDesc::setI32(cudnnBackendAttributeName_t name, int32_t v)
{
    return setAttribute(name, CUDNN_TYPE_INT32, 1, &v);
}

CudnnBackendDesc& CudnnBackendDesc::setF32(cudnnBackendAttributeName_t name, float v)
{
    return setAttribute(name, CUDNN_TYPE_FLOAT, 1, &v);
}

CudnnBackendDesc& CudnnBackendDesc::setF64(cudnnBackendAttributeName_t name, double v)
{
    return setAttribute(name, CUDNN_TYPE_DOUBLE, 1, &v);
}

CudnnBackendDesc& CudnnBackendDesc::setBool(cudnnBackendAttributeName_t name, bool v)
{
    bool b = v;
    return setAttribute(name, CUDNN_TYPE_BOOLEAN, 1, &b);
}

CudnnBackendDesc& CudnnBackendDesc::setDataType(cudnnBackendAttributeName_t name, cudnnDataType_t dt)
{
    return setAttribute(name, CUDNN_TYPE_DATA_TYPE, 1, &dt);
}

CudnnBackendDesc& CudnnBackendDesc::setHandle(cudnnBackendAttributeName_t name, cudnnHandle_t h)
{
    return setAttribute(name, CUDNN_TYPE_HANDLE, 1, &h);
}

CudnnBackendDesc& CudnnBackendDesc::setDesc(cudnnBackendAttributeName_t name, cudnnBackendDescriptor_t d)
{
    return setAttribute(name, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &d);
}

CudnnBackendDesc& CudnnBackendDesc::setDesc(cudnnBackendAttributeName_t name, const CudnnBackendDesc& d)
{
    cudnnBackendDescriptor_t raw = d.get();
    return setAttribute(name, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &raw);
}

CudnnBackendDesc& CudnnBackendDesc::setDescArray(cudnnBackendAttributeName_t name,
    const std::vector<cudnnBackendDescriptor_t>& v)
{
    return setAttribute(name, CUDNN_TYPE_BACKEND_DESCRIPTOR, int64_t(v.size()), v.data());
}

CudnnBackendDesc& CudnnBackendDesc::setVoidPtrArray(cudnnBackendAttributeName_t name, const std::vector<void*>& v)
{
    return setAttribute(name, CUDNN_TYPE_VOID_PTR, int64_t(v.size()), v.data());
}

CudnnBackendDesc& CudnnBackendDesc::setPointwiseMode(cudnnBackendAttributeName_t name, cudnnPointwiseMode_t m)
{
    return setAttribute(name, CUDNN_TYPE_POINTWISE_MODE, 1, &m);
}

CudnnBackendDesc& CudnnBackendDesc::setNormMode(cudnnBackendAttributeName_t name, cudnnBackendNormMode_t m)
{
    return setAttribute(name, CUDNN_TYPE_NORM_MODE, 1, &m);
}

CudnnBackendDesc& CudnnBackendDesc::setNormFwdPhase(cudnnBackendAttributeName_t name, cudnnBackendNormFwdPhase_t p)
{
    return setAttribute(name, CUDNN_TYPE_NORM_FWD_PHASE, 1, &p);
}

CudnnBackendDesc& CudnnBackendDesc::setHeurMode(cudnnBackendAttributeName_t name, cudnnBackendHeurMode_t m)
{
    return setAttribute(name, CUDNN_TYPE_HEUR_MODE, 1, &m);
}

CudnnBackendDesc& CudnnBackendDesc::finalize()
{
    if (!desc_ || finalized_) { return *this; }
    auto s = cudnnBackendFinalize(desc_);
    if (s != CUDNN_STATUS_SUCCESS)
    {
        LOG_ERR("cudnnBackendFinalize(type={}) failed: {}\n", int(type_), cudnnGetErrorString(s));
    }
    else
    {
        finalized_ = true;
    }
    return *this;
}

int64_t CudnnBackendDesc::getI64(cudnnBackendAttributeName_t name) const
{
    int64_t v = 0;
    int64_t got = 0;
    if (!desc_) { return 0; }
    auto s = cudnnBackendGetAttribute(desc_, name, CUDNN_TYPE_INT64, 1, &got, &v);
    if (s != CUDNN_STATUS_SUCCESS)
    {
        LOG_ERR("cudnnBackendGetAttribute(name={}) failed: {}\n", int(name), cudnnGetErrorString(s));
    }
    return v;
}

int64_t CudnnBackendDesc::getElementCount(cudnnBackendAttributeName_t name,
    cudnnBackendAttributeType_t attr_type) const
{
    int64_t got = 0;
    if (!desc_) { return 0; }
    cudnnBackendGetAttribute(desc_, name, attr_type, 0, &got, nullptr);
    return got;
}

// =============================================================================
// CudnnGraphTensor
// =============================================================================
std::vector<int64_t> CudnnGraphTensor::packedStridesNCHW(const std::vector<int64_t>& dims)
{
    // dims 按外→内顺序，packed strides: stride[i] = prod(dims[i+1..end-1]), stride[end-1] = 1
    std::vector<int64_t> strides(dims.size(), 1);
    for (int i = int(dims.size()) - 2; i >= 0; --i)
    {
        strides[i] = strides[i + 1] * dims[i + 1];
    }
    return strides;
}

CudnnBackendDesc CudnnGraphTensor::make(int64_t uid, cudnnDataType_t dt,
    const std::vector<int64_t>& dims, const std::vector<int64_t>& strides,
    bool is_virtual, int64_t alignment)
{
    CudnnBackendDesc desc(CUDNN_BACKEND_TENSOR_DESCRIPTOR);
    desc.setDataType(CUDNN_ATTR_TENSOR_DATA_TYPE, dt)
        .setI64Array(CUDNN_ATTR_TENSOR_DIMENSIONS, dims)
        .setI64Array(CUDNN_ATTR_TENSOR_STRIDES, strides)
        .setI64(CUDNN_ATTR_TENSOR_UNIQUE_ID, uid)
        .setI64(CUDNN_ATTR_TENSOR_BYTE_ALIGNMENT, alignment)
        .setBool(CUDNN_ATTR_TENSOR_IS_VIRTUAL, is_virtual)
        .finalize();
    return desc;
}

CudnnBackendDesc CudnnGraphTensor::fromMatrix(const Matrix& m, int64_t uid,
    bool is_virtual, int64_t alignment)
{
    // cccc dim 顺序 {W, H, C, N}（W 内层）→ cuDNN 期望 [N, C, H, W]（外→内）
    // 若 dim 不足 4，按 cccc 习惯（cudnn_desc 内部也是这样）补 1。
    auto cdim = m.getDim();
    while (cdim.size() < 4) { cdim.push_back(1); }
    std::vector<int64_t> dims(cdim.rbegin(), cdim.rend());
    auto strides = packedStridesNCHW(dims);
    return make(uid, toCudnnDataType(m.getDataType()), dims, strides, is_virtual, alignment);
}

CudnnBackendDesc CudnnGraphTensor::makeScalar(int64_t uid, cudnnDataType_t dt, bool is_virtual)
{
    return make(uid, dt, { 1, 1, 1, 1 }, { 1, 1, 1, 1 }, is_virtual, 16);
}

CudnnBackendDesc CudnnGraphTensor::fromBiasMatrix(const Matrix& bias, int64_t uid, int64_t alignment)
{
    // bias 是一维 channel 向量（cccc 中 dataSize == channel），可能 dim={1,1,C,1}
    // 或 {C,1,1,1}。统一映射到 cuDNN [N,C,H,W] = [1, c, 1, 1]，
    // 用 strides=[c,1,0,0] 在 H/W 维上实现广播。
    int64_t c = bias.getChannel();
    if (c <= 0) { c = (int64_t)bias.getDataSize(); }
    std::vector<int64_t> dims{ 1, c, 1, 1 };
    std::vector<int64_t> strides{ c, 1, 0, 0 };
    return make(uid, toCudnnDataType(bias.getDataType()), dims, strides, false, alignment);
}

// =============================================================================
// CudnnGraphPlan
// =============================================================================
CudnnGraphPlan::CudnnGraphPlan(cudnnHandle_t handle,
    const std::vector<cudnnBackendDescriptor_t>& operations,
    cudnnBackendHeurMode_t heur_mode) :
    handle_(handle)
{
    if (!handle_ || operations.empty())
    {
        LOG_ERR("CudnnGraphPlan: invalid handle or empty operations\n");
        return;
    }

    // 1) OPERATION_GRAPH
    op_graph_ = std::make_unique<CudnnBackendDesc>(CUDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR);
    op_graph_->setHandle(CUDNN_ATTR_OPERATIONGRAPH_HANDLE, handle_)
        .setDescArray(CUDNN_ATTR_OPERATIONGRAPH_OPS, operations)
        .finalize();
    if (!op_graph_->finalized()) { return; }

    // 诊断：算子图全局 engine 数（与 heuristic 无关；为 0 通常意味着算子组合不被任何 engine 支持）
    int64_t engine_global_count = 0;
    {
        int64_t got = 0;
        auto s = cudnnBackendGetAttribute(op_graph_->get(),
            CUDNN_ATTR_OPERATIONGRAPH_ENGINE_GLOBAL_COUNT,
            CUDNN_TYPE_INT64, 1, &got, &engine_global_count);
        if (s != CUDNN_STATUS_SUCCESS)
        {
            LOG_ERR("[CudnnGraph] get ENGINE_GLOBAL_COUNT failed: {}\n", cudnnGetErrorString(s));
        }
    }

    // 2)+3) 依次尝试多个 heuristic 模式：A → B → FALLBACK
    //    标准 cuDNN backend 单算子（pointwise / standalone conv-bwd / pool 等）
    //    在 HEUR_MODE_A 下经常返回 0 候选，需要 FALLBACK 兜底。
    cudnnBackendHeurMode_t modes[] = {
        heur_mode,
        CUDNN_HEUR_MODE_B,
        CUDNN_HEUR_MODE_FALLBACK,
    };
    cudnnBackendDescriptor_t cfg_raw = nullptr;
    cudnnBackendHeurMode_t used_mode = heur_mode;
    bool got_cfg = false;
    for (auto mode : modes)
    {
        // 每次重新建 ENGINEHEUR 描述符（不能修改已 finalize 的 desc）
        heur_ = std::make_unique<CudnnBackendDesc>(CUDNN_BACKEND_ENGINEHEUR_DESCRIPTOR);
        heur_->setDesc(CUDNN_ATTR_ENGINEHEUR_OPERATION_GRAPH, *op_graph_)
            .setHeurMode(CUDNN_ATTR_ENGINEHEUR_MODE, mode)
            .finalize();
        if (!heur_->finalized()) { continue; }

        engine_cfg_ = std::make_unique<CudnnBackendDesc>(CUDNN_BACKEND_ENGINECFG_DESCRIPTOR);
        cfg_raw = engine_cfg_->get();
        int64_t got = 0;
        auto s = cudnnBackendGetAttribute(heur_->get(),
            CUDNN_ATTR_ENGINEHEUR_RESULTS,
            CUDNN_TYPE_BACKEND_DESCRIPTOR,
            1, &got, &cfg_raw);
        if (s == CUDNN_STATUS_SUCCESS && got >= 1)
        {
            used_mode = mode;
            got_cfg = true;
            break;
        }
    }
    if (!got_cfg)
    {
        // Heuristic 完全失败但 op graph 有可用 engine —— 直接按全局索引枚举
        if (engine_global_count > 0)
        {
            for (int64_t idx = 0; idx < engine_global_count && !got_cfg; ++idx)
            {
                CudnnBackendDesc eng(CUDNN_BACKEND_ENGINE_DESCRIPTOR);
                eng.setDesc(CUDNN_ATTR_ENGINE_OPERATION_GRAPH, *op_graph_)
                    .setI64(CUDNN_ATTR_ENGINE_GLOBAL_INDEX, idx)
                    .finalize();
                if (!eng.finalized()) { continue; }

                auto cfg = std::make_unique<CudnnBackendDesc>(CUDNN_BACKEND_ENGINECFG_DESCRIPTOR);
                cfg->setDesc(CUDNN_ATTR_ENGINECFG_ENGINE, eng)
                    .finalize();
                if (!cfg->finalized()) { continue; }

                // 试着 finalize 一个 plan，看是否真的可执行
                auto trial_plan = std::make_unique<CudnnBackendDesc>(CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR);
                trial_plan->setHandle(CUDNN_ATTR_EXECUTION_PLAN_HANDLE, handle_)
                    .setDesc(CUDNN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG, *cfg)
                    .finalize();
                if (!trial_plan->finalized()) { continue; }

                // 保留这套 desc 以便后续执行
                // 这里要把 engine 也持有（cuDNN 在 plan 内引用其句柄）
                // 我们暂存到 op_graph_ 之后的成员里
                // 直接重用 engine_ / engine_cfg_ / plan_：
                engine_ = std::make_unique<CudnnBackendDesc>(std::move(eng));
                engine_cfg_ = std::move(cfg);
                plan_ = std::move(trial_plan);
                workspace_size_ = plan_->getI64(CUDNN_ATTR_EXECUTION_PLAN_WORKSPACE_SIZE);
                ok_ = true;
                LOG_ERR(
                    "[CudnnGraph] heuristic returned 0, using engine global_index={} "
                    "(workspace={} bytes)\n",
                    idx, workspace_size_);
                return;
            }
        }
        LOG_ERR(
            "[CudnnGraph] ENGINEHEUR returned 0 engines for all modes "
            "(A/B/FALLBACK). engine_global_count={}\n",
            engine_global_count);
        return;
    }
    if (used_mode != heur_mode)
    {
        LOG_ERR(
            "[CudnnGraph] heuristic fell back to mode {} (requested {}), "
            "engine_global_count={}\n",
            int(used_mode), int(heur_mode), engine_global_count);
    }

    // 4) EXECUTION_PLAN
    plan_ = std::make_unique<CudnnBackendDesc>(CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR);
    plan_->setHandle(CUDNN_ATTR_EXECUTION_PLAN_HANDLE, handle_)
        .setDesc(CUDNN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG, cfg_raw)
        .finalize();
    if (!plan_->finalized())
    {
        LOG_ERR("[CudnnGraph] EXECUTION_PLAN finalize failed (engine cfg from mode {})\n",
            int(used_mode));
        return;
    }

    workspace_size_ = plan_->getI64(CUDNN_ATTR_EXECUTION_PLAN_WORKSPACE_SIZE);
    ok_ = true;
}

int CudnnGraphPlan::execute(const std::vector<int64_t>& uids,
    const std::vector<void*>& ptrs, void* workspace)
{
    if (!ok_) { return -1; }
    if (uids.size() != ptrs.size())
    {
        LOG_ERR("CudnnGraphPlan::execute uid/ptr size mismatch: {} vs {}\n", uids.size(), ptrs.size());
        return -2;
    }

    CudnnBackendDesc vp(CUDNN_BACKEND_VARIANT_PACK_DESCRIPTOR);
    vp.setI64Array(CUDNN_ATTR_VARIANT_PACK_UNIQUE_IDS, uids)
        .setVoidPtrArray(CUDNN_ATTR_VARIANT_PACK_DATA_POINTERS, ptrs)
        .setAttribute(CUDNN_ATTR_VARIANT_PACK_WORKSPACE, CUDNN_TYPE_VOID_PTR, 1, &workspace)
        .finalize();
    if (!vp.finalized()) { return -3; }

    auto s = cudnnBackendExecute(handle_, plan_->get(), vp.get());
    if (s != CUDNN_STATUS_SUCCESS)
    {
        LOG_ERR("cudnnBackendExecute failed: {}\n", cudnnGetErrorString(s));
        return int(s);
    }
    return 0;
}

int CudnnGraphPlan::execute(const std::vector<int64_t>& uids,
    const std::vector<const Matrix*>& mats, void* workspace)
{
    if (uids.size() != mats.size())
    {
        LOG_ERR("CudnnGraphPlan::execute(Matrix*) uid/mat size mismatch: {} vs {}\n", uids.size(), mats.size());
        return -2;
    }
    std::vector<void*> ptrs;
    ptrs.reserve(mats.size());
    for (auto* m : mats)
    {
        if (!m)
        {
            LOG_ERR("CudnnGraphPlan::execute(Matrix*) null matrix at index {}\n", ptrs.size());
            return -4;
        }
        ptrs.push_back(m->getDataPtr());
    }
    return execute(uids, ptrs, workspace);
}

int CudnnGraphPlan::execute(const std::vector<int64_t>& uids,
    const std::vector<MatrixSP>& mats, void* workspace)
{
    if (uids.size() != mats.size())
    {
        LOG_ERR("CudnnGraphPlan::execute(MatrixSP) uid/mat size mismatch: {} vs {}\n", uids.size(), mats.size());
        return -2;
    }
    std::vector<void*> ptrs;
    ptrs.reserve(mats.size());
    for (const auto& m : mats)
    {
        if (!m)
        {
            LOG_ERR("CudnnGraphPlan::execute(MatrixSP) null matrix at index {}\n", ptrs.size());
            return -4;
        }
        ptrs.push_back(m->getDataPtr());
    }
    return execute(uids, ptrs, workspace);
}

// =============================================================================
// CudnnGraphOps  —  高层算子（带缓存）
// =============================================================================
namespace
{
struct PlanCacheEntry
{
    std::unique_ptr<CudnnGraphPlan> plan;
    Matrix workspace;    // 每个 plan 独立的工作空间
    // keepalive: cuDNN backend API may hold raw pointers into these descriptors
    // even after finalization of the execution plan, so we must keep them alive.
    std::vector<CudnnBackendDesc> descs;
};

std::unordered_map<std::string, PlanCacheEntry>& planCache()
{
    static std::unordered_map<std::string, PlanCacheEntry> g;
    return g;
}
std::mutex& planCacheMutex()
{
    static std::mutex m;
    return m;
}

bool& enabledFlag()
{
    static bool v = false;
    return v;
}

// 一次性日志：每个算子名称只打印一次，用于运行时确认走的是 graph 路径
void logGraphFirstUse(const char* op_name, const std::string& key_short)
{
    static std::mutex m;
    static std::set<std::string> seen;
    std::lock_guard<std::mutex> lk(m);
    if (seen.insert(op_name).second)
    {
        // 用 LOG_ERR 输出到 stderr，便于在打印密集的训练日志中可见
        // （内容是“正在使用 graph 路径”的提示，不是错误）
        LOG_ERR("[CudnnGraph] using cuDNN backend graph for {} (first call). shape: {}\n",
            op_name, key_short);
    }
}

std::string makeKeyConv(const Matrix& X, const Matrix& W, const Matrix& Y,
    const std::vector<int>& stride, const std::vector<int>& padding,
    bool alpha_one, bool beta_zero)
{
    auto pad4 = [](std::vector<int> v)
    {
        while (v.size() < 4)
        {
            v.push_back(1);
        }
        return v;
    };
    auto xd = pad4(X.getDim());
    auto wd = pad4(W.getDim());
    auto yd = pad4(Y.getDim());
    return std::format("C|dt={}|X={},{},{},{}|W={},{},{},{}|Y={},{},{},{}|s={},{}|p={},{}|a{}|b{}",
        int(Y.getDataType()),
        xd[0], xd[1], xd[2], xd[3], wd[0], wd[1], wd[2], wd[3],
        yd[0], yd[1], yd[2], yd[3],
        stride[0], stride[1], padding[0], padding[1],
        int(alpha_one), int(beta_zero));
}

std::string makeKeyPool(const Matrix& X, const Matrix& Y, int pooling_type,
    const std::vector<int>& window, const std::vector<int>& stride,
    const std::vector<int>& padding)
{
    auto pad4 = [](std::vector<int> v)
    {
        while (v.size() < 4)
        {
            v.push_back(1);
        }
        return v;
    };
    auto xd = pad4(X.getDim());
    auto yd = pad4(Y.getDim());
    return std::format("P|dt={}|t={}|X={},{},{},{}|Y={},{},{},{}|w={},{}|s={},{}|p={},{}",
        int(Y.getDataType()), pooling_type,
        xd[0], xd[1], xd[2], xd[3], yd[0], yd[1], yd[2], yd[3],
        window[0], window[1], stride[0], stride[1], padding[0], padding[1]);
}
}    // namespace

bool CudnnGraphOps::enabled() { return enabledFlag(); }
void CudnnGraphOps::setEnabled(bool v) { enabledFlag() = v; }

size_t CudnnGraphOps::cacheSize()
{
    std::lock_guard<std::mutex> lk(planCacheMutex());
    return planCache().size();
}

void CudnnGraphOps::clearCache()
{
    std::lock_guard<std::mutex> lk(planCacheMutex());
    planCache().clear();
}

int CudnnGraphOps::convForward(const Matrix& X, const Matrix& W, Matrix& Y,
    const std::vector<int>& stride, const std::vector<int>& padding,
    float alpha, float beta)
{
    if (!X.isCuda()) { return -100; }
    auto* gpu = X.gpu();
    if (!gpu || !gpu->cudnn_handle_) { return -101; }

    bool alpha_one = (alpha == 1.0f);
    bool beta_zero = (beta == 0.0f);
    auto key = makeKeyConv(X, W, Y, stride, padding, alpha_one, beta_zero);

    PlanCacheEntry* entry = nullptr;
    {
        std::lock_guard<std::mutex> lk(planCacheMutex());
        auto it = planCache().find(key);
        if (it == planCache().end())
        {
            // Try HEUR_MODE_A, rebuild all descriptors fresh for FALLBACK if needed
            auto buildPlan = [&](cudnnBackendHeurMode_t hmode)
                -> std::tuple<std::unique_ptr<CudnnGraphPlan>,
                    CudnnBackendDesc, CudnnBackendDesc, CudnnBackendDesc,
                    CudnnBackendDesc, CudnnBackendDesc>
            {
                auto xT2 = CudnnGraphTensor::fromMatrix(X, 1);
                auto wT2 = CudnnGraphTensor::fromMatrix(W, 2);
                auto yT2 = CudnnGraphTensor::fromMatrix(Y, 3);
                CudnnBackendDesc conv2(CUDNN_BACKEND_CONVOLUTION_DESCRIPTOR);
                conv2.setDataType(CUDNN_ATTR_CONVOLUTION_COMP_TYPE, cudnnTypeOf(Y.getDataType()))
                    .setI64(CUDNN_ATTR_CONVOLUTION_SPATIAL_DIMS, 2)
                    .setI64Array(CUDNN_ATTR_CONVOLUTION_DILATIONS, { 1, 1 })
                    .setI64Array(CUDNN_ATTR_CONVOLUTION_FILTER_STRIDES,
                        { (int64_t)stride[1], (int64_t)stride[0] })
                    .setI64Array(CUDNN_ATTR_CONVOLUTION_PRE_PADDINGS,
                        { (int64_t)padding[1], (int64_t)padding[0] })
                    .setI64Array(CUDNN_ATTR_CONVOLUTION_POST_PADDINGS,
                        { (int64_t)padding[1], (int64_t)padding[0] });
                cudnnConvolutionMode_t cmode2 = CUDNN_CROSS_CORRELATION;
                conv2.setAttribute(CUDNN_ATTR_CONVOLUTION_CONV_MODE,
                    CUDNN_TYPE_CONVOLUTION_MODE, 1, &cmode2);
                conv2.finalize();
                CudnnBackendDesc op2(CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR);
                double a642 = alpha, b642 = beta;
                op2.setDesc(CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_X, xT2)
                    .setDesc(CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_W, wT2)
                    .setDesc(CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_Y, yT2)
                    .setDesc(CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_CONV_DESC, conv2)
                    .setF64(CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_ALPHA, a642)
                    .setF64(CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_BETA, b642)
                    .finalize();
                auto p = std::make_unique<CudnnGraphPlan>(gpu->cudnn_handle_,
                    std::vector{ op2.get() },
                    hmode);
                return { std::move(p), std::move(xT2), std::move(wT2), std::move(yT2),
                    std::move(conv2), std::move(op2) };
            };
            auto [plan, xT, wT, yT, conv, op] = buildPlan(CUDNN_HEUR_MODE_A);
            if (!plan->ok())
            {
                LOG_ERR("[CudnnGraph] HEUR_MODE_A failed, trying FALLBACK\n");
                auto [p2, x2, w2, y2, c2, o2] = buildPlan(CUDNN_HEUR_MODE_FALLBACK);
                plan = std::move(p2);
                xT = std::move(x2);
                wT = std::move(w2);
                yT = std::move(y2);
                conv = std::move(c2);
                op = std::move(o2);
            }
            if (!plan->ok())
            {
                // negative cache to suppress repeated build attempts
                PlanCacheEntry e_neg;
                planCache().emplace(key, std::move(e_neg));
                return -104;
            }

            PlanCacheEntry e;
            e.plan = std::move(plan);
            // workspace must be GPU memory
            int64_t ws = e.plan->workspaceSize();
            if (ws > 0)
            {
                e.workspace.shared_data_->setGpu(gpu);
                e.workspace.resize(1, 1, 1, int(ws / e.workspace.getDataTypeSize() + 2));
            }
            // IMPORTANT: keep all sub-descriptors alive – cuDNN backend API retains
            // raw handles into these even after the execution plan is finalized.
            e.descs.push_back(std::move(xT));
            e.descs.push_back(std::move(wT));
            e.descs.push_back(std::move(yT));
            e.descs.push_back(std::move(conv));
            e.descs.push_back(std::move(op));
            it = planCache().emplace(key, std::move(e)).first;
            logGraphFirstUse("convForward", key);
        }
        entry = &it->second;
    }
    if (!entry->plan) { return -1; }    // cached failure; do not retry

    void* ws_ptr = entry->workspace.getDataSizeInByte() ? entry->workspace.getDataPtr() : nullptr;

    int rc = entry->plan->execute({ 1, 2, 3 },
        std::vector<void*>{ X.data(), W.data(), Y.data() }, ws_ptr);

    return rc;
}

int CudnnGraphOps::poolForward(const Matrix& X, Matrix& Y, int pooling_type,
    const std::vector<int>& window, const std::vector<int>& stride,
    const std::vector<int>& padding, float /*alpha*/, float /*beta*/)
{
    if (!X.isCuda()) { return -100; }
    auto* gpu = X.gpu();
    if (!gpu || !gpu->cudnn_handle_) { return -101; }

    auto key = makeKeyPool(X, Y, pooling_type, window, stride, padding);

    PlanCacheEntry* entry = nullptr;
    {
        std::lock_guard<std::mutex> lk(planCacheMutex());
        auto it = planCache().find(key);
        if (it == planCache().end())
        {
            cudnnResampleMode_t rmode;
            switch (pooling_type)
            {
            case 0:    // POOLING_MAX
                rmode = CUDNN_RESAMPLE_MAXPOOL;
                break;
            case 1:    // POOLING_AVERAGE_PADDING
                rmode = CUDNN_RESAMPLE_AVGPOOL_INCLUDE_PADDING;
                break;
            case 2:    // POOLING_AVERAGE_NOPADDING
                rmode = CUDNN_RESAMPLE_AVGPOOL_EXCLUDE_PADDING;
                break;
            default:
                return -110;
            }

            auto xT = CudnnGraphTensor::fromMatrix(X, /*uid*/ 1);
            auto yT = CudnnGraphTensor::fromMatrix(Y, /*uid*/ 2);

            CudnnBackendDesc rsmp(CUDNN_BACKEND_RESAMPLE_DESCRIPTOR);
            rsmp.setAttribute(CUDNN_ATTR_RESAMPLE_MODE, CUDNN_TYPE_RESAMPLE_MODE, 1, &rmode)
                .setDataType(CUDNN_ATTR_RESAMPLE_COMP_TYPE, cudnnTypeOf(Y.getDataType()))
                .setI64(CUDNN_ATTR_RESAMPLE_SPATIAL_DIMS, 2)
                .setI64Array(CUDNN_ATTR_RESAMPLE_WINDOW_DIMS,
                    { (int64_t)window[1], (int64_t)window[0] })
                .setI64Array(CUDNN_ATTR_RESAMPLE_STRIDES,
                    { (int64_t)stride[1], (int64_t)stride[0] })
                .setI64Array(CUDNN_ATTR_RESAMPLE_PRE_PADDINGS,
                    { (int64_t)padding[1], (int64_t)padding[0] })
                .setI64Array(CUDNN_ATTR_RESAMPLE_POST_PADDINGS,
                    { (int64_t)padding[1], (int64_t)padding[0] });
            cudnnPaddingMode_t pmode = CUDNN_ZERO_PAD;
            rsmp.setAttribute(CUDNN_ATTR_RESAMPLE_PADDING_MODE,
                CUDNN_TYPE_PADDING_MODE, 1, &pmode);
            rsmp.finalize();
            if (!rsmp.finalized()) { return -112; }

            CudnnBackendDesc op(CUDNN_BACKEND_OPERATION_RESAMPLE_FWD_DESCRIPTOR);
            op.setDesc(CUDNN_ATTR_OPERATION_RESAMPLE_FWD_XDESC, xT)
                .setDesc(CUDNN_ATTR_OPERATION_RESAMPLE_FWD_YDESC, yT)
                .setDesc(CUDNN_ATTR_OPERATION_RESAMPLE_FWD_DESC, rsmp)
                .finalize();
            if (!op.finalized()) { return -113; }

            auto plan = std::make_unique<CudnnGraphPlan>(gpu->cudnn_handle_,
                std::vector{ op.get() });
            if (!plan->ok())
            {
                PlanCacheEntry e_neg;
                planCache().emplace(key, std::move(e_neg));
                return -114;
            }

            PlanCacheEntry e;
            e.plan = std::move(plan);
            int64_t ws = e.plan->workspaceSize();
            if (ws > 0)
            {
                e.workspace.resize(1, 1, 1, int(ws / e.workspace.getDataTypeSize() + 2));
            }
            it = planCache().emplace(key, std::move(e)).first;
            logGraphFirstUse("poolForward", key);
        }
        entry = &it->second;
    }
    if (!entry->plan) { return -1; }    // cached failure; do not retry

    void* ws_ptr = entry->workspace.getDataSizeInByte() ? entry->workspace.getDataPtr() : nullptr;
    return entry->plan->execute({ 1, 2 },
        std::vector<void*>{ X.data(), Y.data() }, ws_ptr);
}

// =============================================================================
// 反向卷积 / 激活 / 偏置
// =============================================================================
namespace
{
// 共享：构建 CONVOLUTION_DESCRIPTOR
CudnnBackendDesc buildConvDesc(cudnnDataType_t comp_type,
    const std::vector<int>& stride, const std::vector<int>& padding)
{
    CudnnBackendDesc conv(CUDNN_BACKEND_CONVOLUTION_DESCRIPTOR);
    conv.setDataType(CUDNN_ATTR_CONVOLUTION_COMP_TYPE, comp_type)
        .setI64(CUDNN_ATTR_CONVOLUTION_SPATIAL_DIMS, 2)
        .setI64Array(CUDNN_ATTR_CONVOLUTION_DILATIONS, { 1, 1 })
        .setI64Array(CUDNN_ATTR_CONVOLUTION_FILTER_STRIDES,
            { (int64_t)stride[1], (int64_t)stride[0] })
        .setI64Array(CUDNN_ATTR_CONVOLUTION_PRE_PADDINGS,
            { (int64_t)padding[1], (int64_t)padding[0] })
        .setI64Array(CUDNN_ATTR_CONVOLUTION_POST_PADDINGS,
            { (int64_t)padding[1], (int64_t)padding[0] });
    cudnnConvolutionMode_t cmode = CUDNN_CROSS_CORRELATION;
    conv.setAttribute(CUDNN_ATTR_CONVOLUTION_CONV_MODE,
        CUDNN_TYPE_CONVOLUTION_MODE, 1, &cmode);
    conv.finalize();
    return conv;
}

std::string makeKeyConvBwd(const char* tag, const Matrix& A, const Matrix& B, const Matrix& C,
    const std::vector<int>& stride, const std::vector<int>& padding,
    bool alpha_one, bool beta_zero)
{
    auto pad4 = [](std::vector<int> v)
    {
        while (v.size() < 4)
        {
            v.push_back(1);
        }
        return v;
    };
    auto ad = pad4(A.getDim());
    auto bd = pad4(B.getDim());
    auto cd = pad4(C.getDim());
    return std::format("{}|dt={}|A={},{},{},{}|B={},{},{},{}|C={},{},{},{}|s={},{}|p={},{}|a{}|b{}",
        tag, int(C.getDataType()),
        ad[0], ad[1], ad[2], ad[3], bd[0], bd[1], bd[2], bd[3],
        cd[0], cd[1], cd[2], cd[3],
        stride[0], stride[1], padding[0], padding[1],
        int(alpha_one), int(beta_zero));
}

std::string makeKeyAct(const char* tag, const Matrix& X, int af)
{
    auto pad4 = [](std::vector<int> v)
    {
        while (v.size() < 4)
        {
            v.push_back(1);
        }
        return v;
    };
    auto xd = pad4(X.getDim());
    return std::format("{}|dt={}|af={}|X={},{},{},{}",
        tag, int(X.getDataType()), af, xd[0], xd[1], xd[2], xd[3]);
}

std::string makeKeyBias(const Matrix& X, const Matrix& bias)
{
    auto pad4 = [](std::vector<int> v)
    {
        while (v.size() < 4)
        {
            v.push_back(1);
        }
        return v;
    };
    auto xd = pad4(X.getDim());
    auto bd = pad4(bias.getDim());
    return std::format("B|dt={}|X={},{},{},{}|b={},{},{},{}",
        int(X.getDataType()), xd[0], xd[1], xd[2], xd[3],
        bd[0], bd[1], bd[2], bd[3]);
}

// 把 ActiveFunctionType (cccc) 映射到 cuDNN POINTWISE 模式。失败返回 false。
bool mapActFwd(int af, cudnnPointwiseMode_t& m)
{
    switch (af)
    {
    case 0: /* ACTIVE_FUNCTION_SIGMOID */ m = CUDNN_POINTWISE_SIGMOID_FWD; return true;
    case 1: /* ACTIVE_FUNCTION_RELU    */ m = CUDNN_POINTWISE_RELU_FWD; return true;
    case 2: /* ACTIVE_FUNCTION_TANH    */ m = CUDNN_POINTWISE_TANH_FWD; return true;
    default: return false;
    }
}
bool mapActBwd(int af, cudnnPointwiseMode_t& m)
{
    switch (af)
    {
    case 0: m = CUDNN_POINTWISE_SIGMOID_BWD; return true;
    case 1: m = CUDNN_POINTWISE_RELU_BWD; return true;
    case 2: m = CUDNN_POINTWISE_TANH_BWD; return true;
    default: return false;
    }
}
}    // namespace

int CudnnGraphOps::convBackwardData(const Matrix& W, const Matrix& dY, Matrix& dX,
    const std::vector<int>& stride, const std::vector<int>& padding,
    float alpha, float beta)
{
    if (!dX.isCuda()) { return -100; }
    auto* gpu = dX.gpu();
    if (!gpu || !gpu->cudnn_handle_) { return -101; }

    bool alpha_one = (alpha == 1.0f);
    bool beta_zero = (beta == 0.0f);
    auto key = makeKeyConvBwd("CBD", W, dY, dX, stride, padding, alpha_one, beta_zero);

    PlanCacheEntry* entry = nullptr;
    {
        std::lock_guard<std::mutex> lk(planCacheMutex());
        auto it = planCache().find(key);
        if (it == planCache().end())
        {
            auto wT = CudnnGraphTensor::fromMatrix(W, /*uid*/ 1);
            auto dyT = CudnnGraphTensor::fromMatrix(dY, /*uid*/ 2);
            auto dxT = CudnnGraphTensor::fromMatrix(dX, /*uid*/ 3);
            auto conv = buildConvDesc(cudnnTypeOf(dX.getDataType()), stride, padding);
            if (!conv.finalized()) { return -102; }

            CudnnBackendDesc op(CUDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_DATA_DESCRIPTOR);
            double a64 = alpha, b64 = beta;
            op.setDesc(CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_W, wT)
                .setDesc(CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_DY, dyT)
                .setDesc(CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_DX, dxT)
                .setDesc(CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_CONV_DESC, conv)
                .setF64(CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_ALPHA, a64)
                .setF64(CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_BETA, b64)
                .finalize();
            if (!op.finalized()) { return -103; }

            auto plan = std::make_unique<CudnnGraphPlan>(gpu->cudnn_handle_,
                std::vector{ op.get() });
            if (!plan->ok())
            {
                // negative cache to suppress repeated build attempts
                PlanCacheEntry e_neg;
                planCache().emplace(key, std::move(e_neg));
                return -104;
            }
            PlanCacheEntry e;
            e.plan = std::move(plan);
            int64_t ws = e.plan->workspaceSize();
            if (ws > 0)
            {
                e.workspace.shared_data_->setGpu(gpu);
                e.workspace.resize(1, 1, 1, int(ws / e.workspace.getDataTypeSize() + 2));
            }
            it = planCache().emplace(key, std::move(e)).first;
            logGraphFirstUse("convBackwardData", key);
        }
        entry = &it->second;
    }
    if (!entry->plan) { return -1; }    // cached failure; do not retry
    void* ws_ptr = entry->workspace.getDataSizeInByte() ? entry->workspace.getDataPtr() : nullptr;
    return entry->plan->execute({ 1, 2, 3 },
        std::vector<void*>{ W.data(), dY.data(), dX.data() }, ws_ptr);
}

int CudnnGraphOps::convBackwardFilter(const Matrix& X, const Matrix& dY, Matrix& dW,
    const std::vector<int>& stride, const std::vector<int>& padding,
    float alpha, float beta)
{
    if (!dW.isCuda()) { return -100; }
    auto* gpu = dW.gpu();
    if (!gpu || !gpu->cudnn_handle_) { return -101; }

    bool alpha_one = (alpha == 1.0f);
    bool beta_zero = (beta == 0.0f);
    auto key = makeKeyConvBwd("CBF", X, dY, dW, stride, padding, alpha_one, beta_zero);

    PlanCacheEntry* entry = nullptr;
    {
        std::lock_guard<std::mutex> lk(planCacheMutex());
        auto it = planCache().find(key);
        if (it == planCache().end())
        {
            auto xT = CudnnGraphTensor::fromMatrix(X, /*uid*/ 1);
            auto dyT = CudnnGraphTensor::fromMatrix(dY, /*uid*/ 2);
            auto dwT = CudnnGraphTensor::fromMatrix(dW, /*uid*/ 3);
            auto conv = buildConvDesc(cudnnTypeOf(dW.getDataType()), stride, padding);
            if (!conv.finalized()) { return -102; }

            CudnnBackendDesc op(CUDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_FILTER_DESCRIPTOR);
            double a64 = alpha, b64 = beta;
            op.setDesc(CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_X, xT)
                .setDesc(CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_DY, dyT)
                .setDesc(CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_DW, dwT)
                .setDesc(CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_CONV_DESC, conv)
                .setF64(CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_ALPHA, a64)
                .setF64(CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_BETA, b64)
                .finalize();
            if (!op.finalized()) { return -103; }

            auto plan = std::make_unique<CudnnGraphPlan>(gpu->cudnn_handle_,
                std::vector{ op.get() });
            if (!plan->ok())
            {
                // negative cache to suppress repeated build attempts
                PlanCacheEntry e_neg;
                planCache().emplace(key, std::move(e_neg));
                return -104;
            }
            PlanCacheEntry e;
            e.plan = std::move(plan);
            int64_t ws = e.plan->workspaceSize();
            if (ws > 0)
            {
                e.workspace.shared_data_->setGpu(gpu);
                e.workspace.resize(1, 1, 1, int(ws / e.workspace.getDataTypeSize() + 2));
            }
            it = planCache().emplace(key, std::move(e)).first;
            logGraphFirstUse("convBackwardFilter", key);
        }
        entry = &it->second;
    }
    if (!entry->plan) { return -1; }    // cached failure; do not retry
    void* ws_ptr = entry->workspace.getDataSizeInByte() ? entry->workspace.getDataPtr() : nullptr;
    return entry->plan->execute({ 1, 2, 3 },
        std::vector<void*>{ X.data(), dY.data(), dW.data() }, ws_ptr);
}

int CudnnGraphOps::actForward(const Matrix& X, Matrix& Y, int af, float alpha, float beta)
{
    // cuDNN backend graph engines 仅覆盖 fusion 场景；独立 pointwise（含激活）
    // 在 9.x 上没有可用 engine（HEUR 永远返回 0，cutlass/runtime-fusion 引擎
    // 都拒绝）。直接返回 -1 让调用方静默走 legacy cudnnActivationForward。
    (void)X;
    (void)Y;
    (void)af;
    (void)alpha;
    (void)beta;
    static std::once_flag once;
    std::call_once(once, []
        {
            LOG_ERR(
                "[CudnnGraph] standalone pointwise (act/bias) not supported by "
                "cuDNN graph engines; using legacy path. (informational, once)\n");
        });
    return -1;
}

int CudnnGraphOps::actBackward(const Matrix& X, const Matrix& Y,
    const Matrix& dY, Matrix& dX, int af, float alpha, float beta)
{
    // 见 actForward 的注释；独立 pointwise 反向同样不被 graph engines 支持。
    (void)X;
    (void)Y;
    (void)dY;
    (void)dX;
    (void)af;
    (void)alpha;
    (void)beta;
    return -1;
}

int CudnnGraphOps::biasAdd(const Matrix& X, const Matrix& bias, Matrix& Y,
    float alpha, float beta)
{
    // 独立 pointwise add 不被 graph engines 支持；走 legacy cudnnAddTensor。
    (void)X;
    (void)bias;
    (void)Y;
    (void)alpha;
    (void)beta;
    return -1;
}

namespace
{
// 共享：构建 fused conv(+bias)(+act) plan。include_act=false 时仅 conv+bias。
// 失败返回 nullptr；成功返回 PlanCacheEntry（plan + workspace + descs）。
struct FusedBuildResult
{
    std::unique_ptr<CudnnGraphPlan> plan;
    std::vector<CudnnBackendDesc> descs;    // keepalive
    int64_t ws_bytes = 0;
};

FusedBuildResult buildConvBiasActPlan(GpuControl* gpu,
    const Matrix& X, const Matrix& W, const Matrix& bias, const Matrix& Y,
    const std::vector<int>& stride, const std::vector<int>& padding,
    bool include_act, cudnnPointwiseMode_t act_mode)
{
    FusedBuildResult r;
    auto comp_type = cudnnTypeOf(Y.getDataType());

    // Tensors:
    //   uid 1: X       (real input)
    //   uid 2: W       (real input)
    //   uid 3: bias    (real input, broadcasting)
    //   uid 4: Y       (real output)
    //   uid 10: conv_out  (virtual)
    //   uid 11: bias_out  (virtual, only used if include_act)
    auto xT = CudnnGraphTensor::fromMatrix(X, 1);
    auto wT = CudnnGraphTensor::fromMatrix(W, 2);
    auto bT = CudnnGraphTensor::fromBiasMatrix(bias, 3);
    auto yT = CudnnGraphTensor::fromMatrix(Y, 4);

    // virtual conv output: same shape as Y
    std::vector<int64_t> y_dims, y_strides;
    {
        auto pad4 = [](std::vector<int> v)
        {
            while (v.size() < 4)
            {
                v.push_back(1);
            }
            return v;
        };
        auto yd = pad4(Y.getDim());                 // {W, H, C, N}
        y_dims = { yd[3], yd[2], yd[1], yd[0] };    // [N,C,H,W]
        y_strides = CudnnGraphTensor::packedStridesNCHW(y_dims);
    }
    auto convOutT = CudnnGraphTensor::make(10, comp_type, y_dims, y_strides, /*is_virtual*/ true);
    auto biasOutT = include_act ? CudnnGraphTensor::make(11, comp_type, y_dims, y_strides, /*is_virtual*/ true) : CudnnBackendDesc(CUDNN_BACKEND_TENSOR_DESCRIPTOR);    // unused

    auto conv = buildConvDesc(comp_type, stride, padding);
    if (!conv.finalized()) { return r; }

    // Op 1: CONV_FWD  X * W -> convOut (alpha=1, beta=0)
    CudnnBackendDesc opConv(CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR);
    {
        double a64 = 1.0, b64 = 0.0;
        opConv.setDesc(CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_X, xT)
            .setDesc(CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_W, wT)
            .setDesc(CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_Y, convOutT)
            .setDesc(CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_CONV_DESC, conv)
            .setF64(CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_ALPHA, a64)
            .setF64(CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_BETA, b64)
            .finalize();
    }
    if (!opConv.finalized()) { return r; }

    // POINTWISE_ADD desc  (mode = ADD)
    CudnnBackendDesc pwAddDesc(CUDNN_BACKEND_POINTWISE_DESCRIPTOR);
    pwAddDesc.setPointwiseMode(CUDNN_ATTR_POINTWISE_MODE, CUDNN_POINTWISE_ADD)
        .setDataType(CUDNN_ATTR_POINTWISE_MATH_PREC, comp_type)
        .finalize();
    if (!pwAddDesc.finalized()) { return r; }

    // Op 2: convOut + bias -> biasOut (or directly Y if no act)
    CudnnBackendDesc opAdd(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR);
    opAdd.setDesc(CUDNN_ATTR_OPERATION_POINTWISE_PW_DESCRIPTOR, pwAddDesc)
        .setDesc(CUDNN_ATTR_OPERATION_POINTWISE_XDESC, convOutT)
        .setDesc(CUDNN_ATTR_OPERATION_POINTWISE_BDESC, bT)
        .setDesc(CUDNN_ATTR_OPERATION_POINTWISE_YDESC, include_act ? biasOutT : yT)
        .finalize();
    if (!opAdd.finalized()) { return r; }

    std::vector<cudnnBackendDescriptor_t> ops = { opConv.get(), opAdd.get() };

    // Optional Op 3: ACTIVATION pointwise
    CudnnBackendDesc pwActDesc(CUDNN_BACKEND_POINTWISE_DESCRIPTOR);
    CudnnBackendDesc opAct(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR);
    if (include_act)
    {
        pwActDesc.setPointwiseMode(CUDNN_ATTR_POINTWISE_MODE, act_mode)
            .setDataType(CUDNN_ATTR_POINTWISE_MATH_PREC, comp_type)
            .finalize();
        if (!pwActDesc.finalized()) { return r; }

        opAct.setDesc(CUDNN_ATTR_OPERATION_POINTWISE_PW_DESCRIPTOR, pwActDesc)
            .setDesc(CUDNN_ATTR_OPERATION_POINTWISE_XDESC, biasOutT)
            .setDesc(CUDNN_ATTR_OPERATION_POINTWISE_YDESC, yT)
            .finalize();
        if (!opAct.finalized()) { return r; }
        ops.push_back(opAct.get());
    }

    auto plan = std::make_unique<CudnnGraphPlan>(gpu->cudnn_handle_, ops);
    if (!plan->ok())
    {
        // try fallback
        plan = std::make_unique<CudnnGraphPlan>(gpu->cudnn_handle_, ops, CUDNN_HEUR_MODE_FALLBACK);
    }
    if (!plan->ok()) { return r; }

    r.ws_bytes = plan->workspaceSize();
    r.plan = std::move(plan);
    // keepalive (cuDNN holds raw pointers)
    r.descs.push_back(std::move(xT));
    r.descs.push_back(std::move(wT));
    r.descs.push_back(std::move(bT));
    r.descs.push_back(std::move(yT));
    r.descs.push_back(std::move(convOutT));
    if (include_act)
    {
        r.descs.push_back(std::move(biasOutT));
    }
    r.descs.push_back(std::move(conv));
    r.descs.push_back(std::move(opConv));
    r.descs.push_back(std::move(pwAddDesc));
    r.descs.push_back(std::move(opAdd));
    if (include_act)
    {
        r.descs.push_back(std::move(pwActDesc));
        r.descs.push_back(std::move(opAct));
    }
    return r;
}

std::string makeKeyConvBias(const char* tag, const Matrix& X, const Matrix& W,
    const Matrix& bias, const Matrix& Y,
    const std::vector<int>& stride, const std::vector<int>& padding, int act_mode)
{
    auto pad4 = [](std::vector<int> v)
    {
        while (v.size() < 4)
        {
            v.push_back(1);
        }
        return v;
    };
    auto xd = pad4(X.getDim());
    auto wd = pad4(W.getDim());
    auto bd = pad4(bias.getDim());
    auto yd = pad4(Y.getDim());
    return std::format(
        "{}|dt={}|X={},{},{},{}|W={},{},{},{}|b={},{},{},{}|Y={},{},{},{}|s={},{}|p={},{}|act={}",
        tag, int(Y.getDataType()),
        xd[0], xd[1], xd[2], xd[3], wd[0], wd[1], wd[2], wd[3],
        bd[0], bd[1], bd[2], bd[3], yd[0], yd[1], yd[2], yd[3],
        stride[0], stride[1], padding[0], padding[1], act_mode);
}
}    // namespace

int CudnnGraphOps::convBiasForward(const Matrix& X, const Matrix& W, const Matrix& bias, Matrix& Y,
    const std::vector<int>& stride, const std::vector<int>& padding)
{
    if (!X.isCuda()) { return -100; }
    auto* gpu = X.gpu();
    if (!gpu || !gpu->cudnn_handle_) { return -101; }
    if (stride.size() != 2 || padding.size() != 2) { return -110; }

    auto key = makeKeyConvBias("CB", X, W, bias, Y, stride, padding, /*act*/ -1);
    PlanCacheEntry* entry = nullptr;
    {
        std::lock_guard<std::mutex> lk(planCacheMutex());
        auto it = planCache().find(key);
        if (it == planCache().end())
        {
            auto built = buildConvBiasActPlan(gpu, X, W, bias, Y, stride, padding,
                /*include_act*/ false, CUDNN_POINTWISE_RELU_FWD);
            if (!built.plan)
            {
                PlanCacheEntry e_neg;
                planCache().emplace(key, std::move(e_neg));
                return -104;
            }
            PlanCacheEntry e;
            e.plan = std::move(built.plan);
            e.descs = std::move(built.descs);
            if (built.ws_bytes > 0)
            {
                e.workspace.shared_data_->setGpu(gpu);
                e.workspace.resize(1, 1, 1, int(built.ws_bytes / e.workspace.getDataTypeSize() + 2));
            }
            it = planCache().emplace(key, std::move(e)).first;
            logGraphFirstUse("convBiasForward(fused)", key);
        }
        entry = &it->second;
    }
    if (!entry->plan) { return -1; }
    void* ws_ptr = entry->workspace.getDataSizeInByte() ? entry->workspace.getDataPtr() : nullptr;
    return entry->plan->execute({ 1, 2, 3, 4 },
        std::vector<void*>{ X.data(), W.data(), bias.data(), Y.data() }, ws_ptr);
}

int CudnnGraphOps::convBiasActForward(const Matrix& X, const Matrix& W, const Matrix& bias, Matrix& Y,
    const std::vector<int>& stride, const std::vector<int>& padding, int act_mode)
{
    if (!X.isCuda()) { return -100; }
    auto* gpu = X.gpu();
    if (!gpu || !gpu->cudnn_handle_) { return -101; }
    if (stride.size() != 2 || padding.size() != 2) { return -110; }

    cudnnPointwiseMode_t pwm;
    if (!mapActFwd(act_mode, pwm)) { return -111; }

    auto key = makeKeyConvBias("CBA", X, W, bias, Y, stride, padding, act_mode);
    PlanCacheEntry* entry = nullptr;
    {
        std::lock_guard<std::mutex> lk(planCacheMutex());
        auto it = planCache().find(key);
        if (it == planCache().end())
        {
            auto built = buildConvBiasActPlan(gpu, X, W, bias, Y, stride, padding,
                /*include_act*/ true, pwm);
            if (!built.plan)
            {
                PlanCacheEntry e_neg;
                planCache().emplace(key, std::move(e_neg));
                return -104;
            }
            PlanCacheEntry e;
            e.plan = std::move(built.plan);
            e.descs = std::move(built.descs);
            if (built.ws_bytes > 0)
            {
                e.workspace.shared_data_->setGpu(gpu);
                e.workspace.resize(1, 1, 1, int(built.ws_bytes / e.workspace.getDataTypeSize() + 2));
            }
            it = planCache().emplace(key, std::move(e)).first;
            logGraphFirstUse("convBiasActForward(fused)", key);
        }
        entry = &it->second;
    }
    if (!entry->plan) { return -1; }
    void* ws_ptr = entry->workspace.getDataSizeInByte() ? entry->workspace.getDataPtr() : nullptr;
    return entry->plan->execute({ 1, 2, 3, 4 },
        std::vector<void*>{ X.data(), W.data(), bias.data(), Y.data() }, ws_ptr);
}

// =============================================================================
// MatMul fused: matmul + biasAdd [+ activation]
// =============================================================================
//
// cccc::Matrix::mul 的语义是 col-major GEMM: R(M,N) = A(M,K) * B(K,N)
//   A.row=M, A.number=K   B.row=K, B.number=N   R.row=M, R.number=N
//
// 映射到 cuDNN backend matmul (row-major, dims [B,M,N], C = A*B):
//   将 cccc 的 col-major(M,K) 视为 row-major(K,M)，依此类推。
//   于是令 cuDNN 的 A_desc=B^cccc 视图 [1,N,K]，B_desc=A^cccc 视图 [1,K,M]，
//   C_desc=R^cccc 视图 [1,N,M]，三者的 data ptr 直接复用。
//
// bias: shape (out,1)，dataSize==out。在 C 视图 [1,N,M] 中沿 M 维存在，沿 N 维广播：
//   bias_tensor [1,1,M], strides [0,0,1]
//
namespace
{
struct MatmulShape
{
    int64_t M, K, N;
};

FusedBuildResult buildMatmulBiasActPlan(GpuControl* gpu,
    const Matrix& A, const Matrix& B, const Matrix& bias, const Matrix& C,
    bool include_act, cudnnPointwiseMode_t act_mode)
{
    FusedBuildResult r;
    auto comp_type = cudnnTypeOf(C.getDataType());
    if (A.getRow() != C.getRow() || B.getNumber() != C.getNumber() || A.getNumber() != B.getRow())
    {
        return r;    // shape mismatch
    }
    int64_t M = A.getRow(), K = A.getNumber(), N = B.getNumber();

    auto mkTensor3D = [&](int64_t uid, std::vector<int64_t> dims, std::vector<int64_t> strides, bool is_virtual)
    {
        return CudnnGraphTensor::make(uid, comp_type, dims, strides, is_virtual);
    };

    // A_desc (cuDNN) = B^cccc viewed [1,N,K] (data ptr = B.data)
    auto adT = mkTensor3D(1, { 1, N, K }, { N * K, K, 1 }, false);
    // B_desc (cuDNN) = A^cccc viewed [1,K,M] (data ptr = A.data)
    auto bdT = mkTensor3D(2, { 1, K, M }, { K * M, M, 1 }, false);
    // bias broadcast [1,1,M]
    auto biasT = mkTensor3D(3, { 1, 1, M }, { 0, 0, 1 }, false);
    // C_desc (cuDNN) = R^cccc viewed [1,N,M]
    auto cdT = mkTensor3D(4, { 1, N, M }, { N * M, M, 1 }, false);
    // virtual matmul output (same shape as C)
    auto matOutT = mkTensor3D(10, { 1, N, M }, { N * M, M, 1 }, true);
    auto biasOutT = include_act ? mkTensor3D(11, { 1, N, M }, { N * M, M, 1 }, true) : CudnnBackendDesc(CUDNN_BACKEND_TENSOR_DESCRIPTOR);

    // MATMUL desc
    CudnnBackendDesc mmDesc(CUDNN_BACKEND_MATMUL_DESCRIPTOR);
    mmDesc.setDataType(CUDNN_ATTR_MATMUL_COMP_TYPE, comp_type)
        .finalize();
    if (!mmDesc.finalized()) { return r; }

    CudnnBackendDesc opMM(CUDNN_BACKEND_OPERATION_MATMUL_DESCRIPTOR);
    opMM.setDesc(CUDNN_ATTR_OPERATION_MATMUL_ADESC, adT)
        .setDesc(CUDNN_ATTR_OPERATION_MATMUL_BDESC, bdT)
        .setDesc(CUDNN_ATTR_OPERATION_MATMUL_CDESC, matOutT)
        .setDesc(CUDNN_ATTR_OPERATION_MATMUL_DESC, mmDesc)
        .finalize();
    if (!opMM.finalized()) { return r; }

    // POINTWISE_ADD desc
    CudnnBackendDesc pwAddDesc(CUDNN_BACKEND_POINTWISE_DESCRIPTOR);
    pwAddDesc.setPointwiseMode(CUDNN_ATTR_POINTWISE_MODE, CUDNN_POINTWISE_ADD)
        .setDataType(CUDNN_ATTR_POINTWISE_MATH_PREC, comp_type)
        .finalize();
    if (!pwAddDesc.finalized()) { return r; }

    CudnnBackendDesc opAdd(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR);
    opAdd.setDesc(CUDNN_ATTR_OPERATION_POINTWISE_PW_DESCRIPTOR, pwAddDesc)
        .setDesc(CUDNN_ATTR_OPERATION_POINTWISE_XDESC, matOutT)
        .setDesc(CUDNN_ATTR_OPERATION_POINTWISE_BDESC, biasT)
        .setDesc(CUDNN_ATTR_OPERATION_POINTWISE_YDESC, include_act ? biasOutT : cdT)
        .finalize();
    if (!opAdd.finalized()) { return r; }

    std::vector<cudnnBackendDescriptor_t> ops = { opMM.get(), opAdd.get() };

    CudnnBackendDesc pwActDesc(CUDNN_BACKEND_POINTWISE_DESCRIPTOR);
    CudnnBackendDesc opAct(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR);
    if (include_act)
    {
        pwActDesc.setPointwiseMode(CUDNN_ATTR_POINTWISE_MODE, act_mode)
            .setDataType(CUDNN_ATTR_POINTWISE_MATH_PREC, comp_type)
            .finalize();
        if (!pwActDesc.finalized()) { return r; }
        opAct.setDesc(CUDNN_ATTR_OPERATION_POINTWISE_PW_DESCRIPTOR, pwActDesc)
            .setDesc(CUDNN_ATTR_OPERATION_POINTWISE_XDESC, biasOutT)
            .setDesc(CUDNN_ATTR_OPERATION_POINTWISE_YDESC, cdT)
            .finalize();
        if (!opAct.finalized()) { return r; }
        ops.push_back(opAct.get());
    }

    auto plan = std::make_unique<CudnnGraphPlan>(gpu->cudnn_handle_, ops);
    if (!plan->ok())
    {
        plan = std::make_unique<CudnnGraphPlan>(gpu->cudnn_handle_, ops, CUDNN_HEUR_MODE_FALLBACK);
    }
    if (!plan->ok()) { return r; }

    r.ws_bytes = plan->workspaceSize();
    r.plan = std::move(plan);
    r.descs.push_back(std::move(adT));
    r.descs.push_back(std::move(bdT));
    r.descs.push_back(std::move(biasT));
    r.descs.push_back(std::move(cdT));
    r.descs.push_back(std::move(matOutT));
    if (include_act)
    {
        r.descs.push_back(std::move(biasOutT));
    }
    r.descs.push_back(std::move(mmDesc));
    r.descs.push_back(std::move(opMM));
    r.descs.push_back(std::move(pwAddDesc));
    r.descs.push_back(std::move(opAdd));
    if (include_act)
    {
        r.descs.push_back(std::move(pwActDesc));
        r.descs.push_back(std::move(opAct));
    }
    return r;
}

std::string makeKeyMatmulBias(const char* tag, const Matrix& A, const Matrix& B,
    const Matrix& bias, const Matrix& C, int act_mode)
{
    return std::format("{}|dt={}|MKN={},{},{}|bs={}|act={}",
        tag, int(C.getDataType()),
        A.getRow(), A.getNumber(), B.getNumber(), bias.getDataSize(), act_mode);
}
}    // namespace

int CudnnGraphOps::matmulBiasForward(const Matrix& A, const Matrix& B, const Matrix& bias, Matrix& C)
{
    if (!C.isCuda()) { return -100; }
    auto* gpu = C.gpu();
    if (!gpu || !gpu->cudnn_handle_) { return -101; }

    auto key = makeKeyMatmulBias("MB", A, B, bias, C, /*act*/ -1);
    PlanCacheEntry* entry = nullptr;
    {
        std::lock_guard<std::mutex> lk(planCacheMutex());
        auto it = planCache().find(key);
        if (it == planCache().end())
        {
            auto built = buildMatmulBiasActPlan(gpu, A, B, bias, C, /*include_act*/ false, CUDNN_POINTWISE_RELU_FWD);
            if (!built.plan)
            {
                PlanCacheEntry e_neg;
                planCache().emplace(key, std::move(e_neg));
                return -104;
            }
            PlanCacheEntry e;
            e.plan = std::move(built.plan);
            e.descs = std::move(built.descs);
            if (built.ws_bytes > 0)
            {
                e.workspace.shared_data_->setGpu(gpu);
                e.workspace.resize(1, 1, 1, int(built.ws_bytes / e.workspace.getDataTypeSize() + 2));
            }
            it = planCache().emplace(key, std::move(e)).first;
            logGraphFirstUse("matmulBiasForward(fused)", key);
        }
        entry = &it->second;
    }
    if (!entry->plan) { return -1; }
    void* ws_ptr = entry->workspace.getDataSizeInByte() ? entry->workspace.getDataPtr() : nullptr;
    // 注意 uid 顺序：1=A_desc(=B^cccc), 2=B_desc(=A^cccc), 3=bias, 4=C
    return entry->plan->execute({ 1, 2, 3, 4 },
        std::vector<void*>{ B.data(), A.data(), bias.data(), C.data() }, ws_ptr);
}

int CudnnGraphOps::matmulBiasActForward(const Matrix& A, const Matrix& B, const Matrix& bias, Matrix& C, int act_mode)
{
    if (!C.isCuda()) { return -100; }
    auto* gpu = C.gpu();
    if (!gpu || !gpu->cudnn_handle_) { return -101; }

    cudnnPointwiseMode_t pwm;
    if (!mapActFwd(act_mode, pwm)) { return -111; }

    auto key = makeKeyMatmulBias("MBA", A, B, bias, C, act_mode);
    PlanCacheEntry* entry = nullptr;
    {
        std::lock_guard<std::mutex> lk(planCacheMutex());
        auto it = planCache().find(key);
        if (it == planCache().end())
        {
            auto built = buildMatmulBiasActPlan(gpu, A, B, bias, C, /*include_act*/ true, pwm);
            if (!built.plan)
            {
                PlanCacheEntry e_neg;
                planCache().emplace(key, std::move(e_neg));
                return -104;
            }
            PlanCacheEntry e;
            e.plan = std::move(built.plan);
            e.descs = std::move(built.descs);
            if (built.ws_bytes > 0)
            {
                e.workspace.shared_data_->setGpu(gpu);
                e.workspace.resize(1, 1, 1, int(built.ws_bytes / e.workspace.getDataTypeSize() + 2));
            }
            it = planCache().emplace(key, std::move(e)).first;
            logGraphFirstUse("matmulBiasActForward(fused)", key);
        }
        entry = &it->second;
    }
    if (!entry->plan) { return -1; }
    void* ws_ptr = entry->workspace.getDataSizeInByte() ? entry->workspace.getDataPtr() : nullptr;
    return entry->plan->execute({ 1, 2, 3, 4 },
        std::vector<void*>{ B.data(), A.data(), bias.data(), C.data() }, ws_ptr);
}

// =============================================================================
// 反向融合（option A）：activeBackward + ConvBackward(Data|Filter)
// =============================================================================
//
// 设 forward: Y_act = act(conv(X_in, W) + bias)。
// 反向链：dY_act → activeBwd → dPreAct (= dConvOut, ADD_BIAS 数据通路是恒等)
//                     → conv_bwd_data: dX_in = conv_bwd_data(W, dPreAct)
//                     → conv_bwd_filter: dW    = conv_bwd_filter(X_in, dPreAct)
//
// cuDNN backend 的 POINTWISE_*_BWD 模式定义：
//   relu_bwd(dY, x_or_y) = dY * (x>0 ? 1 : 0)（cuDNN 9 的实现接受 x 或 y 作为参考；
//     对 ReLU 二者等价）
//   sigmoid_bwd(dY, y)   = dY * y * (1-y)
//   tanh_bwd(dY, y)      = dY * (1 - y^2)
// 我们统一以 forward 的输出 Y_act 作为参考 (POINTWISE_*_BWD 的 Y/X 入口)。
//
namespace
{
// Build:
//   Op1: POINTWISE act_bwd(dY_act_real, Y_act_real) -> dPreAct(virtual)
//   Op2: CONV_BWD_(DATA|FILTER) consuming dPreAct(virtual) and the other real tensor
//        -> output real tensor
//
// uids:
//   1 = Y_act        (forward output reference, real)
//   2 = dY_act       (real)
//   3 = other_in     (real: W for bwd_data; X_in for bwd_filter)
//   4 = out          (real: dX_in for bwd_data; dW for bwd_filter)
//   10 = dPreAct     (virtual)
struct BwdKind
{
    bool is_data;
};
FusedBuildResult buildActConvBwdPlan(GpuControl* gpu, BwdKind kind,
    const Matrix& Y_act, const Matrix& dY_act, const Matrix& other_in, const Matrix& out_,
    const std::vector<int>& stride, const std::vector<int>& padding,
    cudnnPointwiseMode_t act_bwd_mode)
{
    FusedBuildResult r;
    auto comp_type = cudnnTypeOf(out_.getDataType());

    auto yT = CudnnGraphTensor::fromMatrix(Y_act, 1);
    auto dyT = CudnnGraphTensor::fromMatrix(dY_act, 2);
    auto inT = CudnnGraphTensor::fromMatrix(other_in, 3);
    auto outT = CudnnGraphTensor::fromMatrix(out_, 4);

    // virtual dPreAct, same shape/strides as Y_act
    std::vector<int64_t> y_dims, y_strides;
    {
        auto pad4 = [](std::vector<int> v)
        {
            while (v.size() < 4)
            {
                v.push_back(1);
            }
            return v;
        };
        auto yd = pad4(Y_act.getDim());
        y_dims = { yd[3], yd[2], yd[1], yd[0] };    // [N,C,H,W]
        y_strides = CudnnGraphTensor::packedStridesNCHW(y_dims);
    }
    auto dPreActT = CudnnGraphTensor::make(10, comp_type, y_dims, y_strides, /*is_virtual*/ true);

    // POINTWISE backward desc
    CudnnBackendDesc pwBwd(CUDNN_BACKEND_POINTWISE_DESCRIPTOR);
    pwBwd.setPointwiseMode(CUDNN_ATTR_POINTWISE_MODE, act_bwd_mode)
        .setDataType(CUDNN_ATTR_POINTWISE_MATH_PREC, comp_type)
        .finalize();
    if (!pwBwd.finalized()) { return r; }

    // Op1: POINTWISE_*_BWD: X = Y_act, B = dY_act, Y = dPreAct
    // (在 backend graph 里，POINTWISE 的 BWD 模式约定 X=ref, B=dY, Y=dX)
    CudnnBackendDesc opBwd(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR);
    opBwd.setDesc(CUDNN_ATTR_OPERATION_POINTWISE_PW_DESCRIPTOR, pwBwd)
        .setDesc(CUDNN_ATTR_OPERATION_POINTWISE_XDESC, yT)
        .setDesc(CUDNN_ATTR_OPERATION_POINTWISE_BDESC, dyT)
        .setDesc(CUDNN_ATTR_OPERATION_POINTWISE_YDESC, dPreActT)
        .finalize();
    if (!opBwd.finalized()) { return r; }

    auto conv = buildConvDesc(comp_type, stride, padding);
    if (!conv.finalized()) { return r; }

    CudnnBackendDesc opConv(kind.is_data ? CUDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_DATA_DESCRIPTOR : CUDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_FILTER_DESCRIPTOR);
    double a64 = 1.0, b64 = 0.0;
    if (kind.is_data)
    {
        // bwd_data: dY=dPreAct, W=other_in, dX=out
        opConv.setDesc(CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_W, inT)
            .setDesc(CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_DY, dPreActT)
            .setDesc(CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_DX, outT)
            .setDesc(CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_CONV_DESC, conv)
            .setF64(CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_ALPHA, a64)
            .setF64(CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_BETA, b64)
            .finalize();
    }
    else
    {
        // bwd_filter: X=other_in (X_in), dY=dPreAct, dW=out
        opConv.setDesc(CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_X, inT)
            .setDesc(CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_DY, dPreActT)
            .setDesc(CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_DW, outT)
            .setDesc(CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_CONV_DESC, conv)
            .setF64(CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_ALPHA, a64)
            .setF64(CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_BETA, b64)
            .finalize();
    }
    if (!opConv.finalized()) { return r; }

    std::vector<cudnnBackendDescriptor_t> ops = { opBwd.get(), opConv.get() };
    auto plan = std::make_unique<CudnnGraphPlan>(gpu->cudnn_handle_, ops);
    if (!plan->ok())
    {
        plan = std::make_unique<CudnnGraphPlan>(gpu->cudnn_handle_, ops, CUDNN_HEUR_MODE_FALLBACK);
    }
    if (!plan->ok()) { return r; }

    r.ws_bytes = plan->workspaceSize();
    r.plan = std::move(plan);
    r.descs.push_back(std::move(yT));
    r.descs.push_back(std::move(dyT));
    r.descs.push_back(std::move(inT));
    r.descs.push_back(std::move(outT));
    r.descs.push_back(std::move(dPreActT));
    r.descs.push_back(std::move(pwBwd));
    r.descs.push_back(std::move(opBwd));
    r.descs.push_back(std::move(conv));
    r.descs.push_back(std::move(opConv));
    return r;
}

std::string makeKeyActConvBwd(const char* tag, const Matrix& Y, const Matrix& dY,
    const Matrix& other, const Matrix& out_,
    const std::vector<int>& stride, const std::vector<int>& padding, int act_mode)
{
    auto pad4 = [](std::vector<int> v)
    {
        while (v.size() < 4)
        {
            v.push_back(1);
        }
        return v;
    };
    auto yd = pad4(Y.getDim());
    auto dd = pad4(dY.getDim());
    auto od = pad4(other.getDim());
    auto rd = pad4(out_.getDim());
    return std::format(
        "{}|dt={}|Y={},{},{},{}|dY={},{},{},{}|in={},{},{},{}|out={},{},{},{}|s={},{}|p={},{}|act={}",
        tag, int(out_.getDataType()),
        yd[0], yd[1], yd[2], yd[3], dd[0], dd[1], dd[2], dd[3],
        od[0], od[1], od[2], od[3], rd[0], rd[1], rd[2], rd[3],
        stride[0], stride[1], padding[0], padding[1], act_mode);
}
}    // namespace

int CudnnGraphOps::actConvBwdData(const Matrix& Y_act, const Matrix& dY_act, const Matrix& W, Matrix& dX_in,
    const std::vector<int>& stride, const std::vector<int>& padding, int act_mode)
{
    if (!dX_in.isCuda()) { return -100; }
    auto* gpu = dX_in.gpu();
    if (!gpu || !gpu->cudnn_handle_) { return -101; }
    if (stride.size() != 2 || padding.size() != 2) { return -110; }

    cudnnPointwiseMode_t pwm;
    if (!mapActBwd(act_mode, pwm)) { return -111; }

    auto key = makeKeyActConvBwd("ACBD", Y_act, dY_act, W, dX_in, stride, padding, act_mode);
    PlanCacheEntry* entry = nullptr;
    {
        std::lock_guard<std::mutex> lk(planCacheMutex());
        auto it = planCache().find(key);
        if (it == planCache().end())
        {
            auto built = buildActConvBwdPlan(gpu, BwdKind{ true }, Y_act, dY_act, W, dX_in, stride, padding, pwm);
            if (!built.plan)
            {
                PlanCacheEntry e_neg;
                planCache().emplace(key, std::move(e_neg));
                return -104;
            }
            PlanCacheEntry e;
            e.plan = std::move(built.plan);
            e.descs = std::move(built.descs);
            if (built.ws_bytes > 0)
            {
                e.workspace.shared_data_->setGpu(gpu);
                e.workspace.resize(1, 1, 1, int(built.ws_bytes / e.workspace.getDataTypeSize() + 2));
            }
            it = planCache().emplace(key, std::move(e)).first;
            logGraphFirstUse("actConvBwdData(fused)", key);
        }
        entry = &it->second;
    }
    if (!entry->plan) { return -1; }
    void* ws_ptr = entry->workspace.getDataSizeInByte() ? entry->workspace.getDataPtr() : nullptr;
    // uids: 1=Y_act, 2=dY_act, 3=W, 4=dX_in
    return entry->plan->execute({ 1, 2, 3, 4 },
        std::vector<void*>{ Y_act.data(), dY_act.data(), W.data(), dX_in.data() }, ws_ptr);
}

int CudnnGraphOps::actConvBwdFilter(const Matrix& Y_act, const Matrix& dY_act, const Matrix& X_in, Matrix& dW,
    const std::vector<int>& stride, const std::vector<int>& padding, int act_mode)
{
    if (!dW.isCuda()) { return -100; }
    auto* gpu = dW.gpu();
    if (!gpu || !gpu->cudnn_handle_) { return -101; }
    if (stride.size() != 2 || padding.size() != 2) { return -110; }

    cudnnPointwiseMode_t pwm;
    if (!mapActBwd(act_mode, pwm)) { return -111; }

    auto key = makeKeyActConvBwd("ACBF", Y_act, dY_act, X_in, dW, stride, padding, act_mode);
    PlanCacheEntry* entry = nullptr;
    {
        std::lock_guard<std::mutex> lk(planCacheMutex());
        auto it = planCache().find(key);
        if (it == planCache().end())
        {
            auto built = buildActConvBwdPlan(gpu, BwdKind{ false }, Y_act, dY_act, X_in, dW, stride, padding, pwm);
            if (!built.plan)
            {
                PlanCacheEntry e_neg;
                planCache().emplace(key, std::move(e_neg));
                return -104;
            }
            PlanCacheEntry e;
            e.plan = std::move(built.plan);
            e.descs = std::move(built.descs);
            if (built.ws_bytes > 0)
            {
                e.workspace.shared_data_->setGpu(gpu);
                e.workspace.resize(1, 1, 1, int(built.ws_bytes / e.workspace.getDataTypeSize() + 2));
            }
            it = planCache().emplace(key, std::move(e)).first;
            logGraphFirstUse("actConvBwdFilter(fused)", key);
        }
        entry = &it->second;
    }
    if (!entry->plan) { return -1; }
    void* ws_ptr = entry->workspace.getDataSizeInByte() ? entry->workspace.getDataPtr() : nullptr;
    // uids: 1=Y_act, 2=dY_act, 3=X_in, 4=dW
    return entry->plan->execute({ 1, 2, 3, 4 },
        std::vector<void*>{ Y_act.data(), dY_act.data(), X_in.data(), dW.data() }, ws_ptr);
}

// =============================================================================
// CudnnOpQueueGraph  – 整网前向一张图
// =============================================================================

namespace
{
// NCHW dims/strides (packed) 从 cccc Matrix 推断
// cccc 存储顺序 (W, H, C, N)；getDim() 返回 {W, H, C, N}。
// 逆序得到 cuDNN 期望的外→内顺序 [N, C, H, W]。
static void nchwOf(const Matrix& m, std::vector<int64_t>& dims, std::vector<int64_t>& strides)
{
    auto cdim = m.getDim();
    while (cdim.size() < 4)
    {
        cdim.push_back(1);
    }
    dims.assign(cdim.rbegin(), cdim.rend());    // [N, C, H, W]
    strides = CudnnGraphTensor::packedStridesNCHW(dims);
}
}    // namespace

CudnnBackendDesc& CudnnOpQueueGraph::addDesc(CudnnBackendDesc&& d)
{
    descs_.push_back(std::move(d));
    return descs_.back();
}

bool CudnnOpQueueGraph::getOrAssignUid(Matrix* m, const std::vector<int64_t>& dims,
    const std::vector<int64_t>& strides, int64_t& uid_out)
{
    if (!m) { return false; }
    void* ptr = m->getDataPtr();
    auto it = matrix_entry_.find(m);
    if (it != matrix_entry_.end())
    {
        // 同一逻辑 Matrix 已有记录，检查形状是否一致。
        // 注意：显存复用后，不同逻辑张量可能共享同一 data 指针；cuDNN graph
        // 必须为它们分配不同 UID，否则会把非重叠生命周期的张量错误折叠成同一节点。
        if (it->second.dims == dims && it->second.strides == strides)
        {
            uid_out = it->second.uid;
            return true;
        }
        // 形状不同 → 该 Matrix 在不同 op 中需要不同视图 → 不支持整网图。
        LOG("[CudnnOpQueueGraph] tensor shape conflict for matrix {} ptr {}: "
            "existing dims[0]={} vs requested dims[0]={} → plain fallback\n",
            (void*)m,
            (void*)ptr,
            it->second.dims.empty() ? -1 : it->second.dims[0],
            dims.empty() ? -1 : dims[0]);
        return false;
    }
    // 首次见到此逻辑 Matrix，分配新 uid。多个 uid 允许绑定到同一 ptr，
    // 由显存复用规划保证这些逻辑张量的实际生命周期不重叠。
    TensorEntry e;
    e.uid = next_uid_++;
    e.dims = dims;
    e.strides = strides;
    uid_out = e.uid;
    matrix_entry_[m] = e;
    vp_uids_.push_back(e.uid);
    vp_ptrs_.push_back(ptr);
    return true;
}

bool CudnnOpQueueGraph::buildOp(MatrixOp& op, std::vector<cudnnBackendDescriptor_t>& op_handles)
{
    auto type = op.getType();
    auto& in = op.getMatrixIn();
    auto& out = op.getMatrixOut();

    // ---- 共用数据类型（取第一个输出的类型）----
    cudnnDataType_t dt = CUDNN_DATA_FLOAT;
    if (!out.empty() && out[0])
    {
        dt = cudnnTypeOf(out[0]->getDataType());
    }

    // ----------------------------------------------------------------
    // CONV
    // ----------------------------------------------------------------
    if (type == MatrixOpType::CONV)
    {
        if (in.size() < 2 || out.size() < 1 || !in[0] || !in[1] || !out[0])
        {
            return false;
        }
        auto& stride = op.getStride();
        auto& padding = op.getPadding();
        if (stride.size() != 2 || padding.size() != 2)
        {
            return false;
        }

        std::vector<int64_t> xd, xs, wd, ws, yd, ys;
        nchwOf(*in[0], xd, xs);
        nchwOf(*in[1], wd, ws);
        nchwOf(*out[0], yd, ys);
        int64_t uid_x, uid_w, uid_y;
        if (!getOrAssignUid(in[0].get(), xd, xs, uid_x))
        {
            return false;
        }
        if (!getOrAssignUid(in[1].get(), wd, ws, uid_w))
        {
            return false;
        }
        if (!getOrAssignUid(out[0].get(), yd, ys, uid_y))
        {
            return false;
        }

        auto& xT = addDesc(CudnnGraphTensor::make(uid_x, dt, xd, xs));
        auto& wT = addDesc(CudnnGraphTensor::make(uid_w, dt, wd, ws));
        auto& yT = addDesc(CudnnGraphTensor::make(uid_y, dt, yd, ys));

        CudnnBackendDesc conv(CUDNN_BACKEND_CONVOLUTION_DESCRIPTOR);
        conv.setDataType(CUDNN_ATTR_CONVOLUTION_COMP_TYPE, dt)
            .setI64(CUDNN_ATTR_CONVOLUTION_SPATIAL_DIMS, 2)
            .setI64Array(CUDNN_ATTR_CONVOLUTION_DILATIONS, { 1, 1 })
            .setI64Array(CUDNN_ATTR_CONVOLUTION_FILTER_STRIDES,
                { (int64_t)stride[1], (int64_t)stride[0] })
            .setI64Array(CUDNN_ATTR_CONVOLUTION_PRE_PADDINGS,
                { (int64_t)padding[1], (int64_t)padding[0] })
            .setI64Array(CUDNN_ATTR_CONVOLUTION_POST_PADDINGS,
                { (int64_t)padding[1], (int64_t)padding[0] });
        cudnnConvolutionMode_t cmode = CUDNN_CROSS_CORRELATION;
        conv.setAttribute(CUDNN_ATTR_CONVOLUTION_CONV_MODE, CUDNN_TYPE_CONVOLUTION_MODE, 1, &cmode);
        conv.finalize();
        if (!conv.finalized())
        {
            return false;
        }

        double alpha = 1.0, beta = 0.0;
        CudnnBackendDesc opConv(CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR);
        opConv.setDesc(CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_X, xT)
            .setDesc(CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_W, wT)
            .setDesc(CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_Y, yT)
            .setDesc(CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_CONV_DESC, addDesc(std::move(conv)))
            .setF64(CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_ALPHA, alpha)
            .setF64(CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_BETA, beta)
            .finalize();
        if (!opConv.finalized())
        {
            return false;
        }
        op_handles.push_back(addDesc(std::move(opConv)).get());
        return true;
    }

    // ----------------------------------------------------------------
    // POOL
    // ----------------------------------------------------------------
    if (type == MatrixOpType::POOL)
    {
        if (in.size() < 1 || out.size() < 1 || !in[0] || !out[0])
        {
            return false;
        }
        auto& window = op.getWindow();
        auto& stride = op.getStride();
        auto& padding = op.getPadding();
        if (window.size() < 2 || stride.size() < 2 || padding.size() < 2)
        {
            return false;
        }

        cudnnResampleMode_t rmode;
        switch (op.getPoolingType())
        {
        case POOLING_MAX: rmode = CUDNN_RESAMPLE_MAXPOOL; break;
        case POOLING_AVERAGE_PADDING: rmode = CUDNN_RESAMPLE_AVGPOOL_INCLUDE_PADDING; break;
        case POOLING_AVERAGE_NOPADDING: rmode = CUDNN_RESAMPLE_AVGPOOL_EXCLUDE_PADDING; break;
        default: return false;
        }

        std::vector<int64_t> xd, xs, yd, ys;
        nchwOf(*in[0], xd, xs);
        nchwOf(*out[0], yd, ys);
        int64_t uid_x, uid_y;
        if (!getOrAssignUid(in[0].get(), xd, xs, uid_x))
        {
            return false;
        }
        if (!getOrAssignUid(out[0].get(), yd, ys, uid_y))
        {
            return false;
        }

        auto& xT = addDesc(CudnnGraphTensor::make(uid_x, dt, xd, xs));
        auto& yT = addDesc(CudnnGraphTensor::make(uid_y, dt, yd, ys));

        CudnnBackendDesc rsmp(CUDNN_BACKEND_RESAMPLE_DESCRIPTOR);
        rsmp.setAttribute(CUDNN_ATTR_RESAMPLE_MODE, CUDNN_TYPE_RESAMPLE_MODE, 1, &rmode)
            .setDataType(CUDNN_ATTR_RESAMPLE_COMP_TYPE, dt)
            .setI64(CUDNN_ATTR_RESAMPLE_SPATIAL_DIMS, 2)
            .setI64Array(CUDNN_ATTR_RESAMPLE_WINDOW_DIMS,
                { (int64_t)window[1], (int64_t)window[0] })
            .setI64Array(CUDNN_ATTR_RESAMPLE_STRIDES,
                { (int64_t)stride[1], (int64_t)stride[0] })
            .setI64Array(CUDNN_ATTR_RESAMPLE_PRE_PADDINGS,
                { (int64_t)padding[1], (int64_t)padding[0] })
            .setI64Array(CUDNN_ATTR_RESAMPLE_POST_PADDINGS,
                { (int64_t)padding[1], (int64_t)padding[0] });
        cudnnPaddingMode_t pmode = CUDNN_ZERO_PAD;
        rsmp.setAttribute(CUDNN_ATTR_RESAMPLE_PADDING_MODE, CUDNN_TYPE_PADDING_MODE, 1, &pmode);
        rsmp.finalize();
        if (!rsmp.finalized())
        {
            return false;
        }

        CudnnBackendDesc opPool(CUDNN_BACKEND_OPERATION_RESAMPLE_FWD_DESCRIPTOR);
        opPool.setDesc(CUDNN_ATTR_OPERATION_RESAMPLE_FWD_XDESC, xT)
            .setDesc(CUDNN_ATTR_OPERATION_RESAMPLE_FWD_YDESC, yT)
            .setDesc(CUDNN_ATTR_OPERATION_RESAMPLE_FWD_DESC, addDesc(std::move(rsmp)))
            .finalize();
        if (!opPool.finalized())
        {
            return false;
        }
        op_handles.push_back(addDesc(std::move(opPool)).get());
        return true;
    }

    // ----------------------------------------------------------------
    // ACTIVE (relu / sigmoid / tanh)
    // ----------------------------------------------------------------
    if (type == MatrixOpType::ACTIVE)
    {
        if (in.size() < 1 || out.size() < 1 || !in[0] || !out[0])
        {
            return false;
        }
        cudnnPointwiseMode_t pwm;
        if (!mapActFwd(int(op.getActiveType()), pwm))
        {
            LOG("[CudnnOpQueueGraph] unsupported activation type {} → plain fallback\n",
                int(op.getActiveType()));
            return false;
        }
        std::vector<int64_t> xd, xs, yd, ys;
        nchwOf(*in[0], xd, xs);
        nchwOf(*out[0], yd, ys);
        int64_t uid_x, uid_y;
        if (!getOrAssignUid(in[0].get(), xd, xs, uid_x))
        {
            return false;
        }
        if (!getOrAssignUid(out[0].get(), yd, ys, uid_y))
        {
            return false;
        }

        auto& xT = addDesc(CudnnGraphTensor::make(uid_x, dt, xd, xs));
        auto& yT = addDesc(CudnnGraphTensor::make(uid_y, dt, yd, ys));

        CudnnBackendDesc pwDesc(CUDNN_BACKEND_POINTWISE_DESCRIPTOR);
        pwDesc.setPointwiseMode(CUDNN_ATTR_POINTWISE_MODE, pwm)
            .setDataType(CUDNN_ATTR_POINTWISE_MATH_PREC, dt)
            .finalize();
        if (!pwDesc.finalized())
        {
            return false;
        }

        CudnnBackendDesc opAct(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR);
        opAct.setDesc(CUDNN_ATTR_OPERATION_POINTWISE_PW_DESCRIPTOR, addDesc(std::move(pwDesc)))
            .setDesc(CUDNN_ATTR_OPERATION_POINTWISE_XDESC, xT)
            .setDesc(CUDNN_ATTR_OPERATION_POINTWISE_YDESC, yT)
            .finalize();
        if (!opAct.finalized())
        {
            return false;
        }
        op_handles.push_back(addDesc(std::move(opAct)).get());
        return true;
    }

    // ----------------------------------------------------------------
    // ADD_BIAS
    //   两种 bias 格式：
    //   a) conv 风格: bias.dataSize == bias.channel → [1,C,1,1] + broadcast
    //   b) matmul 风格: bias.dataSize == out.row → [1,1,M] + broadcast
    // ----------------------------------------------------------------
    if (type == MatrixOpType::ADD_BIAS)
    {
        if (in.size() < 2 || out.size() < 1 || !in[0] || !in[1] || !out[0])
        {
            return false;
        }
        Matrix* X = in[0].get();
        Matrix* bias = in[1].get();
        Matrix* Y = out[0].get();

        // 判断 bias 风格
        bool conv_bias = (bias->getDataSize() == (size_t)bias->getChannel()
            && bias->getChannel() > 0);
        bool mm_bias = !conv_bias && (bias->getDataSize() == (size_t)Y->getRow());

        if (!conv_bias && !mm_bias)
        {
            LOG("[CudnnOpQueueGraph] ADD_BIAS: unrecognized bias shape (size={}, ch={}, Y.row={}) "
                "→ plain fallback\n",
                bias->getDataSize(), bias->getChannel(), Y->getRow());
            return false;
        }

        // 输入/输出张量用自然 NCHW 视图（与前后 conv/matmul 的视图一致）
        // 对 matmul 后的 ADD_BIAS，X/Y 的自然 NCHW 视图 [N,1,1,M] 与 matmul 输出
        // 视图 [1,N,M] 不同 → 形状冲突 → 整网图回退 plain。
        // 这是预期行为：conv→FC 边界处的整网图构建失败，退回 plain。
        std::vector<int64_t> xd, xs, yd, ys;
        nchwOf(*X, xd, xs);
        nchwOf(*Y, yd, ys);
        int64_t uid_x, uid_y;
        if (!getOrAssignUid(X, xd, xs, uid_x))
        {
            return false;
        }
        if (!getOrAssignUid(Y, yd, ys, uid_y))
        {
            return false;
        }

        // bias 张量描述
        int64_t uid_b = 0;
        if (conv_bias)
        {
            int64_t c = (int64_t)bias->getChannel();
            std::vector<int64_t> bd{ 1, c, 1, 1 };
            std::vector<int64_t> bs{ c, 1, 0, 0 };
            if (!getOrAssignUid(bias, bd, bs, uid_b))
            {
                return false;
            }
            addDesc(CudnnGraphTensor::make(uid_b, dt, bd, bs));
        }
        else    // mm_bias: [1,1,M] strides [0,0,1]
        {
            int64_t M = (int64_t)Y->getRow();
            std::vector<int64_t> bd{ 1, 1, M };
            std::vector<int64_t> bs{ 0, 0, 1 };
            if (!getOrAssignUid(bias, bd, bs, uid_b))
            {
                return false;
            }
            addDesc(CudnnGraphTensor::make(uid_b, dt, bd, bs));
        }

        auto& xT = addDesc(CudnnGraphTensor::make(uid_x, dt, xd, xs));
        auto& yT = addDesc(CudnnGraphTensor::make(uid_y, dt, yd, ys));
        auto* bT_ptr = &descs_.back();    // bias tensor just added above

        CudnnBackendDesc pwDesc(CUDNN_BACKEND_POINTWISE_DESCRIPTOR);
        pwDesc.setPointwiseMode(CUDNN_ATTR_POINTWISE_MODE, CUDNN_POINTWISE_ADD)
            .setDataType(CUDNN_ATTR_POINTWISE_MATH_PREC, dt)
            .finalize();
        if (!pwDesc.finalized())
        {
            return false;
        }

        CudnnBackendDesc opAdd(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR);
        opAdd.setDesc(CUDNN_ATTR_OPERATION_POINTWISE_PW_DESCRIPTOR, addDesc(std::move(pwDesc)))
            .setDesc(CUDNN_ATTR_OPERATION_POINTWISE_XDESC, xT)
            .setDesc(CUDNN_ATTR_OPERATION_POINTWISE_BDESC, *bT_ptr)
            .setDesc(CUDNN_ATTR_OPERATION_POINTWISE_YDESC, yT)
            .finalize();
        if (!opAdd.finalized())
        {
            return false;
        }
        op_handles.push_back(addDesc(std::move(opAdd)).get());
        return true;
    }

    // ----------------------------------------------------------------
    // MUL (matmul: cccc col-major C=A*B → cuDNN row-major mapping)
    // ----------------------------------------------------------------
    if (type == MatrixOpType::MUL)
    {
        if (in.size() < 2 || out.size() < 1 || !in[0] || !in[1] || !out[0])
        {
            return false;
        }
        Matrix* A = in[0].get();     // weight: row=M, number=K
        Matrix* B = in[1].get();     // input:  row=K, number=N
        Matrix* C = out[0].get();    // output: row=M, number=N

        int64_t M = A->getRow(), K = A->getNumber(), N = B->getNumber();
        if (A->getNumber() != B->getRow() || C->getRow() != M || C->getNumber() != N)
        {
            LOG("[CudnnOpQueueGraph] MUL shape mismatch → plain fallback\n");
            return false;
        }

        // cuDNN row-major mapping (see buildMatmulBiasActPlan comment)
        // adT = B^cccc: [1, N, K], bdT = A^cccc: [1, K, M], cdT = C^cccc: [1, N, M]
        std::vector<int64_t> adims{ 1, N, K }, astrides{ N * K, K, 1 };
        std::vector<int64_t> bdims{ 1, K, M }, bstrides{ K * M, M, 1 };
        std::vector<int64_t> cdims{ 1, N, M }, cstrides{ N * M, M, 1 };

        int64_t uid_a, uid_b, uid_c;
        // B (cccc input) → cuDNN A_desc: if B was previously assigned a different shape
        // (e.g., NCHW output of pool), getOrAssignUid returns false → plain fallback.
        if (!getOrAssignUid(B, adims, astrides, uid_a))
        {
            return false;
        }
        if (!getOrAssignUid(A, bdims, bstrides, uid_b))
        {
            return false;
        }
        if (!getOrAssignUid(C, cdims, cstrides, uid_c))
        {
            return false;
        }

        auto& aT = addDesc(CudnnGraphTensor::make(uid_a, dt, adims, astrides));
        auto& bT = addDesc(CudnnGraphTensor::make(uid_b, dt, bdims, bstrides));
        auto& cT = addDesc(CudnnGraphTensor::make(uid_c, dt, cdims, cstrides));

        CudnnBackendDesc mmDesc(CUDNN_BACKEND_MATMUL_DESCRIPTOR);
        mmDesc.setDataType(CUDNN_ATTR_MATMUL_COMP_TYPE, dt).finalize();
        if (!mmDesc.finalized())
        {
            return false;
        }

        CudnnBackendDesc opMM(CUDNN_BACKEND_OPERATION_MATMUL_DESCRIPTOR);
        opMM.setDesc(CUDNN_ATTR_OPERATION_MATMUL_ADESC, aT)
            .setDesc(CUDNN_ATTR_OPERATION_MATMUL_BDESC, bT)
            .setDesc(CUDNN_ATTR_OPERATION_MATMUL_CDESC, cT)
            .setDesc(CUDNN_ATTR_OPERATION_MATMUL_DESC, addDesc(std::move(mmDesc)))
            .finalize();
        if (!opMM.finalized())
        {
            return false;
        }
        op_handles.push_back(addDesc(std::move(opMM)).get());
        return true;
    }

    // ----------------------------------------------------------------
    // ADD (elementwise: Y = a0*X + a1*in[1] + b0*prev_Y)
    //   仅支持双输入、输出同形、alpha=1/1/0 的简单形式
    // ----------------------------------------------------------------
    if (type == MatrixOpType::ADD)
    {
        if (in.size() < 2 || out.size() < 1 || !in[0] || !in[1] || !out[0])
        {
            return false;
        }
        std::vector<int64_t> xd, xs, bd, bs, yd, ys;
        nchwOf(*in[0], xd, xs);
        nchwOf(*in[1], bd, bs);
        nchwOf(*out[0], yd, ys);
        int64_t uid_x, uid_b, uid_y;
        if (!getOrAssignUid(in[0].get(), xd, xs, uid_x))
        {
            return false;
        }
        if (!getOrAssignUid(in[1].get(), bd, bs, uid_b))
        {
            return false;
        }
        if (!getOrAssignUid(out[0].get(), yd, ys, uid_y))
        {
            return false;
        }

        auto& xT = addDesc(CudnnGraphTensor::make(uid_x, dt, xd, xs));
        auto& bT = addDesc(CudnnGraphTensor::make(uid_b, dt, bd, bs));
        auto& yT = addDesc(CudnnGraphTensor::make(uid_y, dt, yd, ys));

        CudnnBackendDesc pwDesc(CUDNN_BACKEND_POINTWISE_DESCRIPTOR);
        pwDesc.setPointwiseMode(CUDNN_ATTR_POINTWISE_MODE, CUDNN_POINTWISE_ADD)
            .setDataType(CUDNN_ATTR_POINTWISE_MATH_PREC, dt)
            .finalize();
        if (!pwDesc.finalized())
        {
            return false;
        }

        CudnnBackendDesc opAdd(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR);
        opAdd.setDesc(CUDNN_ATTR_OPERATION_POINTWISE_PW_DESCRIPTOR, addDesc(std::move(pwDesc)))
            .setDesc(CUDNN_ATTR_OPERATION_POINTWISE_XDESC, xT)
            .setDesc(CUDNN_ATTR_OPERATION_POINTWISE_BDESC, bT)
            .setDesc(CUDNN_ATTR_OPERATION_POINTWISE_YDESC, yT)
            .finalize();
        if (!opAdd.finalized())
        {
            return false;
        }
        op_handles.push_back(addDesc(std::move(opAdd)).get());
        return true;
    }

    // ----------------------------------------------------------------
    // ELE_MUL (elementwise multiply)
    // ----------------------------------------------------------------
    if (type == MatrixOpType::ELE_MUL)
    {
        if (in.size() < 2 || out.size() < 1 || !in[0] || !in[1] || !out[0])
        {
            return false;
        }
        std::vector<int64_t> xd, xs, bd, bs, yd, ys;
        nchwOf(*in[0], xd, xs);
        nchwOf(*in[1], bd, bs);
        nchwOf(*out[0], yd, ys);
        int64_t uid_x, uid_b, uid_y;
        if (!getOrAssignUid(in[0].get(), xd, xs, uid_x))
        {
            return false;
        }
        if (!getOrAssignUid(in[1].get(), bd, bs, uid_b))
        {
            return false;
        }
        if (!getOrAssignUid(out[0].get(), yd, ys, uid_y))
        {
            return false;
        }

        auto& xT = addDesc(CudnnGraphTensor::make(uid_x, dt, xd, xs));
        auto& bT = addDesc(CudnnGraphTensor::make(uid_b, dt, bd, bs));
        auto& yT = addDesc(CudnnGraphTensor::make(uid_y, dt, yd, ys));

        CudnnBackendDesc pwDesc(CUDNN_BACKEND_POINTWISE_DESCRIPTOR);
        pwDesc.setPointwiseMode(CUDNN_ATTR_POINTWISE_MODE, CUDNN_POINTWISE_MUL)
            .setDataType(CUDNN_ATTR_POINTWISE_MATH_PREC, dt)
            .finalize();
        if (!pwDesc.finalized())
        {
            return false;
        }

        CudnnBackendDesc opMul(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR);
        opMul.setDesc(CUDNN_ATTR_OPERATION_POINTWISE_PW_DESCRIPTOR, addDesc(std::move(pwDesc)))
            .setDesc(CUDNN_ATTR_OPERATION_POINTWISE_XDESC, xT)
            .setDesc(CUDNN_ATTR_OPERATION_POINTWISE_BDESC, bT)
            .setDesc(CUDNN_ATTR_OPERATION_POINTWISE_YDESC, yT)
            .finalize();
        if (!opMul.finalized())
        {
            return false;
        }
        op_handles.push_back(addDesc(std::move(opMul)).get());
        return true;
    }

    // ----------------------------------------------------------------
    // RESHAPE: 在 cuDNN backend graph 中没有对应节点，但可以用相同数据
    // 重新描述张量视图的方式绕过（下游 op 的 getOrAssignUid 会检查冲突）。
    // 如果 in[0]==out[0]（原地 RESHAPE，cccc 常见），则不需要任何操作。
    // ----------------------------------------------------------------
    if (type == MatrixOpType::RESHAPE)
    {
        // 原地（data ptr 相同）→ 跳过，下游 op 可能有形状冲突，届时回退 plain
        // 非原地 → 不支持
        if (in.size() < 1 || out.size() < 1)
        {
            return false;
        }
        if (in[0].get() == out[0].get())
        {
            return true;    // 原地，跳过
        }
        LOG("[CudnnOpQueueGraph] non-inplace RESHAPE → plain fallback\n");
        return false;
    }

    // 其它所有算子类型 → 不支持
    LOG("[CudnnOpQueueGraph] op type {} not supported in whole-network graph → plain fallback\n",
        int(type));
    return false;
}

bool CudnnOpQueueGraph::build(cudnnHandle_t handle, GpuControl* gpu, std::vector<MatrixOp>& ops)
{
    supported_ = false;
    plan_.reset();
    descs_.clear();
    matrix_entry_.clear();
    vp_uids_.clear();
    vp_ptrs_.clear();
    next_uid_ = 1;

    if (!handle || !gpu) { return false; }

    std::vector<cudnnBackendDescriptor_t> op_handles;
    int n_built = 0;
    for (auto& op : ops)
    {
        if (!buildOp(op, op_handles))
        {
            LOG("[CudnnOpQueueGraph] build aborted at op {} (type={}), total supported before={}, "
                "falling back to plain\n",
                n_built, int(op.getType()), n_built);
            return false;
        }
        ++n_built;
    }

    if (op_handles.empty())
    {
        LOG("[CudnnOpQueueGraph] op_queue empty or all ops produced no graph nodes\n");
        return false;
    }

    plan_ = std::make_unique<CudnnGraphPlan>(handle, op_handles);
    if (!plan_->ok())
    {
        LOG("[CudnnOpQueueGraph] CudnnGraphPlan build failed (total nodes={})\n", op_handles.size());
        plan_.reset();
        return false;
    }

    // 工作空间
    int64_t ws = plan_->workspaceSize();
    if (ws > 0)
    {
        workspace_.shared_data_->setGpu(gpu);
        workspace_.resize(1, 1, 1, int(ws / workspace_.getDataTypeSize() + 2));
    }

    LOG("[CudnnOpQueueGraph] whole-network graph built: ops={}, nodes={}, tensors={}, workspace={} bytes\n",
        ops.size(), op_handles.size(), vp_uids_.size(), ws);
    supported_ = true;
    return true;
}

int CudnnOpQueueGraph::execute()
{
    if (!supported_ || !plan_) { return -1; }
    void* ws_ptr = workspace_.getDataSizeInByte() ? workspace_.getDataPtr() : nullptr;
    return plan_->execute(vp_uids_, vp_ptrs_, ws_ptr);
}

}    // namespace cccc

#endif    // ENABLE_CUDA
