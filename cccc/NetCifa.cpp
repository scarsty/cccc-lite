#include "NetCifa.h"
#include "Log.h"
#include "MatrixData.h"
#include "MatrixEx.h"
#include "Option.h"
#include "strfunc.h"

namespace cccc
{

NetCifa::NetCifa()
{
}

NetCifa::~NetCifa()
{
}

std::vector<cifa::Object> NetCifa::getVector(cifa::ObjectVector& v, int index)
{
    std::vector<cifa::Object> r;
    if (index < 0 || index >= v.size() || !v[index].hasValue())
    {
        return r;
    }
    std::function<void(const cifa::Object&)> expand = [&r, &expand](const cifa::Object& o)
    {
        if (o.isType<std::vector<cifa::Object>>())
        {
            for (auto& o1 : o.ref<std::vector<cifa::Object>>())
            {
                expand(o1);
            }
        }
        else
        {
            r.push_back(o);
        }
    };
    expand(v[index]);
    return r;
}

std::vector<int> NetCifa::getIntVector(cifa::ObjectVector& v, int index)
{
    auto vo = getVector(v, index);
    std::vector<int> r;
    for (auto& o : vo)
    {
        r.push_back(o.toInt());
    }
    return r;
}

std::vector<float> NetCifa::getRealVector(cifa::ObjectVector& v, int index)
{
    auto vo = getVector(v, index);
    std::vector<float> r;
    for (auto& o : vo)
    {
        r.push_back(o.toDouble());
    }
    return r;
}

int NetCifa::init2()
{
    setDeviceSelf();
    registerFunctions();

    //cudnnTensorDescriptor_t t;
    //auto s = sizeof(*t);
    loss_.clear();
    //map_loss_.clear();

    // structure 优先级：外部注入 (structure_script_) > [net] structure > [cifa] structure（向后兼容）
    std::string struct_str = structure_script_;
    if (struct_str.empty())
    {
        struct_str = option_->getString("net", "structure");
    }

    // 参数脚本只写入 [net] 键值，再通过 register_parameter 暴露给 Cifa。
    // 不拼接脚本字符串，避免行号偏移。
    std::string parameter_str = option_->getString("net", "parameter");
    if (!parameter_str.empty())
    {
        auto pairs = strfunc::splitString(parameter_str, ";", true, true);
        for (auto& pair : pairs)
        {
            auto kv = strfunc::splitString(pair, "=", true, true);
            if (kv.size() < 2)
            {
                continue;
            }
            auto key = strfunc::toLowerCase(strfunc::trim(kv[0]));
            auto value = strfunc::trim(kv[1]);
            if (key.empty() || value.empty())
            {
                continue;
            }
            option_->setKey("net", key, value);
        }
    }

    if (runScript(struct_str))
    {
        LOG("Error in script!\n");
        return -1;
    }
    //map_matrix_.clear();
#ifdef _DEBUG
    //LOG("{}\n", MatrixOp::ir(op_queue_));
    //MatrixOp::ir(loss_);
#endif
    if (X_ && A_)
    {
        Y_->resize(A_->getDim());
    }
    else
    {
        LOG("Please set X, Y!\n");
        return -4;
    }
    return 0;
}

int NetCifa::runScript(const std::string& script)
{
    // Register section maps for non-net sections only.
    // [net] parameters are registered as top-level variables below.
    for (auto section : option_->getAllSections())
    {
        if (strfunc::toLowerCase(section) == "net")
        {
            continue;
        }
        std::map<std::string, cifa::Object> m;
        for (auto key : option_->getAllKeys(section))
        {
            m[strfunc::toLowerCase(key)] = cifa::Object(option_->INIReaderNoUnderline::getReal(section, key), option_->INIReaderNoUnderline::getString(section, key));
        }
        cifa_.register_parameter(strfunc::toLowerCase(section), m);
    }

    auto lines = strfunc::splitString(strfunc::replaceAllSubString(script, "\r", ""), "\n", false);
    int i = 1;
    // Avoid dumping the whole CIFA script by default.
    // Use [train] output_cifa_script=1 to enable it when debugging parser issues.
    if (option_->getInt("train", "output_cifa_script", 0))
    {
        for (auto& l : lines)
        {
            LOG("{:3}\t{}\n", i++, l);
        }
        LOG("\n");
    }

    // net.* 参数以顶层变量方式注册，脚本可直接访问 T/SEQ 等。
    for (auto& key : option_->getAllKeys("net"))
    {
        const auto key_trim = strfunc::trim(key);
        if (key_trim.empty())
        {
            continue;
        }
        const auto key_lower = strfunc::toLowerCase(key_trim);
        const auto key_upper = strfunc::toUpperCase(key_trim);
        const auto raw = option_->INIReaderNoUnderline::getString("net", key_trim);
        const auto num = option_->INIReaderNoUnderline::getReal("net", key_trim);

        const cifa::Object value(num, raw);
        cifa_.register_parameter(key_trim, value);
        cifa_.register_parameter(key_lower, value);
        cifa_.register_parameter(key_upper, value);
    }

    cifa_.set_output_error(true);
    auto o = cifa_.run_script(script);
    if (o.getSpecialType() == "Error" && o.isType<std::string>() && o.toString().find("Error") == 0)
    {
        LOG("{}", o.toString());
        return -1;
    }
    return 0;
}

int NetCifa::registerFunctions()
{
    cifa_.register_user_data("__this", this);
    registerFunction("Matrix", [this](cifa::ObjectVector& v)
        {
            std::vector<int> dim;
            for (int i = 0; i < v.size(); i++)
            {
                dim.push_back(v[i]);
            }
            return cifa::Object(makeMatrixSPWithState(dim));
        });
    //MatrixF: 始终使用 float 类型创建矩阵（即使全局 data_type=half），用于需要保持 float 精度的矩阵（如 token_ids）
    registerFunction("MatrixF", [this](cifa::ObjectVector& v)
        {
            std::vector<int> dim;
            for (int i = 0; i < v.size(); i++)
            {
                dim.push_back(v[i]);
            }
            auto m = std::make_shared<Matrix>(dim, DataType::FLOAT, UnitType::GPU, false);
            m->setNeedBack(need_back_state_);
            m->setNeedLoad(false);
            return cifa::Object(MatrixSP(m));
        });
    registerFunction("M", [this](cifa::ObjectVector& v)
        {
            std::vector<int> dim;
            for (int i = 0; i < v.size(); i++)
            {
                dim.push_back(v[i]);
            }
            return cifa::Object(makeMatrixSPWithState(dim));
        });
    //MatrixWithName(name, dim...): 创建矩阵、设置权重名、加入权重列表，三合一简写
    //等价于: W = Matrix(dim...); setWeightName(W, name);
    //名称可含 to_string(i) 等动态拼接，常用于循环中简化层权重定义
    //若 extra_matrixsp_ 中已有同名矩阵（如多组网络共享权重时预先注入），则直接返回已有矩阵
    registerFunction("MatrixWithName", [this](cifa::ObjectVector& v)
        {
            std::string name = v[0].toString();
            // 共享矩阵：若已在 extra_matrixsp_ 中预加载（来自另一组网络），直接复用
            if (extra_matrixsp_.count(name) > 0)
            {
                return cifa::Object(extra_matrixsp_[name]);
            }
            std::vector<int> dim;
            for (int i = 1; i < (int)v.size(); i++)
            {
                dim.push_back(v[i].toInt());
            }
            auto m = makeMatrixSPWithState(dim);
            m->setWeightName(name);
            // 自动注册到 extra_matrixsp_，使后续组网络可通过 preloadMatrices 共享此矩阵
            extra_matrixsp_[name] = m;
            return cifa::Object(m);
        });
    //registerFunction("Mb", [this](cifa::ObjectVector& v)
    //    {
    //        std::vector<int> dim = getIntVector(v, 0);
    //        auto o = cifa::Object(makeMatrixSPWithState(dim));
    //        o.to<MatrixSP>()->setNeedBack(v[1].toInt());
    //        o.to<MatrixSP>()->setNeedLoad(need_load_state_);
    //        return o;
    //    });
    registerFunction("setValue", [this](cifa::ObjectVector& v)
        {
            auto m = v[0].to<MatrixSP>();
            auto values = getVector(v, 1);
            std::vector<float> values_real(values.size());
            for (int i = 0; i < values.size(); i++)
            {
                values_real[i] = values[i].toDouble();
            }
            m->importData(values_real.data(), values_real.size());
            return cifa::Object();
        });
    registerFunction("repeat", [this](cifa::ObjectVector& v)
        {
            auto m = v[0].to<MatrixSP>();
            m->repeat(1);
            return cifa::Object();
        });
    registerFunction("scale", [this](cifa::ObjectVector& v)
        {
            MatrixOp op;
            op.solver_type_ = current_solver_type_;
            auto o = cifa::Object(makeMatrixSPWithState());
            op.as_scale(v[0].to<MatrixSP>(), o.to<MatrixSP>(), float(v[1].toDouble()));
            op_queue_.push_back(op);
            return o;
        });
    registerFunction("print_message", [this](cifa::ObjectVector& v)
        {
            MatrixOp op;
            op.solver_type_ = current_solver_type_;
            auto dummy = cifa::Object(makeMatrixSPWithState());
            std::string label = (v.size() > 1) ? v[1].toString() : "";
            op.as_print_message(v[0].to<MatrixSP>(), dummy.to<MatrixSP>(), label);
            op_queue_.push_back(op);
            return cifa::Object();
        });
    // saveBinary(X, filename): 将矩阵保存为 float32 binary 文件，仅首次 forward 保存，纯诊断
    registerFunction("saveBinary", [this](cifa::ObjectVector& v)
        {
            MatrixOp op;
            op.solver_type_ = current_solver_type_;
            auto dummy = cifa::Object(makeMatrixSPWithState());
            std::string filename = (v.size() > 1) ? v[1].toString() : "";
            op.as_save_binary(v[0].to<MatrixSP>(), dummy.to<MatrixSP>(), filename);
            op_queue_.push_back(op);
            return cifa::Object();
        });
    // print_ratio(A, B [, label]): 每隔若干 forward 打印 RMS(A)/RMS(B)，用于诊断注意力贡献比例
    registerFunction("print_ratio", [this](cifa::ObjectVector& v)
        {
            MatrixOp op;
            op.solver_type_ = current_solver_type_;
            auto dummy = cifa::Object(makeMatrixSPWithState());
            std::string label = (v.size() > 2) ? v[2].toString() : "attn_ratio";
            op.as_print_ratio(v[0].to<MatrixSP>(), v[1].to<MatrixSP>(), dummy.to<MatrixSP>(), label);
            op_queue_.push_back(op);
            return cifa::Object();
        });
    registerFunction("setXY", [this](cifa::ObjectVector& v)
        {
            //为方便理解，注意名字的区别
            auto xsp = v.size() > 0 ? v[0].to<MatrixSP>() : nullptr;
            auto asp = v.size() > 1 ? v[1].to<MatrixSP>() : nullptr;
            setXA(xsp, asp);
            return cifa::Object();
        });
    registerFunction("setXA", [this](cifa::ObjectVector& v)
        {
            setXA(v[0].to<MatrixSP>(), v[1].to<MatrixSP>());
            return cifa::Object();
        });
    registerFunction("setLossWeight", [this](cifa::ObjectVector& v)
        {
            extra_matrixsp_["loss_weight"] = v[0].to<MatrixSP>();
            return cifa::Object();
        });
    registerFunction("clearWeight", [this](cifa::ObjectVector& v)
        {
            weights_.clear();
            return cifa::Object();
        });
    //setWeightName(W, "name"): 为权重矩阵设置命名，配合 named_weights=1 时按名称加载
    registerFunction("setWeightName", [](cifa::ObjectVector& v)
        {
            if (v.size() >= 2)
            {
                v[0].to<MatrixSP>()->setWeightName(v[1].toString());
            }
            return cifa::Object();
        });
    registerFunction("addWeight", [this](cifa::ObjectVector& v)
        {
            for (auto& o : v)
            {
                weights_.push_back(o.to<MatrixSP>().get());
            }
            return cifa::Object();
        });
    registerFunction("addBias", [this](cifa::ObjectVector& v)
        {
            MatrixOp op;
            op.solver_type_ = current_solver_type_;
            auto o = cifa::Object(makeMatrixSPWithState());
            op.as_addBias(v[0].to<MatrixSP>(), v[1].to<MatrixSP>(), o.to<MatrixSP>());
            op_queue_.push_back(op);
            return o;
        });
    //layerNorm(X, scale, bias [, epsilon])
    //X: 输入; scale/bias: 形状 [width_]; 沿 width_ (inner) 维度归一化
    registerFunction("layerNorm", [this](cifa::ObjectVector& v)
        {
            MatrixOp op;
            op.solver_type_ = current_solver_type_;
            auto o = cifa::Object(makeMatrixSPWithState());
            float epsilon = 1e-5f;
            if (v.size() >= 4) { epsilon = float(v[3].toDouble()); }
            auto scale = v[1].to<MatrixSP>();
            auto bias = v[2].to<MatrixSP>();
            //LayerNorm scale/bias 在此初始化为 1/0, 并 addWeight 跳过 initWeights 的随机覆盖
            scale->fillData(1);
            bias->fillData(0);
            weights_.push_back(scale.get());
            weights_.push_back(bias.get());
            op.as_layerNorm(v[0].to<MatrixSP>(), scale, bias, o.to<MatrixSP>(), epsilon);
            op_queue_.push_back(op);
            return o;
        });
    //rmsNorm(X, scale [, epsilon]): RMS Normalization (LLM 标配, 无均值, 无 bias)
    //scale 形状 [width_]
    registerFunction("rmsNorm", [this](cifa::ObjectVector& v)
        {
            MatrixOp op;
            op.solver_type_ = current_solver_type_;
            auto o = cifa::Object(makeMatrixSPWithState());
            float epsilon = 1e-6f;
            if (v.size() >= 3) { epsilon = float(v[2].toDouble()); }
            auto scale = v[1].to<MatrixSP>();
            scale->fillData(1);
            weights_.push_back(scale.get());
            op.as_rmsNorm(v[0].to<MatrixSP>(), scale, o.to<MatrixSP>(), epsilon);
            op_queue_.push_back(op);
            return o;
        });
    //permute(X, {p0, p1, p2, p3}): 4 维任意轴置换
    //输出维度 dim_out[i] = dim_in[perm[i]]; 例如 perm={2,1,0,3} 是把宽高与通道交换
    registerFunction("permute", [this](cifa::ObjectVector& v)
        {
            auto perm = getIntVector(v, 1);
            MatrixOp op;
            auto o = cifa::Object(makeMatrixSPWithState());
            op.as_permute(v[0].to<MatrixSP>(), o.to<MatrixSP>(), perm);
            op_queue_.push_back(op);
            return o;
        });
    //rope(X, cos_tab, sin_tab): RoPE 旋转位置编码 (half-rotate / Llama 风格, ncnn mode=0)
    //X 形状 (D, T, 1, B), cos_tab/sin_tab 形状 (D/2, T, 1, 1); D 必须为偶数
    registerFunction("rope", [this](cifa::ObjectVector& v)
        {
            MatrixOp op;
            auto o = cifa::Object(makeMatrixSPWithState());
            op.as_rope(v[0].to<MatrixSP>(), v[1].to<MatrixSP>(), v[2].to<MatrixSP>(), o.to<MatrixSP>());
            op_queue_.push_back(op);
            return o;
        });
    //rope2(X, cos_tab, sin_tab): RoPE interleaved 风格 (ncnn mode=1)
    //y[2i]=x[2i]*cos[i]-x[2i+1]*sin[i], y[2i+1]=x[2i+1]*cos[i]+x[2i]*sin[i]
    registerFunction("rope2", [this](cifa::ObjectVector& v)
        {
            MatrixOp op;
            auto o = cifa::Object(makeMatrixSPWithState());
            op.as_rope_interleaved(v[0].to<MatrixSP>(), v[1].to<MatrixSP>(), v[2].to<MatrixSP>(), o.to<MatrixSP>());
            op_queue_.push_back(op);
            return o;
        });
    //ropeCosTbl(T, d_head [, base=10000]): 生成 RoPE cos 位置编码表
    //输出形状 (d_head/2, T, 1, 1); 常量矩阵 (need_back=false), 在 CPU 计算后上传 GPU
    //base 为旋转频率基数 (默认 10000, Llama/Qwen 使用 500000)
    registerFunction("ropeCosTbl", [this](cifa::ObjectVector& v)
        {
            const bool old_lazy_mode = MatrixData::isLazyMode();
            MatrixData::setLazyMode(false);
            int T = v[0].toInt();
            int d_head = v[1].toInt();
            float base = (v.size() >= 3) ? (float)v[2].toDouble() : 10000.0f;
            int half_D = d_head / 2;
            auto m = std::make_shared<Matrix>(std::vector<int>{ half_D, T, 1, 1 }, DataType::CURRENT, UnitType::CPU, false);
            for (int t = 0; t < T; t++)
            {
                for (int d = 0; d < half_D; d++)
                {
                    m->setData(d, t, 0, 0, (float)std::cos(t * std::pow(base, -2.0f * d / d_head)));
                }
            }
            m->setNeedBack(false);
            m->setNeedLoad(false);
            m->toGPU();
            MatrixData::setLazyMode(old_lazy_mode);
            return cifa::Object(MatrixSP(m));
        });
    //输出形状 (d_head/2, T, 1, 1); 用法同 ropeCosTbl
    registerFunction("ropeSinTbl", [this](cifa::ObjectVector& v)
        {
            const bool old_lazy_mode = MatrixData::isLazyMode();
            MatrixData::setLazyMode(false);
            int T = v[0].toInt();
            int d_head = v[1].toInt();
            float base = (v.size() >= 3) ? (float)v[2].toDouble() : 10000.0f;
            int half_D = d_head / 2;
            auto m = std::make_shared<Matrix>(std::vector<int>{ half_D, T, 1, 1 }, DataType::CURRENT, UnitType::CPU, false);
            for (int t = 0; t < T; t++)
            {
                for (int d = 0; d < half_D; d++)
                {
                    m->setData(d, t, 0, 0, (float)std::sin(t * std::pow(base, -2.0f * d / d_head)));
                }
            }
            m->setNeedBack(false);
            m->setNeedLoad(false);
            m->toGPU();
            MatrixData::setLazyMode(old_lazy_mode);
            return cifa::Object(MatrixSP(m));
        });
    //ropeCosTblYaRN(T, d_head, base, scale, original_max_len [, beta_fast=32, beta_slow=1]):
    //  生成 YaRN（NTK-by-parts）RoPE cos 表，输出形状 (d_head/2, T, 1, 1)
    //  scale = new_max_len / original_max_len; 对各维度按波长分三段缩放:
    //    wavelen > original_max/beta_slow → 低频，线性插值 inv_freq /= scale
    //    wavelen < original_max/beta_fast → 高频，不缩放
    //    中频 → 平滑混合
    registerFunction("ropeCosTblYaRN", [this](cifa::ObjectVector& v)
        {
            const bool old_lazy_mode = MatrixData::isLazyMode();
            MatrixData::setLazyMode(false);
            int T = v[0].toInt();
            int d_head = v[1].toInt();
            float base = (float)v[2].toDouble();
            float scale = (float)v[3].toDouble();
            float orig_max = (float)v[4].toDouble();
            float beta_fast = (v.size() >= 6) ? (float)v[5].toDouble() : 32.0f;
            float beta_slow = (v.size() >= 7) ? (float)v[6].toDouble() : 1.0f;
            int half_D = d_head / 2;
            auto m = std::make_shared<Matrix>(std::vector<int>{ half_D, T, 1, 1 }, DataType::CURRENT, UnitType::CPU, false);
            const float pi2 = 2.0f * 3.14159265358979323846f;
            for (int d = 0; d < half_D; d++)
            {
                float inv_freq = std::pow(base, -2.0f * d / d_head);
                float wavelen = pi2 / inv_freq;
                float low_wavelen = orig_max / beta_slow;
                float high_wavelen = orig_max / beta_fast;
                float eff_freq;
                if (wavelen > low_wavelen)
                    eff_freq = inv_freq / scale;
                else if (wavelen < high_wavelen)
                    eff_freq = inv_freq;
                else
                {
                    float smooth = (orig_max / wavelen - beta_slow) / (beta_fast - beta_slow);
                    eff_freq = (1.0f - smooth) * (inv_freq / scale) + smooth * inv_freq;
                }
                for (int t = 0; t < T; t++)
                    m->setData(d, t, 0, 0, std::cos((float)t * eff_freq));
            }
            m->setNeedBack(false);
            m->setNeedLoad(false);
            m->toGPU();
            MatrixData::setLazyMode(old_lazy_mode);
            return cifa::Object(MatrixSP(m));
        });
    //ropeSinTblYaRN(T, d_head, base, scale, original_max_len [, beta_fast=32, beta_slow=1]):
    //  同 ropeCosTblYaRN，输出 sin 表
    registerFunction("ropeSinTblYaRN", [this](cifa::ObjectVector& v)
        {
            const bool old_lazy_mode = MatrixData::isLazyMode();
            MatrixData::setLazyMode(false);
            int T = v[0].toInt();
            int d_head = v[1].toInt();
            float base = (float)v[2].toDouble();
            float scale = (float)v[3].toDouble();
            float orig_max = (float)v[4].toDouble();
            float beta_fast = (v.size() >= 6) ? (float)v[5].toDouble() : 32.0f;
            float beta_slow = (v.size() >= 7) ? (float)v[6].toDouble() : 1.0f;
            int half_D = d_head / 2;
            auto m = std::make_shared<Matrix>(std::vector<int>{ half_D, T, 1, 1 }, DataType::CURRENT, UnitType::CPU, false);
            const float pi2 = 2.0f * 3.14159265358979323846f;
            for (int d = 0; d < half_D; d++)
            {
                float inv_freq = std::pow(base, -2.0f * d / d_head);
                float wavelen = pi2 / inv_freq;
                float low_wavelen = orig_max / beta_slow;
                float high_wavelen = orig_max / beta_fast;
                float eff_freq;
                if (wavelen > low_wavelen)
                    eff_freq = inv_freq / scale;
                else if (wavelen < high_wavelen)
                    eff_freq = inv_freq;
                else
                {
                    float smooth = (orig_max / wavelen - beta_slow) / (beta_fast - beta_slow);
                    eff_freq = (1.0f - smooth) * (inv_freq / scale) + smooth * inv_freq;
                }
                for (int t = 0; t < T; t++)
                    m->setData(d, t, 0, 0, std::sin((float)t * eff_freq));
            }
            m->setNeedBack(false);
            m->setNeedLoad(false);
            m->toGPU();
            MatrixData::setLazyMode(old_lazy_mode);
            return cifa::Object(MatrixSP(m));
        });
    //rope2dCosTbl(W, H, d_head [, base=10000]): 生成 2D RoPE cos 位置编码表（图像专用）
    //将 T=W*H 个空间位置的二维坐标 (x=t%W, y=t/W) 分别编码到通道的两半：
    //  d ∈ [0, d_head/4)        → cos(x * base^(-2d / (d_head/2)))   列方向
    //  d ∈ [d_head/4, d_head/2) → cos(y * base^(-2(d-d_head/4) / (d_head/2)))  行方向
    //输出形状 (d_head/2, T, 1, 1); 与 rope() 内核兼容; 常量矩阵 (need_back=false)
    registerFunction("rope2dCosTbl", [this](cifa::ObjectVector& v)
        {
            const bool old_lazy_mode = MatrixData::isLazyMode();
            MatrixData::setLazyMode(false);
            int W = v[0].toInt();
            int H = v[1].toInt();
            int d_head = v[2].toInt();
            float base = (v.size() >= 4) ? (float)v[3].toDouble() : 10000.0f;
            int T = W * H;
            int half_D = d_head / 2;
            int quarter_D = d_head / 4;
            auto m = std::make_shared<Matrix>(std::vector<int>{ half_D, T, 1, 1 }, DataType::CURRENT, UnitType::CPU, false);
            for (int t = 0; t < T; t++)
            {
                int x = t % W;
                int y = t / W;
                for (int d = 0; d < quarter_D; d++)
                {
                    m->setData(d, t, 0, 0, (float)std::cos(x * std::pow(base, -2.0f * d / half_D)));
                }
                for (int d = quarter_D; d < half_D; d++)
                {
                    m->setData(d, t, 0, 0, (float)std::cos(y * std::pow(base, -2.0f * (d - quarter_D) / half_D)));
                }
            }
            m->setNeedBack(false);
            m->setNeedLoad(false);
            m->toGPU();
            MatrixData::setLazyMode(old_lazy_mode);
            return cifa::Object(MatrixSP(m));
        });
    //rope2dSinTbl(W, H, d_head [, base=10000]): 生成 2D RoPE sin 位置编码表（图像专用）
    //通道分配同 rope2dCosTbl, 输出形状 (d_head/2, T, 1, 1); 常量矩阵 (need_back=false)
    registerFunction("rope2dSinTbl", [this](cifa::ObjectVector& v)
        {
            const bool old_lazy_mode = MatrixData::isLazyMode();
            MatrixData::setLazyMode(false);
            int W = v[0].toInt();
            int H = v[1].toInt();
            int d_head = v[2].toInt();
            float base = (v.size() >= 4) ? (float)v[3].toDouble() : 10000.0f;
            int T = W * H;
            int half_D = d_head / 2;
            int quarter_D = d_head / 4;
            auto m = std::make_shared<Matrix>(std::vector<int>{ half_D, T, 1, 1 }, DataType::CURRENT, UnitType::CPU, false);
            for (int t = 0; t < T; t++)
            {
                int x = t % W;
                int y = t / W;
                for (int d = 0; d < quarter_D; d++)
                {
                    m->setData(d, t, 0, 0, (float)std::sin(x * std::pow(base, -2.0f * d / half_D)));
                }
                for (int d = quarter_D; d < half_D; d++)
                {
                    m->setData(d, t, 0, 0, (float)std::sin(y * std::pow(base, -2.0f * (d - quarter_D) / half_D)));
                }
            }
            m->setNeedBack(false);
            m->setNeedLoad(false);
            m->toGPU();
            MatrixData::setLazyMode(old_lazy_mode);
            return cifa::Object(MatrixSP(m));
        });
    //kvcache(X_new, cache): 推理用 KV cache 写入算子
    //X_new 形状 (D, T_new, H, B), cache 形状 (D, T_max, H, B); cache 由用户预先 Matrix(...) 创建并视为持久缓冲
    //每次 forward 把 X_new 写入 cache 当前 pos 偏移处, 输出 Y 与 cache 共享数据 (形状=cache.dim).
    //开始新一段推理前调用 Net::resetKVCache() 重置写入位置.
    registerFunction("kvcache", [this](cifa::ObjectVector& v)
        {
            MatrixOp op;
            auto o = cifa::Object(makeMatrixSPWithState());
            op.as_kvcache(v[0].to<MatrixSP>(), v[1].to<MatrixSP>(), o.to<MatrixSP>());
            op_queue_.push_back(op);
            return o;
        });
    //embed(ids, W): 词表嵌入查表
    //ids 形状 (T,1,1,B) 整数 id 以 float 存储, W 形状 (D,1,1,V) 为词表矩阵
    //输出 Y 形状 (D,T,1,B); W 作为可训练权重
    registerFunction("embed", [this](cifa::ObjectVector& v)
        {
            auto ids = v[0].to<MatrixSP>();
            auto W = v[1].to<MatrixSP>();
            // 默认以 Xavier 初始化 embed 权重（加载权重文件时会被覆盖）--todo 不应在这里填充
            //MatrixEx::fill(*W, RANDOM_FILL_XAVIER, W->getRow(), W->getNumber());
            weights_.push_back(W.get());
            MatrixOp op;
            op.solver_type_ = current_solver_type_;
            auto o = cifa::Object(makeMatrixSPWithState());
            op.as_embed(ids, W, o.to<MatrixSP>());
            op_queue_.push_back(op);
            return o;
        });
    //tile(X, {r0, r1, r2, r3}): 沿各轴重复
    //输出 Y[w,h,c,n] = X[w%W, h%H, c%C, n%N]
    //用于 GQA KV-head 扩展: tile(K, {1,1,2,1}) 把 Hkv=8 扩展到 H=16
    registerFunction("tile", [this](cifa::ObjectVector& v)
        {
            auto repeats = getIntVector(v, 1);
            while (repeats.size() < 4)
            {
                repeats.push_back(1);
            }
            MatrixOp op;
            op.solver_type_ = current_solver_type_;
            auto o = cifa::Object(makeMatrixSPWithState());
            op.as_tile(v[0].to<MatrixSP>(), o.to<MatrixSP>(), repeats);
            op_queue_.push_back(op);
            return o;
        });
    //pixelShuffle(X, r): 像素重组上采样, X (W,H,C*r*r,N) -> Y (W*r, H*r, C, N)
    //r 为上采样倍率; 常配合 1x1 conv 使用实现无块状伪影的上采样
    registerFunction("pixelShuffle", [this](cifa::ObjectVector& v)
        {
            int r = (int)v[1].to<double>();
            MatrixOp op;
            op.solver_type_ = current_solver_type_;
            auto o = cifa::Object(makeMatrixSPWithState());
            op.as_pixelShuffle(v[0].to<MatrixSP>(), o.to<MatrixSP>(), r);
            op_queue_.push_back(op);
            return o;
        });
    //deconv(A, W [, {stride}, {padding}]): 转置卷积 (反卷积)
    //A: 输入 (W_in,H_in,C_in,N), W: 滤波器 (kW,kH,C_out,C_in)
    //C_in 对应 W.getNumber(), C_out 对应 W.getChannel()
    //输出 Y: (W_out,H_out,C_out,N), 其中 W_out=(W_in-1)*stride+kW-2*pad
    //W 作为可训练权重, 以 Xavier 初始化
    registerFunction("deconv", [this](cifa::ObjectVector& v)
        {
            auto stride = getIntVector(v, 2);
            auto padding = getIntVector(v, 3);
            MatrixOp op;
            op.solver_type_ = current_solver_type_;
            auto W = v[1].to<MatrixSP>();
            MatrixEx::fill(*W, RANDOM_FILL_XAVIER, W->getRow(), W->getNumber());
            weights_.push_back(W.get());
            auto o = cifa::Object(makeMatrixSPWithState());
            auto conv_algo = option_->getInt("train", "conv_algo", -1);
            op.as_deconv(v[0].to<MatrixSP>(), W, o.to<MatrixSP>(), stride, padding, conv_algo);
            op_queue_.push_back(op);
            return o;
        });
    //groupNorm(X, G, scale, bias [, epsilon]): Group Normalization
    //X: 输入; G: 组数 (C 必须整除 G); scale/bias: 形状 [C], 可训练参数
    //每组归一化 W*H*C/G 个元素; 用于图像生成 (Diffusion/GAN) 等任务
    registerFunction("groupNorm", [this](cifa::ObjectVector& v)
        {
            auto X = v[0].to<MatrixSP>();
            int G = (int)v[1].toDouble();
            auto scale = v[2].to<MatrixSP>();
            auto bias = v[3].to<MatrixSP>();
            float epsilon = 1e-5f;
            if (v.size() >= 5) { epsilon = float(v[4].toDouble()); }
            weights_.push_back(scale.get());
            weights_.push_back(bias.get());
            MatrixOp op;
            op.solver_type_ = current_solver_type_;
            auto o = cifa::Object(makeMatrixSPWithState());
            op.as_groupNorm(X, scale, bias, o.to<MatrixSP>(), G, epsilon);
            op_queue_.push_back(op);
            return o;
        });
    //reparam(mu, log_var): VAE 重参数化算子
    //mu, log_var: 形状 (D,1,1,B); 输出 z: (D,1,1,B)
    //训练时: z = mu + exp(log_var*0.5) * eps, eps~N(0,1) (eps 存于 workspace 供反向使用)
    //推理时: z = mu (确定性输出)
    registerFunction("reparam", [this](cifa::ObjectVector& v)
        {
            MatrixOp op;
            op.solver_type_ = current_solver_type_;
            auto o = cifa::Object(makeMatrixSPWithState());
            op.as_reparam(v[0].to<MatrixSP>(), v[1].to<MatrixSP>(), o.to<MatrixSP>());
            op_queue_.push_back(op);
            return o;
        });
    //softmaxChannel(X): 沿 X.width_ 维度做 softmax (用于 attention 分数归一化)
    registerFunction("softmaxChannel", [this](cifa::ObjectVector& v)
        {
            MatrixOp op;
            op.solver_type_ = current_solver_type_;
            auto o = cifa::Object(makeMatrixSPWithState());
            op.as_active(v[0].to<MatrixSP>(), o.to<MatrixSP>(), ACTIVE_FUNCTION_SOFTMAX_CHANNEL,
                std::vector<int>{}, std::vector<float>{}, std::vector<Matrix>{});
            op_queue_.push_back(op);
            return o;
        });
    //attention(Q, K, V, [dk, causal]): scaled dot-product attention 单算子 (融合正向/反向).
    //输入 Q/K/V 形状均为 (D, T, 1, B): width_=D, height_=T, channel_=1, number_=B
    //计算: Y = softmax_channel(K^T @ Q / sqrt(dk)) @ V
    //causal=1 时应用因果掩码 (下三角注意力), 用于自回归语言模型
    registerFunction("attention", [this](cifa::ObjectVector& v)
        {
            auto Q = v[0].to<MatrixSP>();
            auto K = v[1].to<MatrixSP>();
            auto V = v[2].to<MatrixSP>();
            float dk = float(Q->getWidth());
            if (v.size() >= 4)
            {
                double dk_arg = v[3].toDouble();
                if (dk_arg > 0) { dk = float(dk_arg); }
            }
            int causal = 0;
            if (v.size() >= 5) { causal = int(v[4].toDouble()); }
            MatrixSP bias_ptr = nullptr;
            if (v.size() >= 6 && v[5].isType<MatrixSP>())
            {
                bias_ptr = v[5].to<MatrixSP>();
            }
            MatrixOp op;
            op.solver_type_ = current_solver_type_;
            auto out = cifa::Object(makeMatrixSPWithState());
            op.as_attention(Q, K, V, out.to<MatrixSP>(), dk, causal, bias_ptr);
            op_queue_.push_back(op);
            return out;
        });
    // roiAlign(feat, boxes, roi_size [, spatial_scale]):
    // feat: (W,H,C,B), boxes: (4,N,1,B) in pixel-space coords,
    // roi_size: integer K, spatial_scale: default 1.0
    // output: (K, K, C, N*B)
    registerFunction("roiAlign", [this](cifa::ObjectVector& v)
        {
            auto feat  = v[0].to<MatrixSP>();
            auto boxes = v[1].to<MatrixSP>();
            int roi_size = int(v[2].toDouble());
            float spatial_scale = 1.0f;
            if (v.size() >= 4) { spatial_scale = float(v[3].toDouble()); }
            MatrixOp op;
            op.solver_type_ = current_solver_type_;
            auto out = cifa::Object(makeMatrixSPWithState());
            op.as_roi_align(feat, boxes, out.to<MatrixSP>(), roi_size, spatial_scale);
            op_queue_.push_back(op);
            return out;
        });
    //attention0(Q, K, V, [dk]): scaled dot-product attention, 展开为4个子操作 (等价对照用).
    //与 attention() 数学上完全等价, 用于对比验证.
    registerFunction("attention0", [this](cifa::ObjectVector& v)
        {
            auto Q = v[0].to<MatrixSP>();
            auto K = v[1].to<MatrixSP>();
            auto V = v[2].to<MatrixSP>();
            float dk = float(Q->getWidth());
            if (v.size() >= 4)
            {
                double dk_arg = v[3].toDouble();
                if (dk_arg > 0) { dk = float(dk_arg); }
            }
            float scale_factor = 1.0f / std::sqrt(dk);
            // op1: scores = K^T @ Q -> (T, T, 1, B), M/N/K/batch 从维度自动推导
            MatrixOp op_scores;
            op_scores.solver_type_ = current_solver_type_;
            auto scores = cifa::Object(makeMatrixSPWithState());
            op_scores.as_batchedMul(K, Q, scores.to<MatrixSP>(), MATRIX_TRANS, MATRIX_NO_TRANS);
            op_queue_.push_back(op_scores);
            // op2: scaled = scores * (1/sqrt(dk))
            MatrixOp op_scale;
            op_scale.solver_type_ = current_solver_type_;
            auto scaled = cifa::Object(makeMatrixSPWithState());
            op_scale.as_scale(scores.to<MatrixSP>(), scaled.to<MatrixSP>(), scale_factor);
            op_queue_.push_back(op_scale);
            // op3: attn = softmaxChannel(scaled)
            MatrixOp op_sm;
            op_sm.solver_type_ = current_solver_type_;
            auto attn = cifa::Object(makeMatrixSPWithState());
            op_sm.as_active(scaled.to<MatrixSP>(), attn.to<MatrixSP>(), ACTIVE_FUNCTION_SOFTMAX_CHANNEL,
                std::vector<int>{}, std::vector<float>{}, std::vector<Matrix>{});
            op_queue_.push_back(op_sm);
            // op4: out = V @ attn -> (D, T, 1, B), M/N/K/batch 从维度自动推导
            MatrixOp op_out;
            op_out.solver_type_ = current_solver_type_;
            auto out = cifa::Object(makeMatrixSPWithState());
            op_out.as_batchedMul(V, attn.to<MatrixSP>(), out.to<MatrixSP>(), MATRIX_NO_TRANS, MATRIX_NO_TRANS);
            op_queue_.push_back(op_out);
            return out;
        });
    registerFunction("add", [this](cifa::ObjectVector& v)
        {
            MatrixOp op;
            op.solver_type_ = current_solver_type_;
            auto o = cifa::Object(makeMatrixSPWithState());
            if (v[0].isType<MatrixSP>())
            {
                op.as_add(v[0].to<MatrixSP>(), v[1].to<MatrixSP>(), o.to<MatrixSP>());
            }
            else
            {
                auto vo = getVector(v, 0);
                std::vector<MatrixSP> vm;
                for (auto& o1 : vo)
                {
                    vm.push_back(o1.to<MatrixSP>());
                }
                std::vector<float> va;
                if (v.size() >= 2)
                {
                    va = getRealVector(v, 1);
                }
                op.as_add(vm, o.to<MatrixSP>(), va);
            }
            op_queue_.push_back(op);
            return o;
        });
    cifa_.user_add.push_back([this](const cifa::Object& l, const cifa::Object& r)
        {
            if (l.isType<MatrixSP>() && r.isType<MatrixSP>())
            {
                MatrixOp op;
                op.solver_type_ = current_solver_type_;
                auto o = cifa::Object(makeMatrixSPWithState());
                if (r.to<MatrixSP>()->getNumber() == 1 && r.to<MatrixSP>()->getDataSize() != l.to<MatrixSP>()->getDataSize())    //注意这个判断不严格
                {
                    op.as_addBias(l.to<MatrixSP>(), r.to<MatrixSP>(), o.to<MatrixSP>());
                }
                else
                {
                    op.as_add(l.to<MatrixSP>(), r.to<MatrixSP>(), o.to<MatrixSP>());
                }
                op_queue_.push_back(op);
                return o;
            }
            if (l.isType<Loss>() && r.isType<Loss>())
            {
                return cifa::Object(l.to<Loss>() + r.to<Loss>());
            }
            return cifa::Object();
        });
    cifa_.user_mul.push_back([this](const cifa::Object& l, const cifa::Object& r)
        {
            if (l.isType<MatrixSP>() && r.isType<MatrixSP>())
            {
                MatrixOp op;
                op.solver_type_ = current_solver_type_;
                auto o = cifa::Object(makeMatrixSPWithState());
                op.as_mul(l.to<MatrixSP>(), r.to<MatrixSP>(), o.to<MatrixSP>());
                op_queue_.push_back(op);
                return o;
            }
            if (l.isType<MatrixSP>() && r.isNumber())
            {
                MatrixOp op;
                op.solver_type_ = current_solver_type_;
                auto o = cifa::Object(makeMatrixSPWithState());
                op.as_scale(l.to<MatrixSP>(), o.to<MatrixSP>(), float(r.toDouble()));
                op_queue_.push_back(op);
                return o;
            }
            if (r.isType<MatrixSP>() && l.isNumber())
            {
                MatrixOp op;
                op.solver_type_ = current_solver_type_;
                auto o = cifa::Object(makeMatrixSPWithState());
                op.as_scale(r.to<MatrixSP>(), o.to<MatrixSP>(), float(l.toDouble()));
                op_queue_.push_back(op);
                return o;
            }
            if (l.isType<Loss>() || r.isType<Loss>())
            {
                Loss q;
                if (l.isType<Loss>())
                {
                    q = r.toDouble() * l.to<Loss>();
                }
                else
                {
                    q = l.toDouble() * r.to<Loss>();
                }
                return cifa::Object(q);
            }
            return cifa::Object();
        });
    registerFunction("conv", [this](cifa::ObjectVector& v)
        {
            auto stride = getIntVector(v, 2);
            auto padding = getIntVector(v, 3);
            MatrixOp op;
            op.solver_type_ = current_solver_type_;
            auto o = cifa::Object(makeMatrixSPWithState());
            auto conv_algo = option_->getInt("train", "conv_algo", -1);
            op.as_conv(v[0].to<MatrixSP>(), v[1].to<MatrixSP>(), o.to<MatrixSP>(), stride, padding, conv_algo);
            op_queue_.push_back(op);
            return o;
        });
    registerFunction("corr", [this](cifa::ObjectVector& v)
        {
            auto stride = getIntVector(v, 2);
            auto padding = getIntVector(v, 3);
            MatrixOp op;
            op.solver_type_ = current_solver_type_;
            auto o = cifa::Object(makeMatrixSPWithState());
            op.as_corr(v[0].to<MatrixSP>(), v[1].to<MatrixSP>(), o.to<MatrixSP>(), stride, padding, 1);
            op_queue_.push_back(op);
            return o;
        });
    registerFunction("pool", [this](cifa::ObjectVector& v)
        {
            auto window = getIntVector(v, 2);
            auto stride = getIntVector(v, 3);
            auto padding = getIntVector(v, 4);
            MatrixOp op;
            op.solver_type_ = current_solver_type_;
            auto o = cifa::Object(makeMatrixSPWithState());
            op.as_pool(v[0].to<MatrixSP>(), o.to<MatrixSP>(), PoolingType(int(v[1])), POOLING_NOT_REVERSE, window, stride, padding);
            op_queue_.push_back(op);
            return o;
        });
    registerFunction("maxPool", [this](cifa::ObjectVector& v)
        {
            auto window = getIntVector(v, 1);
            auto stride = getIntVector(v, 2);
            auto padding = getIntVector(v, 3);
            MatrixOp op;
            op.solver_type_ = current_solver_type_;
            auto o = cifa::Object(makeMatrixSPWithState());
            op.as_pool(v[0].to<MatrixSP>(), o.to<MatrixSP>(), POOLING_MAX, POOLING_NOT_REVERSE, window, stride, padding);
            op_queue_.push_back(op);
            return o;
        });
    registerFunction("averagePool", [this](cifa::ObjectVector& v)
        {
            auto window = getIntVector(v, 1);
            auto stride = getIntVector(v, 2);
            auto padding = getIntVector(v, 3);
            MatrixOp op;
            op.solver_type_ = current_solver_type_;
            auto o = cifa::Object(makeMatrixSPWithState());
            op.as_pool(v[0].to<MatrixSP>(), o.to<MatrixSP>(), POOLING_AVERAGE_NOPADDING, POOLING_NOT_REVERSE, window, stride, padding);
            op_queue_.push_back(op);
            return o;
        });
    registerFunction("reversePool", [this](cifa::ObjectVector& v)
        {
            auto window = getIntVector(v, 1);
            auto stride = getIntVector(v, 2);
            auto padding = getIntVector(v, 3);
            MatrixOp op;
            op.solver_type_ = current_solver_type_;
            auto o = cifa::Object(makeMatrixSPWithState());
            op.as_pool(v[0].to<MatrixSP>(), o.to<MatrixSP>(), POOLING_AVERAGE_NOPADDING, POOLING_REVERSE, window, stride, padding);
            op_queue_.push_back(op);
            return o;
        });
    registerFunction("poolChannel", [this](cifa::ObjectVector& v)
        {
            MatrixOp op;
            op.solver_type_ = current_solver_type_;
            auto o = cifa::Object(makeMatrixSPWithState());
            op.as_poolChannel(v[0].to<MatrixSP>(), o.to<MatrixSP>(), PoolingType(int(v[1])), POOLING_NOT_REVERSE);
            op_queue_.push_back(op);
            return o;
        });
    registerFunction("width", [this](cifa::ObjectVector& v)
        {
            return cifa::Object(v[0].to<MatrixSP>()->getWidth());
        });
    registerFunction("height", [this](cifa::ObjectVector& v)
        {
            return cifa::Object(v[0].to<MatrixSP>()->getHeight());
        });
    registerFunction("channel", [this](cifa::ObjectVector& v)
        {
            return cifa::Object(v[0].to<MatrixSP>()->getChannel());
        });
    registerFunction("row", [this](cifa::ObjectVector& v)
        {
            return cifa::Object(v[0].to<MatrixSP>()->getRow());
        });
    registerFunction("number", [this](cifa::ObjectVector& v)
        {
            return cifa::Object(v[0].to<MatrixSP>()->getNumber());
        });
    registerFunction("reshape", [this](cifa::ObjectVector& v)
        {
            auto dim = getIntVector(v, 1);
            MatrixOp op;
            auto o = cifa::Object(makeMatrixSPWithState());
            op.as_reshape(v[0].to<MatrixSP>(), o.to<MatrixSP>(), dim);
            op_queue_.push_back(op);
            return o;
        });
    //reshape_batch(X, dim): 允许改 batch 维, 仅要求总元素数一致
    registerFunction("reshapeBatch", [this](cifa::ObjectVector& v)
        {
            auto dim = getIntVector(v, 1);
            MatrixOp op;
            auto o = cifa::Object(makeMatrixSPWithState());
            op.as_reshape_batch(v[0].to<MatrixSP>(), o.to<MatrixSP>(), dim);
            op_queue_.push_back(op);
            return o;
        });
    registerFunction("setNeedBack", [this](cifa::ObjectVector& v)
        {
            v[0].to<MatrixSP>()->setNeedBack(v[1].toInt());
            return v[1];
        });
    registerFunction("setAllNeedBack", [this](cifa::ObjectVector& v)
        {
            auto back = v[0].toInt();
            for (auto& op : op_queue_)
            {
                for (auto m : op.getMatrixIn())
                {
                    m->setNeedBack(back);
                }
                for (auto m : op.getMatrixOut())
                {
                    m->setNeedBack(back);
                }
            }
            return cifa::Object();
        });
    registerFunction("setNeedBackState", [this](cifa::ObjectVector& v)
        {
            need_back_state_ = v[0].toInt();
            return cifa::Object();
        });
    registerFunction("setNeedLoadState", [this](cifa::ObjectVector& v)
        {
            need_load_state_ = v[0].toInt();
            return cifa::Object();
        });
    registerFunction("setCurrentSolverType", [this](cifa::ObjectVector& v)
        {
            current_solver_type_ = option_->getEnumFromString<SolverType>(v[0].toString());
            return cifa::Object();
        });
    registerFunction("mul", [this](cifa::ObjectVector& v)
        {
            MatrixTransType ta = MATRIX_NO_TRANS, tb = MATRIX_NO_TRANS;
            if (v.size() >= 3 && int(v[2].toDouble()) != 0) { ta = MATRIX_TRANS; }
            if (v.size() >= 4 && int(v[3].toDouble()) != 0) { tb = MATRIX_TRANS; }
            MatrixOp op;
            auto o = cifa::Object(makeMatrixSPWithState());
            op.as_mul(v[0].to<MatrixSP>(), v[1].to<MatrixSP>(), o.to<MatrixSP>(), 1, {}, ta, tb);
            op_queue_.push_back(op);
            return o;
        });
    //batchedMul(A, B, [ta, tb])
    //M/N/K/batch 从矩阵维度自动推导; ta/tb 为 0 (NoTrans) 或 1 (Trans), 默认均为 0
    registerFunction("batchedMul", [this](cifa::ObjectVector& v)
        {
            MatrixTransType ta = MATRIX_NO_TRANS, tb = MATRIX_NO_TRANS;
            if (v.size() >= 3 && int(v[2].toDouble()) != 0) { ta = MATRIX_TRANS; }
            if (v.size() >= 4 && int(v[3].toDouble()) != 0) { tb = MATRIX_TRANS; }
            MatrixOp op;
            op.solver_type_ = current_solver_type_;
            auto o = cifa::Object(makeMatrixSPWithState());
            op.as_batchedMul(v[0].to<MatrixSP>(), v[1].to<MatrixSP>(), o.to<MatrixSP>(), ta, tb);
            op_queue_.push_back(op);
            return o;
        });
    registerFunction("elementMul", [this](cifa::ObjectVector& v)
        {
            MatrixOp op;
            op.solver_type_ = current_solver_type_;
            auto o = cifa::Object(makeMatrixSPWithState());
            double a = 1.0;
            if (v.size() >= 3)
            {
                a = v[2].toDouble();
            }
            op.as_elementMul(v[0].to<MatrixSP>(), v[1].to<MatrixSP>(), o.to<MatrixSP>(), a);
            op_queue_.push_back(op);
            return o;
        });
    registerFunction("active", [this](cifa::ObjectVector& v)
        {
            MatrixOp op;
            op.solver_type_ = current_solver_type_;
            auto o = cifa::Object(makeMatrixSPWithState());
            op.as_active(v[0].to<MatrixSP>(), o.to<MatrixSP>(), ActiveFunctionType(int(v[1])), getIntVector(v, 2), getRealVector(v, 3), {});
            op_queue_.push_back(op);
            return o;
        });
    registerFunction("tanh", [this](cifa::ObjectVector& v)
        {
            MatrixOp op;
            op.solver_type_ = current_solver_type_;
            auto o = cifa::Object(makeMatrixSPWithState());
            op.as_active(v[0].to<MatrixSP>(), o.to<MatrixSP>(), ACTIVE_FUNCTION_TANH);
            op_queue_.push_back(op);
            return o;
        });
    //prependToken(X, cls): 将 (D,1,1,1) 的可学习 cls 拼到 X (D,T,1,B) 头部 -> (D,T+1,1,B)
    registerFunction("prependToken", [this](cifa::ObjectVector& v)
        {
            MatrixOp op;
            op.solver_type_ = current_solver_type_;
            auto o = cifa::Object(makeMatrixSPWithState());
            auto cls = v[1].to<MatrixSP>();
            cls->fillData(0);    //中性初始化, 让 attention 自己学
            weights_.push_back(cls.get());
            op.as_prependToken(v[0].to<MatrixSP>(), cls, o.to<MatrixSP>());
            op_queue_.push_back(op);
            return o;
        });
    //firstToken(X): 取序列第 0 个 token, X (D,T,1,B) -> Y (D,1,1,B)
    registerFunction("firstToken", [this](cifa::ObjectVector& v)
        {
            MatrixOp op;
            op.solver_type_ = current_solver_type_;
            auto o = cifa::Object(makeMatrixSPWithState());
            op.as_firstToken(v[0].to<MatrixSP>(), o.to<MatrixSP>());
            op_queue_.push_back(op);
            return o;
        });
    registerFunction("concat", [this](cifa::ObjectVector& v)
        {
            MatrixOp op;
            op.solver_type_ = current_solver_type_;
            auto o = cifa::Object(makeMatrixSPWithState());
            if (v[0].isType<MatrixGroup>())
            {
                auto vm = v[0].to<MatrixGroup>();
                op.as_concat(vm, o.to<MatrixSP>());
            }
            else
            {
                MatrixGroup vm;
                for (int i = 0; i < (int)v.size(); i++)
                {
                    auto vo = getVector(v, i);
                    for (auto& o1 : vo)
                    {
                        vm.push_back(o1.to<MatrixSP>());
                    }
                }
                op.as_concat(vm, o.to<MatrixSP>());
            }
            op_queue_.push_back(op);
            return o;
        });
    registerFunction("max", [this](cifa::ObjectVector& v)
        {
            MatrixOp op;
            op.solver_type_ = current_solver_type_;
            auto o = cifa::Object(makeMatrixSPWithState());
            op.as_max(v[0].to<MatrixSP>(), v[1].to<MatrixSP>(), o.to<MatrixSP>());
            op_queue_.push_back(op);
            return o;
        });
    registerFunction("addLoss", [this](cifa::ObjectVector& v)
        {
            for (auto& o : v)
            {
                loss_ = loss_ + o.to<Loss>();
            }
            return cifa::Object();
        });
    registerFunction("crossEntropy", [this](cifa::ObjectVector& v)
        {
            if (v.size() == 2)
            {
                return Loss(crossEntropy(v[0].to<MatrixSP>(), v[1].to<MatrixSP>()));
            }
            else if (v.size() == 3)
            {
                return Loss(crossEntropy(v[0].to<MatrixSP>(), v[1].to<MatrixSP>(), v[2].to<MatrixSP>()));
            }
        });
    //lmCrossEntropy(logits, targets): 语言模型 per-position softmax + cross-entropy loss
    //logits/targets 形状均为 (vocab_size, T, 1, B)
    //内部对 logits 应用 SOFTMAX_CHANNEL_CE (沿 vocab_size 维做 per-position softmax),
    //再用 crossEntropy 计算 CE loss; 梯度为 (softmax_out - targets) / (T*B), 数值稳定
    registerFunction("lmCrossEntropy", [this](cifa::ObjectVector& v)
        {
            auto logits = v[0].to<MatrixSP>();
            auto targets = v[1].to<MatrixSP>();
            //SOFTMAX_CHANNEL_CE: 前向沿 width_=vocab_size 做 per-position softmax,
            //反向直接透传梯度 (配合 crossEntropy 使用, 合并梯度为 softmax_out - targets)
            MatrixOp op_act;
            op_act.solver_type_ = current_solver_type_;
            auto act_out = cifa::Object(makeMatrixSPWithState());
            op_act.as_active(logits, act_out.to<MatrixSP>(), ACTIVE_FUNCTION_SOFTMAX_CHANNEL_CE,
                std::vector<int>{}, std::vector<float>{}, std::vector<Matrix>{});
            op_queue_.push_back(op_act);
            return Loss(crossEntropy(act_out.to<MatrixSP>(), targets));
        });
    registerFunction("focal", [this](cifa::ObjectVector& v)
        {
            if (v.size() == 2)
            {
                return Loss(focal(v[0].to<MatrixSP>(), v[1].to<MatrixSP>()));
            }
            else if (v.size() == 3)
            {
                return Loss(focal(v[0].to<MatrixSP>(), v[1].to<MatrixSP>(), v[2].to<MatrixSP>()));
            }
        });
    registerFunction("L2", [this](cifa::ObjectVector& v)
        {
            return Loss(L2(v[0].to<MatrixSP>()));
        });
    //mseLoss(A, Y): MSE 重建损失 mean((A-Y)^2), 梯度 2*(A-Y)/N; 可乘系数: addLoss(mseLoss(A,Y)*0.5)
    registerFunction("mseLoss", [this](cifa::ObjectVector& v)
        {
            return Loss(mseLoss(v[0].to<MatrixSP>(), v[1].to<MatrixSP>()));
        });
    //l1Loss(A, Y): L1 重建损失 mean(|A-Y|), 梯度 sign(A-Y)/N
    registerFunction("l1Loss", [this](cifa::ObjectVector& v)
        {
            return Loss(l1Loss(v[0].to<MatrixSP>(), v[1].to<MatrixSP>()));
        });
    //klLoss(mu, log_var [, beta]): KL 散度 -0.5*mean(1+lv-mu^2-exp(lv)), 先验 N(0,1)
    //beta 默认 1.0, 用于 β-VAE; 也可 addLoss(klLoss(mu,lv)*beta)
    registerFunction("klLoss", [this](cifa::ObjectVector& v)
        {
            auto loss = klLoss(v[0].to<MatrixSP>(), v[1].to<MatrixSP>());
            if (v.size() >= 3) { loss = loss * v[2].toDouble(); }
            return Loss(loss);
        });
    //upsample(X, sh, sw [, bilinear]): 空间上采样
    //sh/sw: H/W 方向放大倍率 (整数); bilinear=0 最近邻(默认), bilinear=1 双线性
    //输出 Y: (W*sw, H*sh, C, N)
    registerFunction("upsample", [this](cifa::ObjectVector& v)
        {
            int sh = (int)v[1].toDouble();
            int sw = (v.size() >= 3) ? (int)v[2].toDouble() : sh;
            bool bilinear = (v.size() >= 4) ? (int)v[3].toDouble() != 0 : false;
            MatrixOp op;
            op.solver_type_ = current_solver_type_;
            auto o = cifa::Object(makeMatrixSPWithState());
            op.as_upsample(v[0].to<MatrixSP>(), o.to<MatrixSP>(), sh, sw, bilinear);
            op_queue_.push_back(op);
            return o;
        });
    //chunk(X, i, n [, axis=0]): 沿 width(axis=0) 将 X 均分为 n 块, 取第 i 块 (0-based)
    //X: (W, H, C, N), 输出: (W/n, H, C, N)
    //常见用途: adaLN 的 scale/shift 分块, 与 ncnn Slice 对应
    registerFunction("chunk", [this](cifa::ObjectVector& v)
        {
            int chunk_i = (int)v[1].toDouble();
            int n_total = (int)v[2].toDouble();
            MatrixOp op;
            op.solver_type_ = current_solver_type_;
            auto o = cifa::Object(makeMatrixSPWithState());
            op.as_chunk(v[0].to<MatrixSP>(), o.to<MatrixSP>(), chunk_i, n_total);
            op_queue_.push_back(op);
            return o;
        });
    //sliceW(X, start_w, size_w): 沿 width(axis=0) 取任意范围 [start_w, start_w+size_w)
    //X: (W, H, C, N), 输出: (size_w, H, C, N); 不要求等分
    //配合 permute({1,0,2,3}) 可实现对 height(序列)维的切片
    registerFunction("sliceW", [this](cifa::ObjectVector& v)
        {
            int start_w = (int)v[1].toDouble();
            int size_w = (int)v[2].toDouble();
            MatrixOp op;
            op.solver_type_ = current_solver_type_;
            auto o = cifa::Object(makeMatrixSPWithState());
            op.as_sliceW(v[0].to<MatrixSP>(), o.to<MatrixSP>(), start_w, size_w);
            op_queue_.push_back(op);
            return o;
        });
    //sinTimeEmbed(t, d [, base=10000]): 正弦时间步嵌入
    //t: 标量 MatrixSP (1,1,1,B 或 1,1,1,1); d: 输出维度; base: 频率基数
    //输出 (d, 1, 1, B): [cos(t*f_0)..cos(t*f_{d/2-1}), sin(t*f_0)..sin(t*f_{d/2-1})]
    //freq_i = 1/base^(2*i/d); 对应 ncnn t_embedder 的正弦编码部分
    registerFunction("sinTimeEmbed", [this](cifa::ObjectVector& v)
        {
            int d = (int)v[1].toDouble();
            float base = (v.size() >= 3) ? (float)v[2].toDouble() : 10000.0f;
            MatrixOp op;
            op.solver_type_ = current_solver_type_;
            auto o = cifa::Object(makeMatrixSPWithState());
            op.as_sinTimeEmbed(v[0].to<MatrixSP>(), o.to<MatrixSP>(), d, base);
            op_queue_.push_back(op);
            return o;
        });
    registerFunction("crossEntropy2", [this](cifa::ObjectVector& v)
        {
            std::vector<MatrixSP> vm;
            for (auto& o : getVector(v, 0))
            {
                vm.push_back(o.to<MatrixSP>());
            }
            return Loss(commonLoss(MatrixOpType::LOSS, vm, getRealVector(v, 1)));
        });
    registerFunction("commonloss", [this](cifa::ObjectVector& v)
        {
            std::vector<MatrixSP> vm;
            for (auto& o : getVector(v, 1))
            {
                vm.push_back(o.to<MatrixSP>());
            }
            return Loss(commonLoss(MatrixOpType(int(v[0])), vm, getRealVector(v, 2)));
        });
    registerFunction("matrixGroup", [this](cifa::ObjectVector& v)
        {
            return MatrixGroup({});
        });
    registerFunction("addIntoGroup", [this](cifa::ObjectVector& v)
        {
            v[0].to<MatrixGroup>().push_back(v[1].to<MatrixSP>());
            return cifa::Object();
        });
    registerFunction("getA", [this](cifa::ObjectVector& v)
        {
            return cifa::Object(A_);
        });
    registerFunction("getX", [this](cifa::ObjectVector& v)
        {
            return cifa::Object(X_);
        });
    registerFunction("getY", [this](cifa::ObjectVector& v)
        {
            return cifa::Object(Y_);
        });
    registerFunction("getLossWeight", [this](cifa::ObjectVector& v)
        {
            return cifa::Object(extra_matrixsp_["loss_weight"]);
        });
    registerFunction("getMatrix", [this](cifa::ObjectVector& v)
        {
            return cifa::Object(getMatrixByName(v[0].toString()));
        });
    registerFunction("haveMatrix", [this](cifa::ObjectVector& v)
        {
            return cifa::Object(extra_matrixsp_.contains(v[0].toString()));
        });
    registerFunction("showMatrix", [this](cifa::ObjectVector& v)
        {
            extra_matrixsp_[v[0].toString()] = v[1].to<MatrixSP>();
            return cifa::Object();
        });
    //registerMatrix("name", M): showMatrix 的别名，将矩阵注册为可被外部访问的命名矩阵
    registerFunction("registerMatrix", [this](cifa::ObjectVector& v)
        {
            extra_matrixsp_[v[0].toString()] = v[1].to<MatrixSP>();
            return cifa::Object();
        });
    //setIsWeight(M, 0_or_1): 强制设置矩阵是否为权重（is_weight_标志），影响保存/加载和 Solver 更新
    registerFunction("setIsWeight", [](cifa::ObjectVector& v)
        {
            v[0].to<MatrixSP>()->setIsWeight(v[1].toInt() != 0);
            return cifa::Object();
        });
    for (int i = -1; i < 100; i++)
    {
        auto str = option_->getStringFromEnum(ActiveFunctionType(i));
        if (str != "")
        {
            cifa_.register_parameter("active_" + str, i);
            registerFunction(str, [this, i](cifa::ObjectVector& v)
                {
                    MatrixOp op;
                    auto o = cifa::Object(makeMatrixSPWithState());
                    auto Y = o.to<MatrixSP>();
                    op.as_active(v[0].to<MatrixSP>(), Y, ActiveFunctionType(i), getIntVector(v, 1), getRealVector(v, 2), {});
                    op_queue_.push_back(op);
                    return o;
                });
        }
    }
    for (int i = -1; i < 100; i++)
    {
        auto str = option_->getStringFromEnum(PoolingType(i));
        if (str != "")
        {
            cifa_.register_parameter("pool_" + str, i);
        }
    }
    for (int i = -1; i < 100; i++)
    {
        auto str = MatrixOp::getOpName(MatrixOpType(i));
        if (str != "")
        {
            cifa_.register_parameter("op_" + str, i);
        }
    }
    registerFunction("setOption", [this](cifa::ObjectVector& v)
        {
            option_->setKey(v[0].toString(), v[1].toString(), v[2].toString());
            return cifa::Object();
        });

    return 0;
}

inline void NetCifa::registerFunction(std::string name, cifa::Cifa::func_type func)
{
    cifa_.register_function(name, func);
    auto str_lower = strfunc::toLowerCase(name);
    if (str_lower != name)
    {
        cifa_.register_function(str_lower, func);    //再注册一个小写版本，兼容性需求
    }
}

void NetCifa::setXA(const MatrixSP& X, const MatrixSP& A)
{
    X_ = X;
    A_ = A;
}
}    // namespace cccc