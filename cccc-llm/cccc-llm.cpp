#include "cccc-llm.h"
#include "INIReader.h"
#include "Log.h"
#include "MainProcess.h"
#include "Timer.h"
#include "Tokenizer.h"
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <random>
#include <regex>
#include <sstream>
#include <string>
#include <vector>

#define NOMINMAX
#include <windows.h>

// ── DLL entry point ───────────────────────────────────────────────────────────

BOOL WINAPI DllMain(HINSTANCE, DWORD, LPVOID) { return TRUE; }

// ── ini helpers ───────────────────────────────────────────────────────────────

// Process \n and \t escape sequences in values read from ini files.
static std::string unescape_ini(const std::string& src)
{
    std::string r;
    r.reserve(src.size());
    for (size_t i = 0; i < src.size(); i++)
    {
        if (src[i] == '\\' && i + 1 < src.size())
        {
            switch (src[i + 1])
            {
            case 'n':
                r += '\n';
                i++;
                break;
            case 't':
                r += '\t';
                i++;
                break;
            case '\\':
                r += '\\';
                i++;
                break;
            default: r += src[i]; break;
            }
        }
        else
        {
            r += src[i];
        }
    }
    return r;
}

// Parse a comma-separated list of integers (e.g. "151645,151643").
static std::vector<int> parse_int_list(const std::string& s)
{
    std::vector<int> result;
    std::istringstream ss(s);
    std::string tok;
    while (std::getline(ss, tok, ','))
    {
        auto start = tok.find_first_not_of(" \t");
        auto end = tok.find_last_not_of(" \t");
        if (start != std::string::npos)
        {
            try
            {
                result.push_back(std::stoi(tok.substr(start, end - start + 1)));
            } catch (...)
            {
            }
        }
    }
    return result;
}

// ── session ───────────────────────────────────────────────────────────────────

struct LlmSession
{
    cccc::MainProcess mp;    // nets_[0]=prefill (T=N), nets_[1]=decode (T=1, kvcache)
    Tokenizer tok;
    std::vector<int> sys_ids;
    std::vector<int> ctx_ids;
    std::vector<float> x_buf;    // T-element input for prefill
    std::vector<float> y_buf;    // T*V-element output for prefill (half or float)
    std::vector<float> x_dec;    // 1-element input for decode
    std::vector<float> y_dec;    // V-element output for decode (half or float)
    std::vector<int> asst_header;
    std::vector<int> im_end_nl;
    std::vector<int> eos_ids;            // configurable EOS token IDs (any hit → stop)
    int think_open_id = -1;              // token ID for <think>; -1 = disabled
    int think_close_id = -1;             // token ID for </think>; -1 = disabled
    std::vector<int> user_prefix_ids;    // encoded prefix for user turns
    std::vector<int> user_suffix_ids;    // encoded suffix for user turns
    std::vector<int> banned_gen_ids;     // control tokens that should never be emitted as assistant text
    std::string no_think_str;            // appended to asst_header in llm_set_no_think (empty = no-op)
    int T = 0;
    int V = 0;
    int T_kv = 0;              // KV cache capacity (= T for current ini; used to guard RoPE/KV overflow)
    int current_turn_n = 0;    // tokens added by the latest prepare_ctx (user_turn + asst_header)
    int first_turn_n = 0;      // snapshot of current_turn_n from the first prepare_ctx call (original user task)
    bool half_output = false;
    bool has_decode = false;    // true when net group 1 (decode) is present
    cccc::DataType output_dt = cccc::DataType::FLOAT;

    // sampling parameters (read from [llm] ini; defaults = greedy/no-penalty)
    float temperature = 0.0f;           // 0 = greedy argmax
    int top_k = 0;                      // 0 = no top-k filter
    float top_p = 1.0f;                 // 1.0 = no nucleus filter
    float repetition_penalty = 1.0f;    // 1.0 = no penalty
    std::mt19937 rng{ std::random_device{}() };
};

// FP8 E4M3 转 float（CPU 端，无需 scale 即可比较大小，argmax 结果与原始 float 一致）
static float fp8e4m3_to_float(uint8_t b)
{
    // NaN 模式：E=1111, M=111（0x7F 或 0xFF）→ 当作 0
    if ((b & 0x7F) == 0x7F)
    {
        return 0.0f;
    }
    bool neg = (b & 0x80) != 0;
    int e = (b >> 3) & 0x0F;
    int m = b & 0x07;
    float v = (e == 0) ? (m / 8.0f) * (1.0f / 64.0f)    // 次正规数: 2^(1-7) * (m/8)
                         :
                         (1.0f + m / 8.0f) * std::ldexp(1.0f, e - 7);    // 正规数:   2^(e-7) * (1 + m/8)
    return neg ? -v : v;
}

// 对 logit 缓冲区取 argmax（支持 float32 / fp16 / bf16 / fp8_e4m3）
static int argmax_logits(const void* data, int V, cccc::DataType dt)
{
    if (dt == cccc::DataType::HALF)
    {
        auto* p = (const cccc::half*)(data);
        return (int)(std::max_element(p, p + V) - p);
    }
    else if (dt == cccc::DataType::BFLOAT16)
    {
        auto* p = (const cccc::bfloat16*)(data);
        return (int)(std::max_element(p, p + V,
                         [](const cccc::bfloat16& a, const cccc::bfloat16& b)
                         {
                             return (float)a < (float)b;
                         })
            - p);
    }
    else if (dt == cccc::DataType::FP8_E4M3)
    {
        auto* p = (const uint8_t*)(data);
        int best = 0;
        float best_val = fp8e4m3_to_float(p[0]);
        for (int i = 1; i < V; i++)
        {
            float v = fp8e4m3_to_float(p[i]);
            if (v > best_val)
            {
                best_val = v;
                best = i;
            }
        }
        return best;
    }
    else
    {
        auto* p = (const float*)(data);
        return (int)(std::max_element(p, p + V) - p);
    }
}

// Sample (or argmax) from logit buffer with temperature / top-k / top-p / repetition-penalty.
// Falls back to fast argmax when all params are at their defaults (temperature<=0, rep_penalty==1).
static int sample_logits(LlmSession& s, const void* data, int V, cccc::DataType dt)
{
    const float temperature = s.temperature;
    const float rep_penalty = s.repetition_penalty;

    // 转换为 float（temperature/top-p 路径需要）
    std::vector<float> logits(V);
    if (dt == cccc::DataType::HALF)
    {
        auto* p = (const cccc::half*)(data);
        for (int i = 0; i < V; i++)
        {
            logits[i] = (float)p[i];
        }
    }
    else if (dt == cccc::DataType::BFLOAT16)
    {
        auto* p = (const cccc::bfloat16*)(data);
        for (int i = 0; i < V; i++)
        {
            logits[i] = (float)p[i];
        }
    }
    else if (dt == cccc::DataType::FP8_E4M3)
    {
        auto* p = (const uint8_t*)(data);
        for (int i = 0; i < V; i++)
        {
            logits[i] = fp8e4m3_to_float(p[i]);
        }
    }
    else
    {
        auto* p = (const float*)(data);
        std::copy(p, p + V, logits.begin());
    }

    // Never emit chat-control marker tokens as assistant text.
    for (int id : s.banned_gen_ids)
    {
        if (id >= 0 && id < V)
        {
            logits[id] = -1e30f;
        }
    }

    // Repetition penalty: penalise tokens seen in the last 512 context tokens
    if (rep_penalty != 1.0f && !s.ctx_ids.empty())
    {
        int start = (int)s.ctx_ids.size() > 512 ? (int)s.ctx_ids.size() - 512 : 0;
        for (int i = start; i < (int)s.ctx_ids.size(); i++)
        {
            int id = s.ctx_ids[i];
            if (id >= 0 && id < V)
            {
                if (logits[id] > 0.0f)
                {
                    logits[id] /= rep_penalty;
                }
                else
                {
                    logits[id] *= rep_penalty;
                }
            }
        }
    }

    // Greedy after penalty only
    if (temperature <= 0.0f)
    {
        return (int)(std::max_element(logits.begin(), logits.end()) - logits.begin());
    }

    // Apply temperature
    for (auto& l : logits)
    {
        l /= temperature;
    }

    // Top-k: zero out logits below the k-th largest
    if (s.top_k > 0 && s.top_k < V)
    {
        std::vector<float> tmp(logits.begin(), logits.end());
        std::nth_element(tmp.begin(), tmp.begin() + s.top_k - 1, tmp.end(), std::greater<float>());
        float threshold = tmp[s.top_k - 1];
        for (auto& l : logits)
        {
            if (l < threshold)
            {
                l = -1e30f;
            }
        }
    }

    // Softmax
    float max_l = *std::max_element(logits.begin(), logits.end());
    float sum = 0.0f;
    for (auto& l : logits)
    {
        l = std::exp(l - max_l);
        sum += l;
    }
    if (sum > 0.0f)
    {
        for (auto& l : logits)
        {
            l /= sum;
        }
    }

    // Top-p (nucleus): zero out tokens after cumulative prob exceeds top_p
    if (s.top_p < 1.0f)
    {
        std::vector<std::pair<float, int>> sp(V);
        for (int i = 0; i < V; i++)
        {
            sp[i] = { logits[i], i };
        }
        std::sort(sp.begin(), sp.end(), std::greater<std::pair<float, int>>());
        float cum = 0.0f;
        bool cutting = false;
        for (auto& [p, i] : sp)
        {
            if (cutting)
            {
                logits[i] = 0.0f;
                continue;
            }
            cum += p;
            if (cum > s.top_p)
            {
                cutting = true;
            }
        }
        // Renormalise
        sum = 0.0f;
        for (auto& l : logits)
        {
            sum += l;
        }
        if (sum > 0.0f)
        {
            for (auto& l : logits)
            {
                l /= sum;
            }
        }
    }

    // Multinomial sample
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    float r = dist(s.rng);
    float cum = 0.0f;
    for (int i = 0; i < V; i++)
    {
        cum += logits[i];
        if (r <= cum)
        {
            return i;
        }
    }
    return V - 1;
}

// ── generation ────────────────────────────────────────────────────────────────

static std::string run_generate(LlmSession& s, int max_new_tokens, bool show_thinking,
    LlmStreamCallback callback = nullptr, void* userdata = nullptr)
{
    // EOS and think token IDs are configured per model in [llm] ini section.
    std::string result;
    int think_depth = 0;

    // ── prefill ─────────────────────────────────────────────────────────────
    // 初始 prefill：将 ctx_ids 的最后 n 个 token 填入 x_buf（n ≤ T），计算 logit。
    int n = std::min((int)s.ctx_ids.size(), s.T);
    int offset = (int)s.ctx_ids.size() - n;

    // 按需扩容（不清零 y_buf，会被 testExternalData 覆写）
    // y_buf 按实际数据类型字节数分配，以正确存放 BF16/FP16/FP32 输出
    s.x_buf.resize(s.T);
    int elem_bytes = cccc::MatrixData::getDataTypeSize(s.output_dt);
    size_t y_buf_floats = ((size_t)s.V * s.T * elem_bytes + sizeof(float) - 1) / sizeof(float);
    s.y_buf.resize(y_buf_floats);

    // 左对齐：token 置于 x_buf[0..n-1]，后面补 0；logit 正确在 (n-1)*V 处
    std::fill(s.x_buf.begin(), s.x_buf.end(), 0.0f);
    for (int i = 0; i < n; i++)
    {
        s.x_buf[i] = (float)s.ctx_ids[offset + i];
    }

    s.mp.getNet()->resetKVCache();
    Timer t_prefill;
    s.mp.testExternalData(s.x_buf.data(), nullptr, s.y_buf.data(), 1, 0, nullptr);
    double prefill_ms = t_prefill.getElapsedTime() * 1000.0;
    cccc::LOG("[timing] prefill: {} tokens, {:.1f} ms ({:.1f} tok/s)\n",
        n, prefill_ms, n * 1000.0 / prefill_ms);

    if (s.has_decode)
    {
        auto* net_d = s.mp.getNet(0, 1);
        net_d->setKVCachePos(n);
    }

    // 从 prefill 最后一个位置提取 logit
    int next_id;
    {
        int elem_bytes = cccc::MatrixData::getDataTypeSize(s.output_dt);
        const char* logits_ptr = (const char*)(s.y_buf.data()) + (size_t)(n - 1) * s.V * elem_bytes;
        next_id = sample_logits(s, logits_ptr, s.V, s.output_dt);
    }

    // ── decode loop ──────────────────────────────────────────────────────────
    int pos = n;                                 // 绝对 KV cache 位置
    int ctx_at_start = (int)s.ctx_ids.size();    // 生成开始前的 ctx 长度
    int rebuild_count = 0;
    int decode_steps = 0;
    Timer t_decode;

    for (int step = 0; step < max_new_tokens; step++)
    {
        if (!s.eos_ids.empty() && std::find(s.eos_ids.begin(), s.eos_ids.end(), next_id) != s.eos_ids.end())
        {
            break;
        }

        s.ctx_ids.push_back(next_id);

        bool emit_piece = true;
        if (s.think_open_id >= 0 && next_id == s.think_open_id)
        {
            think_depth++;
            emit_piece = show_thinking;
        }
        else if (s.think_close_id >= 0 && next_id == s.think_close_id)
        {
            if (think_depth > 0)
            {
                think_depth--;
                emit_piece = show_thinking;
            }
        }
        else if (!show_thinking && think_depth > 0)
        {
            emit_piece = false;
        }

        if (emit_piece)
        {
            std::string piece = s.tok.decode({ next_id });
            if (callback)
            {
                callback(piece.c_str(), userdata);
            }
            result += piece;
        }

        // KV cache 重建：当 pos 达到 T_kv 时，重新 prefill
        // 布局：[sys_ids | current_turn(user+asst_header) | thinking_marker | gen_tail]
        if (s.has_decode && s.T_kv > 0 && pos >= s.T_kv)
        {
            // 安全上限：最多重建 4 次，防止无限循环
            if (++rebuild_count > 4)
            {
                cccc::LOG_ERR("[llm] KV-cache rebuild limit reached, stopping.\n");
                break;
            }

            int sys_n = (int)s.sys_ids.size();
            int turn_n = s.current_turn_n;    // user_turn + asst_header
            int anchor_n = sys_n + turn_n;
            int gen_so_far = (int)s.ctx_ids.size() - ctx_at_start;
            // 重建后必须留出空间给后续 decode，上限取 min(T, T_kv/2)
            int max_rebuild = std::min(s.T, s.T_kv / 2);
            if (max_rebuild <= anchor_n)
            {
                max_rebuild = anchor_n + 1;
            }
            int max_tail = max_rebuild - anchor_n;

            std::fill(s.x_buf.begin(), s.x_buf.end(), 0.0f);

            // 1. sys_ids
            for (int i = 0; i < sys_n && i < s.T; i++)
            {
                s.x_buf[i] = (float)s.ctx_ids[i];
            }

            // 2. current_turn（user + asst_header）
            // turn_start must always be sys_n, not ctx_at_start - turn_n.
            // After tool injections, ctx_at_start grows beyond sys_n+turn_n, so
            // ctx_at_start - turn_n would point into tool results, not the user task.
            int turn_start = sys_n;
            for (int i = 0; i < turn_n && sys_n + i < s.T; i++)
            {
                s.x_buf[sys_n + i] = (float)s.ctx_ids[turn_start + i];
            }

            // 3. gen_tail：根据思考状态插入标记，帮助模型正确延续
            int tail_n;
            if (think_depth > 0 && s.think_open_id >= 0)
            {
                // 仍在思考中：在 gen_tail 前插入 <think>，保持模型思考状态
                s.x_buf[anchor_n] = (float)s.think_open_id;
                int actual_tail = std::min(gen_so_far, max_tail - 1);
                int gen_tail_start = (int)s.ctx_ids.size() - actual_tail;
                for (int i = 0; i < actual_tail; i++)
                {
                    s.x_buf[anchor_n + 1 + i] = (float)s.ctx_ids[gen_tail_start + i];
                }
                tail_n = 1 + actual_tail;
            }
            else
            {
                // think_depth == 0: not currently thinking.
                // Take the last max_tail tokens from the pre-generation portion of
                // ctx_ids (ctx_ids[anchor_n .. ctx_at_start-1]) so that tool results
                // injected in previous rounds survive the rebuild.  The tail ends
                // cleanly at the assistant header (the last token before generation
                // started), so the model generates a fresh response after rebuild.
                int available = ctx_at_start - anchor_n;
                int actual_tail = std::min(available, max_tail);
                if (actual_tail < 0) { actual_tail = 0; }
                int tail_start = ctx_at_start - actual_tail;
                for (int i = 0; i < actual_tail; i++)
                {
                    s.x_buf[anchor_n + i] = (float)s.ctx_ids[tail_start + i];
                }
                tail_n = actual_tail;
            }

            int rebuild_n = anchor_n + tail_n;

            s.mp.rebuildKVCache(0, 1, s.x_buf.data(), rebuild_n, s.y_buf.data());

            {
                int elem_bytes = cccc::MatrixData::getDataTypeSize(s.output_dt);
                const char* h = (const char*)(s.y_buf.data()) + (size_t)(rebuild_n - 1) * s.V * elem_bytes;
                next_id = sample_logits(s, h, s.V, s.output_dt);
            }

            pos = rebuild_n;
            // 截断 ctx_ids 使其与重建后的 KV cache 完全一致。
            // 不截断的话，rebuild 前的 partial 生成 token 会残留在 ctx_ids 里，
            // 下一轮 llm_continue_stream prefill 时会把它们当有效上下文带入，
            // 导致模型看到"幽灵" token，反复忘记工具结果。
            s.ctx_ids.resize(rebuild_n);
            ctx_at_start = rebuild_n;
            cccc::LOG_ERR("[llm] KV-cache rebuild #{}: {} tokens (sys={} turn={} tail={} think_depth={}), pos={}\n",
                rebuild_count, rebuild_n, sys_n, turn_n, tail_n, think_depth, pos);
            continue;
        }

        // Decode 一步
        if (s.has_decode)
        {
            auto* net_d = s.mp.getNet(0, 1);
            net_d->setKVCachePos(pos);
            net_d->setRopeOffset(pos);
            net_d->setAttentionOffset(pos);

            s.x_dec[0] = (float)next_id;
            s.mp.testExternalData(s.x_dec.data(), nullptr, s.y_dec.data(), 1, 0, nullptr, 1);

            next_id = sample_logits(s, s.y_dec.data(), s.V, s.output_dt);
            pos++;
            decode_steps++;
        }
        else
        {
            // 无 decode 模型时，用 prefill 每步
            int nc = std::min((int)s.ctx_ids.size(), s.T);
            int offc = (int)s.ctx_ids.size() - nc;
            std::fill(s.x_buf.begin(), s.x_buf.end(), 0.0f);
            for (int i = 0; i < nc; i++)
            {
                s.x_buf[i] = (float)s.ctx_ids[offc + i];
            }
            s.mp.testExternalData(s.x_buf.data(), nullptr, s.y_buf.data(), 1, 0, nullptr);
            {
                int elem_bytes = cccc::MatrixData::getDataTypeSize(s.output_dt);
                const char* logits_ptr = (const char*)(s.y_buf.data()) + (size_t)(nc - 1) * s.V * elem_bytes;
                next_id = sample_logits(s, logits_ptr, s.V, s.output_dt);
            }
        }
    }

    double decode_ms = t_decode.getElapsedTime() * 1000.0;
    if (decode_steps > 0)
    {
        cccc::LOG("[timing] decode: {} tokens, {:.1f} ms ({:.1f} tok/s)\n",
            decode_steps, decode_ms, (double)decode_steps * 1000.0 / decode_ms);
    }

    return result;
}

// ── exported functions ────────────────────────────────────────────────────────

// 从 prefill 网络的 cifa 脚本自动推导 decode 网络脚本（T=1, KV cache 使用 T_kv）。
// 规则：
//   T = <N>;                          →  T = 1;\n    T_kv = <N>;
//   ropeCosTbl/ropeSinTbl(T, ...       →  使用 T_kv
//   MatrixWithName("[KV]cache_...", hd, T, 1, HkvB)  →  T_kv
//   reshapeBatch(K/V_cached, {hd, T,   →  T_kv
//   reshapeBatch(K/V_r4,     {hd, T,   →  T_kv
//   registerMatrix(...) 行             →  删除
static std::string make_decode_structure(const std::string& prefill)
{
    std::string s = prefill;
    // 1. rope 查找表（正则处理空格差异）
    s = std::regex_replace(s,
        std::regex(R"(ropeCosTbl\(\s*T\s*,)"),
        "ropeCosTbl(T_kv,");
    s = std::regex_replace(s,
        std::regex(R"(ropeSinTbl\(\s*T\s*,)"),
        "ropeSinTbl(T_kv,");
    // 2. KV cache MatrixWithName 的维度参数（正则处理空格差异）
    s = std::regex_replace(s,
        std::regex(R"(,\s*hd,\s*T,\s*1,\s*HkvB)"),
        ", hd, T_kv, 1, HkvB");
    // 3. KV cache 相关的 reshapeBatch（正则处理空格差异）
    s = std::regex_replace(s,
        std::regex(R"((reshapeBatch\([KV]_(?:cached|r4),\s*\{hd,\s*)T\b)"),
        "$1T_kv");
    // 4. registerMatrix 行注释掉（仅 prefill 网络需要权重共享注册）
    s = std::regex_replace(s,
        std::regex(R"(\n([ \t]*)registerMatrix\()"),
        "\n$1//registerMatrix(");
    // 5. 首次 T = <N>; → T = 1; + T_kv = <N>;（正则处理空格差异，保留行缩进）
    // 用 \n 作锚点，避免 multiline 标志的兼容性问题
    s = std::regex_replace(s,
        std::regex(R"((\n[ \t]*)T\s*=\s*(\d+)\s*;)"),
        "$1T = 1;$1T_kv = $2;",
        std::regex_constants::format_first_only);
    return s;
}

CCCC_LLM_API LlmHandle llm_init(const std::string& ini_path_str, const std::string& system_prompt)
{
    std::filesystem::path ini_path = std::filesystem::path(ini_path_str);
    std::filesystem::path ini_dir = ini_path.has_parent_path() ? ini_path.parent_path() : std::filesystem::path(".");

    auto* s = new LlmSession();

    s->mp.getOption()->setOutput(0);
    {
        std::ifstream ifs(ini_path);
        if (!ifs)
        {
            cccc::LOG_ERR("llm_init: cannot open ini file: {}\n", ini_path.string());
            delete s;
            return nullptr;
        }
        std::string ini_str((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());

        // 读 load_file 并解析为绝对路径
        INIReaderNoUnderline ini_reader;
        ini_reader.loadString(ini_str);
        std::string load_file = ini_reader.getString("train", "load_file", "");
        std::string abs_load_file;
        if (!load_file.empty())
        {
            std::filesystem::path lf(load_file);
            if (!lf.is_absolute())
            {
                lf = (ini_dir / lf).lexically_normal();
            }
            abs_load_file = lf.string();
            if (!std::filesystem::path(load_file).is_absolute())
            {
                ini_str += std::format("\n[train]\nload_file={}\n", abs_load_file);
            }
        }

        ini_str += "\n[train]\noutput_log=0\nneed_free_mem=0\n";

        // 若 net_num>=2 但 ini 中没有 structure1，则从 structure0 自动推导
        {
            INIReaderNoUnderline ini_check;
            ini_check.loadString(ini_str);
            if (ini_check.getInt("net", "net_num", 1) >= 2
                && ini_check.getString("net", "structure1", "").empty())
            {
                std::string s0 = ini_check.getString("net", "structure0", "");
                if (!s0.empty())
                {
                    std::string s1 = make_decode_structure(s0);
                    ini_str += "\n[net]\nstructure1='" + s1 + "'\n";
                    cccc::LOG_ERR("llm_init: structure1 auto-generated from structure0\n");
                }
            }
        }

        if (s->mp.init(ini_str) != 0)
        {
            cccc::LOG_ERR("llm_init: mp.init() failed\n");
            delete s;
            return nullptr;
        }
    }

    // 显示模型加载后的显存使用情况
    {
        auto* net0 = s->mp.getNet(0, 0);
        if (net0 && net0->getGpu())
        {
            size_t free_mem = 0, total_mem = 0;
            net0->getGpu()->getFreeMemory(free_mem, total_mem);
            double total_gb = (double)(total_mem) / (1024.0 * 1024.0 * 1024.0);
            double free_gb = (double)(free_mem) / (1024.0 * 1024.0 * 1024.0);
            double used_gb = total_gb - free_gb;
            cccc::LOG_ERR("[llm] VRAM: {:.2f} GB used / {:.2f} GB total ({:.2f} GB free)\n",
                used_gb, total_gb, free_gb);
        }
    }

    // 从 ini [llm] tokenizer 读取路径；不填则回退到 ini 同目录下的 tokenizer.json
    {
        std::string tok_path = s->mp.getOption()->getString("llm", "tokenizer", "");
        if (tok_path.empty())
        {
            tok_path = (ini_dir / "tokenizer.json").lexically_normal().string();
        }
        else if (!std::filesystem::path(tok_path).is_absolute())
        {
            tok_path = (ini_dir / tok_path).lexically_normal().string();
        }
        if (!s->tok.load(tok_path))
        {
            cccc::LOG_ERR("llm_init: failed to load tokenizer from {}\n", tok_path);
            delete s;
            return nullptr;
        }
    }
    // 主网络（group 0，prefill）
    auto* net0 = s->mp.getNet(0, 0);
    net0->getGpu()->setActivePhase(cccc::ACTIVE_PHASE_TEST);
    s->T = net0->getX().getRow();
    s->V = s->T > 0 ? net0->getA().getRow() / s->T : 151936;
    s->output_dt = net0->getA().getDataType();

    // 若 ini 中 net_num=2 则 group 1（decode 网络）已被一并加载，权重/KV-cache 已共享
    auto* net1 = s->mp.getNet(0, 1);
    if (s->output_dt == cccc::DataType::FP8_E4M3 || s->output_dt == cccc::DataType::FP8_E5M2)
    {
        // Probe: keep the final logits in BF16 even when intermediate activations run in FP8.
        // If this restores non-zero logits, the remaining bug is in the last output quantization,
        // not in the full FP8 compute graph.
        net0->getA() = cccc::Matrix(net0->getA().getDim(), cccc::DataType::BFLOAT16, net0->getA().getDeviceType());
        if (net1 != nullptr)
        {
            net1->getA() = cccc::Matrix(net1->getA().getDim(), cccc::DataType::BFLOAT16, net1->getA().getDeviceType());
        }
        s->output_dt = cccc::DataType::BFLOAT16;
    }
    s->half_output = (s->output_dt == cccc::DataType::HALF);

    if (net1 != nullptr)
    {
        net1->getGpu()->setActivePhase(cccc::ACTIVE_PHASE_TEST);
        s->x_dec.resize(1, 0.0f);
        // y_dec 按原始数据类型字节数分配（BF16=2B/元素, FP32=4B/元素）
        int dec_bytes = s->V * cccc::MatrixData::getDataTypeSize(s->output_dt);
        s->y_dec.resize((dec_bytes + sizeof(float) - 1) / sizeof(float), 0.0f);
        s->has_decode = true;

        // Determine T_kv from shared KV-cache matrix Kcache_0 (dim = {hd, T_kv, 1, HkvB})
        {
            auto& extra = net0->getAllExtraMatrices();
            auto it = extra.find("Kcache_0");
            if (it != extra.end())
            {
                s->T_kv = it->second->getHeight();    // height_ = T_kv
            }
        }
        if (s->T_kv <= 0)
        {
            s->T_kv = s->T;    // fallback
        }
        cccc::LOG_ERR("llm_init: decode model (group 1) loaded, KV-cache sharing enabled, T_kv={}\n", s->T_kv);
    }
    else
    {
        cccc::LOG_ERR("llm_init: no decode group — prefill-only mode\n");
    }

    // ── chat format config (read from [llm] section, \n in values unescaped) ──
    auto getfmt = [&](const std::string& key, const std::string& def) -> std::string
    {
        return unescape_ini(s->mp.getOption()->getString("llm", key, def));
    };

    // EOS tokens (comma-separated token IDs)
    s->eos_ids = parse_int_list(getfmt("eos_tokens", "151645,151643"));

    // Think markers (token IDs, -1 = disabled)
    s->think_open_id = s->mp.getOption()->getInt("llm", "think_open_id", 151667);
    s->think_close_id = s->mp.getOption()->getInt("llm", "think_close_id", 151668);

    // No-think string appended to asst_header by llm_set_no_think (empty = no-op)
    s->no_think_str = getfmt("no_think_str", "<think>\n</think>\n");

    // Sampling parameters (optional; defaults = greedy)
    s->temperature = (float)s->mp.getOption()->getReal("llm", "temperature", 0.0);
    s->top_k = s->mp.getOption()->getInt("llm", "top_k", 0);
    s->top_p = (float)s->mp.getOption()->getReal("llm", "top_p", 1.0);
    s->repetition_penalty = (float)s->mp.getOption()->getReal("llm", "repetition_penalty", 1.0);

    // User turn prefix/suffix
    std::string user_prefix = getfmt("user_prefix", "<|im_start|>user\n");
    std::string user_suffix = getfmt("user_suffix", "<|im_end|>\n");
    if (!user_prefix.empty())
    {
        s->user_prefix_ids = s->tok.encode(user_prefix);
    }
    if (!user_suffix.empty())
    {
        s->user_suffix_ids = s->tok.encode(user_suffix);
    }

    // System turn prefix/suffix
    std::string sys_prefix = getfmt("sys_prefix", "<|im_start|>system\n");
    std::string sys_suffix = getfmt("sys_suffix", "<|im_end|>\n");
    std::vector<int> sys_prefix_ids;
    std::vector<int> sys_suffix_ids;
    if (!sys_prefix.empty())
    {
        sys_prefix_ids = s->tok.encode(sys_prefix);
    }
    if (!sys_suffix.empty())
    {
        sys_suffix_ids = s->tok.encode(sys_suffix);
    }

    // Assistant turn prefix/suffix
    std::string asst_prefix = getfmt("asst_prefix", "<|im_start|>assistant\n");
    std::string asst_suffix = getfmt("asst_suffix", "<|im_end|>\n");
    if (!asst_prefix.empty())
    {
        s->asst_header = s->tok.encode(asst_prefix);
    }
    if (!asst_suffix.empty())
    {
        s->im_end_nl = s->tok.encode(asst_suffix);
    }

    auto append_banned = [&](const std::vector<int>& ids)
    {
        for (int id : ids)
        {
            if (id < 0) { continue; }
            if (std::find(s->eos_ids.begin(), s->eos_ids.end(), id) != s->eos_ids.end()) { continue; }
            if (std::find(s->banned_gen_ids.begin(), s->banned_gen_ids.end(), id) == s->banned_gen_ids.end())
            {
                s->banned_gen_ids.push_back(id);
            }
        }
    };
    append_banned(s->user_prefix_ids);
    append_banned(s->user_suffix_ids);
    append_banned(sys_prefix_ids);
    append_banned(sys_suffix_ids);

    if (!system_prompt.empty())
    {
        if (!sys_prefix.empty())
        {
            auto pf = s->tok.encode(sys_prefix);
            s->sys_ids.insert(s->sys_ids.end(), pf.begin(), pf.end());
        }
        auto content_ids = s->tok.encode(system_prompt);
        s->sys_ids.insert(s->sys_ids.end(), content_ids.begin(), content_ids.end());
        if (!sys_suffix.empty())
        {
            auto sf = s->tok.encode(sys_suffix);
            s->sys_ids.insert(s->sys_ids.end(), sf.begin(), sf.end());
        }
    }
    s->ctx_ids = s->sys_ids;
    return s;
}

CCCC_LLM_API void llm_set_no_think(LlmHandle handle)
{
    auto* s = (LlmSession*)(handle);
    if (!s || s->no_think_str.empty()) { return; }
    // Append the configured no-think string to the base assistant prefix so the
    // model skips internal reasoning and replies directly.
    std::vector<int> no_think_header = s->asst_header;
    auto nt_ids = s->tok.encode(s->no_think_str);
    no_think_header.insert(no_think_header.end(), nt_ids.begin(), nt_ids.end());
    s->asst_header = std::move(no_think_header);
}

CCCC_LLM_API void llm_update_anchor(LlmHandle handle)
{
    auto* s = (LlmSession*)(handle);
    if (!s) { return; }
    // Expand anchor to include the entire non-system context (user turn +
    // model output + tool results injected so far).  This ensures a KV rebuild
    // will re-prefill everything up to this point rather than only the original
    // user task, so the model never loses injected tool responses.
    int non_sys = (int)s->ctx_ids.size() - (int)s->sys_ids.size();
    if (non_sys > 0)
    {
        s->first_turn_n = non_sys;
    }
}

CCCC_LLM_API void llm_destroy(LlmHandle handle)
{
    delete (LlmSession*)(handle);
}

CCCC_LLM_API void llm_reset(LlmHandle handle)
{
    auto* s = (LlmSession*)(handle);
    if (!s)
    {
        return;
    }
    s->ctx_ids = s->sys_ids;
    s->x_buf.clear();
    if (s->has_decode)
    {
        auto* net1 = s->mp.getNet(0, 1);
        if (net1)
        {
            net1->resetKVCache();
        }
    }
}

static void prepare_ctx(LlmSession* s, const std::string& user_input)
{
    std::vector<int> user_turn;
    user_turn.insert(user_turn.end(), s->user_prefix_ids.begin(), s->user_prefix_ids.end());
    auto content_ids = s->tok.encode(user_input);
    user_turn.insert(user_turn.end(), content_ids.begin(), content_ids.end());
    user_turn.insert(user_turn.end(), s->user_suffix_ids.begin(), s->user_suffix_ids.end());
    s->ctx_ids.insert(s->ctx_ids.end(), user_turn.begin(), user_turn.end());
    s->ctx_ids.insert(s->ctx_ids.end(), s->asst_header.begin(), s->asst_header.end());
    s->current_turn_n = (int)user_turn.size() + (int)s->asst_header.size();
    s->first_turn_n = s->current_turn_n;    // remember original task for KV-rebuild anchor

    // Prune very old context to avoid unbounded growth
    if (s->ctx_ids.size() > s->sys_ids.size() + (size_t)s->T * 2)
    {
        std::vector<int> trimmed = s->sys_ids;
        trimmed.insert(trimmed.end(), s->ctx_ids.end() - s->T * 2, s->ctx_ids.end());
        s->ctx_ids = std::move(trimmed);
    }
}

CCCC_LLM_API std::string llm_chat(LlmHandle handle, const std::string& user_input, int max_new_tokens, int show_thinking)
{
    auto* s = (LlmSession*)(handle);
    if (!s)
    {
        return {};
    }
    if (max_new_tokens <= 0)
    {
        max_new_tokens = 200;
    }

    prepare_ctx(s, user_input);
    std::string result = run_generate(*s, max_new_tokens, show_thinking != 0);
    s->ctx_ids.insert(s->ctx_ids.end(), s->im_end_nl.begin(), s->im_end_nl.end());
    return result;
}

CCCC_LLM_API void llm_chat_stream(LlmHandle handle, const std::string& user_input, int max_new_tokens, int show_thinking,
    LlmStreamCallback callback, void* userdata)
{
    auto* s = (LlmSession*)(handle);
    if (!s)
    {
        return;
    }
    if (max_new_tokens <= 0)
    {
        max_new_tokens = 200;
    }

    prepare_ctx(s, user_input);
    run_generate(*s, max_new_tokens, show_thinking != 0, callback, userdata);
    s->ctx_ids.insert(s->ctx_ids.end(), s->im_end_nl.begin(), s->im_end_nl.end());
}

CCCC_LLM_API void llm_inject_tool_response(LlmHandle handle, const std::string& content)
{
    auto* s = (LlmSession*)(handle);
    if (!s)
    {
        return;
    }
    // Qwen3 tool turn: <|im_start|>tool\n<tool_response>\n{content}\n</tool_response>\n<|im_end|>\n
    std::string tool_turn = "<|im_start|>tool\n<tool_response>\n" + content + "\n</tool_response>\n<|im_end|>\n";
    auto ids = s->tok.encode(tool_turn);
    s->ctx_ids.insert(s->ctx_ids.end(), ids.begin(), ids.end());
}

CCCC_LLM_API void llm_continue_stream(LlmHandle handle, int max_new_tokens, int show_thinking,
    LlmStreamCallback callback, void* userdata)
{
    auto* s = (LlmSession*)(handle);
    if (!s)
    {
        return;
    }
    if (max_new_tokens <= 0)
    {
        max_new_tokens = 200;
    }
    // Append assistant header only (no new user turn)
    s->ctx_ids.insert(s->ctx_ids.end(), s->asst_header.begin(), s->asst_header.end());
    // Restore original task as the KV-rebuild anchor so the model never loses
    // the user request even when a large tool response pushed pos beyond T_kv.
    s->current_turn_n = s->first_turn_n;
    run_generate(*s, max_new_tokens, show_thinking != 0, callback, userdata);
    s->ctx_ids.insert(s->ctx_ids.end(), s->im_end_nl.begin(), s->im_end_nl.end());
}
