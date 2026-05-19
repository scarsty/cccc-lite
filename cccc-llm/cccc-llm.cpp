#include "cccc-llm.h"
#include "INIReader.h"
#include "Log.h"
#include "MainProcess.h"
#include "Tokenizer.h"
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#define NOMINMAX
#include <windows.h>

// ── DLL entry point ───────────────────────────────────────────────────────────

BOOL WINAPI DllMain(HINSTANCE, DWORD, LPVOID) { return TRUE; }

static std::string make_turn(const std::string& role, const std::string& content)
{
    return "<|im_start|>" + role + "\n" + content + "<|im_end|>\n";
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
    int T = 0;
    int V = 0;
    int T_kv = 0;              // KV cache capacity (= T for current ini; used to guard RoPE/KV overflow)
    int current_turn_n = 0;    // tokens added by the latest prepare_ctx (user_turn + asst_header)
    bool half_output = false;
    bool has_decode = false;    // true when net group 1 (decode) is present
};

// ── generation ────────────────────────────────────────────────────────────────

static std::string run_generate(LlmSession& s, int max_new_tokens, bool show_thinking,
    LlmStreamCallback callback = nullptr, void* userdata = nullptr)
{
    const int EOS_IM_END = 151645;
    const int EOS_EOT = 151643;
    const int THINK_OPEN = 151667;
    const int THINK_CLOSE = 151668;

    std::string result;
    int think_depth = 0;

    // ── prefill ─────────────────────────────────────────────────────────────
    // 初始 prefill：将 ctx_ids 的最后 n 个 token 填入 x_buf（n ≤ T），计算 logit。
    int n = std::min((int)s.ctx_ids.size(), s.T);
    int offset = (int)s.ctx_ids.size() - n;

    // 按需扩容（不清零 y_buf，会被 testExternalData 覆写）
    s.x_buf.resize(s.T);
    s.y_buf.resize((size_t)s.V * s.T);

    // 左对齐：token 置于 x_buf[0..n-1]，后面补 0；logit 正确在 (n-1)*V 处
    std::fill(s.x_buf.begin(), s.x_buf.end(), 0.0f);
    for (int i = 0; i < n; i++)
    {
        s.x_buf[i] = (float)s.ctx_ids[offset + i];
    }

    s.mp.getNet()->resetKVCache();
    s.mp.testExternalData(s.x_buf.data(), nullptr, s.y_buf.data(), 1, 0, nullptr);

    if (s.has_decode)
    {
        auto* net_d = s.mp.getNet(0, 1);
        net_d->setKVCachePos(n);
    }

    // 从 prefill 最后一个位置提取 logit
    int next_id;
    if (s.half_output)
    {
        const cccc::half* h_logits = reinterpret_cast<const cccc::half*>(s.y_buf.data()) + (size_t)(n - 1) * s.V;
        next_id = (int)(std::max_element(h_logits, h_logits + s.V) - h_logits);
    }
    else
    {
        float* logits = s.y_buf.data() + (size_t)(n - 1) * s.V;
        next_id = (int)(std::max_element(logits, logits + s.V) - logits);
    }

    // ── decode loop ──────────────────────────────────────────────────────────
    int pos = n;                                       // 绝对 KV cache 位置
    const int ctx_at_start = (int)s.ctx_ids.size();    // 生成开始前的 ctx 长度
    int rebuild_count = 0;

    for (int step = 0; step < max_new_tokens; step++)
    {
        if (next_id == EOS_IM_END || next_id == EOS_EOT)
        {
            break;
        }

        s.ctx_ids.push_back(next_id);

        bool emit_piece = true;
        if (next_id == THINK_OPEN)
        {
            think_depth++;
            emit_piece = show_thinking;
        }
        else if (next_id == THINK_CLOSE)
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
            int turn_start = ctx_at_start - turn_n;
            for (int i = 0; i < turn_n && sys_n + i < s.T; i++)
            {
                s.x_buf[sys_n + i] = (float)s.ctx_ids[turn_start + i];
            }

            // 3. gen_tail：根据思考状态插入标记，帮助模型正确延续
            int tail_n;
            if (think_depth > 0)
            {
                // 仍在思考中：在 gen_tail 前插入 <think>，保持模型思考状态
                s.x_buf[anchor_n] = (float)THINK_OPEN;
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
                // 思考已结束：在 max_tail 窗口内查找最后一个 </think>，锚定到它
                int search_start = std::max(ctx_at_start, (int)s.ctx_ids.size() - max_tail);
                int think_close_idx = -1;
                for (int i = (int)s.ctx_ids.size() - 1; i >= search_start; i--)
                {
                    if (s.ctx_ids[i] == THINK_CLOSE)
                    {
                        think_close_idx = i;
                        break;
                    }
                }

                if (think_close_idx >= 0)
                {
                    // </think> 在窗口内：从它开始包含到末尾
                    int actual_tail = (int)s.ctx_ids.size() - think_close_idx;
                    for (int i = 0; i < actual_tail; i++)
                    {
                        s.x_buf[anchor_n + i] = (float)s.ctx_ids[think_close_idx + i];
                    }
                    tail_n = actual_tail;
                }
                else
                {
                    // </think> 已超出窗口：插入 <think></think> 标记，告知模型思考已完成
                    s.x_buf[anchor_n] = (float)THINK_OPEN;
                    s.x_buf[anchor_n + 1] = (float)THINK_CLOSE;
                    int actual_tail = std::min(gen_so_far, max_tail - 2);
                    int gen_tail_start = (int)s.ctx_ids.size() - actual_tail;
                    for (int i = 0; i < actual_tail; i++)
                    {
                        s.x_buf[anchor_n + 2 + i] = (float)s.ctx_ids[gen_tail_start + i];
                    }
                    tail_n = 2 + actual_tail;
                }
            }

            int rebuild_n = anchor_n + tail_n;

            s.mp.rebuildKVCache(0, 1, s.x_buf.data(), rebuild_n, s.y_buf.data());

            if (s.half_output)
            {
                const cccc::half* h = reinterpret_cast<const cccc::half*>(s.y_buf.data()) + (size_t)(rebuild_n - 1) * s.V;
                next_id = (int)(std::max_element(h, h + s.V) - h);
            }
            else
            {
                float* logits = s.y_buf.data() + (size_t)(rebuild_n - 1) * s.V;
                next_id = (int)(std::max_element(logits, logits + s.V) - logits);
            }

            pos = rebuild_n;
            cccc::LOG_ERR("[llm] KV-cache rebuild #{}: {} tokens (sys={} turn={} tail={} think_depth={}), pos={}\n",
                rebuild_count, rebuild_n, sys_n, turn_n, tail_n, think_depth, pos);
            continue;
        }

        // Decode 一步
        if (s.has_decode)
        {
            auto* net_d = s.mp.getNet(0, 1);
            net_d->setRopeOffset(pos);
            net_d->setAttentionOffset(pos);

            s.x_dec[0] = (float)next_id;
            s.mp.testExternalData(s.x_dec.data(), nullptr, s.y_dec.data(), 1, 0, nullptr, 1);

            if (s.half_output)
            {
                const cccc::half* h_logits = reinterpret_cast<const cccc::half*>(s.y_dec.data());
                next_id = (int)(std::max_element(h_logits, h_logits + s.V) - h_logits);
            }
            else
            {
                next_id = (int)(std::max_element(s.y_dec.begin(), s.y_dec.begin() + s.V) - s.y_dec.begin());
            }
            pos++;
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
            if (s.half_output)
            {
                const cccc::half* h_logits = reinterpret_cast<const cccc::half*>(s.y_buf.data()) + (size_t)(nc - 1) * s.V;
                next_id = (int)(std::max_element(h_logits, h_logits + s.V) - h_logits);
            }
            else
            {
                float* logits = s.y_buf.data() + (size_t)(nc - 1) * s.V;
                next_id = (int)(std::max_element(logits, logits + s.V) - logits);
            }
        }
    }

    return result;
}

// ── exported functions ────────────────────────────────────────────────────────

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

        // 用 INIReaderNoUnderline（纯头文件模板，无 DLL 导出问题）读取 load_file
        INIReaderNoUnderline ini_reader;
        ini_reader.loadString(ini_str);
        std::string load_file = ini_reader.getString("train", "load_file", "");
        if (!load_file.empty() && !std::filesystem::path(load_file).is_absolute())
        {
            ini_str += std::format("\n[train]\nload_file={}\n", (ini_dir / load_file).lexically_normal().string());
        }
        ini_str += "\n[train]\noutput_log=0\nneed_free_mem=0\n";
        if (s->mp.init(ini_str) != 0)
        {
            cccc::LOG_ERR("llm_init: mp.init() failed\n");
            delete s;
            return nullptr;
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
    s->half_output = (net0->getA().getDataType() == cccc::DataType::HALF);

    // 若 ini 中 net_num=2 则 group 1（decode 网络）已被一并加载，权重/KV-cache 已共享
    auto* net1 = s->mp.getNet(0, 1);
    if (net1 != nullptr)
    {
        net1->getGpu()->setActivePhase(cccc::ACTIVE_PHASE_TEST);
        s->x_dec.resize(1, 0.0f);
        s->y_dec.resize(s->V, 0.0f);
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

    if (!system_prompt.empty())
    {
        s->sys_ids = s->tok.encode(make_turn("system", system_prompt));
    }
    s->ctx_ids = s->sys_ids;
    s->asst_header = s->tok.encode("<|im_start|>assistant\n");
    s->im_end_nl = s->tok.encode("<|im_end|>\n");

    return s;
}

CCCC_LLM_API void llm_destroy(LlmHandle handle)
{
    delete static_cast<LlmSession*>(handle);
}

CCCC_LLM_API void llm_reset(LlmHandle handle)
{
    auto* s = static_cast<LlmSession*>(handle);
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
    std::vector<int> user_turn = s->tok.encode(make_turn("user", user_input));
    s->ctx_ids.insert(s->ctx_ids.end(), user_turn.begin(), user_turn.end());
    s->ctx_ids.insert(s->ctx_ids.end(), s->asst_header.begin(), s->asst_header.end());
    s->current_turn_n = (int)user_turn.size() + (int)s->asst_header.size();

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
    auto* s = static_cast<LlmSession*>(handle);
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
    auto* s = static_cast<LlmSession*>(handle);
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
