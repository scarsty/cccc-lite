// cccc-sd.cpp  — z-image-turbo text-to-image pipeline using cccc framework
// First-pass implementation: simplified (no adaLN) to confirm pipeline structure.
// Models: text_encoder(35-layer), cap_embedder, t_embedder, x_embedder,
//         context_refiner(2-block), noise_refiner(2-block), unified(30-block),
//         final_layer, vae.

#define CCCC_LLM_BUILDING
#include "cccc-llm.h"
#include "GpuControl.h"
#include "Log.h"
#include "MainProcess.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <random>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

using namespace cccc;

// ============================================================
// Helpers
// ============================================================
static std::string require_ini_file(const std::string& path, const char* stage_name)
{
    std::ifstream f(path);
    if (!f)
    {
        LOG_ERR("Missing ini for {}: {}\n", stage_name, path);
        LOG_ERR("Please provide/edit this ini directly (makeini has been removed).\n");
        return "";
    }
    return path;
}

// Log current GPU VRAM usage.
static void log_vram(cccc::GpuControl* gpu)
{
    if (!gpu)
    {
        return;
    }
    size_t free_b = 0, total_b = 0;
    gpu->getFreeMemory(free_b, total_b);
    const double free_mb = free_b / (1024.0 * 1024.0);
    const double total_mb = total_b / (1024.0 * 1024.0);
    const double used_mb = total_mb - free_mb;
    LOG("VRAM: used={:.0f} MB  free={:.0f} MB  total={:.0f} MB\n", used_mb, free_mb, total_mb);
}

static bool init_model(MainProcess& mp, const std::string& ini, const std::string& net_parameter = "")
{
    std::error_code ec;
    const auto old_cwd = std::filesystem::current_path(ec);
    const bool has_old_cwd = !ec;
    const auto ini_path = std::filesystem::path(ini);
    const auto ini_dir = ini_path.parent_path();
    if (!ini_dir.empty())
    {
        std::filesystem::current_path(ini_dir, ec);
        if (ec)
        {
            LOG_ERR("Cannot set working directory to {}: {}\n", ini_dir.string(), ec.message());
            return false;
        }
    }

    auto* opt = mp.getOption();
    opt->clear();
    opt->loadFile(ini_dir.empty() ? ini : ini_path.filename().string());
    if (!net_parameter.empty())
    {
        opt->setKey("net", "parameter", net_parameter);
    }
    opt->setOutput(false);
    const int init_ret = mp.init("");

    if (has_old_cwd)
    {
        std::filesystem::current_path(old_cwd, ec);
    }

    if (init_ret != 0)
    {
        return false;
    }
    mp.getNet()->getGpu()->setActivePhase(ACTIVE_PHASE_TEST);
    return true;
}

static bool run_model(MainProcess& mp, std::vector<float>& x, std::vector<float>& y)
{
    // MainProcess::initNets sets a global current dtype. When multiple models are
    // kept alive simultaneously, later model loads can overwrite this global state.
    // Restore this model's configured dtype before each inference.
    Matrix::setCurrentDataType(mp.getOption()->getEnum<DataType>("train", "data_type", DataType::FLOAT));

    auto* net = mp.getNet();
    if (!net)
    {
        LOG_ERR("[run_model] getNet() returned nullptr!\n");
        return false;
    }
    auto x_dt = net->getX().getDataType();
    auto a_dt = net->getA().getDataType();
    int64_t n_out = net->getA().getDataSize();
    int64_t n_in = net->getX().getDataSize();
    LOG("[run_model] x_dt={} a_dt={} n_in={} n_out={} x.size={} y.size={}\n",
        (int)x_dt, (int)a_dt, n_in, n_out, x.size(), y.size());

    if (n_out <= 0)
    {
        LOG_ERR("[run_model] n_out={} is invalid\n", n_out);
        return false;
    }
    if ((int64_t)y.size() < n_out)
    {
        LOG_ERR("[run_model] y.size()={} < n_out={}, resizing\n", y.size(), n_out);
        y.resize((size_t)n_out, 0.f);
    }

    if (a_dt == cccc::DataType::HALF)
    {
        std::vector<cccc::half> yh(n_out);
        if (x_dt == cccc::DataType::HALF)
        {
            std::vector<cccc::half> xh(x.size());
            for (size_t i = 0; i < x.size(); i++)
            {
                xh[i] = (cccc::half)(x[i]);
            }
            mp.testExternalData(xh.data(), nullptr, yh.data(), 1, 0, nullptr);
        }
        else
        {
            // x is float (e.g. token IDs in text encoder), a is half
            mp.testExternalData(x.data(), nullptr, yh.data(), 1, 0, nullptr);
        }
        for (int64_t i = 0; i < n_out; i++)
        {
            float fv = (float)(yh[i]);
            // FP16 can legitimately saturate to ±inf; preserve sign direction
            y[i] = std::isfinite(fv) ? fv : std::copysign(65504.f, fv);
        }
    }
    else if (a_dt == cccc::DataType::BFLOAT16)
    {
        // bfloat16 network: convert float↔bfloat16 for I/O
        std::vector<cccc::bfloat16> yb(n_out);
        LOG("[run_model] BF16 path: yb.size={} yb.data={}\n", yb.size(), (void*)yb.data());
        if (x_dt == cccc::DataType::BFLOAT16)
        {
            std::vector<cccc::bfloat16> xb(x.size());
            for (size_t i = 0; i < x.size(); i++)
            {
                xb[i] = cccc::bfloat16(x[i]);
            }
            LOG("[run_model] BF16+BF16: calling testExternalData\n");
            mp.testExternalData(xb.data(), nullptr, yb.data(), 1, 0, nullptr);
        }
        else
        {
            // x is float (e.g. token IDs), a is bfloat16
            LOG("[run_model] FLOAT+BF16: calling testExternalData\n");
            mp.testExternalData(x.data(), nullptr, yb.data(), 1, 0, nullptr);
        }
        LOG("[run_model] BF16 path: testExternalData done, copying {} elements\n", n_out);
        {
            int n_nan = 0, n_inf = 0, n_zero = 0, n_fin = 0;
            int show = std::min((int64_t)8, n_out);
            LOG("[run_model] BF16 raw yb[0:{}]:", show);
            for (int i = 0; i < show; i++)
            {
                float fv = (float)(yb[i]);
                LOG(" {:.5g}", fv);
            }
            LOG("\n");
            for (int64_t i = 0; i < n_out; i++)
            {
                float fv = (float)(yb[i]);
                if (std::isnan(fv))
                {
                    n_nan++;
                }
                else if (std::isinf(fv))
                {
                    n_inf++;
                }
                else if (fv == 0.f)
                {
                    n_zero++;
                }
                else
                {
                    n_fin++;
                }
            }
            LOG("[run_model] BF16 stats: nan={} inf={} zero={} finite_nonzero={}\n", n_nan, n_inf, n_zero, n_fin);
        }
        for (int64_t i = 0; i < n_out; i++)
        {
            float fv = (float)(yb[i]);
            // BF16 inf only occurs from div-by-zero or sqrt(negative); replace with 0
            y[i] = std::isfinite(fv) ? fv : 0.f;
        }
    }
    else
    {
        // Both float
        mp.testExternalData(x.data(), nullptr, y.data(), 1, 0, nullptr);
    }
    LOG("[run_model] done\n");
    return true;
}

class BPETokenizer
{
public:
    static const int ID_EOT = 151643;
    static const int ID_IM_START = 151644;
    static const int ID_IM_END = 151645;

    bool load(const std::string& vocab_path, const std::string& merges_path)
    {
        build_byte_enc();
        {
            std::ifstream f(vocab_path);
            if (!f)
            {
                LOG_ERR("Cannot open vocab: {}\n", vocab_path);
                return false;
            }
            std::string line;
            int id = 0;
            while (std::getline(f, line))
            {
                if (!line.empty() && line.back() == '\r')
                {
                    line.pop_back();
                }
                vocab_[line] = id++;
            }
        }
        vocab_["<|endoftext|>"] = ID_EOT;
        vocab_["<|im_start|>"] = ID_IM_START;
        vocab_["<|im_end|>"] = ID_IM_END;
        {
            std::ifstream f(merges_path);
            if (!f)
            {
                LOG_ERR("Cannot open merges: {}\n", merges_path);
                return false;
            }
            std::string line;
            int rank = 0;
            while (std::getline(f, line))
            {
                if (!line.empty() && line.back() == '\r')
                {
                    line.pop_back();
                }
                if (line.empty() || line[0] == '#')
                {
                    continue;
                }
                auto sp = line.find(' ');
                if (sp == std::string::npos)
                {
                    continue;
                }
                merges_rank_[line.substr(0, sp) + " " + line.substr(sp + 1)] = rank++;
            }
        }
        LOG("BPE loaded: vocab={} merges={}\n", vocab_.size(), merges_rank_.size());
        return true;
    }

    std::vector<int> encode(const std::string& prompt)
    {
        std::vector<int> ids;
        ids.push_back(ID_IM_START);
        append_bpe(ids, "user");
        append_bpe(ids, "\n");
        append_bpe(ids, prompt);
        ids.push_back(ID_IM_END);
        append_bpe(ids, "\n");
        ids.push_back(ID_IM_START);
        append_bpe(ids, "assistant");
        append_bpe(ids, "\n");
        return ids;
    }

private:
    std::unordered_map<std::string, int> vocab_;
    std::unordered_map<std::string, int> merges_rank_;
    std::string byte_enc_[256];

    static std::string cp_to_utf8(uint32_t cp)
    {
        char b[5] = {};
        if (cp < 0x80)
        {
            b[0] = (char)cp;
            return std::string(b, 1);
        }
        if (cp < 0x800)
        {
            b[0] = (char)(0xC0 | (cp >> 6));
            b[1] = (char)(0x80 | (cp & 0x3F));
            return std::string(b, 2);
        }
        if (cp < 0x10000)
        {
            b[0] = (char)(0xE0 | (cp >> 12));
            b[1] = (char)(0x80 | ((cp >> 6) & 0x3F));
            b[2] = (char)(0x80 | (cp & 0x3F));
            return std::string(b, 3);
        }
        b[0] = (char)(0xF0 | (cp >> 18));
        b[1] = (char)(0x80 | ((cp >> 12) & 0x3F));
        b[2] = (char)(0x80 | ((cp >> 6) & 0x3F));
        b[3] = (char)(0x80 | (cp & 0x3F));
        return std::string(b, 4);
    }

    void build_byte_enc()
    {
        std::vector<int> bs, cs;
        for (int b = '!'; b <= '~'; b++)
        {
            bs.push_back(b);
            cs.push_back(b);
        }
        for (int b = 0xA1; b <= 0xAC; b++)
        {
            bs.push_back(b);
            cs.push_back(b);
        }
        for (int b = 0xAE; b <= 0xFF; b++)
        {
            bs.push_back(b);
            cs.push_back(b);
        }
        int n = 0;
        for (int b = 0; b < 256; b++)
        {
            if (std::find(bs.begin(), bs.end(), b) == bs.end())
            {
                bs.push_back(b);
                cs.push_back(256 + n++);
            }
        }
        for (int i = 0; i < 256; i++)
        {
            byte_enc_[bs[i]] = cp_to_utf8((uint32_t)cs[i]);
        }
    }

    std::vector<std::string> apply_bpe(std::vector<std::string> word)
    {
        if (word.size() <= 1)
        {
            return word;
        }
        while (true)
        {
            int best_rank = INT_MAX, best_pos = -1;
            for (int i = 0; i + 1 < (int)word.size(); i++)
            {
                auto it = merges_rank_.find(word[i] + " " + word[i + 1]);
                if (it != merges_rank_.end() && it->second < best_rank)
                {
                    best_rank = it->second;
                    best_pos = i;
                }
            }
            if (best_pos < 0)
            {
                break;
            }
            word[best_pos] += word[best_pos + 1];
            word.erase(word.begin() + best_pos + 1);
        }
        return word;
    }

    void append_bpe(std::vector<int>& ids, const std::string& text)
    {
        if (text.empty())
        {
            return;
        }
        std::vector<std::string> syms;
        for (unsigned char c : text)
        {
            syms.push_back(byte_enc_[c]);
        }
        for (auto& tok : apply_bpe(syms))
        {
            auto it = vocab_.find(tok);
            if (it != vocab_.end())
            {
                ids.push_back(it->second);
            }
            else
            {
                for (unsigned char c : tok)
                {
                    auto it2 = vocab_.find(byte_enc_[c]);
                    if (it2 != vocab_.end())
                    {
                        ids.push_back(it2->second);
                    }
                }
            }
        }
    }
};

// ============================================================
// 3-D RoPE tables for z-image-turbo
// theta=256, axes=[temporal(32), height(48), width(48)]
// Half-dims per axis: [16, 24, 24] → total 64 = hd/2
// ============================================================
static void rope3d(const std::vector<std::array<int, 3>>& ids,
    std::vector<float>& cos_out, std::vector<float>& sin_out)
{
    const float theta = 256.0f;
    const int adims[3] = { 32, 48, 48 };
    const int half_dims[3] = { 16, 24, 24 };
    int S = (int)ids.size();
    cos_out.assign(S * 64, 0.f);
    sin_out.assign(S * 64, 0.f);
    for (int s = 0; s < S; s++)
    {
        int off = 0;
        for (int ax = 0; ax < 3; ax++)
        {
            int pos = ids[s][ax];
            int hd = half_dims[ax];
            int ad = adims[ax];
            for (int i = 0; i < hd; i++)
            {
                float inv = 1.0f / std::pow(theta, (float)(i * 2) / (float)ad);
                float angle = (float)pos * inv;
                cos_out[s * 64 + off + i] = std::cos(angle);
                sin_out[s * 64 + off + i] = std::sin(angle);
            }
            off += hd;
        }
    }
}
// Image patches: temporal position = T+1 (fixed per inference), spatial = (py, px)
static void gen_img_rope(int P, int T, int npw, int nph,
    std::vector<float>& cos_img, std::vector<float>& sin_img)
{
    int start_t = T + 1;
    std::vector<std::array<int, 3>> ids(P);
    for (int py = 0; py < nph; py++)
    {
        for (int px = 0; px < npw; px++)
        {
            ids[py * npw + px][0] = start_t;
            ids[py * npw + px][1] = py;
            ids[py * npw + px][2] = px;
        }
    }
    rope3d(ids, cos_img, sin_img);
}
// Text tokens: temporal position = 1..T, height=0, width=0
static void gen_txt_rope(int T,
    std::vector<float>& cos_txt, std::vector<float>& sin_txt)
{
    std::vector<std::array<int, 3>> ids(T);
    for (int i = 0; i < T; i++)
    {
        ids[i][0] = 1 + i;
        ids[i][1] = 0;
        ids[i][2] = 0;
    }
    rope3d(ids, cos_txt, sin_txt);
}

// ============================================================
// Scheduler: Flow Matching Euler
// ============================================================
struct Scheduler
{
    int steps;
    std::vector<float> sigmas;
    std::vector<float> timesteps;

    void init(int steps_, float shift = 3.0f)
    {
        steps = steps_;
        sigmas.resize(steps + 1);
        timesteps.resize(steps);
        for (int i = 0; i < steps; i++)
        {
            float alpha = 1.0f - (float)i / (steps - 1);
            float sigma = shift * alpha / (1.0f + (shift - 1.0f) * alpha);
            sigmas[i] = sigma;
            timesteps[i] = 1.0f - sigma;
        }
        sigmas[steps] = 0.0f;
    }
    float dt(int step) const { return sigmas[step + 1] - sigmas[step]; }
    float t_val(int step) const { return timesteps[step] * 1000.0f; }
};

// ============================================================
// Patchify / Unpatchify  (ps=2, HWC layout for latent)
// ============================================================
static void patchify(const std::vector<float>& lat, int lH, int lW,
    std::vector<float>& px)
{
    // lat: (lH,lW,16) HWC  →  px: (P,64) row-major
    const int C = 16, ps = 2, ph = lH / ps, pw = lW / ps, P = ph * pw;
    px.resize((size_t)P * 64);
    for (int py = 0; py < ph; py++)
    {
        for (int px_ = 0; px_ < pw; px_++)
        {
            int idx = py * pw + px_;
            // spatial-major: [TL_c0..c15, TR_c0..c15, BL_c0..c15, BR_c0..c15]
            // matches PyTorch rearrange '(p1 p2 c)'
            for (int c = 0; c < C; c++)
            {
                px[(size_t)idx * 64 + 0 * C + c] = lat[(py * 2 + 0) * lW * C + (px_ * 2 + 0) * C + c];    // TL
                px[(size_t)idx * 64 + 1 * C + c] = lat[(py * 2 + 0) * lW * C + (px_ * 2 + 1) * C + c];    // TR
                px[(size_t)idx * 64 + 2 * C + c] = lat[(py * 2 + 1) * lW * C + (px_ * 2 + 0) * C + c];    // BL
                px[(size_t)idx * 64 + 3 * C + c] = lat[(py * 2 + 1) * lW * C + (px_ * 2 + 1) * C + c];    // BR
            }
        }
    }
}

static void unpatchify(const std::vector<float>& px, int lH, int lW,
    std::vector<float>& lat)
{
    const int C = 16, ps = 2, ph = lH / ps, pw = lW / ps;
    lat.resize((size_t)lH * lW * C);
    for (int py = 0; py < ph; py++)
    {
        for (int px_ = 0; px_ < pw; px_++)
        {
            int idx = py * pw + px_;
            // spatial-major: mirrors patchify
            for (int c = 0; c < C; c++)
            {
                lat[(py * 2 + 0) * lW * C + (px_ * 2 + 0) * C + c] = px[(size_t)idx * 64 + 0 * C + c];    // TL
                lat[(py * 2 + 0) * lW * C + (px_ * 2 + 1) * C + c] = px[(size_t)idx * 64 + 1 * C + c];    // TR
                lat[(py * 2 + 1) * lW * C + (px_ * 2 + 0) * C + c] = px[(size_t)idx * 64 + 2 * C + c];    // BL
                lat[(py * 2 + 1) * lW * C + (px_ * 2 + 1) * C + c] = px[(size_t)idx * 64 + 3 * C + c];    // BR
            }
        }
    }
}

// ============================================================
// Save PNG (via stb_image_write)
// ============================================================
// rgb: NCHW (1,3,H,W) float32, values in [-1, 1]
static bool save_image(const std::string& path, const std::vector<float>& rgb,
    int W, int H)
{
    // force .png extension
    std::string p = path;
    auto dot = p.rfind('.');
    if (dot != std::string::npos)
    {
        p = p.substr(0, dot);
    }
    p += ".png";

    // convert NCHW float [-1,1] → HWC uint8 [0,255]
    std::vector<uint8_t> img((size_t)H * W * 3);
    for (int h = 0; h < H; h++)
    {
        for (int w = 0; w < W; w++)
        {
            for (int c = 0; c < 3; c++)
            {
                float v = (rgb[(size_t)c * H * W + h * W + w] + 1.0f) * 127.5f;
                img[(size_t)(h * W + w) * 3 + c] = (uint8_t)std::max(0.0f, std::min(255.0f, v));
            }
        }
    }

    if (!stbi_write_png(p.c_str(), W, H, 3, img.data(), W * 3))
    {
        LOG_ERR("Cannot write PNG: {}\n", p);
        return false;
    }
    LOG("Saved: {}\n", p);
    return true;
}

// ============================================================
// Pipeline
// ============================================================
class ZImagePipeline
{
public:
    std::string model_dir;
    BPETokenizer tok;

    bool init(const std::string& dir)
    {
        model_dir = dir;
        std::replace(model_dir.begin(), model_dir.end(), '\\', '/');
        return tok.load(model_dir + "/vocab.txt", model_dir + "/merges.txt");
    }

    int generate(const std::string& prompt, const std::string& out_path,
        int steps, int width, int height, int seed,
        SdProgressCallback cb, void* ud)
    {
        auto T0 = std::chrono::high_resolution_clock::now();
        auto elapsed = [&]()
        {
            return std::chrono::duration<float>(
                std::chrono::high_resolution_clock::now() - T0)
                .count();
        };

        int lH = height / 8, lW = width / 8, ph = lH / 2, pw = lW / 2, P = ph * pw;
        constexpr int DD = 3968;

        // [1] Tokenize
        LOG("[1] Tokenize '{}'\n", prompt);
        auto ids = tok.encode(prompt);
        int T = (int)ids.size();
        LOG("    T={}\n", T);
        // Precompute text RoPE (needed for ctx refiner and denoising)
        std::vector<float> cos_txt, sin_txt;
        gen_txt_rope(T, cos_txt, sin_txt);

        // [2-4] Text conditioning stages (text encoder → cap embedder → ctx refiner)
        std::vector<float> cap_ref((size_t)3840 * T, 0.0f);
        {
            // [2] Text encoder
            LOG("[2] Text encoder (loading ~7.5GB)...\n");
            std::vector<float> cap((size_t)2560 * T, 0.0f);
            {
                MainProcess m;
                auto ini = require_ini_file(model_dir + "/net_texenc.ini", "text encoder");
                if (ini.empty())
                {
                    return -2;
                }
                if (!init_model(m, ini, "T=" + std::to_string(T) + ";"))
                {
                    LOG_ERR("text_encoder failed\n");
                    return -2;
                }
                std::vector<float> ids_f(T);
                for (int i = 0; i < T; i++)
                {
                    ids_f[i] = (float)(ids[i]);
                }
                run_model(m, ids_f, cap);
                log_vram(m.getNet()->getGpu());
                LOG("    cap range=[{:.4f},{:.4f}]  t={:.1f}s\n",
                    *std::min_element(cap.begin(), cap.end()),
                    *std::max_element(cap.begin(), cap.end()), elapsed());
            }

            // [3] Cap embedder
            LOG("[3] Cap embedder...\n");
            std::vector<float> cap_emb((size_t)3840 * T, 0.0f);
            {
                MainProcess m;
                auto ini = require_ini_file(model_dir + "/net_capemb.ini", "cap embedder");
                if (ini.empty())
                {
                    return -3;
                }
                if (!init_model(m, ini, "T=" + std::to_string(T) + ";"))
                {
                    LOG_ERR("cap_emb failed\n");
                    return -3;
                }
                run_model(m, cap, cap_emb);
            }

            // [4] Context refiner
            LOG("[4] Context refiner...\n");
            {
                MainProcess m;
                auto ini = require_ini_file(model_dir + "/net_ctx.ini", "context refiner");
                if (ini.empty())
                {
                    return -4;
                }
                if (!init_model(m, ini, "T=" + std::to_string(T) + ";"))
                {
                    LOG_ERR("ctx failed\n");
                    return -4;
                }
                // Pack DD=3968 wide: [cap_emb(3840)|cos_txt(64)|sin_txt(64)] per token
                std::vector<float> ctx_in((size_t)DD * T, 0.0f);
                for (int s = 0; s < T; s++)
                {
                    std::memcpy(&ctx_in[(size_t)s * DD], &cap_emb[(size_t)s * 3840], 3840 * 4);
                    std::memcpy(&ctx_in[(size_t)s * DD + 3840], &cos_txt[(size_t)s * 64], 64 * 4);
                    std::memcpy(&ctx_in[(size_t)s * DD + 3904], &sin_txt[(size_t)s * 64], 64 * 4);
                }
                run_model(m, ctx_in, cap_ref);
            }
        }

        // [5] T-embedder (all steps)
        LOG("[5] T-embedder...\n");
        Scheduler sched;
        sched.init(steps, 3.0f);
        std::vector<std::vector<float>> t_embs(steps, std::vector<float>(256, 0.0f));
        {
            MainProcess m;
            auto ini = require_ini_file(model_dir + "/net_temb.ini", "t-embedder");
            if (ini.empty())
            {
                return -5;
            }
            if (!init_model(m, ini))
            {
                LOG_ERR("t_emb failed\n");
                return -5;
            }
            for (int i = 0; i < steps; i++)
            {
                std::vector<float> tin = { sched.t_val(i) };
                run_model(m, tin, t_embs[i]);
            }
        }

        // Init latent & patches
        LOG("Init latent {}x{}...\n", lH, lW);
        std::vector<float> lat_hwc((size_t)lH * lW * 16);
        {
            std::mt19937 rng(seed < 0 ? std::random_device{}() : (uint32_t)seed);
            std::normal_distribution<float> nd;
            for (auto& v : lat_hwc)
            {
                v = nd(rng);
            }
        }
        std::vector<float> x_px;
        patchify(lat_hwc, lH, lW, x_px);    // (P,64)

        LOG("Loading denoising model configs...\n");
        const std::string xemb_ini = require_ini_file(model_dir + "/net_xemb.ini", "x-embedder");
        const std::string noise_ini = require_ini_file(model_dir + "/net_noise.ini", "noise refiner");
        const std::string unified_ini = require_ini_file(model_dir + "/net_unified.ini", "unified transformer");
        const std::string final_ini = require_ini_file(model_dir + "/net_final.ini", "final layer");
        if (xemb_ini.empty() || noise_ini.empty() || unified_ini.empty() || final_ini.empty())
        {
            return -7;
        }

        MainProcess mxemb, mnoise, muni, mfinal;
        LOG("[6-pre] preload 4 denoising models once (default).\n");
        if (!init_model(mxemb, xemb_ini, "P=" + std::to_string(P) + ";"))
        {
            LOG_ERR("x_emb preload failed\n");
            return -7;
        }
        log_vram(mxemb.getNet()->getGpu());
        if (!init_model(mnoise, noise_ini, "P=" + std::to_string(P) + ";"))
        {
            LOG_ERR("noise preload failed\n");
            return -7;
        }
        log_vram(mnoise.getNet()->getGpu());
        if (!init_model(muni, unified_ini, "SEQ=" + std::to_string(P + T) + ";"))
        {
            LOG_ERR("unified preload failed\n");
            return -7;
        }
        log_vram(muni.getNet()->getGpu());
        if (!init_model(mfinal, final_ini, "SEQ=" + std::to_string(P + T) + "; P=" + std::to_string(P) + ";"))
        {
            LOG_ERR("final preload failed\n");
            return -7;
        }
        log_vram(mfinal.getNet()->getGpu());

        // [6] Denoise loop
        LOG("[6] Denoising ({} steps)...\n", steps);
        std::vector<float> x_emb((size_t)3840 * P), x_ref((size_t)3840 * P);
        // Precompute image RoPE (text RoPE already computed at step [1])
        std::vector<float> cos_img, sin_img;
        gen_img_rope(P, T, pw, ph, cos_img, sin_img);
        // noise_in: DD*(P+1): [emb(3840)|cos(64)|sin(64)] per img token + t_emb slot
        // seq: DD*(SEQ+1): same layout; final_in: D*(SEQ+1)
        std::vector<float> noise_in((size_t)DD * (P + 1), 0.0f);
        std::vector<float> seq((size_t)DD * (P + T + 1), 0.0f), uni_out((size_t)3840 * (P + T));
        std::vector<float> final_in((size_t)3840 * (P + T + 1), 0.0f);
        std::vector<float> vel((size_t)64 * P);

        for (int step = 0; step < steps; step++)
        {
            auto ts = std::chrono::high_resolution_clock::now();

            // --- x-embedder ---
            if (!run_model(mxemb, x_px, x_emb))
            {
                LOG_ERR("x_emb run failed at step {}\n", step);
                return -7;
            }

            // Pack noise_in as DD*(P+1): [emb|cos|sin] per token + t_emb slot
            for (int s = 0; s < P; s++)
            {
                std::memcpy(&noise_in[(size_t)s * DD], &x_emb[(size_t)s * 3840], 3840 * 4);
                std::memcpy(&noise_in[(size_t)s * DD + 3840], &cos_img[(size_t)s * 64], 64 * 4);
                std::memcpy(&noise_in[(size_t)s * DD + 3904], &sin_img[(size_t)s * 64], 64 * 4);
            }
            std::memset(&noise_in[(size_t)P * DD], 0, DD * 4);
            std::memcpy(&noise_in[(size_t)P * DD], t_embs[step].data(), 256 * 4);

            // --- noise refiner ---
            if (!run_model(mnoise, noise_in, x_ref))
            {
                LOG_ERR("noise run failed at step {}\n", step);
                return -7;
            }

            // Pack seq as DD=3968 wide: [emb(3840)|cos(64)|sin(64)] per token
            for (int s = 0; s < P; s++)
            {
                std::memcpy(&seq[(size_t)s * DD], &x_ref[(size_t)s * 3840], 3840 * 4);
                std::memcpy(&seq[(size_t)s * DD + 3840], &cos_img[(size_t)s * 64], 64 * 4);
                std::memcpy(&seq[(size_t)s * DD + 3904], &sin_img[(size_t)s * 64], 64 * 4);
            }
            for (int s = 0; s < T; s++)
            {
                std::memcpy(&seq[(size_t)(P + s) * DD], &cap_ref[(size_t)s * 3840], 3840 * 4);
                std::memcpy(&seq[(size_t)(P + s) * DD + 3840], &cos_txt[(size_t)s * 64], 64 * 4);
                std::memcpy(&seq[(size_t)(P + s) * DD + 3904], &sin_txt[(size_t)s * 64], 64 * 4);
            }
            // t_emb slot: zero then write 256-dim t_emb
            std::memset(&seq[(size_t)(P + T) * DD], 0, DD * 4);
            std::memcpy(&seq[(size_t)(P + T) * DD], t_embs[step].data(), 256 * 4);

            // --- unified transformer ---
            if (!run_model(muni, seq, uni_out))
            {
                LOG_ERR("unified run failed at step {}\n", step);
                return -7;
            }

            // Build final_in: unified output (P+T tokens) + t_emb in last slot
            std::memcpy(final_in.data(), uni_out.data(), (size_t)3840 * (P + T) * 4);
            std::memset(final_in.data() + (size_t)3840 * (P + T), 0, (size_t)3840 * 4);    // zero pad
            std::memcpy(final_in.data() + (size_t)3840 * (P + T), t_embs[step].data(), 256 * 4);

            // --- final layer ---
            if (!run_model(mfinal, final_in, vel))
            {
                LOG_ERR("final run failed at step {}\n", step);
                return -7;
            }
            // Report per-step tensor statistics.
            {
                auto stat = [](const std::vector<float>& v, const char* name, int step)
                {
                    float mn = v[0], mx = v[0], sm = 0;
                    for (auto x : v)
                    {
                        if (x < mn) { mn = x; }
                        if (x > mx) { mx = x; }
                        sm += x;
                    }
                    LOG("    s{} {}: min={:.4f} max={:.4f} mean={:.4f}\n", step, name, mn, mx, sm / v.size());
                };
                stat(vel, "vel  ", step);
            }
            float dt = sched.dt(step);
            for (int i = 0; i < P * 64; i++)
            {
                x_px[i] -= dt * vel[i];
            }
            // print x_px stats every step
            {
                float mn = x_px[0], mx = x_px[0], sm = 0;
                for (auto v : x_px)
                {
                    if (v < mn)
                    {
                        mn = v;
                    }
                    if (v > mx)
                    {
                        mx = v;
                    }
                    sm += v;
                }
                LOG("    step x_px stats: min={:.4f} max={:.4f} mean={:.4f}\n", mn, mx, sm / x_px.size());
            }
            float ms = std::chrono::duration<float, std::milli>(
                std::chrono::high_resolution_clock::now() - ts)
                           .count();
            LOG("    step {}/{} sigma={:.4f} dt={:.4f} {:.0f}ms\n",
                step + 1, steps, sched.sigmas[step], dt, ms);
            if (cb)
            {
                cb(step + 1, steps, ud);
            }
        }

        // [7] Unpatchify + VAE scale
        LOG("[7] Unpatchify + VAE...\n");
        unpatchify(x_px, lH, lW, lat_hwc);
        for (auto& v : lat_hwc)
        {
            v = v / 0.3611f + 0.1159f;
        }

        // HWC → NCHW for cccc VAE input
        {
            float lmin = *std::min_element(lat_hwc.begin(), lat_hwc.end());
            float lmax = *std::max_element(lat_hwc.begin(), lat_hwc.end());
            float lmean = 0;
            for (auto v : lat_hwc)
            {
                lmean += v;
            }
            lmean /= (float)lat_hwc.size();
            LOG("    Latent stats (scaled): min={:.3f} max={:.3f} mean={:.4f}\n", lmin, lmax, lmean);
        }
        std::vector<float> lat_nchw((size_t)16 * lH * lW);
        for (int c = 0; c < 16; c++)
        {
            for (int h = 0; h < lH; h++)
            {
                for (int w = 0; w < lW; w++)
                {
                    lat_nchw[(size_t)c * lH * lW + h * lW + w] = lat_hwc[(size_t)h * lW * 16 + w * 16 + c];
                }
            }
        }

        std::vector<float> rgb((size_t)3 * height * width, 0.0f);
        {
            MainProcess m;
            auto ini = require_ini_file(model_dir + "/net_vae.ini", "vae");
            if (ini.empty())
            {
                return -9;
            }
            if (!init_model(m, ini, "lH=" + std::to_string(lH) + "; lW=" + std::to_string(lW) + ";"))
            {
                LOG_ERR("VAE failed\n");
                return -9;
            }
            run_model(m, lat_nchw, rgb);
            LOG("    VAE done t={:.1f}s rgb=[{:.3f},{:.3f}]\n", elapsed(),
                *std::min_element(rgb.begin(), rgb.end()),
                *std::max_element(rgb.begin(), rgb.end()));
        }

        if (!save_image(out_path, rgb, width, height))
        {
            return -10;
        }
        LOG("Total: {:.1f}s\n", elapsed());
        return 0;
    }
};

// ============================================================
// C API
// ============================================================
CCCC_LLM_API SdHandle sd_init(const std::string& model_dir)
{
    auto* p = new ZImagePipeline();
    if (!p->init(model_dir))
    {
        delete p;
        return nullptr;
    }
    return (SdHandle)(p);
}

CCCC_LLM_API void sd_destroy(SdHandle handle)
{
    if (handle)
    {
        delete (ZImagePipeline*)(handle);
    }
}

CCCC_LLM_API int sd_generate(SdHandle handle,
    const std::string& prompt, const std::string& output_path,
    int steps, float /*guidance_scale*/,
    int width, int height, int seed,
    SdProgressCallback callback, void* userdata)
{
    if (!handle)
    {
        return -1;
    }
    return ((ZImagePipeline*)(handle))->generate(
        prompt, output_path, steps, width, height, seed, callback, userdata);
}
