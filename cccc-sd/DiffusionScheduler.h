#pragma once
#include <cmath>
#include <vector>

namespace cccc_sd
{

// ============================================================
// DDIM（Denoising Diffusion Implicit Models）调度器
// 参考：Song et al. 2020 (https://arxiv.org/abs/2010.02502)
//
// 核心公式（eta=0，确定性采样）：
//   x̂₀ = (xₜ - √(1 - ᾱₜ)·ε̂) / √ᾱₜ
//   x_{t-1} = √ᾱ_{t-1}·x̂₀ + √(1 - ᾱ_{t-1})·ε̂
// ============================================================

struct NoiseSchedule
{
    // 在每个推理时间步 t 处的累积 alpha 值
    float alpha_cumprod = 1.0f;       // ᾱₜ
    float alpha_cumprod_prev = 1.0f;  // ᾱ_{t-1}
};

class DiffusionScheduler
{
public:
    // --------------------------------------------------------
    // 初始化参数
    // train_timesteps: 训练时使用的总时间步数（一般为 1000）
    // beta_start / beta_end: 线性噪声调度的端点
    // --------------------------------------------------------
    explicit DiffusionScheduler(int train_timesteps = 1000,
        float beta_start = 0.00085f,
        float beta_end   = 0.012f);

    // --------------------------------------------------------
    // 设置推理时间步序列（从 train_timesteps 中均匀采样）
    // num_steps: 推理步数（如 20、4、1 等）
    // --------------------------------------------------------
    void set_timesteps(int num_steps);

    // --------------------------------------------------------
    // 返回推理时间步序列（降序，从 T-1 到 0）
    // --------------------------------------------------------
    const std::vector<int>& timesteps() const { return timesteps_; }

    // --------------------------------------------------------
    // 获取时间步 t 的噪声调度信息
    // --------------------------------------------------------
    NoiseSchedule get_schedule(int t) const;

    // --------------------------------------------------------
    // 向纯净样本 x0 中加入时间步 t 的高斯噪声，得到 xₜ。
    // x0, noise: 长度相同的浮点数组
    // t:          当前时间步
    // out:        输出 xₜ（与 x0 等长）
    // --------------------------------------------------------
    void add_noise(const float* x0, const float* noise, int t,
        float* out, int n) const;

    // --------------------------------------------------------
    // DDIM 单步去噪（eta=0，确定性）
    // xt:        当前 xₜ（长度 n）
    // eps_pred:  模型预测的噪声 ε̂（长度 n）
    // t:         当前时间步
    // out:       输出 x_{t-1}（长度 n）
    // --------------------------------------------------------
    void step(const float* xt, const float* eps_pred, int t,
        float* out, int n) const;

private:
    int train_timesteps_;
    std::vector<float> alphas_cumprod_;  // 预计算的 ᾱₜ，长度为 train_timesteps_
    std::vector<int>   timesteps_;       // 推理时间步序列（降序）
};


// ============================================================
// 实现（inline，头文件内）
// ============================================================

inline DiffusionScheduler::DiffusionScheduler(int train_timesteps,
    float beta_start, float beta_end)
    : train_timesteps_(train_timesteps)
{
    // 线性 beta 调度
    // beta_t = beta_start + (beta_end - beta_start) * t / (T - 1)
    alphas_cumprod_.resize(train_timesteps_);
    float cumprod = 1.0f;
    for (int t = 0; t < train_timesteps_; ++t)
    {
        float beta_t = beta_start
            + (beta_end - beta_start) * static_cast<float>(t) / (train_timesteps_ - 1);
        float alpha_t = 1.0f - beta_t;
        cumprod *= alpha_t;
        alphas_cumprod_[t] = cumprod;
    }
}

inline void DiffusionScheduler::set_timesteps(int num_steps)
{
    // 从 [0, train_timesteps_) 中均匀选取 num_steps 个时间步，降序排列
    timesteps_.clear();
    timesteps_.resize(num_steps);
    float step = static_cast<float>(train_timesteps_) / num_steps;
    for (int i = 0; i < num_steps; ++i)
    {
        // 从最大到最小，对应去噪方向
        int t = static_cast<int>(std::round(
            (num_steps - 1 - i) * step + step * 0.5f));
        t = (t < train_timesteps_) ? t : train_timesteps_ - 1;
        timesteps_[i] = t;
    }
}

inline NoiseSchedule DiffusionScheduler::get_schedule(int t) const
{
    NoiseSchedule s;
    s.alpha_cumprod = alphas_cumprod_[t];
    s.alpha_cumprod_prev = (t > 0) ? alphas_cumprod_[t - 1] : 1.0f;
    return s;
}

inline void DiffusionScheduler::add_noise(const float* x0,
    const float* noise, int t, float* out, int n) const
{
    float sqrt_alpha = std::sqrt(alphas_cumprod_[t]);
    float sqrt_one_minus = std::sqrt(1.0f - alphas_cumprod_[t]);
    for (int i = 0; i < n; ++i)
    {
        out[i] = sqrt_alpha * x0[i] + sqrt_one_minus * noise[i];
    }
}

inline void DiffusionScheduler::step(const float* xt,
    const float* eps_pred, int t, float* out, int n) const
{
    auto s = get_schedule(t);
    float sqrt_at      = std::sqrt(s.alpha_cumprod);
    float sqrt_at_prev = std::sqrt(s.alpha_cumprod_prev);
    float sqrt_bt      = std::sqrt(1.0f - s.alpha_cumprod);
    float sqrt_bt_prev = std::sqrt(1.0f - s.alpha_cumprod_prev);

    for (int i = 0; i < n; ++i)
    {
        // 预测 x̂₀
        float x0_pred = (xt[i] - sqrt_bt * eps_pred[i]) / sqrt_at;

        // DDIM 确定性步 (eta=0)
        out[i] = sqrt_at_prev * x0_pred + sqrt_bt_prev * eps_pred[i];
    }
}

}  // namespace cccc_sd
