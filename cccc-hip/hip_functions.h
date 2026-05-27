#pragma once

#ifndef C_EXPORT
#ifdef _WIN32
#define C_EXPORT extern "C" __declspec(dllexport)
#else
#define C_EXPORT extern "C"
#endif
#endif

C_EXPORT int hip_half2float(void* p1, void* p2, unsigned int size);
// BF16 转换函数（使用 float 作为中间格式）
// HIP 没有原生 BF16，用 uint16 表示，bits=[sign(1)|exp(8)|mantissa(7)]
// 注意：HIP 推理端仅支持模型参数为 BF16 时的转换
C_EXPORT int hip_bf162float(void* p1, void* p2, unsigned int size);
C_EXPORT int hip_float2bf16(void* p1, void* p2, unsigned int size);

C_EXPORT int hip_reciprocal(int type, void* A, void* B, unsigned int size, float scale, float epsilon);
C_EXPORT int hip_addnumber(int type, void* A, void* R, unsigned int size, float number, float scale);
C_EXPORT int hip_pow(int type, void* A, void* R, unsigned int size, float bias, float a2);
C_EXPORT int hip_sparse(int type, void* p1, void* p2, unsigned int size, float rou, float beta);
C_EXPORT int hip_sign(int type, void* A, void* R, unsigned int size, float v, float section);
C_EXPORT int hip_cross_entropy(int type, void* A, void* B, void* R, unsigned int size, float a, float scale);
C_EXPORT int hip_cross_entropy2(int type, void* A, void* B, void* R, unsigned int size, float a, float scale);

C_EXPORT int hip_div(int type, void* A, void* B, void* R, unsigned int size, float a, float b, float scale);
C_EXPORT int hip_add(int type, void* A, void* B, void* R, unsigned int size, float a, float b);
C_EXPORT int hip_mul(int type, void* A, void* B, void* R, unsigned int size, float a, float b);
C_EXPORT int hip_sectionlimit(int type, void* p1, void* p2, void* p3, unsigned int size, float v0, float v1);
C_EXPORT int hip_ada_delta_update(int type, void* mean_d2, void* mean_ada_d2, void* d, void* ada_d, unsigned int size, float rou, float epsilon);
C_EXPORT int hip_ada_update(int type, void* mean_d2, void* d, void* ada_d, unsigned int size, float rou, float epsilon);
C_EXPORT int hip_adam_update(int type, void* mean_d, void* mean_d2, void* d, void* ada_d, unsigned int size, float beta1, float beta2, float epsilon, float t);
C_EXPORT int hip_rms_prop_update(int type, void* mean_d2, void* d, void* ada_d, unsigned int size, float rou, float epsilon);

C_EXPORT int hip_sin(int type, void* A, void* R, unsigned int size, float a, float b);
C_EXPORT int hip_cos(int type, void* A, void* R, unsigned int size, float a, float b);

C_EXPORT int hip_zigzag(int type, void* A, void* R, unsigned int size, float a1, float a2);
C_EXPORT int hip_zigzagb(int type, void* A, void* dA, void* R, void* dR, unsigned int size, float a1, float a2);

C_EXPORT int hip_step(int type, void* A, void* R, unsigned int size, float unused1, float unused2);

C_EXPORT int hip_leaky_relu(int type, void* A, void* R, unsigned int size, float leak, float a2, float a3);
C_EXPORT int hip_leaky_relub(int type, void* A, void* dA, void* R, void* dR, unsigned int size, float leak, float a2, float a3);

C_EXPORT int hip_max(int type, void* A, void* B, void* R, unsigned int size, float unused1, float unused2, float unused3);
C_EXPORT int hip_maxb(int type, void* A, void* dA, void* B, void* dB, void* R, void* dR, unsigned int size, float alpha, float beta_a, float beta_b);

C_EXPORT int hip_zero_limit(int type, void* A, void* B, void* R, unsigned int size, float beta_a, float beta_b);

C_EXPORT int hip_addbias(float* m, float* b, float* r, unsigned int size_m, unsigned int size_mchannel, unsigned int size_b, float a1, float a2);
C_EXPORT int hip_addbias_bf16(void* r, void* b, unsigned int size_m, unsigned int size_mchannel, unsigned int size_b, float a1, float a2);
C_EXPORT int hip_addbiasb(float* bd, float* rd, unsigned int size_m, unsigned int size_mchannel, unsigned int size_b, float a1, float a2);
C_EXPORT int hip_sigmoid_bf16(void* output, const void* input, unsigned int size, float alpha, float beta);
C_EXPORT int hip_relu_bf16(void* output, const void* input, unsigned int size, float alpha, float beta);
C_EXPORT int hip_tanh_bf16(void* output, const void* input, unsigned int size, float alpha, float beta);
C_EXPORT int hip_softmax_bf16(void* Y, const void* X, unsigned int group_size, unsigned int num_groups, float a, float r);
C_EXPORT int hip_log_softmax_bf16(void* Y, const void* X, unsigned int group_size, unsigned int num_groups, float a, float r);
C_EXPORT int hip_elementwise_add(int type, void* A, void* B, void* R, unsigned int size, float a, float b, float ar);
C_EXPORT int hip_elementwise_mul(int type, void* A, void* B, void* R, unsigned int size, unsigned int b_size, float a, float ar);

//only support 2D, square window, stride = window, no padding
C_EXPORT int hip_pool(float* x, float* y, unsigned int w0, unsigned int h0, unsigned int c, unsigned n, unsigned int w1, unsigned int h1, unsigned int size_win, int type, float a1, float a2);
C_EXPORT int hip_poolb(float* x, float* dx, float* y, float* dy, unsigned int w0, unsigned int h0, unsigned int c, unsigned n, unsigned int w1, unsigned int h1, unsigned int size_win, int type, float a1, float a2);

C_EXPORT int hip_conv2d(float* x, float* w, float* y, int w0, int h0, int c0, int n, int w1, int h1, int c1, int winw, int winh, int stride, int padding, float a1, float a2);
C_EXPORT int hip_conv2db_d(float* dx, float* w, float* dy, int w0, int h0, int c0, int n, int w1, int h1, int c1, int winw, int winh, int stride, int padding, float a1, float a2);
C_EXPORT int hip_conv2db_w(float* x, float* dw, float* dy, int w0, int h0, int c0, int n, int w1, int h1, int c1, int winw, int winh, int stride, int padding, float a1, float a2);
C_EXPORT int hip_im2col(const float* x, float* col, int W0, int H0, int C0, int N, int W1, int H1, int KW, int KH, int stride, int padding);

//Layer Normalization
C_EXPORT int hip_layer_norm_fwd(int type, void* X, void* Y, void* scale, void* bias,
    void* mean_out, void* invstd_out, unsigned int outer, unsigned int inner, float epsilon);
C_EXPORT int hip_layer_norm_bwd(int type, void* X, void* dY, void* dX,
    void* scale, void* mean, void* invstd, void* dscale, void* dbias,
    unsigned int outer, unsigned int inner);

//RMS Normalization
C_EXPORT int hip_rms_norm_fwd(int type, void* X, void* Y, void* scale,
    void* invstd_out, unsigned int outer, unsigned int inner, float epsilon);
C_EXPORT int hip_rms_norm_bwd(int type, void* X, void* dY, void* dX,
    void* scale, void* invstd, void* dscale,
    unsigned int outer, unsigned int inner);

//4D Permute
C_EXPORT int hip_permute4d(int type, const void* X, void* Y,
    int in_d0, int in_d1, int in_d2, int in_d3,
    int p0, int p1, int p2, int p3);

//RoPE (half-rotate / Qwen style)
C_EXPORT int hip_rope_fwd(int type, const void* X, void* Y,
    const void* cos_tab, const void* sin_tab,
    unsigned int D, unsigned int T, unsigned int B,
    unsigned int pos_offset);
C_EXPORT int hip_rope_bwd(int type, const void* dY, void* dX,
    const void* cos_tab, const void* sin_tab,
    unsigned int D, unsigned int T, unsigned int B);

//RoPE interleaved style (ncnn RotaryEmbed mode=1)
C_EXPORT int hip_rope_interleaved_fwd(int type, const void* X, void* Y,
    const void* cos_tab, const void* sin_tab,
    unsigned int D, unsigned int T, unsigned int B,
    unsigned int pos_offset);

//Pixel Shuffle
C_EXPORT int hip_pixel_shuffle_fwd(int type, const void* X, void* Y,
    unsigned int W, unsigned int H, unsigned int C_out, unsigned int r, unsigned int N);
C_EXPORT int hip_pixel_shuffle_bwd(int type, const void* dY, void* dX,
    unsigned int W, unsigned int H, unsigned int C_out, unsigned int r, unsigned int N);

//Embedding lookup
C_EXPORT int hip_embed_fwd(int type, const void* ids, const void* W, void* Y,
    int D, int T, int B);
C_EXPORT int hip_embed_bwd(int type, const void* ids, const void* dY, void* dW,
    int D, int T, int B);

//Tile
C_EXPORT int hip_tile_fwd(int type, const void* X, void* Y,
    int W_in, int H_in, int C_in, int N_in,
    int W_out, int H_out, int C_out, int N_out);
C_EXPORT int hip_tile_bwd(int type, const void* dY, void* dX,
    int W_in, int H_in, int C_in, int N_in,
    int W_out, int H_out, int C_out, int N_out);

//Causal mask
C_EXPORT int hip_causal_mask(int type, void* scores, int T_q, int T_k, int B, int pos_offset);
C_EXPORT int hip_clamp_scores_half(void* scores, unsigned int n, float threshold);

//Group Normalization affine
C_EXPORT int hip_group_norm_affine_fwd(int type,
    const void* X_hat, void* Y, const void* scale, const void* bias,
    int outer, int inner, int G, int CperG, int WH);
C_EXPORT int hip_group_norm_affine_bwd(int type,
    const void* X_hat, const void* dY, void* dX_hat,
    const void* scale, void* dscale, void* dbias,
    int outer, int inner, int G, int CperG, int WH);

//VAE Reparameterization
C_EXPORT int hip_reparam_fwd(int type,
    const void* mu, const void* log_var, const void* eps, void* z,
    unsigned int size);
C_EXPORT int hip_reparam_bwd(int type,
    const void* log_var, const void* eps, const void* dz,
    void* dmu, void* d_log_var,
    unsigned int size, float alpha_mu, float alpha_lv);

//L1 loss backward
C_EXPORT int hip_l1_bwd(int type,
    const void* A, const void* Y, void* dA,
    unsigned int size, float alpha, float beta);

//KL log_var backward
C_EXPORT int hip_kl_lv_bwd(int type,
    const void* log_var, void* dlv,
    unsigned int size, float alpha, float beta);

//Nearest neighbor upsample
C_EXPORT int hip_upsample_nearest_fwd(int type,
    const void* X, void* Y,
    unsigned int W, unsigned int H, unsigned int C, unsigned int N,
    int sh, int sw);
C_EXPORT int hip_upsample_nearest_bwd(int type,
    void* dX, const void* dY,
    unsigned int W, unsigned int H, unsigned int C, unsigned int N,
    int sh, int sw, float alpha, float beta);

//Bilinear upsample
C_EXPORT int hip_upsample_bilinear_fwd(int type,
    const void* X, void* Y,
    unsigned int W_in, unsigned int H_in,
    unsigned int W_out, unsigned int H_out,
    unsigned int C, unsigned int N);
C_EXPORT int hip_upsample_bilinear_bwd(int type,
    const void* dY, void* dX,
    unsigned int W_in, unsigned int H_in,
    unsigned int W_out, unsigned int H_out,
    unsigned int C, unsigned int N, float alpha);

// Unified type conversion (HIP supports half/bf16/float only; no FP8/FP4)
// src_type/dst_type: 0=float, 2=half, 3=bfloat16  (cccc::DataType enum values)
C_EXPORT int hip_convert(const void* src, int src_type, void* dst, int dst_type, unsigned int n, float scale);

