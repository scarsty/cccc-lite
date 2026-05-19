#pragma once

#ifndef C_EXPORT
#ifdef _WIN32
#define C_EXPORT extern "C" __declspec(dllexport)
#else
#define C_EXPORT extern "C"
#endif
#endif

C_EXPORT int cuda_half2float(void* p1, void* p2, unsigned int size);
C_EXPORT int cuda_bf162float(void* p1, void* p2, unsigned int size);
C_EXPORT int cuda_float2bf16(void* p1, void* p2, unsigned int size);

C_EXPORT int cuda_reciprocal(int type, void* A, void* B, unsigned int size, float scale, float epsilon);
C_EXPORT int cuda_addnumber(int type, void* A, void* R, unsigned int size, float number, float scale);
C_EXPORT int cuda_pow(int type, void* A, void* R, unsigned int size, float e, float bias);
C_EXPORT int cuda_sparse(int type, void* p1, void* p2, unsigned int size, float rou, float beta);
C_EXPORT int cuda_sign(int type, void* A, void* R, unsigned int size, float v, float section);
C_EXPORT int cuda_cross_entropy(int type, void* A, void* B, void* R, unsigned int size, float a, float scale);
C_EXPORT int cuda_cross_entropy2(int type, void* A, void* B, void* R, unsigned int size, float a, float scale);

C_EXPORT int cuda_div(int type, void* A, void* B, void* R, unsigned int size, float a, float b, float scale);
C_EXPORT int cuda_sectionlimit(int type, void* p1, void* p2, void* p3, unsigned int size, float v0, float v1);
C_EXPORT int cuda_ada_delta_update(int type, void* mean_d2, void* mean_ada_d2, void* d, void* ada_d, unsigned int size, float rou, float epsilon);
C_EXPORT int cuda_adam_update(int type, void* mean_d, void* mean_d2, void* d, void* ada_d, unsigned int size, float beta1, float beta2, float epsilon, float t);
C_EXPORT int cuda_rms_prop_update(int type, void* mean_d2, void* d, void* ada_d, unsigned int size, float rou, float epsilon);

C_EXPORT int cuda_sin(int type, void* A, void* R, unsigned int size, float a, float b);
C_EXPORT int cuda_cos(int type, void* A, void* R, unsigned int size, float a, float b);

C_EXPORT int cuda_zigzag(int type, void* A, void* R, unsigned int size, float a1, float a2);
C_EXPORT int cuda_zigzagb(int type, void* A, void* dA, void* R, void* dR, unsigned int size, float a1, float a2);

C_EXPORT int cuda_step(int type, void* A, void* R, unsigned int size, float unused1, float unused2);

C_EXPORT int cuda_leaky_relu(int type, void* A, void* R, unsigned int size, float leak, float a2, float a3);
C_EXPORT int cuda_leaky_relub(int type, void* A, void* dA, void* R, void* dR, unsigned int size, float leak, float a2, float a3);

C_EXPORT int cuda_max(int type, void* A, void* B, void* R, unsigned int size, float unused1, float unused2, float unused3);
C_EXPORT int cuda_maxb(int type, void* A, void* dA, void* B, void* dB, void* R, void* dR, unsigned int size, float alpha, float beta_a, float beta_b);

C_EXPORT int cuda_zero_limit(int type, void* A, void* B, void* R, unsigned int size, float beta_a, float beta_b);

//Layer Normalization (按 inner 维度做归一化)
//X, Y: [outer, inner]
//scale, bias, mean_out, invstd_out: [inner] / [outer]
//forward: 每个 outer group 做 (x - mean)/sqrt(var+eps), 然后 * scale + bias; 同时输出 mean 与 1/sqrt(var+eps) 给反向使用
//backward: 输入 X, dY, scale, mean, invstd; 输出 dX, dscale, dbias
C_EXPORT int cuda_layer_norm_fwd(int type, void* X, void* Y, void* scale, void* bias,
    void* mean_out, void* invstd_out, unsigned int outer, unsigned int inner, float epsilon);
C_EXPORT int cuda_layer_norm_bwd(int type, void* X, void* dY, void* dX,
    void* scale, void* mean, void* invstd, void* dscale, void* dbias,
    unsigned int outer, unsigned int inner);

//RMS Normalization (按 inner 维度做归一化, 不减均值, 无 bias)
//y_i = x_i / sqrt(mean(x^2)+eps) * scale_i
//forward 输出 invstd 给反向使用
C_EXPORT int cuda_rms_norm_fwd(int type, void* X, void* Y, void* scale,
    void* invstd_out, unsigned int outer, unsigned int inner, float epsilon);
C_EXPORT int cuda_rms_norm_bwd(int type, void* X, void* dY, void* dX,
    void* scale, void* invstd, void* dscale,
    unsigned int outer, unsigned int inner);

//4 维任意轴置换 (W, H, C, N)
//perm 数组长度为 4, in_dims/out_dims 也为 4
//out_dims[i] = in_dims[perm[i]]
C_EXPORT int cuda_permute4d(int type, const void* X, void* Y,
    int in_d0, int in_d1, int in_d2, int in_d3,
    int p0, int p1, int p2, int p3);

//RoPE (half-rotate / Qwen 风格)
//X / Y: shape (D, T, 1, B), cos/sin: shape (D/2, T)
//forward: 对每个 (b, t), 设 d=D/2, x_l = X[0..d), x_r = X[d..D), 则
//         y_l = x_l*cos - x_r*sin, y_r = x_r*cos + x_l*sin
//backward: dx_l = dy_l*cos + dy_r*sin, dx_r = -dy_l*sin + dy_r*cos
C_EXPORT int cuda_rope_fwd(int type, const void* X, void* Y,
    const void* cos_tab, const void* sin_tab,
    unsigned int D, unsigned int T, unsigned int B,
    unsigned int pos_offset);
C_EXPORT int cuda_rope_bwd(int type, const void* dY, void* dX,
    const void* cos_tab, const void* sin_tab,
    unsigned int D, unsigned int T, unsigned int B);

//RoPE interleaved 风格 (ncnn RotaryEmbed mode=1)
//y[2i] = x[2i]*cos[i] - x[2i+1]*sin[i],  y[2i+1] = x[2i+1]*cos[i] + x[2i]*sin[i]
C_EXPORT int cuda_rope_interleaved_fwd(int type, const void* X, void* Y,
    const void* cos_tab, const void* sin_tab,
    unsigned int D, unsigned int T, unsigned int B,
    unsigned int pos_offset);

//pixel_shuffle (sub-pixel convolution): Input (W, H, C_out*r*r, N) -> Output (W*r, H*r, C_out, N)
C_EXPORT int cuda_pixel_shuffle_fwd(int type, const void* X, void* Y,
    unsigned int W, unsigned int H, unsigned int C_out, unsigned int r, unsigned int N);
C_EXPORT int cuda_pixel_shuffle_bwd(int type, const void* dY, void* dX,
    unsigned int W, unsigned int H, unsigned int C_out, unsigned int r, unsigned int N);

//Embedding lookup: ids (T*B floats cast to int), W (D*V), Y (D*T*B)
//forward:  Y[d + t*D + b*D*T] = W[d + (int)ids[t + b*T] * D]
//backward: atomicAdd(&dW[d + (int)ids[t+b*T]*D], dY[d + t*D + b*D*T])
C_EXPORT int cuda_embed_fwd(int type, const void* ids, const void* W, void* Y,
    int D, int T, int B);
C_EXPORT int cuda_embed_bwd(int type, const void* ids, const void* dY, void* dW,
    int D, int T, int B);

//Tile: X (W_in, H_in, C_in, N_in) -> Y (W_out=W_in*r0, H_out=H_in*r1, C_out=C_in*r2, N_out=N_in*r3)
//forward:  Y[idx] = X[x_coord % each_in_dim]
//backward: atomicAdd into X.d
C_EXPORT int cuda_tile_fwd(int type, const void* X, void* Y,
    int W_in, int H_in, int C_in, int N_in,
    int W_out, int H_out, int C_out, int N_out);
C_EXPORT int cuda_tile_bwd(int type, const void* dY, void* dX,
    int W_in, int H_in, int C_in, int N_in,
    int W_out, int H_out, int C_out, int N_out);

//Causal mask: set scores[k + q*T_k + b*T_q*T_k] = -1e9 where k > q
//scores: (T_k, T_q, 1, B) float or half tensor; type=0 float, type=2 half
C_EXPORT int cuda_causal_mask(int type, void* scores, int T_q, int T_k, int B, int pos_offset);
C_EXPORT int cuda_clamp_scores_half(void* scores, unsigned int n, float threshold);

//Group Normalization affine: 对 layer_norm 归一化后的 X_hat 应用逐通道 scale/bias
//数据布局: [outer=G*N, inner=W*H*CperG], G=groups, CperG=C/G, WH=W*H
//通道索引: c = (k%G)*CperG + i/WH, 其中 k 为 outer 索引, i 为 inner 索引
//forward: Y[k*inner+i] = scale[c] * X_hat[k*inner+i] + bias[c]
C_EXPORT int cuda_group_norm_affine_fwd(int type,
    const void* X_hat, void* Y, const void* scale, const void* bias,
    int outer, int inner, int G, int CperG, int WH);
//backward: dX_hat[k*inner+i] = dY[k*inner+i]*scale[c]; dscale[c]+=dY*X_hat; dbias[c]+=dY
C_EXPORT int cuda_group_norm_affine_bwd(int type,
    const void* X_hat, const void* dY, void* dX_hat,
    const void* scale, void* dscale, void* dbias,
    int outer, int inner, int G, int CperG, int WH);
//VAE 重参数化: z = mu + exp(log_var*0.5) * eps, eps 为预采样 N(0,1) 噪声
//forward: z[i] = mu[i] + exp(log_var[i]*0.5) * eps[i]
C_EXPORT int cuda_reparam_fwd(int type,
    const void* mu, const void* log_var, const void* eps, void* z,
    unsigned int size);
//backward: dmu[i] += alpha_mu*dz[i]; d_log_var[i] += alpha_lv*dz[i]*exp(log_var[i]*0.5)*0.5*eps[i]
C_EXPORT int cuda_reparam_bwd(int type,
    const void* log_var, const void* eps, const void* dz,
    void* dmu, void* d_log_var,
    unsigned int size, float alpha_mu, float alpha_lv);

//L1 loss backward: dA[i] = beta*dA[i] + alpha*sign(A[i]-Y[i])
C_EXPORT int cuda_l1_bwd(int type,
    const void* A, const void* Y, void* dA,
    unsigned int size, float alpha, float beta);

//KL log_var backward: dlv[i] = beta*dlv[i] + alpha*0.5*(exp(lv[i])-1)
C_EXPORT int cuda_kl_lv_bwd(int type,
    const void* log_var, void* dlv,
    unsigned int size, float alpha, float beta);

//Nearest neighbor upsample: X(W,H,C,N) -> Y(W*sw, H*sh, C, N)
C_EXPORT int cuda_upsample_nearest_fwd(int type,
    const void* X, void* Y,
    unsigned int W, unsigned int H, unsigned int C, unsigned int N,
    int sh, int sw);
//backward: dX(W,H,C,N) from dY(W*sw,H*sh,C,N); dX[i] = beta*dX[i] + alpha*sum_over_block(dY)
C_EXPORT int cuda_upsample_nearest_bwd(int type,
    void* dX, const void* dY,
    unsigned int W, unsigned int H, unsigned int C, unsigned int N,
    int sh, int sw, float alpha, float beta);
//Bilinear upsample: X(W_in,H_in,C,N) -> Y(W_out,H_out,C,N); align_corners=False
C_EXPORT int cuda_upsample_bilinear_fwd(int type,
    const void* X, void* Y,
    unsigned int W_in, unsigned int H_in,
    unsigned int W_out, unsigned int H_out,
    unsigned int C, unsigned int N);
//backward with atomicAdd; caller must pre-scale dX by keepWeight before calling (alpha applied to dY)
C_EXPORT int cuda_upsample_bilinear_bwd(int type,
    const void* dY, void* dX,
    unsigned int W_in, unsigned int H_in,
    unsigned int W_out, unsigned int H_out,
    unsigned int C, unsigned int N, float alpha);
