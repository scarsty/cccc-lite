#pragma once

#ifndef C_EXPORT
#ifdef _WIN32
#define C_EXPORT extern "C" __declspec(dllexport)
#else
#define C_EXPORT extern "C"
#endif
#endif

C_EXPORT int hip_half2float(void* p1, void* p2, unsigned int size);

C_EXPORT int hip_reciprocal(int type, void* A, void* B, unsigned int size, float scale, float epsilon);
C_EXPORT int hip_addnumber(int type, void* A, void* R, unsigned int size, float number, float scale);
C_EXPORT int hip_pow(int type, void* A, void* R, unsigned int size, float bias, float a2);
C_EXPORT int hip_sparse(int type, void* p1, void* p2, unsigned int size, float rou, float beta);
C_EXPORT int hip_sign(int type, void* A, void* R, unsigned int size, float v, float section);
C_EXPORT int hip_cross_entropy(int type, void* A, void* B, void* R, unsigned int size, float a, float scale);
C_EXPORT int hip_cross_entropy2(int type, void* A, void* B, void* R, unsigned int size, float a, float scale);

C_EXPORT int hip_div(int type, void* A, void* B, void* R, unsigned int size, float a, float b, float scale);
C_EXPORT int hip_sectionlimit(int type, void* p1, void* p2, void* p3, unsigned int size, float v0, float v1);
C_EXPORT int hip_ada_delta_update(int type, void* mean_d2, void* mean_ada_d2, void* d, void* ada_d, unsigned int size, float rou, float epsilon);
C_EXPORT int hip_adam_update(int type, void* mean_d, void* mean_d2, void* d, void* ada_d, unsigned int size, float beta1, float beta2, float epsilon, float t);
C_EXPORT int hip_rms_prop_update(int type, void* mead_d2, void* d, void* ada_d, unsigned int size, float rou, float epsilon);

C_EXPORT int hip_sin(int type, void* A, void* R, unsigned int size, float a, float b);
C_EXPORT int hip_cos(int type, void* A, void* R, unsigned int size, float a, float b);

C_EXPORT int hip_zigzag(int type, void* A, void* R, unsigned int size, float a1, float a2);
C_EXPORT int hip_zigzagb(int type, void* A, void* dA, void* R, void* dR, unsigned int size, float a1, float a2);

C_EXPORT int hip_step(int type, void* A, void* R, unsigned int size, float unuse1, float unuse2);

C_EXPORT int hip_leaky_relu(int type, void* A, void* R, unsigned int size, float leak, float a2, float a3);
C_EXPORT int hip_leaky_relub(int type, void* A, void* dA, void* R, void* dR, unsigned int size, float leak, float a2, float a3);

C_EXPORT int hip_max(int type, void* A, void* B, void* R, unsigned int size, float unuse1, float unuse2, float unuse3);
C_EXPORT int hip_maxb(int type, void* A, void* dA, void* B, void* dB, void* R, void* dR, unsigned int size, float alpha, float beta_a, float beta_b);

C_EXPORT int hip_addbias(float* m, float* b, float* r, unsigned int size_m, unsigned int size_mchannel, unsigned int size_b, float a1, float a2);
C_EXPORT int hip_addbiasb(float* bd, float* rd, unsigned int size_m, unsigned int size_mchannel, unsigned int size_b, float a1, float a2);
C_EXPORT int hip_softmax(float* x, float* y, unsigned int size, unsigned int channel, float a1, float a2);

//only support 2D, square window, stride = window, no padding
C_EXPORT int hip_pool(float* x, float* y, unsigned int w0, unsigned int h0, unsigned int c, unsigned n, unsigned int w1, unsigned int h1, unsigned int size_win, int type, float a1, float a2);
C_EXPORT int hip_poolb(float* x, float* dx, float* y, float* dy, unsigned int w0, unsigned int h0, unsigned int c, unsigned n, unsigned int w1, unsigned int h1, unsigned int size_win, int type, float a1, float a2);

C_EXPORT int hip_conv2d(float* x, float* w, float* y, int w0, int h0, int c0, int n, int w1, int h1, int c1, int winw, int winh, int stride, int padding, float a1, float a2);
C_EXPORT int hip_conv2db_d(float* dx, float* w, float* dy, int w0, int h0, int c0, int n, int w1, int h1, int c1, int winw, int winh, int stride, int padding, float a1, float a2);
C_EXPORT int hip_conv2db_w(float* x, float* dw, float* dy, int w0, int h0, int c0, int n, int w1, int h1, int c1, int winw, int winh, int stride, int padding, float a1, float a2);
