#pragma once

#ifndef C_EXPORT
#ifdef _WIN32
#define C_EXPORT extern "C" __declspec(dllexport)
#else
#define C_EXPORT extern "C"
#endif
#endif

C_EXPORT int cuda_half2float(void* p1, void* p2, unsigned int size);

C_EXPORT int cuda_reciprocal(int type, void* A, void* B, unsigned int size, float scale, float epsilon);
C_EXPORT int cuda_addnumber(int type, void* A, void* R, unsigned int size, float number, float scale);
C_EXPORT int cuda_pow(int type, void* A, void* R, unsigned int size, float bias, float a2);
C_EXPORT int cuda_sparse(int type, void* p1, void* p2, unsigned int size, float rou, float beta);
C_EXPORT int cuda_sign(int type, void* A, void* R, unsigned int size, float v, float section);
C_EXPORT int cuda_cross_entropy(int type, void* A, void* B, void* R, unsigned int size, float a, float scale);
C_EXPORT int cuda_cross_entropy2(int type, void* A, void* B, void* R, unsigned int size, float a, float scale);

C_EXPORT int cuda_div(int type, void* A, void* B, void* R, unsigned int size, float a, float b, float scale);
C_EXPORT int cuda_sectionlimit(int type, void* p1, void* p2, void* p3, unsigned int size, float v0, float v1);
C_EXPORT int cuda_ada_delta_update(int type, void* mean_d2, void* mean_ada_d2, void* d, void* ada_d, unsigned int size, float rou, float epsilon);
C_EXPORT int cuda_adam_update(int type, void* mean_d, void* mean_d2, void* d, void* ada_d, unsigned int size, float beta1, float beta2, float epsilon, float t);
C_EXPORT int cuda_rms_prop_update(int type, void* mead_d2, void* d, void* ada_d, unsigned int size, float rou, float epsilon);

C_EXPORT int cuda_sin(int type, void* A, void* R, unsigned int size, float a, float b);
C_EXPORT int cuda_cos(int type, void* A, void* R, unsigned int size, float a, float b);

C_EXPORT int cuda_zigzag(int type, void* A, void* R, unsigned int size, float a1, float a2);
C_EXPORT int cuda_zigzagb(int type, void* A, void* dA, void* R, void* dR, unsigned int size, float a1, float a2);

C_EXPORT int cuda_step(int type, void* A, void* R, unsigned int size, float unuse1, float unuse2);

C_EXPORT int cuda_leaky_relu(int type, void* A, void* R, unsigned int size, float leak, float a2, float a3);
C_EXPORT int cuda_leaky_relub(int type, void* A, void* dA, void* R, void* dR, unsigned int size, float leak, float a2, float a3);

C_EXPORT int cuda_max(int type, void* A, void* B, void* R, unsigned int size, float unuse1, float unuse2, float unuse3);
C_EXPORT int cuda_maxb(int type, void* A, void* dA, void* B, void* dB, void* R, void* dR, unsigned int size, float alpha, float beta_a, float beta_b);
