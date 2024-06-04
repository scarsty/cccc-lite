#pragma once

#ifndef C_EXPORT
#ifdef _WIN32
#define C_EXPORT extern "C" __declspec(dllexport)
#else
#define C_EXPORT extern "C"
#endif
#endif

#define HIP_FUNCTION22H(name) C_EXPORT int hip_##name(float* p1, float* p2, unsigned int size, float a1, float a2)
#define HIP_FUNCTION32H(name) C_EXPORT int hip_##name(float* p1, float* p2, float* p3, unsigned int size, float a1, float a2)
#define HIP_FUNCTION42H(name) C_EXPORT int hip_##name(float* p1, float* p2, float* p3, float* p4, unsigned int size, float a1, float a2)
#define HIP_FUNCTION44H(name) C_EXPORT int hip_##name(float* p1, float* p2, float* p3, float* p4, unsigned int size, float a1, float a2, float a3, float a4)

HIP_FUNCTION22H(sigmoid);
HIP_FUNCTION42H(sigmoidb);
HIP_FUNCTION22H(relu);
HIP_FUNCTION42H(relub);

HIP_FUNCTION22H(reciprocal);
HIP_FUNCTION22H(addnumber);
HIP_FUNCTION32H(ada_update);
HIP_FUNCTION42H(ada_delta_update);
HIP_FUNCTION44H(adam_update);
HIP_FUNCTION32H(rms_prop_update);

C_EXPORT int hip_addbias(float* m, float* b, float* r, unsigned int size_m, unsigned int size_mchannel, unsigned int size_b, float a1, float a2);
C_EXPORT int hip_addbiasb(float* bd, float* rd, unsigned int size_m, unsigned int size_mchannel, unsigned int size_b, float a1, float a2);
C_EXPORT int hip_softmax(float* x, float* y, unsigned int size, unsigned int channel, float a1, float a2);

//only support 2D, square window, stride = window, no padding
C_EXPORT int hip_pool(float* x, float* y, unsigned int w0, unsigned int h0, unsigned int c, unsigned n, unsigned int w1, unsigned int h1, unsigned int size_win, int type, float a1, float a2);
C_EXPORT int hip_poolb(float* x, float* dx, float* y, float* dy, unsigned int w0, unsigned int h0, unsigned int c, unsigned n, unsigned int w1, unsigned int h1, unsigned int size_win, int type, float a1, float a2);

C_EXPORT int hip_conv2d(float* x, float* w, float* y, int w0, int h0, int c0, int n, int w1, int h1, int c1, int winw, int winh, int stride, int padding, float a1, float a2);
C_EXPORT int hip_conv2db_d(float* dx, float* w, float* dy, int w0, int h0, int c0, int n, int w1, int h1, int c1, int winw, int winh, int stride, int padding, float a1, float a2);
C_EXPORT int hip_conv2db_w(float* x, float* dw, float* dy, int w0, int h0, int c0, int n, int w1, int h1, int c1, int winw, int winh, int stride, int padding, float a1, float a2);
