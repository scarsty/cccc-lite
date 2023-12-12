#pragma once

#ifndef C_EXPORT
#ifdef _WIN32
#define C_EXPORT extern "C" __declspec(dllexport)
#else
#define C_EXPORT extern "C"
#endif
#endif

#if REAL_PRECISION == 0
typedef float realc;
#elif REAL_PRECISION == 1
typedef double realc;
#elif REAL_PRECISION == 2
typedef half realc;
#endif

#define HIP_FUNCTION22H(name) C_EXPORT int hip_##name(realc* p1, realc* p2, unsigned int size, realc a1, realc a2)
#define HIP_FUNCTION32H(name) C_EXPORT int hip_##name(realc* p1, realc* p2, realc* p3, unsigned int size, realc a1, realc a2)
#define HIP_FUNCTION42H(name) C_EXPORT int hip_##name(realc* p1, realc* p2, realc* p3, realc* p4, unsigned int size, realc a1, realc a2)
#define HIP_FUNCTION44H(name) C_EXPORT int hip_##name(realc* p1, realc* p2, realc* p3, realc* p4, unsigned int size, realc a1, realc a2, realc a3, realc a4)

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

C_EXPORT int hip_addbias(realc* m, realc* b, realc* r, unsigned int size_m, unsigned int size_mchannel, unsigned int size_b, realc a1, realc a2);
C_EXPORT int hip_addbiasb(realc* bd, realc* rd, unsigned int size_m, unsigned int size_mchannel, unsigned int size_b, realc a1, realc a2);
C_EXPORT int hip_softmax(realc* x, realc* y, unsigned int size, unsigned int channel, realc a1, realc a2);

//only support 2D, square window, stride = window, no padding
C_EXPORT int hip_pool(realc* x, realc* y, unsigned int w0, unsigned int h0, unsigned int c, unsigned n, unsigned int w1, unsigned int h1, unsigned int size_win, int type, realc a1, realc a2);
C_EXPORT int hip_poolb(realc* x, realc* dx, realc* y, realc* dy, unsigned int w0, unsigned int h0, unsigned int c, unsigned n, unsigned int w1, unsigned int h1, unsigned int size_win, int type, realc a1, realc a2);

C_EXPORT int hip_conv2d(realc* x, realc* w, realc* y, int w0, int h0, int c0, int n, int w1, int h1, int c1, int winw, int winh, int stride, int padding, realc a1, realc a2);
C_EXPORT int hip_conv2db_d(realc* dx, realc* w, realc* dy, int w0, int h0, int c0, int n, int w1, int h1, int c1, int winw, int winh, int stride, int padding, realc a1, realc a2);
C_EXPORT int hip_conv2db_w(realc* x, realc* dw, realc* dy, int w0, int h0, int c0, int n, int w1, int h1, int c1, int winw, int winh, int stride, int padding, realc a1, realc a2);
