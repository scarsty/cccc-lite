#include <hip/hip_runtime.h>
#include "hip_functions.h"

#define blockMax 1024

#define cal_i() (blockIdx.x * blockDim.x + threadIdx.x)

inline int blockNum(unsigned int size) { return (size + blockMax - 1) / blockMax; }

inline int getError(const char* content)
{
    hipDeviceSynchronize();
    hipError_t hipStatus = hipGetLastError();
    if (hipStatus != hipSuccess)
    {
        fprintf(stderr, "%s kernel launch failed: %s\n", content, hipGetErrorString(hipStatus));
        return 1;
    }
    return 0;
}

#define HIP_FUNCTION22(name, function) \
    static __global__ void name##kernel(realc* p1, realc* p2, unsigned int size, realc a1, realc a2) \
    { \
        int i = cal_i(); \
        if (i < size) \
        { \
            function \
        } \
    } \
    int hip_##name(realc* p1, realc* p2, unsigned int size, realc a1, realc a2) \
    { \
        name##kernel<<<blockNum(size), blockMax>>>(p1, p2, size, a1, a2); \
        return getError(#name); \
    }

#define HIP_FUNCTION42(name, function) \
    static __global__ void name##kernel(realc* p1, realc* p2, realc* p3, realc* p4, unsigned int size, realc a1, realc a2) \
    { \
        int i = cal_i(); \
        if (i < size) \
        { \
            function \
        } \
    } \
    int hip_##name(realc* p1, realc* p2, realc* p3, realc* p4, unsigned int size, realc a1, realc a2) \
    { \
        name##kernel<<<blockNum(size), blockMax>>>(p1, p2, p3, p4, size, a1, a2); \
        return getError(#name); \
    }

#define HIP_FUNCTION44(name, function) \
    static __global__ void name##kernel(realc* p1, realc* p2, realc* p3, realc* p4, unsigned int size, realc a1, realc a2, realc a3, realc a4) \
    { \
        int i = cal_i(); \
        if (i < size) \
        { \
            function \
        } \
    } \
    int hip_##name(realc* p1, realc* p2, realc* p3, realc* p4, unsigned int size, realc a1, realc a2, realc a3, realc a4) \
    { \
        name##kernel<<<blockNum(size), blockMax>>>(p1, p2, p3, p4, size, a1, a2, a3, a4); \
        return getError(#name); \
    }

HIP_FUNCTION22(sigmoid,
    {
        p2[i] = a1 * exp(p1[i]) / (1 + exp(p1[i])) + a2 * p2[i];
    });

HIP_FUNCTION42(sigmoidb,
    {
        p4[i] = a1 * p2[i] * p1[i] * (1 - p1[i]) + a2 * p4[i];
    });

HIP_FUNCTION22(relu,
    {
        p2[i] = a1 * max(0, p1[i]) + a2 * p2[i];
    });

HIP_FUNCTION42(relub,
    {
        if (p3[i] >= 0)
        {
            p4[i] = p2[i] * a1 + p4[i] * a2;
        }
        else
        {
            p4[i] = p4[i] * a2;
        }
    });

HIP_FUNCTION22(reciprocal, { p2[i] = a1 / (p1[i] + a2); });

HIP_FUNCTION22(addnumber, { p2[i] = a1 + p1[i] * a2; });

HIP_FUNCTION44(adam_update,
    {
        realc& beta1 = a1;
        realc& beta2 = a2;
        realc& epsilon = a3;
        realc& t = a4;
        p1[i] = p1[i] * beta1 + p3[i] * (realc(1) - beta1);
        p2[i] = p2[i] * beta2 + p3[i] * p3[i] * (realc(1) - beta2);
        p4[i] = p1[i] / (realc(1) - pow(beta1, t)) / (sqrt(p2[i] / (realc(1) - pow(beta2, t))) + epsilon);
    });

static __global__ void addbiaskernel(realc* m, realc* b, realc* r, unsigned int size_m, unsigned int size_mchannel, unsigned int size_b, realc a1, realc a2)
{
    int i = cal_i();
    if (i < size_m)
    {
        r[i] *= a2;
        r[i] += a1 * b[i / size_mchannel % size_b];
    }
}

int hip_addbias(realc* m, realc* b, realc* r, unsigned int size_m, unsigned int size_mchannel, unsigned int size_b, realc a1, realc a2)
{
    addbiaskernel<<<blockNum(size_m), blockMax>>>(m, b, r, size_m, size_mchannel, size_b, a1, a2);
    return getError("addbias");
}

static __global__ void addbiasbkernel(realc* bd, realc* rd, unsigned int size_m, unsigned int size_mchannel, unsigned int size_b, realc a1, realc a2)
{
    int i = cal_i();
    if (i < size_b)
    {
        bd[i] *= a2;
        for (int i1 = 0; i1 < size_m / size_mchannel / size_b; i1++)
        {
            for (int i2 = 0; i2 < size_mchannel; i2++)
            {
                bd[i] += a1 * rd[i2 + i1 * size_b * size_mchannel + i * size_mchannel];
            }
        }
    }
}

int hip_addbiasb(realc* bd, realc* rd, unsigned int size_m, unsigned int size_mchannel, unsigned int size_b, realc a1, realc a2)
{
    addbiasbkernel<<<blockNum(size_b), blockMax>>>(bd, rd, size_m, size_mchannel, size_b, a1, a2);
    return getError("addbiasb");
}

static __global__ void softmaxkernel(realc* x, realc* y, unsigned int size, unsigned int channel, realc a1, realc a2)
{
    int i = cal_i();
    if (i < size / channel)
    {
        realc min = 0;
        for (int i1 = 0; i1 < channel; i1++)
        {
            //y[i1 + i * channel] = x[i1 + i * channel];
            if (x[i1 + i * channel] < min)
            {
                min = x[i1 + i * channel];
            }
        }
        realc sum = 0;
        for (int i1 = 0; i1 < channel; i1++)
        {
            sum += exp(x[i1 + i * channel] - min);
        }
        for (int i1 = 0; i1 < channel; i1++)
        {
            y[i1 + i * channel] = a1 * exp(x[i1 + i * channel] - min) / sum + a2 * y[i1 + i * channel];
        }
    }
}

int hip_softmax(realc* x, realc* y, unsigned int size, unsigned int channel, realc a1, realc a2)
{
    softmaxkernel<<<blockNum(size / channel), blockMax>>>(x, y, size, channel, a1, a2);
    return getError("softmax");
}

static __global__ void poolkernel(realc* x, realc* y, unsigned int w0, unsigned w0h0, unsigned int w1, unsigned int w1h1, unsigned int size_win, unsigned int size, int type, realc a1, realc a2)
{
    int i = cal_i();
    if (i < size)
    {
        int n = i / w1h1;
        int x1 = i % w1h1 / w1;
        int y1 = i % w1;
        int x0 = x1 * size_win;
        int y0 = y1 * size_win;
        realc v = type == 0 ? -1e8 : 0;
        for (int ix = x0; ix < x0 + size_win; ix++)
        {
            for (int iy = y0; iy < y0 + size_win; iy++)
            {
                int index = n * w0h0 + ix * w0 + iy;
                if (type == 0 && x[index] > v)
                {
                    v = x[index];
                }
                else if (type == 1)
                {
                    v += x[index] / size_win / size_win;
                }
            }
        }
        y[i] = a1 * v + a2 * y[i];
    }
}

int hip_pool(realc* x, realc* y, unsigned int w0, unsigned int h0, unsigned int c, unsigned n, unsigned int w1, unsigned int h1, unsigned int size_win, int type, realc a1, realc a2)
{
    poolkernel<<<blockNum(w1 * h1 * c * n), blockMax>>>(x, y, w0, w0 * h0, w1, w1 * h1, size_win, w1 * h1 * c * n, type, a1, a2);
    return getError("pool");
}

static __global__ void poolbkernel(realc* x, realc* dx, realc* y, realc* dy, unsigned int w0, unsigned w0h0, unsigned int w1, unsigned int w1h1, unsigned int size_win, unsigned int size, int type, realc a1, realc a2)
{
    int i = cal_i();
    if (i < size)
    {
        int n = i / w0h0;
        int x0 = i % w0h0 / w0;
        int y0 = i % w0;
        int x1 = x0 / size_win;
        int y1 = y0 / size_win;
        int index = n * w1h1 + x1 * w1 + y1;
        if (type == 0)
        {
            if (x[i] == y[index])
            {
                dx[i] = a1 * dy[index] + a2 * dx[i];
            }
            else
            {
                dx[i] = a2 * dx[i];
            }
        }
        else
        {
            dx[i] = a1 * dy[index] / size_win / size_win + a2 * dx[i];
        }
    }
}

int hip_poolb(realc* x, realc* dx, realc* y, realc* dy, unsigned int w0, unsigned int h0, unsigned int c, unsigned n, unsigned int w1, unsigned int h1, unsigned int size_win, int type, realc a1, realc a2)
{
    poolbkernel<<<blockNum(w0 * h0 * c * n), blockMax>>>(x, dx, y, dy, w0, w0 * h0, w1, w1 * h1, size_win, w0 * h0 * c * n, type, a1, a2);
    return getError("poolb");
}

static __global__ void conv2dkernel(realc* x, realc* w, realc* y, int w0, int h0, int c0, int n, int w1, int h1, int c1, int winw, int winh, int stride,int padding, realc a1, realc a2)
{
    int i = cal_i();
    if (i < w1 * h1 * c1 * n)
    {
        int in = i / (w1 * h1 * c1);
        int ic1 = i / (w1 * h1) % c1;
        int ih1 = i / w1 % h1;
        int iw1 = i % w1;

        int iw0_begin = iw1 * stride - padding;
        int ih0_begin = ih1 * stride - padding;

        realc v = 0;
        for (int ic0 = 0; ic0 < c0; ic0++)
        {
            for (int iwinh = 0; iwinh < winh; iwinh++)
            {
                for (int iwinw = 0; iwinw < winw; iwinw++)
                {
                    int ih0 = ih0_begin + iwinh;
                    int iw0 = iw0_begin + iwinw;
                    if (iw0 >= 0 && iw0 < w0 && ih0 >= 0 && ih0 < h0)
                    {
                        int index_x = in * w0 * h0 * c0 + ic0 * w0 * h0 + ih0 * w0 + iw0;
                        int index_w = ic1 * winw * winh * c0 + ic0 * winw * winh + iwinh * winw + iwinw;
                        v += x[index_x] * w[index_w];
                    }
                }
            }
        }
        y[i] = a1 * v + a2 * y[i];
    }
}

int hip_conv2d(realc* x, realc* w, realc* y, int w0, int h0, int c0, int n, int w1, int h1, int c1, int winw, int winh, int stride, int padding, realc a1, realc a2)
{
    conv2dkernel<<<blockNum(w1 * h1 * c1 * n), blockMax>>>(x, w, y, w0, h0, c0, n, w1, h1, c1, winw, winh, stride, padding, a1, a2);
    return getError("conv2d");
}


static __global__ void conv2db_dkernel(realc* dx, realc* w, realc* dy, int w0, int h0, int c0, int n, int w1, int h1, int c1, int winw, int winh, int stride, int padding, realc a1, realc a2)
{
    int i = cal_i();
    if (i < w0 * h0 * c0 * n)
    {
        int in = i / (w0 * h0 * c0);
        int ic0 = i / (w0 * h0) % c0;
        int ih0 = i / w0 % h0;
        int iw0 = i % w0;

        int iw1_begin = iw0 * stride + padding;
        int ih1_begin = ih0 * stride + padding;

        realc v = 0;
        for (int ic1 = 0; ic1 < c1; ic1++)
        {
            for (int iwinh = 0; iwinh < winh; iwinh++)
            {
                for (int iwinw = 0; iwinw < winw; iwinw++)
                {
                    int ih1 = ih1_begin - iwinh;
                    int iw1 = iw1_begin - iwinw;
                    if (iw1 >= 0 && iw1 < w1 && ih1 >= 0 && ih1 < h1)
                    {
                        int index_y = in * w1 * h1 * c1 + ic1 * w1 * h1 + ih1 * w1 + iw1;
                        int index_w = ic1 * winw * winh * c0 + ic0 * winw * winh + iwinh * winw + iwinw;
                        v += dy[index_y] * w[index_w];
                    }
                }
            }
        }
        dx[i] = a1 * v + a2 * dx[i];
    }
}


int hip_conv2db_d(realc* dx, realc* w, realc* dy, int w0, int h0, int c0, int n, int w1, int h1, int c1, int winw, int winh, int stride, int padding, realc a1, realc a2)
{
    conv2db_dkernel<<<blockNum(w0 * h0 * c0 * n), blockMax>>>(dx, w, dy, w0, h0, c0, n, w1, h1, c1, winw, winh, stride, padding, a1, a2);
    return getError("conv2db_d");
}


static __global__ void conv2db_wkernel(realc* x, realc* dw, realc* dy, int w0, int h0, int c0, int n, int w1, int h1, int c1, int winw, int winh, int stride, int padding, realc a1, realc a2)
{
    int i = cal_i();
    if (i < winw * winh * c0 * c1)
    {
        int ic1 = i / (winw * winh * c0);
        int ic0 = i / (winw * winh) % c0;
        int iwinh = i / winw % winh;
        int iwinw = i % winw;

        realc v = 0;
        for (int in = 0; in < n; in++)
        {
            for (int iw1 = 0; iw1 < w1; iw1++)
            {
                for (int ih1 = 0; ih1 < h1; ih1++)
                {
                    int ih0 = ih1 + iwinh - padding;
                    int iw0 = iw1 + iwinw - padding;
                    if (iw0 >= 0 && iw0 < w0 && ih0 >= 0 && ih0 < h0)
                    {
                        int index_x = in * w0 * h0 * c0 + ic0 * w0 * h0 + ih0 * w0 + iw0;
                        int index_y = in * w1 * h1 * c1 + ic1 * w1 * h1 + ih1 * w1 + iw1;
                        v += dy[index_y] * x[index_x];
                    }
                }
            }
        }
        dw[i] = a1 * v + a2 * dw[i];
    }
}


int hip_conv2db_w(realc* x, realc* dw, realc* dy, int w0, int h0, int c0, int n, int w1, int h1, int c1, int winw, int winh, int stride, int padding, realc a1, realc a2)
{
    conv2db_wkernel<<<blockNum(winw * winh * c0 * c1), blockMax>>>(x, dw, dy, w0, h0, c0, n, w1, h1, c1, winw, winh, stride, padding, a1, a2);
    return getError("conv2db_d");
}
