#pragma once
#include <cmath>

namespace cccc
{
template <typename T>
T linear_inter(int x, int x1, int x2, T y1, T y2)
{
    T y = y1 + (y2 - y1) / (x2 - x1) * (x - x1);
    return y;
}

template <typename T>
T scale_inter(int x, int x1, int x2, T y1, T y2)
{
    T p = pow(y2 / y1, 1.0 * (x - x1) / (x2 - x1));
    T y = y1 * p;
    return y;
}

template <typename T>
T clip(T x, T y1, T y2)
{
    if (x < y1) { return y1; }
    if (x > y2) { return y2; }
    return x;
}

}
