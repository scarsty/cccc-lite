#pragma once
#include "fmt1.h"

namespace cccc
{

inline int& errorCount()
{
    static int ec = 0;
    return ec;
}

template <typename... Args>
void LOG(Args&&... args)
{
    fmt1::print(args...);
}
template <typename... Args>
void LOG(FILE* fout, Args&&... args)
{
    if (fout == stderr)
    {
        fflush(stdout);
        errorCount()++;
        //fmt1::print("{}\n", errorCount());
        if (errorCount() >= 2000)
        {
            fmt1::print(stderr, "Too many errors, exit program.\n");
            exit(0);
        }
    }
    fmt1::print(fout, args...);
}

}    // namespace cccc