#pragma once
#include "fmt1.h"

namespace cccc
{
static inline int errorCount;
template <typename... Args>
void LOG(Args&&... args)
{
    fmt1::print(std::forward<Args>(args)...);
}

template <typename... Args>
void LOG_ERR(Args&&... args)
{
    if (errorCount++ >= 2000)
    {
        fmt1::print(stderr, "Too many errors, exit program.\n");
        exit(0);
    }
    fmt1::print(stderr, std::forward<Args>(args)...);
}

void showMessageBox(const std::string& str);
void fatalError(const std::string& str);
}    // namespace cccc