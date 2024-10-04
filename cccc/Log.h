#pragma once
#include "fmt1.h"
#include <iostream>

namespace cccc
{
static inline int errorCount;

template <typename... Args>
void LOG(Args&&... args)
{
    std::cout << fmt1::format(std::forward<Args>(args)...);
    fflush(stdout);
}

template <typename... Args>
void LOG_ERR(Args&&... args)
{
    if (errorCount++ >= 2000)
    {
        std::cerr << "Too many errors, exit program.\n";
        fflush(stderr);
        exit(0);
    }
    std::cerr << fmt1::format(std::forward<Args>(args)...);
    fflush(stderr);
}

void showMessageBox(const std::string& str);
void fatalError(const std::string& str);
}    // namespace cccc