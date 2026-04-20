#pragma once
#include <atomic>
#include <format>
#include <iostream>

namespace cccc
{
static inline std::atomic<int> errorCount;

template <typename T>
concept formattable = std::default_initializable<std::formatter<std::remove_cvref_t<T>>>;

template <formattable... Args>
void LOG(const std::format_string<Args...> fmt, Args&&... args)
{
    fputs(std::format(fmt, std::forward<Args>(args)...).c_str(), stdout);
}

template <formattable... Args>
void LOG_ERR(const std::format_string<Args...> fmt, Args&&... args)
{
    if (errorCount++ >= 2000)
    {
        fputs("Too many errors, exit program.\n", stderr);
        exit(0);
    }
    fputs(std::format(fmt, std::forward<Args>(args)...).c_str(), stderr);
}

void showMessageBox(const std::string& str);
void fatalError(const std::string& str);
}    // namespace cccc
