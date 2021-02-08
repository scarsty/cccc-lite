#pragma once
#define FMT_HEADER_ONLY
#include "fmt/format.h"
#include "fmt/ranges.h"

namespace cccc
{

class Log
{
public:
    Log();
    virtual ~Log();
    template <typename... Args>
    static void LOG(Args&&... args)
    {
        if (log_state() == 0)
        {
            return;
        }
        fmt::print(stdout, args...);
    }
    template <typename... Args>
    static void LOG_DEBUG(Args&&... args)
    {
#ifdef _DEBUG
        LOG(args...);
#endif
    }

private:
    static int& log_state()
    {
        static int log = 1;
        return log;
    }

public:
    static void setLog(int log) { log_state() = log; }
};

}    // namespace cccc