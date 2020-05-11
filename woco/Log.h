#pragma once
#include "convert.h"
#include <cstdio>

namespace woco
{

class Log
{
public:
    Log();
    template <typename... Args>
    static void LOG(const std::string& format, Args... args)
    {
        if (log_state() == 0)
        {
            return;
        }
#ifdef _DEBUG
        convert::checkFormatString(format, args...);
#endif
        fprintf(stdout, format.c_str(), args...);
    }

    static void LOG(const std::string& format)
    {
        if (log_state() == 0)
        {
            return;
        }
        fprintf(stdout, "%s", format.c_str());
    }

    template <typename... Args>
    static void LOG_DEBUG(const std::string& format, Args... args)
    {
#ifdef _DEBUG
        LOG(format, args...);
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

}    // namespace woco