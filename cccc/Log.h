#pragma once
#include "fmt1.h"

namespace cccc
{

class LOG
{
public:
    template <typename... Args>
    LOG(Args&&... args)
    {
        if (current_level() >= 1)
        {
            fmt1::print(args...);
        }
    }
    template <typename... Args>
    LOG(int level, Args&&... args)
    {
        if (current_level() >= level)
        {
            fmt1::print(args...);
        }
    }

    static void setLevel(int level)
    {
        prev_level() = current_level();
        current_level() = level;
    }
    static int getLevel() { return current_level(); }
    static void restoreLevel() { current_level() = prev_level(); }

private:
    static int& current_level()
    {
        static int cl = 1;
        return cl;
    }
    static int& prev_level()
    {
        static int pl = 1;
        return pl;
    }
};

}    // namespace cccc