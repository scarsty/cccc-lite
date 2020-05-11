#pragma once
#include <vector>

namespace woco
{

class Util
{
public:
    Util();

    template <class T>
    static void delete_all(std::vector<T*>& pointer_v)
    {
        for (auto& pointer : pointer_v)
        {
            delete pointer;
            pointer = nullptr;
        }
        pointer_v.clear();
    }
    template <typename... Ptr>
    static void delete_all(Ptr&... ptr)
    {
        auto res = { (delete ptr, ptr = nullptr, 0)... };
        (void)res;
    }
};

}    // namespace woco