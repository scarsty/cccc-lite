#pragma once
#include "dll_export.h"
#include <string>
#include <vector>

namespace woco
{

class Net;
class DataPreparer;

class DLL_EXPORT Factory
{
public:
    Factory() = delete;

public:
    static Net* createNet();
    static DataPreparer* createDP(const std::string& section, const std::vector<int>& dimx, const std::vector<int>& dimy);

private:
    static void* getCreator(std::string library_name, std::string function_name);
};

}    // namespace woco