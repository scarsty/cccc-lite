#include "Factory.h"
#include "DataPreparerImage.h"
#include "DynamicLibrary.h"
#include "NetCifa.h"
#include "Option.h"

namespace woco
{

Net* Factory::createNet()
{
    std::string lib_key = "library";
#ifdef _DEBUG
    lib_key = "libraryd";
#endif

    std::string script = Option::getInstance().getString("Net", "structure");
    Net* net = nullptr;
    if (!Option::getInstance().getString("Net", lib_key).empty())
    {
        net = (Net*)getCreator(Option::getInstance().getString("Net", lib_key), Option::getInstance().getString("Net", "function", "net_ext"));
    }
    if (net == nullptr)
    {
        //之前使用vector，但是resize时为复制构造，而实际应每次都默认构造
        static int total = 0;
        static std::map<int, NetCifa> nets;
        //Log::LOG("Net script is:\n%s\n\n", script.c_str());
        net = &nets[total];
        total++;
    }
    if (net)
    {
        net->setMessage(script);
    }
    return net;
}

DataPreparer* Factory::createDP(const std::string& section, const std::vector<int>& dimx, const std::vector<int>& dimy)
{
    std::string lib_key = "library";
#ifdef _DEBUG
    lib_key = "libraryd";
#endif
    auto dp = (DataPreparer*)getCreator(Option::getInstance().getString(section, lib_key), Option::getInstance().getString(section, "function", "dp_ext"));
    if (dp)
    {
        dp->create_by_dll_ = Option::getInstance().getString(section, lib_key);
    }
    else
    {
        auto mode = Option::getInstance().getString(section, "mode", "image");
        if (Option::getInstance().hasSection(section))
        {
            mode = "";
        }
        if (mode == "image")
        {
            static int total = 0;
            static std::map<int, DataPreparerImage> m;
            dp = &m[total++];
            Log::LOG("Create default image data preparer\n");
        }
        else
        {
            static int total = 0;
            static std::map<int, DataPreparer> m;
            dp = &m[total++];
            Log::LOG("Create default data preparer\n");
        }
    }
    dp->section_ = section;
    dp->dimx_ = dimx;
    dp->dimy_ = dimy;
    dp->init();
    return dp;
}

void* Factory::getCreator(std::string library_name, std::string function_name)
{
    using CreatorFunc = void* (*)();
    if (library_name.find(".") == std::string::npos)
    {
#ifdef _WIN32
        library_name = library_name + ".dll";
#else
        library_name = "lib" + library_name + ".so";
#endif
    }
    //Log::LOG("Try to load library and function: %s, %s \n", library_name.c_str(), function_name.c_str());
    if (!library_name.empty() && !function_name.empty())
    {
        //此处请注意：64位Windows中dll的函数名形式上与__cdecl一致，不再支持32位
        CreatorFunc func = nullptr;
        func = (CreatorFunc)DynamicLibrary::getFunction(library_name, function_name);
        if (func)
        {
            //Log::LOG("Create from dynamic library successfully\n");
            return func();
        }
    }
    return nullptr;
}

}    // namespace woco