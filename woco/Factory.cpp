#include "Factory.h"
#include "DataPreparerImage.h"
#include "DynamicLibrary.h"
#include "NetLua.h"

namespace woco
{

Net* Factory::createNet(Option& op)
{
    std::string lib_key = "library";
#ifdef _DEBUG
    lib_key = "libraryd";
#endif
    Net* net = nullptr;
    if (!op.getString("Net", lib_key).empty())
    {
        net = (Net*)getCreator(op.getString("Net", lib_key), op.getString("Net", "function", "net_ext"));
    }
    if (net == nullptr)
    {
        //之前使用vector，但是resize时为复制构造，而实际应每次都默认构造
        static int i = 0;
        static std::map<int, NetLua> nets;
        std::string script = op.getString("Net", "structure");
        //Log::LOG("Net script is:\n%s\n\n", script.c_str());
        nets[i].setScript(script);
        net = &nets[i];
        i++;
    }
    return net;
}

DataPreparer* Factory::createDP(Option& op, const std::string& section, const std::vector<int>& dimx, const std::vector<int>& dimy)
{
    std::string lib_key = "library";
#ifdef _DEBUG
    lib_key = "libraryd";
#endif
    auto dp = (DataPreparer*)getCreator(op.getString(section, lib_key), op.getString(section, "function", "dp_ext"));
    if (dp)
    {
        dp->create_by_dll_ = op.getString(section, lib_key);
    }
    else
    {
        auto mode = op.getString(section, "mode", "image");
        if (op.hasSection(section))
        {
            mode = "";
        }
        if (mode == "image")
        {
            static int i = 0;
            static std::map<int, DataPreparerImage> m;
            dp = &m[i++];
            Log::LOG("Create default image data preparer\n");
        }
        else
        {
            static int i = 0;
            static std::map<int, DataPreparer> m;
            dp = &m[i++];
            Log::LOG("Create default data preparer\n");
        }
    }
    dp->option_ = op;
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