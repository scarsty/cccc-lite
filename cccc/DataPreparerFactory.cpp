#include "DataPreparerFactory.h"
#include "DataPreparerImage.h"
#include "DataPreparerTxt.h"
#include "DynamicLibrary.h"
#include "filefunc.h"

namespace cccc
{

DataPreparer* DataPreparerFactory::create(Option* op, const std::string& section, const std::vector<int>& dim0, const std::vector<int>& dim1)
{
    DataPreparer* dp = nullptr;
    std::string library_name = op->getString(section, "library");
#ifdef _WIN32
#ifdef _DEBUG
    library_name = op->getString(section, "library_dlld", library_name);
#else
    library_name = op->getString(section, "library_dll", library_name);
#endif
#else
    library_name = op->getString(section, "library_so", library_name);
    if (strfunc::toLowerCase(filefunc::getFileExt(library_name)) == "dll")
    {
        library_name = filefunc::getFilenameWithoutPath(filefunc::getFileMainname(library_name)) + ".so";
        if (library_name.find("lib") != 0)
        {
            library_name = "lib" + library_name;
        }
    }
#endif
    auto function_name = op->getString(section, "function", "dp_ext");

    if (library_name.find(".") == std::string::npos)
    {
#ifdef _WIN32
        library_name = library_name + ".dll";
#else
        library_name = "lib" + library_name + ".so";
#endif
    }
    LOG("Try to create data preparer with section \"{}\"\n", section);
    if (!library_name.empty() && !function_name.empty())
    {
        //此处请注意：64位系统中dll的函数名形式上与__cdecl一致。深度学习耗费内存较大，故此处不再对32位系统进行处理
        using MYFUNC = void* (*)();
        MYFUNC func = nullptr;
        func = (MYFUNC)DynamicLibrary::getFunction(library_name, function_name);
        if (func)
        {
            dp = (DataPreparer*)func();
            LOG("Create from {} in {}\n", function_name, library_name);
            dp->create_by_dll_ = library_name;
        }
        else
        {
            LOG("Failed to load {} in {}\n", function_name, library_name);
        }
    }
    if (dp == nullptr)
    {
        auto mode = op->getString(section, "mode", "image");
        if (op->hasSection(section))
        {
            mode = "";
        }
        if (mode == "image")
        {
            dp = new DataPreparerImage();
            LOG("Create default image data preparer\n");
        }
        else if (mode == "txt")
        {
            dp = new DataPreparerTxt();
            LOG("Create default txt data preparer\n");
        }
        else
        {
            dp = new DataPreparer();
            LOG("Create default data preparer\n");
        }
    }

    dp->option_ = op;
    dp->section_ = section;
    dp->dim0_ = dim0;
    dp->dim1_ = dim1;
    dp->init();

    return dp;
}

void DataPreparerFactory::destroy(DataPreparer* dp)
{
    if (dp)
    {
        if (dp->create_by_dll_.empty())
        {
            delete dp;
        }
        else
        {
            //请尽量使用静态方式
            //dp->destroy();
            //using MYFUNC = void (*)(void*);
            //MYFUNC func = nullptr;
            //func = (MYFUNC)DynamicLibrary::getFunction(dp->create_by_dll_, "destroy");
            //if (func)
            //{
            //    func(dp);
            //}
        }
        dp = nullptr;
    }
}

}    // namespace cccc