#include "Application.h"
#include "MainProcess.h"
#include "filefunc.h"
#include "strfunc.h"

namespace cccc
{

void Application::run()
{
    MainProcess mp;

    std::map<std::string, std::string> replace_pairs;
    auto replace_strings = strfunc::splitString(replace_string_, ";");
    for (auto& r : replace_strings)
    {
        auto rs = strfunc::splitString(r, ":");
        if (rs.size() >= 1)
        {
            rs.resize(2);
            replace_pairs[rs[0]] = rs[1];
        }
    }

    auto filenames = strfunc::splitString(ini_file_);
    for (auto filename : filenames)
    {
        if (!filefunc::fileExist(filename))
        {
            LOG_ERR("{} doesn't exist!\n", filename.c_str());
        }
        auto ini_str = filefunc::readFileToString(filename);
        //替换掉一些字符
        for (auto rp : replace_pairs)
        {
            strfunc::replaceAllSubStringRef(ini_str, rp.first, rp.second);
        }
        mp.getOption()->loadString(ini_str);
    }
    auto load_filenames = strfunc::splitString(mp.getOption()->getString("train", "load_ini"), ",");
    for (auto filename : load_filenames)
    {
        if (filename != "")
        {
            mp.getOption()->loadFile(filename);
        }
    }

    //format the string into ini style by inserting '\n'
    strfunc::replaceAllSubStringRef(add_option_string_, "[", "\n[");
    strfunc::replaceAllSubStringRef(add_option_string_, "]", "]\n");
    strfunc::replaceAllSubStringRef(add_option_string_, ";", "\n");
    mp.getOption()->loadString(add_option_string_);

    if (mp.init() != 0)
    {
        return;
    }
    loop_ = true;
    while (loop_)
    {
        mp.run();
        loop_ = false;
    }
}

}    // namespace cccc