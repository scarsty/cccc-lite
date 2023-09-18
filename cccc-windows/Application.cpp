#include "Application.h"
#include "Brain.h"
#include "filefunc.h"

namespace cccc
{

void Application::run()
{
    Brain brain;

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
            LOG(stderr, "{} doesn't exist!\n", filename.c_str());
        }
        auto ini_str = strfunc::readStringFromFile(filename);
        //Ìæ»»µôÒ»Ð©×Ö·û
        for (auto rp : replace_pairs)
        {
            strfunc::replaceAllSubStringRef(ini_str, rp.first, rp.second);
        }
        brain.getOption()->loadString(ini_str);
    }
    auto load_filenames = strfunc::splitString(brain.getOption()->getString("train", "load_ini"), ",");
    for (auto filename : load_filenames)
    {
        if (filename != "")
        {
            brain.getOption()->loadFile(filename);
        }
    }

    //format the string into ini style by inserting '\n'
    strfunc::replaceAllSubStringRef(add_option_string_, "[", "\n[");
    strfunc::replaceAllSubStringRef(add_option_string_, "]", "]\n");
    strfunc::replaceAllSubStringRef(add_option_string_, ";", "\n");
    brain.getOption()->loadString(add_option_string_);

    if (brain.init() != 0)
    {
        return;
    }
    loop_ = true;
    while (loop_)
    {
        brain.run();
        loop_ = false;
    }
}

int Application::getFloatPresicion()
{
    return sizeof(real);
}

}    // namespace cccc