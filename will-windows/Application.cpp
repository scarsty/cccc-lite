#include "Application.h"
#include "Brain.h"
#include "File.h"

namespace cccc
{

void Application::run()
{
    Brain brain;

    std::map<std::string, std::string> replace_pairs;
    auto replace_strings = convert::splitString(replace_string_, ";");
    for (auto& r : replace_strings)
    {
        auto rs = convert::splitString(r, ":");
        if (rs.size() >= 1)
        {
            rs.resize(2);
            replace_pairs[rs[0]] = rs[1];
        }
    }

    auto filenames = convert::splitString(ini_file_);
    for (auto filename : filenames)
    {
        if (!File::fileExist(filename))
        {
            fprintf(stderr, "%s doesn't exist!\n", filename.c_str());
        }
        auto ini_str = convert::readStringFromFile(filename);
        //Ìæ»»µôÒ»Ð©×Ö·û
        for (auto rp : replace_pairs)
        {
            convert::replaceAllSubStringRef(ini_str, rp.first, rp.second);
        }
        brain.getOption()->loadString(ini_str);
    }
    auto load_filenames = convert::splitString(brain.getOption()->getString("", "load_ini"), ",");
    for (auto filename : load_filenames)
    {
        if (filename != "")
        {
            brain.getOption()->loadFile(filename);
        }
    }

    //format the string into ini style by inserting '\n'
    convert::replaceAllSubStringRef(add_option_string_, "[", "\n[");
    convert::replaceAllSubStringRef(add_option_string_, "]", "]\n");
    convert::replaceAllSubStringRef(add_option_string_, ";", "\n");
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