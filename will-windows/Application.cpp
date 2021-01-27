#include "Application.h"
#include "Brain.h"
#include "File.h"

namespace will
{

Application::Application()
{
}

Application::~Application()
{
}

void Application::start()
{
}

void Application::stop()
{
}

void Application::run()
{
    start();
    Brain brain;
    auto filenames = convert::splitString(ini_file_);
    for (auto filename : filenames)
    {
        if (!File::fileExist(filename))
        {
            fprintf(stderr, "%s doesn't exist!\n", filename.c_str());
            return;
        }
        brain.getOption()->loadIniFile(filename);
    }
    auto load_filenames = convert::splitString(brain.getOption()->getString("", "load_ini"), ",");
    for (auto filename : load_filenames)
    {
        if (filename != "")
        {
            brain.getOption()->loadIniFile(filename);
        }
    }

    //format the string into ini style by inserting '\n'
    convert::replaceAllSubStringRef(ini_string_, "[", "\n[");
    convert::replaceAllSubStringRef(ini_string_, "]", "]\n");
    convert::replaceAllSubStringRef(ini_string_, ";", "\n");
    brain.getOption()->loadIniString(ini_string_);

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
    stop();
}

void Application::mainLoop()
{
}

int Application::getFloatPresicion()
{
    return Brain::getFloatPrecision();
}

void Application::callback(void* net_pointer)
{
}

}    // namespace will