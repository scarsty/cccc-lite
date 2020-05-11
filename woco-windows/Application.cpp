#include "Application.h"
#include "File.h"
#include "Neural.h"
#include "convert.h"

namespace woco
{

Application::Application()
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
    Neural neural;
    auto filenames = convert::splitString(ini_file_);
    for (auto filename : filenames)
    {
        if (!File::fileExist(filename))
        {
            fprintf(stderr, "%s doesn't exist!\n", filename.c_str());
            return;
        }
        neural.getOption().loadIniFile(filename);
    }
    auto load_filenames = convert::splitString(neural.getOption().getString("", "load_ini"), ",");
    for (auto filename : load_filenames)
    {
        if (filename != "")
        {
            neural.getOption().loadIniFile(filename);
        }
    }

    //format the string into ini style by inserting '\n'
    convert::replaceAllSubStringRef(ini_string_, "[", "\n[");
    convert::replaceAllSubStringRef(ini_string_, "]", "]\n");
    convert::replaceAllSubStringRef(ini_string_, ";", "\n");
    neural.getOption().loadIniString(ini_string_);

    if (neural.init() != 0)
    {
        return;
    }
    loop_ = true;
    while (loop_)
    {
        neural.run();
        loop_ = false;
    }
    stop();
}

void Application::mainLoop()
{
}

int Application::getFloatPresicion()
{
    return Neural::getFloatPrecision();
}

void Application::callback(void* net_pointer)
{
}

}    // namespace woco