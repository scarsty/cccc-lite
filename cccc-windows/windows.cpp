#include "Application.h"
#include "Log.h"
#include "cmdline.h"
#include <cstdio>

#ifdef _MSC_VER
#include <windows.h>
#endif

int main(int argc, char* argv[])
{
    cmdline::parser cmd;

    cmd.add<std::string>("config", 'c', "config file (ini format) of the net", false, "will.ini");
    cmd.add<std::string>("add-config", 'a', "additional config string ([sec1]key1=v1;key2=v2[sec2]key1=v1...)", false, "");
    cmd.add<std::string>("replace", 'r', "replace strings before loading ini (old1:new1;old2:new2...)", false, "");
    cmd.add("version", 'v', "version information");

#ifdef _MSC_VER
    cmd.parse_check(GetCommandLineA());
#else
    cmd.parse_check(argc, argv);
#endif

    if (cmd.exist("config"))
    {
        cccc::Application app;
        app.ini_file_ = cmd.get<std::string>("config");
        app.add_option_string_ = cmd.get<std::string>("add-config");
        app.replace_string_ = cmd.get<std::string>("replace");
        app.run();
    }
    else if (cmd.exist("version"))
    {
        cccc::LOG("CCCC (A Deep Neural Net library) command line interface\n");
        cccc::LOG("Float precision is %d (", cccc::Application::getFloatPresicion() * 8);
        switch (cccc::Application::getFloatPresicion())
        {
        case 2:
            cccc::LOG("half");
            break;
        case 4:
            cccc::LOG("float");
            break;
        case 8:
            cccc::LOG("double");
            break;
        default:
            cccc::LOG("unknown");
        }
        cccc::LOG(")\n");
        cccc::LOG("Built with ");
#if defined(_MSC_VER)
        cccc::LOG("Microsoft Visual Studio %d\n", _MSC_VER);
#elif defined(__clang__)
        cccc::LOG("Clang %s\n", __clang_version__);
#elif defined(__GNUC__)
        cccc::LOG("GNU C %s\n", __VERSION__);
#else
        cccc::LOG("Unknown complier\n");
#endif
        cccc::LOG("Commemorating my great teacher and friend Dr. Yu Wang\n");
    }
    else
    {
        if (argc >= 1)
        {
            cmd.parse_check({ argv[0], "--help" });
        }
    }

    return 0;
}
