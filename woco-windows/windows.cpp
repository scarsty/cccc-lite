#include "Application.h"
#include "cmdline.h"
#include <cstdio>

#ifdef _MSC_VER
#include <windows.h>
#endif

int main(int argc, char* argv[])
{
    cmdline::parser cmd;

    cmd.add<std::string>("config", 'c', "config file (ini format) of the net", false);
    cmd.add<std::string>("add-config", 'a', "additional config string ([sec1]key1=v1;key2=v2[sec2]key1=v1...)", false, "");
    cmd.add("version", 'v', "version information");

#ifdef _MSC_VER
    cmd.parse_check(GetCommandLineA());
#else
    cmd.parse_check(argc, argv);
#endif

    if (cmd.exist("config"))
    {
        woco::Application woco;
        woco.setIniFile(cmd.get<std::string>("config"));
        woco.setIniString(cmd.get<std::string>("add-config"));
        woco.run();
    }
    else if (cmd.exist("version"))
    {
        fprintf(stdout, "WOCO (A Deep Neural Net library) command line interface\n");
        fprintf(stdout, "Float precision is %d (", woco::Application::getFloatPresicion() * 8);
        switch (woco::Application::getFloatPresicion())
        {
        case 2:
            fprintf(stdout, "half");
            break;
        case 4:
            fprintf(stdout, "float");
            break;
        case 8:
            fprintf(stdout, "double");
            break;
        default:
            fprintf(stdout, "unknown");
        }
        fprintf(stdout, ")\n");
        fprintf(stdout, "Built with ");
#if defined(_MSC_VER)
        fprintf(stdout, "Microsoft Visual Studio %d\n", _MSC_VER);
#elif defined(__clang__)
        fprintf(stdout, "Clang %s\n", __clang_version__);
#elif defined(__GNUC__)
        fprintf(stdout, "GNU C %s\n", __VERSION__);
#else
        fprintf(stdout, "Unknown complier\n");
#endif
        fprintf(stdout, "Commemorating my great teacher and friend Dr. Yu Wang\n");
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
