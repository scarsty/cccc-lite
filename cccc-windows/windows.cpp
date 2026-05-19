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
    cmd.add<std::string>("prompt", 'p', "single prompt for --sd mode (non-interactive)", false, "");
    cmd.add<std::string>("output", 'o', "output image path for --sd mode", false, "output.png");
    cmd.add<int>("seed", 'e', "random seed for image generation (-1=random)", false, -1);
    cmd.add<int>("steps", 't', "denoising steps for --sd mode", false, 9);
    cmd.add<float>("cfg", 'g', "cfg/guidance scale for --sd mode", false, 7.5f);
    cmd.add<int>("width", 'W', "output width for --sd mode (multiple of 16)", false, 512);
    cmd.add<int>("height", 'H', "output height for --sd mode (multiple of 16)", false, 512);
    cmd.add("version", 'v', "version information");
    cmd.add("llm", 'l', "LLM interactive chat mode");
    cmd.add("sd", 's', "Stable Diffusion image generation mode");

#ifdef _MSC_VER
    cmd.parse_check(GetCommandLineA());
#else
    cmd.parse_check(argc, argv);
#endif

    if (cmd.exist("llm"))
    {
        cccc::Application app;
        app.ini_file_ = cmd.get<std::string>("config");
        app.run_llm();
    }
    else if (cmd.exist("sd"))
    {
        cccc::Application app;
        app.ini_file_ = cmd.get<std::string>("config");
        app.sd_prompt_ = cmd.get<std::string>("prompt");
        app.sd_output_ = cmd.get<std::string>("output");
        app.sd_seed_ = cmd.get<int>("seed");
        app.sd_steps_ = cmd.get<int>("steps");
        app.sd_cfg_ = cmd.get<float>("cfg");
        app.sd_width_ = cmd.get<int>("width");
        app.sd_height_ = cmd.get<int>("height");
        app.run_sd();
    }
    else if (cmd.exist("config"))
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
        cccc::LOG("Built with ");
#if defined(_MSC_VER)
        cccc::LOG("Microsoft Visual Studio {}\n", _MSC_VER);
#elif defined(__clang__)
        cccc::LOG("Clang {}\n", __clang_version__);
#elif defined(__GNUC__)
        cccc::LOG("GNU C {}\n", __VERSION__);
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
