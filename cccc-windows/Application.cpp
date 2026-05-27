#include "Application.h"
#include "Log.h"
#include "MainProcess.h"
#include "cccc-llm.h"
#include "filefunc.h"
#include "strfunc.h"
#include <filesystem>
#include <random>
#include <string>

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
    bool ini_dir_set = false;
    for (auto filename : filenames)
    {
        if (!filefunc::fileExist(filename))
        {
            LOG_ERR("{} doesn't exist!\n", filename.c_str());
        }
        auto ini_str = filefunc::readFileToString(filename);
        //替换掉一些字符
        for (auto [str0, str1] : replace_pairs)
        {
            strfunc::replaceAllSubStringRef(ini_str, str0, str1);
        }
        mp.getOption()->loadString(ini_str);
        if (!ini_dir_set && filefunc::fileExist(filename))
        {
            mp.setIniDir(std::filesystem::path(filename).parent_path().string());
            ini_dir_set = true;
        }
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

void Application::run_llm()
{
    SetConsoleOutputCP(CP_UTF8);
    SetConsoleCP(CP_UTF8);

    LlmHandle h = llm_init(ini_file_);
    if (!h)
    {
        LOG_ERR("Failed to initialize LLM from {}\n", ini_file_.c_str());
        return;
    }

    LOG("=== cccc-llm interactive chat ===\n");
    LOG("Commands: /clear (reset history), /quit (exit)\n\n");

    std::string line;
    while (true)
    {
        LOG("You: ");
        //fflush(stdout);
        if (!std::getline(std::cin, line))
        {
            break;
        }
        while (!line.empty() && (line.back() == '\r' || line.back() == '\n' || line.back() == ' '))
        {
            line.pop_back();
        }

        if (line == "/quit" || line == "/exit")
        {
            break;
        }
        if (line == "/clear")
        {
            llm_reset(h);
            LOG("[Conversation history cleared]\n\n");
            continue;
        }
        if (line.empty())
        {
            continue;
        }

        LOG("Assistant: ");
        //fflush(stdout);
        llm_chat_stream(h, line, 8192, 1, [](const std::string& tok, void*)
            {
                LOG("{}", tok);
                //fflush(stdout);
            },
            nullptr);
        LOG("\n\n");
    }

    llm_destroy(h);
    LOG("\nBye!\n");
}

void Application::run_sd()
{
    SetConsoleOutputCP(CP_UTF8);
    SetConsoleCP(CP_UTF8);

    SdHandle h = sd_init(ini_file_);
    if (!h)
    {
        LOG_ERR("Failed to initialize SD pipeline from {}\n", ini_file_.c_str());
        return;
    }

    LOG("=== cccc-sd image generation ===\n");
    LOG("Model dir: {}\n", ini_file_);
    LOG("SD params: steps={} cfg={:.2f} size={}x{}\n", sd_steps_, sd_cfg_, sd_width_, sd_height_);
    if (sd_steps_ <= 0)
    {
        LOG_ERR("Invalid --steps {} (must be > 0)\n", sd_steps_);
        sd_destroy(h);
        return;
    }
    if (sd_width_ <= 0 || sd_height_ <= 0 || (sd_width_ % 16) != 0 || (sd_height_ % 16) != 0)
    {
        LOG_ERR("Invalid --width/--height {}x{} (must be positive multiples of 16)\n", sd_width_, sd_height_);
        sd_destroy(h);
        return;
    }
    if (sd_prompt_.empty())
    {
        LOG("Commands: /quit (exit)\n\n");
    }

    auto generate_once = [&](const std::string& prompt, const std::string& out_path)
    {
        int seed = (sd_seed_ < 0) ? (int)(std::random_device{}()) : sd_seed_;
        LOG("Using seed: {}\n", seed);
        int ret = sd_generate(h, prompt, out_path,
            /*steps=*/sd_steps_, /*cfg=*/sd_cfg_,
            /*w=*/sd_width_, /*h=*/sd_height_, /*seed=*/seed, [](int step, int total, void*)
            {
                LOG("  Step {}/{}\r", step, total);
            },
            nullptr);

        if (ret == 0)
        {
            LOG("\nSaved to {}\n\n", out_path);
        }
        else
        {
            LOG_ERR("\nGeneration failed (code {})\n\n", ret);
        }
        return ret;
    };

    if (!sd_prompt_.empty())
    {
        generate_once(sd_prompt_, sd_output_.empty() ? "output.png" : sd_output_);
        sd_destroy(h);
        LOG("\nBye!\n");
        return;
    }

    std::string line;
    while (true)
    {
        LOG("Prompt: ");
        if (!std::getline(std::cin, line)) { break; }
        while (!line.empty() && (line.back() == '\r' || line.back() == '\n' || line.back() == ' '))
        {
            line.pop_back();
        }
        if (line == "/quit" || line == "/exit") { break; }
        if (line.empty()) { continue; }

        generate_once(line, sd_output_.empty() ? "output.png" : sd_output_);
    }

    sd_destroy(h);
    LOG("\nBye!\n");
}

// ─────────────────────────────────────────────────────────────────────────────
//  Agent
// ─────────────────────────────────────────────────────────────────────────────
void Application::run_agent()
{
    SetConsoleOutputCP(CP_UTF8);
    SetConsoleCP(CP_UTF8);

    LlmHandle h = llm_init(ini_file_, agent_system_prompt());
    if (!h)
    {
        LOG_ERR("Failed to initialize LLM from {}\n", ini_file_.c_str());
        return;
    }

    AgentHandle ag = agent_create(h);
    LOG("=== cccc-agent ===\nTask: {}\n\n", agent_task_);

    agent_set_stream_callback(ag,
        [](const std::string& tok, void*) { LOG("{}", tok); },
        nullptr);

    const std::string& task = agent_task_.empty()
        ? std::string("What would you like me to do?")
        : agent_task_;
    agent_run(ag, task, 0, 0);

    agent_destroy(ag);
    llm_destroy(h);
    LOG("\nDone.\n");
}

}    // namespace cccc
