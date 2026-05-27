#pragma once
#include <string>

namespace cccc
{

class Application
{
private:
    bool loop_ = true;

public:
    std::string ini_file_;
    std::string add_option_string_;
    std::string replace_string_;
    std::string sd_prompt_;
    std::string sd_output_ = "output.png";
    int sd_seed_ = -1;  // -1 means random
    int sd_steps_ = 9;
    float sd_cfg_ = 7.5f;
    int sd_width_ = 512;
    int sd_height_ = 512;
    std::string agent_task_;

public:
    void run();
    void run_llm();
    void run_sd();
    void run_agent();
};

}    // namespace cccc