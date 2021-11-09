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

public:
    void run();
    static int getFloatPresicion();
};

}    // namespace cccc