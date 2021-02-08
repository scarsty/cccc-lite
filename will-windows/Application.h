#pragma once
#include <string>

namespace cccc
{

class Application
{
public:
    Application();
    virtual ~Application();

    void start();
    void stop();

private:
    bool loop_ = true;
    void callback(void*);
    std::string ini_file_;
    std::string ini_string_;

public:
    void run();
    void mainLoop();

    void setIniFile(std::string ini) { ini_file_ = ini; }
    void setIniString(std::string ini) { ini_string_ = ini; }

    static int getFloatPresicion();
};

}    // namespace cccc