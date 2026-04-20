#include "Log.h"

#include <thread>

#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#include <winuser.h>
#endif

namespace cccc
{
void showMessageBox(const std::string& str)
{
#ifdef _WIN32
    std::thread th([str]()
        {
            MessageBoxA(NULL, str.c_str(), "cccc", MB_ICONERROR);
        });
    th.detach();
    //MessageBoxA(NULL, str.c_str(), "cccc", MB_ICONERROR);
    Sleep(30000);
#endif
}

void fatalError(const std::string& str)
{
    LOG_ERR("{}", str);
    showMessageBox(str);
    throw std::runtime_error(str);
}
}    //namespace cccc
