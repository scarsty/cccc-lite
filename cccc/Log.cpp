#include "Log.h"

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
    MessageBoxA(NULL, str.c_str(), "cccc", MB_ICONERROR);
#endif
}

void fatalError(const std::string& str)
{
    LOG_ERR("{}", str);
    showMessageBox(str);
    exit(0);
}
}    //namespace cccc
