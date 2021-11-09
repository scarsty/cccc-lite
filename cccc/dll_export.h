#pragma once

#ifndef DLL_EXPORT
#ifdef _WIN32
#ifdef _WINDLL
#define DLL_EXPORT __declspec(dllexport)
#else
#define DLL_EXPORT
#endif
#else
#define DLL_EXPORT __attribute__((visibility("default")))
#endif
#endif

#ifdef __cplusplus
#define EXTERN_C_BEGIN \
    extern "C" \
    {
#define EXTERN_C_END }
#else
#define EXTERN_C_BEGIN
#define EXTERN_C_END
#endif