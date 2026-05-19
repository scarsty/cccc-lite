#pragma once

#ifndef CCCC_EXPORT
#ifdef _WIN32
#define CCCC_EXPORT __declspec(dllexport)
#else
#define CCCC_EXPORT __attribute__((visibility("default")))
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