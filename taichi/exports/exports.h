#pragma once

#ifndef _CRT_SECURE_NO_WARNINGS
#define _CRT_SECURE_NO_WARNINGS
#endif

// https://gcc.gnu.org/wiki/Visibility
#if defined _WIN32 || defined _WIN64 || defined __CYGWIN__
#ifdef __GNUC__
#define TI_DLL_EXPORT __attribute__((dllexport))
#define TI_API_CALL
#else
#define TI_DLL_EXPORT __declspec(dllexport)
#define TI_API_CALL __stdcall
#endif  //  __GNUC__
#else
#define TI_DLL_EXPORT __attribute__((visibility("default")))
#define TI_API_CALL
#endif  // defined _WIN32 || defined _WIN64 || defined __CYGWIN__

// Windows
#if defined(_WIN64)
#define TI_PLATFORM_WINDOWS
#endif

#if defined(_WIN32) && !defined(_WIN64)
static_assert(false, "32-bit Windows systems are not supported")
#endif

// Linux
#if defined(__linux__)
#if defined(ANDROID)
#define TI_PLATFORM_ANDROID
#else
#define TI_PLATFORM_LINUX
#endif
#endif

// OSX
#if defined(__APPLE__)
#define TI_PLATFORM_OSX
#endif

#if (defined(TI_PLATFORM_LINUX) || defined(TI_PLATFORM_OSX) || \
     defined(__unix__))
#define TI_PLATFORM_UNIX
#endif

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

TI_DLL_EXPORT int TI_API_CALL ticore_hello_world(const char *extra_msg);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus
