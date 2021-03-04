#pragma once

// TO KEEP THE INCLUDE DEPENDENCY CLEAN, PLEASE DO NOT INCLUDE ANY OTHER
// TAICHI HEADERS INTO THIS ONE.
//
// TODO(#2196): Once we can slim down "taichi/common/core.h", consider moving
// the contents back to core.h and delete this file.
#ifndef _CRT_SECURE_NO_WARNINGS
#define _CRT_SECURE_NO_WARNINGS
#endif

// Windows
#if defined(_WIN64)
#define TI_PLATFORM_WINDOWS
#endif

#if defined(_WIN32) && !defined(_WIN64)
static_assert(false, "32-bit Windows systems are not supported")
#endif

// Linux
#if defined(__linux__)
#define TI_PLATFORM_LINUX
#endif

// OSX
#if defined(__APPLE__)
#define TI_PLATFORM_OSX
#endif

#if (defined(TI_PLATFORM_LINUX) || defined(TI_PLATFORM_OSX))
#define TI_PLATFORM_UNIX
#endif
