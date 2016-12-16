#pragma once

#include <cstring>
#include <iostream>
#include <cstdlib>
#include <cstdio>
#include <cstddef>
#include <cstdint>
#include <algorithm>

// Do not disable assert...
#ifdef NDEBUG
#undef NDEBUG
#endif

#ifdef _WIN64
#define __FILENAME__ (strrchr(__FILE__, '\\') ? strrchr(__FILE__, '\\') + 1 : __FILE__)
#pragma warning(push)
#pragma warning(disable:4005)
#include <windows.h>
#pragma warning(pop)
#else
#define __FILENAME__ (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#endif
#define PRINT(x) {printf("%s[%d]: %s = ", __FILENAME__, __LINE__, #x); taichi::print(x);};
#define P(x) PRINT(x)

#ifndef _WIN64
#define sscanf_s sscanf
#define sprintf_s sprintf
#endif

#undef assert
#define assert(x) {bool ret = static_cast<bool>(x); if (!ret) {printf("%s@(Ln %d): Assertion Failed. [%s]\n", __FILENAME__, __LINE__, #x); getchar(); exit(-1);}}
#define assert_info(x, info) {bool ret = static_cast<bool>(x); if (!ret) {printf("%s@(Ln %d): Assertion Failed. [%s]\n", __FILENAME__, __LINE__, &((info)[0])); getchar(); exit(-1);}}
#define error(info) assert_info(false, info)
#define NOT_IMPLEMENTED assert_info(false, "Not Implemented!");

#include <boost/foreach.hpp>

namespace boost
{
	// Suggested work-around for https://svn.boost.org/trac/boost/ticket/6131
	namespace BOOST_FOREACH = foreach;
}

#define foreach   BOOST_FOREACH
#define TC_NAMESPACE_BEGIN namespace taichi {
#define TC_NAMESPACE_END }

#ifdef _WIN64
typedef __int64 int64;
#else
typedef long long int64;
#endif

// Check for inf, nan?
// #define CV_ON
