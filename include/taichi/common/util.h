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
#include <intrin.h>
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
#ifdef _WIN64
#define DEBUG_TRIGGER __debugbreak()
#else
#define DEBUG_TRIGGER getchar()
#endif
#define assert(x) {bool ret = static_cast<bool>(x); if (!ret) {printf("%s@(Ln %d): Assertion Failed. [%s]\n", __FILENAME__, __LINE__, #x); std::cout << std::flush; DEBUG_TRIGGER; exit(-1);}}
#define assert_info(x, info) {bool ___ret___ = static_cast<bool>(x); if (!___ret___) {printf("%s@(Ln %d): Assertion Failed. [%s]\n", __FILENAME__, __LINE__, &((info)[0])); std::cout << std::flush; DEBUG_TRIGGER; exit(-1);}}
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
typedef unsigned __int64 uint64;
#define TC_FORCE_INLINE __forceinline
#else
typedef long long int64;
typedef unsigned long long uint64;
#define TC_FORCE_INLINE __attribute__((always_inline))
#endif

// Check for inf, nan?
// #define CV_ON

#include <type_traits>

template<typename T, typename U>
struct is_type_same : std::false_type { };

template<typename T>
struct is_type_same<T, T> : std::true_type { };

template<typename T, typename U>
constexpr bool same_type() { return is_type_same<T, U>::value; }

template<typename T, typename U>
constexpr bool same_type(const U &u) { return is_type_same<T, U>::value; }
