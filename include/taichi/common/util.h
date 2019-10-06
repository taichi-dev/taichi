/*******************************************************************************
    copyright (c) the taichi authors (2016- ). all rights reserved.
    the use of this software is governed by the license file.
*******************************************************************************/

#pragma once

#ifndef _CRT_SECURE_NO_WARNINGS
#define _CRT_SECURE_NO_WARNINGS
#endif

#include <cstring>
#include <iostream>
#include <cstdlib>
#include <cstdio>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <type_traits>
#include <cstdint>
#include <algorithm>
#include <memory>
#include <csignal>
#include <vector>

//******************************************************************************
//                                 System State
//******************************************************************************

// Reference:
// https://blog.kowalczyk.info/article/j/guide-to-predefined-macros-in-c-compilers-gcc-clang-msvc-etc..html

// Platforms

// Windows
#if defined(_WIN64)
#define TC_PLATFORM_WINDOWS
#endif

#if defined(_WIN32) && !defined(_WIN64)
static_assert(false, "32-bit Windows systems are not supported")
#endif

// Linux
#if defined(__linux__)
#define TC_PLATFORM_LINUX
#endif

// OSX
#if defined(__APPLE__)
#define TC_PLATFORM_OSX
#endif

#if (defined(TC_PLATFORM_LINUX) || defined(TC_PLATFORM_OSX))
#define TC_PLATFORM_UNIX
#endif

// Compilers

// MSVC
#if defined(_MSC_VER)
#define TC_COMPILER_MSVC
#endif

// MINGW
#if defined(__MINGW64__)
#define TC_COMPILER_MINGW
#endif

// gcc
#if defined(__GNUC__)
#define TC_COMPILER__GCC
#endif

// clang
#if defined(__clang__)
#define TC_COMPILER_CLANG
#endif

#if defined(TC_COMPILER_MSVC)
#define TC_ALIGNED(x) __declspec(align(x))
#else
#define TC_ALIGNED(x) __attribute__((aligned(x)))
#endif

#if __cplusplus >= 201703L
#define TC_CPP17
#else
#if defined(TC_COMPILER_CLANG)
static_assert(false, "For clang compilers, use -std=c++17");
#endif
static_assert(__cplusplus >= 201402L, "C++14 required.");
#define TC_CPP14
#endif

// Do not disable assert...
#ifdef NDEBUG
#undef NDEBUG
#endif

#ifdef _WIN64
#define __FILENAME__ \
  (strrchr(__FILE__, '\\') ? strrchr(__FILE__, '\\') + 1 : __FILE__)
#pragma warning(push)
#pragma warning(disable : 4005)
#include <windows.h>
#pragma warning(pop)
#include <intrin.h>
#define TC_EXPORT __declspec(dllexport)
#else
#define __FILENAME__ \
  (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#define TC_EXPORT
#endif
#define TC_P(x) \
  { TC_DEBUG("{}", taichi::TextSerializer::serialize(#x, (x))); }

#ifndef _WIN64
#define sscanf_s sscanf
#define sprintf_s sprintf
#endif

#undef assert
#ifdef _WIN64
#ifndef TC_PASS_EXCEPTION_TO_PYTHON
// For Visual Studio debugging...
#define DEBUG_TRIGGER __debugbreak()
#else
#define DEBUG_TRIGGER
#endif
#else
#define DEBUG_TRIGGER
#endif

#define assert_info(x, info)               \
  {                                        \
    bool ___ret___ = static_cast<bool>(x); \
    if (!___ret___) {                      \
      TC_ERROR(info);                      \
    }                                      \
  }

#define TC_STATIC_ASSERT(x) static_assert((x), #x);
#define TC_ASSERT(x) TC_ASSERT_INFO((x), #x)
#define TC_ASSERT_INFO assert_info
// TODO: this should be part of logging
#define TC_NOT_IMPLEMENTED TC_ERROR("Not Implemented.");

#define TC_NAMESPACE_BEGIN namespace taichi {
#define TC_NAMESPACE_END }

    TC_EXPORT void taichi_raise_assertion_failure_in_python(const char *msg);

TC_NAMESPACE_BEGIN

//******************************************************************************
//                                 System State
//******************************************************************************

class CoreState {
 public:
  bool python_imported = false;
  bool trigger_gdb_when_crash = false;
  bool debug = false;

  static CoreState &get_instance();

  static void set_python_imported(bool val) {
    get_instance().python_imported = val;
  }

  static void set_trigger_gdb_when_crash(bool val) {
    get_instance().trigger_gdb_when_crash = val;
  }

  static void set_debug(bool val) {
    get_instance().debug = val;
  }

  static bool get_debug() {
    return get_instance().debug;
  }
};

//******************************************************************************
//                                 Types
//******************************************************************************

using uchar = unsigned char;

using int8 = int8_t;
using uint8 = uint8_t;

using int16 = int16_t;
using uint16 = uint16_t;

using int32 = int32_t;
using uint32 = uint32_t;
using uint = unsigned int;

using int64 = int64_t;
using uint64 = uint64_t;

#ifdef _WIN64
#define TC_FORCE_INLINE __forceinline
#else
#define TC_FORCE_INLINE inline __attribute__((always_inline))
#endif

using float32 = float;
using float64 = double;

#ifdef TC_USE_DOUBLE
using real = float64;
#else
using real = float32;
#endif

// Float literal for both float32/64
// (Learned from https://github.com/hi2p-perim/lightmetrica-v2)
real constexpr operator"" _f(long double v) {
  return real(v);
}
real constexpr operator"" _f(unsigned long long v) {
  return real(v);
}

float32 constexpr operator"" _f32(long double v) {
  return float32(v);
}
float32 constexpr operator"" _f32(unsigned long long v) {
  return float32(v);
}

float32 constexpr operator"" _fs(long double v) {
  return float32(v);
}
float32 constexpr operator"" _fs(unsigned long long v) {
  return float32(v);
}

float64 constexpr operator"" _f64(long double v) {
  return float64(v);
}
float64 constexpr operator"" _f64(unsigned long long v) {
  return float64(v);
}

float64 constexpr operator"" _fd(long double v) {
  return float64(v);
}
float64 constexpr operator"" _fd(unsigned long long v) {
  return float64(v);
}

TC_EXPORT void print_traceback();

TC_NAMESPACE_END
//******************************************************************************
//                           Meta-programming
//******************************************************************************

#include "meta.h"

//******************************************************************************
//                               Logging
//******************************************************************************
#include "spdlog/fmt/fmt.h"
namespace spdlog {
class logger;
}

TC_NAMESPACE_BEGIN

#define SPD_AUGMENTED_LOG(X, ...)                                        \
  taichi::logger.X(                                                      \
      fmt::format("[{}:{}@{}] ", __FILENAME__, __FUNCTION__, __LINE__) + \
      fmt::format(__VA_ARGS__))

#define TC_TRACE(...) SPD_AUGMENTED_LOG(trace, __VA_ARGS__)
#define TC_DEBUG(...) SPD_AUGMENTED_LOG(debug, __VA_ARGS__)
#define TC_INFO(...) SPD_AUGMENTED_LOG(info, __VA_ARGS__)
#define TC_WARN(...) SPD_AUGMENTED_LOG(warn, __VA_ARGS__)
#define TC_ERROR(...) SPD_AUGMENTED_LOG(error, __VA_ARGS__)
#define TC_CRITICAL(...) SPD_AUGMENTED_LOG(critical, __VA_ARGS__)

#define TC_TRACE_IF(condition, ...) \
  if (condition) {                  \
    TC_TRACE(__VA_ARGS__);          \
  }
#define TC_TRACE_UNLESS(condition, ...) \
  if (!(condition)) {                   \
    TC_TRACE(__VA_ARGS__);              \
  }
#define TC_DEBUG_IF(condition, ...) \
  if (condition) {                  \
    TC_DEBUG(__VA_ARGS__);          \
  }
#define TC_DEBUG_UNLESS(condition, ...) \
  if (!(condition)) {                   \
    TC_DEBUG(__VA_ARGS__);              \
  }
#define TC_INFO_IF(condition, ...) \
  if (condition) {                 \
    TC_INFO(__VA_ARGS__);          \
  }
#define TC_INFO_UNLESS(condition, ...) \
  if (!(condition)) {                  \
    TC_INFO(__VA_ARGS__);              \
  }
#define TC_WARN_IF(condition, ...) \
  if (condition) {                 \
    TC_WARN(__VA_ARGS__);          \
  }
#define TC_WARN_UNLESS(condition, ...) \
  if (!(condition)) {                  \
    TC_WARN(__VA_ARGS__);              \
  }
#define TC_ERROR_IF(condition, ...) \
  if (condition) {                  \
    TC_ERROR(__VA_ARGS__);          \
  }
#define TC_ERROR_UNLESS(condition, ...) \
  if (!(condition)) {                   \
    TC_ERROR(__VA_ARGS__);              \
  }
#define TC_CRITICAL_IF(condition, ...) \
  if (condition) {                     \
    TC_CRITICAL(__VA_ARGS__);          \
  }
#define TC_CRITICAL_UNLESS(condition, ...) \
  if (!(condition)) {                      \
    TC_CRITICAL(__VA_ARGS__);              \
  }

#define TC_STOP TC_ERROR("Stopping here")
#define TC_TAG TC_TRACE("Tagging here")

#define TC_LOG_SET_PATTERN(x) spdlog::set_pattern(x);

#define TC_FLUSH_LOGGER \
  { taichi::logger.flush(); };

class Logger {
  std::shared_ptr<spdlog::logger> console;

 public:
  Logger();
  void trace(const std::string &s);
  void debug(const std::string &s);
  void info(const std::string &s);
  void warn(const std::string &s);
  void error(const std::string &s, bool raise_signal = true);
  void critical(const std::string &s, bool raise_signal = true);
  void flush();
  void set_level(const std::string &level);
};

extern Logger logger;

namespace zip {

void write(std::string fn, const uint8 *data, std::size_t len);
void write(const std::string &fn, const std::string &data);
std::vector<uint8> read(const std::string fn, bool verbose = false);

}  // namespace zip

//******************************************************************************
//                               String Utils
//******************************************************************************

inline std::vector<std::string> split_string(const std::string &s,
                                             const std::string &seperators) {
  std::vector<std::string> ret;
  bool is_seperator[256] = {false};
  for (auto &ch : seperators) {
    is_seperator[(unsigned int)ch] = true;
  }
  int begin = 0;
  for (int i = 0; i <= (int)s.size(); i++) {
    if (is_seperator[(uint8)s[i]] || i == (int)s.size()) {
      ret.push_back(std::string(s.begin() + begin, s.begin() + i));
      begin = i + 1;
    }
  }
  return ret;
}

inline std::string trim_string(const std::string &s) {
  int begin = 0, end = (int)s.size();
  while (begin < end && s[begin] == ' ') {
    begin++;
  }
  while (begin < end && s[end - 1] == ' ') {
    end--;
  }
  return std::string(s.begin() + begin, s.begin() + end);
}

inline bool ends_with(std::string const &str, std::string const &ending) {
  if (ending.size() > str.size())
    return false;
  else
    return std::equal(ending.begin(), ending.end(), str.end() - ending.size());
}

inline bool starts_with(std::string const &str, std::string const &ending) {
  if (ending.size() > str.size())
    return false;
  else
    return std::equal(ending.begin(), ending.end(), str.begin());
}

TC_NAMESPACE_END

//******************************************************************************
//                               Serialization
//******************************************************************************

#include "serialization.h"

//******************************************************************************
//                                   Misc.
//******************************************************************************

TC_NAMESPACE_BEGIN

extern int __trash__;
template <typename T>
void trash(T &&t) {
  static_assert(!std::is_same<T, void>::value, "");
  __trash__ = *reinterpret_cast<uint8 *>(&t);
}

class DeferedExecution {
  std::function<void(void)> statement;

 public:
  DeferedExecution(const std::function<void(void)> &statement)
      : statement(statement) {
  }

  ~DeferedExecution() {
    statement();
  }
};

#define TC_DEFER(x) taichi::DeferedExecution _defered([&]() { x; });

inline bool running_on_windows() {
#if defined(TC_PLATFORM_WINDOWS)
  return true;
#else
  return false;
#endif
}

inline std::string get_repo_dir() {
  auto dir = std::getenv("TAICHI_REPO_DIR");
  if (dir != nullptr || std::string(dir) == "") {
    // release mode. Use ~/.taichi as root
    auto home = std::getenv("HOME");
    TC_ASSERT(home != nullptr);
    return std::string(home) + "/.taichi/";
  } else {
    return std::string(dir);
  }
}

inline std::string assets_dir() {
  return get_repo_dir() + "/assets/";
}

inline std::string absolute_path(std::string path) {
  // If 'path' is actually relative to TAICHI_REPO_DIR, convert it to an
  // absolute one. There are three types of paths:
  //    A. Those who start with / or "C:/" are absolute paths
  //    B. Those who start with "." are relative to cwd
  //    C. Those who start with "$" are relative to assets_dir()
  //    D. Others are relative to $ENV{TAICHI_REPO_DIR}

  TC_ASSERT(!path.empty());
  if (path[0] == '$') {
    path = assets_dir() + path.substr(1, (int)path.size() - 1);
  } else if (path[0] != '.' && path[0] != '/' &&
             (path.size() >= 2 && path[1] != ':')) {
    path = get_repo_dir() + "/" + path;
  }
  return path;
}

std::string cpp_demangle(const std::string &mangled_name);

TC_NAMESPACE_END

#include "asset_manager.h"
