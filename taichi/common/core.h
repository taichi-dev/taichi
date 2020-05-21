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

// Avoid dependency on glibc 2.27
#if defined(TI_PLATFORM_LINUX) && defined(TI_ARCH_x64)
    // objdump -T libtaichi_core.so| grep  GLIBC_2.27
    __asm__(".symver logf,logf@GLIBC_2.2.5");
__asm__(".symver powf,powf@GLIBC_2.2.5");
__asm__(".symver expf,expf@GLIBC_2.2.5");
#endif

// Compilers

// MSVC
#if defined(_MSC_VER)
#define TI_COMPILER_MSVC
#endif

// MINGW
#if defined(__MINGW64__)
#define TI_COMPILER_MINGW
#endif

// gcc
#if defined(__GNUC__)
#define TI_COMPILER__GCC
#endif

// clang
#if defined(__clang__)
#define TI_COMPILER_CLANG
#endif

#if defined(TI_COMPILER_MSVC)
#define TI_ALIGNED(x) __declspec(align(x))
#else
#define TI_ALIGNED(x) __attribute__((aligned(x)))
#endif

#if __cplusplus >= 201703L
#define TI_CPP17
#else
#if defined(TI_COMPILER_CLANG)
static_assert(false, "For clang compilers, use -std=c++17");
#endif
static_assert(__cplusplus >= 201402L, "C++14 required.");
#define TI_CPP14
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
#include "taichi/platform/windows/windows.h"
#pragma warning(pop)
#include <intrin.h>
#define TI_EXPORT __declspec(dllexport)
#else
#define __FILENAME__ \
  (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#define TI_EXPORT
#endif
#define TI_P(x) \
  { TI_INFO("{}", taichi::TextSerializer::serialize(#x, (x))); }

#ifndef _WIN64
#define sscanf_s sscanf
#define sprintf_s sprintf
#endif

#undef assert
#ifdef _WIN64
#ifndef TI_PASS_EXCEPTION_TO_PYTHON
// For Visual Studio debugging...
#define DEBUG_TRIGGER __debugbreak()
#else
#define DEBUG_TRIGGER
#endif
#else
#define DEBUG_TRIGGER
#endif

#define TI_STATIC_ASSERT(x) static_assert((x), #x);
#define TI_ASSERT(x) TI_ASSERT_INFO((x), #x)
#define TI_ASSERT_INFO(x, ...)             \
  {                                        \
    bool ___ret___ = static_cast<bool>(x); \
    if (!___ret___) {                      \
      TI_ERROR(__VA_ARGS__);               \
    }                                      \
  }
#define TI_NOT_IMPLEMENTED TI_ERROR("Not supported.");

#define TI_NAMESPACE_BEGIN namespace taichi {
#define TI_NAMESPACE_END }

#define TLANG_NAMESPACE_BEGIN \
  namespace taichi {          \
  namespace lang {

#define TLANG_NAMESPACE_END \
  }                         \
  }

TI_EXPORT void taichi_raise_assertion_failure_in_python(const char *msg);

TI_NAMESPACE_BEGIN

//******************************************************************************
//                                 System State
//******************************************************************************

class CoreState {
 public:
  bool python_imported = false;
  bool trigger_gdb_when_crash = false;

  static CoreState &get_instance();

  static void set_python_imported(bool val) {
    get_instance().python_imported = val;
  }

  static void set_trigger_gdb_when_crash(bool val) {
    get_instance().trigger_gdb_when_crash = val;
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
#define TI_FORCE_INLINE __forceinline
#else
#define TI_FORCE_INLINE inline __attribute__((always_inline))
#endif

using float32 = float;
using float64 = double;

#ifdef TI_USE_DOUBLE
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

TI_EXPORT void print_traceback();

TI_NAMESPACE_END
//******************************************************************************
//                           Meta-programming
//******************************************************************************

#include "taichi/util/meta.h"

//******************************************************************************
//                               Logging
//******************************************************************************

// Must include "spdlog/common.h" to define SPDLOG_HEADER_ONLY
// before including "spdlog/fmt/fmt.h"
#include "spdlog/common.h"
#include "spdlog/fmt/fmt.h"
namespace spdlog {
class logger;
}

TI_NAMESPACE_BEGIN

#define SPD_AUGMENTED_LOG(X, ...)                                        \
  taichi::logger.X(                                                      \
      fmt::format("[{}:{}@{}] ", __FILENAME__, __FUNCTION__, __LINE__) + \
      fmt::format(__VA_ARGS__))

#if defined(TI_PLATFORM_WINDOWS)
#define TI_UNREACHABLE __assume(0);
#else
#define TI_UNREACHABLE __builtin_unreachable();
#endif

#define TI_TRACE(...) SPD_AUGMENTED_LOG(trace, __VA_ARGS__)
#define TI_DEBUG(...) SPD_AUGMENTED_LOG(debug, __VA_ARGS__)
#define TI_INFO(...) SPD_AUGMENTED_LOG(info, __VA_ARGS__)
#define TI_WARN(...) SPD_AUGMENTED_LOG(warn, __VA_ARGS__)
#define TI_ERROR(...)                      \
  {                                        \
    SPD_AUGMENTED_LOG(error, __VA_ARGS__); \
    TI_UNREACHABLE;                        \
  }
#define TI_CRITICAL(...)                      \
  {                                           \
    SPD_AUGMENTED_LOG(critical, __VA_ARGS__); \
    TI_UNREACHABLE;                           \
  }

#define TI_TRACE_IF(condition, ...) \
  if (condition) {                  \
    TI_TRACE(__VA_ARGS__);          \
  }
#define TI_TRACE_UNLESS(condition, ...) \
  if (!(condition)) {                   \
    TI_TRACE(__VA_ARGS__);              \
  }
#define TI_DEBUG_IF(condition, ...) \
  if (condition) {                  \
    TI_DEBUG(__VA_ARGS__);          \
  }
#define TI_DEBUG_UNLESS(condition, ...) \
  if (!(condition)) {                   \
    TI_DEBUG(__VA_ARGS__);              \
  }
#define TI_INFO_IF(condition, ...) \
  if (condition) {                 \
    TI_INFO(__VA_ARGS__);          \
  }
#define TI_INFO_UNLESS(condition, ...) \
  if (!(condition)) {                  \
    TI_INFO(__VA_ARGS__);              \
  }
#define TI_WARN_IF(condition, ...) \
  if (condition) {                 \
    TI_WARN(__VA_ARGS__);          \
  }
#define TI_WARN_UNLESS(condition, ...) \
  if (!(condition)) {                  \
    TI_WARN(__VA_ARGS__);              \
  }
#define TI_ERROR_IF(condition, ...) \
  if (condition) {                  \
    TI_ERROR(__VA_ARGS__);          \
  }
#define TI_ERROR_UNLESS(condition, ...) \
  if (!(condition)) {                   \
    TI_ERROR(__VA_ARGS__);              \
  }
#define TI_CRITICAL_IF(condition, ...) \
  if (condition) {                     \
    TI_CRITICAL(__VA_ARGS__);          \
  }
#define TI_CRITICAL_UNLESS(condition, ...) \
  if (!(condition)) {                      \
    TI_CRITICAL(__VA_ARGS__);              \
  }

#define TI_STOP TI_ERROR("Stopping here")
#define TI_TAG TI_INFO("Tagging here")

#define TI_LOG_SET_PATTERN(x) spdlog::set_pattern(x);

#define TI_FLUSH_LOGGER \
  { taichi::logger.flush(); };

class Logger {
 private:
  std::shared_ptr<spdlog::logger> console;
  int level;

 public:
  Logger();
  void trace(const std::string &s);
  void debug(const std::string &s);
  void info(const std::string &s);
  void warn(const std::string &s);
  void error(const std::string &s, bool raise_exception = true);
  void critical(const std::string &s);
  void flush();
  void set_level(const std::string &level);
  bool is_level_effective(const std::string &level_name);
  int get_level();
  static int level_enum_from_string(const std::string &level);
  void set_level_default();
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

TI_NAMESPACE_END

//******************************************************************************
//                               Serialization
//******************************************************************************

#include "serialization.h"

//******************************************************************************
//                                   Misc.
//******************************************************************************

TI_NAMESPACE_BEGIN

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

#define TI_DEFER(x) taichi::DeferedExecution _defered([&]() { x; });

bool is_release();

std::string get_repo_dir();

std::string get_python_package_dir();

void set_python_package_dir(const std::string &dir);

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

  TI_ASSERT(!path.empty());
  if (path[0] == '$') {
    path = assets_dir() + path.substr(1, (int)path.size() - 1);
  } else if (path[0] != '.' && path[0] != '/' &&
             (path.size() >= 2 && path[1] != ':')) {
    path = get_repo_dir() + "/" + path;
  }
  return path;
}

std::string cpp_demangle(const std::string &mangled_name);

int get_version_major();

int get_version_minor();

int get_version_patch();

std::string get_version_string();

std::string get_commit_hash();

std::string get_cuda_version_string();

TI_NAMESPACE_END
