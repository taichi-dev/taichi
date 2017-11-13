/*******************************************************************************
    Taichi - Physically based Computer Graphics Library

    Copyright (c) 2016 Yuanming Hu <yuanmhu@gmail.com>

    All rights reserved. Use of this source code is governed by
    the MIT license as written in the LICENSE file.
*******************************************************************************/

#pragma once

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
#include <spdlog/fmt/fmt.h>
#include <memory>

namespace spdlog {
class logger;
}

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
#define TC_PRINT(x)                                           \
  {                                                           \
    std::printf("%s[%d]: %s = ", __FILENAME__, __LINE__, #x); \
    taichi::print(x);                                         \
  };
#define TC_P(x) TC_PRINT(x)

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

#define assert(x) \
  { assert_info(x, ""); }

#define assert_info(x, info)                                         \
  {                                                                  \
    bool ___ret___ = static_cast<bool>(x);                           \
    if (!___ret___) {                                                \
      TC_ERROR(info);                                                \
      taichi::print_traceback();                                     \
      DEBUG_TRIGGER;                                                 \
      taichi_raise_assertion_failure_in_python("Assertion failed."); \
    }                                                                \
  }

#define TC_ASSERT assert
#define TC_ASSERT_INFO assert_info
// TODO: this should be part of logging
#define TC_NOT_IMPLEMENTED TC_ERROR("Not Implemented.");

#define TC_NAMESPACE_BEGIN namespace taichi {
#define TC_NAMESPACE_END }

// Check for inf, nan?
// #define CV_ON

#ifdef CV_ON
#define CV(v)                                              \
  if (abnormal(v)) {                                       \
    for (int i = 0; i < 1; i++)                            \
      printf("Abnormal value %s (Ln %d)\n", #v, __LINE__); \
    taichi::print(v);                                      \
    puts("");                                              \
  }
#else
#define CV(v)
#endif

TC_EXPORT void taichi_raise_assertion_failure_in_python(const char *msg);

TC_NAMESPACE_BEGIN

//******************************************************************************
//                                 Types
//******************************************************************************

using int8 = int8_t;
using uint8 = uint8_t;

using int16 = int16_t;
using uint16 = uint16_t;

using int32 = int32_t;
using uint32 = uint32_t;

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

template <typename T>
inline void print(T t) {
  std::cout << t << std::endl;
}

//******************************************************************************
//                                 Static If
//******************************************************************************

namespace STATIC_IF {
// reference: https://github.com/wichtounet/cpp_utils

struct identity {
  template <typename T>
  T operator()(T &&x) const {
    return std::forward<T>(x);
  }
};

template <bool Cond>
struct statement {
  template <typename F>
  void then(const F &f) {
    f(identity());
  }

  template <typename F>
  void else_(const F &) {
  }
};

template <>
struct statement<false> {
  template <typename F>
  void then(const F &) {
  }

  template <typename F>
  void else_(const F &f) {
    f(identity());
  }
};

template <bool Cond, typename F>
inline statement<Cond> static_if(F const &f) {
  statement<Cond> if_;
  if_.then(f);
  return if_;
}
}

using STATIC_IF::static_if;

#define TC_STATIC_IF(x) taichi::static_if<(x)>([&](const auto& _____) -> void {
#define TC_STATIC_ELSE \
  }).else_([&](const auto &_____) -> void {
#define TC_STATIC_END_IF \
  });

// After we switch to C++17, we should use
// (Note the the behaviour of 'return' is still different.)

/*
#define TC_STATIC_IF(x) if constexpr(x) {
#define TC_STATIC_ELSE \
    } else {
#define TC_STATIC_END_IF \
    }
*/

//******************************************************************************
//                               Logging
//******************************************************************************

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
  void error(const std::string &s);
  void critical(const std::string &s);
  void flush();
};

extern Logger logger;

TC_NAMESPACE_END

//******************************************************************************
//                               Serialization
//******************************************************************************

#include "serialization.h"
