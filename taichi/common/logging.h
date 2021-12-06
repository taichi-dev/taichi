#pragma once

#include <functional>
#include <cstring>

// This is necessary for TI_UNREACHABLE
#include "taichi/common/platform_macros.h"

// Must include "spdlog/common.h" to define SPDLOG_HEADER_ONLY
// before including "spdlog/fmt/fmt.h"
#include "spdlog/common.h"
#include "spdlog/fmt/fmt.h"
namespace spdlog {
class logger;
}

#ifdef _WIN64
#define __FILENAME__ \
  (strrchr(__FILE__, '\\') ? strrchr(__FILE__, '\\') + 1 : __FILE__)
#else
#define __FILENAME__ \
  (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#endif

#define SPD_AUGMENTED_LOG(X, ...)                                        \
  taichi::Logger::get_instance().X(                                      \
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

#define TI_ASSERT(x) TI_ASSERT_INFO((x), "Assertion failure: " #x)
#define TI_ASSERT_INFO(x, ...)             \
  {                                        \
    bool ___ret___ = static_cast<bool>(x); \
    if (!___ret___) {                      \
      TI_ERROR(__VA_ARGS__);               \
    }                                      \
  }
#define TI_NOT_IMPLEMENTED TI_ERROR("Not supported.");

#define TI_STOP TI_ERROR("Stopping here")
#define TI_TAG TI_INFO("Tagging here")

#define TI_LOG_SET_PATTERN(x) spdlog::set_pattern(x);

#define TI_FLUSH_LOGGER \
  { taichi::Logger::get_instance().flush(); };

#define TI_P(x) \
  { TI_INFO("{}", taichi::TextSerializer::serialize(#x, (x))); }

namespace taichi {

class Logger {
 private:
  std::shared_ptr<spdlog::logger> console_;
  int level_;
  std::function<void()> print_stacktrace_fn_;

  Logger();

 public:
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

  // This is mostly to decouple the implementation.
  void set_print_stacktrace_func(std::function<void()> print_fn);

  static Logger &get_instance();
};

}  // namespace taichi
