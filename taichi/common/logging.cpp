/*******************************************************************************
    Copyright (c) The Taichi Authors (2016- ). All Rights Reserved.
    The use of this software is governed by the LICENSE file.
*******************************************************************************/

#include "taichi/common/logging.h"

#include "spdlog/common.h"
#include "spdlog/sinks/stdout_color_sinks.h"
#include "spdlog/spdlog.h"
#include "taichi/common/core.h"

namespace taichi {

const auto default_logging_level = "info";

void Logger::set_level(const std::string &level_name) {
  auto new_level = level_enum_from_string(level_name);
  level_ = new_level;
  spdlog::set_level((spdlog::level::level_enum)level_);
}

int Logger::get_level() {
  return level_;
}

bool Logger::is_level_effective(const std::string &level_name) {
  return get_level() <= level_enum_from_string(level_name);
}

int Logger::level_enum_from_string(const std::string &level_name) {
  if (level_name == "trace") {
    return spdlog::level::trace;
  } else if (level_name == "debug") {
    return spdlog::level::debug;
  } else if (level_name == "info") {
    return spdlog::level::info;
  } else if (level_name == "warn") {
    return spdlog::level::warn;
  } else if (level_name == "error") {
    return spdlog::level::err;
  } else if (level_name == "critical") {
    return spdlog::level::critical;
  } else if (level_name == "off") {
    return spdlog::level::off;
  } else {
    TI_ERROR(
        "Unknown logging level [{}]. Levels = trace, debug, info, warn, error, "
        "critical, off",
        level_name);
  }
}

Logger::Logger() {
  console_ = spdlog::stdout_color_mt("console");
  console_->flush_on(spdlog::level::trace);
  TI_LOG_SET_PATTERN("%^[%L %D %X.%e %t] %v%$");

  set_level_default();
}

void Logger::set_level_default() {
  set_level(default_logging_level);
}

void Logger::trace(const std::string &s) {
  console_->trace(s);
}

void Logger::debug(const std::string &s) {
  console_->debug(s);
}

void Logger::info(const std::string &s) {
  console_->info(s);
}

void Logger::warn(const std::string &s) {
  console_->warn(s);
}

void Logger::error(const std::string &s, bool raise_exception) {
  console_->error(s);
  fmt::print("\n\n");
  if (print_stacktrace_fn_) {
    print_stacktrace_fn_();
  }
  if (taichi::CoreState::get_instance().trigger_gdb_when_crash) {
#if defined(TI_PLATFORM_LINUX)
    trash(system(fmt::format("sudo gdb -p {}", PID::get_pid()).c_str()));
#endif
  }
  if (raise_exception)
    throw s;
}

void Logger::critical(const std::string &s) {
  Logger::error(s);  // simply forward to Logger::error since we actually never
                     // use TI_CRITICAL
}

void Logger::flush() {
  console_->flush();
}

void Logger::set_print_stacktrace_func(std::function<void()> print_fn) {
  print_stacktrace_fn_ = print_fn;
}

// static
Logger &Logger::get_instance() {
  // Use the singleton pattern, instead of defining a global variable. This is
  // because I've moved the signal handler registration + pybind11's
  // py::register_exception_translator to
  // taichi/system/hacked_signal_handler.cpp. We instantiate a global
  // HackedSIgnalHandler (in the anonymous namespace), whose constructor
  // registers the signal handlers.

  // This decouples Logger from pybind11. However, it has introduced a problem
  // if we continue to keep a global Logger instance: the construction order
  // between Logger and HackedSIgnalHandler is unspecified, and it actually
  // crashes on my system. So we use the singleton pattern instead.
  static Logger *l = new Logger();
  return *l;
}

}  // namespace taichi
