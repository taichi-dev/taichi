/*******************************************************************************
    Taichi - Physically based Computer Graphics Library

    Copyright (c) 2017 Yuanming Hu <yuanmhu@gmail.com>

    All rights reserved. Use of this source code is governed by
    the MIT license as written in the LICENSE file.
*******************************************************************************/

#pragma once

#include "util.h"
#include <memory>

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
#define TC_ERR(...) SPD_AUGMENTED_LOG(error, __VA_ARGS__)
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
