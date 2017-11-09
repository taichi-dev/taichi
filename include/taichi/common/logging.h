/*******************************************************************************
    Taichi - Physically based Computer Graphics Library

    Copyright (c) 2017 Yuanming Hu <yuanmhu@gmail.com>

    All rights reserved. Use of this source code is governed by
    the MIT license as written in the LICENSE file.
*******************************************************************************/

#pragma once

#include "util.h"
#include <spdlog/spdlog.h>

TC_NAMESPACE_BEGIN

extern std::shared_ptr<spdlog::logger> console;

#define SPD_AUGMENTED_LOG(X, ...)                                       \
  taichi::console->X(fmt::format("[{}:{}@{}] ", __FILENAME__, \
                                 __FUNCTION__, __LINE__) +              \
                     fmt::format(__VA_ARGS__))

#define TC_TRACE(...) SPD_AUGMENTED_LOG(trace, __VA_ARGS__)
#define TC_DEBUG(...) SPD_AUGMENTED_LOG(debug, __VA_ARGS__)
#define TC_INFO(...) SPD_AUGMENTED_LOG(info, __VA_ARGS__)
#define TC_WARN(...) SPD_AUGMENTED_LOG(warn, __VA_ARGS__)
#define TC_ERR(...) SPD_AUGMENTED_LOG(error, __VA_ARGS__)
#define TC_CRITICAL(...) SPD_AUGMENTED_LOG(critical, __VA_ARGS__)

#define TC_LOG_SET_PATTERN(x) spdlog::set_pattern(x);

TC_NAMESPACE_END
