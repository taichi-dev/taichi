/*******************************************************************************
    Taichi - Physically based Computer Graphics Library

    Copyright (c) 2017 Yuanming Hu <yuanmhu@gmail.com>

    All rights reserved. Use of this source code is governed by
    the MIT license as written in the LICENSE file.
*******************************************************************************/

#include <taichi/common/logging.h>

TC_NAMESPACE_BEGIN

std::shared_ptr<spdlog::logger> console = spdlog::stdout_color_mt("console");

TC_NAMESPACE_END
