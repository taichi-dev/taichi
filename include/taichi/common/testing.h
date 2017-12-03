/*******************************************************************************
    Taichi - Physically based Computer Graphics Library

    Copyright (c) 2017 Yuanming Hu <yuanmhu@gmail.com>

    All rights reserved. Use of this source code is governed by
    the MIT license as written in the LICENSE file.
*******************************************************************************/

#pragma once

#include "util.h"
#define CATCH_CONFIG_RUNNER
#define BENCHMARK CATCH_BENCHMARK
#include <catch.hpp>
#undef BENCHMARK

TC_NAMESPACE_BEGIN

#define TC_TEST(x) TEST_CASE(x, ("[" x "]"))

TC_NAMESPACE_END
