/*******************************************************************************
    Copyright (c) The Taichi Authors (2016- ). All Rights Reserved.
    The use of this software is governed by the LICENSE file.
*******************************************************************************/

#pragma once

#include "util.h"
#define BENCHMARK CATCH_BENCHMARK
#include <catch.hpp>
#undef BENCHMARK

TC_NAMESPACE_BEGIN

#define TC_CHECK_EQUAL(A, B, tolerance)              \
  {                                                  \
    if (!taichi::math::equal(A, B, tolerance)) {     \
      std::cout << A << std::endl << B << std::endl; \
    }                                                \
    CHECK(taichi::math::equal(A, B, tolerance));     \
  }

#define TC_ASSERT_EQUAL(A, B, tolerance)             \
  {                                                  \
    if (!taichi::math::equal(A, B, tolerance)) {     \
      std::cout << A << std::endl << B << std::endl; \
      TC_ERROR(#A " != " #B);                        \
    }                                                \
  }

#define TC_TEST(x) TEST_CASE(x, ("[" x "]"))
#define TC_CHECK(x) CHECK(x)

int run_tests();

TC_NAMESPACE_END
