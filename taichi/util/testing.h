/*******************************************************************************
    Copyright (c) The Taichi Authors (2016- ). All Rights Reserved.
    The use of this software is governed by the LICENSE file.
*******************************************************************************/

#pragma once

#define BENCHMARK CATCH_BENCHMARK
#include <catch.hpp>
#undef BENCHMARK

namespace taichi {

#define TI_CHECK_EQUAL(A, B, tolerance)              \
  {                                                  \
    if (!taichi::math::equal(A, B, tolerance)) {     \
      std::cout << A << std::endl << B << std::endl; \
    }                                                \
    CHECK(taichi::math::equal(A, B, tolerance));     \
  }

#define TI_ASSERT_EQUAL(A, B, tolerance)             \
  {                                                  \
    if (!taichi::math::equal(A, B, tolerance)) {     \
      std::cout << A << std::endl << B << std::endl; \
      TI_ERROR(#A " != " #B);                        \
    }                                                \
  }

#define TI_TEST(x) TEST_CASE(x, ("[" x "]"))
#define TI_CHECK(x) CHECK(x)
#define TI_TEST_PROGRAM                     \
  auto prog_ = std::make_unique<Program>(); \
  prog_->materialize_runtime();

int run_tests(std::vector<std::string> argv);

}  // namespace taichi
