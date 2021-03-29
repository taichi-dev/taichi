/*******************************************************************************
    Copyright (c) The Taichi Authors (2016- ). All Rights Reserved.
    The use of this software is governed by the LICENSE file.
*******************************************************************************/
#include "gtest/gtest.h"

#include "taichi/common/dict.h"
#include "taichi/util/testing.h"

namespace taichi {
namespace lang {

TEST(Dict, Config) {
  Dict dict;

  dict.set("int_a", 123);
  EXPECT_EQ(dict.get<int>("int_a"), 123);

  dict.set("uint_a", 125);
  EXPECT_EQ(dict.get<int>("uint_a"), 125);

  dict.set("float_a", 1.5_f32);
  EXPECT_LT(fabs(dict.get<float32>("float_a") - 1.5_f32), 1e-6_f);

  dict.set("double_b", 0.125_f64);
  EXPECT_LT(fabs(dict.get<float64>("double_b") - 0.125_f64), 1e-6_f);

  dict.set("vec_int", Vector3i(4, 6, 3));
  EXPECT_EQ(dict.get<Vector3i>("vec_int"), Vector3i(4, 6, 3));

  dict.set("str", "Hello");
  EXPECT_EQ(dict.get<std::string>("str"), "Hello");
}

}  // namespace lang
}  // namespace taichi
