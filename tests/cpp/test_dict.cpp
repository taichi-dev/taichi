/*******************************************************************************
    Copyright (c) The Taichi Authors (2016- ). All Rights Reserved.
    The use of this software is governed by the LICENSE file.
*******************************************************************************/

#include "taichi/common/dict.h"
#include "taichi/util/testing.h"

TI_NAMESPACE_BEGIN

TI_TEST("dict") {
  SECTION("config") {
    Dict dict;

    dict.set("int_a", 123);
    TI_CHECK(dict.get<int>("int_a") == 123);

    dict.set("uint_a", 125);
    TI_CHECK(dict.get<int>("uint_a") == 125);

    dict.set("float_a", 1.5_f32);
    TI_CHECK_EQUAL(dict.get<float32>("float_a"), 1.5_f32, 1e-6_f);

    dict.set("double_b", 0.125_f64);
    TI_CHECK_EQUAL(dict.get<float64>("double_b"), 0.125_f64, 1e-6_f);

    dict.set("vec_int", Vector3i(4, 6, 3));
    TI_CHECK(dict.get<Vector3i>("vec_int") == Vector3i(4, 6, 3));

    dict.set("str", "Hello");
    TI_CHECK(dict.get<std::string>("str") == "Hello");
  };
}

TI_NAMESPACE_END
