/*******************************************************************************
    Copyright (c) The Taichi Authors (2016- ). All Rights Reserved.
    The use of this software is governed by the LICENSE file.
*******************************************************************************/

#include <taichi/util.h>
#include <taichi/math.h>
#include <taichi/math/qr_svd.h>

TC_NAMESPACE_BEGIN

class TestConfig : public Task {
  void run() override {
    Config config;

    config.set("int_a", 123);
    assert(config.get<int>("int_a") == 123);

    config.set("uint_a", 125);
    assert(config.get<int>("uint_a") == 125);

    config.set("float_a", 1.5_f32);
    assert(std::abs(config.get<float32>("float_a") - 1.5_f32) < 1e-6_f);

    config.set("double_b", 0.125_f64);
    assert(std::abs(config.get<float64>("double_b") - 0.125_f32) < 1e-6_f);

    config.set("vec_int", Vector3i(4, 6, 3));
    assert(config.get<Vector3i>("vec_int") == Vector3i(4, 6, 3));

    config.set("str", "Hello");
    assert(config.get<std::string>("str") == "Hello");

    std::cout << "Passed." << std::endl;
  }
};

TC_IMPLEMENTATION(Task, TestConfig, "test_config")

TC_NAMESPACE_END
