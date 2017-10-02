/*******************************************************************************
    Taichi - Physically based Computer Graphics Library

    Copyright (c) 2017 Yuanming Hu <yuanmhu@gmail.com>

    All rights reserved. Use of this source code is governed by
    the MIT license as written in the LICENSE file.
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
    assert(config.get<int>("uint_a") == 124);

    config.set("float_a", 1.5_f32);
    assert(config.get<float32>("uint_a") == 1.5_f32);

    config.set("double_b", 0.125_f64);
    assert(config.get<float64>("double64") == 0.125_f32);

    config.set("vec_int", Vector3i(4, 6, 3));
    assert(config.get<Vector3i>("vec_int") == Vector3i(4, 6, 3));


    config.set("str", "Hello");
    assert(config.get<std::string>("str") == "Hello");

    std::cout << "Passed." << std::endl;
  }
};

TC_IMPLEMENTATION(Task, TestConfig, "test_config")

TC_NAMESPACE_END
