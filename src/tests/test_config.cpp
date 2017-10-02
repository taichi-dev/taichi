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
    assert(dot(Vector2i(1, 2), Vector2i(3, 2)) == 7);
    assert((fract(Vector2(1.3f, 2.7f)) - Vector2(0.3f, 0.7f)).length2() <
           1e-10f);
    assert(Vector2(1.3f, 2.7f).sin() == Vector2(sin(1.3f), sin(2.7f)));

    assert(Matrix3(3.0f) + Matrix3(4.0f) == Matrix3(7.0f));
    assert(Matrix3(3.0f) + Matrix3(Vector3(1, 2, 3)) ==
           Matrix3(Vector3(4, 5, 6)));

    assert(Matrix2(Vector2(1, 2)) * Vector2(2, 3) == Vector2(2, 6));
    assert(Matrix3(Vector3(1, 2, 3)) * Vector3(2, 3, 4) == Vector3(2, 6, 12));
    assert(Matrix4(Vector4(1, 2, 3, 4)) * Vector4(2, 3, 4, 5) ==
           Vector4(2, 6, 12, 20));

    test_matrix_inverse<2>();
    test_matrix_inverse<3>();
    test_matrix_inverse<4>();

    test_qr();

    std::cout << "Passed." << std::endl;
  }
};

TC_IMPLEMENTATION(Task, TestConfig, "test_config")

TC_NAMESPACE_END
