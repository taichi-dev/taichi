/*******************************************************************************
    Taichi - Physically based Computer Graphics Library

    Copyright (c) 2016 Yuanming Hu <yuanmhu@gmail.com>

    All rights reserved. Use of this source code is governed by
    the MIT license as written in the LICENSE file.
*******************************************************************************/

#include <taichi/common/util.h>
#include <taichi/common/task.h>
#include <taichi/math/array.h>

TC_NAMESPACE_BEGIN

class TestArray : public Task {
  virtual void run(const std::vector<std::string> &parameters) {
    using Array = Array2D<real>;
    Array A(Vector2i(5, 6));
    Array B(A);
    P(A.get_res());
    P(B.get_res());
  }
};

TC_IMPLEMENTATION(Task, TestArray, "test_array")

TC_NAMESPACE_END
