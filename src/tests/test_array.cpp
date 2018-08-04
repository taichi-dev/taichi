/*******************************************************************************
    Copyright (c) The Taichi Authors (2016- ). All Rights Reserved.
    The use of this software is governed by the LICENSE file.
*******************************************************************************/

#include <taichi/common/util.h>
#include <taichi/common/task.h>
#include <taichi/math/array.h>
#include <taichi/testing.h>

TC_NAMESPACE_BEGIN

TC_TEST("array") {
  using Array = Array2D<real>;
  Array A(Vector2i(5, 6));
  Array B(A);
  TC_P(A.get_res());
  TC_P(B.get_res());
}

TC_NAMESPACE_END
