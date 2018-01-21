/*******************************************************************************
    Copyright (c) The Taichi Authors (2016- ). All Rights Reserved.
    The use of this software is governed by the LICENSE file.
*******************************************************************************/

#pragma once

#include <taichi/common/interface.h>

#include <taichi/math/math.h>
#include <taichi/math/array_2d.h>

#include <taichi/geometry/primitives.h>

TC_NAMESPACE_BEGIN

template <int n, typename T>
using VectorLengthed = std::conditional_t<n != 1, VectorND<n, T>, T>;

template <int n, int m, typename T>
using VectorFunction =
    std::function<VectorLengthed<m, T>(VectorLengthed<n, T>)>;

using Function11 = VectorFunction<1, 1, real>;
using Function12 = VectorFunction<1, 2, real>;
using Function13 = VectorFunction<1, 3, real>;

using Function21 = VectorFunction<2, 1, real>;
using Function22 = VectorFunction<2, 2, real>;
using Function23 = VectorFunction<2, 3, real>;

using Function31 = VectorFunction<3, 1, real>;
using Function32 = VectorFunction<3, 2, real>;
using Function33 = VectorFunction<3, 3, real>;

class Mesh3D {
 public:
  // norm and uv can be null
  static std::vector<Triangle> generate(const Vector2i res,
                                        const Function23 *surf,
                                        const Function23 *norm,
                                        const Function22 *uv,
                                        bool smooth_normal);
};

TC_NAMESPACE_END
