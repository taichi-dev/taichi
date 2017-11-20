/*******************************************************************************
    Taichi - Physically based Computer Graphics Library

    Copyright (c) 2017 Yuanming Hu <yuanmhu@gmail.com>

    All rights reserved. Use of this source code is governed by
    the MIT license as written in the LICENSE file.
*******************************************************************************/

#pragma once

#include <taichi/common/util.h>
#include <taichi/math/geometry_util.h>
#include <taichi/math/array.h>
#include <taichi/math/vector.h>

TC_NAMESPACE_BEGIN

template <typename T>
inline T degrees(T rad) {
  return rad * (type::element<T>(180) / pi);
}

template <typename T>
inline T radians(T deg) {
  return deg * (pi / type::element<T>(180));
}

TC_NAMESPACE_END
