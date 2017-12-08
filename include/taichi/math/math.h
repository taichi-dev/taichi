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
TC_FORCE_INLINE T degrees(T rad) {
  return rad * (type::element<T>(180) / pi);
}

template <typename T>
TC_FORCE_INLINE T radians(T deg) {
  return deg * (pi / type::element<T>(180));
}

template <typename T>
T abs(const T &t) {
  return std::abs(t);
}

template <typename F, typename T>
T map(const T &t, const F &f) {
  T ret;
  TC_STATIC_IF(type::is_VectorND<T>()) {
    for (int i = 0; i < T::dim; i++) {
      ret[i] = f(t[i]);
    }
  }
  TC_STATIC_ELSE {
    TC_STATIC_IF(type::is_MatrixND<T>()){
      for (int i = 0; i < T::dim; i++) {
        for (int j = 0; j < T::dim; j++) {
          ret[i][j] = f(t(i, j));
        }
      }
    }
    TC_STATIC_ELSE {
      ret = f(t);
    }
    TC_STATIC_END_IF
  }
  TC_STATIC_END_IF
  return ret;
}

template <int dim, typename T>
inline bool equal(const MatrixND<dim, T> &A,
                  const MatrixND<dim, T> &B,
                  T tolerance) {
  for (int i = 0; i < dim; i++) {
    for (int j = 0; j < dim; j++) {
      if (abs(A(i, j) - B(i, j)) > tolerance) {
        return false;
      }
    }
  }
  return true;
}

template <int dim, typename T>
inline bool equal(const VectorND<dim, T> &A,
                  const VectorND<dim, T> &B,
                  T tolerance) {
  for (int i = 0; i < dim; i++) {
    if (abs(A(i) - B(i)) > tolerance) {
      return false;
    }
  }
  return true;
}

template <int dim, typename T>
inline bool equal(const T &A, const T &B, T tolerance) {
  if (abs(A - B) > tolerance)
    return false;
  return true;
}

TC_NAMESPACE_END
