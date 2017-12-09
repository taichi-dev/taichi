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

namespace math {
template <typename T>
TC_FORCE_INLINE T degrees(T rad) {
  return rad * (type::element<T>(180) / pi);
}

template <typename T>
TC_FORCE_INLINE T radians(T deg) {
  return deg * (pi / type::element<T>(180));
}

// clang-format off
template <typename F, typename T>
T map(const T &t, const F &f) {
  T ret;
  TC_STATIC_IF(type::is_VectorND<T>()) {
    for (int i = 0; i < T::dim; i++) {
      ret[i] = f(t[i]);
    }
  }
  TC_STATIC_ELSE{
    TC_STATIC_IF(type::is_MatrixND<T>()){
      for (int i = 0; i < T::dim; i++){
        for (int j = 0; j < T::dim; j++){
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
// clang-format on

// clang-format off
template <typename T>
type::element<T> maximum(const T &t) {
  typename type::element<T> ret;
  TC_STATIC_IF(type::is_VectorND<T>()) {
    ret = t(0);
    for (int i = 1; i < T::dim; i++) {
      ret = std::max(ret, t(i));
    }
  }
  TC_STATIC_ELSE {
    ret = t(0, 0);
    TC_STATIC_IF(type::is_MatrixND<T>()) {
      for (int i = 0; i < T::dim; i++){
        for (int j = 0; j < T::dim; j++){
          ret = std::max(ret, t(i, j));
        }
      }
    }
    TC_STATIC_ELSE {
      ret = t;
    }
    TC_STATIC_END_IF
  }
  TC_STATIC_END_IF
  return ret;
}
// clang-format on

#define TC_MAKE_VECTORIZED_FROM_STD(op)                  \
  auto op = [](const auto &t) {                          \
    using Elem = typename type::element<decltype(t)>;    \
    return map(t, static_cast<Elem (*)(Elem)>(std::op)); \
  };

TC_MAKE_VECTORIZED_FROM_STD(abs);
TC_MAKE_VECTORIZED_FROM_STD(log);
TC_MAKE_VECTORIZED_FROM_STD(exp);
TC_MAKE_VECTORIZED_FROM_STD(sin);
TC_MAKE_VECTORIZED_FROM_STD(cos);
TC_MAKE_VECTORIZED_FROM_STD(tan);
TC_MAKE_VECTORIZED_FROM_STD(asin);
TC_MAKE_VECTORIZED_FROM_STD(acos);
TC_MAKE_VECTORIZED_FROM_STD(atan);
TC_MAKE_VECTORIZED_FROM_STD(tanh);
TC_MAKE_VECTORIZED_FROM_STD(ceil);
TC_MAKE_VECTORIZED_FROM_STD(floor);
TC_MAKE_VECTORIZED_FROM_STD(sqrt);

template <typename T>
inline bool equal(const T &A,
                  const T &B,
                  float64 tolerance) {
  return maximum(abs(A - B)) < tolerance;
}

} // namespace math

TC_NAMESPACE_END
