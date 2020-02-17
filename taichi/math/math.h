/*******************************************************************************
    Copyright (c) The Taichi Authors (2016- ). All Rights Reserved.
    The use of this software is governed by the LICENSE file.
*******************************************************************************/

#pragma once

#include <taichi/common/util.h>
#include "geometry_util.h"
#include "array.h"
#include "linalg.h"

TI_NAMESPACE_BEGIN

namespace math {
template <typename T>
TI_FORCE_INLINE T degrees(T rad) {
  return rad * (type::element<T>(180) / pi);
}

template <typename T>
TI_FORCE_INLINE T radians(T deg) {
  return deg * (pi / type::element<T>(180));
}

// clang-format off
template <typename F, typename T>
inline T map(const T &t, const F &f) {
  T ret;
  TI_STATIC_IF(type::is_VectorND<T>()) {
    for (int i = 0; i < std::decay_t<decltype(t)>::dim; i++) {
      ret[i] = f(t[i]);
    }
  }
  TI_STATIC_ELSE{
    TI_STATIC_IF(type::is_MatrixND<T>()){
      for (int i = 0; i < std::decay_t<decltype(t)>::dim; i++){
        for (int j = 0; j < std::decay_t<decltype(t)>::dim; j++){
          ret[i][j] = f(t(i, j));
        }
      }
    }
    TI_STATIC_ELSE {
      ret = f(t);
    }
    TI_STATIC_END_IF
  }
  TI_STATIC_END_IF
  return ret;
}
// clang-format on

// clang-format off
template <typename T>
inline type::element<T> maximum(const T &t) {
  typename type::element<T> ret;
  TI_STATIC_IF(type::is_VectorND<T>()) {
    ret = t(0);
    for (int i = 1; i < T::dim; i++) {
      ret = std::max(ret, t(i));
    }
  }
  TI_STATIC_ELSE {
    TI_STATIC_IF(type::is_MatrixND<T>()) {
      ret = t(0, 0);
      for (int i = 0; i < T::dim; i++){
        for (int j = 0; j < T::dim; j++){
          ret = std::max(ret, t(i, j));
        }
      }
    }
    TI_STATIC_ELSE {
      ret = t;
    }
    TI_STATIC_END_IF
  }
  TI_STATIC_END_IF
  return ret;
}
// clang-format on

// clang-format off
template <typename T>
inline type::element<T> minimum(const T &t) {
  typename type::element<T> ret;
  TI_STATIC_IF(type::is_VectorND<T>()) {
    ret = t(0);
    for (int i = 1; i < T::dim; i++) {
      ret = std::min(ret, t(i));
    }
  }
  TI_STATIC_ELSE {
    TI_STATIC_IF(type::is_MatrixND<T>()) {
      ret = t(0, 0);
      for (int i = 0; i < T::dim; i++){
        for (int j = 0; j < T::dim; j++){
          ret = std::min(ret, t(i, j));
        }
      }
    }
    TI_STATIC_ELSE {
      ret = t;
    }
    TI_STATIC_END_IF
  }
  TI_STATIC_END_IF
  return ret;
}
// clang-format on

// clang-format off
template <typename T>
inline type::element<T> sum(const T &t) {
  typename type::element<T> ret = 0;
  TI_STATIC_IF(type::is_VectorND<T>()) {
    for (int i = 0; i < std::decay_t<decltype(t)>::dim; i++) {
      ret += t(i);
    }
  }
  TI_STATIC_ELSE {
    TI_STATIC_IF(type::is_MatrixND<T>()) {
      for (int i = 0; i < std::decay_t<decltype(t)>::dim; i++){
        for (int j = 0; j < std::decay_t<decltype(t)>::dim; j++){
          ret += t(i, j);
        }
      }
    }
    TI_STATIC_ELSE {
      ret = t;
    }
    TI_STATIC_END_IF
  }
  TI_STATIC_END_IF
  return ret;
}

template <typename T>
inline type::element<T> prod(const T &t) {
  typename type::element<T> ret = 1;
  TI_STATIC_IF(type::is_VectorND<T>()) {
    for (int i = 0; i < T::dim; i++) {
      ret *= t(i);
    }
  } TI_STATIC_ELSE {
    TI_STATIC_IF(type::is_MatrixND<T>()) {
      for (int i = 0; i < T::dim; i++) {
        for (int j = 0; j < T::dim; j++) {
          ret *= t(i, j);
        }
      }
    } TI_STATIC_ELSE {
      ret = t;
    } TI_STATIC_END_IF
  } TI_STATIC_END_IF
  return ret;
}
// clang-format on

#define TI_MAKE_VECTORIZED_FROM_STD(op)                  \
  template <typename T>                                  \
  inline T op(const T &t) {                              \
    using Elem = typename type::element<decltype(t)>;    \
    return map(t, static_cast<Elem (*)(Elem)>(std::op)); \
  }

TI_MAKE_VECTORIZED_FROM_STD(abs);
TI_MAKE_VECTORIZED_FROM_STD(log);
TI_MAKE_VECTORIZED_FROM_STD(exp);
TI_MAKE_VECTORIZED_FROM_STD(sin);
TI_MAKE_VECTORIZED_FROM_STD(cos);
TI_MAKE_VECTORIZED_FROM_STD(tan);
TI_MAKE_VECTORIZED_FROM_STD(asin);
TI_MAKE_VECTORIZED_FROM_STD(acos);
TI_MAKE_VECTORIZED_FROM_STD(atan);
TI_MAKE_VECTORIZED_FROM_STD(tanh);
TI_MAKE_VECTORIZED_FROM_STD(ceil);
TI_MAKE_VECTORIZED_FROM_STD(floor);
TI_MAKE_VECTORIZED_FROM_STD(sqrt);

template <typename T>
TI_FORCE_INLINE
    typename std::enable_if_t<!std::is_floating_point<T>::value, bool>
    equal(const T &A, const T &B, float64 tolerance) {
  return maximum(abs(A - B)) <= tolerance;
}

template <typename T>
TI_FORCE_INLINE
    typename std::enable_if_t<std::is_floating_point<T>::value, bool>
    equal(const T &A, const T &B, float64 tolerance) {
  return std::abs(A - B) <= tolerance;
}

}  // namespace math

template <int dim, typename T, InstSetExt ISE>
template <typename T_,
          typename std::enable_if_t<std::is_same<T_, int>::value, int>>
VectorND<dim, T, ISE>::VectorND(const TIndex<dim> &ind) {
  TI_STATIC_ASSERT(2 <= dim && dim <= 3);
  d[0] = ind.i;
  d[1] = ind.j;
  TI_STATIC_IF(dim == 3) {
    this->d[2] = ind.k;
  }
  TI_STATIC_END_IF
};

TI_NAMESPACE_END
