/*******************************************************************************
    Copyright (c) The Taichi Authors (2016- ). All Rights Reserved.
    The use of this software is governed by the LICENSE file.
*******************************************************************************/

#pragma once

#include <taichi/common/util.h>
#include "geometry_util.h"
#include "array.h"
#include "vector.h"

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
inline T map(const T &t, const F &f) {
  T ret;
  TC_STATIC_IF(type::is_VectorND<T>()) {
    for (int i = 0; i < std::decay_t<decltype(id(t))>::dim; i++) {
      id(ret)[i] = f(id(t)[i]);
    }
  }
  TC_STATIC_ELSE{
    TC_STATIC_IF(type::is_MatrixND<T>()){
      for (int i = 0; i < std::decay_t<decltype(id(t))>::dim; i++){
        for (int j = 0; j < std::decay_t<decltype(id(t))>::dim; j++){
          id(ret)[i][j] = f(id(t)(i, j));
        }
      }
    }
    TC_STATIC_ELSE {
      id(ret) = f(id(t));
    }
    TC_STATIC_END_IF
  }
  TC_STATIC_END_IF
  return ret;
}
// clang-format on

// clang-format off
template <typename T>
inline type::element<T> maximum(const T &t) {
  typename type::element<T> ret;
  TC_STATIC_IF(type::is_VectorND<T>()) {
    ret = id(t)(0);
    for (int i = 1; i < T::dim; i++) {
      ret = std::max(ret, id(t)(i));
    }
  }
  TC_STATIC_ELSE {
    TC_STATIC_IF(type::is_MatrixND<T>()) {
      ret = t(0, 0);
      for (int i = 0; i < T::dim; i++){
        for (int j = 0; j < T::dim; j++){
          ret = std::max(ret, id(t)(i, j));
        }
      }
    }
    TC_STATIC_ELSE {
      ret = id(t);
    }
    TC_STATIC_END_IF
  }
  TC_STATIC_END_IF
  return ret;
}
// clang-format on

// clang-format off
template <typename T>
inline type::element<T> minimum(const T &t) {
  typename type::element<T> ret;
  TC_STATIC_IF(type::is_VectorND<T>()) {
    ret = t(0);
    for (int i = 1; i < T::dim; i++) {
      ret = std::min(ret, id(t)(i));
    }
  }
  TC_STATIC_ELSE {
    TC_STATIC_IF(type::is_MatrixND<T>()) {
      ret = id(t)(0, 0);
      for (int i = 0; i < T::dim; i++){
        for (int j = 0; j < T::dim; j++){
          ret = std::min(ret, id(t)(i, j));
        }
      }
    }
    TC_STATIC_ELSE {
      ret = id(t);
    }
    TC_STATIC_END_IF
  }
  TC_STATIC_END_IF
  return ret;
}
// clang-format on

// clang-format off
template <typename T>
inline type::element<T> sum(const T &t) {
  typename type::element<T> ret = 0;
  TC_STATIC_IF(type::is_VectorND<T>()) {
    for (int i = 0; i < std::decay_t<decltype(id(t))>::dim; i++) {
      ret += id(t)(i);
    }
  }
  TC_STATIC_ELSE {
    TC_STATIC_IF(type::is_MatrixND<T>()) {
      for (int i = 0; i < std::decay_t<decltype(id(t))>::dim; i++){
        for (int j = 0; j < std::decay_t<decltype(id(t))>::dim; j++){
          ret += id(t)(i, j);
        }
      }
    }
    TC_STATIC_ELSE {
      ret = id(t);
    }
    TC_STATIC_END_IF
  }
  TC_STATIC_END_IF
  return ret;
}

template <typename T>
inline type::element<T> prod(const T &t) {
  typename type::element<T> ret = 1;
  TC_STATIC_IF(type::is_VectorND<T>()) {
    for (int i = 0; i < T::dim; i++) {
      ret *= id(t)(i);
    }
  } TC_STATIC_ELSE {
    TC_STATIC_IF(type::is_MatrixND<T>()) {
      for (int i = 0; i < T::dim; i++) {
        for (int j = 0; j < T::dim; j++) {
          ret *= id(t)(i, j);
        }
      }
    } TC_STATIC_ELSE {
      ret = id(t);
    } TC_STATIC_END_IF
  } TC_STATIC_END_IF
  return ret;
}
// clang-format on

#define TC_MAKE_VECTORIZED_FROM_STD(op)                  \
  template <typename T>                                  \
  inline T op(const T &t) {                              \
    using Elem = typename type::element<decltype(t)>;    \
    return map(t, static_cast<Elem (*)(Elem)>(std::op)); \
  }

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
TC_FORCE_INLINE
    typename std::enable_if_t<!std::is_floating_point<T>::value, bool>
    equal(const T &A, const T &B, float64 tolerance) {
  return maximum(abs(A - B)) < tolerance;
}

template <typename T>
TC_FORCE_INLINE
    typename std::enable_if_t<std::is_floating_point<T>::value, bool>
    equal(const T &A, const T &B, float64 tolerance) {
  return std::abs(A - B) < tolerance;
}

}  // namespace math

template <int dim, typename T, InstSetExt ISE>
template <typename T_,
          typename std::enable_if_t<std::is_same<T_, int>::value, int>>
VectorND<dim, T, ISE>::VectorND(const TIndex<dim> &ind) {
  TC_STATIC_ASSERT(2 <= dim && dim <= 3);
  d[0] = ind.i;
  d[1] = ind.j;
  TC_STATIC_IF(dim == 3) {
    this->d[2] = ind.k;
  }
  TC_STATIC_END_IF
};

TC_NAMESPACE_END
