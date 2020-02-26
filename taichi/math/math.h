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
  if constexpr (dim == 3) {
    this->d[2] = ind.k;
  }
};

TI_NAMESPACE_END
