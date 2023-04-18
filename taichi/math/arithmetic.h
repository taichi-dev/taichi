#pragma once

#include <cstdint>

#include "taichi/common/trait.h"

namespace taichi {

// Round up |a| to the closest multiple of |b|, works only for integers.
template <typename T,
          typename U,
          typename = std::enable_if_t<std::is_convertible_v<U, T>>>
T iroundup(T a, U b) {
  static_assert(std::is_integral_v<T>, "LHS must be integral type");
  static_assert(std::is_integral_v<U>, "RHS must be integral type");
  return ((a + b - 1) / b) * b;
}

template <typename T>
uint32_t log2int(T value) {
  static_assert(std::is_integral_v<T>, "Must be integral type");

  uint32_t ret = 0;
  value >>= 1;
  while (value) {
    value >>= 1;
    ret += 1;
  }
  return ret;
}

}  // namespace taichi
