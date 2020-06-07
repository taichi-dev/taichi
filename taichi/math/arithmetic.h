#pragma once

#include "taichi/common/trait.h"

namespace taichi {

// Round up |a| to the closest multiple of |b|, works only for integers.
template <typename T>
T iroundup(T a, T b) {
  if constexpr (std::is_integral_v<T>) {
    return ((a + b - 1) / b) * b;
  } else {
    static_assert(always_false_v<T>, "Must be integral type");
  }
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
