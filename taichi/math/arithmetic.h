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

}  // namespace taichi
