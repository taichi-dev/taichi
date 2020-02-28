#pragma once

#include <type_traits>

namespace taichi {

namespace detail {
template <class T>
struct always_false : std::false_type {};

template <class T>
inline constexpr bool always_false_v = always_false<T>::value;
}  // namespace detail

// Round up |a| to the closest multiple of |b|, works only for integers.
template <typename T>
T iroundup(T a, T b) {
  if constexpr (std::is_integral_v<T>) {
    return ((a + b - 1) / b) * b;
  } else {
    // Have to be type dependent: https://stackoverflow.com/a/53945549/12003165
    static_assert(detail::always_false_v<T>, "Must be integral type");
  }
}

}  // namespace taichi