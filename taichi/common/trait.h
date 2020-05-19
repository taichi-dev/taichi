#pragma once

#include <type_traits>

namespace taichi {

// This allows us to static_assert(always_false_v<T>) when using if constexpr.
// See https://stackoverflow.com/a/53945549/12003165
template <typename T>
struct always_false : std::false_type {};

template <typename T>
inline constexpr bool always_false_v = always_false<T>::value;

}  // namespace taichi
