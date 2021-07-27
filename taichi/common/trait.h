#pragma once

#include <type_traits>

namespace taichi {

// This allows us to static_assert(always_false_v<T>) when using if constexpr.
// See https://stackoverflow.com/a/53945549/12003165
template <typename T>
struct always_false : std::false_type {};

template <typename T>
inline constexpr bool always_false_v = always_false<T>::value;

#define ENUM_FLAG_OPERATOR(T, X)                                   \
  inline T operator X(T lhs, T rhs) {                              \
    return (T)(static_cast<std::underlying_type_t<T>>(lhs)         \
                   X static_cast<std::underlying_type_t<T>>(rhs)); \
  }
#define ENUM_FLAGS(T)                                       \
  enum class T;                                             \
  inline T operator~(T t) {                                 \
    return (T)(~static_cast<std::underlying_type_t<T>>(t)); \
  }                                                         \
  ENUM_FLAG_OPERATOR(T, |)                                  \
  ENUM_FLAG_OPERATOR(T, ^)                                  \
  ENUM_FLAG_OPERATOR(T, &)                                  \
  enum class T

}  // namespace taichi
