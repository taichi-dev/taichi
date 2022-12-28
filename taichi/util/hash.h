#pragma once

#include <functional>
#include <stddef.h>

namespace taichi::hashing {

template <typename T>
struct Hasher {
 public:
  size_t operator()(T const &val) const {
    return std::hash<T>{}(val);
  }
};

namespace {
template <typename T>
inline void hash_combine(size_t &seed, T const &value) {
  // Reference:
  // https://www.boost.org/doc/libs/1_55_0/doc/html/hash/reference.html#boost.hash_combine
  seed ^= Hasher<T>{}(value) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
}
}  // namespace

template <typename T>
struct Hasher<std::vector<T>> {
 public:
  size_t operator()(std::vector<T> const &vec) const {
    size_t ret = 0;
    for (const auto &i : vec) {
      hash_combine(ret, i);
    }
    return ret;
  }
};
}  // namespace taichi::hashing
