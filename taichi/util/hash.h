#pragma once

#include <functional>
#include <stddef.h>

namespace std {

namespace {
inline void hash_combine(size_t &seed, size_t value) {
  seed ^= value + 0x9e3779b9 + (seed << 6) + (seed >> 2);
}
}  // namespace

template <typename T>
struct hash<std::vector<T>> {
 public:
  size_t operator()(std::vector<T> const &vec) const {
    size_t ret = 0;
    for (const auto &i : vec) {
      hash_combine(ret, hash<T>()(i));
    }
    return ret;
  }
};
}  // namespace std
