#pragma once

#include <functional>
#include <stddef.h>

namespace taichi::hashing {

namespace {
inline void hash_combine(size_t &seed, size_t value) {
  seed ^= value + 0x9e3779b9 + (seed << 6) + (seed >> 2);
}
}  // namespace

template <typename T>
struct Hasher {
 public:
  size_t operator()(T const &val) const {
    return std::hash<T>{}(val);
  }
};

template <typename T>
struct Hasher<std::vector<T>> {
 public:
  size_t operator()(std::vector<T> const &vec) const {
    size_t ret = 0;
    for (const auto &i : vec) {
      hash_combine(ret, Hasher<T>()(i));
    }
    return ret;
  }
};

template <typename T1, typename T2>
struct Hasher<std::pair<T1, T2>> {
 public:
  size_t operator()(std::pair<T1, T2> const &val) const {
    size_t ret = Hasher<T1>{}(val.first);
    hash_combine(ret, Hasher<T2>{}(val.second));
    return ret;
  }
};

template <typename... Ts>
struct Hasher<std::tuple<Ts...>> {
 public:
  size_t operator()(std::tuple<Ts...> const &val) const {
    return hash<std::tuple_size_v<std::tuple<Ts...>> - 1>(val);
  };

 private:
  template <int N>
  size_t hash(std::tuple<Ts...> const &val) const {
    size_t ret = hash<N - 1>(val);
    hash_combine(ret, Hasher<std::tuple_element_t<N, std::tuple<Ts...>>>{}(
                          std::get<N>(val)));
    return ret;
  }
  template <>
  size_t hash<0>(std::tuple<Ts...> const &val) const {
    return Hasher<std::tuple_element_t<0, std::tuple<Ts...>>>{}(
        std::get<0>(val));
  }
};
}  // namespace taichi::hashing
