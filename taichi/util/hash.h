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
inline size_t hash_value(T const &value) {
  return Hasher<T>{}(value);
}

template <typename T>
inline void hash_combine(size_t &seed, T const &value) {
  // Reference:
  // https://www.boost.org/doc/libs/1_55_0/doc/html/hash/reference.html#boost.hash_combine
  seed ^= hash_value(value) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
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

template <typename T1, typename T2>
struct Hasher<std::pair<T1, T2>> {
 public:
  size_t operator()(std::pair<T1, T2> const &val) const {
    size_t ret = hash_value(val.first);
    hash_combine(ret, val.second);
    return ret;
  }
};

namespace {
template <int N, typename... Ts>
struct TupleHasher {
  size_t operator()(std::tuple<Ts...> const &val) {
    size_t ret = TupleHasher<N - 1, Ts...>{}(val);
    hash_combine(ret, std::get<N>(val));
    return ret;
  }
};

template <typename... Ts>
struct TupleHasher<0, Ts...> {
  size_t operator()(std::tuple<Ts...> const &val) {
    return hash_value(std::get<0>(val));
  }
};
}  // namespace

template <typename... Ts>
struct Hasher<std::tuple<Ts...>> {
 public:
  size_t operator()(std::tuple<Ts...> const &val) const {
    return TupleHasher<std::tuple_size_v<std::tuple<Ts...>> - 1, Ts...>{}(val);
  };
};

}  // namespace taichi::hashing
