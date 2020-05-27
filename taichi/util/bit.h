/*******************************************************************************
    Copyright (c) The Taichi Authors (2016- ). All Rights Reserved.
    The use of this software is governed by the LICENSE file.
*******************************************************************************/
#pragma once

#include "taichi/common/core.h"

TI_NAMESPACE_BEGIN

namespace bit {

TI_FORCE_INLINE constexpr bool is_power_of_two(int32 x) {
  return x != 0 && (x & (x - 1)) == 0;
}

TI_FORCE_INLINE constexpr bool is_power_of_two(uint32 x) {
  return x != 0 && (x & (x - 1)) == 0;
}

TI_FORCE_INLINE constexpr bool is_power_of_two(int64 x) {
  return x != 0 && (x & (x - 1)) == 0;
}

TI_FORCE_INLINE constexpr bool is_power_of_two(uint64 x) {
  return x != 0 && (x & (x - 1)) == 0;
}

template <int length>
struct Bits {
  static_assert(is_power_of_two(length), "length must be a power of two");
  static_assert(length == 32 || length == 64, "length must be 32/64");

  using T = std::conditional_t<length == 32, uint32, uint64>;

  T data;

  Bits() : data(0) {
  }

  // Uninitialized
  Bits(void *) {
  }

  template <int start, int bits = 1>
  static constexpr T mask() {
    return (((T)1 << bits) - 1) << start;
  }

  template <int start, int bits = 1>
  TI_FORCE_INLINE T get() const {
    return (data >> start) & (((T)1 << bits) - 1);
  }

  template <int start, int bits = 1>
  TI_FORCE_INLINE void set(T val) {
    data =
        (data & ~mask<start, bits>()) | ((val << start) & mask<start, bits>());
  }

  TI_FORCE_INLINE T operator()(T) const {
    return data;
  }

  TI_FORCE_INLINE T get() const {
    return data;
  }

  TI_FORCE_INLINE void set(const T &data) {
    this->data = data;
  }
};

template <int length>
using BitFlags = Bits<length>;

template <typename T>
constexpr int bit_length() {
  return std::is_same<T, bool>() ? 1 : sizeof(T) * 8;
}

#define TI_BIT_FIELD(T, name, start)                    \
  T get_##name() const {                                \
    return (T)Base::get<start, bit::bit_length<T>()>(); \
  }                                                     \
  void set_##name(const T &val) {                       \
    Base::set<start, bit::bit_length<T>()>(val);        \
  }

template <typename T, int N>
TI_FORCE_INLINE constexpr T product(const std::array<T, N> arr) {
  T ret(1);
  for (int i = 0; i < N; i++) {
    ret *= arr[i];
  }
  return ret;
}

constexpr std::size_t least_pot_bound(std::size_t v) {
  if (v > std::numeric_limits<std::size_t>::max() / 2 + 1) {
    TI_ERROR("v({}) too large", v)
  }
  std::size_t ret = 1;
  while (ret < v) {
    ret *= 2;
  }
  return ret;
}

TI_FORCE_INLINE constexpr uint32 pot_mask(int x) {
  return (1u << x) - 1;
}

TI_FORCE_INLINE constexpr uint32 log2int(uint64 value) {
  int ret = 0;
  value >>= 1;
  while (value) {
    value >>= 1;
    ret += 1;
  }
  return ret;
}

template <typename G, typename T>
constexpr TI_FORCE_INLINE copy_refcv_t<T, G> &&reinterpret_bits(T &&t) {
  TI_STATIC_ASSERT(sizeof(G) == sizeof(T));
  return std::forward<copy_refcv_t<T, G>>(*reinterpret_cast<G *>(&t));
};

TI_FORCE_INLINE constexpr float64 compress(float32 h, float32 l) {
  uint64 data =
      ((uint64)reinterpret_bits<uint32>(h) << 32) + reinterpret_bits<uint32>(l);
  return reinterpret_bits<float64>(data);
}

TI_FORCE_INLINE constexpr std::tuple<float32, float32> extract(float64 x) {
  auto data = reinterpret_bits<uint64>(x);
  return std::make_tuple(reinterpret_bits<float32>((uint32)(data >> 32)),
                         reinterpret_bits<float32>((uint32)(data & (-1))));
}

}  // namespace bit

TI_NAMESPACE_END
