/*******************************************************************************
    Taichi - Physically based Computer Graphics Library

    Copyright (c) 2016 Yuanming Hu <yuanmhu@gmail.com>

    All rights reserved. Use of this source code is governed by
    the MIT license as written in the LICENSE file.
*******************************************************************************/
#include <taichi/util.h>

TC_NAMESPACE_BEGIN

namespace bit {

TC_FORCE_INLINE constexpr bool is_power_of_two(int x) {
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
  TC_FORCE_INLINE T get() const {
    return (data >> start) & (((T)1 << bits) - 1);
  }

  template <int start, int bits = 1>
  TC_FORCE_INLINE void set(T val) {
    data =
        (data & ~mask<start, bits>()) | ((val << start) & mask<start, bits>());
  }

  TC_FORCE_INLINE T operator()(T) const {
    return data;
  }

  TC_FORCE_INLINE T get() const {
    return data;
  }

  TC_FORCE_INLINE void set(const T &data) {
    this->data = data;
  }
};

template <typename T>
constexpr int bit_length() {
  return std::is_same<T, bool>() ? 1 : sizeof(T) * 8;
}

#define TC_BIT_FIELD(T, name, start)                    \
  T get_##name() const {                                \
    return (T)Base::get<start, bit::bit_length<T>()>(); \
  }                                                     \
  void set_##name(const T &val) {                       \
    Base::set<start, bit::bit_length<T>()>(val);        \
  }

}  // namespace bit

TC_NAMESPACE_END
