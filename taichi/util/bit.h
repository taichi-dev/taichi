/*******************************************************************************
    Copyright (c) The Taichi Authors (2016- ). All Rights Reserved.
    The use of this software is governed by the LICENSE file.
*******************************************************************************/
#pragma once

#include "taichi/common/core.h"

namespace taichi {
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

TI_FORCE_INLINE uint32 as_uint(const float32 x) {
  return *(uint32 *)&x;
}

TI_FORCE_INLINE float32 as_float(const uint32 x) {
  return *(float32 *)&x;
}

TI_FORCE_INLINE float32 half_to_float(const uint16 x) {
  // Reference: https://stackoverflow.com/a/60047308
  const uint32 e = (x & 0x7C00) >> 10;  // exponent
  const uint32 m = (x & 0x03FF) << 13;  // mantissa
  const uint32 v =
      as_uint((float32)m) >>
      23;  // evil log2 bit hack to count leading zeros in denormalized format
  return as_float(
      (x & 0x8000) << 16 | (e != 0) * ((e + 112) << 23 | m) |
      ((e == 0) & (m != 0)) *
          ((v - 37) << 23 | ((m << (150 - v)) &
                             0x007FE000)));  // sign : normalized : denormalized
}

TI_FORCE_INLINE uint16 float_to_half(const float32 x) {
  // Reference: https://stackoverflow.com/a/60047308
  const uint32 b = as_uint(x) + 0x00001000;  // round-to-nearest-even: add last
                                             // bit after truncated mantissa
  const uint32 e = (b & 0x7F800000) >> 23;   // exponent
  const uint32 m = b & 0x007FFFFF;  // mantissa; in line below: 0x007FF000 =
                                    // 0x00800000-0x00001000 = decimal indicator
                                    // flag - initial rounding
  return (b & 0x80000000) >> 16 |
         (e > 112) * ((((e - 112) << 10) & 0x7C00) | m >> 13) |
         ((e < 113) & (e > 101)) *
             ((((0x007FF000 + m) >> (125 - e)) + 1) >> 1) |
         (e > 143) * 0x7FFF;  // sign : normalized : denormalized : saturate
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
  explicit Bits(void *) {
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

#define TI_BIT_FIELD(T, name, start)                                           \
  T get_##name() const { return (T)Base::get<start, bit::bit_length<T>()>(); } \
  void set_##name(const T &val) { Base::set<start, bit::bit_length<T>()>(val); }

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

TI_FORCE_INLINE constexpr uint32 ceil_log2int(uint64 value) {
  // Returns ceil(log2(value)). When value == 0, it returns 0.
  return log2int(value) + ((value & (value - 1)) != 0);
}

TI_FORCE_INLINE constexpr uint64 lowbit(uint64 x) {
  return x & (-x);
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

class Bitset {
 public:
  using value_t = uint64;
  static constexpr std::size_t kBits = sizeof(value_t) * 8;
  // kBits should be a power of two. However, the function is_power_of_two is
  // ambiguous and can't be called here.
  static_assert((kBits & (kBits - 1)) == 0);
  static constexpr std::size_t kLogBits = log2int(kBits);
  static constexpr value_t kMask = ((value_t)-1);
  class reference {
   public:
    reference(std::vector<value_t> &vec, int x);
    explicit operator bool() const;
    bool operator~() const;
    reference &operator=(bool x);
    reference &operator=(const reference &other);
    reference &flip();

   private:
    value_t *pos_;
    value_t digit_;
  };

  Bitset();
  explicit Bitset(int n);
  std::size_t size() const;
  void reset();
  void flip(int x);
  bool any() const;
  bool none() const;
  reference operator[](int x);
  Bitset &operator&=(const Bitset &other);
  Bitset operator&(const Bitset &other) const;
  Bitset &operator|=(const Bitset &other);
  Bitset operator|(const Bitset &other) const;
  Bitset &operator^=(const Bitset &other);
  Bitset operator~() const;

  // Find the place of the first "1", or return -1 if it doesn't exist.
  int find_first_one() const;
  // Find the place of the first "1" which is not before x, or return -1 if
  // it doesn't exist.
  int lower_bound(int x) const;

  std::vector<int> or_eq_get_update_list(const Bitset &other);

  // output from the lowest bit to the highest bit
  friend std::ostream &operator<<(std::ostream &os, const Bitset &b);

 private:
  std::vector<value_t> vec_;
};

}  // namespace bit
}  // namespace taichi
