/*******************************************************************************
    Copyright (c) The Taichi Authors (2016- ). All Rights Reserved.
    The use of this software is governed by the LICENSE file.
*******************************************************************************/

#include "taichi/util/testing.h"
#include "taichi/util/bit.h"

TI_NAMESPACE_BEGIN

namespace bit {

Bitset::Bitset() {
}

Bitset::Bitset(int n) {
  if (n % kBits != 0) {
    n += kBits - n % kBits;
  }
  vec_ = std::vector<value_t>(n / kBits, 0);
}

std::size_t Bitset::size() const {
  return vec_.size() * kBits;
}

void Bitset::reset() {
  for (auto &value : vec_) {
    value = 0;
  }
}

void Bitset::flip(int x) {
  vec_[x / kBits] ^= ((value_t)1) << (x % kBits);
}

bool Bitset::any() const {
  for (auto &val : vec_) {
    if (val)
      return true;
  }
  return false;
}

bool Bitset::none() const {
  return !any();
}

Bitset::reference Bitset::operator[](int x) {
  return reference(vec_, x);
}

Bitset &Bitset::operator&=(const Bitset &other) {
  const int len = vec_.size();
  TI_ASSERT(len == other.vec_.size());
  for (int i = 0; i < len; i++) {
    vec_[i] &= other.vec_[i];
  }
  return *this;
}

Bitset Bitset::operator&(const Bitset &other) const {
  Bitset result = *this;
  result &= other;
  return result;
}

Bitset &Bitset::operator|=(const Bitset &other) {
  const int len = vec_.size();
  TI_ASSERT(len == other.vec_.size());
  for (int i = 0; i < len; i++) {
    vec_[i] |= other.vec_[i];
  }
  return *this;
}

Bitset Bitset::operator|(const Bitset &other) const {
  Bitset result = *this;
  result |= other;
  return result;
}

Bitset &Bitset::operator^=(const Bitset &other) {
  const int len = vec_.size();
  TI_ASSERT(len == other.vec_.size());
  for (int i = 0; i < len; i++) {
    vec_[i] ^= other.vec_[i];
  }
  return *this;
}

Bitset Bitset::operator~() const {
  Bitset result(size());
  const int len = vec_.size();
  for (int i = 0; i < len; i++) {
    result.vec_[i] = ~vec_[i];
  }
  return result;
}

int Bitset::find_first_one() const {
  return lower_bound(0);
}

int Bitset::lower_bound(int x) const {
  const int len = vec_.size();
  if (x >= len * kBits) {
    return -1;
  }
  if (x < 0) {
    x = 0;
  }
  int i = x / kBits;
  if (x % kBits != 0) {
    if (auto test = vec_[i] & (kMask ^ ((1ULL << (x % kBits)) - 1))) {
      return i * kBits + log2int(lowbit(test));
    }
    i++;
  }
  for (; i < len; i++) {
    if (vec_[i]) {
      return i * kBits + log2int(lowbit(vec_[i]));
    }
  }
  return -1;
}

std::vector<int> Bitset::or_eq_get_update_list(const Bitset &other) {
  const int len = vec_.size();
  TI_ASSERT(len == other.vec_.size());
  std::vector<int> result;
  for (int i = 0; i < len; i++) {
    auto update = other.vec_[i] & ~vec_[i];
    if (update) {
      vec_[i] |= update;
      for (int j = 0; j < kBits; j++) {
        if ((update >> j) & 1) {
          result.push_back((i * kBits) | j);
        }
      }
    }
  }
  return result;
}

Bitset::reference::reference(std::vector<value_t> &vec, int x)
    : pos_(&vec[x / kBits]), digit_(((value_t)1) << (x % kBits)) {
}

Bitset::reference::operator bool() const {
  return *pos_ & digit_;
}

bool Bitset::reference::operator~() const {
  return ~*pos_ & digit_;
}

Bitset::reference &Bitset::reference::operator=(bool x) {
  if (x)
    *pos_ |= digit_;
  else
    *pos_ &= kMask ^ digit_;
  return *this;
}

Bitset::reference &Bitset::reference::operator=(
    const Bitset::reference &other) {
  *this = bool(other);
  return *this;
}

Bitset::reference &Bitset::reference::flip() {
  *pos_ ^= digit_;
  return *this;
}

std::ostream &operator<<(std::ostream &os, const Bitset &b) {
  for (auto &val : b.vec_)
    for (int j = 0; j < Bitset::kBits; j++)
      os << ((val >> j) & 1 ? '1' : '0');
  return os;
}

}  // namespace bit

using namespace bit;

struct Flags : public Bits<32> {
  using Base = Bits<32>;
  TI_BIT_FIELD(bool, apple, 0);
  TI_BIT_FIELD(bool, banana, 1);
  TI_BIT_FIELD(uint8, cherry, 2);
};

TI_TEST("bit") {
  Bits<32> b;
  b.set<5>(1);
  CHECK(b.get() == 32);
  b.set<10, 8>(255);
  CHECK(b.get() == 255 * 1024 + 32);
  b.set<11, 1>(0);
  CHECK(b.get() == 255 * 1024 + 32 - 2048);
  b.set<11, 2>(3);
  CHECK(b.get() == 255 * 1024 + 32);
  b.set<11, 2>(0);
  CHECK(b.get() == 255 * 1024 + 32 - 2 * 3072);
  b.set<11, 2>(1);
  CHECK(b.get() == 255 * 1024 + 32 - 4096);

  Flags f;
  f.set_apple(true);
  CHECK(f.get_apple() == true);
  f.set_apple(false);
  CHECK(f.get_apple() == false);
  f.set_banana(true);
  CHECK(f.get_banana() == true);
  CHECK(f.get_apple() == false);
  f.set_apple(false);
  CHECK(f.get_apple() == false);
  f.set_apple(true);
  f.set_cherry(63);
  CHECK(f.get_cherry() == 63);
  f.set_banana(false);
  CHECK(f.get_cherry() == 63);

  struct Decomp {
    uint8 a, b, c, d;
  };

  uint32 v = 0xabcd1234;
  auto &dec = reinterpret_bits<Decomp>(v);
  CHECK(dec.a == 0x34);
  CHECK(dec.b == 0x12);
  CHECK(dec.c == 0xcd);
  CHECK(dec.d == 0xab);
  dec.d = 0xef;
  CHECK(v == 0xefcd1234);

  CHECK(reinterpret_bits<float32>(reinterpret_bits<uint32>(1.32_f32)) ==
        1.32_f32);

  // float64 t = 123.456789;
  // auto e = extract(t);
  // TI_P(std::get<0>(e));
  // TI_P(std::get<1>(e));
  // CHECK(t == compress(std::get<0>(e), std::get<1>(e)));
}
TI_NAMESPACE_END
