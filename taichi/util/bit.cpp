/*******************************************************************************
    Copyright (c) The Taichi Authors (2016- ). All Rights Reserved.
    The use of this software is governed by the LICENSE file.
*******************************************************************************/

#include "taichi/util/testing.h"
#include "taichi/util/bit.h"

TI_NAMESPACE_BEGIN

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
