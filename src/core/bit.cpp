/*******************************************************************************
    Copyright (c) The Taichi Authors (2016- ). All Rights Reserved.
    The use of this software is governed by the LICENSE file.
*******************************************************************************/

#include <taichi/testing.h>
#include <taichi/common/bit.h>

TC_NAMESPACE_BEGIN

using namespace bit;

struct Flags : public Bits<32> {
  using Base = Bits<32>;
  TC_BIT_FIELD(bool, apple, 0);
  TC_BIT_FIELD(bool, banana, 1);
  TC_BIT_FIELD(uint8, cherry, 2);
};

TC_TEST("bit") {
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
}
TC_NAMESPACE_END
