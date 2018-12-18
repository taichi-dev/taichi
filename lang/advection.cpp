#include "tlang.h"
#include <taichi/util.h>

TC_NAMESPACE_BEGIN

using namespace Tlang;

/*
auto advection = []() {
  int n = 128, nattr = 4;

  Float attr[4], v[2];  // ; vx, vy

  Program prog;

  for (int i = 0; i < 4; i++) {
    prog.buffer(0).stream(0).group(0).place(attr[i]);
  }
  prog.buffer(0).stream(1).group(0).place(v[0], v[1]);

  // TODO: inv_dx, t

  auto offset_x = floor(v[0]);
  auto offset_y = floor(v[1]);
  auto offset = offset_x * n + offset_y;
  auto wx = v[0] - offset_x;
  auto wy = v[1] - offset_y;

  // weights
  auto w00 = (1 - wx) * (1 - wy);
  auto w01 = (1 - wx) * wy;
  auto w10 = wx * (1 - wy);
  auto w11 = wx * wy;

  auto c = prog.cache(0, 1);  // group_size = 1;
  c.store(offset);
  c.store(w00);
  c.store(w01);
  c.store(w10);
  c.store(w11);

  // change of SIMD width

  auto new_offset = c.load();
  auto address = index(0) + new_offset;

  auto load = [&](int i, int offset_x, int offset_y) {
    return prog.load(attr[i], address + offset_x * n + offset_y);
  };

  for (int i = 0; i < nattr; i++) {
    auto new_attr = w00 * load(i, 0, 0) + w01 * load(i, 0, 1) + w10 * load(i, 1,
0) +
               w11 * load(i, 1, 1);
    prog.store(new_attr, new_addr[i]);
  }

  prog();
};
TC_REGISTER_TASK(advection);
*/

auto test_cache = []() {
  Float a, b;
  Vector v(4);

  int n = 16;

  Program prog(Arch::x86_64, 2048);

  prog.buffer(0).stream(0).group(0).place(a, b);
  for (int i = 0; i < 4; i++)
    prog.buffer(1).stream(0).group(0).place(v(i));

  // cache
  auto &c = prog.cache(1);  // group_size = 1
  c.store(a * b, 0);

  auto ab = c.load(0);  // 0th element

  for (int i = 0; i < 4; i++) {
    prog.store(v(i), ab * v(i));
  }

  prog.materialize_layout();

  for (int i = 0; i < n; i++) {
    prog.data(a, i) = i;
    prog.data(b, i) = i * 2;
    for (int j = 0; j < 4; j++) {
      prog.data(v(j), j) = 1.0_f * j / i;
    }
  }

  prog();

  for (int i = 0; i < n; i++) {
    for (int j = 0; j < 4; j++) {
      auto val = prog.data(v(j), i);
      auto gt = i * j * 2;
      if (abs(gt - val) > 1e-5_f) {
        TC_P(i);
        TC_P(j);
        TC_P(val);
        TC_P(gt);
      }
    }
  }
};

// TODO: random access

TC_REGISTER_TASK(test_cache);

TC_NAMESPACE_END