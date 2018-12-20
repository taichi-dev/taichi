#include "tlang.h"
#include <taichi/util.h>

TC_NAMESPACE_BEGIN

using namespace Tlang;

auto test_loop = []() {
  CoreState::set_trigger_gdb_when_crash(true);
  Float a, b;

  int n = 16;

  Program prog(Arch::x86_64, n);
  prog.config.group_size = 8;

  prog.buffer(0).stream(0).group(0).place(a, b);

  Expr i = Expr::index(0);
  for_loop(i, range(0, n), [&]() {
    // ***
    a[i] = a[i] * b[i];
  });

  prog.materialize_layout();

  for (int i = 0; i < n; i++) {
    prog.data(a, i) = i;
    prog.data(b, i) = i * 2;
  }

  prog();

  for (int i = 0; i < n; i++) {
    auto val = prog.data(a, i);
    auto gt = i * i * 2;
    if (abs(gt - val) > 1e-5_f) {
      TC_P(i);
      TC_P(val);
      TC_P(gt);
      TC_ERROR("");
    }
  }
};

TC_REGISTER_TASK(test_loop);

auto advection = []() {
  const int n = 128, nattr = 4;

  Float attr[nattr], v[2];  // ; vx, vy

  Program prog(Arch::x86_64, n);

  for (int i = 0; i < nattr; i++) {
    prog.buffer(0).stream(0).group(0).place(attr[i]);
  }

  prog.buffer(0).stream(1).group(0).place(v[0], v[1]);

  auto index = Expr::index(0);

  // TODO: inv_dx, t

  auto offset_x = floor(v[0]);
  auto offset_y = floor(v[1]);
  auto offset = offset_x * imm(n) + offset_y * imm(1);
  auto wx = v[0] - offset_x;
  auto wy = v[1] - offset_y;

  // weights
  auto w00 = (imm(1.0) - wx) * (imm(1.0) - wy);
  auto w01 = (imm(1.0) - wx) * wy;
  auto w10 = wx * (imm(1.0) - wy);
  auto w11 = wx * wy;

  // change of SIMD width

  auto address = index + offset;

  /*
  auto load = [&](int i, int offset_x, int offset_y) {
    return prog.load(attr[i], address + offset_x * n + offset_y);
  };


  for (int i = 0; i < nattr; i++) {
    auto new_attr = w00 * load(i, 0, 0) + w01 * load(i, 0, 1) +
                    w10 * load(i, 1, 0) + w11 * load(i, 1, 1);
    // prog.store(new_attr, new_addr[i]);
  }
   */

  prog();
};
TC_REGISTER_TASK(advection);

/*
auto test_adapter = []() {
  Float a, b;
  int vec_size = 8;
  Vector v(vec_size);

  int n = 16;

  Program prog(Arch::x86_64, 2048);
  prog.config.group_size = 8;

  prog.buffer(0).stream(0).group(0).place(a, b);
  for (int i = 0; i < vec_size; i++)
    prog.buffer(1).stream(0).group(0).place(v(i));

  // cache
  auto &c = prog.cache(0);  // TODO: specify group_size = 1
  c.store(a * b, 0);

  auto ab = c.load(0);  // 0th element

  for (int d = 0; d < vec_size; d++) {
    v(d) = ab * v(d);
  }

  prog.materialize_layout();

  for (int i = 0; i < n; i++) {
    prog.data(a, i) = i;
    prog.data(b, i) = i * 2;
    for (int j = 0; j < vec_size; j++) {
      prog.data(v(j), j) = 1.0_f * j / i;
    }
  }

  prog();

  for (int i = 0; i < n; i++) {
    for (int j = 0; j < vec_size; j++) {
      auto val = prog.data(v(j), i);
      auto gt = i * j * 2;
      if (abs(gt - val) > 1e-5_f) {
        TC_P(i);
        TC_P(j);
        TC_P(val);
        TC_P(gt);
        TC_ERROR("");
      }
    }
  }
};

// TODO: random access

TC_REGISTER_TASK(test_adapter);
*/

/*
#define For(i, range) \
  {                   \
    Index i; for_loop(i, range, [&](Index i)

#define End )
*/

TC_NAMESPACE_END