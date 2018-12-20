#include "tlang.h"
#include <taichi/util.h>
#include <taichi/visual/gui.h>

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
  const int n = 512, nattr = 1;

  Float attr[2][nattr], v[2];  // ; vx, vy

  Program prog(Arch::x86_64, n * n);

  for (int k = 0; k < 2; k++) {
    for (int i = 0; i < nattr; i++) {
      prog.buffer(k).stream(0).group(0).place(attr[k][i]);
    }
    prog.buffer(2).stream(0).group(0).place(v[k]);
  }
  prog.config.group_size = 1;

  auto index = Expr::index(0);
  Int32 xi = index % imm(n);
  Int32 yi = index / imm(n);

  auto offset_x = floor(v[0][index]);
  auto offset_y = floor(v[1][index]);
  auto offset = cast<int32>(offset_x) * imm(n) + cast<int32>(offset_y) * imm(1);
  auto wx = v[0][index] - offset_x;
  auto wy = v[1][index] - offset_y;

  // weights
  auto w00 = (imm(1.0f) - wx) * (imm(1.0f) - wy);
  auto w01 = (imm(1.0f) - wx) * wy;
  auto w10 = wx * (imm(1.0f) - wy);
  auto w11 = wx * wy;

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

  for (int k = 0; k < nattr; k++) {
    Expr node = index + offset;

    auto v00 = attr[0][k][node];
    auto v01 = attr[0][k][node + imm(1)];
    auto v10 = attr[0][k][node + imm(n)];
    auto v11 = attr[0][k][node + imm(n + 1)];

    attr[1][k][index] =
        w00 * v00 + w01 * v01 + w10 * v10 + w11 * v11;
    // cast<float32>(xi + yi * imm(2)) / imm(3.0f * n);  // imm(0.001_f) +
    // attr[0][k][index];
  }

  prog.compile();

  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      for (int k = 0; k < nattr; k++) {
        prog.data(attr[0][k], i * n + j) = (i) % 128 / 128.0_f;
      }
      real s = 1.0_f / n;
      prog.data(v[0], i * n + j) = s * (j - n / 2);
      prog.data(v[1], i * n + j) = -s * (i - n / 2);
    }
  }

  GUI gui("Advection", n, n);

  while (1) {
    prog();

    for (int i = 0; i < n; i++) {
      for (int j = 0; j < n; j++) {
        for (int k = 0; k < nattr; k++) {
          gui.buffer[i][j] = Vector4(prog.data(attr[1][k], i * n + j));
        }
      }
    }

    gui.update();
    std::swap(prog.buffers[0], prog.buffers[1]);
  }
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