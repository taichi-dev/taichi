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

  Float attr[2][nattr], v[2];

  Program prog(Arch::x86_64, n * n);

  for (int k = 0; k < 2; k++) {
    for (int i = 0; i < nattr; i++) {
      prog.buffer(k).stream(0).group(0).place(attr[k][i]);
    }
    prog.buffer(2).stream(0).group(0).place(v[k]);
  }
  prog.config.group_size = 1;

  auto index = Expr::index(0);
  Int32 xi = index % imm(n) / imm(1);
  Int32 yi = index % imm(n * n) / imm(n);

  auto offset_x = floor(v[0][index]);
  auto offset_y = floor(v[1][index]);
  auto wx = v[0][index] - offset_x;
  auto wy = v[1][index] - offset_y;

  // ** gs=1
  // prog.adapt(offset_x, offset_y, 1); // convert to group_size = 1
  auto offset = cast<int32>(offset_x) * imm(n) + cast<int32>(offset_y) * imm(1);

  // SlowAdapter
  // SlowAdapter adapter;

  // weights
  auto w00 = (imm(1.0f) - wx) * (imm(1.0f) - wy);
  auto w01 = (imm(1.0f) - wx) * wy;
  auto w10 = wx * (imm(1.0f) - wy);
  auto w11 = wx * wy;

  // adapter(w00);
  // adapter(w01);
  // adapter(w10);
  // adapter(w11);

  auto clamp = [](const Expr &e) { return min(max(imm(2), e), imm(n - 2)); };

  for (int k = 0; k < nattr; k++) {
    Expr node = index + offset;
    Int32 i = clamp(node / imm(n));
    Int32 j = clamp(node % imm(n));
    node = i * imm(n) + j;

    auto v00 = attr[0][k][node + imm(0)];
    auto v01 = attr[0][k][node + imm(1)];
    auto v10 = attr[0][k][node + imm(n)];
    auto v11 = attr[0][k][node + imm(n + 1)];

    attr[1][k][index] = w00 * v00 + w01 * v01 + w10 * v10 + w11 * v11;
  }

  prog.compile();

  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      for (int k = 0; k < nattr; k++) {
        prog.data(attr[0][k], i * n + j) = i % 128 / 128.0_f;
      }
      real s = 3.0_f / n;
      prog.data(v[0], i * n + j) = s * (j - n / 2);
      prog.data(v[1], i * n + j) = -s * (i - n / 2);
    }
  }

  GUI gui("Advection", n, n);

  for (int f = 0; f < 1000; f++) {
    for (int t = 0; t < 3; t++) {
      prog();

      for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
          for (int k = 0; k < nattr; k++) {
            gui.buffer[i][j] = Vector4(prog.data(attr[1][k], i * n + j));
          }
        }
      }
    }

    gui.update();
    gui.screenshot(fmt::format("images/{:04d}.png", f));
    prog.swap_buffers(0, 1);
  }
};
TC_REGISTER_TASK(advection);

auto test_adapter = []() {
  Float a, b;
  int vec_size = 1;
  Vector v(vec_size), u(vec_size);

  int n = 16;

  Program prog(Arch::x86_64, 2048);
  prog.config.group_size = 1;

  prog.buffer(0).stream(0).group(0).place(a, b);
  for (int i = 0; i < vec_size; i++) {
    prog.buffer(1).stream(0).group(0).place(v(i));
  }

  auto ind = Expr::index(0);

  // auto &adapter = prog.adapter(0);
  auto ab = a[ind] * b[ind];

  // adapter.convert(ab);
  // adapter.set(1, 8);

  for (int d = 0; d < vec_size; d++) {
    v(d)[ind] = ab * v(d)[ind];
  }

  prog.materialize_layout();

  for (int i = 0; i < n; i++) {
    prog.data(a, i) = i;
    prog.data(b, i) = (i + 1) * 2;
    for (int j = 0; j < vec_size; j++) {
      prog.data(v(j), i) = 1.0_f * j / (i + 1);
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

/*
#define For(i, range) \
  {                   \
    Index i; for_loop(i, range, [&](Index i)

#define End )
*/

TC_NAMESPACE_END