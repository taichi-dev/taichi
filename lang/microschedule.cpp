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
  const int n = 512, nattr = 4;

  Float attr[2][nattr], v[2];

  Program prog(Arch::x86_64, n * n);

  for (int k = 0; k < 2; k++) {
    for (int i = 0; i < nattr; i++) {
      prog.buffer(k).stream(0).group(0).place(attr[k][i]);
    }
    prog.buffer(2).stream(0).group(0).place(v[k]);
  }
  prog.config.group_size = nattr;

  // ** gs = 2
  auto index = Expr::index(0);
  Int32 xi = index % imm(n) / imm(1);
  Int32 yi = index % imm(n * n) / imm(n);

  auto offset_x = floor(v[0][index]);
  auto offset_y = floor(v[1][index]);
  auto wx = v[0][index] - offset_x;
  auto wy = v[1][index] - offset_y;

  prog.adapter(0).set(2, 1);
  prog.adapter(0).convert(offset_x);
  prog.adapter(0).convert(offset_y);

  prog.adapter(1).set(2, 1);
  prog.adapter(1).convert(wx);
  prog.adapter(1).convert(wy);

  // ** gs = 1
  auto offset = cast<int32>(offset_x) * imm(n) + cast<int32>(offset_y) * imm(1);

  auto clamp = [](const Expr &e) { return min(max(imm(2), e), imm(n - 2)); };

  // weights
  auto w00 = (imm(1.0f) - wx) * (imm(1.0f) - wy);
  auto w01 = (imm(1.0f) - wx) * wy;
  auto w10 = wx * (imm(1.0f) - wy);
  auto w11 = wx * wy;

  w00.name("w00");
  w01.name("w01");
  w10.name("w10");
  w11.name("w11");

  Expr node = Expr::index(0) + offset;
  Int32 i = clamp(node / imm(n)).name("i");
  Int32 j = clamp(node % imm(n)).name("j");
  node = i * imm(n) + j;
  node.name("node");

  prog.adapter(2).set(1, 4);
  prog.adapter(2).convert(w00);
  prog.adapter(2).convert(w01);
  prog.adapter(2).convert(w10);
  prog.adapter(2).convert(w11);

  prog.adapter(3).set(1, 4);
  prog.adapter(3).convert(node);

  // ** gs = 4
  for (int k = 0; k < nattr; k++) {
    auto v00 = attr[0][k][node + imm(0)];
    auto v01 = attr[0][k][node + imm(1)];
    auto v10 = attr[0][k][node + imm(n)];
    auto v11 = attr[0][k][node + imm(n + 1)];

    attr[1][k][index] = w00 * v00 + w01 * v01 + w10 * v10 + w11 * v11;
    attr[1][k][index].name(fmt::format("output{}", k));
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

// a * b * vec
void test_adapter1(int vec_size) {
  Float a, b;
  Vector v(vec_size), u(vec_size);

  int n = 128;

  Program prog(Arch::x86_64, n);
  prog.config.group_size = vec_size;
  prog.config.num_groups = 8;

  prog.buffer(0).stream(0).group(0).place(a, b);
  for (int i = 0; i < vec_size; i++) {
    prog.buffer(1).stream(0).group(0).place(v(i));
  }

  auto ind = Expr::index(0);

  auto &adapter = prog.adapter(0);
  auto ab = a[ind] * b[ind];

  adapter.set(1, vec_size);
  adapter.convert(ab);

  for (int d = 0; d < vec_size; d++) {
    v(d)[ind] = ab * v(d)[ind];
  }

  prog.materialize_layout();

  for (int i = 0; i < n; i++) {
    prog.data(a, i) = i;
    prog.data(b, i) = 2.0_f * (i + 1);
    for (int j = 0; j < vec_size; j++) {
      prog.data(v(j), i) = 1.0_f * j / (i + 1);
    }
  }

  prog();

  for (int i = 0; i < n; i++) {
    for (int j = 0; j < vec_size; j++) {
      auto val = prog.data(v(j), i);
      auto gt = i * j * 2;
      if (abs(gt - val) > 1e-3_f) {
        TC_P(i);
        TC_P(j);
        TC_P(val);
        TC_P(gt);
        TC_ERROR("");
      }
    }
  }
}

// Vec<vec_size> reduction
void test_adapter2(int vec_size) {
  Vector v(vec_size);
  Float sum;

  int n = 64;

  Program prog(Arch::x86_64, n);
  prog.config.group_size = 1;
  prog.config.num_groups = 8;

  for (int i = 0; i < vec_size; i++) {
    prog.buffer(1).stream(0).group(0).place(v(i));
  }
  prog.buffer(0).stream(0).group(0).place(sum);

  auto ind = Expr::index(0);

  auto v_ind = v[ind];

  for (int i = 0; i < vec_size; i++) {
    v_ind(i).set(Expr::load_if_pointer(v_ind(i)));
    TC_P(v_ind(i)->node_type_name());
  }

  auto &adapter = prog.adapter(0);
  adapter.set(vec_size, 1);
  for (int i = 0; i < vec_size; i++) {
    adapter.convert(v_ind(i));
  }

  Expr acc = Expr::create_imm(0.0_f);
  for (int d = 0; d < vec_size; d++) {
    acc = acc + v_ind(d);
  }

  sum[ind] = acc;

  prog.materialize_layout();

  for (int i = 0; i < n; i++) {
    for (int j = 0; j < vec_size; j++) {
      prog.data(v(j), i) = j + i;
    }
  }

  prog();

  for (int i = 0; i < n; i++) {
    auto val = prog.data(sum, i);
    auto gt = vec_size * (vec_size - 1) / 2 + i * vec_size;
    if (abs(gt - val) > 1e-5_f) {
      TC_P(i);
      TC_P(val);
      TC_P(gt);
      TC_ERROR("");
    }
  }
}

// reduce(vec_a<n> - vec_b<n>) * vec_c<2n>
void test_adapter3(int vec_size) {
  Vector a(vec_size), b(vec_size), c(vec_size * 2);
  Float sum;

  int n = 64;

  Program prog(Arch::x86_64, n);
  prog.config.group_size = vec_size * 2;
  prog.config.num_groups = 8;

  for (int i = 0; i < vec_size; i++) {
    prog.buffer(0).stream(0).group(0).place(a(i));
    prog.buffer(1).stream(0).group(0).place(b(i));
  }

  for (int i = 0; i < vec_size * 2; i++) {
    prog.buffer(2).stream(0).group(0).place(c(i));
  }

  auto ind = Expr::index(0);

  auto aind = a[ind];
  auto bind = b[ind];
  auto cind = c[ind];

  auto diff = aind.element_wise_prod(aind) - bind.element_wise_prod(bind);

  {
    auto &adapter = prog.adapter(0);
    adapter.set(vec_size, 1);
    for (int i = 0; i < vec_size; i++)
      adapter.convert(diff(i));
  }

  Expr acc = Expr::create_imm(0.0_f);
  for (int d = 0; d < vec_size; d++) {
    acc = acc + diff(d);
  }

  {
    auto &adapter = prog.adapter(1);
    adapter.set(1, vec_size * 2);
    adapter.convert(acc);
    for (int i = 0; i < vec_size * 2; i++)
      c(i)[ind] = c(i)[ind] * acc;
  }

  prog.materialize_layout();

  for (int i = 0; i < n; i++) {
    for (int j = 0; j < vec_size; j++) {
      prog.data(a(j), i) = i + j + 1;
      prog.data(b(j), i) = i + j;
    }
    for (int j = 0; j < vec_size * 2; j++) {
      prog.data(c(j), i) = i - 2 + j;
    }
  }

  prog();

  for (int i = 0; i < n; i++) {
    real s = 0;
    for (int j = 0; j < vec_size; j++) {
      s += sqr(i + j + 1) - sqr(i + j);
    }
    for (int j = 0; j < vec_size * 2; j++) {
      auto val = prog.data(c(j), i);
      auto gt = s * (i - 2 + j);
      if (abs(gt - val) > 1e-3_f) {
        TC_P(i);
        TC_P(j);
        TC_P(val);
        TC_P(gt);
        TC_ERROR("");
      }
    }
  }
}

auto test_adapter = []() {
  test_adapter3(1);
  test_adapter3(2);
  test_adapter3(4);
  test_adapter3(8);

  test_adapter1(1);
  test_adapter1(2);
  test_adapter1(4);
  test_adapter1(8);

  test_adapter2(1);
  test_adapter2(2);
  test_adapter2(4);
  test_adapter2(8);

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