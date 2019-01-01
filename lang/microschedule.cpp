#include "tlang.h"
#include <taichi/util.h>
#include <taichi/visual/gui.h>
#include <taichi/common/bit.h>

TC_NAMESPACE_BEGIN

using namespace Tlang;

#if (0)
auto test_loop = []() {
  TC_NOT_IMPLEMENTED
  // CoreState::set_trigger_gdb_when_crash(true);
  Float a, b;

  int n = 256;

  Program prog(Arch::x86_64, n);
  prog.config.group_size = 1;

  prog.buffer(0).range(n).stream(0).group(0).place(a).place(b);
  prog.materialize_layout();

  Expr i = Expr::index(0);
  for_loop(i, range(0, n), [&]() {
    // ***
    a[i] = a[i] * b[i];
  });

  for (int i = 0; i < n; i++) {
    prog.data(a, i) = i;
    prog.data(b, i) = i * 2;
  }

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
  bool use_adapter = true;

  const int n = 1024, nattr = 4;

  Float attr[2][nattr], v[2];

  Program prog(Arch::x86_64, n * n);

  prog.config.group_size = use_adapter ? nattr : 1;
  prog.config.num_groups = use_adapter ? 8 : 8;

  prog.layout([&]() {
    for (int k = 0; k < 2; k++) {
      if (use_adapter) {
        for (int i = 0; i < nattr; i++) {
          prog.buffer(k).range(n * n).stream(0).group(0).place(attr[k][i]);
        }
        prog.buffer(2).range(n * n).stream(0).group(0).place(v[k]);
      } else {
        for (int i = 0; i < nattr; i++) {
          prog.buffer(k).range(n * n).stream(i).group(0).place(attr[k][i]);
        }
        prog.buffer(2).range(n * n).stream(k).group(0).place(v[k]);
      }
    }
  });

  TC_ASSERT(bit::is_power_of_two(n));

  auto func = prog.def([&]() {
    auto index = Expr::index(0);
    for_loop(index, {0, n * n}, [&] {
      // ** gs = 2

      auto offset_x = floor(v[0][index]).name("offset_x");
      auto offset_y = floor(v[1][index]).name("offset_y");
      auto wx = v[0][index] - offset_x;
      auto wy = v[1][index] - offset_y;
      wx.name("wx");
      wy.name("wy");

      if (use_adapter) {
        prog.adapter(0).set(2, 1).convert(offset_x, offset_y);
        prog.adapter(1).set(2, 1).convert(wx, wy);
      }

      // ** gs = 1
      auto offset =
          cast<int32>(offset_x) * imm(n) + cast<int32>(offset_y) * imm(1);

      auto clamp = [](const Expr &e) {
        return min(max(imm(2), e), imm(n - 2));
      };

      // weights
      auto w00 = (imm(1.0f) - wx) * (imm(1.0f) - wy);
      auto w01 = (imm(1.0f) - wx) * wy;
      auto w10 = wx * (imm(1.0f) - wy);
      auto w11 = wx * wy;

      w00.name("w00");
      w01.name("w01");
      w10.name("w10");
      w11.name("w11");

      Expr node = max(Expr::index(0) + offset, imm(0));
      Int32 i = clamp(node >> imm((int)bit::log2int(n))).name("i");  // node / n
      // Int32 i = clamp(node / imm(n)).name("i"); // node / n
      Int32 j = clamp(node & imm(n - 1)).name("j");  // node % n
      // Int32 j = clamp(node % imm(n)).name("j"); // node % n
      node = (i << imm((int)bit::log2int(n))) + j;
      // node = i * imm(n) + j;
      node.name("node");

      if (use_adapter) {
        prog.adapter(2).set(1, 4).convert(w00, w01, w10, w11);
        prog.adapter(3).set(1, 4).convert(node);
      }

      // ** gs = 4
      for (int k = 0; k < nattr; k++) {
        auto v00 = attr[0][k][node + imm(0)].name("v00");
        auto v01 = attr[0][k][node + imm(1)].name("v01");
        auto v10 = attr[0][k][node + imm(n)].name("v10");
        auto v11 = attr[0][k][node + imm(n + 1)].name("v11");

        attr[1][k][index] = w00 * v00 + w01 * v01 + w10 * v10 + w11 * v11;
        // attr[1][k][index] = w00 * v00;
        attr[1][k][index].name(fmt::format("output{}", k));
      }
    });
  });

  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      for (int k = 0; k < nattr; k++) {
        prog.data(attr[0][k], i * n + j) = i % 128 / 128.0_f;
      }
      real s = 20.0_f / n;
      prog.data(v[0], i * n + j) = s * (j - n / 2);
      prog.data(v[1], i * n + j) = -s * (i - n / 2);
      // prog.data(v[0], i * n + j) = 0;
      // prog.data(v[1], i * n + j) = 0;
    }
  }

  GUI gui("Advection", n, n);

  for (int f = 0; f < 1000; f++) {
    for (int t = 0; t < 100; t++) {
      TC_TIME(func());
      prog.swap_buffers(0, 1);
    }

    /*
    for (int i = 0; i < n; i++) {
      for (int j = 0; j < n; j++) {
        for (int k = 0; k < nattr; k++) {
          gui.buffer[i][j] = Vector4(prog.data(attr[1][k], i * n + j));
        }
      }
    }
    gui.update();
    */
    // gui.screenshot(fmt::format("images/{:04d}.png", f));
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

  prog.materialize_layout();

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

TC_REGISTER_TASK(test_adapter);

auto test_select = []() {
  int n = 128;
  Program prog(Arch::x86_64);
  prog.config.group_size = 1;

  Real a;

  prog.layout([&]() { prog.buffer(0).range(n).stream(0).group().place(a); });

  auto i = Expr::index(0);

  auto func1 = prog.def([&]() {
    auto i = Expr::index(0);
    for_loop(i, {0, n},
             [&]() {  //
               a[i] = select(cmp_ne(imm(0), i % imm(2)), cast<float32>(i),
                             imm(0.0_f));
               //
             });
  });

  func1();

  for (int i = 0; i < n; i++) {
    TC_P(i);
    TC_P(prog.data(a, i));
    TC_ASSERT(prog.data(a, i) == (i % 2) * i);
  }

};
TC_REGISTER_TASK(test_select);
#endif

// TODO: random access

/*
#define For(i, range) \
  {                   \
    Index i; for_loop(i, range, [&](Index i)

#define End )
*/

/*
TODO: arbitrary for loop (bounds using arbitrary constants)
 */
auto test_multiple_programs = []() {
  int n = 128;
  Program prog(Arch::x86_64);
  prog.config.group_size = 1;

  Real a, b, c, d;
  a = placeholder(DataType::f32);
  b = placeholder(DataType::f32);
  c = placeholder(DataType::f32);
  d = placeholder(DataType::f32);

  auto i = Expr::index(0);

  prog.layout([&]() {
    root.fixed(i, n).place(a);
    root.fixed(i, n).place(b);
    root.fixed(i, n).place(c);
    root.fixed(i, n).place(d);
  });

  auto func1 = prog.def([&]() {
    for_loop(i, {0, n}, [&] { b[i] = a[i] + imm(1.0_f); });
  });
  auto func2 = prog.def([&]() {
    for_loop(i, {0, n}, [&] { c[i] = b[i] + imm(1.0_f); });
  });
  auto func3 = prog.def([&]() {
    for_loop(i, {0, n}, [&] { d[i] = c[i] + imm(1.0_f); });
  });

  for (int i = 0; i < n; i++) {
    a.set<float32>(i, i);
  }

  func1();
  func2();
  func3();

  for (int i = 0; i < n; i++) {
    TC_ASSERT(d.get<float32>(i) == i + 3);
  }
};

TC_REGISTER_TASK(test_multiple_programs);

TC_NAMESPACE_END
