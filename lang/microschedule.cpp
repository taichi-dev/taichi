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
