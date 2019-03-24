#include "../tlang.h"
#include <taichi/testing.h>
#include <numeric>
#include <taichi/visual/gui.h>

TLANG_NAMESPACE_BEGIN

TC_TEST("compiler_linalg") {
  CoreState::set_trigger_gdb_when_crash(true);
  Program prog(Arch::x86_64);
  prog.config.print_ir = true;

  Global(a, i32);
  auto i = Index(0);

  layout([&]() { root.fixed(i, 128).place(a); });

  auto func = kernel([&]() {
    Matrix A(2, 2), B(2, 2);
    A(0, 0) = 1;
    A(0, 1) = 1;
    A(1, 0) = 1;
    A(1, 1) = 1;

    B(0, 0) = 1;
    B(0, 0) = Eval(B(0, 0));
    B(0, 0) = Eval(B(0, 0));
    B(0, 0) = Eval(B(0, 0));
    B(0, 0) = Eval(B(0, 0));
    B(0, 1) = 2;
    B(1, 0) = 3;
    B(1, 1) = 4;
    auto C = Eval(A * B + A);
    for (int p = 0; p < 2; p++) {
      for (int q = 0; q < 2; q++) {
        Print(C(p, q));
      }
    }
  });

  func();
};

TC_TEST("compiler_basics") {
  CoreState::set_trigger_gdb_when_crash(true);
  int n = 128;
  Program prog(Arch::x86_64);
  prog.config.print_ir = true;

  Global(a, i32);
  auto i = Index(0);
  layout([&]() { root.fixed(i, n).place(a); });

  auto dou = [](Expr a) { return a * 2; };

  auto func = kernel([&]() {
    Declare(i);
    Local(sum) = 0;
    For(i, 0, n, [&] {
      sum = sum + i;
      Local(ret) = 0;
      If(i % 2 == 0).Then([&] { ret = dou(i); }).Else([&] { ret = i; });
      a[i] = ret;
    });
  });

  func();

  for (int i = 0; i < n; i++) {
    TC_CHECK(a.val<int32>(i) == (2 - i % 2) * i);
  }
};

TC_TEST("simd_if") {
  CoreState::set_trigger_gdb_when_crash(true);
  int n = 128;
  Program prog(Arch::x86_64);

  Global(a, i32);
  auto i = Index(0);
  layout([&]() { root.fixed(i, n).place(a); });

  auto dou = [](Expr a) { return a * 2; };

  auto func = kernel([&]() {
    Declare(i);
    Local(sum) = 0;
    Vectorize(8);
    For(i, 0, n, [&] {
      Local(ret) = 0;
      If(i % 2 == 0).Then([&] { ret = dou(i); }).Else([&] { ret = i; });
      a[i] = ret;
    });
  });

  func();

  for (int i = 0; i < n; i++) {
    TC_CHECK(a.val<int32>(i) == (2 - i % 2) * i);
  }
};

TC_TEST("simd_if2") {
  CoreState::set_trigger_gdb_when_crash(true);
  int n = 32;
  Program prog(Arch::x86_64);

  Global(a, i32);
  auto i = Index(0);
  layout([&]() { root.fixed(i, n).place(a); });

  auto func = kernel([&]() {
    Declare(i);
    Local(sum) = 0;
    Vectorize(8);
    For(i, 0, n, [&] {
      Local(ret) = 0;
      If(i % 3 == 0).Then([&] { ret = i; }).Else([&] {
        If(i % 3 == 1).Then([&] { ret = i * 2; }).Else([&] { ret = i * 3; });
      });
      a[i] = ret;
    });
  });

  func();

  for (int i = 0; i < n; i++) {
    TC_CHECK(a.val<int32>(i) == (1 + i % 3) * i);
  }
};

auto test_circle = [] {
  CoreState::set_trigger_gdb_when_crash(true);
  int n = 128;
  Program prog(Arch::x86_64);

  Global(a, i32);
  auto i = Index(0);

  layout([&]() { root.fixed(i, n * n).place(a); });

  auto func = kernel([&]() {
    Declare(i);

    Vectorize(8);
    For(i, 0, n * n, [&] {
      auto x = i / n - n / 2;
      auto y = i % n - n / 2;
      If(x * x + y * y < n * n / 4).Then([&] { a[i] = 1; }).Else([&] {
        a[i] = 0;
      });
    });
  });

  func();

  GUI gui("circle", Vector2i(n, n));
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      gui.buffer[i][j] = Vector4(a.val<int32>(i * n + j));
    }
  }

  while (1) {
    gui.update();
  }
};
TC_REGISTER_TASK(test_circle);

auto test_ast = []() {
  CoreState::set_trigger_gdb_when_crash(true);

  Program prog(Arch::x86_64);
  int n = 128;

  context = std::make_unique<FrontendContext>();
  Declare(a);
  Declare(x);
  Declare(b);
  Declare(p);
  Declare(q);
  Declare(i);
  Declare(j);

  // var(float32, a);
  x.set(global_new(x, DataType::f32));
  TC_ASSERT(x.is<GlobalVariableExpression>());

  var(float32, a);
  var(float32, b);
  var(int32, p);
  var(int32, q);

  p = p + q;

  Print(a);
  If(a < 500).Then([&] { Print(b); }).Else([&] { Print(a); });

  If(a > 5)
      .Then([&] {
        b = (b + 1) / 3;
        b *= 3;
      })
      .Else([&] {
        b = b + 2;
        b -= 4;
      });

  For(i, 0, 8, [&] {
    x[i] = i;
    For(j, 0, 8, [&] {
      auto k = i + j;
      Print(k);
      // While(k < 500, [&] { Print(k); });
    });
  });
  Print(b);

  auto root = context->root();

  TC_INFO("AST");
  irpass::print(root);

  irpass::lower(root);
  TC_INFO("Lowered");
  irpass::print(root);

  irpass::typecheck(root);
  TC_INFO("TypeChecked");
  irpass::print(root);
};
TC_REGISTER_TASK(test_ast);

TC_TEST("vectorize") {
  CoreState::set_trigger_gdb_when_crash(true);
  int n = 128;
  Program prog(Arch::x86_64);

  Global(a, i32);

  layout([&]() { root.fixed(0, n).place(a); });

  auto func = kernel([&]() {
    Declare(i);

    Vectorize(8);
    For(i, 0, n, [&] { a[i] = i; });
  });

  func();

  for (int i = 0; i < n; i++) {
    TC_CHECK(a.val<int>(i) == i);
  }
};

TC_TEST("simd_fpe") {
  __m128 a = _mm_set_ps(1, 0, -1, -2);
  __m128 b = _mm_set_ps(0, 0, 0, 0);
  a = _mm_sqrt_ps(a);
  for (int i = 0; i < 4; i++) {
    std::cout << a[i] << std::endl;
  }
  a = a / b;
  for (int i = 0; i < 4; i++) {
    std::cout << a[i] << std::endl;
  }
};

TC_TEST("rand") {
  CoreState::set_trigger_gdb_when_crash(true);
  int n = 8;
  Program prog(Arch::x86_64);

  Global(a, i32);

  layout([&]() { root.fixed(0, n).place(a); });

  auto func = kernel([&]() {
    Declare(i);

    For(i, 0, n, [&] { Print(Rand<float>()); });
  });

  func();
};

TC_TEST("while") {
  CoreState::set_trigger_gdb_when_crash(true);
  int n = 4096;
  Program prog(Arch::x86_64);

  Global(a, i32);

  layout([&]() { root.fixed(0, n).place(a); });

  auto func = kernel([&]() {
    Declare(i);

    Vectorize(8);
    For(i, 0, n, [&] {
      Local(j) = 0;
      Local(sum) = 0;
      While(j < i, [&] {
        sum += j;
        j += 1;
      });
      a[i] = sum;
    });
  });

  TC_P(measure_cpe(func, 1));

  for (int i = 0; i < n; i++) {
    TC_CHECK(a.val<int>(i) == (i - 1) * i / 2);
  }
};

TC_TEST("slp") {
  CoreState::set_trigger_gdb_when_crash(true);
  int n = 16;
  Program prog(Arch::x86_64);
  prog.config.print_ir = true;

  Global(a, i32);
  Global(b, i32);
  Global(c, i32);
  Global(d, i32);

  layout([&]() { root.fixed(0, n).place(a, b, c, d); });

  auto func = kernel([&]() {
    Declare(i);

    Vectorize(1);
    For(i, 0, n, [&] {
      SLP(4);
      a[i] = 1;
      b[i] = 2;
      c[i] = 3;
      d[i] = 4;
    });
  });

  func();

  for (int i = 0; i < n; i++) {
    TC_CHECK(a.val<int>(i) == 1);
    TC_CHECK(b.val<int>(i) == 2);
    TC_CHECK(c.val<int>(i) == 3);
    TC_CHECK(d.val<int>(i) == 4);
  }
};

TC_TEST("slp1") {
  CoreState::set_trigger_gdb_when_crash(true);
  int n = 16;
  for (auto slp1 : {true, false}) {
    Program prog(Arch::x86_64);
    prog.config.print_ir = true;
    Vector grid(DataType::f32, 4);
    layout(
        [&]() { root.fixed(0, n).place(grid(0), grid(1), grid(2), grid(3)); });
    auto func = kernel([&]() {
      Declare(i);
      Vectorize(1);
      For(i, 0, n, [&] {
        if (slp1)
          SLP(1);
        Vector v(4);
        for (int i = 0; i < 4; i++) {
          v(i) = real(i);
        }

        SLP(4);
        grid[i] = v;
      });
    });

    func();

    for (int i = 0; i < n; i++) {
      for (int d = 0; d < 4; d++) {
        TC_CHECK(grid(d).val<float32>(i) == d);
      }
    }
  }
};

TC_TEST("slp2") {
  CoreState::set_trigger_gdb_when_crash(true);
  int n = 16;
  Program prog(Arch::x86_64);
  prog.config.print_ir = true;

  Global(a, i32);
  Global(b, i32);

  layout([&]() { root.fixed(0, n).place(a, b); });

  auto func = kernel([&]() {
    Declare(i);

    Vectorize(4);
    For(i, 0, n, [&] {
      SLP(2);
      a[i] = 1 + i * 7;
      b[i] = 2 + i * 9;
    });
  });

  func();

  for (int i = 0; i < n; i++) {
    TC_CHECK(a.val<int>(i) == 1 + i * 7);
    TC_CHECK(b.val<int>(i) == 2 + i * 9);
  }
};

TC_TEST("slp3") {
  CoreState::set_trigger_gdb_when_crash(true);
  int n = 16;
  Program prog(Arch::x86_64);
  prog.config.print_ir = true;

  Global(a, i32);
  Global(b, i32);

  layout([&]() { root.fixed(0, n).place(a, b); });

  auto func = kernel([&]() {
    Declare(i);

    Vectorize(4);
    For(i, 0, n, [&] {
      SLP(2);
      Local(x) = i * 7;
      Local(y) = i * 9;
      a[i] = 1 + x;
      b[i] = 2 + y;
    });
  });

  func();

  for (int i = 0; i < n; i++) {
    TC_CHECK(a.val<int>(i) == 1 + i * 7);
    TC_CHECK(b.val<int>(i) == 2 + i * 9);
  }
};

TC_TEST("slpmatvecmul") {
  CoreState::set_trigger_gdb_when_crash(true);
  int n = 16;
  Program prog(Arch::x86_64);
  prog.config.print_ir = true;

  int dim = 4;

  Matrix A(dim, dim), x(dim), y(dim);
  A.fill_global(DataType::f32);
  x.fill_global(DataType::f32);
  y.fill_global(DataType::f32);

  layout([&]() {
    auto &s = root.fixed(0, n);
    for (int i = 0; i < dim; i++) {
      for (int j = 0; j < dim; j++) {
        s.place(A(i, j));
      }
      s.place(x(i), y(i));
    }
  });

  auto func = kernel([&]() {
    Declare(i);

    Vectorize(8 / dim);
    For(i, 0, n, [&] {
      SLP(dim);
      y[i] = A[i] * x[i];
      SLP(dim);
      y[i] = A[i] * x[i];
    });
  });

  func();

  /*
  for (int i = 0; i < n; i++) {
    TC_CHECK(a.val<int>(i) == 1 + i * 7);
    TC_CHECK(b.val<int>(i) == 2 + i * 9);
  }
  */
};

// scalar a * scalar b * vec c
TC_TEST("mixed_simd1") {
  for (auto vec_size : {4, 8, 16}) {
    Program prog;
    prog.config.print_ir = true;

    Global(a, f32);
    Global(b, f32);
    Vector v(DataType::f32, vec_size);

    int n = 128;
    auto ind = Index(0);

    layout([&] {
      root.fixed(ind, n).place(a, b);
      for (int i = 0; i < vec_size; i++) {
        root.fixed(ind, n).place(v(i));
      }
    });

    auto func = kernel([&]() {
      Declare(i);
      For(i, 0, n, [&]() {
        SLP(1);
        Local(ab) = a[i] * b[i];

        SLP(vec_size);
        v[i] = ab * v[i];
      });
    });

    for (int i = 0; i < n; i++) {
      a.val<float32>(i) = i;
      b.val<float32>(i) = 2.0_f * (i + 1);
      for (int j = 0; j < vec_size; j++) {
        v(j).val<float32>(i) = 1.0_f * j / (i + 1);
      }
    }

    func();

    for (int i = 0; i < n; i++) {
      for (int j = 0; j < vec_size; j++) {
        auto val = v(j).val<float32>(i);
        float32 gt = i * j * 2;
        TC_CHECK_EQUAL(gt, val, 1e-3_f);
      }
    }
  }
}

// Vec<vec_size> reduction
TC_TEST("mixed_simd2") {
  int n = 64;

  for (auto vec_size : {4, 8, 16}) {
    TC_WARN("{}", vec_size);
    Program prog;
    prog.config.max_vector_width = 4;
    prog.config.print_ir = true;

    Vector v(vec_size);
    v.fill_global(DataType::f32);

    Global(sum, f32);

    auto ind = Index(0);

    layout([&] {
      for (int i = 0; i < vec_size; i++) {
        root.fixed(ind, n).place(v(i));
      }
      root.fixed(ind, n).place(sum);
    });

    auto func = kernel([&] {
      Declare(i);
      For(i, 0, n, [&]() {
        SLP(vec_size);
        auto vi = v[i];
        auto v_ind = vi;

        SLP(1);
        Local(acc) = 0.0_f;
        for (int d = 0; d < vec_size; d++) {
          acc = acc + v_ind(d);
        }

        sum[i] = acc;
      });
    });

    for (int i = 0; i < n; i++) {
      for (int j = 0; j < vec_size; j++) {
        v(j).val<float32>(i) = j + i;
      }
    }

    func();

    for (int i = 0; i < n; i++) {
      auto val = sum.val<float32>(i);
      float32 gt = vec_size * (vec_size - 1) / 2 + i * vec_size;
      TC_CHECK_EQUAL(gt, val, 1e-5_f);
    }
  }
}

// reduce(vec_a<n> ** 2 - vec_b<n> ** 2) * vec_c<2n>
TC_TEST("mixed_simd3_slp") {
  for (auto vec_size : {16}) {
    // why vec_size = 16 fails??
    Program prog;
    prog.config.max_vector_width = 8;
    prog.config.print_ir = true;

    Vector a(DataType::f32, vec_size), b(DataType::f32, vec_size),
        c(DataType::f32, vec_size * 2);
    Global(sum, f32);

    int n = 64;

    auto ind = Index(0);

    layout([&] {
      for (int i = 0; i < vec_size; i++) {
        root.fixed(ind, n).place(a(i));
        root.fixed(ind, n).place(b(i));
      }

      for (int i = 0; i < vec_size * 2; i++) {
        root.fixed(ind, n).place(c(i));
      }
    });

    auto func = kernel([&]() {
      Declare(i);
      For(i, 0, n, [&]() {
        SLP(vec_size);
        auto diff_ =
            a[i].element_wise_prod(a[i]) - b[i].element_wise_prod(b[i]);
        auto diff = diff_;

        SLP(1);
        Local(acc) = 0.0_f;
        for (int d = 0; d < vec_size; d++) {
          acc = acc + diff(d);
        }

        SLP(vec_size * 2);
        c[i] *= acc;
      });
    });

    for (int i = 0; i < n; i++) {
      for (int j = 0; j < vec_size; j++) {
        a(j).val<float32>(i) = i + j + 1;
        b(j).val<float32>(i) = i + j;
      }
      for (int j = 0; j < vec_size * 2; j++) {
        c(j).val<float32>(i) = i - 2 + j;
      }
    }

    func();

    for (int i = 0; i < n; i++) {
      real s = 0;
      for (int j = 0; j < vec_size; j++) {
        s += taichi::sqr(i + j + 1) - taichi::sqr(i + j);
      }
      for (int j = 0; j < vec_size * 2; j++) {
        auto val = c(j).val<float32>(i);
        auto gt = s * (i - 2 + j);
        TC_CHECK_EQUAL(gt, val, 1e-3_f);
      }
    }
  }
}

TC_TEST("vector_split1") {
  CoreState::set_trigger_gdb_when_crash(true);
  int n = 32;
  Program prog(Arch::x86_64);
  prog.config.print_ir = true;
  prog.config.max_vector_width = 8;

  Global(a, i32);

  layout([&]() { root.fixed(0, n).place(a); });

  auto func = kernel([&]() {
    Declare(i);
    Vectorize(16);
    For(i, 0, n, [&] { a[i] = i; });
  });

  func();

  for (int i = 0; i < n; i++) {
    TC_CHECK(a.val<int>(i) == i);
  }
};

TC_TEST("vector_split_slp") {
  CoreState::set_trigger_gdb_when_crash(true);
  int n = 256;
  Program prog(Arch::x86_64);
  prog.config.max_vector_width = 8;
  prog.config.print_ir = true;

  Global(a, i32);
  Global(b, i32);
  Global(c, i32);
  Global(d, i32);

  layout([&]() { root.fixed(0, n).place(a, b, c, d); });

  auto func = kernel([&]() {
    Declare(i);

    Vectorize(32);
    For(i, 0, n, [&] {
      SLP(4);
      a[i] = 1 + i;
      b[i] = 2 + i;
      c[i] = 3 + i;
      d[i] = 4 + i;
    });
  });

  func();

  for (int i = 0; i < n; i++) {
    TC_CHECK(a.val<int>(i) == 1 + i);
    TC_CHECK(b.val<int>(i) == 2 + i);
    TC_CHECK(c.val<int>(i) == 3 + i);
    TC_CHECK(d.val<int>(i) == 4 + i);
  }
};

TLANG_NAMESPACE_END
