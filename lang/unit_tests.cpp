#include "ir.h"
#include <numeric>
#include "tlang.h"
#include <taichi/visual/gui.h>

TLANG_NAMESPACE_BEGIN

TC_TEST("compiler_linalg") {
  CoreState::set_trigger_gdb_when_crash(true);
  Program prog(Arch::x86_64);

  declare(a_global);
  auto a = global_new(a_global, DataType::i32);
  auto i = Expr(0);

  layout([&]() { root.fixed(i, 128).place(a); });

  auto func = kernel([&]() {
    declare(i);
    declare(sum);
    var(int32, sum);

    Matrix A(2, 2), B(2, 2);
    A(0, 0) = 1;
    A(0, 1) = 1;
    A(1, 0) = 1;
    A(1, 1) = 1;

    B(0, 0) = 1;
    B(0, 1) = 2;
    B(1, 0) = 3;
    B(1, 1) = 4;
    auto C = A + B;
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

  declare(a_global);
  auto a = global_new(a_global, DataType::i32);
  auto i = Expr(0);
  layout([&]() { root.fixed(i, n).place(a); });

  auto dou = [](ExprH a) { return a * 2; };

  auto func = kernel([&]() {
    declare(i);
    declare(sum);
    var(int32, sum);

    For(i, 0, n, [&] {
      sum = sum + i;
      If(i % 2 == 0).Then([&] { a[i] = dou(i); }).Else([&] { a[i] = i; });
    });
  });

  func();

  for (int i = 0; i < n; i++) {
    TC_CHECK(a.val<int32>(i) == (2 - i % 2) * i);
  }
};

auto test_circle = [] {
  CoreState::set_trigger_gdb_when_crash(true);
  int n = 128;
  Program prog(Arch::x86_64);

  declare(a_global);
  auto a = global_new(a_global, DataType::i32);
  auto i = Expr(0);

  layout([&]() { root.fixed(i, n * n).place(a); });

  auto func = kernel([&]() {
    declare(i);

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
  declare(a);
  declare(x);
  declare(b);
  declare(p);
  declare(q);
  declare(i);
  declare(j);

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
        b = b * 3;
      })
      .Else([&] {
        b = b + 2;
        b = b - 4;
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

  declare(a_global);
  auto a = global_new(a_global, DataType::i32);

  layout([&]() { root.fixed(0, n).place(a); });

  auto func = kernel([&]() {
    declare(i);

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

TC_TEST("while") {
  CoreState::set_trigger_gdb_when_crash(true);
  int n = 4096;
  Program prog(Arch::x86_64);

  declare(a_global);
  auto a = global_new(a_global, DataType::i32);

  layout([&]() { root.fixed(0, n).place(a); });

  auto func = kernel([&]() {
    declare(i);

    Vectorize(8);
    For(i, 0, n, [&] {
      declare_as(j, int);
      declare_as(sum, int);
      j = 0;
      sum = 0;
      While(j < i, [&] {
        sum = sum + j;
        j = j + 1;
      });
      a[i] = sum;
    });
  });

  TC_P(measure_cpe(func, 1));

  for (int i = 0; i < n; i++) {
    TC_CHECK(a.val<int>(i) == (i - 1) * i / 2);
  }
};

Vector complex_mul(const Vector &a, const Vector &b) {
  Vector ret(2);
  ret(0) = a(0) * b(0) - a(1) * b(1);
  ret(1) = a(0) * b(1) + a(1) * b(0);
  return ret;
}

auto mset = [&] {
  CoreState::set_trigger_gdb_when_crash(true);
  int n = 512;
  Program prog(Arch::x86_64);

  declare(a_global);
  auto a = global_new(a_global, DataType::i32);

  layout([&]() { root.fixed(0, n * n).place(a); });

  auto func = kernel([&]() {
    declare(i);
    declare_as(z_re, float);
    declare_as(z_im, float);

    // Vectorize(8);
    For(i, 0, n * n, [&] {
      declare_as(j, int);
      declare_as(sum, int);
      declare_as(c_re, float);
      declare_as(c_im, float);
      c_re = cast<float>(i / n) / float(n / 2) - 1.5f;
      c_im = cast<float>(i % n) / float(n / 2) - 1.0f;
      z_re = c_re;
      z_im = c_im;

      int limit = 50;
      While(j < limit && (z_re * z_re + z_im * z_im) < 4.0f, [&] {
        declare_as(new_re, float);
        declare_as(new_im, float);
        new_re = z_re * z_re - z_im * z_im;
        new_im = 2.0f * z_re * z_im;
        z_re = c_re + new_re;
        z_im = c_im + new_im;
        j = j + 1;
      });
      a[i] = j;
    });
  });

  measure_cpe(func, 1);

  GUI gui("Mandelbrot Set", Vector2i(n));
  for (int i = 0; i < n * n; i++) {
    gui.buffer[i / n][i % n] = Vector4(a.val<int>(i) % 11 * 0.1f);
  }
  while (1)
    gui.update();
};
TC_REGISTER_TASK(mset);

TLANG_NAMESPACE_END
