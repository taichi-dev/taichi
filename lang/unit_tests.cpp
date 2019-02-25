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
    Matrix A(2, 2), B(2, 2);
    A(0, 0) = 1;
    A(0, 1) = 1;
    A(1, 0) = 1;
    A(1, 1) = 1;

    B(0, 0) = 1;
    B(0, 1) = 2;
    B(1, 0) = 3;
    B(1, 1) = 4;
    auto C = A * B + A;
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
    local(sum) = 0;
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

TC_TEST("rand") {
  CoreState::set_trigger_gdb_when_crash(true);
  int n = 4096;
  Program prog(Arch::x86_64);

  declare(a_global);
  auto a = global_new(a_global, DataType::i32);

  layout([&]() { root.fixed(0, n).place(a); });

  auto func = kernel([&]() {
    declare(i);

    For(i, 0, n, [&] { Print(Rand<float>()); });
  });

  func();
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
      local(j) = 0;
      local(sum) = 0;
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
  int n = 1024;
  Program prog(Arch::x86_64);

  declare(a_global);
  auto a = global_new(a_global, DataType::i32);

  layout([&]() { root.fixed(0, n * n).place(a); });

  auto func = kernel([&]() {
    declare(i);

    Vectorize(8);
    For(i, 0, n * n, [&] {
      local(j) = 0;
      int limit = 20;
      if (false) {
        local(c_re) = cast<float>(i / n) / float(n / 2) - 1.5f;
        local(c_im) = cast<float>(i % n) / float(n / 2) - 1.0f;
        local(z_re) = c_re;
        local(z_im) = c_im;

        While(j < limit && (z_re * z_re + z_im * z_im) < 4.0f, [&] {
          local(new_re) = z_re * z_re - z_im * z_im;
          local(new_im) = 2.0f * z_re * z_im;
          z_re = c_re + new_re;
          z_im = c_im + new_im;
          j = j + 1;
        });
      } else {
        Vector c(2);

        c(0) = cast<float>(i / n) / float(n / 2) - 1.5f;
        c(1) = cast<float>(i % n) / float(n / 2) - 1.0f;

        Vector z = c;

        While(j < limit && z.norm2() < 4.0f, [&] {
          z = complex_mul(z, z) + c;
          j = j + 1;
        });
      }
      a[i] = j;
    });
  });

  TC_P(measure_cpe(func, 1));

  GUI gui("Mandelbrot Set", Vector2i(n));
  for (int i = 0; i < n * n; i++) {
    gui.buffer[i / n][i % n] = Vector4(a.val<int>(i) % 11 * 0.1f);
  }
  while (1)
    gui.update();
};
TC_REGISTER_TASK(mset);

auto ray_march = [&] {
  CoreState::set_trigger_gdb_when_crash(true);
  int n = 512;
  Program prog(Arch::x86_64);

  declare(color_r_global);
  declare(color_g_global);
  declare(color_b_global);
  auto color_r = global_new(color_r_global, DataType::f32);
  auto color_g = global_new(color_g_global, DataType::f32);
  auto color_b = global_new(color_b_global, DataType::f32);

  layout([&]() {
    root.fixed(0, n * n).place(color_r).place(color_g).place(color_b);
  });

  auto sdf = [&](Vector p) {
    return min(p(2) + 4.0f, min(p.norm() - 0.9_f, p(1) + 1.0f));
  };

  float32 eps = 1e-5f;
  float32 dist_limit = 1e4;
  int limit = 500;

  auto ray_march = [&](Vector p, Vector dir) {
    local(j) = 0;
    local(dist) = 0.0f;

    While(j < limit && sdf(p + dist * dir) > eps && dist < dist_limit, [&] {
      dist = dist + sdf(p + dist * dir);
      j = j + 1;
    });
    return dist;
  };

  auto normal = [&](Vector p) {
    float d = 1e-3f;
    Vector n(3);
    for (int i = 0; i < 3; i++) {
      Vector inc = p, dec = p;
      inc(i) = inc(i) + d;
      dec(i) = dec(i) - d;
      n(i) = (0.5f / d) * (sdf(inc) - sdf(dec));
    }
    return normalized(n);
  };

  auto out_dir = [&](Vector n) {
    Vector u({1.0f, 0.0f, 0.0f}), v(3);
    If(abs(n(1)) < 1 - 1e-3f, [&] {
      u = normalized(cross(n, Vector({0.0f, 1.0f, 0.0f})));
    });
    v = cross(n, u);
    local(phi) = 2 * pi * Rand<float>();
    local(alpha) = 0.5_f * pi * Rand<float>();
    return sin(alpha) * (cos(phi) * u + sin(phi) * v) + cos(alpha) * n;
  };

  auto background = [](Vector dir) { return max(dir(1) + dir(0), 0.0f); };

  float fov = 0.3;

  auto func = kernel([&]() {
    declare(i);
    Vectorize(8);
    For(i, 0, n * n, [&] {
      Vector orig({0.0f, 0.0f, 7.0f}), c(3);

      c(0) = fov * (cast<float>(i / n) / float(n / 2) - 1.0f);
      c(1) = fov * (cast<float>(i % n) / float(n / 2) - 1.0f);
      c(2) = -1.0f;
      c = normalized(c);

      Vector color(3);
      color = Vector({1.0f, 1.0f, 1.0f});
      int depth_limit = 4;
      local(depth) = 0;

      While(depth < depth_limit, [&] {
        depth = depth + 1;
        local(_dist) = ray_march(orig, c);
        If(_dist < dist_limit,
           [&] {
             orig = orig + _dist * c;
             Vector nor;
             nor = normal(orig);
             c = normalized(out_dir(nor));
             orig = orig + 0.01f * c;
             color = 0.5_f * color;
           })
            .Else([&] {
              color = color * background(c);
              depth = depth_limit;
            });
      });

      color_r[i] = load(color_r[i]) + color(0);
      color_g[i] = load(color_g[i]) + color(1);
      color_b[i] = load(color_b[i]) + color(2);
    });
  });

  /// TC_P(measure_cpe(func, 1));

  GUI gui("ray march", Vector2i(n));
  constexpr int N = 1;
  for (int frame = 0;; frame++) {
    for (int i = 0; i < N; i++)
      func();
    for (int i = 0; i < n * n; i++) {
      gui.buffer[i / n][i % n] =
          1.0_f / ((frame + 1) * N) *
          Vector4(color_r.val<float>(i), color_g.val<float>(i),
                  color_b.val<float>(i), 1);
    }
    gui.update();
  }
};
TC_REGISTER_TASK(ray_march);

TLANG_NAMESPACE_END
