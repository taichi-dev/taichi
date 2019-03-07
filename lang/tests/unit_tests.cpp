#include "../tlang.h"
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

Vector complex_mul(const Vector &a, const Vector &b) {
  Vector ret(2);
  ret(0) = a(0) * b(0) - a(1) * b(1);
  ret(1) = a(0) * b(1) + a(1) * b(0);
  return ret;
}

auto mset = [] {
  CoreState::set_trigger_gdb_when_crash(true);
  int n = 1024;
  Program prog(Arch::x86_64);

  Global(a, i32);

  layout([&]() { root.fixed(0, n * n).place(a); });

  auto func = kernel([&]() {
    Declare(i);

    Vectorize(8);
    For(i, 0, n * n, [&] {
      Local(j) = 0;
      int limit = 20;
      if (false) {
        Local(c_re) = cast<float>(i / n) / float(n / 2) - 1.5f;
        Local(c_im) = cast<float>(i % n) / float(n / 2) - 1.0f;
        Local(z_re) = c_re;
        Local(z_im) = c_im;

        While(j < limit && (z_re * z_re + z_im * z_im) < 4.0f, [&] {
          Local(new_re) = z_re * z_re - z_im * z_im;
          Local(new_im) = 2.0f * z_re * z_im;
          z_re = c_re + new_re;
          z_im = c_im + new_im;
          j += 1;
        });
      } else {
        Vector c(2);

        c(0) = cast<float>(i / n) / float(n / 2) - 1.5f;
        c(1) = cast<float>(i % n) / float(n / 2) - 1.0f;

        Vector z = c;

        While(j < limit && z.norm2() < 4.0f, [&] {
          z = complex_mul(z, z) + c;
          j += 1;
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

auto ray_march = [] {
  CoreState::set_trigger_gdb_when_crash(true);
  int n = 512;
  Program prog(Arch::x86_64);
  prog.config.print_ir = true;

  Global(color_r, f32);
  Global(color_g, f32);
  Global(color_b, f32);

  layout([&]() { root.fixed(0, n * n * 2).place(color_r, color_g, color_b); });

  auto sdf = [&](Vector p_) {
    float alpha = -0.7f;
    Vector p(3);
    p(0) = (cos(alpha) * p_(0) + sin(alpha) * p_(2) + 1.0_f) % 2.0_f - 1.0_f;
    p(1) = p_(1);
    p(2) = -sin(alpha) * p_(0) + cos(alpha) * p_(2);

    auto dist_sphere = p.norm() - 0.5_f;
    auto dist_walls = min(p(2) + 6.0f, p(1) + 1.0f);
    Vector d(3);
    d(0) = abs(p(0) - 1.0_f) - 0.3_f;
    d(1) = abs(p(1) + 0.5_f) - 1.2_f;
    d(2) = abs(p(2) - 1.0_f) - 0.2_f;
    auto dist_cube = norm(d.map([](const Expr &v) { return max(v, 0.0f); })) +
                     min(max(max(d(0), d(1)), d(2)), 0.0_f);
    return min(dist_sphere, min(dist_walls, dist_cube));
  };

  float32 eps = 1e-5f;
  float32 dist_limit = 1e2;
  int limit = 200;

  auto ray_march = [&](Vector p, Vector dir) {
    Local(j) = 0;
    Local(dist) = 0.0f;

    While(j < limit && sdf(p + dist * dir) > eps && dist < dist_limit, [&] {
      dist += sdf(p + dist * dir);
      j += 1;
    });
    return dist;
  };

  auto normal = [&](Vector p) {
    float d = 1e-3f;
    Vector n(3);
    for (int i = 0; i < 3; i++) {
      Vector inc = p, dec = p;
      inc(i) += d;
      dec(i) -= d;
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
    Local(phi) = 2 * pi * Rand<float>();
    Local(r) = Rand<float>();
    Local(alpha) = 0.5_f * pi * (r * r);
    return sin(alpha) * (cos(phi) * u + sin(phi) * v) + cos(alpha) * n;
  };

  auto background = [](Vector dir) {
    return 1.0f * max(dir(1) + dir(0), 0.0f);
  };

  float fov = 0.3;

  auto main = kernel([&]() {
    Declare(i);
    Parallelize(8);
    Vectorize(8);
    For(i, 0, n * n * 2, [&] {
      Vector orig({0.0f, 0.0f, 12.0f}), c(3);

      c(0) = fov * (cast<float>(i / n) / float(n / 2) - 2.0f);
      c(1) = fov * (cast<float>(i % n) / float(n / 2) - 1.0f);
      c(2) = -1.0f;
      c = normalized(c);

      Vector color(3);
      color = Vector({1.0f, 1.0f, 1.0f});
      int depth_limit = 4;
      Local(depth) = 0;

      While(depth < depth_limit, [&] {
        depth += 1;
        Local(_dist) = ray_march(orig, c);
        If(_dist < dist_limit,
           [&] {
             orig += _dist * c;
             Vector nor;
             nor = normal(orig);
             c = normalized(out_dir(nor));
             orig += 0.01f * c;
             color *= 0.7_f;
           })
            .Else([&] {
              color = color * background(c);
              depth = depth_limit;
            });
      });

      color_r[i] += color(0);
      color_g[i] += color(1);
      color_b[i] += color(2);
    });
  });

  /// TC_P(measure_cpe(func, 1));

  GUI gui("ray march", Vector2i(n * 2, n));

  auto tone_map = [](real x) { return x; };
  constexpr int N = 1;
  for (int frame = 0;; frame++) {
    for (int i = 0; i < N; i++)
      main();
    real scale = 1.0_f / ((frame + 1) * N);
    for (int i = 0; i < n * n * 2; i++) {
      gui.buffer[i / n][i % n] =
          Vector4(tone_map(scale * color_r.val<float>(i)),
                  tone_map(scale * color_g.val<float>(i)),
                  tone_map(scale * color_b.val<float>(i)), 1);
    }
    gui.update();
  }
};
TC_REGISTER_TASK(ray_march);

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

    Vectorize(2);
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
  for (auto vec_size : {4}) {
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

  for (auto vec_size : {4}) {
    Program prog;

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
TC_TEST("mixed_simd3") {
  for (auto vec_size : {4}) {
    // why vec_size = 16 fails??
    Program prog;

    Vector a(DataType::f32, vec_size), b(DataType::f32, vec_size),
        c(DataType::f32, vec_size * 2);
    Global(sum, f32);

    int n = 8;

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
        auto diff_ = a[i].element_wise_prod(a[i]) - b[i].element_wise_prod(b[i]);
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

TLANG_NAMESPACE_END
