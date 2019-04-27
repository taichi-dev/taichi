#include <taichi/visual/gui.h>
#include "../tlang.h"

TLANG_NAMESPACE_BEGIN

auto volume_renderer = [] {
  CoreState::set_trigger_gdb_when_crash(true);

  int n = 512;
  int grid_resolution = 256;

  auto f = fopen("snow_density_256.bin", "rb");
  TC_ASSERT_INFO(f, "./snow_density_256.bin not found");
  std::vector<float32> density_field(pow<3>(grid_resolution));
  std::fread(density_field.data(), sizeof(float32), density_field.size(), f);
  std::fclose(f);

  Program prog(Arch::gpu);
  prog.config.print_ir = true;

  Vector buffer(DataType::f32, 3);

  layout([&]() {
    root.dense(0, n * n * 2).place(buffer(0), buffer(1), buffer(2));
  });

  // The signed distance field of the geometry
  auto sdf = [&](Vector q) {
    float32 alpha = -0.7f;
    auto p =
        Vector({(cos(alpha) * q(0) + sin(alpha) * q(2) + 1.0f) % 2.0f - 1.0f,
                q(1), -sin(alpha) * q(0) + cos(alpha) * q(2)});

    auto dist_sphere = p.norm() - 0.5f;
    auto dist_walls = min(p(2) + 6.0f, p(1) + 1.0f);
    Vector d =
        Var(abs(p + Vector({-1.0f, 0.5f, -1.0f})) - Vector({0.3f, 1.2f, 0.2f}));

    auto dist_cube = norm(d.map([](const Expr &v) { return max(v, 0.0f); })) +
                     min(max(max(d(0), d(1)), d(2)), 0.0f);

    return min(dist_sphere, min(dist_walls, dist_cube));
  };

  float32 eps = 1e-5f;
  float32 dist_limit = 1e2;
  int limit = 200;

  // shoot a ray
  auto ray_march = [&](Vector p, Vector dir) {
    auto j = Var(0);
    auto dist = Var(0.0f);
    While(j < limit && sdf(p + dist * dir) > eps && dist < dist_limit, [&] {
      dist += sdf(p + dist * dir);
      j += 1;
    });
    return dist;
  };

  auto background = [](Vector dir) {
    return 1.0f * max(dir(1) + dir(0), 0.0f);
  };

  float32 fov = 0.3;

  Kernel(main).def([&]() {
    Parallelize(8);
    Vectorize(8);
    For(0, n * n * 2, [&](Expr i) {
      auto orig = Var(Vector({0.0f, 0.0f, 12.0f}));

      auto c = Var(Vector({fov * (cast<float32>(i / n) / float32(n / 2) - 2.0f),
                           fov * (cast<float32>(i % n) / float32(n / 2) - 1.0f),
                           -1.0f}));

      c = normalized(c);

      auto color = Var(Vector({1.0f, 1.0f, 1.0f}));

      auto dist = Var(ray_march(orig, c));
      If(dist < dist_limit).Then([&] {
        color = color * (dist / (dist + 1.0f));
      });

      buffer[i] += color;
    });
  });

  GUI gui("Volume Renderer", Vector2i(n * 2, n));

  auto tone_map = [](real x) { return x; };
  constexpr int N = 100;
  for (int frame = 0;; frame++) {
    for (int i = 0; i < N; i++)
      main();
    real scale = 1.0f / ((frame + 1) * N);
    for (int i = 0; i < n * n * 2; i++) {
      gui.buffer[i / n][i % n] =
          Vector4(tone_map(scale * buffer(0).val<float32>(i)),
                  tone_map(scale * buffer(1).val<float32>(i)),
                  tone_map(scale * buffer(2).val<float32>(i)), 1);
    }
    gui.update();
  }
};
TC_REGISTER_TASK(volume_renderer);

TLANG_NAMESPACE_END
