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
  Global(density, f32);

  layout([&]() {
    root.dense(Index(0), n * n * 2).place(buffer(0), buffer(1), buffer(2));
    root.dense(Indices(0, 1, 2), grid_resolution).place(density);
  });

  // If p is in the density field return the density, other 0
  auto query_density = [&](Vector p) {
    auto inside_box = Var(0.0f <= p(0) && p(0) < 1.0f && 0.0f <= p(1) &&
                          p(1) < 1.0f && 0.0f <= p(2) && p(2) < 1.0f);
    auto ret = Var(0.0f);
    If(inside_box).Then([&] {
      auto i = floor(p(0) * float32(grid_resolution));
      auto j = floor(p(1) * float32(grid_resolution));
      auto k = floor(p(2) * float32(grid_resolution));
      ret = density[i, j, k];
    });
    return ret;
  };

  auto background = [](Vector dir) {
    return 1.0f * max(dir(1) + dir(0), 0.0f);
  };

  float32 fov = 0.7;

  Kernel(main).def([&]() {
    For(0, n * n * 2, [&](Expr i) {
      auto orig = Var(Vector({0.5f, 0.3f, 1.5f}));

      auto c = Var(Vector({fov * (cast<float32>(i / n) / float32(n / 2) - 2.0f),
                           fov * (cast<float32>(i % n) / float32(n / 2) - 1.0f),
                           -1.0f}));

      c = normalized(c);

      auto color = Var(Vector({1.0f, 1.0f, 1.0f}));

      For(0, 200, [&](Expr k) {
        auto p = Var(orig + c * ((cast<float32>(k) + Rand<float32>()) * 0.01f));
        color *= (1.0_f - query_density(p) * 0.1f);
      });

      buffer[i] += color;
    });
  });

  for (int i = 0; i < grid_resolution; i++) {
    for (int j = 0; j < grid_resolution; j++) {
      for (int k = 0; k < grid_resolution; k++) {
        density.val<float32>(i, j, k) =
            density_field[i * grid_resolution * grid_resolution +
                          j * grid_resolution + k];
      }
    }
  }

  GUI gui("Volume Renderer", Vector2i(n * 2, n));

  auto tone_map = [](real x) { return x; };
  constexpr int N = 10;
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
