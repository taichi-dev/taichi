#include <taichi/visual/gui.h>
#include "../tlang.h"

TLANG_NAMESPACE_BEGIN

// https://github.com/yuanming-hu/topo_opt_private/blob/master/misc/volume_rendering_kernel.cu

auto voxel_renderer = [] {
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

  auto query_density_int = [&](Vector p) {
    auto inside_box =
        Var(0 <= p(0) && p(0) < grid_resolution && 0 <= p(1) &&
            p(1) < grid_resolution && 0 <= p(2) && p(2) < grid_resolution);
    auto ret = Var(0.0f);
    If(inside_box).Then([&] { ret = density[p]; });
    return ret;
  };

  auto get_next_hit = [&](const Vector &eye_o, const Vector &eye_d, Expr hit,
                          Vector &hit_pos, bool need_position = true) {
    auto d = normalized(eye_d);
    hit = Expr(0);

    /*
    auto tnear, tfar;
    auto box_size = get_box_size();
    auto box_res = get_box_res();

    hit = (bool)intersectBox(eyeRay, -box_size, box_size, &tnear, &tfar);

    if (!hit) {
      hit = false;
      return;
    }

    if (tnear < 0.0f)
      tnear = 0.0f;  // clamp to near plane
    */
    auto tnear = Var(0.0f);

    auto pos = eye_o + d * (tnear + 1e-4f);

    auto ri = Var(1.0f / d);
    auto rs = Vector(3);
    for (int i = 0; i < 3; i++) {
      rs(i) = (d(i) > 0) * 2 - 1;  // sign...
    }
    auto o = Var((pos + 1) * 0.5f * grid_resolution);
    auto ipos = Var(o.cast_elements<int32>());
    auto dis = Var((ipos - o + 0.5f + rs * 0.5f) * ri);

    auto last_sample = Var(0.0f);
    auto running = Var(1);
    auto i = Var(0);
    While(running, [&] {
      last_sample = query_density_int(ipos);

      If(last_sample > 0).Then([&] {
        // intersect the cube
        auto mini = Var(ipos - o + Vector({0.5f, 0.5f, 0.5f}) - rs * 0.5f) * ri;
        auto t = Var(max(max(mini(0), mini(1)), mini(2)) *
                     (1.0f / grid_resolution) * 2.0f);
        hit_pos = pos + t * d;
        hit = 1;
        return;
      });

      auto mm = Var(Vector({0.0f, 0.0f, 0.0f}));
      If(dis(0) <= dis(1) && dis(0) < dis(2))
          .Then([&] {
            mm = Vector({1.0f, 0.0f, 0.0f});
          })
          .Else([&] {
            If(dis(1) <= dis(0) && dis(1) <= dis(2))
                .Then([&] {
                  mm = Vector({0.0f, 1.0f, 0.0f});
                })
                .Else([&] {
                  mm = Vector({0.0f, 0.0f, 1.0f});
                });
          });
      dis += mm.element_wise_prod(rs).element_wise_prod(ri);
      ipos += mm.element_wise_prod(rs);
      i += 1;
    });
    return last_sample;
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

  GUI gui("Voxel Renderer", Vector2i(n * 2, n));

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
TC_REGISTER_TASK(voxel_renderer);

TLANG_NAMESPACE_END
