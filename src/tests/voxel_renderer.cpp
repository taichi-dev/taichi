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

  auto query_density_int = [&](Vector p_) {
    auto p = p_.cast_elements<int32>();
    auto inside_box =
        Var(0 <= p(0) && p(0) < grid_resolution && 0 <= p(1) &&
            p(1) < grid_resolution && 0 <= p(2) && p(2) < grid_resolution);
    auto ret = Var(0.0f);
    If(inside_box).Then([&] { ret = density[p]; });
    return ret;
  };

  auto get_next_hit = [&](const Vector &eye_o, const Vector &eye_d,
                          Expr &hit_distance, Vector &hit_pos, Vector &normal) {
    auto d = normalized(eye_d);
    auto pos = Var(eye_o + d * 1e-4f);

    auto rinv = Var(1.0f / d);
    auto rsign = Vector(3);
    for (int i = 0; i < 3; i++) {
      rsign(i) = cast<float32>(d(i) > 0.0f) * 2.0f - 1.0f;  // sign...
    }

    auto o = Var(pos * float32(grid_resolution));
    auto ipos = Var(floor(o));
    auto dis = Var((ipos - o + 0.5f + rsign * 0.5f).element_wise_prod(rinv));

    auto running = Var(1);
    auto i = Var(0);
    hit_distance = -1.0f;
    While(running, [&] {
      auto last_sample = Var(query_density_int(ipos));
      If(last_sample > 0.0f)
          .Then([&] {
            // intersect the cube
            auto mini =
                Var((ipos - o + Vector({0.5f, 0.5f, 0.5f}) - rsign * 0.5f)
                        .element_wise_prod(rinv));
            hit_distance =
                max(max(mini(0), mini(1)), mini(2)) * (1.0f / grid_resolution);
            hit_pos = pos + hit_distance * d;
            running = 0;
          })
          .Else([&] {
            auto mm = Var(Vector({0.0f, 0.0f, 0.0f}));
            If(dis(0) <= dis(1) && dis(0) < dis(2))
                .Then([&] { mm(0) = 1.0f; })
                .Else([&] {
                  If(dis(1) <= dis(0) && dis(1) <= dis(2))
                      .Then([&] { mm(1) = 1.0f; })
                      .Else([&] { mm(2) = 1.0f; });
                });
            dis += mm.element_wise_prod(rsign).element_wise_prod(rinv);
            ipos += mm.element_wise_prod(rsign);
            normal = -mm.element_wise_prod(rsign);
          });
      i += 1;
      If(i > 500).Then([&] { running = 0; });
    });
  };

  float32 fov = 0.7;

  auto background = [](Vector dir) {
    return 1.0f * max(dir(1) + dir(0), 0.0f);
  };

  auto out_dir = [&](Vector n) {
    auto u = Var(Vector({1.0f, 0.0f, 0.0f})), v = Var(Vector(3));
    If(abs(n(1)) < 1 - 1e-3f, [&] {
      u = normalized(cross(n, Vector({0.0f, 1.0f, 0.0f})));
    });
    v = cross(n, u);
    auto phi = Var(2 * pi * Rand<float32>());
    auto r = Var(Rand<float32>());
    auto alpha = Var(0.5f * pi * (r * r));
    return sin(alpha) * (cos(phi) * u + sin(phi) * v) + cos(alpha) * n;
  };

  Kernel(main).def([&]() {
    For(0, n * n * 2, [&](Expr i) {
      auto orig = Var(Vector({0.5f, 0.2f, 1.0f}));

      auto c = Var(Vector(
          {fov * ((Rand<float32>() + cast<float32>(i / n)) / float32(n / 2) -
                  2.01f),
           fov * ((Rand<float32>() + cast<float32>(i % n)) / float32(n / 2) -
                  1.01f),
           -1.0f}));

      c = normalized(c);

      int depth_limit = 4;
      auto depth = Var(0);

      auto color = Var(Vector({1.0f, 1.0f, 1.0f}));

      While(depth < depth_limit, [&] {
        auto hit_dist = Var(0.0f);
        auto hit_pos = Var(Vector({1.0f, 1.0f, 1.0f}));
        auto normal = Var(Vector({1.0f, 1.0f, 1.0f}));
        get_next_hit(orig, c, hit_dist, hit_pos, normal);

        depth += 1;
        If(hit_dist > 0.0f)
            .Then([&] {
              c = normalized(out_dir(normal));
              orig = hit_pos;
              color *= 0.7f;
            })
            .Else([&] {
              color = color * background(c);
              depth = depth_limit;
            });
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
