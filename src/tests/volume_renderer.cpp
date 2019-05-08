#include <taichi/visual/gui.h>
#include "../tlang.h"

TLANG_NAMESPACE_BEGIN

bool use_gui = false;

auto volume_renderer = [] {
  // CoreState::set_trigger_gdb_when_crash(true);

  int depth_limit = 40;
  int n = 512;
  int grid_resolution = 256;
  Vector3 albedo(0.95, 0.95, 0.95);
  float32 scale = 724.0;
  float32 one_over_four_pi = 0.07957747154;
  float32 pi = 3.14159265359;

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

  auto point_inside_box = [&](Vector p) {
    return Var(0.0f <= p(0) && p(0) < 1.0f && 0.0f <= p(1) && p(1) < 1.0f &&
               0.0f <= p(2) && p(2) < 1.0f);
  };

  // If p is in the density field, return the density, otherwise return 0
  auto query_density = [&](Vector p) {
    auto inside_box = point_inside_box(p);
    auto ret = Var(0.0f);
    If(inside_box).Then([&] {
      auto i = floor(p(0) * float32(grid_resolution));
      auto j = floor(p(1) * float32(grid_resolution));
      auto k = floor(p(2) * float32(grid_resolution));
      ret = density[i, j, k] * scale;
    });
    return ret;
  };

  // Adapted from Mitsuba: include/mitsuba/core/aabb.h#L308
  auto box_intersect = [&](Vector o, Vector d, Expr &near_t, Expr &far_t) {
    auto result = Var(1);

    /* For each pair of AABB planes */
    for (int i = 0; i < 3; i++) {
      auto origin = o(i);
      auto min_val = Var(0.f);
      auto max_val = Var(1.f);
      auto d_rcp = Var(1.f / d(i));

      If(d(i) == 0.f)
          .Then([&] {
            /* The ray is parallel to the planes */
            If(origin < min_val || origin > max_val, [&] { result = 0; });
          })
          .Else([&] {
            /* Calculate intersection distances */
            auto t1 = Var((min_val - origin) * d_rcp);
            auto t2 = Var((max_val - origin) * d_rcp);

            If(t1 > t2, [&] {
              auto tmp = Var(t1);
              t1 = t2;
              t2 = tmp;
            });

            near_t = max(t1, near_t);
            far_t = min(t2, far_t);

            If(near_t > far_t, [&] { result = 0; });
          });
    }

    return result;
  };

  // Adapted from Mitsuba: src/libcore/warp.cpp#L25
  auto sample_phase_isotropic = [&]() {
    auto z = Var(1.0f - 2.0f * Rand<float32>());
    auto r = Var(sqrt(1.0f - z * z));
    auto phi = Var(2.0f * pi * Rand<float32>());
    auto sin_phi = Var(sin(phi));
    auto cos_phi = Var(cos(phi));
    return Var(Vector({r * cos_phi, r * sin_phi, z}));
  };

  auto pdf_phase_isotropic = [&]() { return Var(one_over_four_pi); };

  auto eval_phase_isotropic = [&]() { return pdf_phase_isotropic(); };

  // Direct sample light
  auto sample_light = [&](Vector p, float32 inv_max_density) {
    auto Le = Var(Vector({5.0f, 5.0f, 5.0f}));
    auto light_p = Var(Vector({0.5f, 1.5f, 0.5f}));
    auto dir_to_p = Var(p - light_p);
    auto dist_to_p = Var(dir_to_p.norm());
    auto inv_dist_to_p = Var(1.f / dist_to_p);
    dir_to_p = normalized(dir_to_p);

    auto near_t = Var(-std::numeric_limits<float>::max());
    auto far_t = Var(std::numeric_limits<float>::max());
    auto hit = box_intersect(light_p, dir_to_p, near_t, far_t);
    auto interaction = Var(0);
    auto transmittance = Var(1.f);

    If(hit, [&] {
      auto cond = Var(hit);
      auto t = Var(near_t);

      While(cond, [&] {
        t -= log(1.f - Rand<float32>()) * inv_max_density;

        p = Var(light_p + t * dir_to_p);
        If(t >= dist_to_p || !point_inside_box(p))
            .Then([&] { cond = 0; })
            .Else([&] {
              auto density_at_p = query_density(p);
              If(density_at_p * inv_max_density > Rand<float32>()).Then([&] {
                cond = 0;
                transmittance = Var(0.f);
              });
            });
      });
    });

    return Var(transmittance * Le * inv_dist_to_p * inv_dist_to_p);
  };

  // Woodcock tracking
  auto sample_distance = [&](Vector o, Vector d, float32 inv_max_density,
                             Expr &dist, Vector &sigma_s, Expr &transmittance,
                             Vector &p) {
    auto near_t = Var(-std::numeric_limits<float>::max());
    auto far_t = Var(std::numeric_limits<float>::max());
    auto hit = box_intersect(o, d, near_t, far_t);

    auto cond = Var(hit);
    auto interaction = Var(0);
    auto t = Var(near_t);

    While(cond, [&] {
      t -= log(1.f - Rand<float32>()) * inv_max_density;

      p = Var(o + t * d);
      If(t >= far_t || !point_inside_box(p)).Then([&] { cond = 0; }).Else([&] {
        auto density_at_p = query_density(p);
        If(density_at_p * inv_max_density > Rand<float32>()).Then([&] {
          sigma_s(0) = Var(density_at_p * albedo[0]);
          sigma_s(1) = Var(density_at_p * albedo[1]);
          sigma_s(2) = Var(density_at_p * albedo[2]);
          If(density_at_p != 0.f).Then([&] {
            transmittance = 1.f / density_at_p;
          });
          cond = 0;
          interaction = 1;
        });
      });
    });

    dist = t - near_t;

    return hit && interaction;
  };

  auto background = [](Vector dir) { return Vector({0.4f, 0.4f, 0.4f}); };

  float32 fov = 0.7;

  auto max_density =
      *std::max_element(density_field.begin(), density_field.end()) * scale;
  auto inv_max_density = 0.f;
  if (max_density > 0.f) {
    inv_max_density = 1.f / max_density;
  }

  Kernel(main).def([&]() {
    For(0, n * n * 2, [&](Expr i) {
      auto orig = Var(Vector({0.5f, 0.3f, 1.5f}));

      auto c = Var(Vector(
          {fov * ((Rand<float32>() + cast<float32>(i / n)) / float32(n / 2) -
                  2.0f),
           fov * ((Rand<float32>() + cast<float32>(i % n)) / float32(n / 2) -
                  1.0f),
           -1.0f}));

      c = normalized(c);

      auto color = Var(Vector({1.0f, 1.0f, 1.0f}));
      auto Li = Var(Vector({0.0f, 0.0f, 0.0f}));
      auto throughput = Var(Vector({1.0f, 1.0f, 1.0f}));
      auto depth = Var(0);

      While(depth < depth_limit, [&] {
        auto dist = Var(0.f);
        auto transmittance = Var(0.f);
        auto sigma_s = Var(Vector({0.f, 0.f, 0.f}));
        auto interaction_p = Var(Vector({0.f, 0.f, 0.f}));
        auto interaction =
            sample_distance(orig, c, inv_max_density, dist, sigma_s,
                            transmittance, interaction_p);

        depth += 1;
        If(interaction)
            .Then([&] {
              throughput =
                  throughput.element_wise_prod(sigma_s * transmittance);

              auto phase_value = eval_phase_isotropic();
              auto light_value = sample_light(interaction_p, inv_max_density);
              Li += phase_value * throughput.element_wise_prod(light_value);

              orig = interaction_p;
              c = sample_phase_isotropic();
            })
            .Else([&] {
              Li += throughput.element_wise_prod(background(c));
              depth = depth_limit;
            });
      });

      buffer[i] += Li;
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

  std::unique_ptr<GUI> gui = nullptr;

  if (use_gui) {
    gui = std::make_unique<GUI>("Volume Renderer", Vector2i(n * 2, n));
  }
  Vector2i render_size(n * 2, n);
  Array2D<Vector4> render_buffer;

  auto tone_map = [](real x) { return x; };

  constexpr int N = 100;
  for (int frame = 0; frame < 100; frame++) {
    for (int i = 0; i < N; i++) {
      main();
    }
    prog.profiler_print();

    real scale = 1.0f / ((frame + 1) * N);
    render_buffer.initialize(render_size);
    std::unique_ptr<Canvas> canvas;
    canvas = std::make_unique<Canvas>(render_buffer);
    for (int i = 0; i < n * n * 2; i++) {
      render_buffer[i / n][i % n] =
          Vector4(tone_map(scale * buffer(0).val<float32>(i)),
                  tone_map(scale * buffer(1).val<float32>(i)),
                  tone_map(scale * buffer(2).val<float32>(i)), 1);
    }
    if (use_gui) {
      gui->canvas->img = canvas->img;
      gui->update();
    } else {
      canvas->img.write_as_image(
          fmt::format("{:05d}-{:05d}-{:05d}.png", frame, N, depth_limit));
    }
  }
};
TC_REGISTER_TASK(volume_renderer);

auto volume_renderer_gui = [] {
  use_gui = true;
  volume_renderer();
};

TC_REGISTER_TASK(volume_renderer_gui);

TLANG_NAMESPACE_END
