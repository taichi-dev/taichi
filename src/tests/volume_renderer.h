#pragma once
#include <taichi/visual/gui.h>
#include "../tlang.h"

TLANG_NAMESPACE_BEGIN

class Renderer {
 public:
  int n;
  Vector2i sky_map_size;
  int n_sky_samples;
  Program::Kernel *main, *dilate;
  Vector buffer;
  int depth_limit;
  Expr density;
  int grid_resolution = 256;
  bool use_sky_map = false;
  int block_size = 4;
  float32 one_over_four_pi = 0.07957747154f;
  float32 pi = 3.14159265359f;
  Vector sky_map;
  Vector sky_sample_color;
  Vector sky_sample_uv;

  Renderer() {
    depth_limit = 20;
    n = 512;

    sky_map_size = Vector2i(512, 128);
    n_sky_samples = 1024;

    Vector buffer(DataType::f32, 3);
    this->buffer.set(buffer);
    Global(density, f32);
    this->density.set(density);
    sky_map.declare(DataType::f32, 3);
    sky_sample_color.declare(DataType::f32, 3);
    sky_sample_uv.declare(DataType::f32, 2);
  }

  void place_data() {
    root.dense(Index(0), n * n * 2).place(buffer(0), buffer(1), buffer(2));

    root.dense(Indices(0, 1, 2), 4)
        .bitmasked()
        .dense(Indices(0, 1, 2), grid_resolution / block_size)
        // .pointer()
        .bitmasked()
        .dense(Indices(0, 1, 2), block_size)
        .place(density);

    root.dense(Indices(0, 1), {sky_map_size[0], sky_map_size[1]})
        .place(sky_map);
    root.dense(Indices(0), n_sky_samples)
        .place(sky_sample_color)
        .place(sky_sample_uv);
  }

  void declare_kernels() {
    auto inv_max_density = 1.0f / 724.0f;
    Vector3 albedo(0.9, 0.95, 1);
    auto const block_size = 4;
    auto lower_bound = -0.0f;
    auto upper_bound = 1.0f;

    auto point_inside_box = [&](Vector p) {
      return Var(lower_bound <= p(0) && p(0) < upper_bound &&
                 lower_bound <= p(1) && p(1) < upper_bound &&
                 lower_bound <= p(2) && p(2) < upper_bound);
    };

    auto query_active = [&](Vector p) {
      auto inside_box = Var(1);  // point_inside_box(p);
      auto ret = Var(0);
      If(inside_box).Then([&] {
        auto i = cast<int>(floor(p(0) * float32(grid_resolution)));
        auto j = cast<int>(floor(p(1) * float32(grid_resolution)));
        auto k = cast<int>(floor(p(2) * float32(grid_resolution)));
        ret = Probe(density, (i, j, k));
      });
      return ret;
    };

    // If p is in the density field, return the density, otherwise return 0
    auto query_density = [&](Vector p) {
      auto inside_box = point_inside_box(p);
      auto ret = Var(0.0f);
      If(inside_box).Then([&] {
        auto i = cast<int>(floor(p(0) * float32(grid_resolution)));
        auto j = cast<int>(floor(p(1) * float32(grid_resolution)));
        auto k = cast<int>(floor(p(2) * float32(grid_resolution)));
        ret = density[i, j, k];
      });
      return ret;
    };

    // Adapted from Mitsuba: include/mitsuba/core/aabb.h#L308
    auto box_intersect = [&](Vector o, Vector d, Expr &near_t, Expr &far_t) {
      auto result = Var(1);

      /* For each pair of AABB planes */
      for (int i = 0; i < 3; i++) {
        auto origin = o(i);
        auto min_val = Var(lower_bound);
        auto max_val = Var(upper_bound);
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
      auto ret = Vector({0.0f, 0.0f, 0.0f});
      if (!use_sky_map) {  // point light source
        auto Le = Var(700.0f * Vector({5.0f, 5.0f, 5.0f}));
        auto light_p = Var(10.0f * Vector({2.5f, 1.0f, 0.5f}));
        auto dir_to_p = Var(p - light_p);
        auto dist_to_p = Var(dir_to_p.norm());
        auto inv_dist_to_p = Var(1.f / dist_to_p);
        dir_to_p = normalized(dir_to_p);

        auto transmittance = Var(1.f);

        auto cond = Var(1);
        auto t = Var(0.0f);

        While(cond, [&] {
          t -= log(1.f - Rand<float32>()) * inv_max_density;

          auto q = Var(p - t * dir_to_p);

          If(!point_inside_box(q)).Then([&] { cond = 0; }).Else([&] {
            If(!query_active(q))
                .Then([&] { t += 1.0f * block_size / grid_resolution; })
                .Else([&] {
                  auto density_at_p = query_density(q);
                  If(density_at_p * inv_max_density > Rand<float32>())
                      .Then([&] {
                        cond = 0;
                        transmittance = Var(0.f);
                      });
                });
          });
        });

        ret = Var(transmittance * Le * inv_dist_to_p * inv_dist_to_p);
      } else {
        auto sample = Var(cast<int>(Rand<float32>() * float32(n_sky_samples)));
        auto uv = Var(sky_sample_uv[sample]);
        auto phi = Var(uv(0) * (2 * pi));
        auto theta = Var(uv(1) * (pi / 2));
        // auto phi = Var(0.0f);
        // auto theta = Var(0.9f);

        auto dir_to_sky = Var(
            Vector({cos(phi) * cos(theta), sin(theta), sin(phi) * cos(theta)}));
        /*
        auto dir_to_sky = Var(
            normalized(Vector({2.5f, 1.0f, 0.5f})));
                  */

        auto Le = Var(sky_sample_color[sample]);
        auto near_t = Var(-std::numeric_limits<float>::max());
        auto far_t = Var(std::numeric_limits<float>::max());
        auto hit = box_intersect(p, dir_to_sky, near_t, far_t);
        auto transmittance = Var(1.f);

        If(hit, [&] {
          auto cond = Var(hit);
          auto t = Var(0.0f);

          While(cond, [&] {
            t -= log(1.f - Rand<float32>()) * inv_max_density;
            auto q = Var(p + t * dir_to_sky);
            If(!point_inside_box(q)).Then([&] { cond = 0; }).Else([&] {
              If(!query_active(q))
                  .Then([&] { t += 1.0f * block_size / grid_resolution; })
                  .Else([&] {
                    auto density_at_p = query_density(q);
                    If(density_at_p * inv_max_density > Rand<float32>())
                        .Then([&] {
                          cond = 0;
                          transmittance = Var(0.f);
                        });
                  });
            });
          });
        });

        ret = Var(transmittance * Le);
      }
      return ret;
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
        If(t >= far_t || !point_inside_box(p))
            .Then([&] { cond = 0; })
            .Else([&] {
              If(!query_active(p))
                  .Then([&] { t += 1.0f * block_size / grid_resolution; })
                  .Else([&] {
                    auto density_at_p = query_density(p);
                    If(density_at_p * inv_max_density > Rand<float32>())
                        .Then([&] {
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
      });

      dist = t - near_t;

      return hit && interaction;
    };

    auto background = [&](Vector dir) {
      // return Vector({0.4f, 0.4f, 0.4f});
      auto ret = Var(Vector({0.0f, 0.0f, 0.0f}));
      If(dir(1) >= 0.0f)
          .Then([&] {
            auto phi = Var(atan2(dir(0), dir(2)));
            auto theta = Var(asin(dir(1)));
            auto u = cast<int32>((phi + pi) * (sky_map_size[0] / (2 * pi)));
            auto v = cast<int32>(theta * (sky_map_size[1] / pi * 2));
            ret = sky_map[u, v];
          })
          .Else([&] {
            ret = Vector({0.6f, 0.6f, 0.6f});
          });
      return ret;
    };

    float32 fov = 0.6;

    main = &kernel([&]() {
      BlockDim(32);
      For(0, n * n * 2, [&](Expr i) {
        auto orig = Var(Vector({0.5f, 0.3f, 1.5f}));

        auto bid = Var(i / 32);
        auto tid = Var(i % 32);
        auto x = Var(bid / (n / 4) * 8 + tid / 4),
             y = Var(bid % (n / 4) * 4 + tid % 4);

        auto c = Var(Vector(
            {fov *
                 ((Rand<float32>() + cast<float32>(x)) / float32(n / 2) - 2.0f),
             fov *
                 ((Rand<float32>() + cast<float32>(y)) / float32(n / 2) - 1.0f),
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
                if (use_sky_map) {
                  If(depth == 1).Then([&] {
                    Li += throughput.element_wise_prod(background(c));
                  });
                }
                depth = depth_limit;
              });
        });

        buffer[x * n + y] += Li;
      });
    });

    std::FILE *f;
    if (use_sky_map) {
      f = fopen("sky_map.bin", "rb");
      TC_ASSERT_INFO(f, "./sky_map.bin not found");
      std::vector<uint32> sky_map_data(sky_map_size.prod() * 3);
      std::fread(sky_map_data.data(), sizeof(uint32), sky_map_data.size(), f);

      f = fopen("sky_samples.bin", "rb");
      TC_ASSERT_INFO(f, "./sky_samples.bin not found");
      std::vector<uint32> sky_sample_data(n_sky_samples * 5);
      std::fread(sky_sample_data.data(), sizeof(uint32), sky_sample_data.size(),
                 f);

      for (int i = 0; i < sky_map_size[0]; i++) {
        for (int j = 0; j < sky_map_size[1]; j++) {
          for (int d = 0; d < 3; d++) {
            auto l = sky_map_data[(i * sky_map_size[1] + j) * 3 + d] *
                     (1.0f / (1 << 20));
            sky_map(d).val<float32>(i, j) = l * 1000;
          }
        }
      }

      for (int i = 0; i < n_sky_samples; i++) {
        for (int d = 0; d < 2; d++) {
          sky_sample_uv(d).val<float32>(i) =
              sky_sample_data[i * 5 + 1 - d] * (1.0f / (sky_map_size[d]));
        }
        for (int d = 0; d < 3; d++) {
          sky_sample_color(d).val<float32>(i) =
              sky_sample_data[i * 5 + 2 + d] * (1.0f / (1 << 20));
        }
      }
    }
    // expand blocks
    dilate = &kernel([&] {
      For(density, [&](Expr i, Expr j, Expr k) {
        for (int x = -1; x < 2; x++) {
          for (int y = -1; y < 2; y++) {
            for (int z = -1; z < 2; z++) {
              density[i + x * block_size, j + y * block_size,
                      k + z * block_size] += 0.0f;  // simply activate the block
            }
          }
        }
      });
    });
  }

  void preprocess_volume() {
    (*dilate)();
  }

  void sample() {
    (*main)();
  }
};

TLANG_NAMESPACE_END
