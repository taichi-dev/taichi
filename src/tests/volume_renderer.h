#pragma once
#include <taichi/visual/gui.h>
#include "../tlang.h"

TLANG_NAMESPACE_BEGIN

class TRenderer {
 public:
  Vector2i output_res;
  Vector2i sky_map_size;
  int n_sky_samples;
  Program::Kernel *main, *dilate, *clear_buffer;
  Vector buffer;
  Expr density;
  int grid_resolution;
  bool use_sky_map = true;
  int block_size = 4;
  float32 one_over_four_pi = 0.07957747154f;
  float32 pi = 3.14159265359f;
  Vector sky_map;
  Vector sky_sample_color;
  Vector sky_sample_uv;
  Vector box_min;
  Vector box_max;
  int acc_samples;
  Dict param;

  bool needs_update() {
    for (int i = 0; i < sizeof(Parameters); i++) {
      if (((char *)(&old_parameters))[i] != ((char *)(&parameters))[i]) {
        return true;
      }
    }
    return false;
  }

  TRenderer(Dict param) : param(param) {
    grid_resolution = param.get("grid_resolution", 256);
    old_parameters.depth_limit = -1;  // force initial update
    output_res = param.get("output_res", Vector2i(1024, 512));

    acc_samples = 0;

    TC_ASSERT(bit::is_power_of_two(output_res.x));
    TC_ASSERT(bit::is_power_of_two(output_res.y));

    sky_map_size = Vector2i(512, 128);
    n_sky_samples = 1024;

    density.declare(DataType::f32);
    buffer.declare(DataType::f32, 3);
    sky_map.declare(DataType::f32, 3);
    sky_sample_color.declare(DataType::f32, 3);
    sky_sample_uv.declare(DataType::f32, 2);

    initial = true;

    depth_limit.declare(DataType::i32);
    inv_max_density.declare(DataType::f32);
    max_density.declare(DataType::f32);
    ground_y.declare(DataType::f32);
    light_phi.declare(DataType::f32);
    light_theta.declare(DataType::f32);
    light_smoothness.declare(DataType::f32);
    density_scale.declare(DataType::f32);

    box_min.declare(DataType::f32, 3);
    box_max.declare(DataType::f32, 3);
  }

  void place_data() {
    root.dense(Index(0), output_res.prod())
        .place(buffer(0), buffer(1), buffer(2));

    if (grid_resolution <= 256) {
      root.dense(Indices(0, 1, 2), 4)
          .bitmasked()
          .dense(Indices(0, 1, 2), grid_resolution / block_size)
          // .pointer()
          .bitmasked()
          .dense(Indices(0, 1, 2), block_size)
          .place(density);
    } else {
      root.dense(Indices(0, 1, 2), grid_resolution / block_size)
          // .pointer()
          .bitmasked()
          .dense(Indices(0, 1, 2), block_size)
          .place(density);
    }

    root.dense(Indices(0, 1), {sky_map_size[0], sky_map_size[1]})
        .place(sky_map);
    root.dense(Indices(0), n_sky_samples)
        .place(sky_sample_color)
        .place(sky_sample_uv);

    // parameters
    root.place(depth_limit);
    root.place(inv_max_density);
    root.place(max_density);
    root.place(ground_y);
    root.place(light_phi);
    root.place(light_theta);
    root.place(light_smoothness);
    root.place(density_scale);
    root.place(box_min);
    root.place(box_max);
  }

  struct Parameters {
    int depth_limit;
    float32 max_density;
    float32 ground_y;
    float32 light_phi;
    float32 light_theta;
    float32 light_smoothness;
    float32 box_min[3];
    float32 box_max[3];
    float32 density_scale;
  };

  Expr depth_limit;
  Expr inv_max_density;
  Expr max_density;
  Expr ground_y;
  Expr light_phi, light_theta, light_smoothness;
  Expr density_scale;
  bool initial;

  void update_parameters() {
    (*clear_buffer)();
    depth_limit.val<int32>() = parameters.depth_limit;
    max_density.val<float32>() = parameters.max_density;
    inv_max_density.val<float32>() = 1.0f / parameters.max_density;
    ground_y.val<float32>() = parameters.ground_y;
    light_phi.val<float32>() = parameters.light_phi;
    light_theta.val<float32>() = parameters.light_theta;
    light_smoothness.val<float32>() = parameters.light_smoothness;
    density_scale.val<float32>() = parameters.density_scale;
    for (int i = 0; i < 3; i++) {
      box_min(i).val<float32>() = parameters.box_min[i];
      box_max(i).val<float32>() = parameters.box_max[i];
    }
    old_parameters = parameters;
  }

  Parameters old_parameters;
  Parameters parameters;

  void check_param_update() {
    if (initial) {
      parameters.depth_limit = param.get("depth_limit", 20);
      parameters.max_density = param.get("max_density", 724.0f);
      parameters.ground_y = param.get("ground_y", 0.029f);
      parameters.light_phi = param.get("light_phi", 0.419f);
      parameters.light_theta = param.get("light_theta", 0.218f);
      parameters.light_smoothness = param.get("light_smoothness", 0.05f);
      parameters.density_scale = param.get("density_scale", 400);
      for (int i = 0; i < 3; i++) {
        parameters.box_min[i] = param.get("box_min", Vector3(0.0f))(i);
        parameters.box_max[i] = param.get("box_max", Vector3(1.0f))(i);
      }
      initial = false;
    }
    if (needs_update()) {
      update_parameters();
      acc_samples = 0;
    }
  }

  void declare_kernels() {
    Vector3 albedo(0.9, 0.95, 1);
    auto const block_size = 4;

    auto point_inside_box = [&](Vector p) {
      return Var(box_min(0) <= p(0) && p(0) < box_max(0) &&
                 box_min(1) <= p(1) && p(1) < box_max(1) &&
                 box_min(2) <= p(2) && p(2) < box_max(2));
    };

    auto query_active = [&](Vector p) {
      auto inside_box = point_inside_box(p);
      auto ret = Var(0);
      If(inside_box).Then([&] {
        auto i = cast<int>(floor(p(0) * float32(grid_resolution)));
        auto j = cast<int>(floor(p(1) * float32(grid_resolution)));
        auto k = cast<int>(floor(p(2) * float32(grid_resolution)));
        ret = Probe(density, (i, j, k));
      });
      return ret;
    };

    auto query_density_int = [&](Vector p_) {
      auto p = p_.cast_elements<int32>();
      auto inside_box =
          Var(0 <= p(0) && p(0) < grid_resolution && 0 <= p(1) &&
              p(1) < grid_resolution && 0 <= p(2) && p(2) < grid_resolution);
      auto ret = Var(0.0f);
      If(inside_box).Then([&] { ret = density[p]; });
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
        ret = density[i, j, k] * density_scale;
      });
      return ret;
    };

    // Adapted from Mitsuba: include/mitsuba/core/aabb.h#L308
    auto box_intersect = [&](Vector o, Vector d, Expr &near_t, Expr &far_t) {
      auto result = Var(1);

      /* For each pair of AABB planes */
      for (int i = 0; i < 3; i++) {
        auto origin = o(i);
        auto min_val = Var(box_min(i));
        auto max_val = Var(box_max(i));
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

    // The signed distance field of the geometry
    auto sdf = [&](Vector q) {
      /*
      float32 alpha = -0.7f;
      auto p =
          Vector({(cos(alpha) * q(0) + sin(alpha) * q(2) + 1.0f) % 2.0f - 1.0f,
                  q(1), -sin(alpha) * q(0) + cos(alpha) * q(2)});

      auto dist_sphere = p.norm() - 0.5f;
      auto dist_walls = min(p(2) + 6.0f, p(1) + 1.0f);
      Vector d =
          Var(abs(p + Vector({-1.0f, 0.5f, -1.0f})) - Vector({0.3f, 1.2f,
      0.2f}));

      auto dist_cube = norm(d.map([](const Expr &v) { return max(v, 0.0f); })) +
                       min(max(max(d(0), d(1)), d(2)), 0.0f);

      return min(dist_sphere, min(dist_walls, dist_cube));
      */
      return q(1);
    };

    float32 eps = 1e-5f;
    float32 dist_limit = 1e2;
    int limit = 100;

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

    // normal near the surface
    auto sdf_normal = [&](Vector p) {
      float32 d = 1e-3f;
      Vector n(3);
      // finite difference
      for (int i = 0; i < 3; i++) {
        auto inc = Var(p), dec = Var(p);
        inc(i) += d;
        dec(i) -= d;
        n(i) = (0.5f / d) * (sdf(inc) - sdf(dec));
      }
      return normalized(n);
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

    auto dir_to_sky = [&]() {
      auto phi = light_phi + (Rand<float32>() - 0.5f) * light_smoothness;
      auto theta = light_theta + (Rand<float32>() - 0.5f) * light_smoothness;
      return Vector({cos(phi) * cos(theta), sin(theta), sin(phi) * cos(theta)});
    };

    // Direct sample light
    auto sample_light = [&](Vector p) {
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
        auto dir = Var(dir_to_sky());

        auto Le = Var(3.0f * Vector({1.0f, 1.0f, 1.0f}));
        auto near_t = Var(-std::numeric_limits<float>::max());
        auto far_t = Var(std::numeric_limits<float>::max());
        auto hit = box_intersect(p, dir, near_t, far_t);
        auto transmittance = Var(1.f);

        If(hit, [&] {
          auto cond = Var(hit);
          auto t = Var(max(near_t + 1e-4f, 0.0f));

          While(cond, [&] {
            t -= log(1.f - Rand<float32>()) * inv_max_density;
            auto q = Var(p + t * dir);
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

    auto get_next_hit = [&](const Vector &eye_o, const Vector &eye_d,
                            Expr &hit_distance, Vector &hit_pos,
                            Vector &normal) {
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
              hit_distance = max(max(mini(0), mini(1)), mini(2)) *
                             (1.0f / grid_resolution);
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

    // Woodcock tracking
    auto sample_volume_distance = [&](Vector o, Vector d, Expr &dist,
                                      Vector &sigma_s, Expr &transmittance,
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

    auto background = [&](Vector p, Vector dir) {
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
            auto albedo = Var(Vector({0.1f, 0.12f, 0.14f}));
            auto background = Var(Vector({0.1f, 0.12f, 0.14f}));
            ret = background + albedo.element_wise_prod(sample_light(p));
          });
      return ret;
    };

    float32 fov = param.get("fov", 0.6f);

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

    main = &kernel([&]() {
      kernel_name("main");
      Parallelize(param.get<int>("num_threads", 16));
      Vectorize(param.get<int>("vectorization", 1));
      BlockDim(32);
      For(0, output_res.prod(), [&](Expr i) {
        auto orig_input = param.get("orig", Vector3(0.5, 0.3, 1.5f));
        auto orig = Var(Vector({orig_input.x, orig_input.y, orig_input.z}));

        auto n = output_res.y;
        auto bid = Var(i / 32);
        auto tid = Var(i % 32);
        auto x = Var(bid / (n / 4) * 8 + tid / 4),
             y = Var(bid % (n / 4) * 4 + tid % 4);

        auto c = Var(Vector({fov * ((Rand<float32>() + cast<float32>(x)) /
                                        float32(output_res.y / 2) -
                                    (float32)output_res.x / output_res.y),
                             fov * ((Rand<float32>() + cast<float32>(y)) /
                                        float32(output_res.y / 2) -
                                    1.0f),
                             -1.0f}));

        c = normalized(c);

        auto color = Var(Vector({1.0f, 1.0f, 1.0f}));
        auto Li = Var(Vector({0.0f, 0.0f, 0.0f}));
        auto throughput = Var(Vector({1.0f, 1.0f, 1.0f}));
        auto depth = Var(0);

        If(depth_limit > 0)
            .Then([&] {
              While(depth < depth_limit, [&] {
                auto volume_dist = Var(0.f);
                auto transmittance = Var(0.f);
                auto sigma_s = Var(Vector({0.f, 0.f, 0.f}));
                auto interaction_p = Var(Vector({0.f, 0.f, 0.f}));
                auto volume_interaction =
                    sample_volume_distance(orig, c, volume_dist, sigma_s,
                                           transmittance, interaction_p);
                auto surface_dist = Var(ray_march(orig, c));

                depth += 1;
                If(volume_interaction)
                    .Then([&] {
                      throughput =
                          throughput.element_wise_prod(sigma_s * transmittance);

                      auto phase_value = eval_phase_isotropic();
                      auto light_value = sample_light(interaction_p);
                      Li += phase_value *
                            throughput.element_wise_prod(light_value);

                      orig = interaction_p;
                      c = sample_phase_isotropic();
                    })
                    .Else([&] {
                      if (use_sky_map) {
                        If(depth == 1).Then([&] {
                          auto p =
                              Var(orig - ((orig(1) - ground_y) / c(1) * c));
                          Li += throughput.element_wise_prod(background(p, c));
                        });
                      }
                      depth = depth_limit;
                    });
              });
            })
            .Else([&] {  // negative ones are for voxels
              While(depth < -depth_limit + 1, [&] {
                auto hit_dist = Var(0.0f);
                auto hit_pos = Var(Vector({1.0f, 1.0f, 1.0f}));
                auto normal = Var(Vector({1.0f, 1.0f, 1.0f}));
                get_next_hit(orig, c, hit_dist, hit_pos, normal);
                depth += 1;
                If(hit_dist > 0.0f)
                    .Then([&] {
                      c = normalized(out_dir(normal));
                      orig = hit_pos;
                      throughput = throughput.element_wise_prod(
                          Vector({0.6f, 0.5f, 0.5f}));

                      // direct lighting
                      auto light_dir = Var(dir_to_sky());
                      get_next_hit(orig, light_dir, hit_dist, hit_pos, normal);
                      If(hit_dist < 0.0f).Then([&] {
                        Li += throughput.element_wise_prod(
                            Vector({0.5f, 0.5f, 0.5f}));
                      });
                    })
                    .Else([&] {
                      // buffer[i] +=
                      //    throughput.element_wise_prod(background(p, c));
                      If(depth == 1).Then([&] {
                        Li += Vector({100.0f, 100.0f, 100.0f});
                      });
                      depth = -depth_limit + 1;
                    });
              });
            });

        buffer[x * output_res.y + y] += Li;
      });
    });

    std::FILE *f;
    if (use_sky_map) {
      f = fopen("sky_map.bin", "rb");
      TC_ASSERT_INFO(f, "./sky_map.bin not found");
      std::vector<uint32> sky_map_data(sky_map_size.prod() * 3);
      if (std::fread(sky_map_data.data(), sizeof(uint32), sky_map_data.size(),
                     f)) {
      }

      f = fopen("sky_samples.bin", "rb");
      TC_ASSERT_INFO(f, "./sky_samples.bin not found");
      std::vector<uint32> sky_sample_data(n_sky_samples * 5);
      if (std::fread(sky_sample_data.data(), sizeof(uint32),
                     (int)sky_sample_data.size(), f)) {
      }

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
              sky_sample_data[i * 5 + d] * (1.0f / (sky_map_size[d]));
        }
        for (int d = 0; d < 3; d++) {
          sky_sample_color(d).val<float32>(i) =
              sky_sample_data[i * 5 + 2 + d] * (1.0f / (1 << 20));
        }
      }
    }
    // expand blocks
    dilate = &kernel([&] {
      kernel_name("dilate");
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

    clear_buffer = &kernel([&] {
      kernel_name("clear_buffer");
      For(buffer(0), [&](Expr i) { buffer[i] = Vector({0.0f, 0.0f, 0.0f}); });
    });
  }

  void preprocess_volume() {
    (*dilate)();
  }

  void sample() {
    check_param_update();
    (*main)();
    acc_samples += 1;
  }
};

TLANG_NAMESPACE_END
