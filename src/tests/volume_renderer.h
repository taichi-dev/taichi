#pragma once
#include <taichi/visual/gui.h>
#include "../tlang.h"

TLANG_NAMESPACE_BEGIN

class TRenderer {
 public:
  Vector2i output_res;
  Vector2i sky_map_size;
  int n_sky_samples;
  Program::Kernel *main, *dilate;
  Vector buffer;
  int depth_limit;
  Expr density;
  int grid_resolution;
  bool use_sky_map = false;
  int block_size = 4;
  float32 one_over_four_pi = 0.07957747154f;
  float32 pi = 3.14159265359f;
  Vector sky_map;
  Vector sky_sample_color;
  Vector sky_sample_uv;
  Dict param;

  TRenderer(Dict param) : param(param) {
    grid_resolution = param.get("grid_resolution", 256);
    depth_limit = param.get("depth_limit", 1);
    output_res = param.get("output_res", Vector2i(1024, 512));

    TC_ASSERT(bit::is_power_of_two(output_res.x));
    TC_ASSERT(bit::is_power_of_two(output_res.y));

    sky_map_size = Vector2i(512, 128);
    n_sky_samples = 1024;

    density.declare(DataType::f32);
    buffer.declare(DataType::f32, 3);
    sky_map.declare(DataType::f32, 3);
    sky_sample_color.declare(DataType::f32, 3);
    sky_sample_uv.declare(DataType::f32, 2);
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
  }

  void declare_kernels() {
    auto inv_max_density = 1.0f / 724.0f;
    Vector3 albedo(0.9, 0.95, 1);
    auto const block_size = 4;
    auto lower_bound = -0.0f;
    auto upper_bound = 1.0f;

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

    auto sample_distance = [&](Vector o, Vector d, float32 inv_max_density,
                               Expr &dist, Vector &sigma_s, Expr &transmittance,
                               Vector &p) {
      auto near_t = Var(-std::numeric_limits<float>::max());
      auto far_t = Var(std::numeric_limits<float>::max());
      auto hit = box_intersect(o, d, near_t, far_t);

      return hit;
    };

    float32 fov = param.get("fov", 0.6f);

    main = &kernel([&]() {
      kernel_name("main");
      Parallelize(16);
      Vectorize(param.get<int>("vectorization"));
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

        auto dist = Var(0.f);
        auto transmittance = Var(0.f);
        auto sigma_s = Var(Vector({0.f, 0.f, 0.f}));
        auto interaction_p = Var(Vector({0.f, 0.f, 0.f}));
        auto interaction =
            sample_distance(orig, c, inv_max_density, dist, sigma_s,
                            transmittance, interaction_p);

        If(interaction)
            .Then([&] {
              Li += Vector({1.0f, 1.0f, 1.0f});
            });

        buffer[x * output_res.y + y] += Li;
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
  }

  void preprocess_volume() {
    (*dilate)();
  }

  void sample() {
    (*main)();
  }
};

TLANG_NAMESPACE_END
