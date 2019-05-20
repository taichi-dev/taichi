#include "volume_renderer.h"
#include <tbb/tbb.h>
#include <cuda_runtime_api.h>

TLANG_NAMESPACE_BEGIN

bool use_gui = false;

auto volume_renderer = [](std::vector<std::string> cli_param) {
  auto param = parse_param(cli_param);

  bool gpu = param.get("gpu", true);
  TC_P(gpu);
  std::string fn = param.get("fn", "snow_density_256.bin");
  TC_P(fn);
  CoreState::set_trigger_gdb_when_crash(true);
  Program prog(gpu ? Arch::gpu : Arch::x86_64);
  prog.config.print_ir = true;
  TRenderer renderer((Dict()));

  Vector particle_pos(DataType::f32, 3);

  layout([&] {
    renderer.place_data();
    auto l = Index(3);
    root.dynamic(l, 1024 * 1024 * 4).place(particle_pos);
  });

  renderer.declare_kernels();

  Kernel(rasterize).def([&] {
    const int n = 256;
    auto dx = 1.0_f / n, inv_dx = 1.0_f / dx;
    For(particle_pos(0), [&](Expr l) {
      auto x = particle_pos[l];

      auto base_coord = Var(floor(inv_dx * x - 0.5_f));
      auto fx = x * inv_dx - base_coord;

      Vector w[] = {Var(0.5_f * sqr(1.5_f - fx)), Var(0.75_f - sqr(fx - 1.0_f)),
                    Var(0.5_f * sqr(fx - 0.5_f))};

      // scatter
      for (int a = 0; a < 3; a++) {
        for (int b = 0; b < 3; b++) {
          for (int c = 0; c < 3; c++) {
            auto weight = w[a](0) * w[b](1) * w[c](2);
            auto offset = Vector({a, b, c});
            auto node = base_coord.cast_elements<int32>() + offset;
            Atomic(renderer.density[node]) += weight;
          }
        }
      }
    });
  });

  std::unique_ptr<GUI> gui = nullptr;
  int n = renderer.output_res.y;
  int grid_resolution = renderer.grid_resolution;

  std::vector<Vector3> particles;
  if (fn.back() == 'n') {
    std::vector<float32> density_field(pow<3>(grid_resolution));
    auto f = fopen(fn.c_str(), "rb");
    TC_ERROR_IF(!f, "{} not found", fn);
    if (std::fread(density_field.data(), sizeof(float32), density_field.size(),
                   f)) {
    }
    std::fclose(f);
    float32 target_max_density = 724.0;
    auto max_density = 0.0f;
    for (int i = 0; i < pow<3>(grid_resolution); i++) {
      max_density = std::max(max_density, density_field[i]);
    }

    TC_P(max_density);

    for (int i = 0; i < pow<3>(grid_resolution); i++) {
      density_field[i] /= max_density;             // normalize to 1 first
      density_field[i] *= target_max_density * 1;  // then scale
      density_field[i] = std::min(density_field[i], target_max_density);
    }

    for (int i = 0; i < grid_resolution; i++) {
      for (int j = 0; j < grid_resolution; j++) {
        for (int k = 0; k < grid_resolution; k++) {
          auto d = density_field[i * grid_resolution * grid_resolution +
                                 j * grid_resolution + k];
          if (d != 0) {  // populate non-empty voxels only
            renderer.density.val<float32>(i, j, k) = d;
          }
        }
      }
    }
  } else {
    // load particles and rasterize
    read_from_binary_file(particles, fn);
  }

  if ((int)particles.size()) {
    for (int i = 0; i < particles.size(); i++) {
      for (int d = 0; d < 3; d++) {
        particle_pos(d).val<float32>(i) = particles[i](d);
      }
    }
    rasterize();
  }

  renderer.preprocess_volume();

  float32 exposure = 0.567;
  float32 exposure_linear = 1;
  float32 gamma = 0.5;
  float SPPS = 0;
  if (use_gui) {
    gui = std::make_unique<GUI>("Volume Renderer", Vector2i(n * 2, n));
    gui->label("Sample/pixel/sec", SPPS);
    gui->slider("depth_limit", renderer.parameters.depth_limit, -10, 20);
    gui->slider("density_scale", renderer.parameters.density_scale, 1.0f, 2000.0f);
    gui->slider("max_density", renderer.parameters.max_density, 1.0f, 2000.0f);
    gui->slider("ground_y", renderer.parameters.ground_y, 0.0f, 0.4f);
    gui->slider("light_phi", renderer.parameters.light_phi, 0.0f, pi * 2);
    gui->slider("light_theta", renderer.parameters.light_theta, 0.0f, pi / 2);
    gui->slider("light_smoothness", renderer.parameters.light_smoothness, 0.0f,
                0.7f);
    gui->slider("exposure", exposure, -3.0f, 3.0f);
    gui->slider("gamma", gamma, 0.2f, 2.0f);
  }
  Vector2i render_size(n * 2, n);
  Array2D<Vector4> render_buffer;
  render_buffer.initialize(render_size);

  auto tone_map = [&](real x) { return std::pow(x * exposure_linear, gamma); };

  std::vector<float32> buffer(render_size.prod() * 3);

  constexpr int N = 1;
  auto last_time = Time::get_time();
  for (int frame = 0; frame < 1000000; frame++) {
    for (int i = 0; i < N; i++) {
      renderer.sample();
    }
    if (frame % 10 == 0) {
      auto elapsed = Time::get_time() - last_time;
      last_time = Time::get_time();
      SPPS = 1.0f / (elapsed / 10.0f);
      prog.profiler_print();
      prog.profiler_clear();
    }

    real scale = 1.0f / renderer.acc_samples;
    exposure_linear = std::exp(exposure);

    cudaMemcpy(buffer.data(), &renderer.buffer(0).val<float32>(0),
               buffer.size() * sizeof(float32), cudaMemcpyDeviceToHost);
    tbb::parallel_for(0, n * n * 2, [&](int i) {
      render_buffer[i / n][i % n] =
          Vector4(tone_map(scale * buffer[i * 3 + 0]),
                  tone_map(scale * buffer[i * 3 + 1]),
                  tone_map(scale * buffer[i * 3 + 2]), 1.0f);
    });

    if (use_gui) {
      gui->canvas->img = render_buffer;
      gui->update();
    } else {
      std::unique_ptr<Canvas> canvas;
      canvas = std::make_unique<Canvas>(render_buffer);
      canvas->img.write_as_image(fmt::format("{:05d}-{:05d}-{:05d}.png", frame,
                                             N,
                                             renderer.parameters.depth_limit));
    }
  }
};
TC_REGISTER_TASK(volume_renderer);

auto volume_renderer_gui = [](std::vector<std::string> cli_param) {
  use_gui = true;
  volume_renderer(cli_param);
};

TC_REGISTER_TASK(volume_renderer_gui);

TLANG_NAMESPACE_END
