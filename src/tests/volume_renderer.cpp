#include "volume_renderer.h"

TLANG_NAMESPACE_BEGIN
bool use_gui = false;
TLANG_NAMESPACE_END

#if defined(CUDA_FOUND)
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
  TRenderer renderer((Dict()));

  Vector particle_pos(DataType::f32, 3);

  layout([&] {
    renderer.place_data();
    auto l = Index(3);
    root.dynamic(l, 1024 * 1024 * 4).place(particle_pos);
  });

  renderer.declare_kernels();
  std::unique_ptr<GUI> gui = nullptr;

  Kernel(rasterize).def([&] {
    For(particle_pos(0), [&](Expr l) {
      auto inv_dx = Var(cast<float32>(256));
      auto x = Var(particle_pos[l]);
      auto downscale = Var(256 / renderer.grid_resolution);

      auto base_coord = Var(floor(inv_dx * x - 0.5_f));
      auto fx = x * inv_dx - base_coord;

      Vector w[] = {Var(0.5_f * sqr(1.5_f - fx)), Var(0.75_f - sqr(fx - 1.0_f)),
                    Var(0.5_f * sqr(fx - 0.5_f))};

      auto down_vec = Var(Vector({downscale, downscale, downscale}));

      // scatter
      for (int a = 0; a < 3; a++) {
        for (int b = 0; b < 3; b++) {
          for (int c = 0; c < 3; c++) {
            auto weight = w[a](0) * w[b](1) * w[c](2);
            auto offset = Vector({a, b, c});
            auto node = base_coord.cast_elements<int32>() + offset;
            Atomic(renderer.density[node(0) / downscale, node(1) / downscale,
                                    node(2) / downscale]) += weight;
          }
        }
      }
    });
  });

  int last_voxel_level = 0;
  int voxel_level = 1;

  auto load = [&](std::string fn) {
    static std::string last_fn;
    if (fn == last_fn && voxel_level == last_voxel_level)
      return;
    last_fn = fn;
    last_voxel_level = voxel_level;
    std::vector<Vector3> particles;
    auto f = fopen(fn.c_str(), "rb");
    TC_WARN_IF(!f, "{} not found", fn);

    if (!f)
      return;

    renderer.density.parent().snode()->clear_data_and_deactivate();
    renderer.density.parent().parent().snode()->clear_data_and_deactivate();
    if (fn.back() == 'n') {
      int grid_resolution = 256;
      std::vector<float32> density_field(pow<3>(grid_resolution));
      if (std::fread(density_field.data(), sizeof(float32),
                     density_field.size(), f)) {
      }
      std::fclose(f);
      float32 target_max_density = 1.0;
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
      fclose(f);
      read_from_binary_file(particles, fn);
    }

    if ((int)particles.size()) {
      particle_pos(0).parent().snode()->clear_data_and_deactivate();
      // renderer.check_param_update();
      for (int i = 0; i < (int)particles.size(); i++) {
        for (int d = 0; d < 3; d++) {
          particle_pos(d).val<float32>(i) = particles[i](d);
        }
      }

      int coarsening = 1;
      if (voxel_level == 1) {
        coarsening = 1;
      } else if (voxel_level == 2) {
        coarsening = 4;
      } else if (voxel_level == 3) {
        coarsening = 16;
      } else if (voxel_level == 4) {
        coarsening = 64;
      } else {
        TC_ASSERT(false);
      }
      renderer.parameters.grid_resolution = 256 / coarsening;
      renderer.check_param_update();
      TC_P(particles.size());
      rasterize();
    }
    for (int d = 0; d < 3; d++) {
      renderer.parameters.box_min[d] = 1e16f;
      renderer.parameters.box_max[d] = -1e16f;
    }

    for (int i = 0; i < (int)particles.size(); i++) {
      for (int d = 0; d < 3; d++) {
        renderer.parameters.box_min[d] = std::min(
            renderer.parameters.box_min[d], particle_pos(d).val<float32>(i));
        renderer.parameters.box_max[d] = std::max(
            renderer.parameters.box_max[d], particle_pos(d).val<float32>(i));
      }
    }
    for (int d = 0; d < 3; d++) {
      renderer.parameters.box_min[d] -= 5.0f / 256;
      renderer.parameters.box_max[d] += 5.0f / 256;
    }

    renderer.preprocess_volume();

    renderer.reset();
  };

  float32 exposure = 0.567;
  float32 exposure_linear = 1;
  float32 gamma = 0.5;
  float SPPS = 0;

  int fid = 0;

  Vector2i render_size(1280, 720);
  Array2D<Vector4> render_buffer;
  render_buffer.initialize(render_size);

  int frame = 0;
  int video_mode = 0;
  int output_samples = 10;
  int video_step = 1;
  if (use_gui) {
    gui = std::make_unique<GUI>("Volume Renderer",
                                Vector2i(render_size.x, render_size.y));
    gui->label("Sample/pixel/sec", SPPS);
    gui->slider("depth_limit", renderer.parameters.depth_limit, -10, 20);
    gui->slider("density_scale", renderer.parameters.density_scale, 1.0f,
                2000.0f);
    gui->slider("max_density", renderer.parameters.max_density, 1.0f, 2000.0f);
    gui->slider("ground_y", renderer.parameters.ground_y, 0.0f, 0.4f);
    gui->slider("light_phi", renderer.parameters.light_phi, 0.0f, pi * 2);
    gui->slider("light_theta", renderer.parameters.light_theta, 0.0f, pi / 2);
    gui->slider("light_smoothness", renderer.parameters.light_smoothness, 0.0f,
                0.7f);
    gui->slider("light_ambient", renderer.parameters.light_ambient, 0.0f, 1.0f);
    gui->slider("exposure", exposure, -3.0f, 3.0f);
    gui->slider("gamma", gamma, 0.2f, 2.0f);
    gui->slider("file_id", fid, 0, 500);
    gui->slider("output_samples", output_samples, 0, 100);
    gui->slider("grid_level", voxel_level, 1, 4);
    gui->slider("video_step", video_step, 1, 10);
    gui->button("Save", [&] {
      static int counter = 0;
      render_buffer.write_as_image(fmt::format("screenshot{}.png", counter++));
    });
    gui->button("Render All", [&] {
      create_directories("frames");
      frame = -video_step;
      video_mode = true;
    });
  }

  auto tone_map = [&](real x) { return std::pow(x * exposure_linear, gamma); };

  std::vector<float32> buffer(render_size.prod() * 5);

  auto get_fn = [&]() {
    return fn.find('{') == std::string::npos ? fn : fmt::format(fn, fid);
  };

  constexpr int N = 1;
  auto last_time = Time::get_time();
  for (frame = 0;; frame++) {
    auto ft = Time::get_time();
    if (video_mode)
      fid = frame;
    load(get_fn());
    for (int i = 0; i < (video_mode ? output_samples : N); i++) {
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
    tbb::parallel_for(0, render_size.prod(), [&](int i) {
      render_buffer[i / render_size.y][i % render_size.y] =
          Vector4(tone_map(scale * buffer[i * 3 + 0]),
                  tone_map(scale * buffer[i * 3 + 1]),
                  tone_map(scale * buffer[i * 3 + 2]), 1.0f);
    });

    if (video_mode) {
      render_buffer.write_as_image(fmt::format("frames/{:04d}.png", frame));
    }

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
    if (video_mode)
      frame += video_step - 1;
    gui->canvas->img.write_as_image(fmt::format("gui/{:05d}.png", frame));
    Time::sleep(0.03);
    TC_P(Time::get_time() - ft);
  }
};
TC_REGISTER_TASK(volume_renderer);

auto volume_renderer_gui = [](std::vector<std::string> cli_param) {
  use_gui = true;
  volume_renderer(cli_param);
};

TC_REGISTER_TASK(volume_renderer_gui);

TLANG_NAMESPACE_END
#endif
