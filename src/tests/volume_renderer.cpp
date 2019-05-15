#include "volume_renderer.h"

TLANG_NAMESPACE_BEGIN

bool use_gui = false;

auto volume_renderer = [](std::vector<std::string> cli_param) {
  auto param = parse_param(cli_param);

  bool gpu = param.get("gpu", true);
  TC_P(gpu);
  CoreState::set_trigger_gdb_when_crash(true);
  Program prog(gpu ? Arch::gpu : Arch::x86_64);
  prog.config.print_ir = true;
  TRenderer renderer((Dict()));

  layout([&] { renderer.place_data(); });

  renderer.declare_kernels();

  std::unique_ptr<GUI> gui = nullptr;
  int n = renderer.output_res.y;
  int grid_resolution = renderer.grid_resolution;

  auto f = fopen("snow_density_256.bin", "rb");
  TC_ASSERT_INFO(f, "./snow_density_256.bin not found");
  std::vector<float32> density_field(pow<3>(grid_resolution));
  std::fread(density_field.data(), sizeof(float32), density_field.size(), f);
  std::fclose(f);

  float32 target_max_density = 724.0;
  auto max_density = 0.0f;
  for (int i = 0; i < pow<3>(grid_resolution); i++) {
    max_density = std::max(max_density, density_field[i]);
  }

  TC_P(max_density);

  for (int i = 0; i < pow<3>(grid_resolution); i++) {
    density_field[i] /= max_density;         // normalize to 1 first
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

  renderer.preprocess_volume();

  if (use_gui) {
    gui = std::make_unique<GUI>("Volume Renderer", Vector2i(n * 2, n));
  }
  Vector2i render_size(n * 2, n);
  Array2D<Vector4> render_buffer;

  auto tone_map = [](real x) { return std::sqrt(x); };

  constexpr int N = 10;
  for (int frame = 0; frame < 100; frame++) {
    for (int i = 0; i < N; i++) {
      renderer.sample();
    }
    prog.profiler_print();

    real scale = 1.0f / ((frame + 1) * N);
    render_buffer.initialize(render_size);
    std::unique_ptr<Canvas> canvas;
    canvas = std::make_unique<Canvas>(render_buffer);
    for (int i = 0; i < n * n * 2; i++) {
      render_buffer[i / n][i % n] =
          Vector4(tone_map(scale * renderer.buffer(0).val<float32>(i)),
                  tone_map(scale * renderer.buffer(1).val<float32>(i)),
                  tone_map(scale * renderer.buffer(2).val<float32>(i)), 1);
    }

    for (int i = 0; i < renderer.sky_map_size[0]; i++) {
      for (int j = 0; j < renderer.sky_map_size[1]; j++) {
        for (int d = 0; d < 3; d++) {
          // canvas->img[i][j][d] = sky_map(d).val<float32>(i, j) * 500;
        }
      }
    }

    if (use_gui) {
      gui->canvas->img = canvas->img;
      gui->update();
    } else {
      canvas->img.write_as_image(fmt::format("{:05d}-{:05d}-{:05d}.png", frame,
                                             N, renderer.depth_limit));
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
