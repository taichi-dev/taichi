#include "volume_renderer.h"

TLANG_NAMESPACE_BEGIN

extern bool use_gui;

auto smoke_renderer = [](std::vector<std::string> cli_param_) {
  auto cli_param = parse_param(cli_param_);
  bool gpu = cli_param.get("gpu", true);
  TC_P(gpu);
  int vectorization = cli_param.get("vectorization", 8);
  TC_P(vectorization);
  int num_threads = cli_param.get("num_threads", 8);
  TC_P(num_threads);
  Program prog(gpu ? Arch::gpu : Arch::x86_64);
  prog.config.simplify_before_lower_access = true;
  prog.config.lower_access = true;
  prog.config.simplify_after_lower_access = true;
  prog.config.print_ir = true;
  bool benchmark = true;  // benchmark the bunny cloud against tungsten?
  TC_ASSERT(benchmark);
  // CoreState::set_trigger_gdb_when_crash(true);
  // prog.config.print_ir = true;
  Dict param;
  param.set("grid_resolution", 1024);
  param.set("output_res", Vector2i(512, 512));
  param.set("orig", Vector3(0.25, 0.25, 0.7));
  param.set("fov", 1);
  param.set("vectorization", vectorization);
  param.set("num_threads", num_threads);
  TRenderer renderer(param);

  layout([&] { renderer.place_data(); });

  renderer.declare_kernels();

  std::unique_ptr<GUI> gui = nullptr;

  if (benchmark) {
    auto f = fopen("bunny_cloud.bin", "rb");
    TC_ASSERT_INFO(f, "./bunny_cloud.bin not found");
    int box_sizes[3]{584, 576, 440};
    int total_voxels = box_sizes[0] * box_sizes[1] * box_sizes[2];
    std::vector<float32> density_field(total_voxels);
    if (std::fread(density_field.data(), sizeof(float32), density_field.size(), f)) {}
    std::fclose(f);

    float32 target_max_density = 724.0;
    auto max_density = 0.0f;
    for (int i = 0; i < total_voxels; i++) {
      max_density = std::max(max_density, density_field[i]);
    }

    TC_P(max_density);

    for (int i = 0; i < total_voxels; i++) {
      density_field[i] /= max_density;         // normalize to 1 first
      density_field[i] *= target_max_density;  // then scale
    }

    for (int i = 0; i < box_sizes[0]; i++) {
      for (int j = 0; j < box_sizes[1]; j++) {
        for (int k = 0; k < box_sizes[2]; k++) {
          auto d = density_field[i * box_sizes[2] * box_sizes[1] +
                                 j * box_sizes[2] + k];
          if (d != 0) {  // populate non-empty voxels only
            renderer.density.val<float32>(i, j, k) = d;
          }
        }
      }
    }
  }

  renderer.preprocess_volume();

  if (use_gui) {
    gui = std::make_unique<GUI>("Volume Renderer", renderer.output_res);
  }
  Array2D<Vector4> render_buffer;

  auto tone_map = [](real x) { return std::sqrt(x); };

  int N = gpu ? 10 : 1;
  for (int frame = 0; frame < 100; frame++) {
    for (int i = 0; i < N; i++) {
      renderer.sample();
    }
    prog.profiler_print();

    real scale = 1.0f / ((frame + 1) * N);
    render_buffer.initialize(renderer.output_res);
    std::unique_ptr<Canvas> canvas;
    canvas = std::make_unique<Canvas>(render_buffer);
    for (int i = 0; i < renderer.output_res.prod(); i++) {
      render_buffer[i / renderer.output_res.y][i % renderer.output_res.y] =
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
TC_REGISTER_TASK(smoke_renderer);

auto smoke_renderer_gui = [](std::vector<std::string> cli_param) {
  use_gui = true;
  smoke_renderer(cli_param);
};

TC_REGISTER_TASK(smoke_renderer_gui);

TLANG_NAMESPACE_END
