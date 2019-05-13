#include "volume_renderer.h"

TLANG_NAMESPACE_BEGIN

extern bool use_gui;

auto smoke_renderer = [] {
  bool benchmark = true; // benchmark the bunny cloud against tungsten?
  TC_ASSERT(benchmark);
  // CoreState::set_trigger_gdb_when_crash(true);
  Program prog(Arch::gpu);
  // prog.config.print_ir = true;
  TRenderer renderer(1024);
  TC_TAG;

  layout([&]{
    renderer.place_data();
  });
  TC_TAG;

  renderer.declare_kernels();
  TC_TAG;

  std::unique_ptr<GUI> gui = nullptr;
  int n = renderer.n;
  int grid_resolution = renderer.grid_resolution;


  if (benchmark) {
    auto f = fopen("bunny_cloud.bin", "rb");
    TC_ASSERT_INFO(f, "./bunny_cloud.bin not found");
    int box_sizes[3] {584, 576, 440};
    TC_TAG;
    int total_voxels = box_sizes[0] * box_sizes[1] * box_sizes[2];
    TC_TAG;
    std::vector<float32> density_field(total_voxels);
    std::fread(density_field.data(), sizeof(float32), density_field.size(), f);
    std::fclose(f);

    TC_TAG;

    float32 target_max_density = 724.0;
    auto max_density = 0.0f;
    for (int i = 0; i < total_voxels; i++) {
      max_density = std::max(max_density, density_field[i]);
    }
    TC_TAG;

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
TC_REGISTER_TASK(smoke_renderer);

auto smoke_renderer_gui = [] {
  use_gui = true;
  smoke_renderer();
};

TC_REGISTER_TASK(smoke_renderer_gui);

TLANG_NAMESPACE_END
