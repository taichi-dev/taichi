#include <taichi/lang.h>
#include <taichi/util.h>
#include <taichi/visual/gui.h>
#include <taichi/system/profiler.h>
#include <taichi/visualization/particle_visualization.h>

TC_NAMESPACE_BEGIN

using namespace Tlang;

constexpr int n = 256;

constexpr int num_ch1 = 16, num_ch2 = 16;

auto cnn = [](std::vector<std::string> cli_param) {
  CoreState::set_trigger_gdb_when_crash(true);
  auto param = parse_param(cli_param);
  auto gpu = param.get("gpu", true);
  auto opt = param.get("opt", true);
  auto use_dense = param.get("use_dense", false);
  auto write_input_voxel = param.get("write_input", true);
  TC_P(use_dense);

  Program prog(gpu ? Arch::gpu : Arch::x86_64);
  prog.config.simplify_before_lower_access = opt;
  prog.config.lower_access = opt;
  prog.config.simplify_after_lower_access = opt;
  // prog.config.print_ir = true;

  // constexpr int dim = 3;

  int block_size = gpu ? 4 : 8;

  Global(layer1, f32);
  Global(layer2, f32);
  Global(weights, f32);

  layout([&]() {
    auto ijkl = Indices(0, 1, 2, 3);
    if (use_dense) {
      root.dense(ijkl, {n, n, n, num_ch1}).place(layer1);
      root.dense(ijkl, {n, n, n, num_ch2}).place(layer2);
    } else {
      root.dense(ijkl, {n / block_size, n / block_size, n / block_size, 1})
          .bitmasked()
          .dense(ijkl, {block_size, block_size, block_size, num_ch1})
          .place(layer1);
      root.dense(ijkl, {n / block_size, n / block_size, n / block_size, 1})
          .bitmasked()
          .dense(ijkl, {block_size, block_size, block_size, num_ch2})
          .place(layer2);
    }
    root.dense(ijkl, {4, 4, 4, num_ch1 * num_ch2}).place(weights);
  });

  auto tex = create_instance<Texture>(
      "mesh", Dict()
                  .set("resolution", Vector3(n))
                  .set("translate", Vector3(0.55, 0.35, 0.47))
                  .set("scale", Vector3(0.5))
                  .set("adaptive", false)
                  .set("filename", "$mpm/bunny_small.obj"));
  float *in_data = new float[num_ch1 * n * n * n];
  memset(in_data, 0, sizeof(float) * num_ch1 * n * n * n);
  int count = 0;
  for (int i = 1; i < n - 2; i++) {
    for (int j = 1; j < n - 2; j++) {
      for (int k = 1; k < n - 2; k++) {
        bool inside = tex->sample((Vector3(0.5f) + Vector3(i, j, k)) *
                                  Vector3(1.0f / (n - 1)))
                          .x > 0.5f;
        // inside = pow<2>(i - n / 2) + pow<2>(k - n / 2) < pow<2>(n / 2) / 2;
        // inside = i < n * 0.8 && j < n * 0.8 && k < n * 0.8;
        if (inside) {
          for (int c = 0; c < num_ch1; c++) {
            in_data[c * n * n * n + k * n * n + j * n + i] = 1.f;
            layer1.val<float32>(i, j, k, c) = 1.f;
            count++;
          }
        }
      }
    }
  }
  std::cout << "non_zero:" << count << ", total:" << (num_ch1 * n * n * n)
            << std::endl;
  if (write_input_voxel) {
    auto f = fopen("bunny_sparse.bin", "wb");
    fwrite(in_data, sizeof(float), num_ch1 * n * n * n, f);
    fclose(f);
  }

  Kernel(forward).def([&] {
    // Cache(0, layer1);
    bool use_cache = false;
    if (opt && gpu) {
      use_cache = true;
      Cache(1, weights);
    }
    if (!gpu) {
      Parallelize(8);
      Vectorize(block_size);
    } else {
      BlockDim(256);
    }
    For(layer2, [&](Expr i, Expr j, Expr k, Expr c_out) {
      auto sum = Var(0.0f);
      for (int c_in = 0; c_in < num_ch1; c_in++) {
        for (int dx = -1; dx < 2; dx++) {
          for (int dy = -1; dy < 2; dy++) {
            for (int dz = -1; dz < 2; dz++) {
              auto weight = weights[Expr(dx + 1), Expr(dy + 1), Expr(dz + 1),
                                    c_in * num_ch2 + c_out];
              auto c_in2 = use_cache ? AssumeInRange(c_in, c_out, 0, 1) : c_in;
              sum += weight * layer1[i + dx, j + dy, k + dz, c_in2];
              // layer2[i, j, k, c_out] += weight * layer1[i + dx, j + dy, k +
              // dz, c_in];
            }
          }
        }
      }
      layer2[i, j, k, c_out] = sum;
    });
  });

  // expand blocks
  kernel([&] {
    if (!gpu) {
      Parallelize(8);
      // Vectorize(block_size);
    } else {
      BlockDim(256);
    }

    kernel_name("dilate");
    For(layer1, [&](Expr i, Expr j, Expr k) {
      If(i % block_size == 0 && j % block_size == 0 && k % block_size == 0)
          .Then([&] {
            for (int x = -1; x < 2; x++) {
              for (int y = -1; y < 2; y++) {
                for (int z = -1; z < 2; z++) {
                  layer2[i + x * block_size, j + y * block_size,
                         k + z * block_size, 0] =
                      0.0f;  // simply activate the block
                }
              }
            }
          });
    });
  })();

  for (int c_out = 0; c_out < num_ch2; c_out++) {
    for (int c_in = 0; c_in < num_ch1; c_in++) {
      float inc = 0.1f;
      for (int dx = -1; dx < 2; dx++) {
        for (int dy = -1; dy < 2; dy++) {
          for (int dz = -1; dz < 2; dz++) {
            if (dx == 0 && dy == 0 && dz == 0)
              weights.val<float32>(dx + 1, dy + 1, dz + 1,
                                   c_in * num_ch2 + c_out) = inc;
            inc += 0.1f;
          }
        }
      }
    }
  }

  // prog.config.print_ir = true;
  for (int i = 0; i < 50; i++) {
    forward();
  }
  prog.profiler_print();

  // Write the first layer of output
  float *data = new float[(n - 2) * (n - 2) * (n - 2)];
  int non_zero = 0;
  int zero = 0;
  for (int i = 1; i < (n - 1); i++) {
    for (int j = 1; j < (n - 1); j++) {
      for (int k = 1; k < (n - 1); k++) {
        data[((0 * (n - 2) + (k - 1)) * (n - 2) + (j - 1)) * (n - 2) +
             (i - 1)] = layer2.val<float32>(i, j, k, 0);
        if (layer2.val<float32>(i, j, k, 0) != 0) {
          non_zero++;
        } else {
          zero++;
        }
      }
    }
  }
  std::cout << "Non zero:" << non_zero << ", zero:" << zero << std::endl;
  std::cerr << "Sparsity:" << (double)non_zero / (double)(non_zero + zero)
            << std::endl;
  auto f_out = fopen("our_bunny.bin", "wb");
  fwrite(data, sizeof(float), (n - 2) * (n - 2) * (n - 2), f_out);
  fclose(f_out);

#if 0
  int gui_res = 512;
  GUI gui("Sparse CNN", Vector2i(gui_res + 200, gui_res), false);
  int layer = 1;
  int k = 0;
  int channel = 0;
  gui.slider("z", k, 0, n - 1)
      .slider("Layer", layer, 1, 2)
      .slider("Channel", channel, 0, num_ch1);

  int scale = gui_res / n;
  auto &canvas = gui.get_canvas();
  for (int frame = 1;; frame++) {
    for (int i = 0; i < gui_res - scale; i++) {
      for (int j = 0; j < gui_res - scale; j++) {
        real dx;
        if (layer == 1) {
          dx = layer1.val<float32>(i / scale, j / scale, k, channel);
        } else {
          dx = layer2.val<float32>(i / scale, j / scale, k, channel);
        }
        canvas.img[i][j] = Vector4(0.5f) + Vector4(dx) * 0.5f;
      }
    }
    gui.update();
  }
#endif
};
TC_REGISTER_TASK(cnn);

TC_NAMESPACE_END
