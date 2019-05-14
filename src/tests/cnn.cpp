#include "../tlang.h"
#include <taichi/util.h>
#include <taichi/visual/gui.h>
#include <taichi/system/profiler.h>
#include <taichi/visualization/particle_visualization.h>

TC_NAMESPACE_BEGIN

using namespace Tlang;

auto cnn = []() {
  CoreState::set_trigger_gdb_when_crash(true);

  Program prog(Arch::gpu);
  prog.config.lower_access = false;

  constexpr int dim = 3;
  constexpr int n = 128;

  constexpr int num_ch1 = 16, num_ch2 = 16;

  Global(layer1, f32);
  Global(layer2, f32);
  Global(weights, f32);

  layout([&]() {
    auto ijkl = Indices(0, 1, 2, 3);
    root.dense(ijkl, {n, n, n, num_ch1}).place(layer1);
    root.dense(ijkl, {n, n, n, num_ch2}).place(layer2);
    root.dense(ijkl, {4, 4, 4, num_ch1 * num_ch2}).place(weights);
  });

  for (int c_in = 0; c_in < num_ch1; c_in++) {
    for (int i = 0; i < n; i++) {
      for (int j = 0; j < n; j++) {
        for (int k = 0; k < n; k++) {
          auto len =
              (Vector3i(i, j, k) - Vector3i(n / 2)).length() / (float32) n;
          if (0.4 < len && len < 0.5) {
            layer1.val<float32>(i, j, k, c_in) = len;
          }
        }
      }
    }
  }

  Kernel(forward).def([&] {
    BlockDim(128);
    For(layer2, [&](Expr i, Expr j, Expr k, Expr c_out) {
      for (int c_in = 0; c_in < num_ch1; c_in++) {
        for (int dx = -1; dx < 2; dx++) {
          for (int dy = -1; dy < 2; dy++) {
            for (int dz = -1; dz < 2; dz++) {
              auto weight =
                  weights[Expr(dx + 1), Expr(dy + 1), Expr(dz + 1), c_in * num_ch2 + c_out];
              layer2[i, j, k, c_out] += weight * layer1[i + dx, j + dy, k + dz, c_in];
            }
          }
        }
      }
    });
  });

  for (int c_out = 0; c_out < num_ch1; c_out++) {
    for (int c_in = 0; c_in < num_ch1; c_in++) {
      for (int dx = -1; dx < 2; dx++) {
        for (int dy = -1; dy < 2; dy++) {
          for (int dz = -1; dz < 2; dz++) {
            weights.val<float32>(dx + 1, dy + 1, dz + 1,
                                 c_in * num_ch2 + c_out) = rand() - 0.5f;
          }
        }
      }
    }
  }

  forward();

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
};
TC_REGISTER_TASK(cnn);

TC_NAMESPACE_END
