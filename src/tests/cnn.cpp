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

  constexpr int dim = 3;
  constexpr int n = 256;

  constexpr int num_ch1 = 4, num_ch2 = 4;

  Global(layer1, f32);
  Global(layer2, f32);
  Global(weights, f32);

  layout([&]() {
    auto ijkl = Indices(0, 1, 2, 3);
    root.dense(ijkl, {n, n, n, 4}).place(layer1);
    root.dense(ijkl, {n, n, n, 4}).place(layer2);
    root.dense(ijkl, {4, 4, 4, 16}).place(weights);
  });

  Kernel(forward).def([&] {
    BlockDim(1024);
    For(layer2, [&](Expr i, Expr j, Expr k, Expr c_out) {
      // layer1[]
    });
  });

  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      for (int k = 0; k < n; k++) {
        auto len = (Vector3i(i, j, k) - Vector3i(n / 2)).length() / (float32)n;
        if (0.4 < len && len < 0.5) {
          layer1.val<float32>(i, j, k, 0) = len;
        }
      }
    }
  }

  int gui_res = 512;
  GUI gui("FEM", Vector2i(gui_res + 200, gui_res), false);
  int layer = 1;
  int k = 0;
  int channel = 0;
  gui.slider("z", k, 0, n - 1).slider("Layer", layer, 1, 2);

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
