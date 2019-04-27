#include "../tlang.h"
#include <taichi/util.h>
#include <taichi/visual/gui.h>
#include <taichi/common/bit.h>
#include <Partio.h>
#include <taichi/system/profiler.h>
#include <taichi/visualization/particle_visualization.h>
#include "svd.h"

TC_NAMESPACE_BEGIN

using namespace Tlang;

auto cnn = []() {
  CoreState::set_trigger_gdb_when_crash(true);

  Program prog(Arch::gpu);

  constexpr int dim = 3;
  constexpr int n = 256;

  Global(layer1, f32);
  Global(layer2, f32);
  Global(weights, f32);

  layout([&]() {
    auto ijkl = Indices(0, 1, 2, 3);
    root.dense(ijkl, {n, n, n, 4}).place(layer1);
    root.dense(ijkl, {3, 3, 3, 16}).place(weights);
  });

  Kernel(forward).def([&] {
    BlockDim(1024);
    For(layer2, [&](Expr i, Expr j, Expr k, Expr c_out) {
      // layer1[]
    });
  });
};
TC_REGISTER_TASK(cnn);

TC_NAMESPACE_END
