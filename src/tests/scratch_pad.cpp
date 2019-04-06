#include "../tlang.h"
#include <taichi/testing.h>
#include <numeric>

TLANG_NAMESPACE_BEGIN

TC_TEST("scratch_pad_3d") {
  CoreState::set_trigger_gdb_when_crash(true);
  int n = 10000000;
  Program prog(Arch::x86_64);
  prog.config.print_ir = true;

  Global(x, f32);
  Global(y, f32);

  int domain_size = 1024;
  int block_size = 8;

  auto ijk = Indices(0, 1, 2);
  layout([&]() {
    root.dense(ijk, domain_size / block_size)
        .pointer()
        .dense(ijk, block_size)
        .place(x, y);
  });

  auto dist = [&](Expr i, Expr j, Expr k) {
    auto dx = (1.0_f / domain_size) * (cast<float32>(i) + 0.5f) - 0.5f;
    auto dy = (1.0_f / domain_size) * (cast<float32>(j) + 0.5f) - 0.5f;
    auto dz = (1.0_f / domain_size) * (cast<float32>(k) + 0.5f) - 0.5f;
    return sqrt(dx * dx + dy * dy + dz * dz);
  };

  // Initialize
  kernel([&]() {
    Declare(i);
    Declare(j);
    Declare(k);
    BlockDim(1);
    For(i, 0, domain_size, [&] {
      For(j, 0, domain_size, [&] {
        For(k, 0, domain_size, [&] {
          auto d = Eval(dist(i, j, k));
          If(0.43f < d && d < 0.47f, [&] {
            Activate(x, (i, j, k));
            x[i, j, k] = 1.0f / d;
          });
        });
      });
    });
  })();

  auto laplacian = kernel([&]() {
    Declare(i);
    Declare(j);
    Declare(k);
    For((i, j, k), x, [&]() {
      y[i, j, k] = 6.0f * x[i, j, k] - x[i, j, k - 1] - x[i, j, k + 1] -
                   x[i, j - 1, k] - x[i, j + 1, k] - x[i - 1, j, k] -
                   x[i + 1, j, k];
    });
  });

  laplacian();

  // TC_CHECK(sum.val<int>() == n);
};

TLANG_NAMESPACE_END
