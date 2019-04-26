#include "../tlang.h"
#include <taichi/testing.h>
#include <numeric>

TLANG_NAMESPACE_BEGIN

TC_TEST("range_assumption") {
  CoreState::set_trigger_gdb_when_crash(true);
  Program prog(Arch::gpu);
  prog.config.print_ir = true;

  Global(x, f32);
  Global(y, f32);

  int domain_size = 256;
  int block_size = 8;

  auto i = Indices(0);
  layout([&]() {
    root.dense(i, domain_size / block_size)
        // .pointer()
        .dense(i, block_size)
        .place(x, y);
  });

  // Initialize
  kernel([&]() {
    Declare(i);
    BlockDim(1);
    For(i, 0, domain_size, [&] {
      auto j = AssumeInRange(i + cast<int32>(i % 3 - 1), i, -1, 2);
      Print(j);
    });
  })();
};

TC_TEST("scratch_pad_3d") {
  CoreState::set_trigger_gdb_when_crash(true);
  int n = 10000000;
  Program prog(Arch::gpu);
  prog.config.print_ir = true;

  Global(x, f32);
  Global(y, f32);

  int domain_size = 256;
  int block_size = 8;

  auto ijk = Indices(0, 1, 2);
  layout([&]() {
    root.dense(ijk, domain_size / block_size)
        // .pointer()
        .dense(ijk, block_size)
        .place(x, y);
  });

  auto dist_imm = [&](int i, int j, int k) {
    auto dx = (1.0_f / domain_size) * (i + 0.5f) - 0.5f;
    auto dy = (1.0_f / domain_size) * (j + 0.5f) - 0.5f;
    auto dz = (1.0_f / domain_size) * (k + 0.5f) - 0.5f;
    return std::sqrt(dx * dx + dy * dy + dz * dz);
  };

  auto x_val = [&](int i, int j, int k) {
    auto d = dist_imm(i, j, k);
    if (0.43f < d && d < 0.47f) {
      return d * d * d;
    } else {
      return 0.0f;
    }
  };

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
          auto d = Var(dist(i, j, k));
          If(0.43f < d && d < 0.47f, [&] {
            Activate(x, (i, j, k));
            x[i, j, k] = d * d * d;
          });
        });
      });
    });
  })();

  auto &laplacian = kernel([&]() {
    Declare(i);
    Declare(j);
    Declare(k);
    Cache(0, x);
    Cache(0, y);
    For((i, j, k), x, [&]() {
      y[i, j, k] = 6.0f * x[i, j, k] - x[i, j, k - 1] - x[i, j, k + 1] -
                   x[i, j - 1, k] - x[i, j + 1, k] - x[i - 1, j, k] -
                   x[i + 1, j, k];
    });
  });

  laplacian();

  for (int i = 0; i < domain_size; i++) {
    for (int j = 0; j < domain_size; j++) {
      for (int k = 0; k < domain_size; k++) {
        auto d = dist_imm(i, j, k);
        if (0.44f < d && d < 0.46f) {
          auto gt = 6.0f * x_val(i, j, k) - x_val(i, j, k - 1) -
                    x_val(i, j, k + 1) - x_val(i, j - 1, k) -
                    x_val(i, j + 1, k) - x_val(i - 1, j, k) -
                    x_val(i + 1, j, k);
          if (std::abs(gt - y.val<float32>(i, j, k)) > 1) {
            TC_P(d);
            TC_P(gt);
            TC_P(y.val<float32>(i, j, k));
            TC_P(i);
            TC_P(j);
            TC_P(k);
          }
          TC_CHECK_EQUAL(gt, y.val<float32>(i, j, k),
                         1e-1f / domain_size / domain_size);
        }
      }
    }
  }
};

TLANG_NAMESPACE_END
