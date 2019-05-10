#include "../tlang.h"
#include <taichi/util.h>
#include <taichi/visual/gui.h>
#include <taichi/system/profiler.h>
#include <taichi/visualization/particle_visualization.h>
#include <taichi/visual/texture.h>
#include "fem_interface.h"

TC_NAMESPACE_BEGIN

using namespace Tlang;

constexpr int dim = 3, n = 256;
constexpr int pre_and_post_smoothing = 3, bottom_smoothing = 10;

auto mgpcg_poisson = []() {
  CoreState::set_trigger_gdb_when_crash(true);

  Program prog(Arch::x86_64);
  prog.config.print_ir = true;
  prog.config.lazy_compilation = false;

  Global(x, f32);
  Global(r, f32);
  Global(r2, f32);
  Global(p, f32);
  Global(z, f32);
  Global(z2, f32);
  Global(Ap, f32);
  Global(alpha, f32);
  Global(beta, f32);
  Global(sum, f32);
  Global(phase, i32);

  int block_size = 8;

  bool block_soa = true;

  layout([&]() {
    auto ijk = Indices(0, 1, 2);
    std::function<void(Expr & expr)> place_scalar;

    SNode *block;
    if (block_soa) {
      block = &root.dense(ijk, n / block_size).bitmasked();  //.pointer();
      place_scalar = [&](Expr &s) { block->dense(ijk, block_size).place(s); };
    } else {
      place_scalar = [&](Expr &mat) {
        root.dense(ijk, n / block_size).dense(ijk, block_size).place(mat);
      };
    }

    place_scalar(x);
    place_scalar(p);
    place_scalar(Ap);
    place_scalar(r);
    place_scalar(z);
    root.dense(ijk, n / block_size)
        .bitmasked()
        .dense(ijk, block_size)
        .place(r2, z2);
    root.place(alpha, beta, sum, phase);
  });

  auto get_smoother = [&](const Expr &x, const Expr &r) -> Program::Kernel & {
    return kernel([&] {
      Parallelize(8);
      Vectorize(1);
      For(x, [&](Expr i, Expr j, Expr k) {
        If(((i + j + k) & 1) == phase[Expr(0)]).Then([&] {
          x[i, j, k] =
              (r[i, j, k] + x[i - 1, j, k] + x[i + 1, j, k] + x[i, j + 1, k] +
               x[i, j - 1, k] + x[i, j, k + 1] + x[i, j, k - 1]) *
              (1.0f / 6);
        });
      });
    });
  };

  auto get_restrict = [&](const Expr &x, const Expr &r,
                          const Expr &coarse_r) -> Program::Kernel & {
    return kernel([&] {
      Parallelize(8);
      Vectorize(1);
      For(x, [&](Expr i, Expr j, Expr k) {
        auto res =
            Var(r[i, j, k] - (6.0f * x[i, j, k] - x[i - 1, j, k] -
                              x[i + 1, j, k] - x[i, j + 1, k] - x[i, j - 1, k] -
                              x[i, j, k + 1] - x[i, j, k - 1]));
        coarse_r[i / 2, j / 2, k / 2] += res * 0.5f;  // TODO: x2 or x0.5?
      });
    });
  };

  auto get_prolongate = [&](const Expr &z2,
                            const Expr &z) -> Program::Kernel & {
    return kernel([&] {
      Parallelize(8);
      Vectorize(1);
      For(z, [&](Expr i, Expr j, Expr k) {
        z[i, j, k] += z2[i / 2, j / 2, k / 2];
      });
    });
  };

  Kernel(compute_Ap).def([&] {
    BlockDim(1024);
    Parallelize(8);
    Vectorize(block_size);
    For(Ap, [&](Expr i, Expr j, Expr k) {
      Ap[i, j, k] = 6.0f * p[i, j, k] - p[i - 1, j, k] - p[i + 1, j, k] -
                    p[i, j + 1, k] - p[i, j - 1, k] - p[i, j, k + 1] -
                    p[i, j, k - 1];
    });
  });

  prog.config.print_ir = false;

  Kernel(reduce_r).def([&] {
    For(r, [&](Expr i, Expr j, Expr k) {
      Atomic(sum[Expr(0)]) += r[i, j, k] * r[i, j, k];
    });
  });

  Kernel(reduce_zTr).def([&] {
    For(r, [&](Expr i, Expr j, Expr k) {
      Atomic(sum[Expr(0)]) += z[i, j, k] * r[i, j, k];
    });
  });

  Kernel(reduce_pAp).def([&] {
    For(r, [&](Expr i, Expr j, Expr k) {
      auto tmp = Var(0.0f);
      tmp += p[i, j, k] * Ap[i, j, k];
      Atomic(sum[Expr(0)]) += tmp;
    });
  });

  Kernel(update_x).def([&] {
    For(x, [&](Expr i, Expr j, Expr k) {
      x[i, j, k] += alpha[Expr(0)] * p[i, j, k];
    });
  });

  Kernel(update_r).def([&] {
    For(p, [&](Expr i, Expr j, Expr k) {
      r[i, j, k] -= alpha[Expr(0)] * Ap[i, j, k];
    });
  });

  Kernel(update_p).def([&] {
    For(p, [&](Expr i, Expr j, Expr k) {
      p[i, j, k] = z[i, j, k] + beta[Expr(0)] * p[i, j, k];
    });
  });

  int begin = n / 4, end = n * 3 / 4;
  for (int i = begin; i < end; i++) {
    for (int j = begin; j < end; j++) {
      for (int k = begin; k < end; k++) {
        r.val<float32>(i, j, k) = (i == n / 2) && (j == n / 2) && (k == n / 2);
      }
    }
  }

  auto &smooth_level1 = get_smoother(z, r);
  auto &smooth_level2 = get_smoother(z2, r2);
  auto &restrict = get_restrict(z, r, r2);
  auto &prolongate = get_prolongate(z2, z);

  Kernel(clear_z).def(
      [&] { For(z, [&](Expr i, Expr j, Expr k) { z[i, j, k] = 0.0f; }); });

  Kernel(clear_r2).def(
      [&] { For(r2, [&](Expr i, Expr j, Expr k) { r2[i, j, k] = 0.0f; }); });

  Kernel(clear_z2).def(
      [&] { For(z2, [&](Expr i, Expr j, Expr k) { z2[i, j, k] = 0.0f; }); });

  // z = M^-1 r
  auto apply_preconditioner = [&] {
    clear_z();
    for (int i = 0; i < pre_and_post_smoothing; i++) {
      phase.val<int32>() = 0;
      smooth_level1();
      phase.val<int32>() = 1;
      smooth_level1();
    }
    clear_z2();
    clear_r2();
    restrict();
    for (int i = 0; i < bottom_smoothing; i++) {
      phase.val<int32>() = 0;
      smooth_level2();
      phase.val<int32>() = 1;
      smooth_level2();
    }
    prolongate();
    for (int i = 0; i < pre_and_post_smoothing; i++) {
      phase.val<int32>() = 0;
      smooth_level1();
      phase.val<int32>() = 1;
      smooth_level1();
    }
  };

  // r = b - Ax = b    since x = 0
  // p = r = r + 0 p
  apply_preconditioner();
  update_p();
  sum.val<float32>() = 0;
  reduce_zTr();
  auto old_zTr = sum.val<float32>();

  // CG
  for (int i = 0; i < 40; i++) {
    TC_P(i);
    compute_Ap();
    sum.val<float32>() = 0;
    reduce_pAp();
    auto pAp = sum.val<float32>();
    // alpha = rTr / pTAp
    alpha.val<float32>() = old_zTr / pAp;
    TC_P(old_zTr);
    // TC_P(pAp);
    // TC_P(alpha.val<float32>());
    // x = x + alpha p
    update_x();
    // r = r - alpha Ap
    update_r();
    // return if |r| small
    // z = M^-1 r
    apply_preconditioner();
    sum.val<float32>() = 0;
    reduce_zTr();
    auto new_zTr = sum.val<float32>();
    TC_P(new_zTr);
    sum.val<float32>() = 0;
    reduce_r();
    auto rTr = sum.val<float32>();
    if (rTr < 1e-7f)
      break;
    // beta = new rTr / old rTr
    beta.val<float32>() = new_zTr / old_zTr;
    TC_P(beta.val<float32>());
    // p = z + beta p
    update_p();
    old_zTr = new_zTr;
  }
  get_current_program().profiler_print();

  compute_Ap();
  auto residual = 0.0f;
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      for (int k = 0; k < n; k++) {
        for (int d = 0; d < dim; d++) {
          // residual += std::abs(R[i][j][k][d] - Ap(d).val<float32>(i, j, k));
        }
      }
    }
  }
  TC_P(residual);
  // TC_P(difference_max);

  int gui_res = 512;
  GUI gui("MGPCG Poisson", Vector2i(gui_res + 200, gui_res), false);
  int gt = 0;
  int k = 0;
  gui.slider("z", k, 0, n - 1).slider("Ground truth", gt, 0, 1);

  int scale = gui_res / n;
  auto &canvas = gui.get_canvas();
  for (int frame = 1;; frame++) {
    for (int i = 0; i < gui_res - scale; i++) {
      for (int j = 0; j < gui_res - scale; j++) {
        real dx = x.val<float32>(i / scale, j / scale, k);
        canvas.img[i][j] = Vector4(0.5f) + Vector4(dx, dx, dx, 0) * 15.0f;
      }
    }
    gui.update();
  }
};
TC_REGISTER_TASK(mgpcg_poisson);

TC_NAMESPACE_END