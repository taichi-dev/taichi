#include "../tlang.h"
#include <taichi/util.h>
#include <taichi/visual/gui.h>
#include <taichi/system/profiler.h>
#include <taichi/visualization/particle_visualization.h>
#include <taichi/visual/texture.h>
#include "fem_interface.h"

TC_NAMESPACE_BEGIN

using namespace Tlang;

constexpr int dim = 3, n = 512; // double the size. actual box is 256^3
constexpr int pre_and_post_smoothing = 2, bottom_smoothing = 50;
constexpr int mg_levels = 6;

auto mgpcg_poisson = [](std::vector<std::string> cli_param) {
  auto param = parse_param(cli_param);

  int block_size = 8;

  int threads = param.get("threads", 8);
  TC_P(threads);
  int vec = param.get("vec", 8);
  TC_P(vec);
  TC_ASSERT(vec == 1 || vec == 8);

  CoreState::set_trigger_gdb_when_crash(true);

  Program prog(Arch::x86_64);
  // prog.config.print_ir = true;
  prog.config.lazy_compilation = false;

  auto r = Vector(DataType::f32, mg_levels);
  auto z = Vector(DataType::f32, mg_levels);
  Global(x, f32);
  Global(p, f32);
  Global(Ap, f32);
  Global(alpha, f32);
  Global(beta, f32);
  Global(sum, f32);
  Global(phase, i32);

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
    place_scalar(r(0));
    place_scalar(z(0));
    for (int i = 1; i < mg_levels; i++) {
      auto &fork = root.dense(ijk, n / block_size).bitmasked();
      fork.dense(ijk, block_size).place(r(i));
      fork.dense(ijk, block_size).place(z(i));
    }
    root.place(alpha, beta, sum, phase);
  });

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

  Kernel(reduce_r).def([&] {
    mark_reduction();
    Parallelize(threads);
    Vectorize(block_size);
    For(p, [&](Expr i, Expr j, Expr k) {
      Atomic(sum[Expr(0)]) += r(0)[i, j, k] * r(0)[i, j, k];
    });
  });

  Kernel(reduce_zTr).def([&] {
    mark_reduction();
    Parallelize(threads);
    Vectorize(block_size);
    For(p, [&](Expr i, Expr j, Expr k) {
      Atomic(sum[Expr(0)]) += z(0)[i, j, k] * r(0)[i, j, k];
    });
  });

  Kernel(reduce_pAp).def([&] {
    mark_reduction();
    Parallelize(threads);
    Vectorize(block_size);
    For(p, [&](Expr i, Expr j, Expr k) {
      auto tmp = Var(0.0f);
      tmp += p[i, j, k] * Ap[i, j, k];
      Atomic(sum[Expr(0)]) += tmp;
    });
  });

  Kernel(update_x).def([&] {
    Parallelize(8);
    Vectorize(block_size);
    For(x, [&](Expr i, Expr j, Expr k) {
      x[i, j, k] += alpha[Expr(0)] * p[i, j, k];
    });
  });

  Kernel(update_r).def([&] {
    Parallelize(8);
    Vectorize(block_size);
    For(p, [&](Expr i, Expr j, Expr k) {
      r(0)[i, j, k] -= alpha[Expr(0)] * Ap[i, j, k];
    });
  });

  Kernel(update_p).def([&] {
    Parallelize(8);
    Vectorize(block_size);
    For(p, [&](Expr i, Expr j, Expr k) {
      p[i, j, k] = z(0)[i, j, k] + beta[Expr(0)] * p[i, j, k];
    });
  });

  int begin = n / 4, end = n * 3 / 4;
  for (int i = begin; i < end; i++) {
    for (int j = begin; j < end; j++) {
      for (int k = begin; k < end; k++) {
        /*
        r(0).val<float32>(i, j, k) =
            (i == n / 2) && (j == n / 2) && (k == n / 2);
            */
        float x = (i - begin) * 2.0f / n;
        float y = (i - begin) * 2.0f / n;
        float z = (i - begin) * 2.0f / n;
        r(0).val<float32>(i, j, k) =
            sin(2 * pi * x) * cos(2 * pi * y) * sin(2 * pi * z);
      }
    }
  }

  std::vector<std::function<void()>> smoothers(mg_levels),
      smoothers2(mg_levels), restrictors(mg_levels - 1),
      prolongators(mg_levels - 1), clearer_z(mg_levels), clearer_r(mg_levels);

  for (int l = 0; l < mg_levels; l++) {
    if (l < mg_levels - 1) {
      prog.config.print_ir = true;
      restrictors[l] =
          kernel([&] {
            kernel_name(fmt::format("restrict_lv{}", l));
            Parallelize(8);
            Vectorize(1);
            For(r(l), [&](Expr i, Expr j, Expr k) {
              auto res =
                  Var(r(l)[i, j, k] - (6.0f * z(l)[i, j, k] -
                                       z(l)[i - 1, j, k] - z(l)[i + 1, j, k] -
                                       z(l)[i, j + 1, k] - z(l)[i, j - 1, k] -
                                       z(l)[i, j, k + 1] - z(l)[i, j, k - 1]));
              r(l + 1)[i / 2, j / 2, k / 2] += res * 0.5f;
            });
          })
              .func();
      prog.config.print_ir = false;
      prolongators[l] = kernel([&] {
                          kernel_name(fmt::format("prolongate_lv{}", l));
                          Parallelize(8);
                          Vectorize(1);
                          For(z(l), [&](Expr i, Expr j, Expr k) {
                            z(l)[i, j, k] += z(l + 1)[i / 2, j / 2, k / 2];
                          });
                        })
                            .func();
    }
    smoothers[l] =
        kernel([&] {
          kernel_name(fmt::format("smooth_lv{}", l));
          Parallelize(8);
          Vectorize(block_size);
          For(z(l), [&](Expr i, Expr j, Expr k) {
            auto ret = Var(z(l)[i, j, k]);
            If(((i + j + k) & 1) == phase[Expr(0)]).Then([&] {
              ret = (r(l)[i, j, k] + z(l)[i - 1, j, k] + z(l)[i + 1, j, k] +
                     z(l)[i, j + 1, k] + z(l)[i, j - 1, k] + z(l)[i, j, k + 1] +
                     z(l)[i, j, k - 1]) *
                    (1.0f / 6);
            });
            z(l)[i, j, k] = ret;
          });
        })
            .func();
    clearer_r[l] =
        kernel([&] {
          kernel_name(fmt::format("clear_r_lv{}", l));
          Parallelize(8);
          Vectorize(block_size);
          For(r(l), [&](Expr i, Expr j, Expr k) { r(l)[i, j, k] = 0.0f; });
        })
            .func();
    clearer_z[l] =
        kernel([&] {
          kernel_name(fmt::format("clear_z_lv{}", l));
          Parallelize(8);
          Vectorize(8);
          For(z(l), [&](Expr i, Expr j, Expr k) { z(l)[i, j, k] = 0.0f; });
        })
            .func();
  }

  // z = M^-1 r
  auto apply_preconditioner = [&] {
    clearer_z[0]();
    for (int l = 0; l < mg_levels - 1; l++) {
      for (int i = 0; i < (pre_and_post_smoothing << l); i++) {
        phase.val<int32>() = 0;
        smoothers[l]();
        phase.val<int32>() = 1;
        smoothers[l]();
      }
      clearer_z[l + 1]();
      clearer_r[l + 1]();
      restrictors[l]();
    }
    for (int i = 0; i < bottom_smoothing; i++) {
      phase.val<int32>() = 0;
      smoothers[mg_levels - 1]();
      phase.val<int32>() = 1;
      smoothers[mg_levels - 1]();
    }
    for (int l = mg_levels - 2; l >= 0; l--) {
      prolongators[l]();
      for (int i = 0; i < (pre_and_post_smoothing << l); i++) {
        phase.val<int32>() = 0;
        smoothers[l]();
        phase.val<int32>() = 1;
        smoothers[l]();
      }
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
  auto t = Time::get_time();
  for (int i = 0; i < 40; i++) {
    TC_P(i);
    compute_Ap();
    sum.val<float32>() = 0;
    reduce_pAp();
    auto pAp = sum.val<float32>();
    // alpha = rTr / pTAp
    alpha.val<float32>() = old_zTr / pAp;
    // TC_P(old_zTr);
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
    // TC_P(new_zTr);
    sum.val<float32>() = 0;
    reduce_r();
    auto rTr = sum.val<float32>();
    TC_P(rTr);
    if (rTr < 1e-7f)
      break;
    // beta = new rTr / old rTr
    beta.val<float32>() = new_zTr / old_zTr;
    // TC_P(beta.val<float32>());
    // p = z + beta p
    update_p();
    old_zTr = new_zTr;
  }
  TC_P(Time::get_time() - t);
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
        canvas.img[i][j] = Vector4(0.5f) + Vector4(dx, dx, dx, 0) * 0.01f;
      }
    }
    gui.update();
  }
};
TC_REGISTER_TASK(mgpcg_poisson);

TC_NAMESPACE_END