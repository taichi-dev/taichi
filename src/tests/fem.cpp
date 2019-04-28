#include "../tlang.h"
#include <taichi/util.h>
#include <taichi/visual/gui.h>
#include <taichi/system/profiler.h>
#include <taichi/visualization/particle_visualization.h>

TC_NAMESPACE_BEGIN

#include "fem_coeff.h"

using namespace Tlang;

auto fem = []() {
  CoreState::set_trigger_gdb_when_crash(true);

  Program prog(Arch::gpu);

  constexpr int dim = 3, n = 256;

  Vector x(dim), r(dim), p(dim), Ap(dim);

  Global(alpha, f32);
  Global(beta, f32);
  Global(sum, f32);

  Global(mu, f32);
  Global(lambda, f32);

  layout([&]() {
    auto ijk = Indices(0, 1, 2);
    root.dense(ijk, {n, n, n}).place(x);
    root.dense(ijk, {n, n, n}).place(r);
    root.place(alpha, beta, sum);
  });

  Kernel(compute_ku).def([&] {
    BlockDim(1024);
    For(Ap(0), [&](Expr i, Expr j, Expr k) {
      auto Ku_tmp = Var(Vector(dim));
      for (int cell = 0; cell < pow<dim>(2); cell++) {
        for (int node = 0; node < pow<dim>(2); node++) {
          for (int p = 0; p < dim; p++) {
            for (int q = 0; q < dim; q++) {
              auto weight = 1.0f;
              K_la[cell][node][p][q];
              K_mu[cell][node][p][q];
              Ku_tmp(p) += weight * x[i, j, k](q);
            }
          }
        }
      }
      Ap[i, j, k] = Ku_tmp;
    });
  });

  Kernel(reduce_r).def([&] {
    For(r(0),
        [&](Expr i, Expr j, Expr k) { sum[Expr(0)] += r[i, j, k].norm2(); });
  });

  Kernel(update_x).def([&] {
    For(x(0),
        [&](Expr i, Expr j, Expr k) { x[i, j, k] += alpha * p[i, j, k]; });
  });

  Kernel(update_r).def([&] {
    For(p(0),
        [&](Expr i, Expr j, Expr k) { r[i, j, k] -= alpha * p[i, j, k]; });
  });

  Kernel(update_p).def([&] {
    For(p(0), [&](Expr i, Expr j, Expr k) {
      p[i, j, k] = r[i, j, k] + beta * p[i, j, k];
    });
  });

  Kernel(enforce_bc).def([&] {
    For(x(0), [&](Expr i, Expr j, Expr k) {
      If(j == 0).Then([&] { x[i, j, k] = Vector({0.0f, 0.0f, 0.0f}); });
    });
  });

  // r = n - Ax
  // p = r
  for (int i = 0; i < 1000; i++) {
    // CG
    // alpha = rTr / pTAp
    // x = x + alpha p
    update_x();
    // r = r - alpha Ap
    update_r();
    // return if |r| small
    sum.val<float32>() = 0;
    reduce_r();
    auto rTr = 0.0_f;
    auto old_rTr = rTr;
    auto new_rTr = sum.val<float32>();
    // beta = new rTr / old rTr
    beta.val<float32>() = new_rTr / old_rTr;
    // p = r + beta p
    update_p();
  }
};
TC_REGISTER_TASK(fem);

TC_NAMESPACE_END
