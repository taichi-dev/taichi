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

  Program prog(Arch::x86_64);
  prog.config.print_ir = true;
  prog.config.lazy_compilation = false;

  constexpr int dim = 3, n = 2;

  Vector x(DataType::f32, dim), r(DataType::f32, dim), p(DataType::f32, dim),
      Ap(DataType::f32, dim);

  Global(alpha, f32);
  Global(beta, f32);
  Global(sum, f32);

  AmbientGlobal(lambda, f32, 0.0f);
  AmbientGlobal(mu, f32, 0.0f);

  layout([&]() {
    auto ijk = Indices(0, 1, 2);
    root.dense(ijk, {n, n, n}).place(x);
    root.dense(ijk, {n, n, n}).place(p);
    root.dense(ijk, {n, n, n}).place(Ap);
    root.dense(ijk, {n, n, n}).place(r);
    root.dense(ijk, {n, n, n}).place(lambda);
    root.dense(ijk, {n, n, n}).place(mu);
    root.place(alpha, beta, sum);
  });

  Kernel(compute_Ap).def([&] {
    BlockDim(1024);
    For(Ap(0), [&](Expr i, Expr j, Expr k) {
      /*
      auto cell_coord = Var(Vector({i, j, k}));
      auto Ku_tmp = Var(Vector(dim));
      Ku_tmp = Vector({0.0f, 0.0f, 0.0f});
      for (int cell = 0; cell < pow<dim>(2); cell++) {
        auto cell_offset = Var(Vector({cell / 4, cell / 2 % 2, cell % 2}));
        auto cell_lambda = lambda[cell_coord + cell_offset];
        auto cell_mu = mu[cell_coord + cell_offset];
        for (int node = 0; node < pow<dim>(2); node++) {
          auto node_offset = Var(Vector({node / 4, node / 2 % 2, node % 2}));
          for (int u = 0; u < dim; u++) {
            for (int v = 0; v < dim; v++) {
              auto weight = 1.0f;
              Ku_tmp(u) += (cell_lambda * K_la[cell][node][u][v] +
                            cell_mu * K_mu[cell][node][u][v]) *
                           p[i, j, k](v);
            }
          }
        }
      }
      // boundary condition
      If(j == 0).Then([&] { Ku_tmp = Vector({0.0f, 0.0f, 0.0f}); });
      Ap[i, j, k] = Ku_tmp;
      */
      Ap[i, j, k] = p[i, j, k] * 0.5f;
    });
  });

  Kernel(reduce_r).def([&] {
    For(r(0), [&](Expr i, Expr j, Expr k) {
      Atomic(sum[Expr(0)]) += r[i, j, k].norm2();
    });
  });

  Kernel(reduce_pAp).def([&] {
    For(r(0), [&](Expr i, Expr j, Expr k) {
      auto tmp = Var(0.0f);
      for (int d = 0; d < dim; d++) {
        tmp += p[i, j, k](d) * Ap[i, j, k](d);
      }
      Atomic(sum[Expr(0)]) += tmp;
    });
  });

  Kernel(update_x).def([&] {
    For(x(0), [&](Expr i, Expr j, Expr k) {
      x[i, j, k] += alpha[Expr(0)] * p[i, j, k];
    });
  });

  Kernel(update_r).def([&] {
    For(p(0), [&](Expr i, Expr j, Expr k) {
      r[i, j, k] -= alpha[Expr(0)] * p[i, j, k];
    });
  });

  Kernel(copy_b_to_r).def([&] {
    For(p(0), [&](Expr i, Expr j, Expr k) {
      If(i == 0 && j == n - 1 && k == 0)
          .Then([&] {
            r[i, j, k] = Vector({0.0f, -1.0f, 0.0f});
          })
          .Else([&] {
            r[i, j, k] = Vector({0.0f, 0.0f, 0.0f});
          });
    });
  });

  Kernel(update_p).def([&] {
    For(p(0), [&](Expr i, Expr j, Expr k) {
      p[i, j, k] = r[i, j, k] + beta[Expr(0)] * p[i, j, k];
    });
  });

  Kernel(enforce_bc).def([&] {
    For(x(0), [&](Expr i, Expr j, Expr k) {
      If(j == 0).Then([&] { x[i, j, k] = Vector({0.0f, 0.0f, 0.0f}); });
    });
  });

  for (int i = 0; i < n - 1; i++) {
    for (int j = 0; j < n - 1; j++) {
      for (int k = 0; k < n - 1; k++) {
        lambda.val<float32>(i, j, k) = 1.0f;
        mu.val<float32>(i, j, k) = 1.0f;
      }
    }
  }

  // r = b - Ax = b    since x = 0
  copy_b_to_r();
  // p = r = r + 0 p
  update_p();
  sum.val<float32>() = 0;
  reduce_r();
  auto old_rTr = sum.val<float32>();
  TC_P(old_rTr);
  for (int i = 0; i < 10; i++) {
    // CG
    compute_Ap();
    sum.val<float32>() = 0;
    reduce_pAp();
    auto pAp = sum.val<float32>();
    // alpha = rTr / pTAp
    alpha.val<float32>() = old_rTr / pAp;
    TC_P(alpha.val<float32>());
    // x = x + alpha p
    update_x();
    // r = r - alpha Ap
    update_r();
    // return if |r| small
    sum.val<float32>() = 0;
    reduce_r();
    auto new_rTr = sum.val<float32>();
    TC_P(new_rTr);
    if (new_rTr < 1e-5f)
      break;
    // beta = new rTr / old rTr
    beta.val<float32>() = new_rTr / old_rTr;
    TC_P(beta.val<float32>());
    // p = r + beta p
    update_p();
    old_rTr = new_rTr;
  }
};
TC_REGISTER_TASK(fem);

TC_NAMESPACE_END
