#include "../tlang.h"
#include <taichi/util.h>
#include <taichi/visual/gui.h>
#include <taichi/system/profiler.h>
#include <taichi/visualization/particle_visualization.h>
#include <taichi/visual/texture.h>

TC_NAMESPACE_BEGIN

#include "fem_coeff.h"

using namespace Tlang;

constexpr int dim = 3, n = 64;
bool active[n][n][n];

auto fem = []() {
  CoreState::set_trigger_gdb_when_crash(true);

  Program prog(Arch::x86_64);
  prog.config.print_ir = true;
  prog.config.lazy_compilation = false;

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
      auto cell_coord = Var(Vector({i, j, k}));
      auto Ku_tmp = Var(Vector(dim));
      Ku_tmp = Vector({0.0f, 0.0f, 0.0f});
      for (int cell = 0; cell < pow<dim>(2); cell++) {
        auto cell_offset =
            Var(Vector({-(cell / 4), -(cell / 2 % 2), -(cell % 2)}));
        auto cell_lambda = lambda[cell_coord + cell_offset];
        auto cell_mu = mu[cell_coord + cell_offset];
        for (int node = 0; node < pow<dim>(2); node++) {
          auto node_offset = Var(Vector({node / 4, node / 2 % 2, node % 2}));
          for (int u = 0; u < dim; u++) {
            for (int v = 0; v < dim; v++) {
              Ku_tmp(u) += (cell_lambda * K_la[cell][node][u][v] +
                            cell_mu * K_mu[cell][node][u][v]) *
                           p[cell_coord + cell_offset + node_offset](v);
            }
          }
        }
      }
      // boundary condition
      If(j < 2).Then([&] { Ku_tmp = Vector({0.0f, 0.0f, 0.0f}); });
      Ap[i, j, k] = Ku_tmp;
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
      r[i, j, k] -= alpha[Expr(0)] * Ap[i, j, k];
    });
  });

  Kernel(copy_b_to_r).def([&] {
    For(p(0), [&](Expr i, Expr j, Expr k) {
      If(i == n / 2 && j == n / 2 && k == n / 2)
          // If(j == n / 2)
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

  const auto E = 1_f;     // Young's Modulus
  const auto nu = 0.2_f;  // Poisson ratio

  const real mu_0 = E / (2 * (1 + nu));
  const real lambda_0 = E * nu / ((1 + nu) * (1 - 2 * nu));

  auto tex = create_instance<Texture>(
      "mesh", Dict()
                  .set("resolution", Vector3(n))
                  .set("translate", Vector3(0.55, 0.40, 0.47))
                  .set("scale", Vector3(1.1))
                  .set("adaptive", false)
                  .set("filename", "$mpm/bunny_small.obj"));

  for (int i = 1; i < n - 2; i++) {
    for (int j = 1; j < n - 2; j++) {
      for (int k = 1; k < n - 2; k++) {
        bool inside = tex->sample((Vector3(0.5f) + Vector3(i, j, k)) *
                                  Vector3(1.0f / (n - 1)))
                          .x > 0.5f;
        inside = pow<2>(i - n / 2) + pow<2>(k - n / 2) < pow<2>(n / 2) / 2;
        // bool inside = i + j > n * 0.2;
        if (inside) {
          active[i][j][k] = true;
          lambda.val<float32>(i, j, k) = lambda_0;
          mu.val<float32>(i, j, k) = mu_0;
          if (j == n / 3 && i == n / 2)
            r(0).val<float32>(i, j, k) = -1.0f;
        }
        // if (j == n / 2 && i == n / 2)
        // r(0).val<float32>(i, j, k) = -1.0f;
      }
    }
  }

  std::deque<Vector3i> q;
  q.push_back(Vector3i(n / 2));  // Assuming the center voxel is active
  std::function<void(int, int, int)> dfs = [&](int i, int j, int k) {
    if (active[i][j][k]) {
      active[i][j][k] = 0;
      q.push_back(Vector3i(i, j, k));
    }
  };
  while (!q.empty()) {
    auto x = q.front();
    q.pop_front();
    auto i = x[0];
    auto j = x[1];
    auto k = x[2];
    // TC_INFO("{} {} {}", i, j, k);
    dfs(i + 1, j, k);
    dfs(i - 1, j, k);
    dfs(i, j + 1, k);
    dfs(i, j - 1, k);
    dfs(i, j, k + 1);
    dfs(i, j, k - 1);
  }

  for (int i = 0; i < n - 1; i++) {
    for (int j = 0; j < n - 1; j++) {
      for (int k = 0; k < n - 1; k++) {
        TC_ASSERT(!active[i][j][k]);
      }
    }
  }

  // r = b - Ax = b    since x = 0
  // copy_b_to_r();
  // p = r = r + 0 p
  update_p();
  sum.val<float32>() = 0;
  reduce_r();
  auto old_rTr = sum.val<float32>();

  for (int i = 0; i < 1000; i++) {
    compute_Ap();
    sum.val<float32>() = 0;
    reduce_pAp();
    auto pAp = sum.val<float32>();
    // alpha = rTr / pTAp
    alpha.val<float32>() = old_rTr / pAp;
    TC_P(old_rTr);
    // TC_P(pAp);
    // TC_P(alpha.val<float32>());
    // x = x + alpha p
    update_x();
    // r = r - alpha Ap
    update_r();
    // return if |r| small
    sum.val<float32>() = 0;
    reduce_r();
    auto new_rTr = sum.val<float32>();
    // TC_P(new_rTr);
    if (new_rTr < 1e-3f)
      break;
    // beta = new rTr / old rTr
    beta.val<float32>() = new_rTr / old_rTr;
    // TC_P(beta.val<float32>());
    // p = r + beta p
    update_p();
    old_rTr = new_rTr;
  }

  int gui_res = 512;
  GUI gui("FEM", Vector2i(gui_res + 200, gui_res), false);
  int k = 0;
  gui.slider("z", k, 0, n);

  int scale = gui_res / n;
  auto &canvas = gui.get_canvas();
  for (int frame = 1;; frame++) {
    for (int i = 0; i < gui_res - scale; i++) {
      for (int j = 0; j < gui_res - scale; j++) {
        auto dx = x(0).val<float32>(i / scale, j / scale, k);
        auto dy = x(1).val<float32>(i / scale, j / scale, k);
        auto dz = x(2).val<float32>(i / scale, j / scale, k);
        canvas.img[i][j] = Vector4(0.5f) + Vector4(dx, dy, dz, 0) * 0.5f;
      }
    }
    gui.update();
  }
};
TC_REGISTER_TASK(fem);

TC_NAMESPACE_END
