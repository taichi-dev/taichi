#include "../tlang.h"
#include <taichi/util.h>
#include <taichi/visual/gui.h>
#include <taichi/system/profiler.h>
#include <taichi/visualization/particle_visualization.h>
#include <taichi/visual/texture.h>
#include "fem_interface.h"

TC_NAMESPACE_BEGIN

#include "fem_coeff.h"

using namespace Tlang;

constexpr int dim = 3, n = 64;
bool active[n][n][n];
real F = -10;
real R[n][n][n][dim], D[n][n][n], X[n][n][n][dim];

const auto E = 1_f;     // Young's Modulus
const auto nu = 0.2_f;  // Poisson ratio

const real mu_0 = E / (2 * (1 + nu));
const real lambda_0 = E * nu / ((1 + nu) * (1 - 2 * nu));

void fem_solve() {
  Dict config;
  using namespace fem_interface;
  FEMInterface interface;
  auto &param = interface.param;
  constexpr int padding = 8;

  // TC_P(container_bounds);
  // Cell size, instead of node size
  for (int i = 0; i < 3; i++)
    param.resolution[i] = n + padding * 3;

  param.global_mu = mu_0;
  param.global_lambda = lambda_0;
  param.use_density_only = true;
  param.dx = 1;

  param.krylov.tolerance = config.get("cg_tolerance", 1e-7_f);
  param.krylov.max_iterations = config.get("cg_max_iterations", 50);
  param.krylov.print_residuals = config.get("print_residuals", true);
  param.krylov.restart_iterations = config.get("restart_iterations", 0);
  TC_ERROR_IF(config.has_key("global_smoothing_iter"),
              "global_smoothing_iter is obselete.");
  param.defect_correction_cg_iter() =
      config.get("defect_correction_cg_iter", 3);
  param.penalty() = 1;
  param.minimum_stiffness() = 0;

  // Do not do warm starting for a pure objective solve
  param.keep_state() = false;
  param.forced_reuse() = (int)config.get<bool>("forced_reuse", true);

  // Bool's solver type option
  param.solver_type() = config.get<int>("solver_type", 1);

  // Multigrid Parameters
  param.mg_level = 3;
  param.explicit_mg_level = 0;

  if (param.explicit_mg_level >= param.mg_level) {
    TC_WARN("Clamping explicit_mg_level ({}) to {}!", param.explicit_mg_level,
            param.mg_level - 1);
    param.explicit_mg_level = param.mg_level - 1;
  }
  param.pre_and_post_smoothing_iter = 1;
  param.boundary_smoothing() = 3;
  param.bottom_smoothing_iter = 600;
  param.jacobi_damping = 0.4f;

  using scalar_block_size = FEMInputs::ScalarGrid::block_size;

  for (int i = 0; i < 3; i++) {
    param.mu.resolution[i] = param.resolution[i];
    param.lambda.resolution[i] = param.resolution[i];
    param.density.resolution[i] = param.resolution[i];
  }

  Vector3i block_size(scalar_block_size::x, scalar_block_size::y,
                      scalar_block_size::z);

  double bounds[2] = {1e30f, -1e30f};

  bool has_nan = false;

  for (int I = 0; I < n; I += block_size.x)
    for (int J = 0; J < n; J += block_size.y)
      for (int K = 0; K < n; K += block_size.z) {
        Vector3i base_coord(I, J, K);

        TC_ASSERT(n % scalar_block_size::x == 0);
        TC_ASSERT(n % scalar_block_size::y == 0);
        TC_ASSERT(n % scalar_block_size::z == 0);

        for (int i = base_coord[0]; i < base_coord[0] + block_size.x;
             i += scalar_block_size::x) {
          for (int j = base_coord[1]; j < base_coord[1] + block_size.y;
               j += scalar_block_size::y) {
            for (int k = base_coord[2]; k < base_coord[2] + block_size.z;
                 k += scalar_block_size::z) {
              FEMInputs::ScalarGrid::Block density_block;

              density_block.base_coordinates[0] = i + padding;
              density_block.base_coordinates[1] = j + padding;
              density_block.base_coordinates[2] = k + padding;

              for (int ii = 0; ii < scalar_block_size::x; ii++) {
                for (int jj = 0; jj < scalar_block_size::y; jj++) {
                  for (int kk = 0; kk < scalar_block_size::z; kk++) {
                    auto scale = (double)D[i + ii][j + jj][k + kk];
                    // auto scale = 1.0;
                    density_block.get(ii, jj, kk) = scale;
                    bounds[0] = std::min(bounds[0], scale);
                    bounds[1] = std::max(bounds[1], scale);
                    has_nan = has_nan || (scale != scale);
                  }
                }
              }
              param.density.blocks.push_back(density_block);
            }
          }
        }
      }

  TC_P(bounds);
  TC_ERROR_IF(has_nan, "density field contains nan!");

  for (int i = 1; i < n - 2; i++) {
    for (int j = 0; j < 2; j++) {
      for (int k = 1; k < n - 2; k++) {
        for (int d = 0; d < 3; d++) {
          DirichletOnNode dirichlet;
          dirichlet.coord[0] = i + padding;
          dirichlet.coord[1] = j + padding;
          dirichlet.coord[2] = k + padding;
          dirichlet.axis = d;
          dirichlet.value = 0;
          param.dirichlet_nodes.push_back(dirichlet);
        }
      }
    }
  }

  param.forces.push_back(fem_interface::ForceOnNode{
      {n / 2 + padding, n / 2 + padding, n / 2 + padding}, {0, F, 0}});

  param.caller_method = "taichi_benchmark";
  interface.preserve_output(param.density.blocks.size());

  TC_INFO("Invoking FEM Solver");

  param.min_fraction() = 0;
  std::string solver = config.get<std::string>("solver", "");
  TC_PROFILE("solver", interface.run());
  TC_INFO("FEM Solver returned");

  using vector_block_size = FEMInputs::ScalarGrid::block_size;

  bounds[0] = 1e30f;
  bounds[1] = -1e30f;

  has_nan = false;

  for (auto &block : interface.outputs.displacements.blocks) {
    int i = block.base_coordinates[0];
    int j = block.base_coordinates[1];
    int k = block.base_coordinates[2];
    for (int ii = 0; ii < vector_block_size::x; ii++) {
      for (int jj = 0; jj < vector_block_size::y; jj++) {
        for (int kk = 0; kk < vector_block_size::z; kk++) {
          /*
          auto sparse_v0 = grid->v0();
          sparse_v2(i + ii, j + jj, k + kk) = block.get(ii, jj, kk)[2];
          */

          for (int d = 0; d < dim; d++) {
            int I = i + ii - padding, J = j + jj - padding,
                K = k + kk - padding;
            if (0 <= I && I < n && 0 <= J && J < n && 0 <= K && K < n)
              X[I][J][K][d] = block.get(ii, jj, kk)[d];
          }

          for (int r = 0; r < dim; r++) {
            bounds[0] = std::min(bounds[0], block.get(ii, jj, kk)[r]);
            bounds[1] = std::max(bounds[1], block.get(ii, jj, kk)[r]);
            has_nan = has_nan ||
                      (block.get(ii, jj, kk)[r] != block.get(ii, jj, kk)[r]);
          }
        }
      }
    }
  }
  TC_P(bounds);
  TC_WARN_UNLESS(interface.outputs.success, "FEM solve has failed!");
  TC_WARN_UNLESS(!has_nan, "FEM solution contains nan!");

  bool success = !has_nan && interface.outputs.success;
  TC_ASSERT(success);
}

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

  int block_size = 8;

  bool block_soa = true;

  layout([&]() {
    auto ijk = Indices(0, 1, 2);
    std::function<void(Matrix & mat)> place;
    std::function<void(Expr & expr)> place_scalar;

    SNode *block;
    if (block_soa) {
      block = &root.dense(ijk, n / block_size);
      place_scalar = [&](Expr &s) { block->dense(ijk, block_size).place(s); };
      place = [&](Matrix &mat) {
        for (auto &e : mat.entries) {
          place_scalar(e);
        }
      };
    } else {
      place = [&](Matrix &mat) {
        root.dense(ijk, n / block_size).dense(ijk, block_size).place(mat);
      };
      place_scalar = [&](Expr &mat) {
        root.dense(ijk, n / block_size).dense(ijk, block_size).place(mat); };
    }

    place(x);
    place(p);
    place(Ap);
    place(r);
    place_scalar(lambda);
    place_scalar(mu);
    root.place(alpha, beta, sum);
  });

  Kernel(compute_Ap).def([&] {
    BlockDim(1024);
    Parallelize(8);
    // Vectorize(1);
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

  prog.config.print_ir = false;

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

  auto tex = create_instance<Texture>(
      "mesh", Dict()
                  .set("resolution", Vector3(n))
                  .set("translate", Vector3(0.55, 0.35, 0.47))
                  .set("scale", Vector3(1.1))
                  .set("adaptive", false)
                  .set("filename", "$mpm/bunny_small.obj"));

  for (int i = 1; i < n - 2; i++) {
    for (int j = 1; j < n - 2; j++) {
      for (int k = 1; k < n - 2; k++) {
        bool inside = tex->sample((Vector3(0.5f) + Vector3(i, j, k)) *
                                  Vector3(1.0f / (n - 1)))
                          .x > 0.5f;
        // inside = pow<2>(i - n / 2) + pow<2>(k - n / 2) < pow<2>(n / 2) / 2;
        // inside = i < n * 0.8 && j < n * 0.8 && k < n * 0.8;
        if (inside) {
          D[i][j][k] = 1;
          active[i][j][k] = true;
          lambda.val<float32>(i, j, k) = lambda_0;
          mu.val<float32>(i, j, k) = mu_0;
        }
      }
    }
  }
  r(1).val<float32>(n / 2, n / 2, n / 2) = F;
  R[n / 2][n / 2][n / 2][1] = F;

  std::deque<Vector3i> q;
  q.push_back(Vector3i(n / 2));  // Assuming the center voxel is active
  std::function<void(int, int, int)> enqueue = [&](int i, int j, int k) {
    if (active[i][j][k]) {
      active[i][j][k] = false;
      q.push_back(Vector3i(i, j, k));
    }
  };
  while (!q.empty()) {
    auto x = q.front();
    q.pop_front();
    auto i = x[0];
    auto j = x[1];
    auto k = x[2];
    enqueue(i + 1, j, k);
    enqueue(i - 1, j, k);
    enqueue(i, j + 1, k);
    enqueue(i, j - 1, k);
    enqueue(i, j, k + 1);
    enqueue(i, j, k - 1);
  }

  for (int i = 0; i < n - 1; i++) {
    for (int j = 0; j < n - 1; j++) {
      for (int k = 0; k < n - 1; k++) {
        TC_ASSERT(!active[i][j][k]);
      }
    }
  }

  fem_solve();

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
    if (new_rTr < 1e-5f)
      break;
    // beta = new rTr / old rTr
    beta.val<float32>() = new_rTr / old_rTr;
    // TC_P(beta.val<float32>());
    // p = r + beta p
    update_p();
    old_rTr = new_rTr;
    get_current_program().profiler_print();
  }

  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      for (int k = 0; k < n; k++) {
        for (int d = 0; d < dim; d++) {
          p(d).val<float32>(i, j, k) = x(d).val<float32>(i, j, k);
        }
      }
    }
  }
  compute_Ap();
  auto residual = 0.0f;
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      for (int k = 0; k < n; k++) {
        for (int d = 0; d < dim; d++) {
          residual += std::abs(R[i][j][k][d] - Ap(d).val<float32>(i, j, k));
        }
      }
    }
  }
  TC_P(residual);
  auto difference = 0.0f;
  auto difference_max = 0.0f;
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      for (int k = 0; k < n; k++) {
        for (int d = 0; d < dim; d++) {
          difference += std::abs(X[i][j][k][d] - x(d).val<float32>(i, j, k));
          difference_max =
              std::max(difference_max,
                       std::abs(X[i][j][k][d] - x(d).val<float32>(i, j, k)));
        }
      }
    }
  }
  TC_P(difference);
  TC_P(difference_max);

  int gui_res = 512;
  GUI gui("FEM", Vector2i(gui_res + 200, gui_res), false);
  int gt = 0;
  int k = 0;
  gui.slider("z", k, 0, n - 1).slider("Ground truth", gt, 0, 1);

  int scale = gui_res / n;
  auto &canvas = gui.get_canvas();
  for (int frame = 1;; frame++) {
    for (int i = 0; i < gui_res - scale; i++) {
      for (int j = 0; j < gui_res - scale; j++) {
        real dx, dy, dz;
        if (!gt) {
          dx = x(0).val<float32>(i / scale, j / scale, k);
          dy = x(1).val<float32>(i / scale, j / scale, k);
          dz = x(2).val<float32>(i / scale, j / scale, k);
        } else {
          dx = X[i / scale][j / scale][k][0];
          dy = X[i / scale][j / scale][k][1];
          dz = X[i / scale][j / scale][k][2];
        }
        canvas.img[i][j] = Vector4(0.5f) + Vector4(dx, dy, dz, 0) * 0.5f;
      }
    }
    gui.update();
  }
};
TC_REGISTER_TASK(fem);

// FEM ret bounds: [-7.73062, 0.458078]

TC_NAMESPACE_END
