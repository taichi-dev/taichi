#include <algorithm>
#include <random>
#include <taichi/lang.h>
#include <taichi/util.h>
#include <taichi/visual/gui.h>
#include <taichi/common/bit.h>
#include <taichi/system/profiler.h>
#include <taichi/visualization/particle_visualization.h>
#include "svd.h"

TC_NAMESPACE_BEGIN

using namespace Tlang;

auto diff_mpm = [](std::vector<std::string> cli_param) {
  CoreState::set_trigger_gdb_when_crash(true);
  Program prog(Arch::gpu);

  auto param = parse_param(cli_param);
  bool particle_soa = param.get("particle_soa", false);
  TC_P(particle_soa);
  bool block_soa = param.get("block_soa", true);
  TC_P(block_soa);
  bool use_cache = param.get("use_cache", true);
  TC_P(use_cache);
  bool initial_reorder = param.get("initial_reorder", true);
  TC_P(initial_reorder);
  bool initial_shuffle = param.get("initial_shuffle", false);
  TC_P(initial_shuffle);
  prog.config.lower_access = param.get("lower_access", false);
  int stagger = param.get("stagger", true);
  TC_P(stagger);

  constexpr int dim = 2, n = 512, grid_block_size = 8, n_particles = 1024 * 1024;
  const real dt = 1e-5_f * 512 / n, dx = 1.0_f / n, inv_dx = 1.0_f / dx;
  auto particle_mass = 1.0_f, vol = 1.0_f, E = 1e4_f, nu = 0.3f;
  real mu = E / (2 * (1 + nu)), lambda = E * nu / ((1 + nu) * (1 - 2 * nu));
  auto f32 = DataType::f32;

  Vector particle_x(f32, dim), particle_v(f32, dim), grid_v(f32, dim);
  Matrix particle_F(f32, dim, dim), particle_C(f32, dim, dim);

  Global(grid_m, f32);
  Global(l, i32);
  Global(gravity_x, f32);
  int max_n_particles = 1024 * 1024;
  std::vector<Vector2> p_x;
  p_x.resize(n_particles);

  for (int i = 0; i < n_particles; i++) {
    for (int j = 0; j < dim; j++)
      p_x[i][j] = rand() * 0.4f + 0.3f;
  }

  layout([&]() {
    auto space = dim == 2 ? Indices(0, 1) : Indices(0, 1, 2);
    auto p = Index(dim);
    SNode *fork = nullptr;
    if (!particle_soa)
      fork = &root.dynamic(p, max_n_particles);
    auto place = [&](Expr &expr) {
      if (particle_soa) {
        root.dynamic(p, max_n_particles).place(expr);
      } else {
        fork->place(expr);
      }
    };
    for (int i = 0; i < dim; i++)
      for (int j = 0; j < dim; j++)
        place(particle_F(i, j));
    for (int i = 0; i < dim; i++)
      for (int j = 0; j < dim; j++)
        place(particle_C(i, j));
    for (int i = 0; i < dim; i++)
      place(particle_x(i));
    for (int i = 0; i < dim; i++)
      place(particle_v(i));
    TC_ASSERT(n % grid_block_size == 0);
    auto &block = root.dense(space, n / grid_block_size).pointer();
    if (block_soa) {
      for (int i = 0; i < dim; i++)
        block.dense(space, grid_block_size).place(grid_v(i));
      block.dense(space, grid_block_size).place(grid_m);
    } else {
      block.dense(space, grid_block_size).place(grid_v).place(grid_m);
    }

    block.dynamic(p, pow<dim>(grid_block_size) * 64).place(l);
    root.place(gravity_x);
  });
  Kernel(sort).def([&] {
    BlockDim(1024);
    For(particle_x(0), [&](Expr p) {
      auto node_coord = floor(particle_x[p] * inv_dx + (0.5_f - stagger));
      Append(l.parent(),
             (cast<int32>(node_coord(0)), cast<int32>(node_coord(1))), p);
    });
  });
  Kernel(p2g_sorted).def([&] {
    BlockDim(128);
    if (use_cache) {
      for (int i = 0; i < dim; i++)
        Cache(0, grid_v(i));
      Cache(0, grid_m);
    }
    For(l, [&](Expr i, Expr j, Expr p_ptr) {
      auto p = Var(l[i, j, p_ptr]);
      auto x = Var(particle_x[p]), v = Var(particle_v[p]),
           C = Var(particle_C[p]);
      auto base_coord = floor(inv_dx * x - 0.5_f), fx = x * inv_dx - base_coord;
      Matrix F = Var(Matrix::identity(dim) + dt * C) * particle_F[p];
      particle_F[p] = F;
      Vector w[] = {Var(0.5_f * sqr(1.5_f - fx)), Var(0.75_f - sqr(fx - 1.0_f)),
                    Var(0.5_f * sqr(fx - 0.5_f))};

      // polar 2d
      auto R = Matrix(2, 2);
      {
        auto x = Var(F(0, 0) + F(1, 1));
        auto y = Var(F(1, 0) - F(0, 1));
        auto scale = Var(Expr(1.0f) / sqrt(x * x + y * y));
        auto c = Var(x * scale);
        auto s = Var(y * scale);
        R(0, 0) = c;
        R(0, 1) = -s;
        R(1, 0) = s;
        R(1, 1) = c;
      }
      auto S = Var(transposed(R) * F);

      auto J = Var(S(0, 0) * S(1, 1));
      auto cauchy = Var(2.0_f * mu * (F - R) * transposed(F) +
                        (Matrix::identity(2) * lambda) * (J - 1.0f) * J);
      auto affine =
          Var(particle_mass * C - (4 * inv_dx * inv_dx * dt * vol) * cauchy);
      int low = -1 + stagger, high = stagger;
      auto base_coord_i =
          AssumeInRange(cast<int32>(base_coord(0)), i, low, high);
      auto base_coord_j =
          AssumeInRange(cast<int32>(base_coord(1)), j, low, high);
      for (int a = 0; a < 3; a++)
        for (int b = 0; b < 3; b++) {
          auto dpos =
              dx * (Vector::from_list({a, b}).cast_elements<float32>() - fx);
          auto weight = w[a](0) * w[b](1);
          auto node = (base_coord_i + a, base_coord_j + b);
          Atomic(grid_v[node]) += weight * (particle_mass * v + affine * dpos);
          Atomic(grid_m[node]) += weight * particle_mass;
        }
    });
  });
  Kernel(grid_op).def([&]() {
    For(grid_m, [&](Expr i, Expr j) {
      auto v = Var(grid_v[i, j]);
      auto m = Var(grid_m[i, j]);
      int bound = 8;
      If(m > 0.0f, [&]() {
        auto inv_m = Var(1.0f / m);
        v *= inv_m;
        auto f = gravity_x[Expr(0)];
        v(1) += dt * (-1000_f + abs(f));
        v(0) += dt * f;
      });
      v(0) = select(n - bound < i, min(v(0), Expr(0.0_f)), v(0));
      v(1) = select(n - bound < j, min(v(1), Expr(0.0_f)), v(1));
      v(0) = select(i < bound, max(v(0), Expr(0.0_f)), v(0));
      If(j < bound, [&] { v(1) = max(v(1), Expr(0.0_f)); });
      grid_v[i, j] = v;
    });
  });
  Kernel(g2p).def([&]() {
    BlockDim(128);
    if (use_cache) {
      for (int i = 0; i < dim; i++)
        Cache(0, grid_v(i));
    }
    For(l, [&](Expr i, Expr j, Expr p_ptr) {
      auto p = Var(l[i, j, p_ptr]);
      auto x = Var(particle_x[p]), v = Var(Vector(dim)),
           C = Var(Matrix(dim, dim));
      for (int i = 0; i < dim; i++) {
        v(i) = Expr(0.0_f);
        for (int j = 0; j < dim; j++) {
          C(i, j) = Expr(0.0_f);
        }
      }
      auto base_coord = floor(inv_dx * x - 0.5_f);
      auto fx = x * inv_dx - base_coord;
      Vector w[] = {Var(0.5_f * sqr(1.5_f - fx)), Var(0.75_f - sqr(fx - 1.0_f)),
                    Var(0.5_f * sqr(fx - 0.5_f))};
      int low = -1 + stagger, high = stagger;
      auto base_coord_i =
          AssumeInRange(cast<int32>(base_coord(0)), i, low, high);
      auto base_coord_j =
          AssumeInRange(cast<int32>(base_coord(1)), j, low, high);
      for (int p = 0; p < 3; p++)
        for (int q = 0; q < 3; q++) {
          auto dpos = Vector::from_list({p, q}).cast_elements<float32>() - fx;
          auto weight = w[p](0) * w[q](1);
          auto wv = weight * grid_v[base_coord_i + p, base_coord_j + q];
          v += wv;
          C += outer_product(wv, dpos);
        }
      particle_C[p] = (4 * inv_dx) * C;
      particle_v[p] = v;
      particle_x[p] = x + dt * v;
    });
  });
  auto block_id = [&](Vector2 x) {
    auto xi = (x * inv_dx - Vector2(0.5f)).floor().template cast<int>() /
              Vector2i(grid_block_size);
    return xi.x * pow<2>(n / grid_block_size) + xi.y * n / grid_block_size;
  };
  if (initial_reorder) {
    std::sort(p_x.begin(), p_x.end(),
              [&](Vector2 a, Vector2 b) { return block_id(a) < block_id(b); });
  }
  if (initial_shuffle) {
    std::random_device rng;
    std::mt19937 urng(rng());
    std::shuffle(p_x.begin(), p_x.end(), urng);
  }
  for (int i = 0; i < n_particles; i++) {
    for (int d = 0; d < dim; d++) {
      particle_x(d).val<float32>(i) = p_x[i][d];
      particle_v(d).val<float32>(i) = d == 1 ? -3 : 0;
    }
    for (int p = 0; p < dim; p++)
      for (int q = 0; q < dim; q++)
        particle_F(p, q).val<float32>(i) = (p == q);
  }
  auto simulate_frame = [&]() {
    grid_m.parent().parent().snode()->clear_data_and_deactivate();
    auto t = Time::get_time();
    for (int f = 0; f < 200; f++) {
      grid_m.parent().parent().snode()->clear_data();
      sort();
      p2g_sorted();
      grid_op();
      g2p();
    }
    prog.profiler_print();
    auto ms_per_substep = (Time::get_time() - t) / 200 * 1000;
    TC_P(ms_per_substep);
  };

  // Visualization
  Vector2i cam_res(1024, 1024);
  GUI gui("MPM", cam_res);
  auto renderer = create_instance_unique<ParticleRenderer>("shadow_map");

  auto &canvas = gui.get_canvas();
  for (int frame = 1;; frame++) {
    simulate_frame();
    canvas.clear(Vector4(1.0f));
    for (int i = 0; i < n_particles; i++) {
      auto x = particle_x(0).val<float32>(i), y = particle_x(1).val<float32>(i);
      canvas.circle(x, y).radius(2).color(Vector4(0.5, 0.5, 0.5, 1.f));
    }
    print_profile_info();
    gui.update();
  }
};
TC_REGISTER_TASK(diff_mpm);

TC_NAMESPACE_END
