#include <taichi/lang.h>
#include <taichi/util.h>
#include <taichi/visual/gui.h>
#include <taichi/common/bit.h>
#include <taichi/system/profiler.h>
#include "svd.h"
TC_NAMESPACE_BEGIN
using namespace Tlang;
auto mpm_benchmark = []() {
  Program prog(Arch::gpu);
  prog.config.lower_access = false;
  constexpr int dim = 3, n = 256, grid_block_size = 4, n_particles = 775196;
  const real dt = 1e-5_f * 256 / n, dx = 1.0_f / n, inv_dx = 1.0_f / dx;
  auto particle_mass = 1.0_f, vol = 1.0_f, E = 1e4_f, nu = 0.3f;
  real mu = E / (2 * (1 + nu)), lambda = E * nu / ((1 + nu) * (1 - 2 * nu));
  auto f32 = DataType::f32;
  Vector particle_x(f32, dim), particle_v(f32, dim), grid_v(f32, dim);
  Matrix particle_F(f32, dim, dim), particle_C(f32, dim, dim);
  Global(grid_m, f32);
  Global(l, i32);
  Global(gravity_x, f32);
  int max_n_particles = 1024 * 1024;
  std::vector<Vector3> p_x;
  p_x.resize(n_particles);
  std::vector<float> benchmark_particles;
  auto f = fopen("dragon_particles.bin", "rb");
  TC_ASSERT_INFO(f, "./dragon_particles.bin not found");
  benchmark_particles.resize(n_particles * 3);
  std::fread(benchmark_particles.data(), sizeof(float), n_particles * 3, f);
  std::fclose(f);
  for (int i = 0; i < n_particles; i++) {
    for (int j = 0; j < dim; j++)
      p_x[i][j] = benchmark_particles[i * dim + j];
  }
  bool particle_SOA = false;
  layout([&]() {
    auto i = Index(0), j = Index(1), k = Index(2), p = Index(3);
    SNode *fork;
    if (!particle_SOA)
      fork = &root.dynamic(p, max_n_particles);
    auto place = [&](Expr &expr) {
      if (particle_SOA) {
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
    auto &block = root.dense({i, j, k}, n / grid_block_size).pointer();
    constexpr bool block_soa = true;
    if (block_soa) {
      block.dense({i, j, k}, grid_block_size).place(grid_v(0));
      block.dense({i, j, k}, grid_block_size).place(grid_v(1));
      block.dense({i, j, k}, grid_block_size).place(grid_v(2));
      block.dense({i, j, k}, grid_block_size).place(grid_m);
    } else {
      block.dense({i, j, k}, grid_block_size)
          .place(grid_v(0), grid_v(1), grid_v(2), grid_m);
    }
    block.dynamic(p, pow<dim>(grid_block_size) * 128).place(l);
    root.place(gravity_x);
  });
  Kernel(sort).def([&] {
    BlockDim(1024);
    For(particle_x(0), [&](Expr p) {
      auto node_coord = floor(particle_x[p] * inv_dx - 0.5_f);
      Append(l.parent(),
             (cast<int32>(node_coord(0)), cast<int32>(node_coord(1)),
              cast<int32>(node_coord(2))),
             p);
    });
  });
  Kernel(sort_print).def([&] {
    BlockDim(1024);
    For(particle_x(0), [&](Expr p) {
      auto node_coord = floor(particle_x[p] * inv_dx - 0.5_f);
      auto len = Var(Probe(
          l.parent(), (cast<int32>(node_coord(0)), cast<int32>(node_coord(1)),
                       cast<int32>(node_coord(2)))));
      Print(len);
      Assert(len == 0);
    });
  });
  Kernel(p2g_sorted).def([&] {
    BlockDim(128);
    Cache(0, grid_v(0));
    Cache(0, grid_v(1));
    Cache(0, grid_v(2));
    Cache(0, grid_m);
    For(l, [&](Expr i, Expr j, Expr k, Expr p_ptr) {
      auto p = Var(l[i, j, k, p_ptr]);
      auto x = Var(particle_x[p]), v = Var(particle_v[p]),
           C = Var(particle_C[p]);
      auto base_coord = floor(inv_dx * x - 0.5_f), fx = x * inv_dx - base_coord;
      Matrix F = Var(Matrix::identity(dim) + dt * C) * particle_F[p];
      particle_F[p] = F;
      Vector w[] = {Var(0.5_f * sqr(1.5_f - fx)), Var(0.75_f - sqr(fx - 1.0_f)),
                    Var(0.5_f * sqr(fx - 0.5_f))};
      auto svd = sifakis_svd(F);
      auto R = Var(std::get<0>(svd) * transposed(std::get<2>(svd)));
      auto sig = Var(std::get<1>(svd));
      auto J = Var(sig(0) * sig(1) * sig(2));
      auto cauchy = Var(2.0_f * mu * (F - R) * transposed(F) +
                        (Matrix::identity(3) * lambda) * (J - 1.0f) * J);
      auto affine =
          Var(particle_mass * C - (4 * inv_dx * inv_dx * dt * vol) * cauchy);
      int low = 0, high = 1;
      auto base_coord_i =
          AssumeInRange(cast<int32>(base_coord(0)), i, low, high);
      auto base_coord_j =
          AssumeInRange(cast<int32>(base_coord(1)), j, low, high);
      auto base_coord_k =
          AssumeInRange(cast<int32>(base_coord(2)), k, low, high);
      for (int a = 0; a < 3; a++)
        for (int b = 0; b < 3; b++)
          for (int c = 0; c < 3; c++) {
            auto dpos = dx * (Vector({a, b, c}).cast_elements<float32>() - fx);
            auto weight = w[a](0) * w[b](1) * w[c](2);
            auto node = (base_coord_i + a, base_coord_j + b, base_coord_k + c);
            Atomic(grid_v[node]) +=
                weight * (particle_mass * v + affine * dpos);
            Atomic(grid_m[node]) += weight * particle_mass;
          }
    });
  });
  Kernel(grid_op).def([&]() {
    For(grid_m, [&](Expr i, Expr j, Expr k) {
      auto v = Var(grid_v[i, j, k]);
      auto m = Var(grid_m[i, j, k]);
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
      v(2) = select(n - bound < k, min(v(2), Expr(0.0_f)), v(2));
      v(0) = select(i < bound, max(v(0), Expr(0.0_f)), v(0));
      v(2) = select(k < bound, max(v(2), Expr(0.0_f)), v(2));
      If(j < bound, [&] { v(1) = max(v(1), Expr(0.0_f)); });
      grid_v[i, j, k] = v;
    });
  });
  Kernel(g2p).def([&]() {
    BlockDim(128);
    Cache(0, grid_v(0));
    Cache(0, grid_v(1));
    Cache(0, grid_v(2));
    For(l, [&](Expr i, Expr j, Expr k, Expr p_ptr) {
      auto p = Var(l[i, j, k, p_ptr]);
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
      int low = 0, high = 1;
      auto base_coord_i =
          AssumeInRange(cast<int32>(base_coord(0)), i, low, high);
      auto base_coord_j =
          AssumeInRange(cast<int32>(base_coord(1)), j, low, high);
      auto base_coord_k =
          AssumeInRange(cast<int32>(base_coord(2)), k, low, high);
      for (int p = 0; p < 3; p++)
        for (int q = 0; q < 3; q++)
          for (int r = 0; r < 3; r++) {
            auto dpos = Vector({p, q, r}).cast_elements<float32>() - fx;
            auto weight = w[p](0) * w[q](1) * w[r](2);
            auto wv =
                weight *
                grid_v[base_coord_i + p, base_coord_j + q, base_coord_k + r];
            v += wv;
            C += outer_product(wv, dpos);
          }
      particle_C[p] = (4 * inv_dx) * C;
      particle_v[p] = v;
      particle_x[p] = x + dt * v;
    });
  });
  auto block_id = [&](Vector3 x) {
    auto xi = (x * inv_dx - Vector3(0.5f)).floor().template cast<int>() /
              Vector3i(grid_block_size);
    return xi.x * pow<2>(n / grid_block_size) + xi.y * n / grid_block_size +
           xi.z;
  };
  std::sort(p_x.begin(), p_x.end(),
            [&](Vector3 a, Vector3 b) { return block_id(a) < block_id(b); });
  for (int i = 0; i < n_particles; i++) {
    for (int d = 0; d < dim; d++) {
      particle_x(d).val<float32>(i) = p_x[i][d];
    }
    particle_v(0).val<float32>(i) = 0._f;
    particle_v(1).val<float32>(i) = -3.0_f;
    particle_v(2).val<float32>(i) = 0._f;
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
  // Visualization omitted...
};
TC_REGISTER_TASK(mpm_benchmark);
TC_NAMESPACE_END
