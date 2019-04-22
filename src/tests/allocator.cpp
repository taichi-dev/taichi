#include <cuda_runtime.h>
#include "../tlang.h"
#include <taichi/testing.h>
#include <numeric>

TLANG_NAMESPACE_BEGIN

TC_TEST("gpu_gc_basics") {
  for (auto arch : {Arch::gpu}) {
    int n = 32;
    Program prog(arch);

    Global(x, i32);
    layout([&]() {
      auto i = Index(0);
      auto j = Index(1);
      root.dense(i, n).pointer().dense(j, n).place(x);
    });

    kernel([&]() {
      Declare(i);
      Declare(j);
      For(i, 0, n, [&] { For(j, 0, i, [&] { Activate(x.snode(), {i, j}); }); });
    })();

    kernel([&]() {
      Declare(i);
      Declare(j);
      For(i, 0, n, [&] { For(j, 0, i, [&] { x[i, j] = i + j; }); });
    })();

    auto stat = x.parent().parent().snode()->stat();
    TC_CHECK(stat.num_resident_blocks == n - 1);
    for (int i = 0; i < n; i++) {
      for (int j = 0; j < i; j++) {
        TC_CHECK(x.val<int>(i, j) == i + j);
      }
    }
    x.parent().parent().snode()->clear(1);
    stat = x.parent().parent().snode()->stat();
    TC_CHECK(stat.num_resident_blocks == 0);
    TC_CHECK(stat.num_recycled_blocks == 0);

    for (int i = 0; i < n; i++) {
      for (int j = 0; j < i; j++) {
        TC_CHECK(x.val<int>(i, j) == 0);
      }
    }
  }
};

TC_TEST("parallel_particle_sort") {
  bool benchmark_dragon = false;
  Program prog(Arch::gpu);
  constexpr bool highres = true;
  CoreState::set_trigger_gdb_when_crash(true);

  constexpr int n = highres ? 256 : 128;  // grid_resolution
  const real dt = 1e-5_f * 256 / n, dx = 1.0_f / n, inv_dx = 1.0_f / dx;
  auto particle_mass = 1.0_f, vol = 1.0_f;
  auto E = 1e4_f, nu = 0.3f;
  real mu_0 = E / (2 * (1 + nu)), lambda_0 = E * nu / ((1 + nu) * (1 - 2 * nu));

  auto friction_angle = 45._f;
  real sin_phi = std::sin(friction_angle / 180._f * real(3.141592653));
  auto alpha = std::sqrt(2._f / 3._f) * 2._f * sin_phi / (3._f - sin_phi);

  constexpr int dim = 3;

  auto f32 = DataType::f32;
  int grid_block_size = 4;
  int particle_block_size = 1;

  Vector particle_x(f32, dim), particle_v(f32, dim);
  Matrix particle_F(f32, dim, dim), particle_C(f32, dim, dim);
  Global(l, i32);
  Global(particle_J, f32);
  Global(gravity_x, f32);

  Vector grid_v(f32, dim);
  Global(grid_m, f32);

  bool sorted = true;

  int max_n_particles = 1024 * 1024;

  int n_particles = 0;
  std::vector<float> benchmark_particles;
  std::vector<Vector3> p_x;
  n_particles = max_n_particles / (highres ? 1 : 8);
  p_x.resize(n_particles);
  for (int i = 0; i < n_particles; i++) {
    Vector3 offset = Vector3::rand() - Vector3(0.5_f);
    while (offset.length() > 0.5f) {
      offset = Vector3::rand() - Vector3(0.5_f);
    }
    p_x[i] = Vector3(0.5_f) + offset * 0.3f;
  }

  TC_ASSERT(n_particles <= max_n_particles);

  auto i = Index(0), j = Index(1), k = Index(2);
  auto p = Index(3);

  bool SOA = false;

  layout([&]() {
    SNode *fork;
    if (!SOA)
      fork = &root.dynamic(p, max_n_particles);
    auto place = [&](Expr &expr) {
      if (SOA) {
        if (particle_block_size == 1) {
          root.dynamic(p, max_n_particles).place(expr);
        } else {
          TC_ASSERT(max_n_particles % particle_block_size == 0);
          root.dense(p, max_n_particles / particle_block_size)
              .dense(p, particle_block_size)
              .place(expr);
        }
      } else {
        fork->place(expr);
      }
    };
    for (int i = 0; i < dim; i++) {
      for (int j = 0; j < dim; j++) {
        place(particle_F(i, j));
      }
    }
    for (int i = 0; i < dim; i++) {
      for (int j = 0; j < dim; j++) {
        place(particle_C(i, j));
      }
    }
    for (int i = 0; i < dim; i++) {
      place(particle_x(i));
    }
    for (int i = 0; i < dim; i++)
      place(particle_v(i));
    place(particle_J);

    TC_ASSERT(n % grid_block_size == 0);
    auto &block = root.dense({i, j, k}, n / grid_block_size).pointer();
    constexpr bool block_soa = false;

    if (block_soa) {
      block.dense({i, j, k}, grid_block_size).place(grid_v(0));
      block.dense({i, j, k}, grid_block_size).place(grid_v(1));
      block.dense({i, j, k}, grid_block_size).place(grid_v(2));
      block.dense({i, j, k}, grid_block_size).place(grid_m);
    } else {
      block.dense({i, j, k}, grid_block_size)
          .place(grid_v(0), grid_v(1), grid_v(2), grid_m);
    }

    block.dynamic(p, pow<dim>(grid_block_size) * 64).place(l);

    root.place(gravity_x);
  });

  TC_ASSERT(bit::is_power_of_two(n));

  Kernel(sort).def([&] {
    Declare(p);
    BlockDim(256);
    For(p, particle_x(0), [&] {
      auto node_coord = floor(particle_x[p] * inv_dx - 0.5_f);
      Append(l.parent(),
             (cast<int32>(node_coord(0)), cast<int32>(node_coord(1)),
              cast<int32>(node_coord(2))),
             p);
    });
  });

  auto check_fluctuation = [&] {
    int last_nb = -1;
    while (1) {
      grid_m.parent().parent().snode()->clear(1);
      sort();
      auto stat = grid_m.parent().parent().snode()->stat();
      int nb = stat.num_resident_blocks;
      TC_P(nb);
      if (last_nb == -1) {
        last_nb = nb;
      } else {
        TC_ASSERT(last_nb == nb);
      }
    }
  };

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
  }

  check_fluctuation();
};

TLANG_NAMESPACE_END
