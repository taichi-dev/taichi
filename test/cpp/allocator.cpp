#include <taichi/lang.h>
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
      For(0, n, [&](Expr i) {
        For(0, i, [&](Expr j) { Activate(x.snode(), {i, j}); });
      });
    })();

    kernel([&]() {
      For(0, n, [&](Expr i) { For(0, i, [&](Expr j) { x[i, j] = i + j; }); });
    })();

    auto stat = x.parent().parent().snode()->stat();
    TC_CHECK(stat.num_resident_blocks == n - 1);
    for (int i = 0; i < n; i++) {
      for (int j = 0; j < i; j++) {
        TC_CHECK(x.val<int>(i, j) == i + j);
      }
    }
    x.parent().parent().snode()->clear_data_and_deactivate();
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
  Program prog(Arch::gpu);
  CoreState::set_trigger_gdb_when_crash(true);

  constexpr int n = 256;
  const real dx = 1.0_f / n, inv_dx = 1.0_f / dx;

  constexpr int dim = 3;

  auto f32 = DataType::f32;
  auto i32 = DataType::i32;
  int grid_block_size = 4;

  Vector particle_x(i32, dim);

  Global(grid_m, f32);
  Global(flag, i32);

  int max_n_particles = 1024 * 1024;

  int n_particles = 0;
  std::vector<float> benchmark_particles;
  std::vector<Vector3> p_x;
  n_particles = max_n_particles;
  p_x.resize(n_particles);
  for (int i = 0; i < n_particles; i++) {
    Vector3 offset = Vector3::rand() - Vector3(0.5_f);
    p_x[i] = Vector3(0.5_f) + offset * 0.7f;
  }

  TC_ASSERT(n_particles <= max_n_particles);

  auto i = Index(0), j = Index(1), k = Index(2);
  auto p = Index(3);

  layout([&]() {
    auto &fork = root.dynamic(p, max_n_particles);
    for (int i = 0; i < dim; i++) {
      fork.place(particle_x(i));
    }

    root.dense(i, max_n_particles).place(flag);

    TC_ASSERT(n % grid_block_size == 0);
    root.dense({i, j, k}, n / grid_block_size)
        .pointer()
        .dense({i, j, k}, grid_block_size)
        .place(grid_m);
  });

  TC_ASSERT(bit::is_power_of_two(n));

  Kernel(sort).def([&] {
    BlockDim(256);
    For(particle_x(0), [&](Expr p) {
      grid_m[particle_x[p]] = 1;
      Atomic(flag[p]) += 1;
    });
    // For(0, max_n_particles, [&](Expr p) { grid_m[particle_x[p]] = 1; });
  });

  for (int i = 0; i < n_particles; i++) {
    for (int d = 0; d < dim; d++) {
      particle_x(d).val<int32>(i) = p_x[i][d] * inv_dx;
    }
  }

  int last_nb = -1;
  for (int i = 0; i < 2048; i++) {
    for (int k = 0; k < max_n_particles; k++)
      flag.val<int32>(k) = 0;
    grid_m.parent().parent().snode()->clear_data_and_deactivate();
    sort();
    prog.synchronize();
    for (int k = 0; k < max_n_particles; k++) {
      TC_CHECK(flag.val<int32>(k) == 1);
    }
    auto stat = grid_m.parent().parent().snode()->stat();
    int nb = stat.num_resident_blocks;
    if (last_nb == -1) {
      last_nb = nb;
      TC_P(last_nb);
    } else {
      if (last_nb != nb) {
        TC_P(i);
      }
      TC_CHECK(last_nb == nb);
    }
  }
};

TC_TEST("struct_for") {
  Program prog(Arch::gpu);
  CoreState::set_trigger_gdb_when_crash(true);

  constexpr int n = 256;
  const real dx = 1.0_f / n, inv_dx = 1.0_f / dx;

  constexpr int dim = 3;

  auto f32 = DataType::f32;
  auto i32 = DataType::i32;
  int grid_block_size = 4;

  Vector particle_x(i32, dim);

  Global(grid_m, f32);
  Global(flag, i32);

  int max_n_particles = 1024 * 1024;

  int n_particles = 0;
  std::vector<float> benchmark_particles;
  std::vector<Vector3> p_x;
  n_particles = max_n_particles;
  p_x.resize(n_particles);
  for (int i = 0; i < n_particles; i++) {
    Vector3 offset = Vector3::rand() - Vector3(0.5_f);
    p_x[i] = Vector3(0.5_f) + offset * 0.7f;
  }

  TC_ASSERT(n_particles <= max_n_particles);

  auto i = Index(0), j = Index(1), k = Index(2);
  auto p = Index(3);

  layout([&]() {
    auto &fork = root.dynamic(p, max_n_particles);
    for (int i = 0; i < dim; i++) {
      fork.place(particle_x(i));
    }
    root.dense(i, max_n_particles).place(flag);
  });

  Kernel(sort).def([&] {
    BlockDim(256);
    For(particle_x(0), [&](Expr p) { Atomic(flag[p]) += 1; });
  });

  for (int i = 0; i < n_particles; i++) {
    particle_x(0).val<int32>(i) = 1;
  }

  for (int i = 0; i < 2048; i++) {
    for (int k = 0; k < max_n_particles; k++)
      flag.val<int32>(k) = 0;
    sort();
    int zero_count = 0, big_count = 0;
    for (int k = 0; k < max_n_particles; k++) {
      if (flag.val<int32>(k) < 1) {
        zero_count += 1;
      }
      if (flag.val<int32>(k) > 1) {
        big_count += 1;
      }
    }
    TC_CHECK(zero_count == 0);
    TC_CHECK(big_count == 0);
  }
};

TLANG_NAMESPACE_END
