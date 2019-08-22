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
  constexpr bool highres = true;
  CoreState::set_trigger_gdb_when_crash(true);

  constexpr int n = 256;
  const real dx = 1.0_f / n, inv_dx = 1.0_f / dx;

  constexpr int dim = 3;

  auto f32 = DataType::f32;
  int grid_block_size = 4;

  Vector particle_x(f32, dim);
  Global(l, i32);

  Vector grid_v(f32, dim);
  Global(grid_m, f32);

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

  layout([&]() {
    SNode *fork = nullptr;
    fork = &root.dynamic(p, max_n_particles);
    for (int i = 0; i < dim; i++) {
      fork->place(particle_x(i));
    }

    TC_ASSERT(n % grid_block_size == 0);
    auto &block = root.dense({i, j, k}, n / grid_block_size).pointer();

    block.dense({i, j, k}, grid_block_size)
        .place(grid_v(0), grid_v(1), grid_v(2), grid_m);

    block.dynamic(p, pow<dim>(grid_block_size) * 64).place(l);
  });

  TC_ASSERT(bit::is_power_of_two(n));

  Kernel(sort).def([&] {
    BlockDim(256);
    For(particle_x(0), [&](Expr p) {
      auto node_coord = floor(particle_x[p] * inv_dx - 0.5_f);
      Append(l.parent(),
             (cast<int32>(node_coord(0)), cast<int32>(node_coord(1)),
              cast<int32>(node_coord(2))),
             p);
    });
  });

  for (int i = 0; i < n_particles; i++) {
    for (int d = 0; d < dim; d++) {
      particle_x(d).val<float32>(i) = p_x[i][d];
    }
  }

  int last_nb = -1;
  for (int i = 0; i < 128; i++) {
    grid_m.parent().parent().snode()->clear_data_and_deactivate();
    sort();
    auto stat = grid_m.parent().parent().snode()->stat();
    int nb = stat.num_resident_blocks;
    if (last_nb == -1) {
      last_nb = nb;
    } else {
      TC_CHECK(last_nb == nb);
    }
  }
};

TLANG_NAMESPACE_END
