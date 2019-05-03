#include <taichi/testing.h>
#include "../tlang.h"

TLANG_NAMESPACE_BEGIN

TC_TEST("snode") {
  Program prog(Arch::x86_64);

  auto i = Index(0);
  Global(u, i32);

  int n = 128;

  // All data structure originates from a "root", which is a forked node.
  prog.layout([&] { root.dense(i, n).place(u); });

  for (int i = 0; i < n; i++) {
    u.val<int32>(i) = i + 1;
  }

  for (int i = 0; i < n; i++) {
    TC_CHECK_EQUAL(u.val<int32>(i), i + 1, 0);
  }
}

TC_TEST("snode_loop") {
  for (auto arch : {Arch::x86_64, Arch::gpu}) {
    Program prog(arch);
    CoreState::set_trigger_gdb_when_crash(true);
    prog.config.print_ir = true;

    auto i = Index(0);
    Global(u, i32);

    int n = 8192;

    // All data structure originates from a "root", which is a forked node.
    prog.layout([&] { root.dense(i, n).place(u); });

    kernel([&] {
      Declare(i);
      BlockDim(256);
      For(i, u, [&] { u[i] = i * 2; });
    })();

    for (int i = 0; i < n; i++) {
      TC_CHECK_EQUAL(u.val<int32>(i), i * 2, 0);
    }
  }
}

TC_TEST("snode_loop2") {
  for (auto arch : {Arch::x86_64, Arch::gpu}) {
    Program prog(arch);
    CoreState::set_trigger_gdb_when_crash(true);
    prog.config.print_ir = true;

    auto i = Index(0), j = Index(1);
    Global(u, i32);
    Global(v, i32);

    int n = 128;

    // All data structure originates from a "root", which is a forked node.
    prog.layout([&] {
      root.dense(i, n).place(u);
      root.dense(j, n).place(v);
    });

    TC_ASSERT(
        u.cast<GlobalVariableExpression>()->snode->physical_index_position[0] ==
        0);
    TC_ASSERT(
        v.cast<GlobalVariableExpression>()->snode->physical_index_position[0] ==
        1);

    kernel([&] {
      Declare(i);
      For(i, u, [&] { u[i] = i * 2; });
    })();

    kernel([&] {
      Declare(j);
      For(j, v, [&] { v[j] = j * 3; });
    })();

    for (int i = 0; i < n; i++) {
      TC_CHECK_EQUAL(u.val<int32>(i), i * 2, 0);
      TC_CHECK_EQUAL(v.val<int32>(i), i * 3, 0);
    }
  }
}

TC_TEST("2d_blocked_array") {
  int n = 8, block_size = 4;

  for (auto arch : {Arch::x86_64, Arch::gpu})
    for (auto blocked : {false, true}) {
      Program prog(arch);

      Global(a, i32);
      Global(b, i32);

      layout([&] {
        auto i = Index(0);
        auto j = Index(1);
        if (blocked) {
          TC_ASSERT(n % block_size == 0);
          root.dense({i, j}, {n / block_size, n * 2 / block_size})
              .dense({i, j}, {block_size, block_size})
              .place(a, b);
        } else {
          root.dense({i, j}, {n, n * 2}).place(a);
          root.dense({i, j}, {n, n * 2}).place(b);
        }
      });

      for (int i = 0; i < n; i++) {
        for (int j = 0; j < n * 2; j++) {
          a.val<int32>(i, j) = i + j * 3;
        }
      }

      kernel(
          [&]() { For(a, [&](Expr i, Expr j) { b[i, j] = a[i, j] + i; }); })();

      for (int i = 0; i < n; i++) {
        for (int j = 0; j < n * 2; j++) {
          TC_CHECK(a.val<int32>(i, j) == i + j * 3);
          TC_CHECK(b.val<int32>(i, j) == i * 2 + j * 3);
        }
      }
    }
}

TC_TEST("2d_blocked_array_vec") {
  int n = 8, block_size = 4;

  for (auto arch : {Arch::x86_64})
    for (auto blocked : {false, true}) {
      Program prog(arch);
      prog.config.print_ir = true;

      Global(a, i32);
      Global(b, i32);

      layout([&] {
        auto i = Index(0);
        auto j = Index(1);
        if (blocked) {
          TC_ASSERT(n % block_size == 0);
          root.dense({i, j}, {n / block_size, n * 2 / block_size})
              .dense({i, j}, {block_size, block_size})
              .place(a, b);
        } else {
          root.dense({i, j}, {n, n * 2}).place(a);
          root.dense({i, j}, {n, n * 2}).place(b);
        }
      });

      for (int i = 0; i < n; i++) {
        for (int j = 0; j < n * 2; j++) {
          a.val<int32>(i, j) = i + j * 3;
        }
      }

      kernel([&]() {
        Vectorize(block_size);
        For(a, [&](Expr i, Expr j) { b[i, j] = a[i, j] + i; });
      })();

      for (int i = 0; i < n; i++) {
        for (int j = 0; j < n * 2; j++) {
          TC_CHECK(a.val<int32>(i, j) == i + j * 3);
          TC_CHECK(b.val<int32>(i, j) == i * 2 + j * 3);
        }
      }
    }
}

TC_TEST("loop_over_blocks") {
  int n = 64, block_size = 4;

  for (auto arch : {Arch::x86_64, Arch::gpu}) {
    Program prog(arch);

    Global(a, i32);
    Global(sum_i, i32);
    Global(sum_j, i32);

    layout([&] {
      auto ij = Indices(0, 1);
      TC_ASSERT(n % block_size == 0);
      root.dense(ij, n / block_size)
          .dense(ij, {block_size, block_size * 2})
          .place(a);
      root.place(sum_i, sum_j);
    });

    kernel([&]() {
      For(a.parent(), [&](Expr i, Expr j) {
        Atomic(sum_i[Expr(0)]) += i;
        Atomic(sum_j[Expr(0)]) += j;
      });
    })();

    int sum_i_gt = 0;
    int sum_j_gt = 0;
    for (int i = 0; i < n; i += block_size) {
      for (int j = 0; j < n * 2; j += block_size * 2) {
        sum_i_gt += i;
        sum_j_gt += j;
      }
    }
    TC_CHECK(sum_i.val<int32>() == sum_i_gt);
    TC_CHECK(sum_j.val<int32>() == sum_j_gt);
  }
}

#if (0)

TC_TEST("spmv") {
  initialize_benchmark();
  int n = 8192;
  int band = 256;
  int k = 128;
  TC_ASSERT(k <= band);
  int m = n * k;

  Eigen::SparseMatrix<float32, Eigen::RowMajor> M(n, n);
  Eigen::VectorXf V(n), Vret(n);

  Program prog;
  prog.config.simd_width = 4;
  prog.config.external_optimization_level = 4;

  auto result = var<float32>();
  auto mat_col = var<int32>();
  auto mat_row = var<int32>();
  auto mat_val = var<float32>();
  auto vec_val = var<float32>();
  auto compressed_col = var<int32>();
  auto compressed_val = var<float32>();

  auto i = ind(), j = ind(), p = ind();

  std::vector<Eigen::Triplet<float32>> entries;

  layout([&] {
    // indirect puts an int32
    root.dense(p, m).place(mat_row);
    root.dense(p, m).place(mat_col);
    root.dense(p, m).place(mat_val);
    auto &mat = root.dense(i, n).multi_threaded();
    mat.dynamic(j, k).place(compressed_col);
    mat.dynamic(j, k).place(compressed_val);
    root.dense(i, n).place(vec_val);
    root.dense(i, n).place(result);
  });

  auto populate = kernel(mat_row, [&]() {
    touch(compressed_col, mat_row[p], mat_col[p]);
    touch(compressed_val, mat_row[p], mat_val[p]);
  });

  auto matvecmul = kernel(compressed_col, [&]() {
    auto entry = compressed_val[i, j] * vec_val[compressed_col[i, j]];
    reduce(result[i], entry);
  });

  for (int i = 0; i < n; i++) {
    std::set<int> cols;
    for (int j = 0; j < k; j++) {
      int col;
      while (true) {
        col = std::rand() % (2 * band) - band + i;
        if (col < 0 || col >= n)
          continue;
        if (cols.find(col) == cols.end()) {
          break;
        }
      }
      cols.insert(col);
      auto val = rand();
      entries.push_back({i, col, val});
      mat_row.val<int32>(i * k + j) = i;
      mat_col.val<int32>(i * k + j) = col;
      mat_val.val<float32>(i * k + j) = val;
    }
    auto val = rand();
    V(i) = val;
    vec_val.val<float32>(i) = val;
  }

  TC_TIME(M.setFromTriplets(entries.begin(), entries.end()));

  TC_TIME(populate());

  int T = 1;
  for (int i = 0; i < T; i++) {
    TC_TIME(matvecmul());
  }

  TC_P(n = Eigen::nbThreads());
  for (int i = 0; i < T; i++) {
    TC_TIME(Vret = M * V);
  }

  for (int i = 0; i < n; i++) {
    TC_CHECK_EQUAL(Vret(i), result.val<float32>(i) / T, 1e-3_f);
  }
}

TC_TEST("spmv_dynamic") {
  initialize_benchmark();
  int n = 8192;
  int band = 256;
  int k = 128;
  TC_ASSERT(k <= band);
  int m = n * k;

  Eigen::SparseMatrix<float32, Eigen::RowMajor> M(n, n);
  Eigen::VectorXf V(n), Vret(n);

  Program prog;
  prog.config.external_optimization_level = 4;

  auto result = var<float32>();
  auto mat_col = var<int32>();
  auto mat_row = var<int32>();
  auto mat_val = var<float32>();
  auto vec_val = var<float32>();
  auto compressed_col = var<int32>();
  auto compressed_val = var<float32>();

  auto i = ind(), j = ind(), p = ind();

  std::vector<Eigen::Triplet<float32>> entries;

  layout([&] {
    // indirect puts an int32
    root.dynamic(p, m).place(mat_row);
    root.dynamic(p, m).place(mat_col);
    root.dynamic(p, m).place(mat_val);
    auto &mat = root.dense(i, n).multi_threaded();
    mat.dynamic(j, k).place(compressed_col);
    mat.dynamic(j, k).place(compressed_val);
    root.dense(i, n).place(vec_val);
    root.dense(i, n).place(result);
  });

  auto populate = kernel(mat_row, [&]() {
    touch(compressed_col, mat_row[p], mat_col[p]);
    touch(compressed_val, mat_row[p], mat_val[p]);
  });

  auto matvecmul = kernel(compressed_col, [&]() {
    auto entry = compressed_val[i, j] * vec_val[compressed_col[i, j]];
    reduce(result[i], entry);
  });

  int num_entries = 0;
  for (int i = 0; i < n; i++) {
    std::set<int> cols;
    int sub = rand_int() % 32;
    for (int j = 0; j < k - sub; j++) {
      int col;
      while (true) {
        col = std::rand() % (2 * band) - band + i;
        if (col < 0 || col >= n)
          continue;
        if (cols.find(col) == cols.end()) {
          break;
        }
      }
      cols.insert(col);
      auto val = rand();
      entries.push_back({i, col, val});
      mat_row.val<int32>(num_entries) = i;
      mat_col.val<int32>(num_entries) = col;
      mat_val.val<float32>(num_entries) = val;
      num_entries++;
    }
    auto val = rand();
    V(i) = val;
    vec_val.val<float32>(i) = val;
  }

  TC_TIME(M.setFromTriplets(entries.begin(), entries.end()));

  TC_TIME(populate());

  int T = 30;
  for (int i = 0; i < T; i++) {
    TC_TIME(matvecmul());
  }

  TC_P(n = Eigen::nbThreads());
  for (int i = 0; i < T; i++) {
    TC_TIME(Vret = M * V);
  }

  for (int i = 0; i < n; i++) {
    TC_CHECK_EQUAL(Vret(i), result.val<float32>(i) / T, 1e-3_f);
  }
}

// array of linked list
TC_TEST("indirect") {
  Program prog;

  int n = 4;
  int k = 8;
  int m = n * k;

  Global(a, i32);
  Global(sum, i32);

  SNode *snode;

  layout([&] {
    auto i = Index(0), j = Index(1);
    // indirect puts an int32
    snode = &root.dense(i, n).indirect(j, k * 2);
    root.dense(j, m).place(a);
    root.dense(i, n).place(sum);
  });

  auto populate = kernel([&]() {
    Declare(j);

    // the second
    touch(snode, load(a[j]) / imm(k), j);  // put main index into snode sparsity
  });

  auto inc = kernel(a, [&]() { a[j] = a[j] + imm(1); });

  auto red = kernel(snode, [&]() { reduce(sum[i], a[j]); });

  for (int i = 0; i < m; i++) {
    a.val<int32>(i) = i;
  }

  populate();
  inc();
  red();

  for (int i = 0; i < n; i++) {
    auto reduced = sum.val<int32>(i);
    TC_CHECK(reduced == (i * k + (i + 1) * k + 1) * k / 2);
  }
}
#endif

TC_TEST("leaf_context") {
  Program prog;

  int n = 64;
  int k = 32;
  int m = n * k;

  Global(a, i32);
  Global(sum, i32);

  layout([&] {
    auto i = Index(0);
    root.dense(i, n).dense(i, k).place(a);
    root.place(sum);
  });

  int sum_gt = 0;
  for (int i = 0; i < m; i++) {
    if (i / k % 3 == 1) {
      a.val<int32>(i) = i;
      sum_gt += i;
    }
  }

  kernel([&]() {
    Declare(i);
    For(i, a, [&] { sum[Expr(0)] += a[i]; });
  })();

  TC_CHECK(sum.val<int32>() == sum_gt);
}

TC_TEST("pointer") {
  Program prog;

  int n = 32;
  int k = 64;
  int m = n * k;

  Global(a, i32);
  Global(sum, i32);

  layout([&] {
    auto i = Index(0);
    root.dense(i, n).pointer().dense(i, k).place(a);
    root.place(sum);
  });

  int sum_gt = 0;
  for (int i = 0; i < m; i++) {
    if (i / k % 3 == 1) {
      a.val<int32>(i) = i;
      sum_gt += i;
    }
  }

  kernel([&]() {
    Declare(i);
    For(i, a, [&] { sum[Expr(0)] += a[i]; });
  })();

  auto reduced = sum.val<int32>();
  TC_CHECK(reduced == sum_gt);
}

TC_TEST("misaligned") {
  // On the same tree, x has indices i while y has indices i & j
  Program prog;

  int n = 32;
  int k = 64;

  Global(x, i32);
  Global(y, i32);

  layout([&] {
    auto i = Index(0);
    auto j = Index(1);
    auto &fork = root.dense(i, n);
    fork.place(x);
    fork.dense(j, k).place(y);
  });

  std::vector<int> x_gt(n, 0);
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < k; j++) {
      int val = rand<int>() % 10;
      y.val<int32>(i, j) = val;
      x_gt[i] += val;
    }
  }

  kernel([&]() {
    Declare(i);
    Declare(j);
    For(i, 0, n, [&] { For(j, 0, k, [&] { x[i] += y[i, j]; }); });
  })();

  for (int i = 0; i < n; i++) {
    TC_CHECK(x_gt[i] == x.val<int32>(i));
  }
}

TC_TEST("hashed") {
  Program prog;

  int n = 64;
  int k = 128;
  int m = n * k;

  Global(a, i32);
  Global(sum, i32);

  layout([&] {
    auto i = Index(0);
    root.hashed(i, n).dense(i, k).place(a);
    root.place(sum);
  });

  sum.val<int32>() = 0;

  int sum_gt = 0;
  for (int i = 0; i < m; i++) {
    if (i / k % 3 == 1) {
      a.val<int32>(i) = 1;
      sum_gt += 1;
    }
  }

  kernel([&]() {
    Declare(i);
    For(i, a, [&] { sum[Expr(0)] += a[i]; });
  })();

  auto reduced = sum.val<int32>();
  TC_CHECK(reduced == sum_gt);
}

TC_TEST("mpm_layout") {
  Program prog(Arch::gpu);
  constexpr int dim = 3;
  constexpr bool highres = true;

  constexpr int n = 256;
  constexpr int grid_n = n * 4;
  int max_n_particles = 1024 * 1024;

  auto f32 = DataType::f32;
  int grid_block_size = 4;

  Vector particle_x("x", f32, dim), particle_v("v", f32, dim);
  Matrix particle_F("F", f32, dim, dim), particle_C("C", f32, dim, dim);

  NamedScalar(l, l, i32);
  NamedScalar(particle_J, J, f32);
  NamedScalar(gravity_x, g, f32);

  Vector grid_v("v^{g}", f32, dim);
  NamedScalar(grid_m, m ^ {p}, f32);

  auto i = Index(0), j = Index(1), k = Index(2);
  auto p = Index(3);

  bool particle_soa = true;

  layout([&]() {
    SNode *fork;
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
    place(particle_J);

    TC_ASSERT(n % grid_block_size == 0);
    auto &block = root.dense({i, j, k}, grid_n / 4 / grid_block_size)
                      .pointer()
                      .dense({i, j, k}, 4)
                      .pointer();
    constexpr bool block_soa = true;

    if (block_soa) {
      block.dense({i, j, k}, grid_block_size).place(grid_v(0));
      block.dense({i, j, k}, grid_block_size).place(grid_v(1));
      block.dense({i, j, k}, grid_block_size).place(grid_v(2));
      block.dense({i, j, k}, grid_block_size).place(grid_m);
    } else {
      block.dense({i, j, k}, grid_block_size).place(grid_v).place(grid_m);
    }

    block.dynamic(p, pow<dim>(grid_block_size) * 64).place(l);

    root.place(gravity_x);
  });
}

TLANG_NAMESPACE_END
