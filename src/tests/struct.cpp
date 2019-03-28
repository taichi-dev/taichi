#include <taichi/testing.h>
#include "../tlang.h"

TLANG_NAMESPACE_BEGIN

TC_TEST("snode") {
  Program prog(Arch::x86_64);

  auto i = Index(0);
  Global(u, i32);

  int n = 128;

  // All data structure originates from a "root", which is a forked node.
  prog.layout([&] { root.fixed(i, n).place(u); });

  for (int i = 0; i < n; i++) {
    u.val<int32>(i) = i + 1;
  }

  for (int i = 0; i < n; i++) {
    TC_CHECK_EQUAL(u.val<int32>(i), i + 1, 0);
  }
}

TC_TEST("snode_loop") {
  Program prog(Arch::x86_64);
  CoreState::set_trigger_gdb_when_crash(true);
  prog.config.print_ir = true;

  auto i = Index(0);
  Global(u, i32);

  int n = 128;

  // All data structure originates from a "root", which is a forked node.
  prog.layout([&] { root.fixed(i, n).place(u); });

  auto set = kernel([&] {
    Declare(i);
    For(i, u, [&] { u[i] = i * 2; });
  });

  set();

  for (int i = 0; i < n; i++) {
    TC_CHECK_EQUAL(u.val<int32>(i), i * 2, 0);
  }
}

TC_TEST("snode_loop2") {
  Program prog(Arch::x86_64);
  CoreState::set_trigger_gdb_when_crash(true);
  prog.config.print_ir = true;

  auto i = Index(0), j = Index(1);
  Global(u, i32);
  Global(v, i32);

  int n = 128;

  // All data structure originates from a "root", which is a forked node.
  prog.layout([&] {
    root.fixed(i, n).place(u);
    root.fixed(j, n).place(v);
  });

  TC_ASSERT(
      u.cast<GlobalVariableExpression>()->snode->physical_index_position[0] ==
      0);
  TC_ASSERT(
      v.cast<GlobalVariableExpression>()->snode->physical_index_position[0] ==
      1);

  auto set1 = kernel([&] {
    Declare(i);
    For(i, u, [&] { u[i] = i * 2; });
  });

  auto set2 = kernel([&] {
    Declare(j);
    For(j, v, [&] { v[j] = j * 3; });
  });

  set1();
  set2();

  for (int i = 0; i < n; i++) {
    TC_CHECK_EQUAL(u.val<int32>(i), i * 2, 0);
    TC_CHECK_EQUAL(v.val<int32>(i), i * 3, 0);
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
          root.fixed({i, j}, {n / block_size, n * 2 / block_size})
              .fixed({i, j}, {block_size, block_size})
              .place(a, b);
        } else {
          root.fixed({i, j}, {n, n * 2}).place(a);
          root.fixed({i, j}, {n, n * 2}).place(b);
        }
      });

      auto inc = kernel([&]() {
        Declare(i);
        Declare(j);
        For({i, j}, a, [&] { b[i, j] = a[i, j] + i; });
      });

      for (int i = 0; i < n; i++) {
        for (int j = 0; j < n * 2; j++) {
          a.val<int32>(i, j) = i + j * 3;
        }
      }

      inc();

      for (int i = 0; i < n; i++) {
        for (int j = 0; j < n * 2; j++) {
          TC_CHECK(a.val<int32>(i, j) == i + j * 3);
          TC_CHECK(b.val<int32>(i, j) == i * 2 + j * 3);
        }
      }
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
    root.fixed(p, m).place(mat_row);
    root.fixed(p, m).place(mat_col);
    root.fixed(p, m).place(mat_val);
    auto &mat = root.fixed(i, n).multi_threaded();
    mat.dynamic(j, k).place(compressed_col);
    mat.dynamic(j, k).place(compressed_val);
    root.fixed(i, n).place(vec_val);
    root.fixed(i, n).place(result);
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
    auto &mat = root.fixed(i, n).multi_threaded();
    mat.dynamic(j, k).place(compressed_col);
    mat.dynamic(j, k).place(compressed_val);
    root.fixed(i, n).place(vec_val);
    root.fixed(i, n).place(result);
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
    snode = &root.fixed(i, n).indirect(j, k * 2);
    root.fixed(j, m).place(a);
    root.fixed(i, n).place(sum);
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
    root.fixed(i, n).fixed(i, k).place(a);
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
    root.fixed(i, n).pointer().fixed(i, k).place(a);
    root.place(sum);
  });

  auto red = kernel([&]() {
    Declare(i);
    For(i, a, [&] { sum[Expr(0)] += a[i]; });
  });

  int sum_gt = 0;
  for (int i = 0; i < m; i++) {
    if (i / k % 3 == 1) {
      a.val<int32>(i) = i;
      sum_gt += i;
    }
  }

  red();

  auto reduced = sum.val<int32>();
  TC_CHECK(reduced == sum_gt);
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
    root.hashed(i, n).fixed(i, k).place(a);
    root.place(sum);
  });

  auto red = kernel([&]() {
    Declare(i);
    For(i, a, [&] { sum[Expr(0)] += a[i]; });
  });
  sum.val<int32>() = 0;

  int sum_gt = 0;
  for (int i = 0; i < m; i++) {
    if (i / k % 3 == 1) {
      a.val<int32>(i) = 1;
      sum_gt += 1;
    }
  }

  red();

  auto reduced = sum.val<int32>();
  TC_CHECK(reduced == sum_gt);
}

TC_TEST("box_filter") {
  return;
  Program prog;

  int n = 64;
  int k = 128;
  int m = n * k;

  Global(x, i32);
  Global(y, i32);

  layout([&] {
    auto i = Index(0);
    root.hashed(i, n).fixed(i, k).place(x, y);
  });

  auto red = kernel([&]() {
    Declare(i);
    For(i, x, [&] { y[i] = x[i - 1] + x[i] + x[i + 1]; });
  });

  red();

  // auto reduced = sum.val<int32>();
  // TC_CHECK(reduced == sum_gt);
}

TLANG_NAMESPACE_END
