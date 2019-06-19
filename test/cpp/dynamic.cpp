#include <taichi/lang.h>
#include <taichi/testing.h>
#include <numeric>
#include <taichi/taichi>

TLANG_NAMESPACE_BEGIN

TC_TEST("append_and_probe") {
  CoreState::set_trigger_gdb_when_crash(true);
  for (auto arch : {Arch::x86_64, Arch::gpu}) {
    int n = 32;
    Program prog(arch);

    Global(x, i32);
    Global(len, i32);
    SNode *list;
    layout([&]() {
      auto i = Index(0);
      list = &root.dynamic(i, n);
      list->place(x);
      root.place(len);
    });

    kernel([&]() {
      Declare(i);
      For(i, 0, n, [&] { Append(list, i, i); });
    })();

    kernel([&]() {
      Declare(i);
      len[Expr(0)] = Probe(list, Expr(0));
    });

    for (int i = 0; i < n; i++) {
      TC_CHECK(x.val<int>(i) == i);
    }
  }
};

TC_TEST("activate") {
  for (auto arch : {Arch::x86_64, Arch::gpu}) {
    int n = 32;
    Program prog(arch);

    Global(x, i32);
    layout([&]() {
      auto i = Index(0);
      auto j = Index(1);
      root.dense(i, n).pointer().dense(j, n).place(x);
    });

    /*
    kernel([&]() {
      Declare(i);
      Declare(j);
      For(i, 0, n, [&] {
        For(j, 0, i, [&] {
          Activate(x.snode(), {i, j});
        });
      });
    })();
    */

    kernel([&]() {
      For(0, n, [&](Expr i) { For(0, i, [&](Expr j) { x[i, j] = i + j; }); });
    })();

    for (int i = 0; i < n; i++) {
      for (int j = 0; j < i; j++) {
        TC_CHECK(x.val<int>(i, j) == i + j);
      }
    }
  }
};

TC_TEST("task_list") {
  for (auto arch : {Arch::gpu}) {
    int n = 262144;
    int m = 64;
    Program prog(arch);

    Global(x, i32);
    layout([&]() {
      auto i = Index(0);
      auto j = Index(1);
      root.dense(i, n).pointer().dense(j, m).place(x);
    });

    kernel([&]() {
      For(0, n, [&](Expr i) {
        For(0, max(i % 5 - 3, 0), [&](Expr j) { Activate(x.snode(), {i, j}); });
      });
    })();

    kernel([&]() { For(x, [&](Expr i, Expr j) { x[i, j] = i + j; }); })();

    auto &inc =
        kernel([&]() { For(x, [&](Expr i, Expr j) { x[i, j] += 1; }); });

    int P = 10;

    for (int i = 0; i < P; i++) {
      inc();
    }

    for (int i = 0; i < n; i++) {
      if (i % 5 == 4)
        for (int j = 0; j < m; j++) {
          TC_CHECK(x.val<int>(i, j) == i + j + P);
        }
    }
  }
};

TC_TEST("task_list_dynamic") {
  for (auto arch : {Arch::gpu}) {
    int n = 262144;
    int m = 64;
    Program prog(arch);

    Global(x, i32);
    layout([&]() {
      auto i = Index(0);
      auto j = Index(1);
      root.dense(i, n).dynamic(j, m).place(x);
    });

    for (int i = 0; i < n; i++) {
      for (int j = 0; j < m; j++) {
        x.val<int32>(i, j) = i + j;
      }
    }

    auto &inc =
        kernel([&]() { For(x, [&](Expr i, Expr j) { x[i, j] += 1; }); });

    int P = 10;

    for (int i = 0; i < P; i++) {
      inc();
    }

    for (int i = 0; i < n; i++) {
      if (i % 5 == 4)
        for (int j = 0; j < 1; j++) {
          TC_CHECK(x.val<int>(i, j) == i + j + P);
        }
    }
  }
};

TC_TEST("parallel_append") {
  for (auto arch : {Arch::gpu}) {
    int n = 32;
    Program prog(arch);

    Global(x, i32);
    SNode *list;
    layout([&]() {
      auto i = Index(0);
      auto j = Index(1);
      root.dense(i, n).pointer().dynamic(j, n).place(x);
    });

    Kernel(append).def([&]() {
      Declare(i);
      For(i, 0, n * n, [&] { Append(x.parent(), (i % n, 0), i); });
    });

    Kernel(activate).def([&]() {
      Declare(i);
      For(i, 0, n * n, [&] { Activate(x.parent(), (i % n, 0)); });
    });

    for (int i = 0; i < 32; i++) {
      x.parent().parent().snode()->clear_data_and_deactivate();
      if (i % 2)
        activate();
      else
        append();
      auto stat = x.parent().parent().snode()->stat();
      TC_CHECK(stat.num_resident_blocks == n);
      if (i % 2)
        for (int i = 0; i < n; i++) {
          for (int j = 0; j < n; j++) {
            TC_CHECK(x.val<int>(i, j) == 0);
          }
        }
    }
  }
};

TC_TEST("append_2d") {
  for (auto arch : {Arch::x86_64, Arch::gpu}) {
    int n = 32;
    Program prog(arch);

    Global(x, i32);
    SNode *list;
    layout([&]() {
      auto i = Index(0);
      auto j = Index(1);
      list = &root.dense(i, n).dynamic(j, n);
      list->place(x);
    });

    kernel([&]() {
      Declare(i);
      Declare(j);
      For(i, 0, n, [&] { For(j, 0, i, [&] { Append(list, i, i + j); }); });
    })();

    for (int i = 0; i < n; i++) {
      for (int j = 0; j < i; j++) {
        TC_CHECK(x.val<int>(i, j) == i + j);
      }
    }
  }
};

TC_TEST("clear") {
  CoreState::set_trigger_gdb_when_crash(true);
  for (auto arch : {Arch::x86_64, Arch::gpu}) {
    int n = 32;
    Program prog(arch);

    Global(x, i32);
    SNode *list;
    layout([&]() {
      auto i = Index(0);
      auto j = Index(1);
      list = &root.dense(i, n).dynamic(j, n);
      list->place(x);
    });

    auto &append = kernel([&]() {
      Declare(i);
      Declare(j);
      For(i, 0, n, [&] { For(j, 0, i, [&] { Append(list, i, i + j); }); });
    });
    append();

    kernel([&]() {
      Declare(i);
      For(i, 0, n, [&] { Clear(list, i); });
    })();

    append();

    for (int i = 0; i < n; i++) {
      for (int j = 0; j < n; j++) {
        if (j < i)
          TC_CHECK(x.val<int>(i, j) == i + j);
        else
          TC_CHECK(x.val<int>(i, j) == 0);
      }
    }
  }
};

TC_TEST("sort") {
  for (auto arch : {Arch::x86_64, Arch::gpu}) {
    int n = 4;
    Program prog(arch);

    std::vector<int> particles(n * n);
    std::vector<int> count(n * n, 0);

    for (int i = 0; i < n * n; i++) {
      particles[i] = rand_int() % (n * n);
      count[particles[i]]++;
    }

    Global(c, i32);
    Global(coord, i32);
    Global(p, i32);
    TC_P(particles);
    layout([&]() {
      auto i = Index(0);
      auto j = Index(1);
      root.dense(i, n * n).place(coord);
      auto &fork = root.dense(i, n);
      fork.dense(i, n).place(c);
      fork.dynamic(j, n * n).place(p);
    });

    for (int i = 0; i < n * n; i++) {
      coord.val<int32>(i) = particles[i];
    }

    auto &sort = kernel([&]() {
      Declare(i);
      Declare(j);
      For(i, 0, n * n, [&] { Append(p.parent().snode(), coord[i], i); });
    });

    sort();

    kernel([&]() {
      Declare(i);
      Declare(j);
      For(i, p.parent(), [&] {
        auto len = Var(Probe(p.parent().snode(), i));
        Print(len);
        For(j, 0, len, [&] {
          auto pos = coord[p[i, j]];
          c[pos] += 1;
        });
      });
    })();

    for (int i = 0; i < n * n; i++) {
      TC_CHECK(c.val<int>(i) == count[i]);
    }
  }
};

TC_TEST("dilate") {
  for (auto arch : {Arch::x86_64, Arch::gpu}) {
    for (auto ds : {1}) {
      int n = 32;
      int bs = 4;
      Program prog(arch);

      Global(x, i32);
      Global(y, i32);
      layout([&]() {
        auto i = Index(0);
        if (ds) {
          root.dense(i, n / bs).pointer().dense(i, bs).place(x);
        } else {
          root.dense(i, n / bs).bitmasked().dense(i, bs).place(x);
        }
        root.dense(i, n / bs).place(y);
      });

      x.val<int32>(bs * 2);

      // dilate
      kernel([&]() {
        For(x, [&](Expr i) {
          x[i - 1] = 0;
          x[i + 1] = 0;
        });
      })();

      // dilate
      kernel([&]() { For(x, [&](Expr i) { x[i] += 1; }); })();

      kernel([&] { For(x, [&](Expr i) { y[i / bs] = Probe(x, i); }); })();

      for (int i = 0; i < n; i++) {
        int bid = i / bs;
        TC_CHECK(x.val<int32>(i) == (1 <= bid && bid < 4));
      }
      for (int i = 0; i < n / bs; i++) {
        TC_CHECK(y.val<int32>(i) == (1 <= i && i < 4));
      }
    }
  }
};

TC_TEST("dynamic_sort") {
  for (auto arch : {Arch::x86_64, Arch::gpu}) {
    int n = 4;
    Program prog(arch);

    std::vector<int> particles(n * n);
    std::vector<int> count(n * n, 0);

    for (int i = 0; i < n * n; i++) {
      particles[i] = rand_int() % (n * n);
      count[particles[i]]++;
    }

    Global(c, i32);
    Global(coord, i32);
    Global(p, i32);
    TC_P(particles);
    layout([&]() {
      auto i = Index(0);
      auto j = Index(1);
      root.dense(i, n * n).place(coord);
      auto &fork = root.dense(i, n).pointer();
      fork.dense(i, n).place(c);
      fork.dynamic(j, n * n).place(p);
    });

    for (int i = 0; i < n * n; i++) {
      coord.val<int32>(i) = particles[i];
    }

    auto &sort = kernel([&]() {
      Declare(i);
      Declare(j);
      For(i, 0, n * n, [&] {
        Activate(p.snode(), coord[i]);
        Append(p.parent().snode(), coord[i], i);
      });
    });

    sort();

    kernel([&]() {
      Declare(i);
      Declare(j);
      For(i, p.parent(), [&] {
        auto len = Var(Probe(p.parent().snode(), i));
        Print(len);
        For(j, 0, len, [&] {
          auto pos = coord[p[i, j]];
          c[pos] += 1;
        });
      });
    })();

    for (int i = 0; i < n * n; i++) {
      TC_CHECK(c.val<int>(i) == count[i]);
    }
  }
};

auto reset_grid_benchmark = []() {
  Program prog(Arch::gpu);

  constexpr int n = 256;  // grid_resolution
  constexpr int dim = 3;

  auto f32 = DataType::f32;
  int grid_block_size = 4;

  Vector grid_v(f32, dim);
  Global(grid_m, f32);

  auto i = Index(0), j = Index(1), k = Index(2);

  layout([&]() {
    TC_ASSERT(n % grid_block_size == 0);
    auto &block = root.dense({i, j, k}, n / grid_block_size);
    constexpr bool block_soa = false;
    if (block_soa) {
      block.dense({i, j, k}, grid_block_size).place(grid_v(0));
      block.dense({i, j, k}, grid_block_size).place(grid_v(1));
      block.dense({i, j, k}, grid_block_size).place(grid_v(2));
      block.dense({i, j, k}, grid_block_size).place(grid_m);
    } else {
      block.dense({i, j, k}, grid_block_size)
          .place(grid_v(0), grid_v(1), grid_v(2), grid_m);
      //.place(grid_m);
    }
  });

  TC_ASSERT(bit::is_power_of_two(n));

  auto &reset_grid = kernel([&]() {
    Declare(i);
    Declare(j);
    Declare(k);
    For((i, j, k), grid_m, [&] {
      grid_v(0)[i, j, k] = 0.0_f;
      grid_v(1)[i, j, k] = 0.0_f;
      grid_v(2)[i, j, k] = 0.0_f;
      grid_m[i, j, k] = 0.0_f;
    });
  });

  while (1)
    TC_TIME(reset_grid());
};
TC_REGISTER_TASK(reset_grid_benchmark);

TLANG_NAMESPACE_END
