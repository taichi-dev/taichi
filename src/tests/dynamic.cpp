#include "../tlang.h"
#include <taichi/testing.h>
#include <numeric>

TLANG_NAMESPACE_BEGIN

TC_TEST("append_and_probe") {
  CoreState::set_trigger_gdb_when_crash(true);
  for (auto arch : {Arch::x86_64, Arch::gpu}) {
    int n = 32;
    Program prog(arch);
    prog.config.print_ir = true;

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

TC_TEST("append_2d") {
  for (auto arch : {Arch::x86_64, Arch::gpu}) {
    int n = 32;
    Program prog(arch);
    prog.config.print_ir = true;

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
    prog.config.print_ir = true;

    Global(x, i32);
    SNode *list;
    layout([&]() {
      auto i = Index(0);
      auto j = Index(1);
      list = &root.dense(i, n).dynamic(j, n);
      list->place(x);
    });

    auto append = kernel([&]() {
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
  return;
  for (auto arch : {Arch::x86_64, Arch::gpu}) {
    int n = 32;
    Program prog(arch);

    std::vector<int> particles(n * n);
    std::vector<int> count(n * n, 0);

    for (int i = 0; i < n * n; i++) {
      particles[i] = rand<int>() % (n * n);
      count[particles[i]]++;
    }

    Global(c, i32);
    Global(coord, i32);
    Global(p, i32);
    SNode *list;
    layout([&]() {
      auto i = Index(0);
      auto j = Index(1);
      root.dense(i, n * n).place(coord);
      auto &fork = root.dense(i, n);
      fork.dense(i, n).place(c);
      list = &fork.dynamic(j, n * n);
      list->place(p);
    });

    for (int i = 0; i < n * n; i++) {
      coord.val<int32>(i) = particles[i];
    }

    auto sort = kernel([&]() {
      Declare(i);
      Declare(j);
      For(i, 0, n, [&] { Append(list, coord[i] / n, i); });
    });

    sort();

    kernel([&]() {
      Declare(i);
      Declare(j);
      // TODO: should be parent node of p block
      For(i, p, [&] {
        auto len = Eval(Probe(list, i));
        Print(len);
        For(j, 0, len, [&] {
          auto pos = load(coord[load(p[i, j])]);
          c[pos] += 1;
          Print(pos);
        });
      });
    })();

    for (int i = 0; i < n * n; i++) {
      TC_CHECK(c.val<int>(i) == count[i]);
    }
  }
};

TLANG_NAMESPACE_END
