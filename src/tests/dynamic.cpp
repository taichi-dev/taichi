#include "../tlang.h"
#include <taichi/testing.h>
#include <numeric>

TLANG_NAMESPACE_BEGIN

TC_TEST("append") {
  CoreState::set_trigger_gdb_when_crash(true);
  for (auto arch : {Arch::x86_64, Arch::gpu}) {
    int n = 32;
    Program prog(arch);
    prog.config.print_ir = true;

    Global(x, i32);
    SNode *list;
    layout([&]() {
      auto i = Index(0);
      list = &root.dynamic(i, n);
      list->place(x);
    });

    auto func = kernel([&]() {
      Declare(i);
      For(i, 0, n, [&] { Append(list, i, i); });
    });

    func();

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

TLANG_NAMESPACE_END
