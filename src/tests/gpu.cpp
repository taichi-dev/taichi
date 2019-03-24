#include "../tlang.h"
#include <taichi/testing.h>
#include <numeric>

TLANG_NAMESPACE_BEGIN

TC_TEST("compiler_basics_gpu") {
  return;
  CoreState::set_trigger_gdb_when_crash(true);
  int n = 128;
  Program prog(Arch::x86_64);
  prog.config.print_ir = true;
  prog.config.arch = Arch::gpu;

  Global(a, i32);
  auto i = Index(0);
  layout([&]() { root.fixed(i, n).place(a); });

  auto dou = [](Expr a) { return a * 2; };

  auto func = kernel([&]() {
    Declare(i);
    For(i, 0, n, [&] {
      Local(ret) = 0;
      If(i % 2 == 0).Then([&] { ret = dou(i); }).Else([&] { ret = i; });
      a[i] = ret;
    });
  });

  func();

  for (int i = 0; i < n; i++) {
    TC_CHECK(a.val<int32>(i) == (2 - i % 2) * i);
  }
};

TLANG_NAMESPACE_END
