#include "../tlang.h"
#include <taichi/testing.h>
#include <numeric>

TLANG_NAMESPACE_BEGIN

TC_TEST("atomics") {
  CoreState::set_trigger_gdb_when_crash(true);
  int n = 100000000;
  std::atomic<int> a;
  a += 100;
  Program prog(Arch::x86_64);
  prog.config.print_ir = true;

  Global(sum, i32);
  layout([&]() { root.place(sum); });

  TC_P(Atomic(sum).atomic);

  auto func = kernel([&]() {
    Declare(i);
    Parallelize(4);
    For(i, 0, n, [&] { Atomic(sum[Expr(0)]) += 1; });
  });

  func();

  TC_CHECK(sum.val<int>() == n);
};

TLANG_NAMESPACE_END
