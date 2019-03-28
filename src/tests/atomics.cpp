#include "../tlang.h"
#include <taichi/testing.h>
#include <numeric>

TLANG_NAMESPACE_BEGIN

TC_TEST("atomics") {
  CoreState::set_trigger_gdb_when_crash(true);
  int n = 10000000;
  Program prog(Arch::x86_64);
  prog.config.print_ir = true;

  Global(sum, i32);
  Global(fsum, f32);
  layout([&]() { root.place(sum, fsum); });

  auto func = kernel([&]() {
    Declare(i);
    Parallelize(4);
    For(i, 0, n, [&] {
      Atomic(sum[Expr(0)]) += 1;
      // Atomic(fsum[Expr(0)]) += 1 - 2 * (i % 2);
      Atomic(fsum[Expr(0)]) += cast<float32>(1 - 2 * (i % 2));
    });
  });

  func();

  TC_CHECK(sum.val<int>() == n);
  TC_CHECK(fsum.val<float32>() == 0);
};

TLANG_NAMESPACE_END
