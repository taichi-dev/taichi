#include <taichi/lang.h>
#include <taichi/testing.h>
#include <numeric>

TLANG_NAMESPACE_BEGIN

TI_TEST("atomics") {
  CoreState::set_trigger_gdb_when_crash(true);
  int n = 10000000;
  Program prog(Arch::x64);

  Global(sum, i32);
  Global(fsum, f32);
  layout([&]() { root.place(sum, fsum); });

  auto &func = kernel([&]() {
    Parallelize(4);
    For(0, n, [&](Expr i) {
      Atomic(sum[Expr(0)]) += 1;
      Atomic(fsum[Expr(0)]) += cast<float32>(1 - 2 * (i % 2));
    });
  });

  func();

  TI_CHECK(sum.val<int>() == n);
  TI_CHECK(fsum.val<float32>() == 0);
};

TI_TEST("atomics2") {
  CoreState::set_trigger_gdb_when_crash(true);
  int n = 1000;
  Program prog(Arch::x64);

  Global(fsum, f32);
  layout([&]() { root.place(fsum); });

  auto &func = kernel([&]() {
    Parallelize(4);
    For(0, n, [&](Expr i) { Atomic(fsum[Expr(0)]) += 1.0f; });
  });

  func();

  TI_CHECK(fsum.val<float32>() == 1000);
};

TI_TEST("parallel_reduce") {
  CoreState::set_trigger_gdb_when_crash(true);
  int n = 1024 * 1024 * 32;
  Program prog(Arch::x64);

  Global(fsum, i32);
  Global(a, i32);
  layout([&]() {
    root.place(fsum).dense(Index(0), n / 1024).dense(Index(0), 1024).place(a);
  });

  for (int i = 0; i < n; i++)
    a.val<int32>(i) = i;

  Kernel(reduce).def([&]() {
    Parallelize(8);
    Vectorize(8);
    mark_reduction();
    For(a, [&](Expr i) { Atomic(fsum[Expr(0)]) += a[i]; });
  });

  for (int i = 0; i < 10; i++)
    reduce();
  prog.profiler_print();

  TI_CHECK(fsum.val<int32>() == (n / 2) * (n - 1) * 10);
};

TLANG_NAMESPACE_END
