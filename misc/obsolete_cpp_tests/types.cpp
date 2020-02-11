#include <taichi/lang.h>
#include <taichi/testing.h>
#include <numeric>

TLANG_NAMESPACE_BEGIN

TC_TEST("float64") {
  CoreState::set_trigger_gdb_when_crash(true);
  int n = 32;
  Program prog(Arch::gpu);

  Global(a, f64);
  // Global(b, f32);
  layout([&]() { root.dense(Index(0), n).place(a); });

  auto &func = kernel([&]() {
    Parallelize(4);
    For(0, n, [&](Expr i) { a[i] = cast<float64>(i) * 2.0; });
  });

  func();

  // TC_CHECK(sum.val<int>() == n);
  for (int i = 0; i < n; i++)
    TC_CHECK(a.val<float64>(i) == i * 2);
};

TC_TEST("llvm_exception") {
  CoreState::set_trigger_gdb_when_crash(true);
  int n = 1;
  for (int i = 0; i < 2; i++) {
    // Program prog(Arch::gpu);
    Program prog(Arch::x86_64);
    Global(a, f32);
    // Global(b, f32);
    layout([&]() { root.dense(Index(0), n).place(a); });
    auto &func = kernel([&]() { For(0, n, [&](Expr i) { a[i] = 1; }); });

    func();
  };
  try {
    throw IRModified();
  } catch (IRModified) {
    TC_TAG;
  }
};

TLANG_NAMESPACE_END
