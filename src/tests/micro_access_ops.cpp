#include "../tlang.h"
#include <taichi/testing.h>

TLANG_NAMESPACE_BEGIN

TC_TEST("lower_access_basics") {
  for (auto vec: {1, 4, 8}) {
    CoreState::set_trigger_gdb_when_crash(true);
    int n = 32;
    Program prog(Arch::x86_64);

    Global(a, i32);
    layout([&]() { root.dense(Index(0), n).place(a); });

    Kernel(func).def([&]() {
      Vectorize(vec);
      For(0, n, [&](Expr i) { a[i] = i; });
    });

    func();

    for (int i = 0; i < n; i++) {
      TC_CHECK(a.val<int32>(i) == i);
    }
  }
};

TLANG_NAMESPACE_END
