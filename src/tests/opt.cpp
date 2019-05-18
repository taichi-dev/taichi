#include "../tlang.h"
#include <taichi/testing.h>
#include <numeric>

TLANG_NAMESPACE_BEGIN

TC_TEST("access_simp") {
  CoreState::set_trigger_gdb_when_crash(true);
  int n = 16;
  Program prog(Arch::x86_64);
  prog.config.print_ir = true;
  prog.config.lower_access = true;

  Global(sum, i32);
  Global(val, i32);

  layout([&]() {
    root.dense(Index(0), n).place(val);
    root.place(sum);
  });

  for (int i = 0; i < n; i++) {
    val.val<int32>(i) = i;
  }

  kernel([&]() {
    For(val, [&](Expr i) {
      sum[Expr(0)] += val[Expr(1)];
      Print(val[Expr(1)]);
    });
  })();

  TC_CHECK(sum.val<int32>() == 16);
};

TLANG_NAMESPACE_END
