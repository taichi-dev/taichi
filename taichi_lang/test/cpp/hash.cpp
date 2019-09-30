#include <taichi/testing.h>
#include <taichi/lang.h>

TLANG_NAMESPACE_BEGIN

TC_TEST("hash") {
  for (auto arch : {Arch::gpu, Arch::x86_64}) {
    Program prog(arch);
    CoreState::set_trigger_gdb_when_crash(true);

    auto i = Index(0);
    Global(u, i32);

    int n = 256;

    prog.layout([&] { root.hash(i, n).dense(i, n).place(u); });

    kernel([&] {
      Declare(i);
      BlockDim(256);
      For(i, 0, n * n / 2, [&] { u[i] = i * 2; });
    })();

    for (int i = 0; i < n * n / 2; i++) {
      TC_CHECK(u.val<int32>(i) == i * 2);
    }
    for (int i = n * n / 2; i < n * n; i++) {
      TC_CHECK(u.val<int32>(i) == 0);
    }
  }
}

TLANG_NAMESPACE_END
