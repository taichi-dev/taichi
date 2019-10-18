#include <taichi/lang.h>
#include <taichi/testing.h>
#include <numeric>

int main() {
  using namespace taichi;
  using namespace taichi::Tlang;
  CoreState::set_trigger_gdb_when_crash(true);
  int n = 1;
  default_compile_config.use_llvm = true;
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

