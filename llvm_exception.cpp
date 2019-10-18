#include <taichi/lang.h>
#include <taichi/testing.h>
#include <numeric>

void test_throw() {
  using namespace taichi::Tlang;
  try {
    throw IRModified();
  } catch (IRModified) {
    TC_TAG;
  }
}

int main() {
  using namespace taichi;
  using namespace taichi::Tlang;
  CoreState::set_trigger_gdb_when_crash(true);
  int n = 1;
  default_compile_config.use_llvm = true;
  for (int i = 0; i < 2; i++) {
    test_throw();
    // Program prog(Arch::gpu);
    Program prog(Arch::x86_64);
    test_throw();
    Global(a, f32);
    test_throw();
    // Global(b, f32);
    layout([&]() { root.dense(Index(0), n).place(a); });
    test_throw();
    auto &func = kernel([&]() { For(0, n, [&](Expr i) { a[i] = 1; }); });
    test_throw();
    func();
    test_throw();
  };
};

