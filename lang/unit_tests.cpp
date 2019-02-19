#include "ir.h"
#include <numeric>
#include "tlang.h"
#include <Eigen/Dense>

TLANG_NAMESPACE_BEGIN

TC_TEST("test_compiler") {
  int n = 128;
  Program prog(Arch::x86_64);

  auto a = global<float32>();
  auto i = ind();

  layout([&]() { root.fixed(i, n).place(a); });

  auto func = kernel(a, [&]() {
    a[i] = select(cmp_ne(imm(0), i % imm(2)), cast<float32>(i), imm(0.0_f));
  });

  func();

  for (int i = 0; i < n; i++) {
    TC_CHECK(a.val<float32>(i) == (i % 2) * i);
  }
};

auto test_ast = []() {
  CoreState::set_trigger_gdb_when_crash(true);
  context = std::make_unique<FrontendContext>();
  declare(a);
  declare(b);
  declare(p);
  declare(q);
  declare(i);
  declare(j);

  var(float32, a);
  var(float32, b);
  var(int32, p);
  var(int32, q);

  a = a + b;
  p = p + q;

  Print(a);
  If(a < 500).Then([&] { Print(b); }).Else([&] { Print(a); });

  If(a > 5)
      .Then([&] {
        b = (b + 1) / 3;
        b = b * 3;
      })
      .Else([&] {
        b = b + 2;
        b = b - 4;
      });

  For(i, 0, 8, [&] {
    For(j, 0, 8, [&] {
      auto k = i + j;
      Print(k);
      // While(k < 500, [&] { Print(k); });
    });
  });
  Print(b);

  auto root = context->root();

  TC_INFO("AST");
  irpass::print(root);

  irpass::lower(root);
  TC_INFO("Lowered");
  irpass::print(root);

  irpass::typecheck(root);
  TC_INFO("TypeChecked");
  irpass::print(root);
};
TC_REGISTER_TASK(test_ast);

TLANG_NAMESPACE_END
