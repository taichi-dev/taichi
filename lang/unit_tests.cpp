#include "ir.h"
#include <numeric>
#include "tlang.h"
#include <Eigen/Dense>

TLANG_NAMESPACE_BEGIN

TC_TEST("test_compiler") {
  CoreState::set_trigger_gdb_when_crash(true);
  int n = 128;
  Program prog(Arch::x86_64);

  declare(a_global);
  auto a = global_new(a_global, DataType::f32);
  auto i = ind();

  layout([&]() { root.fixed(i, n).place_new(a); });

  auto func = kernel([&]() {
    declare(i);
    declare(sum);
    var(int32, sum);

    For(i, 0, n, [&] {
      sum = sum + i;
      If(i % 2 == 0).Then([&] { a[i] = i + i; }).Else([&] { a[i] = i; });
      Print(a[i]);
    });
    Print(sum);
  });

  func();

  /*
  for (int i = 0; i < n; i++) {
    TC_CHECK(a.val<float32>(i) == (i % 2) * i);
  }
  */
};

auto test_ast = []() {
  CoreState::set_trigger_gdb_when_crash(true);

  Program prog(Arch::x86_64);
  auto index = ind();
  int n = 128;

  // layout([&]() { root.fixed(index, n).place(x); });

  context = std::make_unique<FrontendContext>();
  declare(a);
  declare(x);
  declare(b);
  declare(p);
  declare(q);
  declare(i);
  declare(j);

  // var(float32, a);
  x.set(global_new(x, DataType::f32));
  TC_ASSERT(x.is<GlobalVariableExpression>());

  var(float32, a);
  var(float32, b);
  var(int32, p);
  var(int32, q);

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
    x[i] = i;
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
