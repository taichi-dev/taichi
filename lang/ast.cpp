#include "ast.h"

TLANG_NAMESPACE_BEGIN

auto test_ast = []() {
  Id a, b, i, j;

  Var(a);
  Var(b);
  Var(i);
  Var(j);

  /*
  a = a + 1;

  If(a > 5)
      .Then([&] {
        b = (b + 1) / 3;
        b = b * 3;
      })
      .Else([&] {
        b = b + 2;
        b = b - 4;
      });

  For(i, 0, 100, [&] {
    For(j, 0, 200, [&] {
      Id k = i + j;
      While(k < 500, [&] { Print(k); });
    });
  });

  Print(b);
  */

  ASTPrinter::run(context.root());
};

TC_REGISTER_TASK(test_ast);

TLANG_NAMESPACE_END
