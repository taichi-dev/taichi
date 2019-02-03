#include "ast.h"

TLANG_NAMESPACE_BEGIN

class ASTPrinter : public ASTVisitor {
 public:
  static void run(ASTNode &node) {
    auto p = ASTPrinter();
    node.accept(p);
  }

  void visit(StatementList &stmt_list) {
    for (auto &stmt : stmt_list.statements) {
      stmt->accept(*this);
    }
  }

  void visit(AssignmentStatement &assign) {
    fmt::print("{} <- {}\n", assign.lhs.name(), assign.rhs.name());
  }

  void visit(AllocaStatement &alloca) {
    fmt::print("{} <- alloca {}\n", alloca.lhs.name(),
               data_type_name(alloca.type));
  }

  void visit(BinaryOpStatement &bin) {
    fmt::print("{} <- {} {} {}\n", bin.lhs.name(), binary_type_name(bin.type),
               bin.rhs1.name(), bin.rhs2.name());
  }
};

auto test_ast = []() {
  Id a, b, i, j;

  Var(a);
  Var(b);
  Var(i);
  Var(j);

  a = a + b;
  /*

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
