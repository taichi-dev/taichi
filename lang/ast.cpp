#include "ast.h"
#include <numeric>
#include <Eigen/Dense>

TLANG_NAMESPACE_BEGIN

class ASTPrinter : public ASTVisitor {
 public:
  int current_indent;

  ASTPrinter() {
    current_indent = -1;
  }

  template <typename... Args>
  void print(std::string f, Args &&... args) {
    for (int i = 0; i < current_indent; i++)
      fmt::print("  ");
    fmt::print(f, std::forward<Args>(args)...);
    fmt::print("\n");
  }

  static void run(ASTNode &node) {
    auto p = ASTPrinter();
    node.accept(p);
  }

  void visit(StatementList &stmt_list) {
    current_indent++;
    for (auto &stmt : stmt_list.statements) {
      stmt->accept(*this);
    }
    current_indent--;
  }

  void visit(AssignmentStatement &assign) {
    print("{} <- {}", assign.id.name(), assign.rhs->serialize());
  }

  void visit(AllocaStatement &alloca) {
    print("{} <- alloca {}", alloca.lhs.name(), data_type_name(alloca.type));
  }

  void visit(BinaryOpStatement &bin) {
    print("{} <- {} {} {}", bin.lhs.name(), binary_type_name(bin.type),
          bin.rhs1.name(), bin.rhs2.name());
  }

  void visit(IfStatement &if_stmt) {
    print("if {} {{", if_stmt.condition->serialize());
    if (if_stmt.true_statements)
      if_stmt.true_statements->accept(*this);
    if (if_stmt.false_statements) {
      print("}} else {{");
      if_stmt.false_statements->accept(*this);
    }
    print("}}");
  }

  void visit(PrintStatement &print_stmt) {
    print("print {}", print_stmt.expr.serialize());
  }

  void visit(ConstStatement &const_stmt) {
    print("{} = const<{}>({})", const_stmt.id.name(),
          data_type_name(const_stmt.data_type), const_stmt.value);
  }

  void visit(ForStatement &for_stmt) {
    print("for {} in range({}, {}) {{", for_stmt.loop_var_id.name(),
          for_stmt.begin->serialize(), for_stmt.end->serialize());
    for_stmt.body->accept(*this);
    print("}}");
  }
};

#define declare(x) auto x = ExpressionHandle(std::make_shared<IdExpression>());

auto test_ast = []() {
  declare(a);
  declare(b);
  declare(i);
  declare(j);

  Var(a);
  Var(b);
  Var(i);
  Var(j);

  a = a + b;
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

  For(i, 0, 100, [&] {
    For(j, 0, 200, [&] {
      ExprH k = i + j;
      Print(k);
      // While(k < 500, [&] { Print(k); });
    });
  });
  Print(b);

  ASTPrinter::run(context.root());
};
TC_REGISTER_TASK(test_ast);

// https://stackoverflow.com/questions/36781881/why-denormalized-floats-are-so-much-slower-than-other-floats-from-hardware-arch
// https://software.intel.com/en-us/forums/intel-performance-bottleneck-analyzer/topic/487262
auto test_mmul_cpe = []() {
  constexpr int n = 1024;
  Eigen::Matrix4f a[n], b[n];
  auto measure = [&] {
    return measure_cpe([&] {
      for (int i = 0; i < n; i++) {
        // TC_P(a[i](0, 0));
        b[i] = a[i] * a[i];
      }
    }, n);
  };

  // muls resulting in denormed floats can make things 700x slower
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < 4; j++) {
      for (int k = 0; k < 4; k++) {
        a[i](j, k) = rand() * 1e-21_f;
      }
    }
  }

  TC_INFO("denormed cpe {}", measure());

  for (int i = 0; i < n; i++) {
    for (int j = 0; j < 4; j++) {
      for (int k = 0; k < 4; k++) {
        a[i](j, k) = rand();
      }
    }
  }
  TC_INFO("regular cpe {}", measure());
};
TC_REGISTER_TASK(test_mmul_cpe);

TLANG_NAMESPACE_END
