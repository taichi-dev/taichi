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

  static void run(ASTNode *node) {
    auto p = ASTPrinter();
    node->accept(&p);
  }

  void visit(StatementList *stmt_list) {
    current_indent++;
    for (auto &stmt : stmt_list->statements) {
      stmt->accept(this);
    }
    current_indent--;
  }

  void visit(AssignmentStatement *assign) {
    print("{} <- {}", assign->id.name(), assign->rhs->serialize());
  }

  void visit(AllocaStatement *alloca) {
    print("{} <- alloca {}", alloca->lhs.name(), data_type_name(alloca->type));
  }

  void visit(BinaryOpStatement *bin) {
    print("{} <- {} {} {}", bin->ret.name(), binary_type_name(bin->type),
          bin->lhs->name(), bin->rhs->name());
  }

  void visit(IfStatement *if_stmt) {
    print("if {} {{", if_stmt->condition->serialize());
    if (if_stmt->true_statements)
      if_stmt->true_statements->accept(this);
    if (if_stmt->false_statements) {
      print("}} else {{");
      if_stmt->false_statements->accept(this);
    }
    print("}}");
  }

  void visit(PrintStatement *print_stmt) {
    print("print {}", print_stmt->expr.serialize());
  }

  void visit(ConstStatement *const_stmt) {
    print("{} = const<{}>({})", const_stmt->ret.name(),
          data_type_name(const_stmt->data_type), const_stmt->value);
  }

  void visit(ForStatement *for_stmt) {
    print("for {} in range({}, {}) {{", for_stmt->loop_var_id.name(),
          for_stmt->begin->serialize(), for_stmt->end->serialize());
    for_stmt->body->accept(this);
    print("}}");
  }

  void visit(LocalLoadStmt *stmt) {
    print("{} <- load {}", stmt->name(), stmt->id.name());
  }

  void visit(LocalStoreStmt *stmt) {
    print("store {} -> {}", stmt->stmt->name(), stmt->id.name());
  }
};

class ASTModifiedException {};

// Lower Expr tree to a bunch of binary/unary(binary/unary) statements
// Goal: eliminate Expression, and mutable local variables. Make AST SSA.
class LowerAST : public ASTVisitor {
 public:
  LowerAST() {
  }

  VecStatement expand(ExprH expr) {
    auto ret = VecStatement();
    expr->flatten(ret);
    return ret;
  }

  void visit(StatementList *stmt_list) {
    for (auto &stmt : stmt_list->statements) {
      stmt->accept(this);
    }
  }

  void visit(AllocaStatement *alloca) {
    // print("{} <- alloca {}", alloca->lhs.name(),
    // data_type_name(alloca->type));
  }

  void visit(BinaryOpStatement *bin) {  // this will not appear here
  }

  void visit(IfStatement *if_stmt) {
    // TODO
    if (if_stmt->true_statements)
      if_stmt->true_statements->accept(this);
    if (if_stmt->false_statements) {
      if_stmt->false_statements->accept(this);
    }
  }

  void visit(LocalLoadStmt *) {
  }

  void visit(LocalStoreStmt *) {
  }

  void visit(PrintStatement *print_stmt) {
  }

  void visit(ConstStatement *const_stmt) {  // this will not appear here
  }

  void visit(ForStatement *for_stmt) {
    for_stmt->body->accept(this);
  }

  void visit(AssignmentStatement *assign) {
    // expand rhs
    auto expr = assign->rhs;
    VecStatement flattened;
    expr->flatten(flattened);
    if (true) {  // local variable
      // emit local store stmt
      auto local_store =
          std::make_unique<LocalStoreStmt>(assign->id, flattened.back().get());
      flattened.push_back(std::move(local_store));
    } else {  // global variable
    }
    assign->parent->replace_with(assign, flattened);
    throw ASTModifiedException();
  }

  static void run(ASTNode *node) {
    LowerAST inst;
    while (true) {
      bool modified = false;
      try {
        node->accept(&inst);
      } catch (ASTModifiedException) {
        modified = true;
      }
      if (!modified)
        break;
    }
  }
};

class TypeCheck : public ASTVisitor {};

#define declare(x) auto x = ExpressionHandle(std::make_shared<IdExpression>());

auto test_ast = []() {
  CoreState::set_trigger_gdb_when_crash(true);
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

  LowerAST::run(context.root());
  ASTPrinter::run(context.root());

};
TC_REGISTER_TASK(test_ast);

TLANG_NAMESPACE_END
