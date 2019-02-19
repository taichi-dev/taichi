#include "ir.h"

TLANG_NAMESPACE_BEGIN

class IRPrinter : public IRVisitor {
 public:
  int current_indent;

  IRPrinter() {
    current_indent = -1;
  }

  template <typename... Args>
  void print(std::string f, Args &&... args) {
    for (int i = 0; i < current_indent; i++)
      fmt::print("  ");
    fmt::print(f, std::forward<Args>(args)...);
    fmt::print("\n");
  }

  static void run(IRNode *node) {
    auto p = IRPrinter();
    node->accept(&p);
  }

  void visit(Block *stmt_list) {
    current_indent++;
    for (auto &stmt : stmt_list->statements) {
      stmt->accept(this);
    }
    current_indent--;
  }

  void visit(AssignStmt *assign) {
    print("{} = {}", assign->id.name(), assign->rhs->serialize());
  }

  void visit(AllocaStmt *alloca) {
    print("{}alloca {}", alloca->type_hint(), alloca->ident.name());
  }

  void visit(BinaryOpStmt *bin) {
    print("{}{} = {} {} {}", bin->type_hint(), bin->name(),
          binary_type_name(bin->op_type), bin->lhs->name(), bin->rhs->name());
  }

  void visit(IfStmt *if_stmt) {
    print("if {} {{", if_stmt->cond->name());
    if (if_stmt->true_statements)
      if_stmt->true_statements->accept(this);
    if (if_stmt->false_statements) {
      print("}} else {{");
      if_stmt->false_statements->accept(this);
    }
    print("}}");
  }

  void visit(FrontendIfStmt *if_stmt) {
    print("if {} {{", if_stmt->condition->serialize());
    if (if_stmt->true_statements)
      if_stmt->true_statements->accept(this);
    if (if_stmt->false_statements) {
      print("}} else {{");
      if_stmt->false_statements->accept(this);
    }
    print("}}");
  }

  void visit(FrontendPrintStmt *print_stmt) {
    print("print {}", print_stmt->expr.serialize());
  }

  void visit(PrintStmt *print_stmt) {
    print("{}print {}", print_stmt->type_hint(), print_stmt->stmt->name());
  }

  void visit(ConstStmt *const_stmt) {
    print("{}{} = const {}", const_stmt->type_hint(), const_stmt->name(),
          const_stmt->value);
  }

  void visit(FrontendForStmt *for_stmt) {
    print("for {} in range({}, {}) {{", for_stmt->loop_var_id.name(),
          for_stmt->begin->serialize(), for_stmt->end->serialize());
    for_stmt->body->accept(this);
    print("}}");
  }

  void visit(RangeForStmt *for_stmt) {
    print("for {} in range({}, {}) {{", for_stmt->loop_var.name(),
          for_stmt->begin->name(), for_stmt->end->name());
    for_stmt->body->accept(this);
    print("}}");
  }

  void visit(LocalLoadStmt *stmt) {
    print("{}{} = load {}", stmt->type_hint(), stmt->name(),
          stmt->ident.name());
  }

  void visit(LocalStoreStmt *stmt) {
    print("[store] {} = {}", stmt->ident.name(), stmt->stmt->name());
  }
};

namespace irpass {

void print(IRNode *root) {
  return IRPrinter::run(root);
}

}  // namespace irpass

TLANG_NAMESPACE_END
