#include "taichi/ir/ir.h"

TLANG_NAMESPACE_BEGIN

class FieldsRegisteredChecker : public BasicStmtVisitor {
 public:
  using BasicStmtVisitor::visit;

  FieldsRegisteredChecker() {
    allow_undefined_visitor = true;
    invoke_default_visitor = true;
  }

  void visit(Stmt *stmt) override {
    TI_ASSERT(stmt->fields_registered);
  }

  void visit(IfStmt *if_stmt) override {
    TI_ASSERT(if_stmt->fields_registered);
    if (if_stmt->true_statements)
      if_stmt->true_statements->accept(this);
    if (if_stmt->false_statements) {
      if_stmt->false_statements->accept(this);
    }
  }

  void visit(WhileStmt *stmt) override {
    TI_ASSERT(stmt->fields_registered);
    stmt->body->accept(this);
  }

  void visit(RangeForStmt *for_stmt) override {
    TI_ASSERT(for_stmt->fields_registered);
    for_stmt->body->accept(this);
  }

  void visit(StructForStmt *for_stmt) override {
    TI_ASSERT(for_stmt->fields_registered);
    for_stmt->body->accept(this);
  }

  void visit(OffloadedStmt *stmt) override {
    TI_ASSERT(stmt->fields_registered);
    if (stmt->body)
      stmt->body->accept(this);
  }

  static void run(IRNode *root) {
    FieldsRegisteredChecker checker;
    root->accept(&checker);
  }
};

namespace irpass {
void check_fields_registered(IRNode *root) {
  return FieldsRegisteredChecker::run(root);
}
}  // namespace irpass

TLANG_NAMESPACE_END
