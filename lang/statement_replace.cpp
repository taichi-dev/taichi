#include "ir.h"

TLANG_NAMESPACE_BEGIN

class StatementReplace : public IRVisitor {
 public:
  Stmt *old_stmt, *new_stmt;

  StatementReplace(Stmt *old_stmt, Stmt *new_stmt)
      : old_stmt(old_stmt), new_stmt(new_stmt) {
    allow_undefined_visitor = true;
    invoke_default_visitor = true;
  }

  void visit(Statement *stmt) {
    for (int i = 0; i < stmt->num_operands(); i++) {
      if (stmt->operand(i) == old_stmt) {
        stmt->operand(i) = new_stmt;
      }
    }
  }

  void visit(IfStmt *if_stmt) {
    if (if_stmt->true_statements)
      if_stmt->true_statements->accept(this);
    if (if_stmt->false_statements) {
      if_stmt->false_statements->accept(this);
    }
  }

  void visit(Block *stmt_list) {
    for (auto &stmt : stmt_list->statements) {
      stmt->accept(this);
    }
  }

  void visit(RangeForStmt *stmt) {
    stmt->body->accept(this);
  }

  static void run(IRNode *node, Stmt *old_stmt, Stmt *new_stmt) {
    StatementReplace inst(old_stmt, new_stmt);
    node->accept(&inst);
  }
};

namespace irpass {

void replace_all_usages_with(IRNode *root, Stmt *old_stmt, Stmt *new_stmt) {
  StatementReplace::run(root, old_stmt, new_stmt);
}

}  // namespace irpass

TLANG_NAMESPACE_END
