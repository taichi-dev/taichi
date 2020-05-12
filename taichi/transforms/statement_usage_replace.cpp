#include "taichi/ir/ir.h"
#include "taichi/ir/transforms.h"
#include "taichi/ir/visitors.h"

TLANG_NAMESPACE_BEGIN

// Replace all usages statement A with a new statement B.
// Note that the original statement A is NOT replaced.
class StatementUsageReplace : public IRVisitor {
 public:
  Stmt *old_stmt, *new_stmt;

  StatementUsageReplace(Stmt *old_stmt, Stmt *new_stmt)
      : old_stmt(old_stmt), new_stmt(new_stmt) {
    allow_undefined_visitor = true;
    invoke_default_visitor = true;
  }

  void visit(Stmt *stmt) {
    // if (stmt->may_have_operand(old_stmt))
    stmt->replace_operand_with(old_stmt, new_stmt);
  }

  void visit(WhileStmt *stmt) {
    stmt->replace_operand_with(old_stmt, new_stmt);
    stmt->body->accept(this);
  }

  void visit(IfStmt *if_stmt) {
    if_stmt->replace_operand_with(old_stmt, new_stmt);
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
    stmt->replace_operand_with(old_stmt, new_stmt);
    stmt->body->accept(this);
  }

  void visit(StructForStmt *stmt) {
    stmt->body->accept(this);
  }

  void visit(OffloadedStmt *stmt) {
    if (stmt->body)
      stmt->body->accept(this);
  }

  static void run(IRNode *node, Stmt *old_stmt, Stmt *new_stmt) {
    StatementUsageReplace inst(old_stmt, new_stmt);
    node->accept(&inst);
  }
};

namespace irpass {

void replace_all_usages_with(IRNode *root, Stmt *old_stmt, Stmt *new_stmt) {
  StatementUsageReplace::run(root, old_stmt, new_stmt);
}

}  // namespace irpass

TLANG_NAMESPACE_END
