#include "../ir.h"

TLANG_NAMESPACE_BEGIN

// Dead Instruction Elimination
class DIE : public IRVisitor {
 public:
  int id_counter;

  DIE(IRNode *node) {
    allow_undefined_visitor = true;
    invoke_default_visitor = true;
    id_counter = 0;
    node->accept(this);
  }

  void visit(Stmt *stmt) {
  }

  void visit(Block *stmt_list) {
    for (auto &stmt : stmt_list->statements) {
      stmt->accept(this);
    }
  }

  void visit(IfStmt *if_stmt) {
    if (if_stmt->true_statements)
      if_stmt->true_statements->accept(this);
    if (if_stmt->false_statements) {
      if_stmt->false_statements->accept(this);
    }
  }

  void visit(WhileStmt *stmt) {
    stmt->body->accept(this);
  }

  void visit(RangeForStmt *for_stmt) {
    for_stmt->body->accept(this);
  }

  void visit(StructForStmt *for_stmt) {
    for_stmt->body->accept(this);
  }
};

namespace irpass {

void die(IRNode *root) {
  DIE instance(root);
}

}  // namespace irpass

TLANG_NAMESPACE_END
