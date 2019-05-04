#include "../ir.h"

TLANG_NAMESPACE_BEGIN

class LowerAccess : public IRVisitor {
 public:
  LowerAccess() {
    // TODO: change this to false
    allow_undefined_visitor = true;
  }

  /*
  void visit(Block *stmt_list) override {
    auto backup_block = current_block;
    current_block = stmt_list;
    for (auto &stmt : stmt_list->statements) {
      stmt->accept(this);
    }
    current_block = backup_block;
  }

  void visit(IfStmt *if_stmt) override {
    if (if_stmt->true_statements)
      if_stmt->true_statements->accept(this);
    if (if_stmt->false_statements) {
      if_stmt->false_statements->accept(this);
    }
  }

  void visit(WhileStmt *stmt) override {
    stmt->body->accept(this);
  }

  void visit(RangeForStmt *for_stmt) override {
    for_stmt->body->accept(this);
  }

  void visit(StructForStmt *for_stmt) override {
    for_stmt->body->accept(this);
  }
  */

  static void run(IRNode *node) {
    LowerAccess inst;
    while (true) {
      bool modified = false;
      try {
        node->accept(&inst);
      } catch (IRModifiedException) {
        modified = true;
      }
      if (!modified)
        break;
    }
  }
};

namespace irpass {

void lower_access(IRNode *root) {
  return LowerAccess::run(root);
}

}  // namespace irpass

TLANG_NAMESPACE_END