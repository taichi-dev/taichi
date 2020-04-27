#include "taichi/ir/ir.h"

TLANG_NAMESPACE_BEGIN

// Flag accesses to be either weak (non-activating) or strong (activating)
class FlagAccess : public IRVisitor {
 public:
  FlagAccess(IRNode *node) {
    allow_undefined_visitor = true;
    invoke_default_visitor = false;
    node->accept(this);
  }

  void visit(Block *stmt_list) {  // block itself has no id
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

  void visit(OffloadedStmt *stmt) {
    if (stmt->body)
      stmt->body->accept(this);
  }

  // Assuming pointers will be visited before global load/st
  void visit(GlobalPtrStmt *stmt) {
    stmt->activate = false;
  }

  void visit(GlobalStoreStmt *stmt) {
    if (stmt->ptr->is<GlobalPtrStmt>()) {
      stmt->ptr->as<GlobalPtrStmt>()->activate = true;
    }
  }

  void visit(AtomicOpStmt *stmt) {
    if (stmt->dest->is<GlobalPtrStmt>()) {
      stmt->dest->as<GlobalPtrStmt>()->activate = true;
    }
  }
};

namespace irpass {

void flag_access(IRNode *root) {
  FlagAccess instance(root);
}

}  // namespace irpass

TLANG_NAMESPACE_END
