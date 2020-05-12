// Dead Instruction Elimination

#include "taichi/ir/ir.h"
#include "taichi/ir/transforms.h"
#include "taichi/ir/visitors.h"
#include <unordered_set>

TLANG_NAMESPACE_BEGIN

// Dead Instruction Elimination
class DIE : public IRVisitor {
 public:
  std::unordered_set<int> used;
  int phase;  // 0: mark usage 1: eliminate

  DIE(IRNode *node) {
    allow_undefined_visitor = true;
    invoke_default_visitor = true;
    while (1) {
      bool modified = false;
      phase = 0;
      used.clear();
      node->accept(this);
      phase = 1;
      while (1) {
        try {
          node->accept(this);
        } catch (IRModified) {
          modified = true;
          continue;
        }
        break;
      }
      if (!modified)
        break;
    }
  }

  void register_usage(Stmt *stmt) {
    for (auto op : stmt->get_operands()) {
      if (op) {  // might be nullptr
        if (used.find(op->instance_id) == used.end()) {
          used.insert(op->instance_id);
        }
      }
    }
  }

  void visit(Stmt *stmt) {
    TI_ASSERT(!stmt->erased);
    if (phase == 0) {
      register_usage(stmt);
    } else {
      if (!stmt->has_global_side_effect() &&
          used.find(stmt->instance_id) == used.end()) {
        stmt->parent->erase(stmt);
        throw IRModified();
      }
    }
  }

  void visit(Block *stmt_list) {
    for (auto &stmt : stmt_list->statements) {
      stmt->accept(this);
    }
  }

  void visit(IfStmt *if_stmt) {
    register_usage(if_stmt);
    if (if_stmt->true_statements)
      if_stmt->true_statements->accept(this);
    if (if_stmt->false_statements) {
      if_stmt->false_statements->accept(this);
    }
  }

  void visit(WhileStmt *stmt) {
    register_usage(stmt);
    stmt->body->accept(this);
  }

  void visit(RangeForStmt *for_stmt) {
    register_usage(for_stmt);
    for_stmt->body->accept(this);
  }

  void visit(StructForStmt *for_stmt) {
    register_usage(for_stmt);
    for_stmt->body->accept(this);
  }

  void visit(OffloadedStmt *stmt) {
    if (stmt->body)
      stmt->body->accept(this);
  }
};

namespace irpass {

void die(IRNode *root) {
  DIE instance(root);
}

}  // namespace irpass

TLANG_NAMESPACE_END
