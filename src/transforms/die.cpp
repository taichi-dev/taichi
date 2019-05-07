#include "../ir.h"
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
      irpass::print(node);
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
          break;
        }
        break;
      }
      if (!modified)
        break;
    }
  }

  void register_usage(Stmt *stmt) {
    TC_TAG;
    TC_P(stmt->id);
    TC_P(stmt->num_operands());
    int counter = 0;
    for (auto op : stmt->get_operands()) {
      TC_P(counter++);
      if (op) {  // might be nullptr
        TC_ASSERT(!op->erased);
        TC_P(op);
        if (used.find(op->instance_id) == used.end()) {
          used.insert(op->instance_id);
        }
        TC_P(op->id);
        TC_P(op->instance_id);
      }
    }
  }

  void visit(Stmt *stmt) {
    TC_ASSERT(!stmt->erased);
    if (phase == 0) {
      register_usage(stmt);
    } else {
      if (!stmt->has_side_effect() &&
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
};

namespace irpass {

void die(IRNode *root) {
  DIE instance(root);
}

}  // namespace irpass

TLANG_NAMESPACE_END
