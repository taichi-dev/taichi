// Dead Instruction Elimination

#include "taichi/ir/ir.h"
#include "taichi/ir/statements.h"
#include "taichi/ir/transforms.h"
#include "taichi/ir/visitors.h"
#include "taichi/system/profiler.h"

#include <unordered_set>

TLANG_NAMESPACE_BEGIN

// Dead Instruction Elimination
class DIE : public IRVisitor {
 public:
  std::unordered_set<int> used;
  int phase;  // 0: mark usage 1: eliminate
  DelayedIRModifier modifier;
  bool modified_ir;

  DIE(IRNode *node) {
    allow_undefined_visitor = true;
    invoke_default_visitor = true;
    modified_ir = false;
    while (true) {
      bool modified = false;
      phase = 0;
      used.clear();
      node->accept(this);
      phase = 1;
      while (true) {
        node->accept(this);
        if (modifier.modify_ir()) {
          modified = true;
          modified_ir = true;
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

  void visit(Stmt *stmt) override {
    TI_ASSERT(!stmt->erased);
    if (phase == 0) {
      register_usage(stmt);
    } else {
      if (stmt->dead_instruction_eliminable() &&
          used.find(stmt->instance_id) == used.end()) {
        modifier.erase(stmt);
      }
    }
  }

  void visit(Block *stmt_list) override {
    for (auto &stmt : stmt_list->statements) {
      stmt->accept(this);
    }
  }

  void visit(IfStmt *if_stmt) override {
    register_usage(if_stmt);
    if (if_stmt->true_statements)
      if_stmt->true_statements->accept(this);
    if (if_stmt->false_statements) {
      if_stmt->false_statements->accept(this);
    }
  }

  void visit(WhileStmt *stmt) override {
    register_usage(stmt);
    stmt->body->accept(this);
  }

  void visit(RangeForStmt *for_stmt) override {
    register_usage(for_stmt);
    for_stmt->body->accept(this);
  }

  void visit(StructForStmt *for_stmt) override {
    register_usage(for_stmt);
    for_stmt->body->accept(this);
  }

  void visit(MeshForStmt *for_stmt) override {
    register_usage(for_stmt);
    for_stmt->body->accept(this);
  }

  void visit(OffloadedStmt *stmt) override {
    // TODO: A hack to make sure end_stmt is registered.
    // Ideally end_stmt should be its own Block instead.
    if (stmt->end_stmt &&
        used.find(stmt->end_stmt->instance_id) == used.end()) {
      used.insert(stmt->end_stmt->instance_id);
    }
    stmt->all_blocks_accept(this, true);
  }

  void visit(MatrixInitStmt *stmt) override {
    register_usage(stmt);
    for (auto &elts : stmt->values) {
      elts->accept(this);
    }
  }
};

namespace irpass {

bool die(IRNode *root) {
  TI_AUTO_PROF;
  DIE instance(root);
  return instance.modified_ir;
}

}  // namespace irpass

TLANG_NAMESPACE_END
