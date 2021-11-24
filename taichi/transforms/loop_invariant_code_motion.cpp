#include "taichi/ir/ir.h"
#include "taichi/ir/statements.h"
#include "taichi/ir/transforms.h"
#include "taichi/ir/visitors.h"
#include "taichi/system/profiler.h"

#include <stack>

TLANG_NAMESPACE_BEGIN

class LoopInvariantCodeMotion : public BasicStmtVisitor {
 public:
  using BasicStmtVisitor::visit;

  std::stack<Block *> loop_blocks;

  const CompileConfig &config;

  DelayedIRModifier modifier;

  explicit LoopInvariantCodeMotion(const CompileConfig &config)
      : config(config) {
    allow_undefined_visitor = true;
  }

  bool stmt_can_be_moved(Stmt *stmt) {
    if (loop_blocks.size() <= 1 || (!config.move_loop_invariant_outside_if &&
                                    stmt->parent != loop_blocks.top()))
      return false;

    bool can_be_moved = true;

    Block *current_scope = stmt->parent;

    for (Stmt *operand : stmt->get_operands()) {
      if (operand->parent == current_scope) {
        // This statement has an operand that is in the current scope,
        // so it can not be moved out of the scope.
        can_be_moved = false;
        break;
      }
      if (config.move_loop_invariant_outside_if &&
          stmt->parent != loop_blocks.top()) {
        // If we enable moving code from a nested if block, we need to check
        // visibility. Example:
        // for i in range(10):
        //   a = x[0]
        //   if b:
        //     c = a + 1
        // Since we are moving statements outside the cloest for scope,
        // We need to check the scope of the operand
        Stmt *operand_parent = operand;
        while (operand_parent && operand_parent->parent) {
          operand_parent = operand_parent->parent->parent_stmt;
          if (!operand_parent)
            break;
          // If the one of the parent of the operand is the top loop scope
          // Then it will not be visible if we move it outside the top loop
          // scope
          if (operand_parent == loop_blocks.top()->parent_stmt) {
            can_be_moved = false;
            break;
          }
        }
        if (!can_be_moved)
          break;
      }
    }

    return can_be_moved;
  }

  void visit(BinaryOpStmt *stmt) override {
    if (stmt_can_be_moved(stmt)) {
      auto replacement = stmt->clone();
      stmt->replace_usages_with(replacement.get());

      modifier.insert_before(stmt->parent->parent_stmt, std::move(replacement));
      modifier.erase(stmt);
    }
  }

  void visit(UnaryOpStmt *stmt) override {
    if (stmt_can_be_moved(stmt)) {
      auto replacement = stmt->clone();
      stmt->replace_usages_with(replacement.get());

      modifier.insert_before(stmt->parent->parent_stmt, std::move(replacement));
      modifier.erase(stmt);
    }
  }

  void visit(Block *stmt_list) override {
    for (auto &stmt : stmt_list->statements)
      stmt->accept(this);
  }

  void visit_loop(Block *body) {
    loop_blocks.push(body);

    body->accept(this);

    loop_blocks.pop();
  }

  void visit(RangeForStmt *stmt) override {
    visit_loop(stmt->body.get());
  }

  void visit(StructForStmt *stmt) override {
    visit_loop(stmt->body.get());
  }

  void visit(MeshForStmt *stmt) override {
    visit_loop(stmt->body.get());
  }

  void visit(WhileStmt *stmt) override {
    visit_loop(stmt->body.get());
  }

  void visit(OffloadedStmt *stmt) override {
    if (stmt->tls_prologue)
      stmt->tls_prologue->accept(this);

    if (stmt->mesh_prologue)
      stmt->mesh_prologue->accept(this);

    if (stmt->bls_prologue)
      stmt->bls_prologue->accept(this);

    if (stmt->body) {
      if (stmt->task_type == OffloadedStmt::TaskType::range_for ||
          stmt->task_type == OffloadedStmt::TaskType::struct_for)
        visit_loop(stmt->body.get());
      else
        stmt->body->accept(this);
    }

    if (stmt->bls_epilogue)
      stmt->bls_epilogue->accept(this);

    if (stmt->tls_epilogue)
      stmt->tls_epilogue->accept(this);
  }

  static bool run(IRNode *node, const CompileConfig &config) {
    bool modified = false;

    while (true) {
      LoopInvariantCodeMotion eliminator(config);
      node->accept(&eliminator);
      if (eliminator.modifier.modify_ir())
        modified = true;
      else
        break;
    };

    return modified;
  }
};

namespace irpass {
bool loop_invariant_code_motion(IRNode *root, const CompileConfig &config) {
  TI_AUTO_PROF;
  return LoopInvariantCodeMotion::run(root, config);
}
}  // namespace irpass

TLANG_NAMESPACE_END
