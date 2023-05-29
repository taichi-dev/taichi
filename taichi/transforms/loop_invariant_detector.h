#include "taichi/ir/ir.h"
#include "taichi/ir/statements.h"
#include "taichi/ir/transforms.h"
#include "taichi/ir/visitors.h"
#include "taichi/system/profiler.h"

#include <stack>

namespace taichi::lang {

class LoopInvariantDetector : public BasicStmtVisitor {
 public:
  using BasicStmtVisitor::visit;

  std::vector<Block *> loop_blocks;

  const CompileConfig &config;

  explicit LoopInvariantDetector(const CompileConfig &config) : config(config) {
    allow_undefined_visitor = true;
  }

  bool is_operand_loop_invariant_impl(Stmt *operand,
                                      Block *current_scope,
                                      Block *loop_block = nullptr) {
    if (!loop_block) {
      loop_block = loop_blocks.back();
    }
    if (operand->parent == current_scope) {
      // This statement has an operand that is in the current scope,
      // so it can not be moved out of the scope.
      return false;
    }
    if (current_scope != loop_block) {
      // If we enable moving code from a nested if block, we need to check
      // visibility. Example:
      // for i in range(10):
      //   a = x[0]
      //   if b:
      //     c = a + 1
      // Since we are moving statements outside the closest for scope,
      // We need to check the scope of the operand
      Stmt *operand_parent = operand;
      while (operand_parent->parent) {
        operand_parent = operand_parent->parent->parent_stmt();
        if (!operand_parent)
          break;
        // If the one of the current_scope of the operand is the top loop
        // scope Then it will not be visible if we move it outside the top
        // loop scope
        if (operand_parent == loop_block->parent_stmt()) {
          return false;
        }
      }
    }
    return true;
  }

  bool is_operand_loop_invariant(Stmt *operand,
                                 Block *current_scope,
                                 int depth = -1) {
    if (depth == -1) {
      depth = loop_blocks.size() - 1;
    }
    if (depth <= 0)
      return false;
    return is_operand_loop_invariant_impl(operand, current_scope,
                                          loop_blocks[depth]);
  }

  bool is_loop_invariant(Stmt *stmt, Block *current_scope) {
    if (loop_blocks.size() <= 1 || (!config.move_loop_invariant_outside_if &&
                                    current_scope != loop_blocks.back()))
      return false;

    bool is_invariant = true;

    for (Stmt *operand : stmt->get_operands()) {
      if (operand == nullptr)
        continue;
      is_invariant &= is_operand_loop_invariant_impl(operand, current_scope);
    }

    return is_invariant;
  }

  Stmt *get_loop_stmt(int depth) {
    return loop_blocks[depth]->parent_stmt();
  }

  Stmt *current_loop_stmt() {
    return get_loop_stmt(loop_blocks.size() - 1);
  }
  void visit(Block *stmt_list) override {
    for (auto &stmt : stmt_list->statements)
      stmt->accept(this);
  }

  virtual void visit_loop(Block *body) {
    loop_blocks.push_back(body);

    body->accept(this);

    loop_blocks.pop_back();
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
          stmt->task_type == OffloadedTaskType::mesh_for ||
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
};

}  // namespace taichi::lang
