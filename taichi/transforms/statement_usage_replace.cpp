#include "taichi/ir/ir.h"
#include "taichi/ir/statements.h"
#include "taichi/ir/transforms.h"
#include "taichi/ir/visitors.h"
#include "taichi/system/profiler.h"

TLANG_NAMESPACE_BEGIN

// Replace all usages statement A with a new statement B.
// Note that the original statement A is NOT replaced.
class StatementUsageReplace : public IRVisitor {
  // The reason why we don't use BasicStmtVisitor is we don't want to go into
  // FrontendForStmt.
 public:
  Stmt *old_stmt, *new_stmt;

  StatementUsageReplace(Stmt *old_stmt, Stmt *new_stmt)
      : old_stmt(old_stmt), new_stmt(new_stmt) {
    allow_undefined_visitor = true;
    invoke_default_visitor = true;
  }

  void visit(Stmt *stmt) override {
    stmt->replace_operand_with(old_stmt, new_stmt);
  }

  void visit(WhileStmt *stmt) override {
    stmt->replace_operand_with(old_stmt, new_stmt);
    stmt->body->accept(this);
  }

  void visit(IfStmt *if_stmt) override {
    if_stmt->replace_operand_with(old_stmt, new_stmt);
    if (if_stmt->true_statements)
      if_stmt->true_statements->accept(this);
    if (if_stmt->false_statements) {
      if_stmt->false_statements->accept(this);
    }
  }

  void visit(Block *stmt_list) override {
    for (auto &stmt : stmt_list->statements) {
      stmt->accept(this);
    }
  }

  void visit(RangeForStmt *stmt) override {
    stmt->replace_operand_with(old_stmt, new_stmt);
    stmt->body->accept(this);
  }

  void visit(StructForStmt *stmt) override {
    stmt->body->accept(this);
  }

  void visit(MeshForStmt *stmt) override {
    stmt->body->accept(this);
  }

  void visit(OffloadedStmt *stmt) override {
    stmt->all_blocks_accept(this);
  }

  static void run(IRNode *root, Stmt *old_stmt, Stmt *new_stmt) {
    StatementUsageReplace replacer(old_stmt, new_stmt);
    if (root != nullptr) {
      // If root is specified, simply traverse the root.
      root->accept(&replacer);
      return;
    }

    // statements inside old_stmt->parent
    TI_ASSERT(old_stmt->parent != nullptr);
    old_stmt->parent->accept(&replacer);
    auto current_block = old_stmt->parent->parent_block();

    // statements outside old_stmt->parent: bottom-up
    while (current_block != nullptr) {
      for (auto &stmt : current_block->statements) {
        stmt->replace_operand_with(old_stmt, new_stmt);
      }
      current_block = current_block->parent_block();
    }
  }
};

namespace irpass {

void replace_all_usages_with(IRNode *root, Stmt *old_stmt, Stmt *new_stmt) {
  TI_AUTO_PROF;
  StatementUsageReplace::run(root, old_stmt, new_stmt);
}

}  // namespace irpass

TLANG_NAMESPACE_END
