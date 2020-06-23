#include "taichi/ir/ir.h"
#include "taichi/ir/visitors.h"
#include "taichi/ir/transforms.h"

TLANG_NAMESPACE_BEGIN

// Eliminate useless ContinueStmt
class UselessContinueEliminator : public IRVisitor {
 public:
  bool modified;

  UselessContinueEliminator() : modified(false) {
    allow_undefined_visitor = true;
  }

  void visit(ContinueStmt *stmt) override {
    stmt->parent->erase(stmt);
    modified = true;
  }

  void visit(IfStmt *if_stmt) override {
    if (if_stmt->true_statements && if_stmt->true_statements->size())
      if_stmt->true_statements->back()->accept(this);
    if (if_stmt->false_statements && if_stmt->false_statements->size())
      if_stmt->false_statements->back()->accept(this);
  }
};

// Eliminate useless ContinueStmt and the statements after ContinueStmt
class ContinueStmtOptimizer : public BasicStmtVisitor {
 public:
  using BasicStmtVisitor::visit;
  bool modified;
  UselessContinueEliminator useless_continue_eliminator;

  ContinueStmtOptimizer() : modified(false) {
    allow_undefined_visitor = true;
  }

  void visit_loop(Stmt *loop_stmt, Block *body) {
    const int body_size = body->size();
    for (int i = 0; i < body_size - 1; i++) {
      if (auto continue_stmt = body->statements[i]->cast<ContinueStmt>()) {
        TI_ASSERT(continue_stmt->scope == loop_stmt ||
                  continue_stmt->scope == nullptr);
        // Eliminate statements after ContinueStmt
        for (int j = body_size - 1; j > i; j--)
          body->erase(j);
        modified = true;
      }
    }
    if (body->size())
      body->back()->accept(&useless_continue_eliminator);
    body->accept(this);
  }

  void visit(RangeForStmt *stmt) override {
    visit_loop(stmt, stmt->body.get());
  }

  void visit(StructForStmt *stmt) override {
    visit_loop(stmt, stmt->body.get());
  }

  void visit(WhileStmt *stmt) override {
    visit_loop(stmt, stmt->body.get());
  }

  void visit(OffloadedStmt *stmt) override {
    if (stmt->prologue)
      stmt->prologue->accept(this);
    if (stmt->task_type == OffloadedStmt::TaskType::range_for ||
        stmt->task_type == OffloadedStmt::TaskType::struct_for)
      visit_loop(stmt, stmt->body.get());
    else if (stmt->body)
      stmt->body->accept(this);
    if (stmt->epilogue)
      stmt->epilogue->accept(this);
  }

  static bool run(IRNode *node) {
    ContinueStmtOptimizer optimizer;
    node->accept(&optimizer);
    return optimizer.modified || optimizer.useless_continue_eliminator.modified;
  }
};

namespace irpass {
bool continue_stmt_optimization(IRNode *root) {
  TI_AUTO_PROF;
  return ContinueStmtOptimizer::run(root);
}
}  // namespace irpass

TLANG_NAMESPACE_END
