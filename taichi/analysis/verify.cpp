#include "taichi/ir/ir.h"
#include "taichi/ir/analysis.h"
#include "taichi/ir/visitors.h"

#include <vector>
#include <unordered_set>

TLANG_NAMESPACE_BEGIN

class IRVerifier : public BasicStmtVisitor {
 private:
  Block *current_block;
  // each scope corresponds to an unordered_set
  std::vector<std::unordered_set<Stmt *>> visible_stmts;

 public:
  using BasicStmtVisitor::visit;

  explicit IRVerifier(IRNode *root) : current_block(nullptr) {
    allow_undefined_visitor = true;
    invoke_default_visitor = true;
    TI_ASSERT(root->is<Block>() && root->as<Block>()->parent == nullptr);
  }

  void basic_verify(Stmt *stmt) {
    TI_ASSERT_INFO(stmt->parent == current_block,
                   "stmt({})->parent({}) != current_block({})", stmt->id,
                   fmt::ptr(stmt->parent), fmt::ptr(current_block));
    for (auto &op : stmt->get_operands()) {
      if (op == nullptr)
        continue;
      bool found = false;
      for (int depth = (int)visible_stmts.size() - 1; depth >= 0; depth--) {
        if (visible_stmts[depth].find(op) != visible_stmts[depth].end()) {
          found = true;
          break;
        }
      }
      TI_ASSERT_INFO(
          found,
          "IR broken: stmt {} cannot have operand {}."
          " Consider adding `ti.core.toggle_advance_optimization(False)`?"
          " If that fixes the problem, please report this bug by opening an"
          " issue at https://github.com/taichi-dev/taichi to help us improve,"
          " many thanks!",
          stmt->id, op->id);
    }
    visible_stmts.back().insert(stmt);
  }

  void preprocess_container_stmt(Stmt *stmt) override {
    basic_verify(stmt);
  }

  void visit(Stmt *stmt) override {
    basic_verify(stmt);
  }

  void visit(Block *stmt_list) override {
    TI_ASSERT(stmt_list->parent == current_block);
    auto backup_block = current_block;
    current_block = stmt_list;
    visible_stmts.emplace_back();
    for (auto &stmt : stmt_list->statements) {
      stmt->accept(this);
    }
    current_block = backup_block;
    visible_stmts.pop_back();
  }

  void visit(LocalLoadStmt *stmt) override {
    basic_verify(stmt);
    for (int i = 0; i < stmt->width(); i++) {
      TI_ASSERT(stmt->ptr[i].var->is<AllocaStmt>());
    }
  }

  void visit(LocalStoreStmt *stmt) override {
    basic_verify(stmt);
    TI_ASSERT(stmt->ptr->is<AllocaStmt>());
  }

  void visit(LoopIndexStmt *stmt) override {
    basic_verify(stmt);
    TI_ASSERT(stmt->loop);
    if (stmt->loop->is<OffloadedStmt>()) {
      TI_ASSERT(stmt->loop->as<OffloadedStmt>()->task_type ==
                    OffloadedStmt::TaskType::struct_for ||
                stmt->loop->as<OffloadedStmt>()->task_type ==
                    OffloadedStmt::TaskType::range_for);
    } else {
      TI_ASSERT(stmt->loop->is<StructForStmt>() ||
                stmt->loop->is<RangeForStmt>());
    }
  }

  static void run(IRNode *root) {
    IRVerifier verifier(root);
    root->accept(&verifier);
  }
};

namespace irpass::analysis {
void verify(IRNode *root) {
  TI_AUTO_PROF;
  IRVerifier::run(root);
}
}  // namespace irpass::analysis

TLANG_NAMESPACE_END
