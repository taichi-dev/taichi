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
      TI_ASSERT_INFO(found, "stmt {} cannot have operand {}.", stmt->id,
                     op->id);
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

  void visit(RangeForStmt *for_stmt) override {
    basic_verify(for_stmt);
    if (for_stmt->loop_var) {
      TI_ASSERT(for_stmt->loop_var->is<AllocaStmt>());
      TI_ASSERT_INFO(irpass::analysis::gather_statements(
                         for_stmt->loop_var->parent,
                         [&](Stmt *s) {
                           if (auto store = s->cast<LocalStoreStmt>())
                             return store->ptr == for_stmt->loop_var;
                           else if (auto atomic = s->cast<AtomicOpStmt>()) {
                             return atomic->dest == for_stmt->loop_var;
                           } else {
                             return false;
                           }
                         })
                         .empty(),
                     "loop_var of {} modified", for_stmt->id);
    }
    for_stmt->body->accept(this);
  }

  void visit(StructForStmt *for_stmt) override {
    basic_verify(for_stmt);
    for (auto &loop_var : for_stmt->loop_vars) {
      if (!loop_var)
        continue;
      TI_ASSERT(loop_var->is<AllocaStmt>());
      TI_ASSERT_INFO(irpass::analysis::gather_statements(
                         loop_var->parent,
                         [&](Stmt *s) {
                           if (auto store = s->cast<LocalStoreStmt>())
                             return store->ptr == loop_var;
                           else if (auto atomic = s->cast<AtomicOpStmt>()) {
                             return atomic->dest == loop_var;
                           } else {
                             return false;
                           }
                         })
                         .empty(),
                     "loop_var of {} modified", for_stmt->id);
    }
    for_stmt->body->accept(this);
  }

  static void run(IRNode *root) {
    IRVerifier verifier(root);
    root->accept(&verifier);
  }
};

namespace irpass::analysis {
void verify(IRNode *root) {
  IRVerifier::run(root);
}
}  // namespace irpass::analysis

TLANG_NAMESPACE_END
