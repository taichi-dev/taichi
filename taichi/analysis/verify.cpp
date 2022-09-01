#include <vector>
#include <unordered_set>

#include "taichi/ir/ir.h"
#include "taichi/ir/analysis.h"
#include "taichi/ir/statements.h"
#include "taichi/ir/visitors.h"
#include "taichi/ir/transforms.h"
#include "taichi/system/profiler.h"

TLANG_NAMESPACE_BEGIN

class IRVerifier : public BasicStmtVisitor {
 private:
  Block *current_block_;
  Stmt *current_container_stmt_;
  // each scope corresponds to an unordered_set
  std::vector<std::unordered_set<Stmt *>> visible_stmts_;

 public:
  using BasicStmtVisitor::visit;

  explicit IRVerifier(IRNode *root)
      : current_block_(nullptr), current_container_stmt_(nullptr) {
    allow_undefined_visitor = true;
    invoke_default_visitor = true;
    if (!root->is<Block>())
      visible_stmts_.emplace_back();
    if (root->is<Stmt>() && root->as<Stmt>()->is_container_statement()) {
      current_container_stmt_ = root->as<Stmt>();
    }
  }

  void basic_verify(Stmt *stmt) {
    TI_ASSERT_INFO(stmt->parent == current_block_,
                   "stmt({})->parent({}) != current_block({})", stmt->id,
                   fmt::ptr(stmt->parent), fmt::ptr(current_block_));
    for (auto &op : stmt->get_operands()) {
      if (op == nullptr)
        continue;
      bool found = false;
      for (int depth = (int)visible_stmts_.size() - 1; depth >= 0; depth--) {
        if (visible_stmts_[depth].find(op) != visible_stmts_[depth].end()) {
          found = true;
          break;
        }
      }
      TI_ASSERT_INFO(found,
                     "IR broken: stmt {} {} cannot have operand {} {}."
                     " If you are using autodiff, please check out"
                     " https://docs.taichi-lang.org/docs/"
                     "differences_between_taichi_and_python_programs"
                     " If it doesn't help, please open an issue at"
                     " https://github.com/taichi-dev/taichi to help us improve."
                     " Thanks in advance!",
                     stmt->type(), stmt->id, op->type(), op->id);
    }
    visible_stmts_.back().insert(stmt);
  }

  void preprocess_container_stmt(Stmt *stmt) override {
    basic_verify(stmt);
  }

  void visit(Stmt *stmt) override {
    basic_verify(stmt);
  }

  void visit(Block *block) override {
    TI_ASSERT_INFO(
        block->parent_stmt == current_container_stmt_,
        "block({})->parent({}) != current_container_stmt({})", fmt::ptr(block),
        block->parent_stmt ? block->parent_stmt->name() : "nullptr",
        current_container_stmt_ ? current_container_stmt_->name() : "nullptr");
    auto backup_block = current_block_;
    current_block_ = block;
    auto backup_container_stmt = current_container_stmt_;
    if (!block->parent_stmt || !block->parent_stmt->is<OffloadedStmt>())
      visible_stmts_.emplace_back();
    for (auto &stmt : block->statements) {
      if (stmt->is_container_statement())
        current_container_stmt_ = stmt.get();
      stmt->accept(this);
      if (stmt->is_container_statement())
        current_container_stmt_ = backup_container_stmt;
    }
    current_block_ = backup_block;
    if (!block->parent_stmt || !block->parent_stmt->is<OffloadedStmt>())
      current_block_ = backup_block;
  }

  void visit(OffloadedStmt *stmt) override {
    basic_verify(stmt);
    if (stmt->has_body() && !stmt->body) {
      TI_ERROR("offloaded {} ({})->body is nullptr",
               offloaded_task_type_name(stmt->task_type), stmt->name());
    } else if (!stmt->has_body() && stmt->body) {
      TI_ERROR("offloaded {} ({})->body is {} (should be nullptr)",
               offloaded_task_type_name(stmt->task_type), stmt->name(),
               fmt::ptr(stmt->body));
    }
    stmt->all_blocks_accept(this);
  }

  void visit(LocalLoadStmt *stmt) override {
    basic_verify(stmt);
    TI_ASSERT(stmt->src->is<AllocaStmt>() || stmt->src->is<PtrOffsetStmt>());
  }

  void visit(LocalStoreStmt *stmt) override {
    basic_verify(stmt);
    TI_ASSERT(stmt->dest->is<AllocaStmt>() ||
              (stmt->dest->is<PtrOffsetStmt>() &&
               stmt->dest->cast<PtrOffsetStmt>()->is_local_ptr()));
  }

  void visit(LoopIndexStmt *stmt) override {
    basic_verify(stmt);
    TI_ASSERT(stmt->loop);
    if (stmt->loop->is<OffloadedStmt>()) {
      TI_ASSERT(stmt->loop->as<OffloadedStmt>()->task_type ==
                    OffloadedStmt::TaskType::struct_for ||
                stmt->loop->as<OffloadedStmt>()->task_type ==
                    OffloadedStmt::TaskType::mesh_for ||
                stmt->loop->as<OffloadedStmt>()->task_type ==
                    OffloadedStmt::TaskType::range_for);
    } else {
      TI_ASSERT(stmt->loop->is<StructForStmt>() ||
                stmt->loop->is<MeshForStmt>() ||
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
  if (!root->is<Block>() && !root->is<OffloadedStmt>()) {
    TI_WARN(
        "IR root is neither a Block nor an OffloadedStmt."
        " Skipping verification.");
  } else {
    IRVerifier::run(root);
  }
}
}  // namespace irpass::analysis

TLANG_NAMESPACE_END
