#include "taichi/ir/ir.h"
#include "taichi/ir/analysis.h"
#include "taichi/ir/statements.h"
#include "taichi/ir/transforms.h"
#include "taichi/ir/visitors.h"
#include "taichi/system/profiler.h"

#include <typeindex>

TLANG_NAMESPACE_BEGIN

// A helper class to maintain WholeKernelCSE::visited
class MarkUndone : public BasicStmtVisitor {
 private:
  std::unordered_set<int> *const visited_;
  Stmt *const modified_operand_;

 public:
  using BasicStmtVisitor::visit;

  MarkUndone(std::unordered_set<int> *visited, Stmt *modified_operand)
      : visited_(visited), modified_operand_(modified_operand) {
    allow_undefined_visitor = true;
    invoke_default_visitor = true;
  }

  void visit(Stmt *stmt) override {
    if (stmt->has_operand(modified_operand_)) {
      visited_->erase(stmt->instance_id);
    }
  }

  void preprocess_container_stmt(Stmt *stmt) override {
    if (stmt->has_operand(modified_operand_)) {
      visited_->erase(stmt->instance_id);
    }
  }

  static void run(std::unordered_set<int> *visited, Stmt *modified_operand) {
    MarkUndone marker(visited, modified_operand);
    modified_operand->get_ir_root()->accept(&marker);
  }
};

// Whole Kernel Common Subexpression Elimination
class WholeKernelCSE : public BasicStmtVisitor {
 private:
  std::unordered_set<int> visited_;
  // each scope corresponds to an unordered_set
  std::vector<std::unordered_map<std::size_t, std::unordered_set<Stmt *> > >
      visible_stmts_;
  DelayedIRModifier modifier_;

 public:
  using BasicStmtVisitor::visit;

  WholeKernelCSE() {
    allow_undefined_visitor = true;
    invoke_default_visitor = true;
  }

  bool is_done(Stmt *stmt) {
    return visited_.find(stmt->instance_id) != visited_.end();
  }

  void set_done(Stmt *stmt) {
    visited_.insert(stmt->instance_id);
  }

  static std::size_t operand_type_hash(const Stmt *stmt) {
    std::size_t hash_code{0};
    auto hash_type =
        std::hash<std::type_index>{}(std::type_index(typeid(stmt)));
    auto op = stmt->get_operands();
    for (auto &x : op) {
      if (x == nullptr)
        continue;
      hash_code =
          (hash_code) ^
          (std::hash<unsigned long>{}(reinterpret_cast<unsigned long>(x)));
    }
    return hash_type ^ hash_code;
  }

  static bool common_statement_eliminable(Stmt *this_stmt, Stmt *prev_stmt) {
    // Is this_stmt eliminable given that prev_stmt appears before it and has
    // the same type with it?
    if (this_stmt->type() != prev_stmt->type())
      return false;
    if (this_stmt->is<GlobalPtrStmt>()) {
      auto this_ptr = this_stmt->as<GlobalPtrStmt>();
      auto prev_ptr = prev_stmt->as<GlobalPtrStmt>();
      return irpass::analysis::definitely_same_address(this_ptr, prev_ptr) &&
             (this_ptr->activate == prev_ptr->activate || prev_ptr->activate);
    }
    if (this_stmt->is<LoopUniqueStmt>()) {
      auto this_loop_unique = this_stmt->as<LoopUniqueStmt>();
      auto prev_loop_unique = prev_stmt->as<LoopUniqueStmt>();
      if (irpass::analysis::same_value(this_loop_unique->input,
                                       prev_loop_unique->input)) {
        // Merge the "covers" information into prev_loop_unique.
        // Notice that this_loop_unique->covers is corrupted here.
        prev_loop_unique->covers.insert(this_loop_unique->covers.begin(),
                                        this_loop_unique->covers.end());
        return true;
      }
      return false;
    }
    return irpass::analysis::same_statements(this_stmt, prev_stmt);
  }

  void visit(Stmt *stmt) override {
    if (!stmt->common_statement_eliminable())
      return;
    // container_statement is no need to be CSE
    if (stmt->is_container_statement())
      return;
    // Generic visitor for all CSE-able statements.
    if (is_done(stmt)) {
      visible_stmts_.back()[operand_type_hash(stmt)].insert(stmt);
      return;
    }
    for (auto &scope : visible_stmts_) {
      for (auto &prev_stmt : scope[operand_type_hash(stmt)]) {
        if (common_statement_eliminable(stmt, prev_stmt)) {
          MarkUndone::run(&visited_, stmt);
          stmt->replace_usages_with(prev_stmt);
          modifier_.erase(stmt);
          return;
        }
      }
    }
    visible_stmts_.back()[operand_type_hash(stmt)].insert(stmt);
    set_done(stmt);
  }

  void visit(Block *stmt_list) override {
    visible_stmts_.emplace_back();
    for (auto &stmt : stmt_list->statements) {
      stmt->accept(this);
    }
    visible_stmts_.pop_back();
  }

  void visit(IfStmt *if_stmt) override {
    if (if_stmt->true_statements) {
      if (if_stmt->true_statements->statements.empty()) {
        if_stmt->set_true_statements(nullptr);
      }
    }

    if (if_stmt->false_statements) {
      if (if_stmt->false_statements->statements.empty()) {
        if_stmt->set_false_statements(nullptr);
      }
    }

    // Move common statements at the beginning or the end of both branches
    // outside.
    if (if_stmt->true_statements && if_stmt->false_statements) {
      auto &true_clause = if_stmt->true_statements;
      auto &false_clause = if_stmt->false_statements;
      if (irpass::analysis::same_statements(
              true_clause->statements[0].get(),
              false_clause->statements[0].get())) {
        // Directly modify this because it won't invalidate any iterators.
        auto common_stmt = true_clause->extract(0);
        irpass::replace_all_usages_with(false_clause.get(),
                                        false_clause->statements[0].get(),
                                        common_stmt.get());
        modifier_.insert_before(if_stmt, std::move(common_stmt));
        false_clause->erase(0);
      }
      if (!true_clause->statements.empty() &&
          !false_clause->statements.empty() &&
          irpass::analysis::same_statements(
              true_clause->statements.back().get(),
              false_clause->statements.back().get())) {
        // Directly modify this because it won't invalidate any iterators.
        auto common_stmt = true_clause->extract((int)true_clause->size() - 1);
        irpass::replace_all_usages_with(false_clause.get(),
                                        false_clause->statements.back().get(),
                                        common_stmt.get());
        modifier_.insert_after(if_stmt, std::move(common_stmt));
        false_clause->erase((int)false_clause->size() - 1);
      }
    }

    if (if_stmt->true_statements)
      if_stmt->true_statements->accept(this);
    if (if_stmt->false_statements)
      if_stmt->false_statements->accept(this);
  }

  static bool run(IRNode *node) {
    WholeKernelCSE eliminator;
    bool modified = false;
    while (true) {
      node->accept(&eliminator);
      if (eliminator.modifier_.modify_ir())
        modified = true;
      else
        break;
    }
    return modified;
  }
};

namespace irpass {
bool whole_kernel_cse(IRNode *root) {
  TI_AUTO_PROF;
  return WholeKernelCSE::run(root);
}
}  // namespace irpass

TLANG_NAMESPACE_END
