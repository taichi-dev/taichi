#include "taichi/ir/ir.h"
#include "taichi/ir/transforms.h"
#include "taichi/ir/analysis.h"
#include "taichi/ir/visitors.h"

TLANG_NAMESPACE_BEGIN

// Whole Kernel Common Subexpression Elimination
class WholeKernelCSE : public BasicStmtVisitor {
 public:
  using BasicStmtVisitor::visit;

  WholeKernelCSE() {
  }

  void visit(IfStmt *if_stmt) override {
    if (if_stmt->true_statements) {
      if (if_stmt->true_statements->statements.empty()) {
        if_stmt->true_statements = nullptr;
        throw IRModified();
      }
    }

    if (if_stmt->false_statements) {
      if (if_stmt->false_statements->statements.empty()) {
        if_stmt->false_statements = nullptr;
        throw IRModified();
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
        auto common_stmt = true_clause->extract(0);
        irpass::replace_all_usages_with(false_clause.get(),
                                        false_clause->statements[0].get(),
                                        common_stmt.get());
        if_stmt->insert_before_me(std::move(common_stmt));
        false_clause->erase(0);
        throw IRModified();
      }
      if (irpass::analysis::same_statements(
              true_clause->statements.back().get(),
              false_clause->statements.back().get())) {
        auto common_stmt = true_clause->extract((int)true_clause->size() - 1);
        irpass::replace_all_usages_with(false_clause.get(),
                                        false_clause->statements.back().get(),
                                        common_stmt.get());
        if_stmt->insert_before_me(std::move(common_stmt));
        false_clause->erase((int)false_clause->size() - 1);
        throw IRModified();
      }
    }
  }

  static void run(IRNode *node) {
    WholeKernelCSE eliminator;
    while (true) {
      bool modified = false;
      try {
        node->accept(&eliminator);
      } catch (IRModified) {
        modified = true;
      }
      if (!modified)
        break;
    }
  }
};

namespace irpass {

void whole_kernel_cse(IRNode *root) {
  WholeKernelCSE::run(root);
}

}  // namespace irpass

TLANG_NAMESPACE_END
