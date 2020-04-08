#include "taichi/ir/ir.h"

TLANG_NAMESPACE_BEGIN

// Whole Kernel Common Subexpression Elimination
class WholeKernelCSE : public BasicStmtVisitor {
 public:
  using BasicStmtVisitor::visit;

  WholeKernelCSE() {}

  void visit(IfStmt *if_stmt) override {
    // Move common statements at the beginning or the end of both branches
    // outside.
    if (if_stmt->true_statements && if_stmt->false_statements) {
      auto block = if_stmt->parent;
      int current_stmt_id = block->locate(if_stmt);
      std::cout << "wholekernel cse if" << std::endl;
      irpass::print(block->statements[current_stmt_id - 1].get());
      irpass::print(if_stmt);
      auto &true_clause = if_stmt->true_statements;
      auto &false_clause = if_stmt->false_statements;
      if (irpass::same_statements(true_clause->statements[0].get(),
                                  false_clause->statements[0].get())) {
        std::cout << "same!";

        irpass::print(true_clause->statements[0].get());
        irpass::print(false_clause->statements[0].get());
        auto common_stmt = std::move(true_clause->statements[0]);
        irpass::replace_all_usages_with(false_clause.get(),
                                        false_clause->statements[0].get(),
                                        common_stmt.get());
        true_clause->statements.erase(true_clause->statements.begin());
        if_stmt->insert_before_me(std::move(common_stmt));
        false_clause->erase(0);
        irpass::print(block->statements[current_stmt_id].get());
        irpass::print(if_stmt);
        throw IRModified();
      }
      std::cout << "wholekernel cse if done" << std::endl;
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
        std::cout << "modified!\n";
        irpass::print(node);
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
