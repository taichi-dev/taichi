#include "taichi/ir/ir.h"
#include <vector>
#include <unordered_set>

TLANG_NAMESPACE_BEGIN

class IRVerifier : public IRVisitor {
 private:
  Block *current_block;

 public:
  explicit IRVerifier(IRNode *root) {
    allow_undefined_visitor = true;
    invoke_default_visitor = true;
    if (root->is<Block>())
      // for checking the Block's parent
      current_block = root->as<Block>()->parent;
    else {
      TI_ASSERT(root->is<Stmt>());
      current_block = root->as<Stmt>()->parent;
    }
  }

  void basic_verify(Stmt *stmt) {
    TI_ASSERT(stmt->parent == current_block);
  }

  void visit(Stmt *stmt) {
    basic_verify(stmt);
  }

  void visit(Block *stmt_list) {
    TI_ASSERT(stmt_list->parent == current_block);
    auto backup_block = current_block;
    current_block = stmt_list;
    for (auto &stmt : stmt_list->statements) {
      stmt->accept(this);
    }
    current_block = backup_block;
  }

  static void run(IRNode *root) {
    IRVerifier verifier(root);
    root->accept(&verifier);
  }
};

namespace irpass {
void verify(IRNode *root) {
  IRVerifier::run(root);
}
}  // namespace irpass

TLANG_NAMESPACE_END
