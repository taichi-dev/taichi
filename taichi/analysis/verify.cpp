#include "taichi/ir/ir.h"
#include <vector>
#include <unordered_set>

TLANG_NAMESPACE_BEGIN

class IRVerifier : public IRVisitor {
 private:
  Block *current_block;
  // each scope corresponds to an unordered_set
  std::vector<std::unordered_set<Stmt *>> visible_stmts;

 public:
  explicit IRVerifier(IRNode *root) {
    allow_undefined_visitor = true;
    invoke_default_visitor = true;

    visible_stmts.emplace_back();
    if (root->is<Block>()) {
      TI_ASSERT(root->as<Block>()->parent == nullptr);
      // for checking the Block's parent
      current_block = root->as<Block>()->parent;
    }
    else {
      TI_ASSERT(root->is<Stmt>());
      current_block = root->as<Stmt>()->parent;
    }
  }

  void basic_verify(Stmt *stmt) {
    TI_ASSERT(stmt->parent == current_block);
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
//      TI_ASSERT(found);
    }
    visible_stmts.back().insert(stmt);
  }

  void visit(Stmt *stmt) {
    basic_verify(stmt);
  }

  void visit(Block *stmt_list) {
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
