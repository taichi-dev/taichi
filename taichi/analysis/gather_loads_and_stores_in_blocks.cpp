#include "taichi/ir/ir.h"
#include <unordered_set>

TLANG_NAMESPACE_BEGIN

// Gather all loads, stores and atomic operations for each block.
class LoadsAndStoresSearcher : public BasicStmtVisitor {
 private:
  std::unordered_map<Block *, std::vector<Stmt *>> loads_and_stores;

 public:
  LoadsAndStoresSearcher() {
    allow_undefined_visitor = true;
    invoke_default_visitor = true;
  }

  void found(Stmt *store_stmt) {
    for (Block *block = store_stmt->parent; block != nullptr;
         block = block->parent) {
      stores[block].push_back(store_stmt);
    }
  }

  void visit(AtomicOpStmt *stmt) override {
    found(stmt);
  }
  void visit(LocalStoreStmt *stmt) override {
    found(stmt);
  }
  void visit(GlobalStoreStmt *stmt) override {
    found(stmt);
  }
  void visit(LocalLoadStmt *stmt) override {
    found(stmt);
  }
  void visit(GlobalLoadStmt *stmt) override {
    found(stmt);
  }

  static const std::unordered_map<Block *, std::vector<Stmt *>> &run(
      IRNode *root) {
    LoadsAndStoresSearcher searcher;
    root->accept(&searcher);
    return searcher.stores;
  }
};

namespace irpass::analysis {
const std::unordered_map<Block *, std::vector<Stmt *>>
    &gather_loads_and_stores_in_blocks(IRNode *root) {
  return LoadsAndStoresSearcher::run(root);
}
}  // namespace irpass::analysis

TLANG_NAMESPACE_END
