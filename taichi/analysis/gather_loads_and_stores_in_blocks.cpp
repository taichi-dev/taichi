#include "taichi/ir/ir.h"
#include <unordered_set>

TLANG_NAMESPACE_BEGIN

// Gather all loads, stores and atomic operations for each block.
class LoadsAndStoresSearcher : public BasicStmtVisitor {
 private:
  std::unique_ptr<std::unordered_map<Block *, std::vector<Stmt *>>> loads_and_stores;

 public:
  LoadsAndStoresSearcher() {
    allow_undefined_visitor = true;
    invoke_default_visitor = true;
    loads_and_stores = std::make_unique<std::unordered_map<Block *, std::vector<Stmt *>>>();
  }

  void found(Stmt *store_stmt) {
    for (Block *block = store_stmt->parent; block != nullptr;
         block = block->parent) {
      (*loads_and_stores)[block].push_back(store_stmt);
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

  static std::unique_ptr<std::unordered_map<Block *, std::vector<Stmt *>>> run(
      IRNode *root) {
    LoadsAndStoresSearcher searcher;
    root->accept(&searcher);
    return std::move(searcher.loads_and_stores);
  }
};

namespace irpass::analysis {
std::unique_ptr<std::unordered_map<Block *, std::vector<Stmt *>>>
    gather_loads_and_stores_in_blocks(IRNode *root) {
  return LoadsAndStoresSearcher::run(root);
}
}  // namespace irpass::analysis

TLANG_NAMESPACE_END
