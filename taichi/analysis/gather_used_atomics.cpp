#include "taichi/ir/ir.h"
#include "taichi/ir/analysis.h"
#include "taichi/ir/visitors.h"
#include <unordered_set>

TLANG_NAMESPACE_BEGIN

class UsedAtomicsSearcher : public BasicStmtVisitor {
 private:
  std::unique_ptr<std::unordered_set<AtomicOpStmt *>> used_atomics;

 public:
  using BasicStmtVisitor::visit;

  UsedAtomicsSearcher() {
    allow_undefined_visitor = true;
    invoke_default_visitor = true;
    used_atomics = std::make_unique<std::unordered_set<AtomicOpStmt *>>();
  }

  void search_operands(Stmt *stmt) {
    for (auto &op : stmt->get_operands()) {
      if (op != nullptr && op->is<AtomicOpStmt>()) {
        used_atomics->insert(op->as<AtomicOpStmt>());
      }
    }
  }

  void preprocess_container_stmt(Stmt *stmt) override {
    search_operands(stmt);
  }

  void visit(Stmt *stmt) override {
    search_operands(stmt);
  }

  static std::unique_ptr<std::unordered_set<AtomicOpStmt *>> run(IRNode *root) {
    UsedAtomicsSearcher searcher;
    root->accept(&searcher);
    return std::move(searcher.used_atomics);
  }
};

namespace irpass::analysis {
std::unique_ptr<std::unordered_set<AtomicOpStmt *>> gather_used_atomics(
    IRNode *root) {
  return UsedAtomicsSearcher::run(root);
}
}  // namespace irpass::analysis

TLANG_NAMESPACE_END
