#include "taichi/ir/ir.h"
#include "taichi/ir/analysis.h"
#include "taichi/ir/visitors.h"

TLANG_NAMESPACE_BEGIN

// Find if there is a store (or AtomicOpStmt).
class LocalStoreSearcher : public BasicStmtVisitor {
 private:
  const std::vector<Stmt *> &vars;
  bool result;

 public:
  using BasicStmtVisitor::visit;

  explicit LocalStoreSearcher(const std::vector<Stmt *> &vars)
      : vars(vars), result(false) {
    for (auto var : vars) {
      TI_ASSERT(var->is<AllocaStmt>());
    }
    allow_undefined_visitor = true;
    invoke_default_visitor = true;
  }

  void visit(LocalStoreStmt *stmt) override {
    for (auto var : vars) {
      if (stmt->ptr == var) {
        result = true;
        break;
      }
    }
  }

  void visit(AtomicOpStmt *stmt) override {
    for (auto var : vars) {
      if (stmt->dest == var) {
        result = true;
        break;
      }
    }
  }

  static bool run(IRNode *root, const std::vector<Stmt *> &vars) {
    LocalStoreSearcher searcher(vars);
    root->accept(&searcher);
    return searcher.result;
  }
};

namespace irpass::analysis {
bool has_store_or_atomic(IRNode *root, const std::vector<Stmt *> &vars) {
  return LocalStoreSearcher::run(root, vars);
}
}  // namespace irpass::analysis

TLANG_NAMESPACE_END
