#include "taichi/ir/ir.h"

TLANG_NAMESPACE_BEGIN

// Find if there is a load (or AtomicOpStmt).
class LocalLoadSearcher : public BasicStmtVisitor {
 private:
  Stmt *var;
  bool result;

 public:
  using BasicStmtVisitor::visit;

  explicit LocalLoadSearcher(Stmt *var) : var(var), result(false) {
    TI_ASSERT(var->is<AllocaStmt>());
    allow_undefined_visitor = true;
    invoke_default_visitor = true;
  }

  void visit(LocalLoadStmt *stmt) override {
    if (stmt->has_source(var)) {
      result = true;
    }
  }

  void visit(AtomicOpStmt *stmt) override {
    if (stmt->dest == var) {
      result = true;
    }
  }

  static bool run(IRNode *root, Stmt *var) {
    LocalLoadSearcher searcher(var);
    root->accept(&searcher);
    return searcher.result;
  }
};

namespace irpass::analysis {
bool has_load_or_atomic(IRNode *root, Stmt *var) {
  return LocalLoadSearcher::run(root, var);
}
}  // namespace irpass::analysis

TLANG_NAMESPACE_END
