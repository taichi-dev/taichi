#include "taichi/ir/ir.h"
#include "taichi/ir/analysis.h"
#include "taichi/ir/visitors.h"

TLANG_NAMESPACE_BEGIN

// Count all statements (including containers)
class StmtCounter : public BasicStmtVisitor {
 private:
  StmtCounter() {
    counter = 0;
    allow_undefined_visitor = true;
    invoke_default_visitor = true;
  }

  using BasicStmtVisitor::visit;

 public:
  void preprocess_container_stmt(Stmt *stmt) override {
    counter++;
  }

  void visit(Stmt *stmt) override {
    counter++;
  }

  static int run(IRNode *root) {
    StmtCounter stmt_counter;
    root->accept(&stmt_counter);
    return stmt_counter.counter;
  }

 private:
  int counter;
};

namespace irpass::analysis {
int count_statements(IRNode *root) {
  TI_ASSERT(root);
  return StmtCounter::run(root);
}
}  // namespace irpass::analysis

TLANG_NAMESPACE_END
