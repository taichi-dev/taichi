#include "taichi/ir/ir.h"

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
  void visit(Stmt *stmt) override {
    counter++;
  }

  void visit(IfStmt *stmt) override {
    counter++;
    BasicStmtVisitor::visit(stmt);
  }

  void visit(WhileStmt *stmt) override {
    counter++;
    BasicStmtVisitor::visit(stmt);
  }

  void visit(OffloadedStmt *stmt) override {
    counter++;
    BasicStmtVisitor::visit(stmt);
  }

  void visit(RangeForStmt *stmt) override {
    counter++;
    BasicStmtVisitor::visit(stmt);
  }

  void visit(StructForStmt *stmt) override {
    counter++;
    BasicStmtVisitor::visit(stmt);
  }

  static int run(IRNode *root) {
    StmtCounter stmt_counter;
    root->accept(&stmt_counter);
    return stmt_counter.counter;
  }

 private:
  int counter;
};

namespace irpass {
int count_statements(IRNode *root) {
  TI_ASSERT(root);
  return StmtCounter::run(root);
}
}  // namespace irpass

TLANG_NAMESPACE_END
