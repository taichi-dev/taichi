#include "taichi/ir/ir.h"
#include "taichi/ir/analysis.h"
#include "taichi/ir/visitors.h"
#include "taichi/ir/frontend_ir.h"

#include <unordered_set>

TLANG_NAMESPACE_BEGIN

class DetectForsWithBreak : public BasicStmtVisitor {
 public:
  using BasicStmtVisitor::visit;

  std::vector<Stmt *> loop_stack;
  std::unordered_set<Stmt *> fors_with_break;
  IRNode *root;

  DetectForsWithBreak(IRNode *root) : root(root) {
  }

  void visit(FrontendBreakStmt *stmt) override {
    TI_ASSERT_INFO(loop_stack.size() != 0, "break statement out of loop scope");
    auto loop = loop_stack.back();
    if (loop->is<FrontendForStmt>())
      fors_with_break.insert(loop);
  }

  void visit(FrontendWhileStmt *stmt) override {
    loop_stack.push_back(stmt);
    stmt->body->accept(this);
    loop_stack.pop_back();
  }

  void visit(FrontendForStmt *stmt) override {
    loop_stack.push_back(stmt);
    stmt->body->accept(this);
    loop_stack.pop_back();
  }

  std::unordered_set<Stmt *> run() {
    root->accept(this);
    return fors_with_break;
  }
};

namespace irpass::analysis {
std::unordered_set<Stmt *> detect_fors_with_break(IRNode *root) {
  DetectForsWithBreak detective(root);
  return detective.run();
}
}  // namespace irpass::analysis

TLANG_NAMESPACE_END
