#include "taichi/ir/ir.h"

#include <unordered_set>

TLANG_NAMESPACE_BEGIN

class DetectForWithBreak : public BasicStmtVisitor {
 public:
  using BasicStmtVisitor::visit;

  std::vector<Stmt *> loop_stack;
  std::unordered_set<Stmt *> fors_with_break;
  IRNode *root;

  DetectForWithBreak(IRNode *root) : root(root) {
  }

  void visit(FrontendBreakStmt *stmt) override {
    TI_ASSERT_INFO(loop_stack.size() != 0, "break statement out of loop scope");
    auto loop = loop_stack.back();
    if (loop->is<FrontendForStmt>())
      fors_with_break.insert(loop);
  }

  std::unordered_set<Stmt *> run() {
    root->accept(this);
    return fors_with_break;
  }
};

namespace irpass {
std::unordered_set<Stmt *> detect_fors_with_break(IRNode *root) {
  DetectForWithBreak detective(root);
  return detective.run();
}
}  // namespace irpass

TLANG_NAMESPACE_END
