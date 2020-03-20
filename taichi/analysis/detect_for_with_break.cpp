#include "taichi/ir/ir.h"
#include <set>

TLANG_NAMESPACE_BEGIN

class DetectForWithBreak : public BasicStmtVisitor {
 public:
  using BasicStmtVisitor::visit;

  std::vector<Stmt *> loop_stack;
  std::set<Stmt *> fors_with_break;
  IRNode *root;

  DetectForWithBreak(IRNode *root) : root(root) {
  }

  void visit(FrontendWhileStmt *stmt) override {
    TI_INFO("FOUND WHILE");
    loop_stack.push_back(stmt);
    stmt->body->accept(this);
    loop_stack.pop_back();
  }

  void visit(FrontendForStmt *stmt) override {
    TI_INFO("FOUND FOR");
    loop_stack.push_back(stmt);
    stmt->body->accept(this);
    loop_stack.pop_back();
  }

  void visit(FrontendIfStmt *stmt) override {
    TI_INFO("FOUND IF");
    if (stmt->true_statements)
      stmt->true_statements->accept(this);
    if (stmt->false_statements)
      stmt->false_statements->accept(this);
  }

  void visit(FrontendBreakStmt *stmt) override {
    TI_INFO("FOUND BREAK!");
    TI_ASSERT_INFO(loop_stack.size() != 0, "break statement out of loop scope");
    auto loop = loop_stack.back();
    if (loop->is<FrontendForStmt>())
      fors_with_break.insert(loop);
  }

  std::vector<Stmt *> run() {
    root->accept(this);
    return std::vector<Stmt *>(fors_with_break.begin(), fors_with_break.end());
  }
};

namespace irpass {
std::vector<Stmt *> detect_fors_with_break(IRNode *root) {
  DetectForWithBreak detective(root);
  return detective.run();
}
}  // namespace irpass

TLANG_NAMESPACE_END
