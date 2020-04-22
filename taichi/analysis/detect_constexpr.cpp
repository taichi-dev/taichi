#include "taichi/ir/ir.h"

#include <unordered_set>

TLANG_NAMESPACE_BEGIN

class DetectConstexpr : public BasicStmtVisitor {
 public:
  using BasicStmtVisitor::visit;

  bool has_non_const{false};
  IRNode *root;

  DetectConstexpr(IRNode *root) : root(root) {
    invoke_default_visitor = true;
  }

  void visit(Stmt *stmt) override {
    has_non_const = true;
  }

  void visit(UnaryOpStmt *stmt) override {
    stmt->operand->accept(this);
  }

  void visit(BinaryOpStmt *stmt) override {
    stmt->lhs->accept(this);
    stmt->rhs->accept(this);
  }

  void visit(ConstStmt *stmt) override {
  }

  bool run() {
    root->accept(this);
    return !has_non_const;
  }
};

namespace irpass::analysis {
bool detect_constexpr(IRNode *root) {
  DetectConstexpr detective(root);
  return detective.run();
}
}  // namespace irpass::analysis

TLANG_NAMESPACE_END
