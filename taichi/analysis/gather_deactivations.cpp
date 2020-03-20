#include "taichi/ir/ir.h"
#include <set>

TLANG_NAMESPACE_BEGIN

class GatherDeactivations : public BasicStmtVisitor {
 public:
  using BasicStmtVisitor::visit;

  std::set<SNode *> snodes;
  IRNode *root;

  GatherDeactivations(IRNode *root) : root(root) {
  }

  void visit(SNodeOpStmt *stmt) override {
    if (stmt->op_type == SNodeOpType::deactivate) {
      if (snodes.find(stmt->snode) == snodes.end()) {
        snodes.insert(stmt->snode);
      }
    }
  }

  std::vector<SNode *> run() {
    root->accept(this);
    return std::vector<SNode *>(snodes.begin(), snodes.end());
  }
};

namespace irpass {
std::vector<SNode *> gather_deactivations(IRNode *root) {
  GatherDeactivations gather(root);
  return gather.run();
}
}  // namespace irpass

TLANG_NAMESPACE_END
