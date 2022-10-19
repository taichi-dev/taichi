#include "taichi/ir/ir.h"
#include "taichi/ir/analysis.h"
#include "taichi/ir/statements.h"
#include "taichi/ir/visitors.h"

namespace taichi::lang {

class GatherDeactivations : public BasicStmtVisitor {
 public:
  using BasicStmtVisitor::visit;

  std::unordered_set<SNode *> snodes;
  IRNode *root;

  explicit GatherDeactivations(IRNode *root) : root(root) {
  }

  void visit(SNodeOpStmt *stmt) override {
    if (stmt->op_type == SNodeOpType::deactivate) {
      if (snodes.find(stmt->snode) == snodes.end()) {
        snodes.insert(stmt->snode);
      }
    }
  }

  std::unordered_set<SNode *> run() {
    root->accept(this);
    return snodes;
  }
};

namespace irpass::analysis {
std::unordered_set<SNode *> gather_deactivations(IRNode *root) {
  GatherDeactivations gather(root);
  return gather.run();
}
}  // namespace irpass::analysis

}  // namespace taichi::lang
