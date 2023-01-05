#include "taichi/ir/ir.h"
#include "taichi/ir/statements.h"
#include "taichi/ir/transforms.h"
#include "taichi/ir/visitors.h"

namespace taichi::lang {

class GatherStatementUsages : public BasicStmtVisitor {
 private:
  using BasicStmtVisitor::visit;

  // maps a stmt to all its usages <stmt, operand>
  std::unordered_map<Stmt *, std::vector<std::pair<Stmt *, int>>> stmt_usages_;

 public:
  explicit GatherStatementUsages() {
    invoke_default_visitor = true;
  }

  void default_visit(Stmt *stmt) {
    auto ops = stmt->get_operands();
    for (int i = 0; i < ops.size(); i++) {
      auto &op = ops[i];
      if (op != nullptr) {
        stmt_usages_[op].push_back({stmt, i});
      }
    }
  }

  void visit(Stmt *stmt) override {
    default_visit(stmt);
  }

  void preprocess_container_stmt(Stmt *stmt) override {
    default_visit(stmt);
  }

  static std::unordered_map<Stmt *, std::vector<std::pair<Stmt *, int>>> run(
      IRNode *node) {
    GatherStatementUsages pass;
    node->accept(&pass);
    return pass.stmt_usages_;
  }
};

namespace irpass::analysis {

std::unordered_map<Stmt *, std::vector<std::pair<Stmt *, int>>>
gather_statement_usages(IRNode *root) {
  return GatherStatementUsages::run(root);
}

}  // namespace irpass::analysis

}  // namespace taichi::lang
