#include <set>
#include "taichi/ir/ir.h"

TLANG_NAMESPACE_BEGIN

class CheckOutOfBound : public BasicStmtVisitor {
 public:
  using BasicStmtVisitor::visit;
  std::set<int> visited;

  CheckOutOfBound() : BasicStmtVisitor(), visited() {
  }

  bool is_done(Stmt *stmt) {
    return visited.find(stmt->instance_id) != visited.end();
  }

  void set_done(Stmt *stmt) {
    visited.insert(stmt->instance_id);
  }

  void visit(GlobalPtrStmt *stmt) override {
    if (is_done(stmt))
      return;
    TI_ASSERT(stmt->snodes.size() == 1);
    auto snode = stmt->snodes[0];
    auto new_stmts = VecStatement();
    auto zero = new_stmts.push_back<ConstStmt>(LaneAttribute<TypedConstant>(0));
    Stmt *result = new_stmts.push_back<ConstStmt>(LaneAttribute<TypedConstant>(true));
    for (int i = 0; i < stmt->indices.size(); i++) {
      auto check_zero = new_stmts.push_back<BinaryOpStmt>(
          BinaryOpType::cmp_ge, stmt->indices[i], zero);
      auto bound = new_stmts.push_back<ConstStmt>(LaneAttribute<TypedConstant>(
          snode->extractors[snode->physical_index_position[i]].num_elements));
      auto check_bound = new_stmts.push_back<BinaryOpStmt>(
          BinaryOpType::cmp_lt, stmt->indices[i], bound);
      auto check_i = new_stmts.push_back<BinaryOpStmt>(
          BinaryOpType::bit_and, check_zero, check_bound);
      result = new_stmts.push_back<BinaryOpStmt>(
          BinaryOpType::bit_and, result, check_i);
    }
    new_stmts.push_back<AssertStmt>(result, "Accessing Tensor of Size [...] with indices (...)", std::vector<Stmt *>());
    stmt->parent->insert_before(stmt, std::move(new_stmts));
    set_done(stmt);
    throw IRModified();
  }

  static void run(IRNode *node) {
    CheckOutOfBound checker;
    while (true) {
      bool modified = false;
      try {
        node->accept(&checker);
      } catch (IRModified) {
        modified = true;
      }
      if (!modified)
        break;
    }
  }
};

namespace irpass {

void check_out_of_bound(IRNode *root) {
  return CheckOutOfBound::run(root);
}

}  // namespace irpass

TLANG_NAMESPACE_END
