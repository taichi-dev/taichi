#include "taichi/ir/ir.h"
#include "taichi/ir/transforms.h"
#include "taichi/ir/visitors.h"
#include "taichi/program/program.h"

TLANG_NAMESPACE_BEGIN

// Demote Operations into pieces for backends to deal easier
class DemoteOperations : public BasicStmtVisitor {
 public:
  using BasicStmtVisitor::visit;
  DelayedIRModifier modifier;

  DemoteOperations() : BasicStmtVisitor() {
  }

  void visit(BinaryOpStmt *stmt) override {
    auto lhs = stmt->lhs;
    auto rhs = stmt->rhs;
    if (stmt->op_type == BinaryOpType::floordiv) {
      if (is_integral(rhs->element_type()) && is_integral(lhs->element_type())) {
        // @ti.func
        // def ifloordiv(a, b):
        //     return (a if (a < 0) == (b < 0) else (a - b + 1)) / b
        auto zero = Stmt::make<ConstStmt>(LaneAttribute<TypedConstant>(0));
        auto one = Stmt::make<ConstStmt>(LaneAttribute<TypedConstant>(1));
        auto lhs_ltz = Stmt::make<BinaryOpStmt>(
            BinaryOpType::cmp_lt, lhs, zero.get());
        auto rhs_ltz = Stmt::make<BinaryOpStmt>(
            BinaryOpType::cmp_lt, rhs, zero.get());
        auto lhs_ltz_eq_rhs_ltz = Stmt::make<BinaryOpStmt>(
            BinaryOpType::cmp_eq, lhs_ltz.get(), rhs_ltz.get());
        auto lhs_sub_rhs = Stmt::make<BinaryOpStmt>(
            BinaryOpType::sub, lhs, rhs);
        auto lhs_sub_rhs_add_one = Stmt::make<BinaryOpStmt>(
            BinaryOpType::add, lhs_sub_rhs.get(), one.get());
        auto ternary = Stmt::make<TernaryOpStmt>(
            TernaryOpType::select, lhs_ltz_eq_rhs_ltz.get(),
            lhs, lhs_sub_rhs_add_one.get());
        auto ret = Stmt::make<BinaryOpStmt>(
            BinaryOpType::div, ternary.get(), rhs);

        modifier.insert_before(stmt, std::move(zero));
        modifier.insert_before(stmt, std::move(one));
        modifier.insert_before(stmt, std::move(lhs_ltz));
        modifier.insert_before(stmt, std::move(rhs_ltz));
        modifier.insert_before(stmt, std::move(lhs_ltz_eq_rhs_ltz));
        modifier.insert_before(stmt, std::move(lhs_sub_rhs));
        modifier.insert_before(stmt, std::move(lhs_sub_rhs_add_one));
        modifier.insert_before(stmt, std::move(ternary));
        stmt->replace_with(std::move(ret));
      } else {
        // @ti.func
        // def ffloordiv(a, b):
        //     r = ti.raw_div(a, b)
        //     return ti.floor(r)
        auto div = Stmt::make<BinaryOpStmt>(
            BinaryOpType::div, lhs, rhs);
        auto floor = Stmt::make<UnaryOpStmt>(
            UnaryOpType::floor, div.get());
        modifier.insert_before(stmt, std::move(div));
        stmt->replace_with(std::move(floor));
      }
    }
  }

  static bool run(IRNode *node) {
    DemoteOperations demoter;
    bool modified = false;
    while (true) {
      node->accept(&demoter);
      if (demoter.modifier.modify_ir())
        modified = true;
      else
        break;
    }
    return modified;
  }
};

namespace irpass {

bool demote_operations(IRNode *root) {
  TI_AUTO_PROF;
  return DemoteOperations::run(root);
}

}  // namespace irpass

TLANG_NAMESPACE_END
