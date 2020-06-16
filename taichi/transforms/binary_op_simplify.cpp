#include "taichi/ir/ir.h"
#include "taichi/ir/transforms.h"
#include "taichi/ir/visitors.h"
#include "taichi/program/program.h"

TLANG_NAMESPACE_BEGIN

class BinaryOpSimp : public BasicStmtVisitor {
 public:
  using BasicStmtVisitor::visit;
  bool fast_math;
  DelayedIRModifier modifier;

  explicit BinaryOpSimp(bool fast_math_)
      : BasicStmtVisitor(), fast_math(fast_math_) {
  }

  void visit(BinaryOpStmt *stmt) override {
    // swap lhs and rhs if lhs is a const and op is commutative
    auto const_lhs = stmt->lhs->cast<ConstStmt>();
    if (const_lhs && is_commutative(stmt->op_type) &&
        !stmt->rhs->is<ConstStmt>()) {
      auto rhs_stmt = stmt->rhs;
      stmt->lhs = rhs_stmt;
      stmt->rhs = const_lhs;
    }
    if (!fast_math) {
      return;
    }
    auto binary_lhs = stmt->lhs->cast<BinaryOpStmt>();
    auto const_rhs = stmt->rhs->cast<ConstStmt>();
    if (!binary_lhs || !const_rhs) {
      return;
    }
    auto const_lhs_rhs = binary_lhs->rhs->cast<ConstStmt>();
    if (!const_lhs_rhs || binary_lhs->lhs->is<ConstStmt>()) {
      return;
    }
    // original:
    // stmt = (a op1 b) op2 c
    // rearrange to:
    // stmt = a op1 (b op2 c)
    if (can_rearrange_associative(binary_lhs->op_type, stmt->op_type)) {
      auto bin_op =
          Stmt::make<BinaryOpStmt>(stmt->op_type, const_lhs_rhs, const_rhs);
      bin_op->ret_type.data_type = stmt->ret_type.data_type;
      auto new_stmt = Stmt::make<BinaryOpStmt>(binary_lhs->op_type,
                                               binary_lhs->lhs, bin_op.get());
      new_stmt->ret_type.data_type = stmt->ret_type.data_type;

      modifier.insert_before(stmt, std::move(bin_op));
      stmt->replace_with(new_stmt.get());
      modifier.insert_before(stmt, std::move(new_stmt));
      modifier.erase(stmt);
    }
  }

  static bool can_rearrange_associative(BinaryOpType op1, BinaryOpType op2) {
    if (op1 == BinaryOpType::add &&
        (op2 == BinaryOpType::add || op2 == BinaryOpType::sub)) {
      return true;
    }
    if (op1 == BinaryOpType::mul &&
        (op2 == BinaryOpType::mul || op2 == BinaryOpType::div)) {
      return true;
    }
    // for bit operations it only holds when two ops are the same
    if ((op1 == BinaryOpType::bit_and || op1 == BinaryOpType::bit_or ||
         op1 == BinaryOpType::bit_xor) &&
        op1 == op2) {
      return true;
    }
    return false;
  }

  static bool is_commutative(BinaryOpType op) {
    return op == BinaryOpType::add || op == BinaryOpType::mul ||
           op == BinaryOpType::bit_and || op == BinaryOpType::bit_or ||
           op == BinaryOpType::bit_xor;
  }

  static bool run(IRNode *node, bool fast_math) {
    BinaryOpSimp simplifier(fast_math);
    bool modified = false;
    while (true) {
      node->accept(&simplifier);
      if (simplifier.modifier.modify_ir()) {
        modified = true;
      } else
        break;
    }
    return modified;
  }
};

namespace irpass {

bool binary_op_simplify(IRNode *root) {
  TI_AUTO_PROF;
  return BinaryOpSimp::run(root, hack::use_fast_math(root));
}

}  // namespace irpass

TLANG_NAMESPACE_END
