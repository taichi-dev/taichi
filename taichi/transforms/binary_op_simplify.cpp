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
    auto const_lhs = stmt->lhs->cast<ConstStmt>();
    if (const_lhs) {
      auto lhs_stmt = stmt->lhs;
      stmt->lhs->replace_with(stmt->rhs);
      stmt->rhs->replace_with(lhs_stmt);
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
    if (!const_lhs_rhs) {
      return;
    }
    if (is_associative(binary_lhs->op_type, stmt->op_type)) {
      auto bin_op =
          Stmt::make<BinaryOpStmt>(stmt->op_type, const_lhs_rhs, const_rhs);
      bin_op->ret_type.data_type = stmt->ret_type.data_type;
      stmt->lhs->replace_with(binary_lhs->lhs);
      stmt->op_type = binary_lhs->op_type;
      stmt->rhs->replace_with(bin_op.get());
      modifier.insert_before(binary_lhs, std::move(bin_op));
      modifier.erase(binary_lhs);
    }
  }

  static bool is_associative(BinaryOpType op1, BinaryOpType op2) {
    if (op1 == BinaryOpType::add &&
        (op2 == BinaryOpType::add || op2 == BinaryOpType::sub)) {
      return true;
    }
    if (op1 == BinaryOpType::mul && op2 == BinaryOpType::mul) {
      return true;
    }
    return false;
  }

  static bool run(IRNode *node, bool fast_math) {
    BinaryOpSimp simplifier(fast_math);
    bool modified = false;
    while (true) {
      node->accept(&simplifier);
      if (simplifier.modifier.modify_ir())
        modified = true;
      else
        break;
    }
    return modified;
  }
};

namespace irpass {

bool binary_op_simplify(IRNode *root) {
  const auto &config = root->get_kernel()->program.config;
  return BinaryOpSimp::run(root, config.fast_math);
}

}  // namespace irpass

TLANG_NAMESPACE_END
