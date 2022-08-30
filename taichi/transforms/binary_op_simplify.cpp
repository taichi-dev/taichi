#include "taichi/ir/analysis.h"
#include "taichi/ir/ir.h"
#include "taichi/ir/statements.h"
#include "taichi/ir/transforms.h"
#include "taichi/ir/visitors.h"

TLANG_NAMESPACE_BEGIN

class BinaryOpSimp : public BasicStmtVisitor {
 public:
  using BasicStmtVisitor::visit;
  bool fast_math;
  DelayedIRModifier modifier;
  bool operand_swapped;

  explicit BinaryOpSimp(bool fast_math_)
      : BasicStmtVisitor(), fast_math(fast_math_), operand_swapped(false) {
  }

  bool try_rearranging_const_rhs(BinaryOpStmt *stmt) {
    // Returns true if the statement is modified.
    auto binary_lhs = stmt->lhs->cast<BinaryOpStmt>();
    auto const_rhs = stmt->rhs->cast<ConstStmt>();
    if (!binary_lhs || !const_rhs) {
      return false;
    }
    auto const_lhs_rhs = binary_lhs->rhs->cast<ConstStmt>();
    if (!const_lhs_rhs || binary_lhs->lhs->is<ConstStmt>()) {
      return false;
    }
    auto op1 = binary_lhs->op_type;
    auto op2 = stmt->op_type;
    // Disables (a / b) * c -> a / (b / c), (a * b) / c -> a * (b / c)
    // when the data type is integral.
    if (is_integral(stmt->ret_type) &&
        ((op1 == BinaryOpType::div && op2 == BinaryOpType::mul) ||
         (op1 == BinaryOpType::mul && op2 == BinaryOpType::div))) {
      return false;
    }
    BinaryOpType new_op2;
    // original:
    // stmt = (a op1 b) op2 c
    // rearrange to:
    // stmt = a op1 (b op2 c)
    if (can_rearrange_associative(op1, op2, new_op2)) {
      auto bin_op = Stmt::make<BinaryOpStmt>(new_op2, const_lhs_rhs, const_rhs);
      bin_op->ret_type = stmt->ret_type;
      auto new_stmt =
          Stmt::make<BinaryOpStmt>(op1, binary_lhs->lhs, bin_op.get());
      new_stmt->ret_type = stmt->ret_type;

      modifier.insert_before(stmt, std::move(bin_op));
      // Replace stmt now to avoid being "simplified" again
      stmt->replace_usages_with(new_stmt.get());
      modifier.insert_before(stmt, std::move(new_stmt));
      modifier.erase(stmt);
      return true;
    }
    // original:
    // stmt = (a >> b) << b
    // rearrange to:
    // stmt = a & (-(1 << b))
    if ((op1 == BinaryOpType::bit_shr || op1 == BinaryOpType::bit_sar) &&
        op2 == BinaryOpType::bit_shl &&
        irpass::analysis::same_value(const_lhs_rhs, const_rhs)) {
      int64 mask = -((int64)1 << (uint64)const_rhs->val.val_as_int64());
      auto mask_stmt =
          Stmt::make<ConstStmt>(TypedConstant(stmt->ret_type, mask));
      auto new_stmt = Stmt::make<BinaryOpStmt>(
          BinaryOpType::bit_and, binary_lhs->lhs, mask_stmt.get());
      new_stmt->ret_type = stmt->ret_type;

      modifier.insert_before(stmt, std::move(mask_stmt));
      // Replace stmt now to avoid being "simplified" again
      stmt->replace_usages_with(new_stmt.get());
      modifier.insert_before(stmt, std::move(new_stmt));
      modifier.erase(stmt);
      return true;
    }
    return false;
  }

  void visit(BinaryOpStmt *stmt) override {
    // Swap lhs and rhs if lhs is a const and op is commutative.
    auto const_lhs = stmt->lhs->cast<ConstStmt>();
    if (const_lhs && is_commutative(stmt->op_type) &&
        !stmt->rhs->is<ConstStmt>()) {
      stmt->lhs = stmt->rhs;
      stmt->rhs = const_lhs;
      operand_swapped = true;
    }
    // Disable other optimizations if fast_math=True and the data type is not
    // integral.
    if (!fast_math && !is_integral(stmt->ret_type)) {
      return;
    }

    if (try_rearranging_const_rhs(stmt)) {
      return;
    }

    // Miscellaneous optimizations.
    // original:
    // stmt = a - (a & b)
    // rearrange to:
    // stmt = a & ~b
    auto *binary_rhs = stmt->rhs->cast<BinaryOpStmt>();
    if (binary_rhs && stmt->op_type == BinaryOpType::sub &&
        binary_rhs->op_type == BinaryOpType::bit_and &&
        irpass::analysis::same_value(stmt->lhs, binary_rhs->lhs)) {
      auto mask_stmt =
          Stmt::make<UnaryOpStmt>(UnaryOpType::bit_not, binary_rhs->rhs);
      mask_stmt->ret_type = binary_rhs->rhs->ret_type;
      auto new_stmt = Stmt::make<BinaryOpStmt>(BinaryOpType::bit_and, stmt->lhs,
                                               mask_stmt.get());
      new_stmt->ret_type = stmt->ret_type;

      modifier.insert_before(stmt, std::move(mask_stmt));
      // Replace stmt now to avoid being "simplified" again
      stmt->replace_usages_with(new_stmt.get());
      modifier.insert_before(stmt, std::move(new_stmt));
      modifier.erase(stmt);
      return;
    }
  }

  static bool can_rearrange_associative(BinaryOpType op1,
                                        BinaryOpType op2,
                                        BinaryOpType &new_op2) {
    if ((op1 == BinaryOpType::add || op1 == BinaryOpType::sub) &&
        (op2 == BinaryOpType::add || op2 == BinaryOpType::sub)) {
      if (op1 == BinaryOpType::add)
        new_op2 = op2;
      else
        new_op2 =
            (op2 == BinaryOpType::add ? BinaryOpType::sub : BinaryOpType::add);
      return true;
    }
    if ((op1 == BinaryOpType::mul || op1 == BinaryOpType::div) &&
        (op2 == BinaryOpType::mul || op2 == BinaryOpType::div)) {
      if (op1 == BinaryOpType::mul)
        new_op2 = op2;
      else
        new_op2 =
            (op2 == BinaryOpType::mul ? BinaryOpType::div : BinaryOpType::mul);
      return true;
    }
    // for bit operations it holds when two ops are the same
    if ((op1 == BinaryOpType::bit_and || op1 == BinaryOpType::bit_or ||
         op1 == BinaryOpType::bit_xor) &&
        op1 == op2) {
      new_op2 = op2;
      return true;
    }
    if ((op1 == BinaryOpType::bit_shl || op1 == BinaryOpType::bit_shr ||
         op1 == BinaryOpType::bit_sar) &&
        op1 == op2) {
      // (a << b) << c -> a << (b + c)
      // (a >> b) >> c -> a >> (b + c)
      new_op2 = BinaryOpType::add;
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
    return modified || simplifier.operand_swapped;
  }
};

namespace irpass {

bool binary_op_simplify(IRNode *root, const CompileConfig &config) {
  TI_AUTO_PROF;
  return BinaryOpSimp::run(root, config.fast_math);
}

}  // namespace irpass

TLANG_NAMESPACE_END
