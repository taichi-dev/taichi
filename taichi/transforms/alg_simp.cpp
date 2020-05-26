#include "taichi/ir/ir.h"
#include "taichi/ir/transforms.h"
#include "taichi/ir/visitors.h"
#include "taichi/program/program.h"

TLANG_NAMESPACE_BEGIN

// Algebraic Simplification
class AlgSimp : public BasicStmtVisitor {
 public:
  using BasicStmtVisitor::visit;
  bool fast_math;
  std::vector<Stmt *> to_erase;
  std::vector<std::pair<std::unique_ptr<Stmt>, Stmt *>> to_insert_before;

  explicit AlgSimp(bool fast_math_)
      : BasicStmtVisitor(), fast_math(fast_math_) {
  }

  void visit(BinaryOpStmt *stmt) override {
    auto lhs = stmt->lhs->cast<ConstStmt>();
    auto rhs = stmt->rhs->cast<ConstStmt>();
    if (!lhs && !rhs)
      return;
    if (stmt->width() != 1) {
      return;
    }
    if (stmt->op_type == BinaryOpType::add ||
        stmt->op_type == BinaryOpType::sub ||
        stmt->op_type == BinaryOpType::bit_or ||
        stmt->op_type == BinaryOpType::bit_xor) {
      if (alg_is_zero(rhs)) {
        // a +-|^ 0 -> a
        stmt->replace_with(stmt->lhs);
        to_erase.push_back(stmt);
      } else if (stmt->op_type != BinaryOpType::sub && alg_is_zero(lhs)) {
        // 0 +|^ a -> a
        stmt->replace_with(stmt->rhs);
        to_erase.push_back(stmt);
      }
    } else if (stmt->op_type == BinaryOpType::mul ||
               stmt->op_type == BinaryOpType::div) {
      if (alg_is_one(rhs)) {
        // a */ 1 -> a
        stmt->replace_with(stmt->lhs);
        to_erase.push_back(stmt);
      } else if (stmt->op_type == BinaryOpType::mul && alg_is_one(lhs)) {
        // 1 * a -> a
        stmt->replace_with(stmt->rhs);
        to_erase.push_back(stmt);
      } else if ((fast_math || is_integral(stmt->ret_type.data_type)) &&
                 stmt->op_type == BinaryOpType::mul &&
                 (alg_is_zero(lhs) || alg_is_zero(rhs))) {
        // fast_math or integral operands: 0 * a -> 0, a * 0 -> 0
        if (alg_is_zero(lhs) &&
            lhs->ret_type.data_type == stmt->ret_type.data_type) {
          stmt->replace_with(stmt->lhs);
          to_erase.push_back(stmt);
        } else if (alg_is_zero(rhs) &&
                   rhs->ret_type.data_type == stmt->ret_type.data_type) {
          stmt->replace_with(stmt->rhs);
          to_erase.push_back(stmt);
        } else {
          auto zero = Stmt::make<ConstStmt>(
              LaneAttribute<TypedConstant>(stmt->ret_type.data_type));
          stmt->replace_with(zero.get());
          to_insert_before.emplace_back(std::move(zero), stmt);
          to_erase.push_back(stmt);
        }
      }
    } else if (stmt->op_type == BinaryOpType::bit_and) {
      if (alg_is_minus_one(rhs)) {
        // a & -1 -> a
        stmt->replace_with(stmt->lhs);
        to_erase.push_back(stmt);
      } else if (alg_is_minus_one(lhs)) {
        // -1 & a -> a
        stmt->replace_with(stmt->rhs);
        to_erase.push_back(stmt);
      }
    }
  }

  void visit(AssertStmt *stmt) override {
    auto cond = stmt->cond->cast<ConstStmt>();
    if (!cond)
      return;
    if (!alg_is_zero(cond)) {
      // this statement has no effect
      to_erase.push_back(stmt);
    }
  }

  void visit(WhileControlStmt *stmt) override {
    auto cond = stmt->cond->cast<ConstStmt>();
    if (!cond)
      return;
    if (!alg_is_zero(cond)) {
      // this statement has no effect
      to_erase.push_back(stmt);
    }
  }

  static bool alg_is_zero(ConstStmt *stmt) {
    if (!stmt)
      return false;
    if (stmt->width() != 1)
      return false;
    auto val = stmt->val[0];
    auto data_type = stmt->ret_type.data_type;
    if (is_real(data_type))
      return val.val_float() == 0;
    else if (is_signed(data_type))
      return val.val_int() == 0;
    else if (is_unsigned(data_type))
      return val.val_uint() == 0;
    else {
      TI_NOT_IMPLEMENTED
    }
  }

  static bool alg_is_one(ConstStmt *stmt) {
    if (!stmt)
      return false;
    if (stmt->width() != 1)
      return false;
    auto val = stmt->val[0];
    auto data_type = stmt->ret_type.data_type;
    if (is_real(data_type))
      return val.val_float() == 1;
    else if (is_signed(data_type))
      return val.val_int() == 1;
    else if (is_unsigned(data_type))
      return val.val_uint() == 1;
    else {
      TI_NOT_IMPLEMENTED
    }
  }

  static bool alg_is_minus_one(ConstStmt *stmt) {
    if (!stmt)
      return false;
    if (stmt->width() != 1)
      return false;
    auto val = stmt->val[0];
    auto data_type = stmt->ret_type.data_type;
    if (is_real(data_type))
      return val.val_float() == -1;
    else if (is_signed(data_type))
      return val.val_int() == -1;
    else if (is_unsigned(data_type))
      return false;
    else {
      TI_NOT_IMPLEMENTED
    }
  }

  static bool run(IRNode *node, bool fast_math) {
    AlgSimp simplifier(fast_math);
    bool modified = false;
    while (true) {
      node->accept(&simplifier);
      if (simplifier.to_erase.empty() && simplifier.to_insert_before.empty())
        break;
      modified = true;
      for (auto &i : simplifier.to_insert_before) {
        i.second->insert_before_me(std::move(i.first));
      }
      for (auto &stmt : simplifier.to_erase) {
        stmt->parent->erase(stmt);
      }
      simplifier.to_insert_before.clear();
      simplifier.to_erase.clear();
    }
    return modified;
  }
};

namespace irpass {

bool alg_simp(IRNode *root, const CompileConfig &config) {
  return AlgSimp::run(root, config.fast_math);
}

}  // namespace irpass

TLANG_NAMESPACE_END
