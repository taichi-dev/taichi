#include "taichi/ir/ir.h"
#include "taichi/ir/transforms.h"
#include "taichi/ir/visitors.h"
#include "taichi/program/program.h"

TLANG_NAMESPACE_BEGIN

// Algebraic Simplification and Strength Reduction
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
      } else if (stmt->op_type == BinaryOpType::mul &&
                 (alg_is_two(lhs) || alg_is_two(rhs))) {
        // 2 * a -> a + a, a * 2 -> a + a
        auto a = stmt->lhs;
        if (alg_is_two(lhs))
          a = stmt->rhs;
        if (stmt->ret_type.data_type != a->ret_type.data_type) {
          auto cast = Stmt::make_typed<UnaryOpStmt>(UnaryOpType::cast_value, a);
          cast->cast_type = stmt->ret_type.data_type;
          cast->ret_type.data_type = stmt->ret_type.data_type;
          a = cast.get();
          to_insert_before.emplace_back(std::move(cast), stmt);
        }
        auto sum = Stmt::make<BinaryOpStmt>(BinaryOpType::add, a, a);
        sum->ret_type.data_type = a->ret_type.data_type;
        stmt->replace_with(sum.get());
        to_insert_before.emplace_back(std::move(sum), stmt);
        to_erase.push_back(stmt);
      } else if (fast_math && stmt->op_type == BinaryOpType::div && rhs &&
                 is_real(rhs->ret_type.data_type)) {
        if (alg_is_zero(rhs)) {
          TI_WARN("Potential division by 0");
        } else {
          // a / const -> a * (1 / const)
          auto reciprocal = Stmt::make_typed<ConstStmt>(
              LaneAttribute<TypedConstant>(rhs->ret_type.data_type));
          if (rhs->ret_type.data_type == DataType::f64) {
            reciprocal->val[0].val_float64() =
                (float64)1.0 / rhs->val[0].val_float64();
          } else if (rhs->ret_type.data_type == DataType::f32) {
            reciprocal->val[0].val_float32() =
                (float64)1.0 / rhs->val[0].val_float32();
          } else {
            TI_NOT_IMPLEMENTED
          }
          auto product = Stmt::make<BinaryOpStmt>(BinaryOpType::mul, stmt->lhs,
                                                  reciprocal.get());
          product->ret_type.data_type = stmt->ret_type.data_type;
          stmt->replace_with(product.get());
          to_insert_before.emplace_back(std::move(reciprocal), stmt);
          to_insert_before.emplace_back(std::move(product), stmt);
          to_erase.push_back(stmt);
        }
      }
    } else if (stmt->op_type == BinaryOpType::pow) {
      if (alg_is_one(rhs)) {
        // a ** 1 -> a
        stmt->replace_with(stmt->lhs);
        to_erase.push_back(stmt);
      } else if (alg_is_two(rhs)) {
        // a ** 2 -> a * a
        auto a = stmt->lhs;
        if (stmt->ret_type.data_type != a->ret_type.data_type) {
          auto cast = Stmt::make_typed<UnaryOpStmt>(UnaryOpType::cast_value, a);
          cast->cast_type = stmt->ret_type.data_type;
          cast->ret_type.data_type = stmt->ret_type.data_type;
          a = cast.get();
          to_insert_before.emplace_back(std::move(cast), stmt);
        }
        auto product = Stmt::make<BinaryOpStmt>(BinaryOpType::mul, a, a);
        product->ret_type.data_type = a->ret_type.data_type;
        stmt->replace_with(product.get());
        to_insert_before.emplace_back(std::move(product), stmt);
        to_erase.push_back(stmt);
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

  static bool alg_is_two(ConstStmt *stmt) {
    if (!stmt)
      return false;
    if (stmt->width() != 1)
      return false;
    auto val = stmt->val[0];
    auto data_type = stmt->ret_type.data_type;
    if (is_real(data_type))
      return val.val_float() == 2;
    else if (is_signed(data_type))
      return val.val_int() == 2;
    else if (is_unsigned(data_type))
      return val.val_uint() == 2;
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
      return val.val_uint() == (uint64)-1;
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

bool alg_simp(IRNode *root) {
  const auto &config = root->get_kernel()->program.config;
  return AlgSimp::run(root, config.fast_math);
}

}  // namespace irpass

TLANG_NAMESPACE_END
