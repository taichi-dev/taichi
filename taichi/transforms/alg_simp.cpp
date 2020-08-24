#include "taichi/ir/ir.h"
#include "taichi/ir/transforms.h"
#include "taichi/ir/visitors.h"
#include "taichi/program/program.h"

TLANG_NAMESPACE_BEGIN

// Algebraic Simplification and Strength Reduction
class AlgSimp : public BasicStmtVisitor {
 private:
  void cast_to_result_type(Stmt *&a, Stmt *stmt) {
    if (stmt->ret_type.data_type != a->ret_type.data_type) {
      auto cast = Stmt::make_typed<UnaryOpStmt>(UnaryOpType::cast_value, a);
      cast->cast_type = stmt->ret_type.data_type;
      cast->ret_type.data_type = stmt->ret_type.data_type;
      a = cast.get();
      modifier.insert_before(stmt, std::move(cast));
    }
  }

 public:
  static constexpr int max_weaken_exponent = 32;
  using BasicStmtVisitor::visit;
  bool fast_math;
  DelayedIRModifier modifier;

  explicit AlgSimp(bool fast_math_)
      : BasicStmtVisitor(), fast_math(fast_math_) {
  }

  void visit(UnaryOpStmt *stmt) override {
    if (stmt->is_cast() &&
        stmt->cast_type == stmt->operand->ret_type.data_type) {
      stmt->replace_with(stmt->operand);
      modifier.erase(stmt);
    }
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
        modifier.erase(stmt);
      } else if (stmt->op_type != BinaryOpType::sub && alg_is_zero(lhs)) {
        // 0 +|^ a -> a
        stmt->replace_with(stmt->rhs);
        modifier.erase(stmt);
      }
    } else if (stmt->op_type == BinaryOpType::mul ||
               stmt->op_type == BinaryOpType::div) {
      if (alg_is_one(rhs)) {
        // a */ 1 -> a
        stmt->replace_with(stmt->lhs);
        modifier.erase(stmt);
      } else if (stmt->op_type == BinaryOpType::mul && alg_is_one(lhs)) {
        // 1 * a -> a
        stmt->replace_with(stmt->rhs);
        modifier.erase(stmt);
      } else if ((fast_math || is_integral(stmt->ret_type.data_type)) &&
                 stmt->op_type == BinaryOpType::mul &&
                 (alg_is_zero(lhs) || alg_is_zero(rhs))) {
        // fast_math or integral operands: 0 * a -> 0, a * 0 -> 0
        if (alg_is_zero(lhs) &&
            lhs->ret_type.data_type == stmt->ret_type.data_type) {
          stmt->replace_with(stmt->lhs);
          modifier.erase(stmt);
        } else if (alg_is_zero(rhs) &&
                   rhs->ret_type.data_type == stmt->ret_type.data_type) {
          stmt->replace_with(stmt->rhs);
          modifier.erase(stmt);
        } else {
          auto zero = Stmt::make<ConstStmt>(
              LaneAttribute<TypedConstant>(stmt->ret_type.data_type));
          stmt->replace_with(zero.get());
          modifier.insert_before(stmt, std::move(zero));
          modifier.erase(stmt);
        }
      } else if (stmt->op_type == BinaryOpType::mul &&
                 (alg_is_two(lhs) || alg_is_two(rhs))) {
        // 2 * a -> a + a, a * 2 -> a + a
        auto a = stmt->lhs;
        if (alg_is_two(lhs))
          a = stmt->rhs;
        cast_to_result_type(a, stmt);
        auto sum = Stmt::make<BinaryOpStmt>(BinaryOpType::add, a, a);
        sum->ret_type.data_type = a->ret_type.data_type;
        stmt->replace_with(sum.get());
        modifier.insert_before(stmt, std::move(sum));
        modifier.erase(stmt);
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
                (float32)1.0 / rhs->val[0].val_float32();
          } else {
            TI_NOT_IMPLEMENTED
          }
          auto product = Stmt::make<BinaryOpStmt>(BinaryOpType::mul, stmt->lhs,
                                                  reciprocal.get());
          product->ret_type.data_type = stmt->ret_type.data_type;
          stmt->replace_with(product.get());
          modifier.insert_before(stmt, std::move(reciprocal));
          modifier.insert_before(stmt, std::move(product));
          modifier.erase(stmt);
        }
      }
    } else if (rhs && stmt->op_type == BinaryOpType::pow) {
      float64 exponent = rhs->val[0].val_cast_to_float64();
      if (exponent == 1) {
        // a ** 1 -> a
        stmt->replace_with(stmt->lhs);
        modifier.erase(stmt);
      } else if (exponent == 0) {
        // a ** 0 -> 1
        auto one = Stmt::make<ConstStmt>(LaneAttribute<TypedConstant>(1));
        auto one_raw = one.get();
        modifier.insert_before(stmt, std::move(one));
        cast_to_result_type(one_raw, stmt);
        stmt->replace_with(one_raw);
        modifier.erase(stmt);
      } else if (exponent == 0.5) {
        // a ** 0.5 -> sqrt(a)
        auto a = stmt->lhs;
        cast_to_result_type(a, stmt);
        auto result = Stmt::make<UnaryOpStmt>(UnaryOpType::sqrt, a);
        result->ret_type.data_type = a->ret_type.data_type;
        stmt->replace_with(result.get());
        modifier.insert_before(stmt, std::move(result));
        modifier.erase(stmt);
      } else if (exponent == std::round(exponent) && exponent > 0 &&
                 exponent <= max_weaken_exponent) {
        // a ** n -> Exponentiation by squaring
        auto a = stmt->lhs;
        cast_to_result_type(a, stmt);
        const int exp = exponent;
        Stmt *result = nullptr;
        auto a_power_of_2 = a;
        int current_exponent = 1;
        while (true) {
          if (exp & current_exponent) {
            if (!result)
              result = a_power_of_2;
            else {
              auto new_result = Stmt::make<BinaryOpStmt>(BinaryOpType::mul,
                                                         result, a_power_of_2);
              new_result->ret_type.data_type = a->ret_type.data_type;
              result = new_result.get();
              modifier.insert_before(stmt, std::move(new_result));
            }
          }
          current_exponent <<= 1;
          if (current_exponent > exp)
            break;
          auto new_a_power = Stmt::make<BinaryOpStmt>(
              BinaryOpType::mul, a_power_of_2, a_power_of_2);
          new_a_power->ret_type.data_type = a->ret_type.data_type;
          a_power_of_2 = new_a_power.get();
          modifier.insert_before(stmt, std::move(new_a_power));
        }
        stmt->replace_with(result);
        modifier.erase(stmt);
      } else if (exponent == std::round(exponent) && exponent < 0 &&
                 exponent >= -max_weaken_exponent) {
        // a ** -n -> 1 / a ** n
        auto one = Stmt::make<ConstStmt>(LaneAttribute<TypedConstant>(1));
        auto one_raw = one.get();
        modifier.insert_before(stmt, std::move(one));
        cast_to_result_type(one_raw, stmt);
        auto new_exponent = Stmt::make<UnaryOpStmt>(UnaryOpType::neg, rhs);
        auto a_to_n = Stmt::make<BinaryOpStmt>(BinaryOpType::pow, stmt->lhs,
                                               new_exponent.get());
        a_to_n->ret_type.data_type = stmt->ret_type.data_type;
        auto result =
            Stmt::make<BinaryOpStmt>(BinaryOpType::div, one_raw, a_to_n.get());
        stmt->replace_with(result.get());
        modifier.insert_before(stmt, std::move(new_exponent));
        modifier.insert_before(stmt, std::move(a_to_n));
        modifier.insert_before(stmt, std::move(result));
        modifier.erase(stmt);
      }
    } else if (stmt->op_type == BinaryOpType::bit_and) {
      if (alg_is_minus_one(rhs)) {
        // a & -1 -> a
        stmt->replace_with(stmt->lhs);
        modifier.erase(stmt);
      } else if (alg_is_minus_one(lhs)) {
        // -1 & a -> a
        stmt->replace_with(stmt->rhs);
        modifier.erase(stmt);
      }
    } else if (rhs && stmt->op_type == BinaryOpType::floordiv) {
      if (is_integral(rhs->element_type()) && is_integral(lhs->element_type())) {
        // @ti.func
        // def ifloordiv(a, b):
        //     r = ti.raw_div(a, b)
        //     if (a < 0) != (b < 0) and a and b * r != a:
        //         r = r - 1
        //     return r
        auto ret = Stmt::make<BinaryOpStmt>(
            BinaryOpType::div, lhs, rhs);
        auto zero = Stmt::make<ConstStmt>(LaneAttribute<TypedConstant>(0));
        auto lhs_ltz = Stmt::make<BinaryOpStmt>(
            BinaryOpType::cmp_lt, lhs, zero.get());
        auto rhs_ltz = Stmt::make<BinaryOpStmt>(
            BinaryOpType::cmp_lt, rhs, zero.get());
        auto rhs_mul_ret = Stmt::make<BinaryOpStmt>(
            BinaryOpType::mul, rhs, ret.get());
        auto cond1 = Stmt::make<BinaryOpStmt>(
            BinaryOpType::cmp_ne, lhs_ltz.get(), rhs_ltz.get());
        auto cond2 = Stmt::make<BinaryOpStmt>(
            BinaryOpType::cmp_eq, lhs, zero.get());
        auto cond3 = Stmt::make<BinaryOpStmt>(
            BinaryOpType::cmp_eq, rhs_mul_ret.get(), lhs);
        auto cond12 = Stmt::make<BinaryOpStmt>(
            BinaryOpType::bit_and, cond1.get(), cond2.get());
        auto cond = Stmt::make<BinaryOpStmt>(
            BinaryOpType::bit_and, cond12.get(), cond3.get());
        auto real_ret = Stmt::make<BinaryOpStmt>(
            BinaryOpType::add, ret.get(), cond.get());

        modifier.insert_before(stmt, std::move(ret));
        modifier.insert_before(stmt, std::move(zero));
        modifier.insert_before(stmt, std::move(lhs_ltz));
        modifier.insert_before(stmt, std::move(rhs_ltz));
        modifier.insert_before(stmt, std::move(rhs_mul_ret));
        modifier.insert_before(stmt, std::move(cond1));
        modifier.insert_before(stmt, std::move(cond2));
        modifier.insert_before(stmt, std::move(cond3));
        modifier.insert_before(stmt, std::move(cond12));
        modifier.insert_before(stmt, std::move(cond));
        modifier.insert_before(stmt, std::move(real_ret));
        modifier.erase(stmt);
      } else {
        // @ti.func
        // def ffloordiv(a, b):
        //     r = ti.raw_div(a, b)
        //     return ti.floor(r)
        // TODO
      }
    }
  }

  void visit(AssertStmt *stmt) override {
    auto cond = stmt->cond->cast<ConstStmt>();
    if (!cond)
      return;
    if (!alg_is_zero(cond)) {
      // this statement has no effect
      modifier.erase(stmt);
    }
  }

  void visit(WhileControlStmt *stmt) override {
    auto cond = stmt->cond->cast<ConstStmt>();
    if (!cond)
      return;
    if (!alg_is_zero(cond)) {
      // this statement has no effect
      modifier.erase(stmt);
    }
  }

  static bool alg_is_zero(ConstStmt *stmt) {
    if (!stmt || stmt->width() != 1)
      return false;
    return stmt->val[0].equal_value(0);
  }

  static bool alg_is_one(ConstStmt *stmt) {
    if (!stmt || stmt->width() != 1)
      return false;
    return stmt->val[0].equal_value(1);
  }

  static bool alg_is_two(ConstStmt *stmt) {
    if (!stmt || stmt->width() != 1)
      return false;
    return stmt->val[0].equal_value(2);
  }

  static bool alg_is_minus_one(ConstStmt *stmt) {
    if (!stmt || stmt->width() != 1)
      return false;
    return stmt->val[0].equal_value(-1);
  }

  static bool run(IRNode *node, bool fast_math) {
    AlgSimp simplifier(fast_math);
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

namespace hack {
bool use_fast_math(IRNode *root) {
  const Kernel *kernel = root->get_kernel();
  if (!kernel) {
    return false;
  }
  return kernel->program.config.fast_math;
}
}  // namespace hack

bool alg_simp(IRNode *root) {
  TI_AUTO_PROF;
  return AlgSimp::run(root, hack::use_fast_math(root));
}

}  // namespace irpass

TLANG_NAMESPACE_END
