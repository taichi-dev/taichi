#include "taichi/ir/analysis.h"
#include "taichi/ir/ir.h"
#include "taichi/ir/statements.h"
#include "taichi/ir/transforms.h"
#include "taichi/ir/visitors.h"
#include "taichi/program/program.h"
#include "taichi/util/bit.h"

namespace taichi::lang {

// Algebraic Simplification and Strength Reduction
class AlgSimp : public BasicStmtVisitor {
  static constexpr int max_weaken_exponent = 32;

 private:
  void cast_to_result_type(Stmt *&a, Stmt *stmt) {
    if (stmt->ret_type != a->ret_type) {
      auto cast = Stmt::make_typed<UnaryOpStmt>(UnaryOpType::cast_value, a);
      cast->cast_type = stmt->ret_type;
      cast->ret_type = stmt->ret_type;
      a = cast.get();
      modifier.insert_before(stmt, std::move(cast));
    }
  }

  void replace_with_zero(Stmt *stmt) {
    auto stmts = get_const_stmt_with_value(stmt->ret_type, 0);
    auto zero = stmts.back().get();

    for (auto &s : stmts) {
      modifier.insert_before(stmt, std::move(s));
    }
    stmt->replace_usages_with(zero);
    modifier.erase(stmt);
  }

  void replace_with_one(Stmt *stmt) {
    auto stmts = get_const_stmt_with_value(stmt->ret_type, 1);
    auto one = stmts.back().get();

    for (auto &s : stmts) {
      modifier.insert_before(stmt, std::move(s));
    }
    stmt->replace_usages_with(one);
    modifier.erase(stmt);
  }

  std::vector<float64> get_exponent_values(BinaryOpStmt *stmt) {
    std::vector<float64> rets;
    if (stmt->rhs->is<ConstStmt>()) {
      rets.push_back(stmt->rhs->as<ConstStmt>()->val.val_cast_to_float64());
      return rets;
    } else if (stmt->rhs->is<MatrixInitStmt>()) {
      auto matrix_init = stmt->rhs->as<MatrixInitStmt>();
      for (auto &value : matrix_init->values) {
        if (value->is<ConstStmt>()) {
          rets.push_back(value->as<ConstStmt>()->val.val_cast_to_float64());
        } else {
          return {};
        }
      }
      return rets;
    }
    return {};
  }

  Stmt *get_inverse(BinaryOpStmt *stmt) {
    auto rhs = stmt->rhs;
    if (rhs->is<ConstStmt>()) {
      auto reciprocal =
          Stmt::make_typed<ConstStmt>(TypedConstant(rhs->ret_type));
      if (rhs->ret_type->is_primitive(PrimitiveTypeID::f64)) {
        reciprocal->val.val_float64() =
            (float64)1.0 / rhs->as<ConstStmt>()->val.val_float64();
      } else if (rhs->ret_type->is_primitive(PrimitiveTypeID::f32)) {
        reciprocal->val.val_float32() =
            (float32)1.0 / rhs->as<ConstStmt>()->val.val_float32();
      } else {
        TI_NOT_IMPLEMENTED
      }
      Stmt *reciprocal_ptr = reciprocal.get();
      modifier.insert_before(stmt, std::move(reciprocal));
      return reciprocal_ptr;
    } else {
      auto matrix_rhs = rhs->cast<MatrixInitStmt>();
      TI_ASSERT(matrix_rhs != nullptr);
      std::vector<Stmt *> values;
      for (auto scalar_stmt : matrix_rhs->values) {
        auto const_stmt = scalar_stmt->cast<ConstStmt>();
        TI_ASSERT(const_stmt != nullptr);
        auto reciprocal =
            Stmt::make_typed<ConstStmt>(TypedConstant(scalar_stmt->ret_type));
        if (scalar_stmt->ret_type->is_primitive(PrimitiveTypeID::f64)) {
          reciprocal->val.val_float64() =
              (float64)1.0 / const_stmt->val.val_float64();
        } else if (scalar_stmt->ret_type->is_primitive(PrimitiveTypeID::f32)) {
          reciprocal->val.val_float32() =
              (float32)1.0 / const_stmt->val.val_float32();
        } else {
          TI_NOT_IMPLEMENTED
        }
        values.push_back(reciprocal.get());
        modifier.insert_before(stmt, std::move(reciprocal));
      }
      auto new_rhs = Stmt::make<MatrixInitStmt>(values);
      new_rhs->ret_type = rhs->ret_type;

      Stmt *new_rhs_ptr = new_rhs.get();
      modifier.insert_before(stmt, std::move(new_rhs));
      return new_rhs_ptr;
    }
  }

  Stmt *get_log2rhs(BinaryOpStmt *stmt) {
    auto rhs = stmt->rhs;
    if (auto const_rhs = stmt->rhs->cast<ConstStmt>()) {
      int log2rhs =
          bit::log2int((uint64)rhs->as<ConstStmt>()->val.val_as_int64());
      auto new_rhs =
          Stmt::make<ConstStmt>(TypedConstant(stmt->lhs->ret_type, log2rhs));
      Stmt *new_rhs_ptr = new_rhs.get();
      modifier.insert_before(stmt, std::move(new_rhs));
      return new_rhs_ptr;
    } else {
      auto matrix_rhs = rhs->cast<MatrixInitStmt>();
      TI_ASSERT(matrix_rhs != nullptr);

      std::vector<Stmt *> values;
      for (auto scalar_stmt : matrix_rhs->values) {
        auto const_stmt = scalar_stmt->cast<ConstStmt>();
        TI_ASSERT(const_stmt != nullptr);
        int log2int = bit::log2int((uint64)const_stmt->val.val_as_int64());
        auto log2int_stmt = Stmt::make<ConstStmt>(
            TypedConstant(scalar_stmt->ret_type, log2int));

        values.push_back(log2int_stmt.get());
        modifier.insert_before(stmt, std::move(log2int_stmt));
      }

      auto new_rhs = Stmt::make<MatrixInitStmt>(values);
      new_rhs->ret_type = rhs->ret_type;

      Stmt *new_rhs_ptr = new_rhs.get();
      modifier.insert_before(stmt, std::move(new_rhs));
      return new_rhs_ptr;
    }
  }

  bool exponent_one_optimize(BinaryOpStmt *stmt) {
    auto exponents = get_exponent_values(stmt);
    if (exponents.empty()) {
      return false;
    }

    for (auto exponent : exponents) {
      if (exponent != 1) {
        return false;
      }
    }

    stmt->replace_usages_with(stmt->lhs);
    modifier.erase(stmt);
    return true;
  }

  bool exponent_zero_optimize(BinaryOpStmt *stmt) {
    auto exponents = get_exponent_values(stmt);
    if (exponents.empty()) {
      return false;
    }

    for (auto exponent : exponents) {
      if (exponent != 0) {
        return false;
      }
    }

    replace_with_one(stmt);
    return true;
  }

  bool exponent_half_optimize(BinaryOpStmt *stmt) {
    auto exponents = get_exponent_values(stmt);
    if (exponents.empty()) {
      return false;
    }

    for (auto exponent : exponents) {
      if (exponent != 0.5) {
        return false;
      }
    }
    auto a = stmt->lhs;
    cast_to_result_type(a, stmt);
    auto result = Stmt::make<UnaryOpStmt>(UnaryOpType::sqrt, a);
    result->ret_type = a->ret_type;
    stmt->replace_usages_with(result.get());
    modifier.insert_before(stmt, std::move(result));
    modifier.erase(stmt);
    return true;
  }

  bool exponent_n_optimize(BinaryOpStmt *stmt) {
    auto exponents = get_exponent_values(stmt);
    if (exponents.empty()) {
      return false;
    }

    // Equal exponents
    for (auto exponent : exponents) {
      if (exponent != exponents[0]) {
        return false;
      }
    }

    float64 exponent = exponents[0];
    if (!(exponent == std::round(exponent) && exponent > 0 &&
          exponent <= max_weaken_exponent)) {
      return false;
    }

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
          auto new_result =
              Stmt::make<BinaryOpStmt>(BinaryOpType::mul, result, a_power_of_2);
          new_result->ret_type = a->ret_type;
          result = new_result.get();
          modifier.insert_before(stmt, std::move(new_result));
        }
      }
      current_exponent <<= 1;
      if (current_exponent > exp)
        break;
      auto new_a_power = Stmt::make<BinaryOpStmt>(BinaryOpType::mul,
                                                  a_power_of_2, a_power_of_2);
      new_a_power->ret_type = a->ret_type;
      a_power_of_2 = new_a_power.get();
      modifier.insert_before(stmt, std::move(new_a_power));
    }
    stmt->replace_usages_with(result);
    modifier.erase(stmt);
    return true;
  }

  bool exponent_negative_optimize(BinaryOpStmt *stmt) {
    auto exponents = get_exponent_values(stmt);
    if (exponents.empty()) {
      return false;
    }

    // All negative exponent
    for (auto exponent : exponents) {
      if (!(exponent == std::round(exponent) && exponent < 0 &&
            exponent >= -max_weaken_exponent)) {
        return false;
      }
    }

    if (is_integral(stmt->lhs->ret_type.get_element_type())) {
      TI_ERROR("Negative exponent in pow(int, int) is not allowed.");
    }

    auto stmts = get_const_stmt_with_value(stmt->ret_type, 1);
    auto one = stmts.back().get();

    for (auto &s : stmts) {
      modifier.insert_before(stmt, std::move(s));
    }

    cast_to_result_type(one, stmt);
    auto new_exponent = Stmt::make<UnaryOpStmt>(UnaryOpType::neg, stmt->rhs);
    new_exponent->ret_type = stmt->rhs->ret_type;
    auto a_to_n = Stmt::make<BinaryOpStmt>(BinaryOpType::pow, stmt->lhs,
                                           new_exponent.get());
    a_to_n->ret_type = stmt->ret_type;
    auto result =
        Stmt::make<BinaryOpStmt>(BinaryOpType::div, one, a_to_n.get());
    result->ret_type = stmt->ret_type;
    stmt->replace_usages_with(result.get());
    modifier.insert_before(stmt, std::move(new_exponent));
    modifier.insert_before(stmt, std::move(a_to_n));
    modifier.insert_before(stmt, std::move(result));
    modifier.erase(stmt);
    return true;
  }

 public:
  using BasicStmtVisitor::visit;
  bool fast_math;
  DelayedIRModifier modifier;

  explicit AlgSimp(bool fast_math_) : fast_math(fast_math_) {
  }

  [[nodiscard]] bool is_redundant_cast(const DataType &first_cast,
                                       const DataType &second_cast) const {
    // Tests if second_cast(first_cast(a)) is guaranteed to be equivalent to
    // second_cast(a).
    if (!first_cast->is<PrimitiveType>() || !second_cast->is<PrimitiveType>()) {
      // TODO(type): handle this case
      return false;
    }
    if (is_real(second_cast)) {
      // float(...(a))
      return is_real(first_cast) &&
             data_type_bits(second_cast) <= data_type_bits(first_cast);
    }
    if (is_integral(first_cast)) {
      // int(int(a)), note it's not always equivalent when signedness differ,
      // see #7915
      return data_type_bits(second_cast) <= data_type_bits(first_cast) &&
             is_signed(second_cast) == is_signed(first_cast);
    }
    // int(float(a))
    if (data_type_bits(second_cast) <= data_type_bits(first_cast) * 2) {
      // f64 can hold any i32 values.
      return true;
    } else {
      // Assume a floating point type can hold any integer values when
      // fast_math=True.
      return fast_math;
    }
  }

  void visit(UnaryOpStmt *stmt) override {
    if (stmt->is_cast()) {
      if (stmt->cast_type == stmt->operand->ret_type) {
        stmt->replace_usages_with(stmt->operand);
        modifier.erase(stmt);
      } else if (stmt->operand->is<UnaryOpStmt>() &&
                 stmt->operand->as<UnaryOpStmt>()->is_cast()) {
        auto prev_cast = stmt->operand->as<UnaryOpStmt>();
        if (stmt->op_type == UnaryOpType::cast_bits &&
            prev_cast->op_type == UnaryOpType::cast_bits) {
          stmt->operand = prev_cast->operand;
          modifier.mark_as_modified();
        } else if (stmt->op_type == UnaryOpType::cast_value &&
                   prev_cast->op_type == UnaryOpType::cast_value &&
                   is_redundant_cast(prev_cast->cast_type, stmt->cast_type)) {
          stmt->operand = prev_cast->operand;
          modifier.mark_as_modified();
        }
      }
    }
  }

  bool optimize_multiplication(BinaryOpStmt *stmt) {
    // return true iff the IR is modified
    auto lhs = stmt->lhs;
    auto rhs = stmt->rhs;
    TI_ASSERT(stmt->op_type == BinaryOpType::mul);
    if (alg_is_one(lhs) || alg_is_one(rhs)) {
      // 1 * a -> a, a * 1 -> a
      stmt->replace_usages_with(alg_is_one(lhs) ? stmt->rhs : stmt->lhs);
      modifier.erase(stmt);
      return true;
    }
    if ((fast_math || is_integral(stmt->ret_type.get_element_type())) &&
        (alg_is_zero(lhs) || alg_is_zero(rhs))) {
      // fast_math or integral operands: 0 * a -> 0, a * 0 -> 0
      replace_with_zero(stmt);
      return true;
    }
    if (is_integral(stmt->ret_type.get_element_type()) &&
        (alg_is_pot(lhs) || alg_is_pot(rhs))) {
      // a * pot -> a << log2(pot)
      if (alg_is_pot(lhs)) {
        std::swap(stmt->lhs, stmt->rhs);
        std::swap(lhs, rhs);
      }

      Stmt *new_rhs = get_log2rhs(stmt);
      auto result =
          Stmt::make<BinaryOpStmt>(BinaryOpType::bit_shl, stmt->lhs, new_rhs);
      result->ret_type = stmt->ret_type;

      result->dbg_info = stmt->dbg_info;
      stmt->replace_usages_with(result.get());
      modifier.insert_before(stmt, std::move(result));
      modifier.erase(stmt);
      return true;
    }
    if (alg_is_two(lhs) || alg_is_two(rhs)) {
      // 2 * a -> a + a, a * 2 -> a + a
      auto a = stmt->lhs;
      if (alg_is_two(lhs))
        a = stmt->rhs;
      cast_to_result_type(a, stmt);
      auto sum = Stmt::make<BinaryOpStmt>(BinaryOpType::add, a, a);
      sum->ret_type = a->ret_type;
      sum->dbg_info = stmt->dbg_info;
      stmt->replace_usages_with(sum.get());
      modifier.insert_before(stmt, std::move(sum));
      modifier.erase(stmt);
      return true;
    }
    return false;
  }

  bool optimize_division(BinaryOpStmt *stmt) {
    // return true iff the IR is modified
    auto rhs = stmt->rhs;
    TI_ASSERT(stmt->op_type == BinaryOpType::div ||
              stmt->op_type == BinaryOpType::floordiv);
    if (alg_is_one(rhs) && !(is_real(stmt->lhs->ret_type.get_element_type()) &&
                             stmt->op_type == BinaryOpType::floordiv)) {
      // a / 1 -> a
      stmt->replace_usages_with(stmt->lhs);
      modifier.erase(stmt);
      return true;
    }
    if ((fast_math || is_integral(stmt->ret_type.get_element_type())) &&
        irpass::analysis::same_value(stmt->lhs, stmt->rhs)) {
      // fast_math or integral operands: a / a -> 1
      replace_with_one(stmt);
      return true;
    }
    if (fast_math && alg_is_optimizable(rhs) &&
        is_real(rhs->ret_type.get_element_type()) &&
        stmt->op_type != BinaryOpType::floordiv) {
      if (alg_is_zero(rhs)) {
        TI_WARN("Potential division by 0\n{}", stmt->get_tb());
      } else {
        // a / const -> a * (1 / const)
        Stmt *new_rhs = get_inverse(stmt);

        auto product =
            Stmt::make<BinaryOpStmt>(BinaryOpType::mul, stmt->lhs, new_rhs);
        product->ret_type = stmt->ret_type;
        stmt->replace_usages_with(product.get());
        modifier.insert_before(stmt, std::move(product));
        modifier.erase(stmt);
        return true;
      }
    }
    if (is_integral(stmt->lhs->ret_type.get_element_type()) &&
        is_unsigned(stmt->lhs->ret_type.get_element_type()) &&
        alg_is_pot(rhs)) {
      // (unsigned)a / pot -> a >> log2(pot)

      Stmt *new_rhs = get_log2rhs(stmt);
      auto result =
          Stmt::make<BinaryOpStmt>(BinaryOpType::bit_sar, stmt->lhs, new_rhs);
      result->ret_type = stmt->ret_type;
      stmt->replace_usages_with(result.get());
      modifier.insert_before(stmt, std::move(result));
      modifier.erase(stmt);
      return true;
    }
    return false;
  }

  void visit(BinaryOpStmt *stmt) override {
    auto lhs = stmt->lhs;
    auto rhs = stmt->rhs;
    if (stmt->op_type == BinaryOpType::mul) {
      optimize_multiplication(stmt);
    } else if (stmt->op_type == BinaryOpType::div ||
               stmt->op_type == BinaryOpType::floordiv) {
      optimize_division(stmt);
    } else if (stmt->op_type == BinaryOpType::add ||
               stmt->op_type == BinaryOpType::sub ||
               stmt->op_type == BinaryOpType::bit_or ||
               stmt->op_type == BinaryOpType::bit_xor) {
      if (alg_is_zero(rhs)) {
        // a +-|^ 0 -> a
        stmt->replace_usages_with(stmt->lhs);
        modifier.erase(stmt);
      } else if (stmt->op_type != BinaryOpType::sub && alg_is_zero(lhs)) {
        // 0 +|^ a -> a
        stmt->replace_usages_with(stmt->rhs);
        modifier.erase(stmt);
      } else if (stmt->op_type == BinaryOpType::bit_or &&
                 irpass::analysis::same_value(stmt->lhs, stmt->rhs)) {
        // a | a -> a
        stmt->replace_usages_with(stmt->lhs);
        modifier.erase(stmt);
      } else if ((stmt->op_type == BinaryOpType::sub ||
                  stmt->op_type == BinaryOpType::bit_xor) &&
                 (fast_math ||
                  is_integral(stmt->ret_type.get_element_type())) &&
                 irpass::analysis::same_value(stmt->lhs, stmt->rhs)) {
        // fast_math or integral operands: a -^ a -> 0
        replace_with_zero(stmt);
      }
    } else if (stmt->op_type == BinaryOpType::pow) {
      if (exponent_one_optimize(stmt)) {
        // a ** 1 -> a
      } else if (exponent_zero_optimize(stmt)) {
        // a ** 0 -> 1
      } else if (exponent_half_optimize(stmt)) {
        // a ** 0.5 -> sqrt(a)
      } else if (exponent_n_optimize(stmt)) {
        // a ** n -> Exponentiation by squaring
      } else if (exponent_negative_optimize(stmt)) {
        // a ** -n -> 1 / a ** n
      }
    } else if (stmt->op_type == BinaryOpType::bit_and) {
      if (alg_is_minus_one(rhs)) {
        // a & -1 -> a
        stmt->replace_usages_with(stmt->lhs);
        modifier.erase(stmt);
      } else if (alg_is_minus_one(lhs)) {
        // -1 & a -> a
        stmt->replace_usages_with(stmt->rhs);
        modifier.erase(stmt);
      } else if (alg_is_zero(lhs) || alg_is_zero(rhs)) {
        // 0 & a -> 0, a & 0 -> 0
        replace_with_zero(stmt);
      } else if (irpass::analysis::same_value(stmt->lhs, stmt->rhs)) {
        // a & a -> a
        stmt->replace_usages_with(stmt->lhs);
        modifier.erase(stmt);
      }
    } else if (stmt->op_type == BinaryOpType::bit_sar ||
               stmt->op_type == BinaryOpType::bit_shl ||
               stmt->op_type == BinaryOpType::bit_shr) {
      if (alg_is_zero(rhs) || alg_is_zero(lhs)) {
        // a >> 0 -> a
        // a << 0 -> a
        // 0 << a -> 0
        // 0 >> a -> 0
        TI_ASSERT(stmt->lhs->ret_type == stmt->ret_type);
        stmt->replace_usages_with(stmt->lhs);
        modifier.erase(stmt);
      }
    } else if (is_comparison(stmt->op_type)) {
      if ((fast_math || is_integral(stmt->lhs->ret_type.get_element_type())) &&
          irpass::analysis::same_value(stmt->lhs, stmt->rhs)) {
        // fast_math or integral operands: a == a -> 1, a != a -> 0
        if (stmt->op_type == BinaryOpType::cmp_eq ||
            stmt->op_type == BinaryOpType::cmp_ge ||
            stmt->op_type == BinaryOpType::cmp_le) {
          replace_with_one(stmt);
        } else if (stmt->op_type == BinaryOpType::cmp_ne ||
                   stmt->op_type == BinaryOpType::cmp_gt ||
                   stmt->op_type == BinaryOpType::cmp_lt) {
          replace_with_zero(stmt);
        } else {
          TI_NOT_IMPLEMENTED
        }
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

  static bool alg_is_zero(Stmt *stmt) {
    if (!stmt)
      return false;
    if (auto const_stmt = stmt->cast<ConstStmt>()) {
      return const_stmt->val.equal_value(0);
    } else if (auto matrix_stmt = stmt->cast<MatrixInitStmt>()) {
      for (auto &val : matrix_stmt->values) {
        if (!alg_is_zero(val)) {
          return false;
        }
      }
      return true;
    }
    return false;
  }

  static bool alg_is_optimizable(Stmt *stmt) {
    if (!stmt)
      return false;
    if (auto const_stmt = stmt->cast<ConstStmt>()) {
      return true;
    } else if (auto matrix_stmt = stmt->cast<MatrixInitStmt>()) {
      for (auto &val : matrix_stmt->values) {
        if (!alg_is_optimizable(val)) {
          return false;
        }
      }
      return true;
    }
    return false;
  }

  static bool alg_is_one(Stmt *stmt) {
    if (!stmt)
      return false;
    if (auto const_stmt = stmt->cast<ConstStmt>()) {
      return const_stmt->val.equal_value(1);
    } else if (auto matrix_stmt = stmt->cast<MatrixInitStmt>()) {
      for (auto &val : matrix_stmt->values) {
        if (!alg_is_one(val)) {
          return false;
        }
      }
      return true;
    }
    return false;
  }

  static bool alg_is_two(Stmt *stmt) {
    if (!stmt)
      return false;
    if (auto const_stmt = stmt->cast<ConstStmt>()) {
      return const_stmt->val.equal_value(2);
    } else if (auto matrix_stmt = stmt->cast<MatrixInitStmt>()) {
      for (auto &val : matrix_stmt->values) {
        if (!alg_is_two(val)) {
          return false;
        }
      }
      return true;
    }
    return false;
  }

  static bool alg_is_minus_one(Stmt *stmt) {
    if (!stmt)
      return false;
    if (auto const_stmt = stmt->cast<ConstStmt>()) {
      return const_stmt->val.equal_value(-1);
    } else if (auto matrix_stmt = stmt->cast<MatrixInitStmt>()) {
      for (auto &val : matrix_stmt->values) {
        if (!alg_is_minus_one(val)) {
          return false;
        }
      }
      return true;
    }
    return false;
  }

  static bool alg_is_pot(Stmt *stmt) {
    if (!stmt)
      return false;

    if (auto const_stmt = stmt->cast<ConstStmt>()) {
      if (!is_integral(const_stmt->val.dt))
        return false;
      if (is_signed(const_stmt->val.dt)) {
        return bit::is_power_of_two(const_stmt->val.val_int());
      } else {
        return bit::is_power_of_two(const_stmt->val.val_uint());
      }
    } else if (auto matrix_stmt = stmt->cast<MatrixInitStmt>()) {
      for (auto &val : matrix_stmt->values) {
        if (!alg_is_pot(val)) {
          return false;
        }
      }
      return true;
    }
    return false;
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

bool alg_simp(IRNode *root, const CompileConfig &config) {
  TI_AUTO_PROF;
  return AlgSimp::run(root, config.fast_math);
}

}  // namespace irpass

}  // namespace taichi::lang
