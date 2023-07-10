#include <cmath>
#include <deque>
#include <set>
#include <thread>

#include "taichi/ir/ir.h"
#include "taichi/ir/snode.h"
#include "taichi/ir/statements.h"
#include "taichi/ir/transforms.h"
#include "taichi/ir/visitors.h"
#include "taichi/transforms/constant_fold.h"
#include "taichi/program/program.h"

namespace taichi::lang {
namespace {
template <typename T>
T sar(T value, unsigned int amount) {
  return value < 0 ? ~(~value >> amount) : value >> amount;
}

template <typename T>
T shr(T value, unsigned int shift) {
  return static_cast<T>(
      static_cast<typename std::make_unsigned<T>::type>(value) >> shift);
}
}  // namespace

class ConstantFold : public BasicStmtVisitor {
 public:
  using BasicStmtVisitor::visit;
  DelayedIRModifier modifier;

  static bool is_good_type(DataType dt) {
    // ConstStmt of `bad` types like `i8` is not supported by LLVM.
    // Discussion:
    // https://github.com/taichi-dev/taichi/pull/839#issuecomment-625902727
    if (dt->is_primitive(PrimitiveTypeID::i32) ||
        dt->is_primitive(PrimitiveTypeID::i64) ||
        dt->is_primitive(PrimitiveTypeID::u1) ||
        dt->is_primitive(PrimitiveTypeID::u32) ||
        dt->is_primitive(PrimitiveTypeID::u64) ||
        dt->is_primitive(PrimitiveTypeID::f32) ||
        dt->is_primitive(PrimitiveTypeID::f64))
      return true;
    else
      return false;
  }

  std::optional<TypedConstant> get_scalar_value_to_replace(BinaryOpStmt *stmt,
                                                           ConstStmt *lhs,
                                                           ConstStmt *rhs,
                                                           DataType dst_type) {
    TypedConstant new_constant(dst_type);

    if (stmt->op_type == BinaryOpType::pow) {
      if (is_integral(rhs->ret_type)) {
        auto rhs_val = rhs->val.val_int();
        if (rhs_val < 0 && is_integral(dst_type)) {
          TI_ERROR("Negative exponent in pow(int, int) is not allowed.");
        }
      }
    }

    // Type check should have been done at this point.
    auto dt = lhs->val.dt;

    std::optional<TypedConstant> res = std::nullopt;
    switch (stmt->op_type) {
#define COMMA ,
#define HANDLE_REAL_AND_INTEGRAL_BINARY(OP_TYPE, PREFIX, OP_CPP)              \
  case BinaryOpType::OP_TYPE: {                                               \
    if (dt->is_primitive(PrimitiveTypeID::f32) ||                             \
        dt->is_primitive(PrimitiveTypeID::f64)) {                             \
      res = TypedConstant(dst_type,                                           \
                          PREFIX(lhs->val.val_cast_to_float64()               \
                                     OP_CPP rhs->val.val_cast_to_float64())); \
    } else if (dt->is_primitive(PrimitiveTypeID::i32) ||                      \
               dt->is_primitive(PrimitiveTypeID::i64)) {                      \
      res = TypedConstant(                                                    \
          dst_type, PREFIX(lhs->val.val_int() OP_CPP rhs->val.val_int()));    \
    } else if (dt->is_primitive(PrimitiveTypeID::u32) ||                      \
               dt->is_primitive(PrimitiveTypeID::u64)) {                      \
      res = TypedConstant(                                                    \
          dst_type, PREFIX(lhs->val.val_uint() OP_CPP rhs->val.val_uint()));  \
    } else if (dt->is_primitive(PrimitiveTypeID::u1)) {                       \
      res = TypedConstant(dst_type,                                           \
                          PREFIX(int32(lhs->val.val_uint1())                  \
                                     OP_CPP int32(rhs->val.val_uint1())));    \
    }                                                                         \
    break;                                                                    \
  }

      HANDLE_REAL_AND_INTEGRAL_BINARY(mul, , *)
      HANDLE_REAL_AND_INTEGRAL_BINARY(add, , +)
      HANDLE_REAL_AND_INTEGRAL_BINARY(sub, , -)
      HANDLE_REAL_AND_INTEGRAL_BINARY(floordiv, std::floor, /)
      HANDLE_REAL_AND_INTEGRAL_BINARY(div, , /)
      HANDLE_REAL_AND_INTEGRAL_BINARY(cmp_lt, , <)
      HANDLE_REAL_AND_INTEGRAL_BINARY(cmp_le, , <=)
      HANDLE_REAL_AND_INTEGRAL_BINARY(cmp_gt, , >)
      HANDLE_REAL_AND_INTEGRAL_BINARY(cmp_ge, , >=)
      HANDLE_REAL_AND_INTEGRAL_BINARY(cmp_eq, , ==)
      HANDLE_REAL_AND_INTEGRAL_BINARY(cmp_ne, , !=)

      HANDLE_REAL_AND_INTEGRAL_BINARY(max, std::max, COMMA)
      HANDLE_REAL_AND_INTEGRAL_BINARY(min, std::min, COMMA)
      HANDLE_REAL_AND_INTEGRAL_BINARY(atan2, std::atan2, COMMA)
      HANDLE_REAL_AND_INTEGRAL_BINARY(pow, std::pow, COMMA)
#undef HANDLE_REAL_AND_INTEGRAL_BINARY

#define HANDLE_INTEGRAL_BINARY(OP_TYPE, PREFIX, OP_CPP)                        \
  case BinaryOpType::OP_TYPE: {                                                \
    if (dt->is_primitive(PrimitiveTypeID::i32)) {                              \
      res = TypedConstant(                                                     \
          dst_type, PREFIX(lhs->val.val_int32() OP_CPP rhs->val.val_int32())); \
    } else if (dt->is_primitive(PrimitiveTypeID::i64)) {                       \
      res = TypedConstant(                                                     \
          dst_type, PREFIX(lhs->val.val_int() OP_CPP rhs->val.val_int()));     \
    } else if (dt->is_primitive(PrimitiveTypeID::u32) ||                       \
               dt->is_primitive(PrimitiveTypeID::u64)) {                       \
      res = TypedConstant(                                                     \
          dst_type, PREFIX(lhs->val.val_uint() OP_CPP rhs->val.val_uint()));   \
    }                                                                          \
    break;                                                                     \
  }

      HANDLE_INTEGRAL_BINARY(mod, , %)
      HANDLE_INTEGRAL_BINARY(bit_and, , &)
      HANDLE_INTEGRAL_BINARY(bit_or, , |)
      HANDLE_INTEGRAL_BINARY(bit_xor, , ^)
      HANDLE_INTEGRAL_BINARY(bit_shl, , <<)
      HANDLE_INTEGRAL_BINARY(bit_shr, shr, COMMA)
      HANDLE_INTEGRAL_BINARY(bit_sar, sar, COMMA)
#undef HANDLE_INTEGRAL_BINARY
#undef COMMA

      case BinaryOpType::truediv:
        TI_ERROR("{} should have been lowered.",
                 binary_op_type_name(stmt->op_type));
        break;

      default:
        break;
    }

    return res;
  }

  void visit(BinaryOpStmt *stmt) override {
    auto lhs = stmt->lhs;
    auto rhs = stmt->rhs;

    if (lhs->is<ConstStmt>() && rhs->is<ConstStmt>()) {
      auto typed_constant = get_scalar_value_to_replace(
          stmt, lhs->as<ConstStmt>(), rhs->as<ConstStmt>(), stmt->ret_type);
      if (!typed_constant)
        return;

      TypedConstant new_constant = *typed_constant;
      insert_and_erase(stmt, new_constant);
    } else if (lhs->is<MatrixInitStmt>() && rhs->is<MatrixInitStmt>()) {
      int num_values = rhs->as<MatrixInitStmt>()->values.size();

      std::vector<TypedConstant> typed_constants;
      for (int i = 0; i < num_values; i++) {
        auto scalar_lhs =
            lhs->as<MatrixInitStmt>()->values[i]->cast<ConstStmt>();
        auto scalar_rhs =
            rhs->as<MatrixInitStmt>()->values[i]->cast<ConstStmt>();
        if (!scalar_lhs || !scalar_rhs)
          return;

        auto typed_constant = get_scalar_value_to_replace(
            stmt, scalar_lhs, scalar_rhs, stmt->ret_type.get_element_type());
        if (!typed_constant)
          return;

        TypedConstant new_constant = *typed_constant;
        typed_constants.push_back(new_constant);
      }
      insert_and_erase(stmt, typed_constants);
    }
  }

  std::optional<TypedConstant> get_scalar_value_to_replace(UnaryOpStmt *stmt,
                                                           ConstStmt *operand,
                                                           DataType dst_type) {
    if (stmt->is_cast() && stmt->op_type == UnaryOpType::cast_bits) {
      TypedConstant new_constant(dst_type);
      new_constant.value_bits = operand->val.value_bits;
      return new_constant;
    }
    const auto dt = operand->val.dt;
    if (!is_good_type(dt))
      return std::nullopt;

    std::optional<TypedConstant> res = std::nullopt;
    switch (stmt->op_type) {
#define HANDLE_REAL_AND_INTEGRAL_UNARY(OP_TYPE, OP_CPP)                \
  case UnaryOpType::OP_TYPE: {                                         \
    if (dt->is_primitive(PrimitiveTypeID::f32) ||                      \
        dt->is_primitive(PrimitiveTypeID::f64)) {                      \
      res = TypedConstant(dst_type, OP_CPP(operand->val.val_float())); \
    } else if (dt->is_primitive(PrimitiveTypeID::i32) ||               \
               dt->is_primitive(PrimitiveTypeID::i64)) {               \
      res = TypedConstant(dst_type, OP_CPP(operand->val.val_int()));   \
    } else if (dt->is_primitive(PrimitiveTypeID::u32) ||               \
               dt->is_primitive(PrimitiveTypeID::u64)) {               \
      res = TypedConstant(dst_type, OP_CPP(operand->val.val_uint()));  \
    }                                                                  \
    break;                                                             \
  }

      HANDLE_REAL_AND_INTEGRAL_UNARY(neg, -)
      HANDLE_REAL_AND_INTEGRAL_UNARY(sqrt, std::sqrt)
      HANDLE_REAL_AND_INTEGRAL_UNARY(round, std::round)
      HANDLE_REAL_AND_INTEGRAL_UNARY(floor, std::floor)
      HANDLE_REAL_AND_INTEGRAL_UNARY(ceil, std::ceil)
      HANDLE_REAL_AND_INTEGRAL_UNARY(abs, std::fabs)
      HANDLE_REAL_AND_INTEGRAL_UNARY(sin, std::sin)
      HANDLE_REAL_AND_INTEGRAL_UNARY(asin, std::asin)
      HANDLE_REAL_AND_INTEGRAL_UNARY(cos, std::cos)
      HANDLE_REAL_AND_INTEGRAL_UNARY(acos, std::acos)
      HANDLE_REAL_AND_INTEGRAL_UNARY(tan, std::tan)
      HANDLE_REAL_AND_INTEGRAL_UNARY(tanh, std::tanh)
      HANDLE_REAL_AND_INTEGRAL_UNARY(log, std::log)
      HANDLE_REAL_AND_INTEGRAL_UNARY(exp, std::exp)
      HANDLE_REAL_AND_INTEGRAL_UNARY(rsqrt, 1.0 / std::sqrt)
#undef HANDLE_REAL_AND_INTEGRAL_UNARY

#define HANDLE_INTEGRAL_UNARY(OP_TYPE, OP_CPP)                        \
  case UnaryOpType::OP_TYPE: {                                        \
    if (dt->is_primitive(PrimitiveTypeID::i32) ||                     \
        dt->is_primitive(PrimitiveTypeID::i64)) {                     \
      res = TypedConstant(dst_type, OP_CPP(operand->val.val_int()));  \
    } else if (dt->is_primitive(PrimitiveTypeID::u32) ||              \
               dt->is_primitive(PrimitiveTypeID::u64)) {              \
      res = TypedConstant(dst_type, OP_CPP(operand->val.val_uint())); \
    } else if (dt->is_primitive(PrimitiveTypeID::u1)) {               \
      res = TypedConstant(dst_type, !operand->val.val_uint1());       \
    }                                                                 \
    break;                                                            \
  }

      HANDLE_INTEGRAL_UNARY(bit_not, ~)
      HANDLE_INTEGRAL_UNARY(logic_not, !)
#undef HANDLE_INTEGRAL_UNARY

      case UnaryOpType::cast_value: {
        if (dt->is_primitive(PrimitiveTypeID::f32) ||
            dt->is_primitive(PrimitiveTypeID::f64)) {
          res = TypedConstant(dst_type, operand->val.val_float());
        } else if (dt->is_primitive(PrimitiveTypeID::i32) ||
                   dt->is_primitive(PrimitiveTypeID::i64)) {
          res = TypedConstant(dst_type, operand->val.val_int());
        } else if (dt->is_primitive(PrimitiveTypeID::u32) ||
                   dt->is_primitive(PrimitiveTypeID::u64)) {
          res = TypedConstant(dst_type, operand->val.val_uint());
        } else if (dt->is_primitive(PrimitiveTypeID::u1)) {
          res = TypedConstant(dst_type, operand->val.val_uint1());
        }
        break;
      }
      default:
        return std::nullopt;
    }
    return res;
  }

  void visit(UnaryOpStmt *stmt) override {
    if (stmt->is_cast() && stmt->cast_type == stmt->operand->ret_type) {
      stmt->replace_usages_with(stmt->operand);
      modifier.erase(stmt);
      return;
    }

    if (auto operand = stmt->operand->cast<ConstStmt>()) {
      auto typed_constant =
          get_scalar_value_to_replace(stmt, operand, stmt->ret_type);
      if (!typed_constant)
        return;

      TypedConstant new_constant = *typed_constant;
      insert_and_erase(stmt, new_constant);
    } else if (auto operand = stmt->operand->cast<MatrixInitStmt>()) {
      std::vector<TypedConstant> typed_constants;
      for (auto &scalar_operand : operand->values) {
        auto const_scalar_operand = scalar_operand->cast<ConstStmt>();
        if (!const_scalar_operand)
          return;

        auto typed_constant = get_scalar_value_to_replace(
            stmt, const_scalar_operand, stmt->ret_type.get_element_type());
        if (!typed_constant)
          return;

        TypedConstant new_constant = *typed_constant;
        typed_constants.push_back(new_constant);
      }
      insert_and_erase(stmt, typed_constants);
    }
  }

  static bool run(IRNode *node) {
    ConstantFold folder;
    bool modified = false;

    while (true) {
      node->accept(&folder);
      if (folder.modifier.modify_ir()) {
        modified = true;
      } else {
        break;
      }
    }

    return modified;
  }

 private:
  void insert_and_erase(Stmt *stmt, const TypedConstant &new_constant) {
    auto evaluated = Stmt::make<ConstStmt>(new_constant);
    stmt->replace_usages_with(evaluated.get());
    modifier.insert_before(stmt, std::move(evaluated));
    modifier.erase(stmt);
  }

  void insert_and_erase(Stmt *stmt,
                        const std::vector<TypedConstant> &new_constants) {
    std::vector<Stmt *> values;
    for (auto &new_constant : new_constants) {
      auto const_stmt = Stmt::make<ConstStmt>(new_constant);
      values.push_back(const_stmt.get());
      modifier.insert_before(stmt, std::move(const_stmt));
    }

    auto evaluated = Stmt::make<MatrixInitStmt>(values);
    evaluated->ret_type = stmt->ret_type;

    stmt->replace_usages_with(evaluated.get());
    modifier.insert_before(stmt, std::move(evaluated));
    modifier.erase(stmt);
  }
};

const PassID ConstantFoldPass::id = "ConstantFoldPass";

namespace irpass {

bool constant_fold(IRNode *root) {
  TI_AUTO_PROF;
  return ConstantFold::run(root);
}

}  // namespace irpass

}  // namespace taichi::lang
