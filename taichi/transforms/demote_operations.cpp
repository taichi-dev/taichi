#include "taichi/ir/ir.h"
#include "taichi/ir/ir_builder.h"
#include "taichi/ir/statements.h"
#include "taichi/ir/transforms.h"
#include "taichi/ir/visitors.h"
#include "taichi/program/program.h"

namespace taichi::lang {

// Demote Operations into pieces for backends to deal easier
class DemoteOperations : public BasicStmtVisitor {
 public:
  using BasicStmtVisitor::visit;
  DelayedIRModifier modifier;

  DemoteOperations() {
  }

  std::unique_ptr<Stmt> demote_ifloordiv(BinaryOpStmt *stmt,
                                         Stmt *lhs,
                                         Stmt *rhs) {
    auto ret = Stmt::make<BinaryOpStmt>(BinaryOpType::div, lhs, rhs);
    auto zero = Stmt::make<ConstStmt>(TypedConstant(0));

    if (lhs->ret_type->is<TensorType>()) {
      int num_elements = lhs->ret_type->cast<TensorType>()->get_num_elements();
      std::vector<Stmt *> values(num_elements, zero.get());

      auto matrix_zero = Stmt::make<MatrixInitStmt>(values);
      matrix_zero->ret_type = lhs->ret_type;

      modifier.insert_before(stmt, std::move(zero));
      zero = std::move(matrix_zero);
    }

    auto lhs_ltz =
        Stmt::make<BinaryOpStmt>(BinaryOpType::cmp_lt, lhs, zero.get());
    auto rhs_ltz =
        Stmt::make<BinaryOpStmt>(BinaryOpType::cmp_lt, rhs, zero.get());
    auto rhs_mul_ret =
        Stmt::make<BinaryOpStmt>(BinaryOpType::mul, rhs, ret.get());
    auto cond1 = Stmt::make<BinaryOpStmt>(BinaryOpType::cmp_ne, lhs_ltz.get(),
                                          rhs_ltz.get());
    auto cond2 =
        Stmt::make<BinaryOpStmt>(BinaryOpType::cmp_ne, lhs, zero.get());
    auto cond3 =
        Stmt::make<BinaryOpStmt>(BinaryOpType::cmp_ne, rhs_mul_ret.get(), lhs);
    auto cond12 = Stmt::make<BinaryOpStmt>(BinaryOpType::logical_and,
                                           cond1.get(), cond2.get());
    auto cond = Stmt::make<BinaryOpStmt>(BinaryOpType::logical_and,
                                         cond12.get(), cond3.get());
    auto real_ret =
        Stmt::make<BinaryOpStmt>(BinaryOpType::sub, ret.get(), cond.get());

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
    return real_ret;
  }

  std::unique_ptr<Stmt> demote_ffloor(BinaryOpStmt *stmt,
                                      Stmt *lhs,
                                      Stmt *rhs) {
    auto div = Stmt::make<BinaryOpStmt>(BinaryOpType::div, lhs, rhs);
    auto floor = Stmt::make<UnaryOpStmt>(UnaryOpType::floor, div.get());
    modifier.insert_before(stmt, std::move(div));
    return floor;
  }

  void visit(BinaryOpStmt *stmt) override {
    auto lhs = stmt->lhs;
    auto rhs = stmt->rhs;

    auto lhs_prim_type = lhs->ret_type.get_element_type();
    auto rhs_prim_type = rhs->ret_type.get_element_type();
    if (stmt->op_type == BinaryOpType::floordiv) {
      if (is_integral(rhs_prim_type) && is_integral(lhs_prim_type)) {
        // @ti.func
        // def ifloordiv(a, b):
        //     r = ti.raw_div(a, b)
        //     if (a < 0) != (b < 0) and a and b * r != a:
        //         r = r - 1
        //     return r
        //
        // simply `a * b < 0` may leads to overflow (#969)
        //
        // Formal Anti-Regression Verification (FARV):
        //
        // old = a * b < 0
        // new = (a < 0) != (b < 0) && a
        //
        //  a  b old new
        //  -  -  f = f (f&t)
        //  -  +  t = t (t&t)
        //  0  -  f = f (t&f)
        //  0  +  f = f (f&f)
        //  +  -  t = t (t&t)
        //  +  +  f = f (f&t)
        //
        // the situation of `b = 0` is ignored since we get FPE anyway.
        auto real_ret = demote_ifloordiv(stmt, lhs, rhs);
        real_ret->ret_type = stmt->ret_type;
        stmt->replace_usages_with(real_ret.get());
        modifier.insert_before(stmt, std::move(real_ret));
        modifier.erase(stmt);

      } else if (is_real(rhs_prim_type) || is_real(lhs_prim_type)) {
        // @ti.func
        // def ffloordiv(a, b):
        //     r = ti.raw_div(a, b)
        //     return ti.floor(r)
        auto floor = demote_ffloor(stmt, lhs, rhs);
        floor->ret_type = stmt->ret_type;
        stmt->replace_usages_with(floor.get());
        modifier.insert_before(stmt, std::move(floor));
        modifier.erase(stmt);
      }
    } else if (stmt->op_type == BinaryOpType::bit_shr) {
      // @ti.func
      // def bit_shr(a, b):
      //     unsigned_a = ti.cast(a, ti.uXX)
      //     shifted = ti.bit_sar(unsigned_a, b)
      //     ret = ti.cast(shifted, ti.iXX)
      //     return ret
      TI_ASSERT(is_integral(lhs_prim_type) && is_integral(rhs_prim_type));
      auto unsigned_cast = Stmt::make<UnaryOpStmt>(UnaryOpType::cast_bits, lhs);
      unsigned_cast->as<UnaryOpStmt>()->cast_type =
          is_signed(lhs_prim_type) ? to_unsigned(lhs_prim_type) : lhs_prim_type;
      auto shift = Stmt::make<BinaryOpStmt>(BinaryOpType::bit_sar,
                                            unsigned_cast.get(), rhs);
      auto signed_cast =
          Stmt::make<UnaryOpStmt>(UnaryOpType::cast_bits, shift.get());
      signed_cast->as<UnaryOpStmt>()->cast_type = lhs_prim_type;
      signed_cast->ret_type = stmt->ret_type;
      stmt->replace_usages_with(signed_cast.get());
      modifier.insert_before(stmt, std::move(unsigned_cast));
      modifier.insert_before(stmt, std::move(shift));
      modifier.insert_before(stmt, std::move(signed_cast));
      modifier.erase(stmt);
    } else if (stmt->op_type == BinaryOpType::pow &&
               is_integral(rhs_prim_type)) {
      // @ti.func
      // def pow(lhs, rhs):
      //     a = lhs
      //     b = abs(rhs)
      //     result = 1
      //     while b > 0:
      //         if b & 1:
      //             result *= a
      //         a *= a
      //         b >>= 1
      //     if rhs < 0:              # for real lhs
      //         result = 1 / result  # for real lhs
      //     return result
      IRBuilder builder;
      auto one_lhs = builder.get_constant(lhs_prim_type, 1);
      auto one_rhs = builder.get_constant(rhs_prim_type, 1);
      auto zero_rhs = builder.get_constant(rhs_prim_type, 0);
      auto a = builder.create_local_var(lhs_prim_type);
      builder.create_local_store(a, lhs);
      auto b = builder.create_local_var(rhs_prim_type);
      builder.create_local_store(b, builder.create_abs(rhs));
      auto result = builder.create_local_var(lhs_prim_type);
      builder.create_local_store(result, one_lhs);
      auto loop = builder.create_while_true();
      {
        auto loop_guard = builder.get_loop_guard(loop);
        auto current_a = builder.create_local_load(a);
        auto current_b = builder.create_local_load(b);
        auto if_stmt =
            builder.create_if(builder.create_cmp_le(current_b, zero_rhs));
        {
          auto _ = builder.get_if_guard(if_stmt, true);
          builder.create_break();
        }
        auto bit_and = builder.create_and(current_b, one_rhs);
        if_stmt = builder.create_if(builder.create_cmp_ne(bit_and, zero_rhs));
        {
          auto _ = builder.get_if_guard(if_stmt, true);
          auto current_result = builder.create_local_load(result);
          auto new_result = builder.create_mul(current_result, current_a);
          builder.create_local_store(result, new_result);
        }
        auto new_a = builder.create_mul(current_a, current_a);
        builder.create_local_store(a, new_a);
        auto new_b = builder.create_sar(current_b, one_rhs);
        builder.create_local_store(b, new_b);
      }
      if (is_real(lhs_prim_type)) {
        auto if_stmt = builder.create_if(builder.create_cmp_le(rhs, zero_rhs));
        {
          auto _ = builder.get_if_guard(if_stmt, true);
          auto current_result = builder.create_local_load(result);
          auto new_result = builder.create_div(one_lhs, current_result);
          builder.create_local_store(result, new_result);
        }
      }
      auto final_result = builder.create_local_load(result);
      stmt->replace_usages_with(final_result);
      modifier.insert_before(
          stmt, VecStatement(std::move(builder.extract_ir()->statements)));
      modifier.erase(stmt);
    }
  }

  static bool run(IRNode *node, const CompileConfig &config) {
    DemoteOperations demoter;
    bool modified = false;
    while (true) {
      node->accept(&demoter);
      if (demoter.modifier.modify_ir())
        modified = true;
      else
        break;
      irpass::type_check(node, config);
    }
    if (modified) {
      irpass::type_check(node, config);
    }
    return modified;
  }
};

namespace irpass {

bool demote_operations(IRNode *root, const CompileConfig &config) {
  TI_AUTO_PROF;
  bool modified = DemoteOperations::run(root, config);
  return modified;
}

}  // namespace irpass

}  // namespace taichi::lang
