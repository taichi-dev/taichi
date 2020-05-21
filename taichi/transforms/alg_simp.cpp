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
        stmt->parent->erase(stmt);
        throw IRModified();
      } else if (stmt->op_type != BinaryOpType::sub && alg_is_zero(lhs)) {
        // 0 +|^ a -> a
        stmt->replace_with(stmt->rhs);
        stmt->parent->erase(stmt);
        throw IRModified();
      }
    } else if (stmt->op_type == BinaryOpType::mul ||
               stmt->op_type == BinaryOpType::div) {
      if (alg_is_one(rhs)) {
        // a */ 1 -> a
        stmt->replace_with(stmt->lhs);
        stmt->parent->erase(stmt);
        throw IRModified();
      } else if (stmt->op_type == BinaryOpType::mul && alg_is_one(lhs)) {
        // 1 * a -> a
        stmt->replace_with(stmt->rhs);
        stmt->parent->erase(stmt);
        throw IRModified();
      } else if ((fast_math || is_integral(stmt->ret_type.data_type)) &&
                 stmt->op_type == BinaryOpType::mul &&
                 (alg_is_zero(lhs) || alg_is_zero(rhs))) {
        // fast_math or integral operands: 0 * a -> 0, a * 0 -> 0
        auto zero = Stmt::make<ConstStmt>(
            LaneAttribute<TypedConstant>(stmt->ret_type.data_type));
        stmt->replace_with(zero.get());
        stmt->parent->insert_before(stmt, VecStatement(std::move(zero)));
        stmt->parent->erase(stmt);
        throw IRModified();
      }
    } else if (stmt->op_type == BinaryOpType::bit_and) {
      if (alg_is_minus_one(rhs)) {
        // a & -1 -> a
        stmt->replace_with(stmt->lhs);
        stmt->parent->erase(stmt);
        throw IRModified();
      } else if (alg_is_minus_one(lhs)) {
        // -1 & a -> a
        stmt->replace_with(stmt->rhs);
        stmt->parent->erase(stmt);
        throw IRModified();
      }
    }
  }

  void visit(AssertStmt *stmt) override {
    auto cond = stmt->cond->cast<ConstStmt>();
    if (!cond)
      return;
    if (!alg_is_zero(cond)) {
      // this statement has no effect
      stmt->parent->erase(stmt);
      throw IRModified();
    }
  }

  void visit(WhileControlStmt *stmt) override {
    auto cond = stmt->cond->cast<ConstStmt>();
    if (!cond)
      return;
    if (!alg_is_zero(cond)) {
      // this statement has no effect
      stmt->parent->erase(stmt);
      throw IRModified();
    }
  }

  static bool alg_is_zero(ConstStmt *stmt) {
    if (!stmt)
      return false;
    if (stmt->width() != 1)
      return false;
    auto val = stmt->val[0];
    auto data_type = stmt->ret_type.data_type;
    if (data_type == DataType::i32)
      return val.val_int32() == 0;
    else if (data_type == DataType::f32)
      return val.val_float32() == 0;
    else if (data_type == DataType::i64)
      return val.val_int64() == 0;
    else if (data_type == DataType::f64)
      return val.val_float64() == 0;
    else {
      TI_NOT_IMPLEMENTED
      return false;
    }
  }

  static bool alg_is_one(ConstStmt *stmt) {
    if (!stmt)
      return false;
    if (stmt->width() != 1)
      return false;
    auto val = stmt->val[0];
    auto data_type = stmt->ret_type.data_type;
    if (data_type == DataType::i32)
      return val.val_int32() == 1;
    else if (data_type == DataType::f32)
      return val.val_float32() == 1;
    else if (data_type == DataType::i64)
      return val.val_int64() == 1;
    else if (data_type == DataType::f64)
      return val.val_float64() == 1;
    else {
      TI_NOT_IMPLEMENTED
      return false;
    }
  }

  static bool alg_is_minus_one(ConstStmt *stmt) {
    if (!stmt)
      return false;
    if (stmt->width() != 1)
      return false;
    auto val = stmt->val[0];
    auto data_type = stmt->ret_type.data_type;
    if (data_type == DataType::i32)
      return val.val_int32() == -1;
    else if (data_type == DataType::f32)
      return val.val_float32() == -1;
    else if (data_type == DataType::i64)
      return val.val_int64() == -1;
    else if (data_type == DataType::f64)
      return val.val_float64() == -1;
    else {
      TI_NOT_IMPLEMENTED
      return false;
    }
  }

  static void run(IRNode *node, bool fast_math) {
    AlgSimp simplifier(fast_math);
    while (true) {
      bool modified = false;
      try {
        node->accept(&simplifier);
      } catch (IRModified) {
        modified = true;
      }
      if (!modified)
        break;
    }
  }
};

namespace irpass {

void alg_simp(IRNode *root, const CompileConfig &config) {
  return AlgSimp::run(root, config.fast_math);
}

}  // namespace irpass

TLANG_NAMESPACE_END
