#include "../ir.h"

TLANG_NAMESPACE_BEGIN

// Algebraic Simplification
class AlgSimp : public BasicStmtVisitor {
 public:
  using BasicStmtVisitor::visit;

  AlgSimp() : BasicStmtVisitor() {
  }

  void visit(BinaryOpStmt *stmt) override {
    auto lhs = stmt->lhs->cast<ConstStmt>();
    auto rhs = stmt->rhs->cast<ConstStmt>();
    if (!lhs && !rhs)
      return;
    if (stmt->width() != 1) {
      return;
    }
    if (stmt->op_type == BinaryOpType::add || stmt->op_type == BinaryOpType::sub) {
      if (alg_is_zero(rhs)) {
        stmt->replace_with(stmt->lhs);
      } else if (stmt->op_type == BinaryOpType::add && alg_is_zero(lhs)) {
        stmt->replace_with(stmt->rhs);
      }
    } else if (stmt->op_type == BinaryOpType::mul || stmt->op_type == BinaryOpType::div) {
      if (alg_is_one(rhs)) {
        stmt->replace_with(stmt->lhs);
      } else if (stmt->op_type == BinaryOpType::mul && alg_is_one(lhs)) {
        stmt->replace_with(stmt->rhs);
      }
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
      TC_NOT_IMPLEMENTED
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
      TC_NOT_IMPLEMENTED
      return false;
    }
  }

  static void run(IRNode *node) {
    AlgSimp simplifier;
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

void alg_simp(IRNode *root) {
  return AlgSimp::run(root);
}

}  // namespace irpass

TLANG_NAMESPACE_END
