#include "../ir.h"
#include <deque>
#include <set>

TLANG_NAMESPACE_BEGIN

class ConstantFold : public BasicStmtVisitor {
 public:
  using BasicStmtVisitor::visit;

  ConstantFold() : BasicStmtVisitor() {
  }

  void visit(UnaryOpStmt *stmt) override {
    if (stmt->width() == 1 && stmt->op_type == UnaryOpType::cast &&
        stmt->cast_by_value && stmt->operand->is<ConstStmt>()) {
      auto input = stmt->operand->as<ConstStmt>()->val[0];
      auto src_type = stmt->operand->ret_type.data_type;
      auto dst_type = stmt->ret_type.data_type;
      TypedConstant new_constant(dst_type);
      bool success = false;
      if (src_type == DataType::f32) {
        auto v = input.val_float32();
        if (dst_type == DataType::i32) {
          new_constant.val_i32 = int32(v);
          success = true;
        }
      } else if (src_type == DataType::i32) {
        auto v = input.val_int32();
        if (dst_type == DataType::f32) {
          new_constant.val_f32 = float32(v);
          success = true;
        }
      }

      if (success) {
        auto evaluated =
            Stmt::make<ConstStmt>(LaneAttribute<TypedConstant>(new_constant));
        stmt->replace_with(evaluated.get());
        stmt->parent->insert_before(stmt, VecStatement(std::move(evaluated)));
        stmt->parent->erase(stmt);
        throw IRModified();
      }
    }
  }

  void visit(BinaryOpStmt *stmt) override {
    auto lhs = stmt->lhs->cast<ConstStmt>();
    auto rhs = stmt->rhs->cast<ConstStmt>();
    if (!lhs || !rhs)
      return;
    if (stmt->width() != 1 || stmt->ret_type.data_type != DataType::i32) {
      return;
    }
    auto dst_type = DataType::i32;
    TypedConstant new_constant(dst_type);
    if (stmt->op_type == BinaryOpType::sub) {
      new_constant.val_int32() =
          lhs->val[0].val_int32() - rhs->val[0].val_int32();
      auto evaluated =
          Stmt::make<ConstStmt>(LaneAttribute<TypedConstant>(new_constant));
      stmt->replace_with(evaluated.get());
      stmt->parent->insert_before(stmt, VecStatement(std::move(evaluated)));
      stmt->parent->erase(stmt);
      throw IRModified();
    } else if (stmt->op_type == BinaryOpType::mul) {
      new_constant.val_int32() =
          lhs->val[0].val_int32() * rhs->val[0].val_int32();
      auto evaluated =
          Stmt::make<ConstStmt>(LaneAttribute<TypedConstant>(new_constant));
      stmt->replace_with(evaluated.get());
      stmt->parent->insert_before(stmt, VecStatement(std::move(evaluated)));
      stmt->parent->erase(stmt);
      throw IRModified();
    }
  }

  static void run(IRNode *node) {
    ConstantFold folder;
    while (true) {
      bool modified = false;
      try {
        node->accept(&folder);
      } catch (IRModified) {
        modified = true;
      }
      if (!modified)
        break;
    }
  }
};

namespace irpass {

void constant_fold(IRNode *root) {
  return ConstantFold::run(root);
}

}  // namespace irpass

TLANG_NAMESPACE_END
