#include "../ir.h"
#include <deque>
#include <set>

TLANG_NAMESPACE_BEGIN

// Visits all non-containing statements
class BasicStmtVisitor : public IRVisitor {
 public:
  StructForStmt *current_struct_for;

  BasicStmtVisitor() {
    current_struct_for = nullptr;
    allow_undefined_visitor = true;
  }

  void visit(Block *stmt_list) override {
    auto backup_block = current_block;
    current_block = stmt_list;
    for (auto &stmt : stmt_list->statements) {
      stmt->accept(this);
    }
    current_block = backup_block;
  }

  void visit(IfStmt *if_stmt) override {
    if (if_stmt->true_statements)
      if_stmt->true_statements->accept(this);
    if (if_stmt->false_statements) {
      if_stmt->false_statements->accept(this);
    }
  }

  void visit(WhileStmt *stmt) override {
    stmt->body->accept(this);
  }

  void visit(RangeForStmt *for_stmt) override {
    for_stmt->body->accept(this);
  }

  void visit(StructForStmt *for_stmt) override {
    current_struct_for = for_stmt;
    for_stmt->body->accept(this);
    current_struct_for = nullptr;
  }
};

class ConstantFold : public BasicStmtVisitor {
 public:
  ConstantFold() : BasicStmtVisitor() {
  }

  void visit(UnaryOpStmt *stmt) {
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
      }

      if (success) {
        auto evaluated = Stmt::make<ConstStmt>(LaneAttribute<TypedConstant>(new_constant));
        stmt->replace_with(evaluated.get());
        stmt->parent->insert_before(stmt, VecStatement(std::move(evaluated)));
        stmt->parent->erase(stmt);
        throw IRModified();
      }
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
