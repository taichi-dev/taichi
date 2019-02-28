#include "ir.h"

TLANG_NAMESPACE_BEGIN

// "Type" here does not include vector width
// Variable lookup and Type inference
class TypeCheck : public IRVisitor {
 public:
  TypeCheck() {
    allow_undefined_visitor = true;
  }

  static void mark_as_if_const(Statement *stmt, VectorType t) {
    if (stmt->is<ConstStmt>()) {
      stmt->ret_type = t;
    }
  }

  void visit(AllocaStmt *stmt) {
  }

  void visit(TmpValStmt *stmt) {
    stmt->ret_type = stmt->val->ret_type;
  }

  void visit(IfStmt *if_stmt) {
    if (if_stmt->true_statements)
      if_stmt->true_statements->accept(this);
    if (if_stmt->false_statements) {
      if_stmt->false_statements->accept(this);
    }
  }

  void visit(Block *stmt_list) {
    for (auto &stmt : stmt_list->statements) {
      stmt->accept(this);
    }
  }

  void visit(LocalLoadStmt *stmt) {
    auto block = stmt->parent;
    auto lookup = stmt->ident->ret_type;
    stmt->ret_type = lookup;
  }

  void visit(LocalStoreStmt *stmt) {
    // auto block = stmt->parent;
    // auto lookup = block->local_var_alloca.find(stmt->ident);
    if (stmt->ident->ret_type.data_type == DataType::unknown) {
      stmt->ident->ret_type = stmt->stmt->ret_type;
    }
    TC_ASSERT(stmt->ident->ret_type == stmt->stmt->ret_type);
  }


  void visit(GlobalLoadStmt *stmt) {
    stmt->ret_type = stmt->ptr->ret_type;
  }

  void visit(GlobalPtrStmt *stmt) {
    if (stmt->snode)
      stmt->ret_type.data_type = stmt->snode->dt;
    else
      TC_WARN("Type inference failed: snode is nullptr.");
  }

  void visit(GlobalStoreStmt *stmt) {
    if (stmt->ptr->ret_type != stmt->data->ret_type) {
      TC_ERROR("Global store type mismatch: {} <- {}",
               stmt->ptr->ret_data_type_name(),
               stmt->data->ret_data_type_name());
    }
  }

  void visit(RangeForStmt *stmt) {
    auto block = stmt->parent;
    /*
    TC_ASSERT(block->local_variables.find(stmt->loop_var) ==
              block->local_variables.end());
              */
    mark_as_if_const(stmt->begin, VectorType(1, DataType::i32));
    mark_as_if_const(stmt->end, VectorType(1, DataType::i32));
    /*
    block->local_variables.insert(
        std::make_pair(stmt->loop_var, VectorType(1, DataType::i32)));
    */
    stmt->body->accept(this);
  }

  void visit(WhileStmt *stmt) {
    stmt->body->accept(this);
  }

  void visit(UnaryOpStmt *stmt) {
    stmt->ret_type = stmt->rhs->ret_type;
    if (stmt->op_type == UnaryType::cast) {
      stmt->ret_type.data_type = stmt->cast_type;
    }
  }

  void visit(BinaryOpStmt *stmt) {
    TC_ASSERT(stmt->lhs->ret_type.data_type != DataType::unknown ||
              stmt->rhs->ret_type.data_type != DataType::unknown);
    if (stmt->lhs->ret_type.data_type == DataType::unknown &&
        stmt->lhs->is<ConstStmt>()) {
      stmt->lhs->ret_type = stmt->rhs->ret_type;
    }
    if (stmt->rhs->ret_type.data_type == DataType::unknown &&
        stmt->rhs->is<ConstStmt>()) {
      stmt->rhs->ret_type = stmt->lhs->ret_type;
    }
    TC_ASSERT(stmt->lhs->ret_type.width == stmt->rhs->ret_type.width);
    TC_ASSERT(stmt->lhs->ret_type.data_type != DataType::unknown);
    TC_ASSERT(stmt->rhs->ret_type.data_type != DataType::unknown);
    TC_ASSERT(stmt->lhs->ret_type == stmt->rhs->ret_type);
    if (is_comparison(stmt->op_type)) {
      stmt->ret_type = VectorType(stmt->lhs->ret_type.width, DataType::i32);
    } else {
      stmt->ret_type = stmt->lhs->ret_type;
    }
  }

  static void run(IRNode *node) {
    TypeCheck inst;
    node->accept(&inst);
  }
};

namespace irpass {

void typecheck(IRNode *root) {
  return TypeCheck::run(root);
}

}  // namespace irpass

TLANG_NAMESPACE_END
