#include "../ir.h"

TLANG_NAMESPACE_BEGIN

// "Type" here does not include vector width
// Var lookup and Type inference
class TypeCheck : public IRVisitor {
 public:
  TypeCheck() {
    allow_undefined_visitor = true;
  }

  static void mark_as_if_const(Stmt *stmt, VectorType t) {
    if (stmt->is<ConstStmt>()) {
      stmt->ret_type = t;
    }
  }

  void visit(AllocaStmt *stmt) {
    // Do nothing.
    // Alloca type is determined by first (compile-time) LocalStore
  }

  void visit(IfStmt *if_stmt) {
    if (if_stmt->true_statements)
      if_stmt->true_statements->accept(this);
    if (if_stmt->false_statements) {
      if_stmt->false_statements->accept(this);
    }
  }

  void visit(Block *stmt_list) {
    std::vector<Stmt *> stmts;
    // Make a copy since type casts may be inserted for type promotion
    for (auto &stmt : stmt_list->statements) {
      stmts.push_back(stmt.get());
      // stmt->accept(this);
    }
    for (auto stmt : stmts)
      stmt->accept(this);
  }

  void visit(AtomicOpStmt *stmt) {
    TC_ASSERT(stmt->width() == 1);
    TC_ASSERT(stmt->dest->ret_type == stmt->val->ret_type);
  }

  void visit(LocalLoadStmt *stmt) {
    TC_ASSERT(stmt->width() == 1);
    auto lookup = stmt->ptr[0].var->ret_type;
    stmt->ret_type = lookup;
  }

  void visit(LocalStoreStmt *stmt) {
    if (stmt->ptr->ret_type.data_type == DataType::unknown) {
      // Infer data type for alloca
      stmt->ptr->ret_type = stmt->data->ret_type;
    }
    auto ret_type = promoted_type(stmt->ptr->ret_type.data_type,
                                  stmt->data->ret_type.data_type);
    if (ret_type != stmt->data->ret_type.data_type) {
      stmt->data = insert_type_cast_before(stmt, stmt->data, ret_type);
    }
    if (stmt->ptr->ret_type != stmt->data->ret_type) {
      TC_WARN(
          "Error: type mismatch in local store (target = {}, value = {}, "
          "stmt_id = {}) at",
          stmt->ptr->ret_data_type_name(), stmt->data->ret_data_type_name(),
          stmt->id);
      fmt::print(stmt->tb);
      TC_WARN("Compilation stopped due to type mismatch.");
      exit(-1);
    }
    stmt->ret_type = stmt->ptr->ret_type;
  }

  void visit(GlobalLoadStmt *stmt) {
    stmt->ret_type = stmt->ptr->ret_type;
  }

  void visit(SNodeOpStmt *stmt) {
    stmt->ret_type = VectorType(1, DataType::i32);
  }

  void visit(GlobalPtrStmt *stmt) {
    if (stmt->snodes)
      stmt->ret_type.data_type = stmt->snodes[0]->dt;
    else
      TC_WARN("Type inference failed: snode is nullptr.");
    for (int l = 0; l < stmt->snodes.size(); l++) {
      if (stmt->snodes[l]->num_active_indices != stmt->indices.size()) {
        TC_ERROR("{} has {} indices. Indexed with {}.",
                 stmt->snodes[l]->node_type_name,
                 stmt->snodes[l]->num_active_indices, stmt->indices.size());
      }
    }
  }

  void visit(GlobalStoreStmt *stmt) {
    auto ret_type = promoted_type(stmt->ptr->ret_type.data_type,
                                  stmt->data->ret_type.data_type);
    if (ret_type != stmt->data->ret_type.data_type) {
      stmt->data = insert_type_cast_before(stmt, stmt->data, ret_type);
    }
    if (stmt->ptr->ret_type != stmt->data->ret_type) {
      TC_ERROR("Global store type mismatch: {} <- {}",
               stmt->ptr->ret_data_type_name(),
               stmt->data->ret_data_type_name());
    }
    stmt->ret_type = stmt->ptr->ret_type;
  }

  void visit(RangeForStmt *stmt) {
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

  void visit(StructForStmt *stmt) {
    stmt->body->accept(this);
  }

  void visit(WhileStmt *stmt) {
    stmt->body->accept(this);
  }

  void visit(UnaryOpStmt *stmt) {
    stmt->ret_type = stmt->rhs->ret_type;
    if (stmt->op_type == UnaryOpType::cast) {
      stmt->ret_type.data_type = stmt->cast_type;
    }
    if (is_trigonometric(stmt->op_type) &&
        !is_real(stmt->rhs->ret_type.data_type)) {
      TC_ERROR("Trigonometric operator takes real inputs only.");
    }
  }

  Stmt *insert_type_cast_before(Stmt *anchor,
                                Stmt *input,
                                DataType output_type) {
    auto &&cast_stmt = Stmt::make_typed<UnaryOpStmt>(UnaryOpType::cast, input);
    cast_stmt->cast_type = output_type;
    cast_stmt->cast_by_value = true;
    cast_stmt->accept(this);
    auto stmt = cast_stmt.get();
    anchor->insert_before_me(std::move(cast_stmt));
    return stmt;
  }

  void visit(BinaryOpStmt *stmt) {
    auto error = [&] {
      TC_WARN("Error: type mismatch (left = {}, right = {}, stmt_id = {}) at",
              stmt->lhs->ret_data_type_name(), stmt->rhs->ret_data_type_name(),
              stmt->id);
      fmt::print(stmt->tb);
      TC_WARN("Compilation stopped due to type mismatch.");
      exit(-1);
    };
    if (!(stmt->lhs->ret_type.data_type != DataType::unknown ||
          stmt->rhs->ret_type.data_type != DataType::unknown))
      error();
    if (stmt->lhs->ret_type.data_type != stmt->rhs->ret_type.data_type) {
      auto ret_type = promoted_type(stmt->lhs->ret_type.data_type,
                                    stmt->rhs->ret_type.data_type);
      if (ret_type != stmt->lhs->ret_type.data_type) {
        // promote rhs
        auto cast_stmt = insert_type_cast_before(stmt, stmt->lhs, ret_type);
        stmt->lhs = cast_stmt;
      }
      if (ret_type != stmt->rhs->ret_type.data_type) {
        // promote rhs
        auto cast_stmt = insert_type_cast_before(stmt, stmt->rhs, ret_type);
        stmt->rhs = cast_stmt;
      }
    }
    bool matching = true;
    matching =
        matching && (stmt->lhs->ret_type.width == stmt->rhs->ret_type.width);
    matching = matching && (stmt->lhs->ret_type.data_type != DataType::unknown);
    matching = matching && (stmt->rhs->ret_type.data_type != DataType::unknown);
    matching = matching && (stmt->lhs->ret_type == stmt->rhs->ret_type);
    if (!matching) {
      error();
    }
    if (is_comparison(stmt->op_type)) {
      stmt->ret_type = VectorType(stmt->lhs->ret_type.width, DataType::i32);
    } else {
      stmt->ret_type = stmt->lhs->ret_type;
    }
  }

  void visit(TernaryOpStmt *stmt) {
    if (stmt->op_type == TernaryOpType::select) {
      TC_ASSERT(stmt->op2->ret_type == stmt->op3->ret_type);
      TC_ASSERT(stmt->op1->ret_type.data_type == DataType::i32)
      TC_ASSERT(stmt->op1->ret_type.width == stmt->op2->ret_type.width);
      stmt->ret_type = stmt->op2->ret_type;
    } else {
      TC_NOT_IMPLEMENTED
    }
  }

  void visit(ElementShuffleStmt *stmt) {
    TC_ASSERT(stmt->elements.size() != 0);
    stmt->element_type() = stmt->elements[0].stmt->element_type();
  }

  void visit(RangeAssumptionStmt *stmt) {
    TC_ASSERT(stmt->input->ret_type == stmt->base->ret_type);
    stmt->ret_type = stmt->input->ret_type;
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
