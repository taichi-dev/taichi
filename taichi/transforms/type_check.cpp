// Type checking

#include "taichi/ir/ir.h"
#include "taichi/ir/transforms.h"
#include "taichi/ir/analysis.h"
#include "taichi/ir/visitors.h"
#include "taichi/program/kernel.h"
#include "taichi/ir/frontend.h"

TLANG_NAMESPACE_BEGIN

// "Type" here does not include vector width
// Var lookup and Type inference
class TypeCheck : public IRVisitor {
 private:
  Kernel *kernel;
  CompileConfig config;

 public:
  TypeCheck(Kernel *kernel) : kernel(kernel) {
    // TODO: remove dependency on get_current_program here
    if (current_program != nullptr)
      config = get_current_program().config;
    allow_undefined_visitor = true;
  }

  static void mark_as_if_const(Stmt *stmt, VectorType t) {
    if (stmt->is<ConstStmt>()) {
      stmt->ret_type = t;
    }
  }

  void visit(AllocaStmt *stmt) {
    // Do nothing. Alloca type is determined by the first LocalStore in IR
    // visiting order, at compile time.

    // ret_type stands for its element type.
    stmt->ret_type.set_is_pointer(false);
  }

  void visit(IfStmt *if_stmt) {
    // TODO: use DataType::u1 when it's supported
    TI_ASSERT(if_stmt->cond->ret_type.data_type == DataType::i32);
    if (if_stmt->true_statements)
      if_stmt->true_statements->accept(this);
    if (if_stmt->false_statements) {
      if_stmt->false_statements->accept(this);
    }
  }

  void visit(Block *stmt_list) {
    std::vector<Stmt *> stmts;
    // Make a copy since type casts may be inserted for type promotion.
    for (auto &stmt : stmt_list->statements) {
      stmts.push_back(stmt.get());
    }
    for (auto stmt : stmts)
      stmt->accept(this);
  }

  void visit(AtomicOpStmt *stmt) {
    TI_ASSERT(stmt->width() == 1);
    if (stmt->val->ret_type.data_type != stmt->dest->ret_type.data_type) {
      TI_WARN("[{}] Atomic add ({} to {}) may lose precision.", stmt->name(),
              data_type_name(stmt->val->ret_type.data_type),
              data_type_name(stmt->dest->ret_type.data_type));
      stmt->val = insert_type_cast_before(stmt, stmt->val,
                                          stmt->dest->ret_type.data_type);
    }
    if (stmt->element_type() == DataType::unknown) {
      stmt->ret_type = stmt->dest->ret_type;
    }
    stmt->ret_type.set_is_pointer(false);
  }

  void visit(LocalLoadStmt *stmt) {
    TI_ASSERT(stmt->width() == 1);
    auto lookup = stmt->ptr[0].var->ret_type;
    stmt->ret_type = lookup;
    stmt->ret_type.set_is_pointer(false);
  }

  void visit(LocalStoreStmt *stmt) {
    if (stmt->ptr->ret_type.data_type == DataType::unknown) {
      // Infer data type for alloca
      stmt->ptr->ret_type = stmt->data->ret_type;
    }
    auto common_container_type = promoted_type(stmt->ptr->ret_type.data_type,
                                               stmt->data->ret_type.data_type);
    if (stmt->ptr->ret_type.data_type != stmt->data->ret_type.data_type) {
      stmt->data = insert_type_cast_before(stmt, stmt->data,
                                           stmt->ptr->ret_type.data_type);
    }
    if (stmt->ptr->ret_type.data_type != common_container_type) {
      TI_WARN(
          "[{}] Local store may lose precision (target = {}, value = {}, at",
          stmt->name(), stmt->ptr->ret_data_type_name(),
          stmt->data->ret_data_type_name(), stmt->id);
      fmt::print(stmt->tb);
    }
    stmt->ret_type = stmt->ptr->ret_type;
    stmt->ret_type.set_is_pointer(false);
  }

  void visit(GlobalLoadStmt *stmt) {
    stmt->ret_type = stmt->ptr->ret_type;
    stmt->ret_type.set_is_pointer(false);
  }

  void visit(SNodeOpStmt *stmt) {
    stmt->ret_type = VectorType(1, DataType::i32);
  }

  void visit(GlobalPtrStmt *stmt) {
    stmt->ret_type.set_is_pointer(true);
    if (stmt->snodes)
      stmt->ret_type.data_type = stmt->snodes[0]->dt;
    else
      TI_WARN("[{}] Type inference failed: snode is nullptr.", stmt->name());
    for (int l = 0; l < stmt->snodes.size(); l++) {
      if (stmt->snodes[l]->parent->num_active_indices != 0 &&
          stmt->snodes[l]->parent->num_active_indices != stmt->indices.size()) {
        TI_ERROR("[{}] {} has {} indices. Indexed with {}.", stmt->name(),
                 stmt->snodes[l]->parent->node_type_name,
                 stmt->snodes[l]->parent->num_active_indices,
                 stmt->indices.size());
      }
    }
    for (int i = 0; i < stmt->indices.size(); i++) {
      TI_ASSERT_INFO(
          is_integral(stmt->indices[i]->ret_type.data_type),
          "[{}] Taichi tensors must be accessed with integral indices (e.g., "
          "i32/i64). It seems that you have used a float point number as "
          "an index. You can cast that to an integer using int(). Also note "
          "that ti.floor(ti.f32) returns f32.",
          stmt->name());
      TI_ASSERT(stmt->indices[i]->ret_type.width == stmt->snodes.size());
    }
  }

  void visit(GlobalStoreStmt *stmt) {
    auto promoted = promoted_type(stmt->ptr->ret_type.data_type,
                                  stmt->data->ret_type.data_type);
    auto input_type = stmt->data->ret_data_type_name();
    if (stmt->ptr->ret_type.data_type != stmt->data->ret_type.data_type) {
      stmt->data = insert_type_cast_before(stmt, stmt->data,
                                           stmt->ptr->ret_type.data_type);
    }
    if (stmt->ptr->ret_type.data_type != promoted) {
      TI_WARN("[{}] Global store may lose precision: {} <- {}, at",
              stmt->name(), stmt->ptr->ret_data_type_name(), input_type,
              stmt->tb);
    }
    stmt->ret_type = stmt->ptr->ret_type;
  }

  void visit(RangeForStmt *stmt) {
    mark_as_if_const(stmt->begin, VectorType(1, DataType::i32));
    mark_as_if_const(stmt->end, VectorType(1, DataType::i32));
    stmt->body->accept(this);
  }

  void visit(StructForStmt *stmt) {
    stmt->body->accept(this);
  }

  void visit(WhileStmt *stmt) {
    stmt->body->accept(this);
  }

  void visit(UnaryOpStmt *stmt) {
    stmt->ret_type = stmt->operand->ret_type;
    if (stmt->is_cast()) {
      stmt->ret_type.data_type = stmt->cast_type;
    }
    if (is_trigonometric(stmt->op_type) &&
        !is_real(stmt->operand->ret_type.data_type)) {
      TI_ERROR("[{}] Trigonometric operator takes real inputs only. At {}",
               stmt->name(), stmt->tb);
    }
    if ((stmt->op_type == UnaryOpType::floor ||
         stmt->op_type == UnaryOpType::ceil) &&
        !is_real(stmt->operand->ret_type.data_type)) {
      TI_ERROR("[{}] floor/ceil takes real inputs only. At {}", stmt->name(),
               stmt->tb);
    }
  }

  Stmt *insert_type_cast_before(Stmt *anchor,
                                Stmt *input,
                                DataType output_type) {
    auto &&cast_stmt =
        Stmt::make_typed<UnaryOpStmt>(UnaryOpType::cast_value, input);
    cast_stmt->cast_type = output_type;
    cast_stmt->accept(this);
    auto stmt = cast_stmt.get();
    anchor->insert_before_me(std::move(cast_stmt));
    return stmt;
  }

  Stmt *insert_type_cast_after(Stmt *anchor,
                               Stmt *input,
                               DataType output_type) {
    auto &&cast_stmt =
        Stmt::make_typed<UnaryOpStmt>(UnaryOpType::cast_value, input);
    cast_stmt->cast_type = output_type;
    cast_stmt->accept(this);
    auto stmt = cast_stmt.get();
    anchor->insert_after_me(std::move(cast_stmt));
    return stmt;
  }

  void cast(Stmt *&val, DataType dt) {
    auto cast_stmt = insert_type_cast_after(val, val, dt);
    val = cast_stmt;
  }

  void visit(BinaryOpStmt *stmt) {
    auto error = [&](std::string comment = "") {
      if (comment == "") {
        TI_WARN(
            "[{}] Error: type mismatch (left = {}, right = {}, stmt_id = {}) "
            "at",
            stmt->name(), stmt->lhs->ret_data_type_name(),
            stmt->rhs->ret_data_type_name(), stmt->id);
      } else {
        TI_WARN("[{}] {} at", stmt->name(), comment);
      }
      fmt::print(stmt->tb);
      TI_WARN("Compilation stopped due to type mismatch.");
      throw std::runtime_error("Binary operator type mismatch");
    };
    if (stmt->lhs->ret_type.data_type == DataType::unknown &&
        stmt->rhs->ret_type.data_type == DataType::unknown)
      error();

    // lower truediv into div

    if (stmt->op_type == BinaryOpType::truediv) {
      auto default_fp = config.default_fp;
      if (!is_real(stmt->lhs->ret_type.data_type)) {
        cast(stmt->lhs, default_fp);
      }
      if (!is_real(stmt->rhs->ret_type.data_type)) {
        cast(stmt->rhs, default_fp);
      }
      stmt->op_type = BinaryOpType::div;
    }

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
    if (binary_is_bitwise(stmt->op_type)) {
      if (!is_integral(stmt->lhs->ret_type.data_type)) {
        error("Error: bitwise operations can only apply to integral types.");
      }
    }
    if (is_comparison(stmt->op_type)) {
      stmt->ret_type = VectorType(stmt->lhs->ret_type.width, DataType::i32);
    } else {
      stmt->ret_type = stmt->lhs->ret_type;
    }
  }

  void visit(TernaryOpStmt *stmt) {
    if (stmt->op_type == TernaryOpType::select) {
      auto ret_type = promoted_type(stmt->op2->ret_type.data_type,
                                    stmt->op3->ret_type.data_type);
      TI_ASSERT(stmt->op1->ret_type.data_type == DataType::i32)
      TI_ASSERT(stmt->op1->ret_type.width == stmt->op2->ret_type.width);
      TI_ASSERT(stmt->op2->ret_type.width == stmt->op3->ret_type.width);
      if (ret_type != stmt->op2->ret_type.data_type) {
        auto cast_stmt = insert_type_cast_before(stmt, stmt->op2, ret_type);
        stmt->op2 = cast_stmt;
      }
      if (ret_type != stmt->op3->ret_type.data_type) {
        auto cast_stmt = insert_type_cast_before(stmt, stmt->op3, ret_type);
        stmt->op3 = cast_stmt;
      }
      stmt->ret_type = VectorType(stmt->op1->width(), ret_type);
    } else {
      TI_NOT_IMPLEMENTED
    }
  }

  void visit(ElementShuffleStmt *stmt) {
    TI_ASSERT(stmt->elements.size() != 0);
    stmt->element_type() = stmt->elements[0].stmt->element_type();
  }

  void visit(RangeAssumptionStmt *stmt) {
    TI_ASSERT(stmt->input->ret_type == stmt->base->ret_type);
    stmt->ret_type = stmt->input->ret_type;
  }

  void visit(ArgLoadStmt *stmt) {
    Kernel *current_kernel = kernel;
    if (current_kernel == nullptr) {
      current_kernel = &get_current_program().get_current_kernel();
    }
    auto &args = current_kernel->args;
    TI_ASSERT(0 <= stmt->arg_id && stmt->arg_id < args.size());
    stmt->ret_type = VectorType(1, args[stmt->arg_id].dt);
  }

  void visit(KernelReturnStmt *stmt) {
    Kernel *current_kernel = kernel;
    if (current_kernel == nullptr) {
      current_kernel = &get_current_program().get_current_kernel();
    }
    auto &rets = current_kernel->rets;
    TI_ASSERT(rets.size() >= 1);
    auto ret = rets[0];  // TODO: stmt->ret_id?
    auto ret_type = ret.dt;
    TI_ASSERT(stmt->value->ret_type.data_type == ret_type);
    stmt->ret_type = VectorType(1, ret_type);
  }

  void visit(ExternalPtrStmt *stmt) {
    stmt->ret_type.set_is_pointer(true);
    stmt->ret_type = VectorType(stmt->base_ptrs.size(),
                                stmt->base_ptrs[0]->ret_type.data_type);
  }

  void visit(LoopIndexStmt *stmt) {
    stmt->ret_type = VectorType(1, DataType::i32);
  }

  void visit(GetRootStmt *stmt) {
    stmt->ret_type = VectorType(1, DataType::gen, true);
  }

  void visit(SNodeLookupStmt *stmt) {
    stmt->ret_type = VectorType(1, DataType::gen, true);
  }

  void visit(GetChStmt *stmt) {
    stmt->ret_type = VectorType(1, stmt->output_snode->dt);
    stmt->ret_type.set_is_pointer(true);
  }

  void visit(OffloadedStmt *stmt) {
    if (stmt->body)
      stmt->body->accept(this);
  }

  void visit(OffsetAndExtractBitsStmt *stmt) {
    stmt->ret_type.data_type = DataType::i32;
  }

  void visit(LinearizeStmt *stmt) {
    stmt->ret_type.data_type = DataType::i32;
  }

  void visit(IntegerOffsetStmt *stmt) {
    stmt->ret_type.data_type = DataType::i32;
  }

  void visit(StackAllocaStmt *stmt) {
    stmt->ret_type.data_type = stmt->dt;
    // ret_type stands for its element type.
    stmt->ret_type.set_is_pointer(false);
  }

  void visit(StackLoadTopStmt *stmt) {
    stmt->ret_type = stmt->stack->ret_type;
    stmt->ret_type.set_is_pointer(false);
  }

  void visit(StackLoadTopAdjStmt *stmt) {
    stmt->ret_type = stmt->stack->ret_type;
    stmt->ret_type.set_is_pointer(false);
  }

  void visit(StackPushStmt *stmt) {
    stmt->ret_type = stmt->stack->ret_type;
    stmt->ret_type.set_is_pointer(false);
    TI_ASSERT(stmt->ret_type == stmt->v->ret_type);
  }

  void visit(StackPopStmt *stmt) {
    stmt->ret_type = stmt->stack->ret_type;
    stmt->ret_type.set_is_pointer(false);
  }

  void visit(StackAccAdjointStmt *stmt) {
    stmt->ret_type = stmt->stack->ret_type;
    stmt->ret_type.set_is_pointer(false);
    TI_ASSERT(stmt->ret_type == stmt->v->ret_type);
  }

  void visit(GlobalTemporaryStmt *stmt) {
    stmt->ret_type.set_is_pointer(true);
  }
};

namespace irpass {

void typecheck(IRNode *root, Kernel *kernel) {
  analysis::check_fields_registered(root);
  TypeCheck inst(kernel);
  root->accept(&inst);
}

}  // namespace irpass

TLANG_NAMESPACE_END
