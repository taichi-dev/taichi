// Type checking

#include "taichi/ir/ir.h"
#include "taichi/ir/statements.h"
#include "taichi/ir/transforms.h"
#include "taichi/ir/analysis.h"
#include "taichi/ir/frontend_ir.h"

TLANG_NAMESPACE_BEGIN

static_assert(
    sizeof(real) == sizeof(float32),
    "Please build the taichi compiler with single precision (TI_USE_DOUBLE=0)");

// "Type" here does not include vector width
// Var lookup and Type inference
class TypeCheck : public IRVisitor {
 private:
  CompileConfig config_;

 public:
  explicit TypeCheck(const CompileConfig &config) : config_(config) {
    allow_undefined_visitor = true;
  }

  static void mark_as_if_const(Stmt *stmt, DataType t) {
    if (stmt->is<ConstStmt>()) {
      stmt->ret_type = t;
    }
  }

  void visit(AllocaStmt *stmt) override {
    // Do nothing. Alloca type is determined by the first LocalStore in IR
    // visiting order, at compile time.

    // ret_type stands for its element type.
  }

  void visit(IfStmt *if_stmt) override {
    if (if_stmt->true_statements)
      if_stmt->true_statements->accept(this);
    if (if_stmt->false_statements) {
      if_stmt->false_statements->accept(this);
    }
  }

  void visit(Block *stmt_list) override {
    std::vector<Stmt *> stmts;
    // Make a copy since type casts may be inserted for type promotion.
    for (auto &stmt : stmt_list->statements) {
      stmts.push_back(stmt.get());
    }
    for (auto stmt : stmts)
      stmt->accept(this);
  }

  void visit(AtomicOpStmt *stmt) override {
    TI_ASSERT(stmt->width() == 1);
    // TODO(type): test_ad_for fails if we assume dest is a pointer type.
    auto dst_type = stmt->dest->ret_type.ptr_removed();
    if (auto cit = dst_type->cast<CustomIntType>()) {
      dst_type = cit->get_physical_type();
    } else if (auto cft = dst_type->cast<CustomFloatType>()) {
      auto cit = cft->get_digits_type()->as<CustomIntType>();
      dst_type = cit->get_physical_type();
    } else if (stmt->val->ret_type != dst_type) {
      TI_WARN("[{}] Atomic {} ({} to {}) may lose precision, at\n{}",
              stmt->name(), atomic_op_type_name(stmt->op_type),
              data_type_name(stmt->val->ret_type), data_type_name(dst_type),
              stmt->tb);
      stmt->val = insert_type_cast_before(stmt, stmt->val, dst_type);
    }
    stmt->ret_type = dst_type;
  }

  void visit(LocalLoadStmt *stmt) override {
    TI_ASSERT(stmt->width() == 1);
    TI_ASSERT_INFO(stmt->src.size() == 1, "Vectorization has been disabled.");
    TI_ASSERT(stmt->src[0].var->is<AllocaStmt>() ||
              stmt->src[0].var->is<PtrOffsetStmt>());
    if (auto ptr_offset_stmt = stmt->src[0].var->cast<PtrOffsetStmt>()) {
      TI_ASSERT(ptr_offset_stmt->origin->is<AllocaStmt>() ||
                ptr_offset_stmt->origin->is<GlobalTemporaryStmt>());
      if (auto alloca_stmt = ptr_offset_stmt->origin->cast<AllocaStmt>()) {
        auto lookup =
            DataType(
                alloca_stmt->ret_type->as<TensorType>()->get_element_type())
                .ptr_removed();
        stmt->ret_type = lookup;
      }
      if (auto global_temporary_stmt =
              ptr_offset_stmt->origin->cast<GlobalTemporaryStmt>()) {
        auto lookup = DataType(global_temporary_stmt->ret_type->as<TensorType>()
                                   ->get_element_type())
                          .ptr_removed();
        stmt->ret_type = lookup;
      }
    } else {
      auto lookup = stmt->src[0].var->ret_type;
      stmt->ret_type = lookup;
    }
  }

  void visit(LocalStoreStmt *stmt) override {
    if (stmt->dest->is<PtrOffsetStmt>() &&
        stmt->dest->cast<PtrOffsetStmt>()->is_local_ptr()) {
      auto dst_value_type = stmt->dest->ret_type.ptr_removed();
      if (dst_value_type->is<CustomIntType>() ||
          dst_value_type->is<CustomFloatType>()) {
        // We force the value type to be the compute_type of the bit pointer.
        // Casting from compute_type to physical_type is handled in codegen.
        dst_value_type = dst_value_type->get_compute_type();
      }
      auto promoted = promoted_type(dst_value_type, stmt->val->ret_type);
      auto input_type = stmt->val->ret_data_type_name();
      if (dst_value_type != stmt->val->ret_type) {
        stmt->val = insert_type_cast_before(stmt, stmt->val, dst_value_type);
      }
      if (dst_value_type != promoted && dst_value_type != stmt->val->ret_type) {
        TI_WARN("[{}] Local store may lose precision: {} <- {}, at\n{}",
                stmt->name(), dst_value_type->to_string(), input_type,
                stmt->tb);
      }
      stmt->ret_type = dst_value_type;
      return;
    }

    if (stmt->dest->ret_type->is_primitive(PrimitiveTypeID::unknown)) {
      // Infer data type for alloca
      stmt->dest->ret_type = stmt->val->ret_type;
    }
    auto common_container_type =
        promoted_type(stmt->dest->ret_type, stmt->val->ret_type);

    auto old_data = stmt->val;
    if (stmt->dest->ret_type != stmt->val->ret_type) {
      stmt->val =
          insert_type_cast_before(stmt, stmt->val, stmt->dest->ret_type);
    }
    if (stmt->dest->ret_type != common_container_type) {
      TI_WARN(
          "[{}] Local store may lose precision (target = {}, value = {}), "
          "at\n{}",
          stmt->name(), stmt->dest->ret_data_type_name(),
          old_data->ret_data_type_name(), stmt->id, stmt->tb);
    }
    stmt->ret_type = stmt->dest->ret_type;
  }

  void visit(GlobalLoadStmt *stmt) override {
    auto pointee_type = stmt->src->ret_type.ptr_removed();
    if (auto bit_struct = pointee_type->cast<BitStructType>()) {
      stmt->ret_type = bit_struct->get_physical_type();
    } else {
      stmt->ret_type = pointee_type->get_compute_type();
    }
  }

  void visit(SNodeOpStmt *stmt) override {
    if (stmt->op_type == SNodeOpType::get_addr) {
      stmt->ret_type =
          TypeFactory::create_vector_or_scalar_type(1, PrimitiveType::u64);
    } else {
      stmt->ret_type =
          TypeFactory::create_vector_or_scalar_type(1, PrimitiveType::i32);
    }
  }

  void visit(ExternalTensorShapeAlongAxisStmt *stmt) override {
    stmt->ret_type =
        TypeFactory::create_vector_or_scalar_type(1, PrimitiveType::i32);
  }

  void visit(GlobalPtrStmt *stmt) override {
    if (stmt->is_bit_vectorized) {
      return;
    }
    stmt->ret_type.set_is_pointer(true);
    if (stmt->snodes) {
      stmt->ret_type =
          TypeFactory::get_instance().get_pointer_type(stmt->snodes[0]->dt);
    } else
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
      if (!is_integral(stmt->indices[i]->ret_type)) {
        TI_WARN(
            "[{}] Field index {} not integral, casting into int32 implicitly",
            stmt->name(), i);
        stmt->indices[i] =
            insert_type_cast_before(stmt, stmt->indices[i], PrimitiveType::i32);
      }
      TI_ASSERT(stmt->indices[i]->width() == stmt->snodes.size());
    }
  }

  void visit(PtrOffsetStmt *stmt) override {
    TI_ASSERT(stmt->offset->ret_type->is_primitive(PrimitiveTypeID::i32));
    stmt->ret_type.set_is_pointer(true);
  }

  void visit(GlobalStoreStmt *stmt) override {
    auto dst_value_type = stmt->dest->ret_type.ptr_removed();
    if (dst_value_type->is<CustomIntType>() ||
        dst_value_type->is<CustomFloatType>()) {
      // We force the value type to be the compute_type of the bit pointer.
      // Casting from compute_type to physical_type is handled in codegen.
      dst_value_type = dst_value_type->get_compute_type();
    }
    auto promoted = promoted_type(dst_value_type, stmt->val->ret_type);
    auto input_type = stmt->val->ret_data_type_name();
    if (dst_value_type != stmt->val->ret_type) {
      stmt->val = insert_type_cast_before(stmt, stmt->val, dst_value_type);
    }
    // TODO: do not use "promoted" here since u8 + u8 = i32 in C++ and storing
    // u8 to u8 leads to extra warnings.
    if (dst_value_type != promoted && dst_value_type != stmt->val->ret_type) {
      TI_WARN("[{}] Global store may lose precision: {} <- {}, at\n{}",
              stmt->name(), dst_value_type->to_string(), input_type, stmt->tb);
    }
  }

  void visit(RangeForStmt *stmt) override {
    mark_as_if_const(stmt->begin, TypeFactory::create_vector_or_scalar_type(
                                      1, PrimitiveType::i32));
    mark_as_if_const(stmt->end, TypeFactory::create_vector_or_scalar_type(
                                    1, PrimitiveType::i32));
    stmt->body->accept(this);
  }

  void visit(StructForStmt *stmt) override {
    stmt->body->accept(this);
  }

  void visit(MeshForStmt *stmt) override {
    stmt->body->accept(this);
  }

  void visit(WhileStmt *stmt) override {
    stmt->body->accept(this);
  }

  void visit(UnaryOpStmt *stmt) override {
    stmt->ret_type = stmt->operand->ret_type;
    if (stmt->is_cast()) {
      stmt->ret_type = stmt->cast_type;
    }
    if (!is_real(stmt->operand->ret_type)) {
      if (stmt->op_type == UnaryOpType::sqrt ||
          stmt->op_type == UnaryOpType::exp ||
          stmt->op_type == UnaryOpType::log) {
        cast(stmt->operand, config_.default_fp);
        stmt->ret_type = config_.default_fp;
      }
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

  void visit(BinaryOpStmt *stmt) override {
    auto error = [&](std::string comment = "") {
      if (comment == "") {
        TI_WARN(
            "[{}] Error: type mismatch (left = {}, right = {}, stmt_id = {}), "
            "at\n{}",
            stmt->name(), stmt->lhs->ret_data_type_name(),
            stmt->rhs->ret_data_type_name(), stmt->id, stmt->tb);
      } else {
        TI_WARN("[{}] {} at\n{}", stmt->name(), comment, stmt->tb);
      }
      TI_WARN("Compilation stopped due to type mismatch.");
      throw std::runtime_error("Binary operator type mismatch");
    };
    if (stmt->lhs->ret_type->is_primitive(PrimitiveTypeID::unknown) &&
        stmt->rhs->ret_type->is_primitive(PrimitiveTypeID::unknown))
      error();

    // lower truediv into div

    if (stmt->op_type == BinaryOpType::truediv) {
      auto default_fp = config_.default_fp;
      if (!is_real(stmt->lhs->ret_type)) {
        cast(stmt->lhs, default_fp);
      }
      if (!is_real(stmt->rhs->ret_type)) {
        cast(stmt->rhs, default_fp);
      }
      stmt->op_type = BinaryOpType::div;
    }

    if (stmt->lhs->ret_type != stmt->rhs->ret_type) {
      auto promote_custom_int_type = [&](Stmt *stmt, Stmt *hs) {
        if (auto cit = hs->ret_type->cast<CustomIntType>()) {
          return insert_type_cast_before(stmt, hs, cit->get_compute_type());
        }
        return hs;
      };
      stmt->lhs = promote_custom_int_type(stmt, stmt->lhs);
      stmt->rhs = promote_custom_int_type(stmt, stmt->rhs);
      auto ret_type = promoted_type(stmt->lhs->ret_type, stmt->rhs->ret_type);
      if (ret_type != stmt->lhs->ret_type) {
        // promote lhs
        auto cast_stmt = insert_type_cast_before(stmt, stmt->lhs, ret_type);
        stmt->lhs = cast_stmt;
      }
      if (ret_type != stmt->rhs->ret_type) {
        // promote rhs
        auto cast_stmt = insert_type_cast_before(stmt, stmt->rhs, ret_type);
        stmt->rhs = cast_stmt;
      }
    }
    bool matching = true;
    matching = matching && (stmt->lhs->width() == stmt->rhs->width());
    matching = matching && (stmt->lhs->ret_type != PrimitiveType::unknown);
    matching = matching && (stmt->rhs->ret_type != PrimitiveType::unknown);
    matching = matching && (stmt->lhs->ret_type == stmt->rhs->ret_type);
    if (!matching) {
      error();
    }
    if (is_comparison(stmt->op_type)) {
      stmt->ret_type = TypeFactory::create_vector_or_scalar_type(
          stmt->lhs->width(), PrimitiveType::i32);
    } else {
      stmt->ret_type = stmt->lhs->ret_type;
    }
  }

  void visit(TernaryOpStmt *stmt) override {
    if (stmt->op_type == TernaryOpType::select) {
      auto ret_type = promoted_type(stmt->op2->ret_type, stmt->op3->ret_type);
      TI_ASSERT(stmt->op1->ret_type->is_primitive(PrimitiveTypeID::i32))
      TI_ASSERT(stmt->op1->width() == stmt->op2->width());
      TI_ASSERT(stmt->op2->width() == stmt->op3->width());
      if (ret_type != stmt->op2->ret_type) {
        auto cast_stmt = insert_type_cast_before(stmt, stmt->op2, ret_type);
        stmt->op2 = cast_stmt;
      }
      if (ret_type != stmt->op3->ret_type) {
        auto cast_stmt = insert_type_cast_before(stmt, stmt->op3, ret_type);
        stmt->op3 = cast_stmt;
      }
      stmt->ret_type = TypeFactory::create_vector_or_scalar_type(
          stmt->op1->width(), ret_type);
    } else {
      TI_NOT_IMPLEMENTED
    }
  }

  void visit(ElementShuffleStmt *stmt) override {
    TI_ASSERT(stmt->elements.size() != 0);
    stmt->element_type() = stmt->elements[0].stmt->element_type();
  }

  void visit(RangeAssumptionStmt *stmt) override {
    stmt->ret_type = stmt->input->ret_type;
  }

  void visit(LoopUniqueStmt *stmt) override {
    stmt->ret_type = stmt->input->ret_type;
  }

  void visit(FuncCallStmt *stmt) override {
    auto *func = stmt->func;
    TI_ASSERT(func);
    TI_ASSERT(func->rets.size() <= 1);
    if (func->rets.size() == 1) {
      stmt->ret_type = func->rets[0].dt;
    }
  }

  void visit(ArgLoadStmt *stmt) override {
    // TODO: Maybe have a type_inference() pass, which takes in the args/rets
    // defined by the kernel. After that, type_check() pass will purely do
    // verification, without modifying any types.
    TI_ASSERT(stmt->width() == 1);
    stmt->ret_type.set_is_pointer(stmt->is_ptr);
  }

  void visit(ReturnStmt *stmt) override {
    // TODO: Support stmt->ret_id?
    TI_ASSERT(stmt->width() == 1);
  }

  void visit(ExternalPtrStmt *stmt) override {
    stmt->ret_type.set_is_pointer(true);
    stmt->ret_type = TypeFactory::create_vector_or_scalar_type(
        stmt->base_ptrs.size(), stmt->base_ptrs[0]->ret_type);
  }

  void visit(LoopIndexStmt *stmt) override {
    stmt->ret_type =
        TypeFactory::create_vector_or_scalar_type(1, PrimitiveType::i32);
  }

  void visit(LoopLinearIndexStmt *stmt) override {
    stmt->ret_type =
        TypeFactory::create_vector_or_scalar_type(1, PrimitiveType::i32);
  }

  void visit(BlockCornerIndexStmt *stmt) override {
    stmt->ret_type =
        TypeFactory::create_vector_or_scalar_type(1, PrimitiveType::i32);
  }

  void visit(GetRootStmt *stmt) override {
    stmt->ret_type =
        TypeFactory::create_vector_or_scalar_type(1, PrimitiveType::gen, true);
  }

  void visit(SNodeLookupStmt *stmt) override {
    if (stmt->snode->type == SNodeType::bit_array) {
      auto bit_array_type = stmt->snode->dt;
      auto element_type =
          bit_array_type->cast<BitArrayType>()->get_element_type();
      auto pointer_type =
          TypeFactory::get_instance().get_pointer_type(element_type, true);
      stmt->ret_type = pointer_type;
    } else {
      stmt->ret_type = TypeFactory::create_vector_or_scalar_type(
          1, PrimitiveType::gen, true);
    }
  }

  void visit(GetChStmt *stmt) override {
    if (stmt->is_bit_vectorized) {
      auto physical_type = stmt->output_snode->physical_type;
      auto ptr_ret_type =
          TypeFactory::get_instance().get_pointer_type(physical_type);
      stmt->ret_type = DataType(ptr_ret_type);
      return;
    }
    TI_ASSERT(stmt->width() == 1);
    auto element_type = stmt->output_snode->dt;
    // For bit_struct SNodes, their component SNodes must have
    // is_bit_level=true
    auto pointer_type = TypeFactory::get_instance().get_pointer_type(
        element_type, stmt->output_snode->is_bit_level);
    stmt->ret_type = pointer_type;
  }

  void visit(OffloadedStmt *stmt) override {
    stmt->all_blocks_accept(this);
  }

  void visit(BitExtractStmt *stmt) override {
    stmt->ret_type = stmt->input->ret_type;
  }

  void visit(LinearizeStmt *stmt) override {
    stmt->ret_type = PrimitiveType::i32;
  }

  void visit(IntegerOffsetStmt *stmt) override {
    stmt->ret_type = PrimitiveType::i32;
  }

  void visit(AdStackAllocaStmt *stmt) override {
    stmt->ret_type = stmt->dt;
    // ret_type stands for its element type.
    stmt->ret_type.set_is_pointer(false);
  }

  void visit(AdStackLoadTopStmt *stmt) override {
    stmt->ret_type = stmt->stack->ret_type;
    stmt->ret_type.set_is_pointer(false);
  }

  void visit(AdStackLoadTopAdjStmt *stmt) override {
    stmt->ret_type = stmt->stack->ret_type;
    stmt->ret_type.set_is_pointer(false);
  }

  void visit(AdStackPushStmt *stmt) override {
    stmt->ret_type = stmt->stack->ret_type;
    stmt->ret_type.set_is_pointer(false);
    TI_ASSERT(stmt->ret_type == stmt->v->ret_type);
  }

  void visit(AdStackPopStmt *stmt) override {
    stmt->ret_type = stmt->stack->ret_type;
    stmt->ret_type.set_is_pointer(false);
  }

  void visit(AdStackAccAdjointStmt *stmt) override {
    stmt->ret_type = stmt->stack->ret_type;
    stmt->ret_type.set_is_pointer(false);
    TI_ASSERT(stmt->ret_type == stmt->v->ret_type);
  }

  void visit(GlobalTemporaryStmt *stmt) override {
    if (!stmt->ret_type->is<TensorType>())
      stmt->ret_type.set_is_pointer(true);
  }

  void visit(InternalFuncStmt *stmt) override {
    // TODO: support return type specification
    stmt->ret_type =
        TypeFactory::create_vector_or_scalar_type(1, PrimitiveType::i32);
  }

  void visit(BitStructStoreStmt *stmt) override {
    // do nothing
  }
};

namespace irpass {

void type_check(IRNode *root, const CompileConfig &config) {
  TI_AUTO_PROF;
  analysis::check_fields_registered(root);
  TypeCheck inst(config);
  root->accept(&inst);
}

}  // namespace irpass

TLANG_NAMESPACE_END
