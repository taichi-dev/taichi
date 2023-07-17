// Type checking

#include "taichi/ir/ir.h"
#include "taichi/ir/statements.h"
#include "taichi/ir/transforms.h"
#include "taichi/ir/analysis.h"
#include "taichi/ir/frontend_ir.h"
#include "taichi/transforms/utils.h"

namespace taichi::lang {

static_assert(
    sizeof(real) == sizeof(float32),
    "Please build the taichi compiler with single precision (TI_USE_DOUBLE=0)");

// Var lookup and Type inference
class TypeCheck : public IRVisitor {
 private:
  CompileConfig config_;

  Type *type_check_store(Stmt *stmt, Stmt *dst, Stmt *&val) {
    auto dst_type = dst->ret_type.ptr_removed();
    auto val_type = val->ret_type.ptr_removed();
    if (is_quant(dst_type)) {
      // We force the value type to be the compute_type of the bit pointer.
      // Casting from compute_type to physical_type is handled in codegen.
      dst_type = dst_type->get_compute_type();
    }
    if (dst_type != val_type) {
      val = insert_type_cast_before(stmt, val, dst_type);
    }
    return dst_type;
  }

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
    // TODO(type): test_ad_for fails if we assume dest is a pointer type.

    stmt->ret_type = type_check_store(stmt, stmt->dest, stmt->val);
  }

  void visit(LocalLoadStmt *stmt) override {
    TI_ASSERT(stmt->src->is<AllocaStmt>() || stmt->src->is<MatrixPtrStmt>() ||
              stmt->src->is<MatrixOfMatrixPtrStmt>());
    if (auto ptr_offset_stmt = stmt->src->cast<MatrixPtrStmt>()) {
      auto lookup = DataType(ptr_offset_stmt->origin->ret_type.ptr_removed()
                                 ->as<TensorType>()
                                 ->get_element_type())
                        .ptr_removed();
      stmt->ret_type = lookup;
    } else {
      auto lookup = stmt->src->ret_type;
      stmt->ret_type = lookup.ptr_removed();
    }
  }

  void visit(LocalStoreStmt *stmt) override {
    if (stmt->dest->ret_type->is_primitive(PrimitiveTypeID::unknown)) {
      // Infer data type for alloca
      stmt->dest->ret_type = stmt->val->ret_type;
    }
    stmt->ret_type = type_check_store(stmt, stmt->dest, stmt->val);
  }

  void visit(GlobalLoadStmt *stmt) override {
    auto pointee_type = stmt->src->ret_type.ptr_removed();
    stmt->ret_type = pointee_type->get_compute_type();
  }

  void visit(SNodeOpStmt *stmt) override {
    if (stmt->op_type == SNodeOpType::get_addr) {
      stmt->ret_type = PrimitiveType::u64;
    } else if (stmt->op_type == SNodeOpType::allocate) {
      stmt->ret_type = PrimitiveType::gen;
      stmt->ret_type.set_is_pointer(true);
    } else if (stmt->op_type == SNodeOpType::is_active) {
      stmt->ret_type = PrimitiveType::u1;
    } else {
      stmt->ret_type = PrimitiveType::i32;
    }
  }

  void visit(ExternalTensorShapeAlongAxisStmt *stmt) override {
    stmt->ret_type = PrimitiveType::i32;
  }

  void visit(GlobalPtrStmt *stmt) override {
    if (stmt->is_bit_vectorized) {
      return;
    }
    stmt->ret_type.set_is_pointer(true);

    if (!stmt->ret_type.ptr_removed().get_element_type()->is_primitive(
            PrimitiveTypeID::unknown)) {
      // pass
    } else if (stmt->snode) {
      stmt->ret_type =
          TypeFactory::get_instance().get_pointer_type(stmt->snode->dt);
    } else
      ErrorEmitter(TaichiTypeWarning(), stmt,
                   "Type inference failed: snode is nullptr.");
    auto check_indices = [&](SNode *snode) {
      if (snode->num_active_indices != stmt->indices.size()) {
        ErrorEmitter(
            TaichiRuntimeError(), stmt,
            fmt::format("{} has {} indices. Indexed with {}.",
                        snode->node_type_name, snode->num_active_indices,
                        stmt->indices.size()));
      }
    };
    check_indices(stmt->is_cell_access ? stmt->snode : stmt->snode->parent);
    for (int i = 0; i < stmt->indices.size(); i++) {
      if (!stmt->indices[i]->ret_type->is_primitive(PrimitiveTypeID::i32)) {
        ErrorEmitter(
            TaichiCastWarning(), stmt,
            fmt::format(
                "Field index {} not int32, casting into int32 implicitly", i));
        stmt->indices[i] =
            insert_type_cast_before(stmt, stmt->indices[i], PrimitiveType::i32);
      }
    }
  }

  void visit(MatrixPtrStmt *stmt) override {
    TI_ASSERT(stmt->offset->ret_type.get_element_type()->is_primitive(
        PrimitiveTypeID::i32));
    stmt->ret_type.set_is_pointer(true);
  }

  void visit(GlobalStoreStmt *stmt) override {
    type_check_store(stmt, stmt->dest, stmt->val);
  }

  void visit(RangeForStmt *stmt) override {
    mark_as_if_const(stmt->begin, PrimitiveType::i32);
    mark_as_if_const(stmt->end, PrimitiveType::i32);
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
    if (stmt->op_type == UnaryOpType::frexp) {
      // frexp returns a struct type of {double, i32} and it's set in frontend
      // IR lowering.
      return;
    }
    auto operand_type = stmt->operand->ret_type;
    stmt->ret_type = operand_type;
    if (stmt->is_cast()) {
      stmt->ret_type = stmt->cast_type;
      if (operand_type->is<TensorType>() &&
          stmt->cast_type->is<PrimitiveType>()) {
        auto ret_tensor_type = operand_type->as<TensorType>();
        auto tensor_shape = ret_tensor_type->get_shape();
        stmt->ret_type = TypeFactory::get_instance().create_tensor_type(
            tensor_shape, stmt->cast_type);
      }
    }

    DataType primitive_dtype = stmt->operand->ret_type.get_element_type();
    if (!is_real(primitive_dtype)) {
      if (stmt->op_type == UnaryOpType::sqrt ||
          stmt->op_type == UnaryOpType::exp ||
          stmt->op_type == UnaryOpType::log) {
        DataType target_dtype = config_.default_fp;
        if (stmt->operand->ret_type->is<TensorType>()) {
          target_dtype = TypeFactory::get_instance().create_tensor_type(
              stmt->operand->ret_type->as<TensorType>()->get_shape(),
              target_dtype);
        }

        cast(stmt->operand, target_dtype);
        stmt->ret_type = target_dtype;
      } else if (stmt->op_type == UnaryOpType::logic_not) {
        DataType target_dtype = PrimitiveType::u1;
        if (stmt->operand->ret_type->is<TensorType>()) {
          target_dtype = TypeFactory::get_instance().create_tensor_type(
              stmt->operand->ret_type->as<TensorType>()->get_shape(),
              target_dtype);
        }

        cast(stmt->operand, target_dtype);
        stmt->ret_type = target_dtype;
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

  void insert_shift_op_assertion_before(Stmt *stmt, Stmt *lhs, Stmt *rhs) {
    int rhs_limit = data_type_bits(lhs->ret_type);
    auto const_stmt =
        Stmt::make<ConstStmt>(TypedConstant(rhs->ret_type, rhs_limit));
    auto cond_stmt =
        Stmt::make<BinaryOpStmt>(BinaryOpType::cmp_le, rhs, const_stmt.get());

    std::string msg =
        "Detected overflow for bit_shift_op with rhs = %d, exceeding limit of "
        "%d.";
    msg += "\n" + stmt->get_tb();
    std::vector<Stmt *> args = {rhs, const_stmt.get()};
    auto assert_stmt =
        Stmt::make<AssertStmt>(cond_stmt.get(), msg, std::move(args));

    const_stmt->accept(this);
    cond_stmt->accept(this);
    assert_stmt->accept(this);

    stmt->insert_before_me(std::move(const_stmt));
    stmt->insert_before_me(std::move(cond_stmt));
    stmt->insert_before_me(std::move(assert_stmt));
  }

  void cast(Stmt *&val, DataType dt) {
    if (val->ret_type == dt)
      return;

    auto cast_stmt = insert_type_cast_after(val, val, dt);
    val = cast_stmt;
  }

  void visit(BinaryOpStmt *stmt) override {
    auto error = [&]() {
      ErrorEmitter(
          TaichiTypeError(), stmt,
          fmt::format("Type mismatch (left = {}, right = {}, stmt_id = {})",
                      stmt->lhs->ret_data_type_name(),
                      stmt->rhs->ret_data_type_name(), stmt->id));
    };
    if (stmt->lhs->ret_type->is_primitive(PrimitiveTypeID::unknown) &&
        stmt->rhs->ret_type->is_primitive(PrimitiveTypeID::unknown))
      error();
    if (stmt->op_type == BinaryOpType::pow &&
        (is_integral(stmt->rhs->ret_type.get_element_type()))) {
      stmt->ret_type = stmt->lhs->ret_type;
      return;
    }

    auto make_dt = [stmt](DataType dt) {
      if (auto tensor_ty = stmt->lhs->ret_type->cast<TensorType>()) {
        return TypeFactory::create_tensor_type(tensor_ty->get_shape(), dt);
      } else {
        return dt;
      }
    };

    // lower truediv into div

    if (stmt->op_type == BinaryOpType::truediv) {
      auto default_fp = config_.default_fp;
      if (!is_real(stmt->lhs->ret_type.get_element_type())) {
        cast(stmt->lhs, make_dt(default_fp));
      }
      if (!is_real(stmt->rhs->ret_type.get_element_type())) {
        cast(stmt->rhs, make_dt(default_fp));
      }
      stmt->op_type = BinaryOpType::div;
    }

    // Some backends such as vulkan doesn't support fp64
    // Always promote to fp32 unless necessary
    if (stmt->op_type == BinaryOpType::atan2) {
      if (stmt->rhs->ret_type == PrimitiveType::f64 ||
          stmt->lhs->ret_type == PrimitiveType::f64) {
        stmt->ret_type = make_dt(PrimitiveType::f64);
        cast(stmt->rhs, make_dt(PrimitiveType::f64));
        cast(stmt->lhs, make_dt(PrimitiveType::f64));
      } else {
        stmt->ret_type = make_dt(PrimitiveType::f32);
        cast(stmt->rhs, make_dt(PrimitiveType::f32));
        cast(stmt->lhs, make_dt(PrimitiveType::f32));
      }
    }

    if (stmt->lhs->ret_type != stmt->rhs->ret_type) {
      DataType ret_type;
      if (is_shift_op(stmt->op_type)) {
        // shift_ops does not follow the same type promotion rule as numerical
        // ops numerical ops: u8 + i32 = i32 shift_ops:     u8 << i32 = u8
        // (return dtype follows that of the lhs)
        //
        // In the above example, while truncating rhs(i32) to u8 risks an
        // overflow, the runtime value of rhs is very likely less than 8
        // (otherwise meaningless). Nevertheless, we insert an AssertStmt here
        // to warn user of this potential overflow.
        ret_type = stmt->lhs->ret_type;

        // Insert AssertStmt
        if (config_.debug) {
          insert_shift_op_assertion_before(stmt, stmt->lhs, stmt->rhs);
        }
      } else {
        ret_type = promoted_type(stmt->lhs->ret_type, stmt->rhs->ret_type);
      }

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
    matching = matching && (stmt->lhs->ret_type != PrimitiveType::unknown);
    matching = matching && (stmt->rhs->ret_type != PrimitiveType::unknown);
    matching = matching && (stmt->lhs->ret_type == stmt->rhs->ret_type);
    if (!matching) {
      error();
    }
    if (is_comparison(stmt->op_type)) {
      stmt->ret_type = make_dt(PrimitiveType::u1);
    } else {
      stmt->ret_type = stmt->lhs->ret_type;
    }
  }

  void visit(TernaryOpStmt *stmt) override {
    if (stmt->op_type == TernaryOpType::select) {
      auto ret_type = promoted_type(stmt->op2->ret_type, stmt->op3->ret_type);
      TI_ASSERT(is_integral(stmt->op1->ret_type.get_element_type()));
      if (ret_type != stmt->op2->ret_type) {
        auto cast_stmt = insert_type_cast_before(stmt, stmt->op2, ret_type);
        stmt->op2 = cast_stmt;
      }
      if (ret_type != stmt->op3->ret_type) {
        auto cast_stmt = insert_type_cast_before(stmt, stmt->op3, ret_type);
        stmt->op3 = cast_stmt;
      }
      stmt->ret_type = ret_type;
    } else {
      TI_NOT_IMPLEMENTED
    }
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
    stmt->ret_type = func->ret_type;
    stmt->ret_type.set_is_pointer(true);
  }

  void visit(FrontendFuncCallStmt *stmt) override {
    auto *func = stmt->func;
    TI_ASSERT(func);
    stmt->ret_type = func->ret_type;
    stmt->ret_type.set_is_pointer(true);
  }

  void visit(GetElementStmt *stmt) override {
    TI_ASSERT(stmt->src->ret_type->is<PointerType>());
    stmt->ret_type =
        stmt->src->ret_type.ptr_removed()->as<StructType>()->get_element_type(
            stmt->index);
  }

  void visit(ArgLoadStmt *stmt) override {
    // Do nothing
  }

  void visit(ReturnStmt *stmt) override {
    // TODO: Support stmt->ret_id?
  }

  void visit(ExternalPtrStmt *stmt) override {
    TI_ASSERT(stmt->base_ptr->is<ArgLoadStmt>());
    auto arg_load_stmt = stmt->base_ptr->cast<ArgLoadStmt>();

    if (stmt->overrided_dtype) {
      // pass
    } else {
      stmt->ret_type = arg_load_stmt->ret_type.ptr_removed()
                           ->as<StructType>()
                           ->get_element_type({1});
    }

    stmt->ret_type.set_is_pointer(true);
    for (int i = 0; i < stmt->indices.size(); i++) {
      TI_ASSERT(is_integral(stmt->indices[i]->ret_type));
      if (stmt->indices[i]->ret_type != PrimitiveType::i32) {
        stmt->indices[i] =
            insert_type_cast_before(stmt, stmt->indices[i], PrimitiveType::i32);
      }
    }
  }

  void visit(LoopIndexStmt *stmt) override {
    stmt->ret_type = PrimitiveType::i32;
  }

  void visit(LoopLinearIndexStmt *stmt) override {
    stmt->ret_type = PrimitiveType::i32;
  }

  void visit(BlockCornerIndexStmt *stmt) override {
    stmt->ret_type = PrimitiveType::i32;
  }

  void visit(GetRootStmt *stmt) override {
    stmt->ret_type =
        TypeFactory::get_instance().get_pointer_type(PrimitiveType::gen);
  }

  void visit(SNodeLookupStmt *stmt) override {
    if (stmt->snode->type == SNodeType::quant_array) {
      auto quant_array_type = stmt->snode->dt;
      auto element_type =
          quant_array_type->cast<QuantArrayType>()->get_element_type();
      auto pointer_type =
          TypeFactory::get_instance().get_pointer_type(element_type, true);
      stmt->ret_type = pointer_type;
    } else {
      stmt->ret_type =
          TypeFactory::get_instance().get_pointer_type(PrimitiveType::gen);
    }
  }

  void visit(GetChStmt *stmt) override {
    if (stmt->overrided_dtype)
      return;

    if (stmt->is_bit_vectorized) {
      auto physical_type = stmt->output_snode->physical_type;
      auto ptr_ret_type =
          TypeFactory::get_instance().get_pointer_type(physical_type);
      stmt->ret_type = DataType(ptr_ret_type);
      return;
    }
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
    stmt->ret_type.set_is_pointer(true);
  }

  void visit(InternalFuncStmt *stmt) override {
    // TODO: support return type specification
    stmt->ret_type = PrimitiveType::i32;
  }

  void visit(BitStructStoreStmt *stmt) override {
    // do nothing
  }

  void visit(ReferenceStmt *stmt) override {
    stmt->ret_type = stmt->var->ret_type;
    stmt->ret_type.set_is_pointer(true);
  }

  void visit(MatrixInitStmt *stmt) override {
    TI_ASSERT_INFO(stmt->ret_type->is<TensorType>(),
                   "Matrix should have tensor type, got {}",
                   stmt->ret_type->to_string());
    auto tensor_type = stmt->ret_type->as<TensorType>();
    auto element_dtype = tensor_type->get_element_type();
    for (int i = 0; i < stmt->values.size(); ++i) {
      if (element_dtype != stmt->values[i]->ret_type) {
        cast(stmt->values[i], element_dtype);
      }
    }
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

}  // namespace taichi::lang
