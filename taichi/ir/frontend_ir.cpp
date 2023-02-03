#include "taichi/ir/frontend_ir.h"

#include "taichi/ir/expression_printer.h"
#include "taichi/ir/statements.h"
#include "taichi/program/program.h"
#include "taichi/common/exceptions.h"

#include <numeric>

namespace taichi::lang {

#define TI_ASSERT_TYPE_CHECKED(x)                       \
  TI_ASSERT_INFO(x->ret_type != PrimitiveType::unknown, \
                 "[{}] was not type-checked",           \
                 ExpressionHumanFriendlyPrinter::expr_to_string(x))

static bool is_primitive_or_tensor_type(DataType &type) {
  return type->is<PrimitiveType>() || type->is<TensorType>();
}

FrontendSNodeOpStmt::FrontendSNodeOpStmt(SNodeOpType op_type,
                                         SNode *snode,
                                         const ExprGroup &indices,
                                         const Expr &val)
    : op_type(op_type), snode(snode), indices(indices), val(val) {
  if (val.expr != nullptr) {
    TI_ASSERT(op_type == SNodeOpType::append);
  } else {
    TI_ASSERT(op_type != SNodeOpType::append);
  }
}

FrontendReturnStmt::FrontendReturnStmt(const ExprGroup &group) : values(group) {
}

FrontendAssignStmt::FrontendAssignStmt(const Expr &lhs, const Expr &rhs)
    : lhs(lhs), rhs(rhs) {
  TI_ASSERT(lhs->is_lvalue());
  if (lhs.is<IdExpression>() && lhs->ret_type == PrimitiveType::unknown) {
    lhs.expr->ret_type = rhs->ret_type;
  }
}

FrontendForStmt::FrontendForStmt(const ExprGroup &loop_vars,
                                 SNode *snode,
                                 Arch arch,
                                 const ForLoopConfig &config)
    : snode(snode) {
  init_config(arch, config);
  init_loop_vars(loop_vars);
}

FrontendForStmt::FrontendForStmt(const ExprGroup &loop_vars,
                                 const Expr &external_tensor,
                                 Arch arch,
                                 const ForLoopConfig &config)
    : external_tensor(external_tensor) {
  init_config(arch, config);
  init_loop_vars(loop_vars);
}

FrontendForStmt::FrontendForStmt(const ExprGroup &loop_vars,
                                 const mesh::MeshPtr &mesh,
                                 const mesh::MeshElementType &element_type,
                                 Arch arch,
                                 const ForLoopConfig &config)
    : mesh(mesh.ptr.get()), element_type(element_type) {
  init_config(arch, config);
  init_loop_vars(loop_vars);
}

FrontendForStmt::FrontendForStmt(const Expr &loop_var,
                                 const Expr &begin,
                                 const Expr &end,
                                 Arch arch,
                                 const ForLoopConfig &config)
    : begin(begin), end(end) {
  init_config(arch, config);
  add_loop_var(loop_var);
}

void FrontendForStmt::init_config(Arch arch, const ForLoopConfig &config) {
  is_bit_vectorized = config.is_bit_vectorized;
  strictly_serialized = config.strictly_serialized;
  mem_access_opt = config.mem_access_opt;
  block_dim = config.block_dim;
  if (arch == Arch::cuda || arch == Arch::amdgpu) {
    num_cpu_threads = 1;
    TI_ASSERT(block_dim <= taichi_max_gpu_block_dim);
  } else {  // cpu
    if (config.num_cpu_threads == 0) {
      num_cpu_threads = std::thread::hardware_concurrency();
    } else {
      num_cpu_threads = config.num_cpu_threads;
    }
  }
}

void FrontendForStmt::init_loop_vars(const ExprGroup &loop_vars) {
  loop_var_ids.reserve(loop_vars.size());
  for (int i = 0; i < (int)loop_vars.size(); i++) {
    add_loop_var(loop_vars[i]);
  }
}

void FrontendForStmt::add_loop_var(const Expr &loop_var) {
  loop_var_ids.push_back(loop_var.cast<IdExpression>()->id);
  loop_var.expr->ret_type = PrimitiveType::i32;
}

void ArgLoadExpression::type_check(const CompileConfig *) {
  TI_ASSERT_INFO(dt->is<PrimitiveType>() && dt != PrimitiveType::unknown,
                 "Invalid dt [{}] for ArgLoadExpression", dt->to_string());
  ret_type = dt;
}

void ArgLoadExpression::flatten(FlattenContext *ctx) {
  auto arg_load = std::make_unique<ArgLoadStmt>(arg_id, dt, is_ptr);
  ctx->push_back(std::move(arg_load));
  stmt = ctx->back_stmt();
}

void TexturePtrExpression::type_check(const CompileConfig *config) {
}

void TexturePtrExpression::flatten(FlattenContext *ctx) {
  ctx->push_back<ArgLoadStmt>(arg_id, PrimitiveType::f32, true);
  ctx->push_back<TexturePtrStmt>(ctx->back_stmt(), num_dims, is_storage,
                                 num_channels, channel_format, lod);
  stmt = ctx->back_stmt();
}

void RandExpression::type_check(const CompileConfig *) {
  TI_ASSERT_INFO(dt->is<PrimitiveType>() && dt != PrimitiveType::unknown,
                 "Invalid dt [{}] for RandExpression", dt->to_string());
  ret_type = dt;
}

void RandExpression::flatten(FlattenContext *ctx) {
  auto ran = std::make_unique<RandStmt>(dt);
  ctx->push_back(std::move(ran));
  stmt = ctx->back_stmt();
}

void UnaryOpExpression::type_check(const CompileConfig *config) {
  TI_ASSERT_TYPE_CHECKED(operand);

  TI_ASSERT(config != nullptr);
  /*
    Dtype inference for both TensorType and PrimitiveType are essentially
    the same. Therefore we extract the primitive type to perform the type
    inference, and then reconstruct the TensorType once neccessary.
  */

  auto operand_primitive_type = operand->ret_type.get_element_type();
  auto ret_primitive_type = ret_type;

  if (!operand_primitive_type->is<PrimitiveType>()) {
    throw TaichiTypeError(fmt::format(
        "unsupported operand type(s) for '{}': '{}'", unary_op_type_name(type),
        operand_primitive_type->to_string()));
  }

  if ((type == UnaryOpType::round || type == UnaryOpType::floor ||
       type == UnaryOpType::ceil || is_trigonometric(type)) &&
      !is_real(operand_primitive_type))
    throw TaichiTypeError(fmt::format(
        "'{}' takes real inputs only, however '{}' is provided",
        unary_op_type_name(type), operand_primitive_type->to_string()));

  if ((type == UnaryOpType::sqrt || type == UnaryOpType::exp ||
       type == UnaryOpType::log) &&
      !is_real(operand_primitive_type)) {
    ret_primitive_type = config->default_fp;
  } else {
    ret_primitive_type = is_cast() ? cast_type : operand_primitive_type;
  }

  if (operand->ret_type->is<TensorType>()) {
    ret_type = taichi::lang::TypeFactory::get_instance().get_tensor_type(
        operand->ret_type.get_shape(), ret_primitive_type);
  } else {
    TI_ASSERT(operand->ret_type->is<PrimitiveType>());
    ret_type = ret_primitive_type;
  }
}

bool UnaryOpExpression::is_cast() const {
  return unary_op_is_cast(type);
}

void UnaryOpExpression::flatten(FlattenContext *ctx) {
  auto operand_stmt = flatten_rvalue(operand, ctx);
  auto unary = std::make_unique<UnaryOpStmt>(type, operand_stmt);
  if (is_cast()) {
    unary->cast_type = cast_type;
  }
  stmt = unary.get();
  stmt->tb = tb;
  stmt->ret_type = ret_type;
  ctx->push_back(std::move(unary));
}

Expr to_broadcast_tensor(const Expr &elt, const DataType &dt) {
  if (!elt->ret_type->is<TensorType>() && !dt->is<TensorType>())
    return elt;

  if (elt->ret_type->is<TensorType>() && dt->is<TensorType>()) {
    // Only tensor shape will be checked here, since the dtype will
    // be promoted later at irpass::type_check()
    if (elt->ret_type.get_shape() != dt.get_shape()) {
      TI_ERROR("Cannot broadcast tensor to tensor");
    } else {
      return elt;
    }
  }

  auto tensor_type = dt->as<TensorType>();
  auto elt_type = tensor_type->get_element_type();
  TI_ASSERT_INFO(elt_type->is<PrimitiveType>(),
                 "Only primitive types are supported in Tensors, got {}",
                 elt_type->to_string());
  std::vector<Expr> broadcast_values(tensor_type->get_num_elements(), elt);
  auto matrix_expr = Expr::make<MatrixExpression>(
      broadcast_values, tensor_type->get_shape(), elt->ret_type);
  matrix_expr->type_check(nullptr);
  return matrix_expr;
}

std::tuple<Expr, Expr> unify_binop_operands(const Expr &e1, const Expr &e2) {
  if (e1->ret_type->is<PrimitiveType>() && e2->ret_type->is<TensorType>()) {
    return std::tuple(to_broadcast_tensor(e1, e2->ret_type), e2);
  } else if (e1->ret_type->is<TensorType>() &&
             e2->ret_type->is<PrimitiveType>()) {
    return std::tuple(e1, to_broadcast_tensor(e2, e1->ret_type));
  } else {
    return std::tuple(e1, e2);
  }
}

void BinaryOpExpression::type_check(const CompileConfig *config) {
  TI_ASSERT_TYPE_CHECKED(lhs);
  TI_ASSERT_TYPE_CHECKED(rhs);
  auto lhs_type = lhs->ret_type;
  auto rhs_type = rhs->ret_type;
  auto error = [&]() {
    throw TaichiTypeError(
        fmt::format("unsupported operand type(s) for '{}': '{}' and '{}'",
                    binary_op_type_symbol(type), lhs->ret_type->to_string(),
                    rhs->ret_type->to_string()));
  };

  if (!is_primitive_or_tensor_type(lhs_type) ||
      !is_primitive_or_tensor_type(rhs_type)) {
    error();
  }

  if ((lhs_type->is<PrimitiveType>() && rhs_type->is<TensorType>()) ||
      (lhs_type->is<TensorType>() && rhs_type->is<PrimitiveType>())) {
    // convert Tensor/Scalar | Scalar/Tensor operations to broadcasting
    auto [unified_l, unified_r] = unify_binop_operands(lhs, rhs);
    lhs = unified_l;
    rhs = unified_r;
    if (lhs->ret_type == PrimitiveType::unknown)
      lhs.type_check(config);
    if (rhs->ret_type == PrimitiveType::unknown)
      rhs.type_check(config);
    TI_ASSERT(lhs->ret_type->is<TensorType>());
    TI_ASSERT(rhs->ret_type->is<TensorType>());
    lhs_type = lhs->ret_type;
    rhs_type = rhs->ret_type;
  }

  bool is_tensor_op = false;

  if (lhs_type->is<TensorType>()) {
    is_tensor_op = true;
    auto rhs_tensor_type = rhs_type->cast<TensorType>();
    if (rhs_tensor_type->get_shape() !=
        lhs_type->cast<TensorType>()->get_shape())
      // current assume element-wise binary op
      error();
  }

  auto make_dt = [&is_tensor_op, this](DataType dt) {
    if (is_tensor_op) {
      return TypeFactory::create_tensor_type(
          this->lhs->ret_type->cast<TensorType>()->get_shape(), dt);
    } else {
      return dt;
    }
  };

  if (binary_is_bitwise(type) && (!is_integral(lhs_type.get_element_type()) ||
                                  !is_integral(rhs_type.get_element_type())))
    error();
  if (binary_is_logical(type) &&
      (is_tensor_op || lhs_type != PrimitiveType::i32 ||
       rhs_type != PrimitiveType::i32))
    error();
  if (is_comparison(type) || binary_is_logical(type)) {
    ret_type = make_dt(PrimitiveType::i32);
    return;
  }
  if (is_shift_op(type) ||
      (type == BinaryOpType::pow && is_integral(rhs_type))) {
    ret_type = lhs_type;
    return;
  }

  // Some backends such as vulkan doesn't support fp64
  // Try not promoting to fp64 unless necessary
  if (type == BinaryOpType::atan2) {
    if (lhs_type == PrimitiveType::f64 || rhs_type == PrimitiveType::f64) {
      ret_type = make_dt(PrimitiveType::f64);
    } else {
      ret_type = make_dt(PrimitiveType::f32);
    }
    return;
  }

  if (type == BinaryOpType::truediv) {
    auto default_fp = config->default_fp;
    if (!is_real(lhs_type.get_element_type())) {
      lhs_type = make_dt(default_fp);
    }
    if (!is_real(rhs_type.get_element_type())) {
      rhs_type = make_dt(default_fp);
    }
  }
  ret_type = promoted_type(lhs_type, rhs_type);
}

void BinaryOpExpression::flatten(FlattenContext *ctx) {
  // if (stmt)
  //  return;
  auto lhs_stmt = flatten_rvalue(lhs, ctx);

  if (binary_is_logical(type)) {
    auto result = ctx->push_back<AllocaStmt>(ret_type);
    ctx->push_back<LocalStoreStmt>(result, lhs_stmt);
    auto cond = ctx->push_back<LocalLoadStmt>(result);
    auto if_stmt = ctx->push_back<IfStmt>(cond);

    FlattenContext rctx;
    rctx.current_block = ctx->current_block;
    auto rhs_stmt = flatten_rvalue(rhs, &rctx);
    rctx.push_back<LocalStoreStmt>(result, rhs_stmt);

    auto true_block = std::make_unique<Block>();
    if (type == BinaryOpType::logical_and) {
      true_block->set_statements(std::move(rctx.stmts));
    }
    if_stmt->set_true_statements(std::move(true_block));

    auto false_block = std::make_unique<Block>();
    if (type == BinaryOpType::logical_or) {
      false_block->set_statements(std::move(rctx.stmts));
    }
    if_stmt->set_false_statements(std::move(false_block));

    auto ret = ctx->push_back<LocalLoadStmt>(result);
    ret->tb = tb;
    stmt = ret;
    stmt->ret_type = ret_type;
    return;
  }
  auto rhs_stmt = flatten_rvalue(rhs, ctx);
  ctx->push_back(std::make_unique<BinaryOpStmt>(type, lhs_stmt, rhs_stmt));
  ctx->stmts.back()->tb = tb;
  stmt = ctx->back_stmt();
  stmt->ret_type = ret_type;
}

void make_ifte(Expression::FlattenContext *ctx,
               DataType ret_type,
               Expr cond,
               Expr true_val,
               Expr false_val) {
  auto result = ctx->push_back<AllocaStmt>(ret_type);
  auto cond_stmt = flatten_rvalue(cond, ctx);
  auto if_stmt = ctx->push_back<IfStmt>(cond_stmt);

  Expression::FlattenContext lctx;
  lctx.current_block = ctx->current_block;
  auto true_val_stmt = flatten_rvalue(true_val, &lctx);
  lctx.push_back<LocalStoreStmt>(result, true_val_stmt);

  Expression::FlattenContext rctx;
  rctx.current_block = ctx->current_block;
  auto false_val_stmt = flatten_rvalue(false_val, &rctx);
  rctx.push_back<LocalStoreStmt>(result, false_val_stmt);

  auto true_block = std::make_unique<Block>();
  true_block->set_statements(std::move(lctx.stmts));
  if_stmt->set_true_statements(std::move(true_block));

  auto false_block = std::make_unique<Block>();
  false_block->set_statements(std::move(rctx.stmts));
  if_stmt->set_false_statements(std::move(false_block));

  ctx->push_back<LocalLoadStmt>(result);
  return;
}

static std::tuple<Expr, Expr, Expr> unify_ternaryop_operands(const Expr &e1,
                                                             const Expr &e2,
                                                             const Expr &e3) {
  auto target_dtype = PrimitiveType::unknown;
  // Since we don't support broadcasting between two TensorTypes,
  // we can simply use the first TensorType's dtype as the target dtype.
  if (e1->ret_type->is<TensorType>()) {
    target_dtype = e1->ret_type;
  } else if (e2->ret_type->is<TensorType>()) {
    target_dtype = e2->ret_type;
  } else if (e3->ret_type->is<TensorType>()) {
    target_dtype = e3->ret_type;
  }

  if (target_dtype == PrimitiveType::unknown) {
    return std::tuple(e1, e2, e3);
  }

  return std::tuple(to_broadcast_tensor(e1, target_dtype),
                    to_broadcast_tensor(e2, target_dtype),
                    to_broadcast_tensor(e3, target_dtype));
}

void TernaryOpExpression::type_check(const CompileConfig *config) {
  TI_ASSERT_TYPE_CHECKED(op1);
  TI_ASSERT_TYPE_CHECKED(op2);
  TI_ASSERT_TYPE_CHECKED(op3);

  bool is_valid = true;
  bool is_tensor = false;

  auto [unified_cond, unified_l, unified_r] =
      unify_ternaryop_operands(op1, op2, op3);
  op1 = unified_cond;
  op2 = unified_l;
  op3 = unified_r;
  auto op1_type = op1->ret_type;
  auto op2_type = op2->ret_type;
  auto op3_type = op3->ret_type;

  auto error = [&]() {
    throw TaichiTypeError(
        fmt::format("unsupported operand type(s) for '{}': '{}', '{}' and '{}'",
                    ternary_type_name(type), op1->ret_type->to_string(),
                    op2->ret_type->to_string(), op3->ret_type->to_string()));
  };

  if (op1_type->is<TensorType>() && op2_type->is<TensorType>() &&
      op3_type->is<TensorType>()) {
    // valid
    is_tensor = true;
    if (op1_type->cast<TensorType>()->get_shape() !=
        op2_type->cast<TensorType>()->get_shape()) {
      is_valid = false;
    }
    if (op2_type->cast<TensorType>()->get_shape() !=
        op3_type->cast<TensorType>()->get_shape()) {
      is_valid = false;
    }
    op1_type = op1_type->cast<TensorType>()->get_element_type();
    op2_type = op2_type->cast<TensorType>()->get_element_type();
    op3_type = op3_type->cast<TensorType>()->get_element_type();

  } else if (op1_type->is<PrimitiveType>() && op2_type->is<PrimitiveType>() &&
             op3_type->is<PrimitiveType>()) {
    // valid
  } else {
    is_valid = false;
  }

  if (op1_type != PrimitiveType::i32) {
    is_valid = false;
  }
  if (!op2_type->is<PrimitiveType>() || !op3_type->is<PrimitiveType>()) {
    is_valid = false;
  }

  if (!is_valid)
    error();

  if (is_tensor) {
    auto primitive_dtype = promoted_type(op2_type, op3_type);
    auto shape = op2->ret_type->cast<TensorType>()->get_shape();
    ret_type = TypeFactory::create_tensor_type(shape, primitive_dtype);
  } else {
    ret_type = promoted_type(op2_type, op3_type);
  }
}

void TernaryOpExpression::flatten(FlattenContext *ctx) {
  // if (stmt)
  //  return;
  if (type == TernaryOpType::select) {
    auto op1_stmt = flatten_rvalue(op1, ctx);
    auto op2_stmt = flatten_rvalue(op2, ctx);
    auto op3_stmt = flatten_rvalue(op3, ctx);
    ctx->push_back(
        std::make_unique<TernaryOpStmt>(type, op1_stmt, op2_stmt, op3_stmt));
  } else if (type == TernaryOpType::ifte) {
    make_ifte(ctx, ret_type, op1, op2, op3);
  }
  stmt = ctx->back_stmt();
  stmt->tb = tb;
  stmt->ret_type = ret_type;
}

void InternalFuncCallExpression::type_check(const CompileConfig *) {
  for (auto &arg : args) {
    TI_ASSERT_TYPE_CHECKED(arg);
    // no arg type compatibility check for now due to lack of specification
  }
  // internal func calls have default return type
  ret_type = PrimitiveType::i32;
}

void InternalFuncCallExpression::flatten(FlattenContext *ctx) {
  std::vector<Stmt *> args_stmts(args.size());
  for (int i = 0; i < (int)args.size(); ++i) {
    args_stmts[i] = flatten_rvalue(args[i], ctx);
  }
  ctx->push_back<InternalFuncStmt>(func_name, args_stmts, nullptr,
                                   with_runtime_context);
  stmt = ctx->back_stmt();
  stmt->tb = tb;
}

void ExternalTensorExpression::flatten(FlattenContext *ctx) {
  // https://github.com/taichi-dev/taichi/issues/5819
  // ArgLoadStmt keeps primitive types since all matrix-type gets
  // scalarized at python-scope
  //
  // FIXME(zhanlue): ArgLoadStmt should use TensorType once real_matrix is
  // turned-on by default.
  //                 The scalarization should happen after
  //                 irpass::lower_access()
  auto prim_dt = dt;
  auto ptr = Stmt::make<ArgLoadStmt>(arg_id, prim_dt, /*is_ptr=*/true,
                                     /*is_grad=*/is_grad);

  int external_dims = dim - std::abs(element_dim);
  ptr->cast<ArgLoadStmt>()->set_extern_dims(external_dims);

  ptr->tb = tb;
  ctx->push_back(std::move(ptr));
  stmt = ctx->back_stmt();
}

std::vector<Stmt *> make_index_stmts(Expression::FlattenContext *ctx,
                                     const ExprGroup &indices,
                                     const std::vector<int> &offsets) {
  std::vector<Stmt *> index_stmts;
  for (int i = 0; i < (int)indices.size(); i++) {
    Stmt *ind = flatten_rvalue(indices.exprs[i], ctx);
    if (!offsets.empty()) {
      auto offset = ctx->push_back<ConstStmt>(TypedConstant(offsets[i]));
      ind = ctx->push_back<BinaryOpStmt>(BinaryOpType::sub, ind, offset);
    }
    index_stmts.push_back(ind);
  }
  return index_stmts;
}

Stmt *make_field_access(Expression::FlattenContext *ctx,
                        const FieldExpression &field,
                        ExprGroup indices) {
  return ctx->push_back(std::make_unique<GlobalPtrStmt>(
      field.snode, make_index_stmts(ctx, indices, field.snode->index_offsets)));
}

Stmt *make_matrix_field_access(Expression::FlattenContext *ctx,
                               const MatrixFieldExpression &matrix_field,
                               ExprGroup indices,
                               DataType ret_type) {
  std::vector<SNode *> snodes;
  for (auto &field : matrix_field.fields) {
    snodes.push_back(field.cast<FieldExpression>()->snode);
  }
  return ctx->push_back(std::make_unique<MatrixOfGlobalPtrStmt>(
      snodes, make_index_stmts(ctx, indices, snodes[0]->index_offsets),
      matrix_field.dynamic_indexable, matrix_field.dynamic_index_stride,
      ret_type));
}

Stmt *make_ndarray_access(Expression::FlattenContext *ctx,
                          Expr var,
                          ExprGroup indices) {
  std::vector<Stmt *> index_stmts;
  for (int i = 0; i < (int)indices.size(); i++) {
    Stmt *ind = flatten_rvalue(indices.exprs[i], ctx);
    index_stmts.push_back(ind);
  }
  auto var_stmt = flatten_lvalue(var, ctx);
  auto expr = var.cast<ExternalTensorExpression>();
  auto external_ptr_stmt = std::make_unique<ExternalPtrStmt>(
      var_stmt, index_stmts, expr->dt.get_shape(), expr->element_dim);
  if (expr->dim == indices.size()) {
    // Indexing into an scalar element
    external_ptr_stmt->ret_type = expr->dt.ptr_removed().get_element_type();
  } else {
    // Indexing outer dimensions
    external_ptr_stmt->ret_type = expr->dt.ptr_removed();
  }

  return ctx->push_back(std::move(external_ptr_stmt));
}

Stmt *make_tensor_access_single_element(Expression::FlattenContext *ctx,
                                        Stmt *var_stmt,
                                        const ExprGroup &indices,
                                        const std::vector<int> &shape,
                                        const std::string &tb) {
  bool needs_dynamic_index = false;
  for (int i = 0; i < (int)indices.size(); ++i) {
    if (!indices[i].is<ConstExpression>()) {
      needs_dynamic_index = true;
    }
  }
  Stmt *offset_stmt = nullptr;
  if (needs_dynamic_index) {
    offset_stmt = ctx->push_back<ConstStmt>(TypedConstant(0));
    for (int i = 0; i < (int)indices.size(); ++i) {
      auto index_stmt = flatten_rvalue(indices[i], ctx);
      Stmt *shape_stmt = ctx->push_back<ConstStmt>(TypedConstant(shape[i]));
      Stmt *mul_stmt = ctx->push_back<BinaryOpStmt>(BinaryOpType::mul,
                                                    offset_stmt, shape_stmt);
      offset_stmt =
          ctx->push_back<BinaryOpStmt>(BinaryOpType::add, mul_stmt, index_stmt);
    }
  } else {
    int offset = 0;
    for (int i = 0; i < (int)indices.size(); ++i) {
      offset =
          offset * shape[i] + indices[i].cast<ConstExpression>()->val.val_int();
    }
    offset_stmt = ctx->push_back<ConstStmt>(TypedConstant(offset));
  }
  return ctx->push_back<MatrixPtrStmt>(var_stmt, offset_stmt, tb);
}

Stmt *make_tensor_access(Expression::FlattenContext *ctx,
                         Expr var,
                         const std::vector<ExprGroup> &indices_group,
                         DataType ret_type,
                         std::vector<int> shape,
                         const std::string &tb) {
  auto var_stmt = flatten_lvalue(var, ctx);
  if (!var->is_lvalue()) {
    auto alloca_stmt = ctx->push_back<AllocaStmt>(var->ret_type);
    ctx->push_back<LocalStoreStmt>(alloca_stmt, var_stmt);
    var_stmt = alloca_stmt;
  }
  if (is_tensor(ret_type)) {
    std::vector<Stmt *> stmts;
    for (auto &indices : indices_group) {
      stmts.push_back(
          make_tensor_access_single_element(ctx, var_stmt, indices, shape, tb));
    }
    return ctx->push_back<MatrixOfMatrixPtrStmt>(stmts, ret_type);
  }
  return make_tensor_access_single_element(ctx, var_stmt, indices_group[0],
                                           shape, tb);
}

void MatrixExpression::type_check(const CompileConfig *config) {
  TI_ASSERT(dt->as<TensorType>()->get_num_elements() == elements.size());

  for (auto &arg : elements) {
    TI_ASSERT_TYPE_CHECKED(arg);
    if (arg->ret_type != dt.get_element_type()) {
      arg = cast(arg, dt.get_element_type());
      arg->type_check(config);
    }
  }
  ret_type = dt;
}

void MatrixExpression::flatten(FlattenContext *ctx) {
  TI_ASSERT(this->dt->is<TensorType>());
  std::vector<Stmt *> values;
  for (auto &elt : elements) {
    values.push_back(flatten_rvalue(elt, ctx));
  }
  stmt = ctx->push_back<MatrixInitStmt>(values);
  stmt->ret_type = this->dt;
}

IndexExpression::IndexExpression(const Expr &var,
                                 const ExprGroup &indices,
                                 std::string tb)
    : var(var), indices_group({indices}) {
  this->tb = tb;
}

IndexExpression::IndexExpression(const Expr &var,
                                 const std::vector<ExprGroup> &indices_group,
                                 const std::vector<int> &ret_shape,
                                 std::string tb)
    : var(var), indices_group(indices_group), ret_shape(ret_shape) {
  // IndexExpression with ret_shape is used for matrix slicing, where each entry
  // of ExprGroup is interpreted as a group of indices to return within each
  // axis. For example, mat[0, 3:5] has indices_group={0, [3, 4]}, where [3, 4]
  // means "m"-axis will return a TensorType with size of 2. In this case, we
  // should not expand indices_group due to its special semantics.
  this->tb = tb;
}

bool IndexExpression::is_field() const {
  return var.is<FieldExpression>();
}

bool IndexExpression::is_matrix_field() const {
  return var.is<MatrixFieldExpression>();
}

bool IndexExpression::is_ndarray() const {
  return var.is<ExternalTensorExpression>();
}

bool IndexExpression::is_tensor() const {
  return var->ret_type->is<TensorType>();
}

bool IndexExpression::is_local() const {
  return !is_global();
}

bool IndexExpression::is_global() const {
  // Special case: Indexing into TensorType-element of ExternalPtrStmt
  // or GlobalPtrStmt should be treated as global ptrs
  if (var.is<IndexExpression>()) {
    TI_ASSERT(var.cast<IndexExpression>()->is_matrix_field() ||
              var.cast<IndexExpression>()->is_ndarray());
    return true;
  }

  // Only Ndarray and Field comes outside from a kernel
  return is_field() || is_matrix_field() || is_ndarray();
}

static void field_validation(FieldExpression *field_expr, int index_dim) {
  TI_ASSERT(field_expr != nullptr);
  TI_ASSERT(field_expr->snode != nullptr);
  int field_dim = field_expr->snode->num_active_indices;

  if (field_dim != index_dim) {
    throw TaichiIndexError(
        fmt::format("Field with dim {} accessed with indices of dim {}",
                    field_dim, index_dim));
  }
}

void IndexExpression::type_check(const CompileConfig *) {
  // TODO: Change to type-based solution
  // Currently, dimension compatibility check happens in Python
  TI_ASSERT(indices_group.size() == std::accumulate(begin(ret_shape),
                                                    end(ret_shape), 1,
                                                    std::multiplies<>()));
  int index_dim = indices_group.empty() ? 0 : indices_group[0].size();
  bool has_slice = !ret_shape.empty();
  if (has_slice) {
    TI_ASSERT_INFO(is_tensor(), "Slice or swizzle can only apply on matrices");
    auto element_type = var->ret_type->as<TensorType>()->get_element_type();
    ret_type = TypeFactory::create_tensor_type(ret_shape, element_type);

  } else if (is_field()) {  // field
    auto field_expr = var.cast<FieldExpression>();
    field_validation(field_expr.get(), index_dim);
    ret_type = field_expr->dt->get_compute_type();

  } else if (is_matrix_field()) {
    auto matrix_field_expr = var.cast<MatrixFieldExpression>();

    TI_ASSERT(!matrix_field_expr->fields.empty());
    auto field_expr = matrix_field_expr->fields[0].cast<FieldExpression>();
    field_validation(field_expr.get(), index_dim);

    ret_type = TypeFactory::create_tensor_type(matrix_field_expr->element_shape,
                                               matrix_field_expr->fields[0]
                                                   .cast<FieldExpression>()
                                                   ->dt->get_compute_type());
  } else if (is_ndarray()) {  // ndarray
    auto external_tensor_expr = var.cast<ExternalTensorExpression>();
    int total_dim = external_tensor_expr->dim;
    int element_dim = external_tensor_expr->dt.get_shape().size();
    if (total_dim != index_dim + element_dim) {
      throw TaichiTypeError(
          fmt::format("Array with dim {} accessed with indices of dim {}",
                      total_dim - element_dim, index_dim));
    }

    if (index_dim == total_dim) {
      // Access all the way to a single element
      ret_type = var.cast<ExternalTensorExpression>()->dt.get_element_type();
    } else {
      // Access to a Tensor
      ret_type = var.cast<ExternalTensorExpression>()->dt;
    }
  } else if (is_tensor()) {  // local tensor
    auto shape = var->ret_type->as<TensorType>()->get_shape();
    if (indices_group[0].size() != shape.size()) {
      TI_ERROR("Expected {} indices, got {}.", shape.size(),
               indices_group[0].size());
    }
    ret_type = var->ret_type->cast<TensorType>()->get_element_type();
  } else {
    throw TaichiTypeError(
        "Invalid IndexExpression: the source is not among field, ndarray or "
        "local tensor");
  }

  for (auto &indices : indices_group) {
    for (int i = 0; i < indices.exprs.size(); i++) {
      auto &expr = indices.exprs[i];
      TI_ASSERT_TYPE_CHECKED(expr);
      if (!is_integral(expr->ret_type))
        throw TaichiTypeError(
            fmt::format("indices must be integers, however '{}' is "
                        "provided as index {}",
                        expr->ret_type->to_string(), i));
    }
  }
}

void IndexExpression::flatten(FlattenContext *ctx) {
  if (is_field()) {
    stmt =
        make_field_access(ctx, *var.cast<FieldExpression>(), indices_group[0]);
  } else if (is_matrix_field()) {
    stmt = make_matrix_field_access(ctx, *var.cast<MatrixFieldExpression>(),
                                    indices_group[0], ret_type);
  } else if (is_ndarray()) {
    stmt = make_ndarray_access(ctx, var, indices_group[0]);
  } else if (is_tensor()) {
    stmt =
        make_tensor_access(ctx, var, indices_group, ret_type,
                           var->ret_type->cast<TensorType>()->get_shape(), tb);
  } else {
    throw TaichiTypeError(
        "Invalid IndexExpression: the source is not among field, ndarray or "
        "local tensor");
  }
  stmt->tb = tb;
}

void RangeAssumptionExpression::type_check(const CompileConfig *) {
  TI_ASSERT_TYPE_CHECKED(input);
  TI_ASSERT_TYPE_CHECKED(base);
  if (!input->ret_type->is<PrimitiveType>() ||
      !base->ret_type->is<PrimitiveType>() || input->ret_type != base->ret_type)
    throw TaichiTypeError(
        fmt::format("unsupported operand type(s) for "
                    "'range_assumption': '{}' and '{}'",
                    input->ret_type->to_string(), base->ret_type->to_string()));
  ret_type = input->ret_type;
}

void RangeAssumptionExpression::flatten(FlattenContext *ctx) {
  auto input_stmt = flatten_rvalue(input, ctx);
  auto base_stmt = flatten_rvalue(base, ctx);
  ctx->push_back(
      Stmt::make<RangeAssumptionStmt>(input_stmt, base_stmt, low, high));
  stmt = ctx->back_stmt();
}

void LoopUniqueExpression::type_check(const CompileConfig *) {
  TI_ASSERT_TYPE_CHECKED(input);
  if (!input->ret_type->is<PrimitiveType>())
    throw TaichiTypeError(
        fmt::format("unsupported operand type(s) for 'loop_unique': '{}'",
                    input->ret_type->to_string()));
  ret_type = input->ret_type;
}

void LoopUniqueExpression::flatten(FlattenContext *ctx) {
  auto input_stmt = flatten_rvalue(input, ctx);
  ctx->push_back(Stmt::make<LoopUniqueStmt>(input_stmt, covers));
  stmt = ctx->back_stmt();
}

void IdExpression::flatten(FlattenContext *ctx) {
  stmt = ctx->current_block->lookup_var(id);
  if (!ret_type->is_primitive(PrimitiveTypeID::unknown)) {
    stmt->ret_type = ret_type;
  }
}

void AtomicOpExpression::type_check(const CompileConfig *config) {
  TI_ASSERT_TYPE_CHECKED(dest);
  TI_ASSERT_TYPE_CHECKED(val);
  auto error = [&]() {
    throw TaichiTypeError(fmt::format(
        "unsupported operand type(s) for 'atomic_{}': '{}' and '{}'",
        atomic_op_type_name(op_type), dest->ret_type->to_string(),
        val->ret_type->to_string()));
  };

  // Broadcast val to dest if neccessary
  auto val_dtype = val->ret_type;
  auto dest_dtype = dest->ret_type.ptr_removed();
  if (dest_dtype->is<PrimitiveType>() && val_dtype->is<TensorType>()) {
    error();
  }

  if (val_dtype->is<PrimitiveType>() && dest_dtype->is<TensorType>()) {
    auto broadcasted_expr = to_broadcast_tensor(val, dest_dtype);
    val = std::move(broadcasted_expr);
    val.type_check(config);
  }

  // Validate dtype
  auto dtype = val->ret_type;
  if (dtype->is<TensorType>()) {
    dtype = dtype.get_element_type();
  }

  if (!dtype->is<PrimitiveType>()) {
    error();
  }

  if (is_quant(dest->ret_type)) {
    ret_type = dest->ret_type->get_compute_type();
  } else if (dest->ret_type->is<PrimitiveType>() ||
             dest->ret_type->is<TensorType>()) {
    ret_type = dest->ret_type;
  } else {
    error();
  }
}

void AtomicOpExpression::flatten(FlattenContext *ctx) {
  TI_ASSERT(
      dest.is<IdExpression>() || dest.is<IndexExpression>() ||
      (dest.is<ArgLoadExpression>() && dest.cast<ArgLoadExpression>()->is_ptr));
  // replace atomic sub with negative atomic add
  if (op_type == AtomicOpType::sub) {
    if (val->ret_type != ret_type) {
      val.set(Expr::make<UnaryOpExpression>(UnaryOpType::cast_value, val,
                                            ret_type));
    }

    val.set(Expr::make<UnaryOpExpression>(UnaryOpType::neg, val));
    op_type = AtomicOpType::add;
  }
  // expand rhs
  auto val_stmt = flatten_rvalue(val, ctx);
  auto dest_stmt = flatten_lvalue(dest, ctx);
  stmt = ctx->push_back<AtomicOpStmt>(op_type, dest_stmt, val_stmt);
  stmt->ret_type = stmt->as<AtomicOpStmt>()->dest->ret_type;
  stmt->tb = tb;
}

SNodeOpExpression::SNodeOpExpression(SNode *snode,
                                     SNodeOpType op_type,
                                     const ExprGroup &indices)
    : snode(snode), op_type(op_type), indices(indices) {
}

SNodeOpExpression::SNodeOpExpression(SNode *snode,
                                     SNodeOpType op_type,
                                     const ExprGroup &indices,
                                     const std::vector<Expr> &values)
    : SNodeOpExpression(snode, op_type, indices) {
  this->values = values;
}

void SNodeOpExpression::type_check(const CompileConfig *config) {
  if (op_type == SNodeOpType::get_addr) {
    ret_type = PrimitiveType::u64;
  } else {
    ret_type = PrimitiveType::i32;
  }
  if (op_type == SNodeOpType::append) {
    TI_ASSERT(snode->ch.size() == values.size());
    for (int i = 0; i < values.size(); i++) {
      TI_ASSERT_TYPE_CHECKED(values[i]);
      auto &dst_type = snode->ch[i]->dt;
      auto promoted = promoted_type(dst_type, values[i]->ret_type);
      if (dst_type != promoted) {
        TI_WARN("Append may lose precision: {} <- {}\n{}",
                dst_type->to_string(), values[i]->ret_type->to_string(), tb);
      }
      values[i] = cast(values[i], dst_type);
      values[i]->type_check(config);
    }
  }
}

void SNodeOpExpression::flatten(FlattenContext *ctx) {
  std::vector<Stmt *> indices_stmt;
  for (int i = 0; i < (int)indices.size(); i++) {
    indices_stmt.push_back(flatten_rvalue(indices[i], ctx));
  }
  auto is_cell_access = SNodeOpStmt::activation_related(op_type) &&
                        snode->type != SNodeType::dynamic;
  auto ptr =
      ctx->push_back<GlobalPtrStmt>(snode, indices_stmt, true, is_cell_access);
  ptr->tb = tb;
  if (op_type == SNodeOpType::is_active) {
    TI_ERROR_IF(snode->type != SNodeType::pointer &&
                    snode->type != SNodeType::hash &&
                    snode->type != SNodeType::bitmasked,
                "ti.is_active only works on pointer, hash or bitmasked nodes.");
    ctx->push_back<SNodeOpStmt>(SNodeOpType::is_active, snode, ptr, nullptr);
  } else if (op_type == SNodeOpType::length) {
    ctx->push_back<SNodeOpStmt>(SNodeOpType::length, snode, ptr, nullptr);
  } else if (op_type == SNodeOpType::get_addr) {
    ctx->push_back<SNodeOpStmt>(SNodeOpType::get_addr, snode, ptr, nullptr);
  } else if (op_type == SNodeOpType::append) {
    auto alloca = ctx->push_back<AllocaStmt>(PrimitiveType::i32);
    alloca->set_tb(tb);
    auto addr =
        ctx->push_back<SNodeOpStmt>(SNodeOpType::allocate, snode, ptr, alloca);
    addr->set_tb(tb);
    for (int i = 0; i < values.size(); i++) {
      auto value_stmt = flatten_rvalue(values[i], ctx);
      auto ch_addr = ctx->push_back<GetChStmt>(addr, snode, i);
      ch_addr->set_tb(tb);
      ctx->push_back<GlobalStoreStmt>(ch_addr, value_stmt)->set_tb(tb);
    }
    ctx->push_back<LocalLoadStmt>(alloca)->set_tb(tb);
    TI_ERROR_IF(snode->type != SNodeType::dynamic,
                "ti.append only works on dynamic nodes.");
  }
  stmt = ctx->back_stmt();
}

TextureOpExpression::TextureOpExpression(TextureOpType op,
                                         Expr texture_ptr,
                                         const ExprGroup &args)
    : op(op), texture_ptr(texture_ptr), args(args) {
}

void TextureOpExpression::type_check(const CompileConfig *config) {
  TI_ASSERT(texture_ptr.is<TexturePtrExpression>());
  auto ptr = texture_ptr.cast<TexturePtrExpression>();
  if (op == TextureOpType::kSampleLod) {
    // UV, Lod
    TI_ASSERT_INFO(args.size() == ptr->num_dims + 1,
                   "Invalid number of args for sample_lod Texture op with a "
                   "{}-dimension texture",
                   ptr->num_dims);
    for (int i = 0; i < ptr->num_dims; i++) {
      TI_ASSERT_TYPE_CHECKED(args[i]);
      if (args[i].get_ret_type() != PrimitiveType::f32) {
        throw TaichiTypeError(
            fmt::format("Invalid type for texture sample_lod: '{}', all "
                        "arguments must be f32",
                        args[i].get_ret_type()->to_string()));
      }
    }
  } else if (op == TextureOpType::kFetchTexel) {
    // index, int LOD
    TI_ASSERT_INFO(args.size() == ptr->num_dims + 1,
                   "Invalid number of args for fetch_texel Texture op with a "
                   "{}-dimension texture",
                   ptr->num_dims);
    for (int i = 0; i < ptr->num_dims; i++) {
      TI_ASSERT_TYPE_CHECKED(args[i]);
      if (args[i].get_ret_type() != PrimitiveType::i32) {
        throw TaichiTypeError(
            fmt::format("Invalid type for texture fetch_texel: '{}', all "
                        "arguments must be i32",
                        args[i].get_ret_type()->to_string()));
      }
    }
  } else if (op == TextureOpType::kLoad) {
    // index
    TI_ASSERT_INFO(args.size() == ptr->num_dims,
                   "Invalid number of args for load Texture op with a "
                   "{}-dimension texture",
                   ptr->num_dims);
    for (int i = 0; i < ptr->num_dims; i++) {
      TI_ASSERT_TYPE_CHECKED(args[i]);
      if (args[i].get_ret_type() != PrimitiveType::i32) {
        throw TaichiTypeError(
            fmt::format("Invalid type for texture load: '{}', all "
                        "arguments must be i32",
                        args[i].get_ret_type()->to_string()));
      }
    }
  } else if (op == TextureOpType::kStore) {
    // index, value
    TI_ASSERT_INFO(args.size() == ptr->num_dims + 4,
                   "Invalid number of args for store Texture op with a "
                   "{}-dimension texture",
                   ptr->num_dims);
    for (int i = 0; i < ptr->num_dims; i++) {
      TI_ASSERT_TYPE_CHECKED(args[i]);
      if (args[i].get_ret_type() != PrimitiveType::i32) {
        throw TaichiTypeError(
            fmt::format("Invalid type for texture load: '{}', index "
                        "arguments must be i32",
                        args[i].get_ret_type()->to_string()));
      }
    }
    for (int i = ptr->num_dims; i < ptr->num_dims + 4; i++) {
      TI_ASSERT_TYPE_CHECKED(args[i]);
      if (args[i].get_ret_type() != PrimitiveType::f32) {
        throw TaichiTypeError(
            fmt::format("Invalid type for texture load: '{}', value "
                        "arguments must be f32",
                        args[i].get_ret_type()->to_string()));
      }
    }
  } else {
    TI_ERROR("Invalid TextureOpType");
  }
  ret_type =
      TypeFactory::get_instance().get_pointer_type(PrimitiveType::f32,
                                                   /*is_bit_pointer=*/false);
}

void TextureOpExpression::flatten(FlattenContext *ctx) {
  auto texture_ptr_stmt = flatten_rvalue(texture_ptr, ctx);
  std::vector<Stmt *> arg_stmts;
  for (Expr &arg : args.exprs) {
    arg_stmts.push_back(flatten_rvalue(arg, ctx));
  }
  ctx->push_back<TextureOpStmt>(op, texture_ptr_stmt, arg_stmts);
  stmt = ctx->back_stmt();
}

void ConstExpression::type_check(const CompileConfig *) {
  TI_ASSERT_INFO(
      val.dt->is<PrimitiveType>() && val.dt != PrimitiveType::unknown,
      "Invalid dt [{}] for ConstExpression", val.dt->to_string());
  ret_type = val.dt;
}

void ConstExpression::flatten(FlattenContext *ctx) {
  ctx->push_back(Stmt::make<ConstStmt>(val));
  stmt = ctx->back_stmt();
}

void ExternalTensorShapeAlongAxisExpression::type_check(const CompileConfig *) {
  TI_ASSERT_INFO(
      ptr.is<ExternalTensorExpression>() || ptr.is<TexturePtrExpression>(),
      "Invalid ptr [{}] for ExternalTensorShapeAlongAxisExpression",
      ExpressionHumanFriendlyPrinter::expr_to_string(ptr));
  ret_type = PrimitiveType::i32;
}

void ExternalTensorShapeAlongAxisExpression::flatten(FlattenContext *ctx) {
  auto temp = ptr.cast<ExternalTensorExpression>();
  TI_ASSERT(0 <= axis && axis < temp->dim);
  ctx->push_back<ExternalTensorShapeAlongAxisStmt>(axis, temp->arg_id);
  stmt = ctx->back_stmt();
}

void GetElementExpression::type_check(const CompileConfig *config) {
  TI_ASSERT_TYPE_CHECKED(src);

  ret_type = src->ret_type->as<StructType>()->get_element_type(index);
}

void GetElementExpression::flatten(FlattenContext *ctx) {
  ctx->push_back<GetElementStmt>(flatten_rvalue(src, ctx), index);
  stmt = ctx->back_stmt();
}
// Mesh related.

void MeshPatchIndexExpression::flatten(FlattenContext *ctx) {
  auto pid_stmt = std::make_unique<MeshPatchIndexStmt>();
  ctx->push_back(std::move(pid_stmt));
  stmt = ctx->back_stmt();
}

void MeshPatchIndexExpression::type_check(const CompileConfig *) {
  ret_type = PrimitiveType::i32;
}

void MeshRelationAccessExpression::type_check(const CompileConfig *) {
  ret_type = PrimitiveType::i32;
}

void MeshRelationAccessExpression::flatten(FlattenContext *ctx) {
  auto mesh_idx_stmt = flatten_rvalue(mesh_idx, ctx);
  if (neighbor_idx) {
    auto neighbor_idx_stmt = flatten_rvalue(neighbor_idx, ctx);
    ctx->push_back<MeshRelationAccessStmt>(mesh, mesh_idx_stmt, to_type,
                                           neighbor_idx_stmt);
  } else {
    ctx->push_back<MeshRelationAccessStmt>(mesh, mesh_idx_stmt, to_type);
  }
  stmt = ctx->back_stmt();
}

MeshIndexConversionExpression::MeshIndexConversionExpression(
    mesh::Mesh *mesh,
    mesh::MeshElementType idx_type,
    const Expr idx,
    mesh::ConvType conv_type)
    : mesh(mesh), idx_type(idx_type), idx(idx), conv_type(conv_type) {
}

void MeshIndexConversionExpression::type_check(const CompileConfig *) {
  ret_type = PrimitiveType::i32;
}

void MeshIndexConversionExpression::flatten(FlattenContext *ctx) {
  auto idx_stmt = flatten_rvalue(idx, ctx);
  ctx->push_back<MeshIndexConversionStmt>(mesh, idx_type, idx_stmt, conv_type);
  stmt = ctx->back_stmt();
}

void ReferenceExpression::type_check(const CompileConfig *) {
  ret_type = var->ret_type;
}

void ReferenceExpression::flatten(FlattenContext *ctx) {
  auto var_stmt = flatten_lvalue(var, ctx);
  ctx->push_back<ReferenceStmt>(var_stmt);
  stmt = ctx->back_stmt();
}

Block *ASTBuilder::current_block() {
  if (stack_.empty())
    return nullptr;
  else
    return stack_.back();
}

Stmt *ASTBuilder::get_last_stmt() {
  TI_ASSERT(!stack_.empty());
  return stack_.back()->back();
}

void ASTBuilder::insert(std::unique_ptr<Stmt> &&stmt, int location) {
  TI_ASSERT(!stack_.empty());
  stack_.back()->insert(std::move(stmt), location);
}

void ASTBuilder::stop_gradient(SNode *snode) {
  TI_ASSERT(!stack_.empty());
  stack_.back()->stop_gradients.push_back(snode);
}

void ASTBuilder::insert_assignment(Expr &lhs,
                                   const Expr &rhs,
                                   const std::string &tb) {
  // Inside a kernel or a function
  // Create an assignment in the IR
  if (lhs.expr == nullptr) {
    lhs.set(rhs);
  } else if (lhs.expr->is_lvalue()) {
    auto stmt = std::make_unique<FrontendAssignStmt>(lhs, rhs);
    stmt->tb = tb;
    this->insert(std::move(stmt));

  } else {
    TI_ERROR("Cannot assign to non-lvalue: {}",
             ExpressionHumanFriendlyPrinter::expr_to_string(lhs));
  }
}

Expr ASTBuilder::make_var(const Expr &x, std::string tb) {
  auto var = this->expr_alloca();
  this->insert_assignment(var, x, tb);
  return var;
}

Expr ASTBuilder::make_id_expr(const std::string &name) {
  return Expr::make<IdExpression>(get_next_id(name));
}

void ASTBuilder::insert_for(const Expr &s,
                            const Expr &e,
                            const std::function<void(Expr)> &func) {
  auto i = Expr(std::make_shared<IdExpression>(get_next_id()));
  auto stmt_unique = std::make_unique<FrontendForStmt>(i, s, e, this->arch_,
                                                       for_loop_dec_.config);
  for_loop_dec_.reset();
  auto stmt = stmt_unique.get();
  this->insert(std::move(stmt_unique));
  this->create_scope(stmt->body);
  func(i);
  this->pop_scope();
}

Expr ASTBuilder::insert_thread_idx_expr() {
  auto loop = stack_.size() ? stack_.back()->parent_stmt : nullptr;
  TI_ERROR_IF(
      arch_ != Arch::cuda && !arch_is_cpu(arch_) && arch_ != Arch::amdgpu,
      "ti.thread_idx() is only available in cuda or cpu or amdgpu context.");
  if (loop != nullptr) {
    auto i = stack_.size() - 1;
    while (!(loop->is<FrontendForStmt>())) {
      loop = i > 0 ? stack_[--i]->parent_stmt : nullptr;
      if (loop == nullptr)
        break;
    }
  }
  TI_ERROR_IF(!(loop && loop->is<FrontendForStmt>()),
              "ti.thread_idx() is only valid within loops.");
  return Expr::make<InternalFuncCallExpression>(
      "linear_thread_idx", std::vector<Expr>{}, /*with_runtime_context=*/true);
}

Expr ASTBuilder::insert_patch_idx_expr() {
  auto loop = stack_.size() ? stack_.back()->parent_stmt : nullptr;
  if (loop != nullptr) {
    auto i = stack_.size() - 1;
    while (!(loop->is<FrontendForStmt>())) {
      loop = i > 0 ? stack_[--i]->parent_stmt : nullptr;
      if (loop == nullptr)
        break;
    }
  }
  TI_ERROR_IF(!(loop && loop->is<FrontendForStmt>() &&
                loop->as<FrontendForStmt>()->mesh),
              "ti.mesh_patch_idx() is only valid within mesh-for loops.");
  return Expr::make<MeshPatchIndexExpression>();
}

void ASTBuilder::create_kernel_exprgroup_return(const ExprGroup &group) {
  auto expanded_exprs = this->expand_exprs(group.exprs);
  ExprGroup expanded_expr_group;
  expanded_expr_group.exprs = std::move(expanded_exprs);
  this->insert(Stmt::make<FrontendReturnStmt>(expanded_expr_group));
}

void ASTBuilder::create_print(
    std::vector<std::variant<Expr, std::string>> contents) {
  this->insert(std::make_unique<FrontendPrintStmt>(contents));
}

void ASTBuilder::begin_func(const std::string &funcid) {
  auto stmt_unique = std::make_unique<FrontendFuncDefStmt>(funcid);
  auto stmt = stmt_unique.get();
  this->insert(std::move(stmt_unique));
  this->create_scope(stmt->body);
}

void ASTBuilder::end_func(const std::string &funcid) {
  this->pop_scope();
}

void ASTBuilder::begin_frontend_if(const Expr &cond) {
  auto stmt_tmp = std::make_unique<FrontendIfStmt>(cond);
  this->insert(std::move(stmt_tmp));
}

void ASTBuilder::begin_frontend_if_true() {
  auto if_stmt = this->get_last_stmt()->as<FrontendIfStmt>();
  this->create_scope(if_stmt->true_statements);
}

void ASTBuilder::begin_frontend_if_false() {
  auto if_stmt = this->get_last_stmt()->as<FrontendIfStmt>();
  this->create_scope(if_stmt->false_statements);
}

void ASTBuilder::insert_external_func_call(std::size_t func_addr,
                                           std::string source,
                                           std::string filename,
                                           std::string funcname,
                                           const ExprGroup &args,
                                           const ExprGroup &outputs) {
  auto stmt = Stmt::make<FrontendExternalFuncStmt>(
      (void *)func_addr, source, filename, funcname, args.exprs, outputs.exprs);
  this->insert(std::move(stmt));
}

Expr ASTBuilder::expr_alloca() {
  auto var = Expr(std::make_shared<IdExpression>(get_next_id()));
  this->insert(std::make_unique<FrontendAllocaStmt>(
      std::static_pointer_cast<IdExpression>(var.expr)->id,
      PrimitiveType::unknown));
  return var;
}

std::optional<Expr> ASTBuilder::insert_func_call(Function *func,
                                                 const ExprGroup &args) {
  ExprGroup expanded_args;
  expanded_args.exprs = this->expand_exprs(args.exprs);
  if (func->ret_type) {
    auto var = Expr(std::make_shared<IdExpression>(get_next_id()));
    this->insert(std::make_unique<FrontendFuncCallStmt>(
        func, expanded_args,
        std::static_pointer_cast<IdExpression>(var.expr)->id));
    var.expr->ret_type = func->ret_type;
    return var;
  } else {
    this->insert(std::make_unique<FrontendFuncCallStmt>(func, expanded_args));
    return std::nullopt;
  }
}

Expr ASTBuilder::make_matrix_expr(const std::vector<int> &shape,
                                  const DataType &dt,
                                  const std::vector<Expr> &elements) {
  /*
    Since we have both "shape" and "element_type" in MatrixExpression,
    we should flatten all the elements and disallow recursive TensorType in
    element Expr
  */
  TI_ASSERT(dt->is<PrimitiveType>());
  auto expanded_elements = this->expand_exprs(elements);
  auto mat =
      Expr(std::make_shared<MatrixExpression>(expanded_elements, shape, dt));
  return mat;
}

Expr ASTBuilder::expr_alloca_shared_array(const std::vector<int> &shape,
                                          const DataType &element_type) {
  auto var = Expr(std::make_shared<IdExpression>(get_next_id()));
  this->insert(std::make_unique<FrontendAllocaStmt>(
      std::static_pointer_cast<IdExpression>(var.expr)->id, shape, element_type,
      true));
  var->ret_type = this->get_last_stmt()->ret_type;
  return var;
}

void ASTBuilder::expr_assign(const Expr &lhs, const Expr &rhs, std::string tb) {
  TI_ASSERT(lhs->is_lvalue());
  auto stmt = std::make_unique<FrontendAssignStmt>(lhs, rhs);
  stmt->set_tb(tb);
  this->insert(std::move(stmt));
}

Expr ASTBuilder::expr_subscript(const Expr &expr,
                                const ExprGroup &indices,
                                std::string tb) {
  TI_ASSERT(expr.is<FieldExpression>() || expr.is<MatrixFieldExpression>() ||
            expr.is<ExternalTensorExpression>() ||
            is_tensor(expr.expr->ret_type));

  // IndexExpression without ret_shape is used for matrix indexing,
  // where each entry of ExprGroup is interpreted as indexing into a specific
  // axis. For example, mat[3, 4] has indices_group={[3, 4]}, where [3, 4]
  // corresponds to "n"-axis and "m"-axis of the matrix. Therefore we expand
  // indices_group={[3, 4]} into {3, 4} to avoid TensorType in indices.
  std::vector<Expr> expanded_indices = this->expand_exprs(indices.exprs);
  auto expanded_expr_group = ExprGroup();
  expanded_expr_group.exprs = expanded_indices;

  return Expr::make<IndexExpression>(expr, expanded_expr_group, tb);
}

void ASTBuilder::create_assert_stmt(const Expr &cond,
                                    const std::string &msg,
                                    const std::vector<Expr> &args) {
  auto stmt_unique = std::make_unique<FrontendAssertStmt>(cond, msg, args);
  this->insert(std::move(stmt_unique));
}

void ASTBuilder::begin_frontend_range_for(const Expr &i,
                                          const Expr &s,
                                          const Expr &e) {
  auto stmt_unique =
      std::make_unique<FrontendForStmt>(i, s, e, arch_, for_loop_dec_.config);
  auto stmt = stmt_unique.get();
  this->insert(std::move(stmt_unique));
  this->create_scope(stmt->body,
                     for_loop_dec_.config.strictly_serialized ? While : For);
  for_loop_dec_.reset();
}

void ASTBuilder::begin_frontend_struct_for_on_snode(const ExprGroup &loop_vars,
                                                    SNode *snode) {
  TI_WARN_IF(
      for_loop_dec_.config.strictly_serialized,
      "ti.loop_config(serialize=True) does not have effect on the struct for. "
      "The execution order is not guaranteed.");
  auto stmt_unique = std::make_unique<FrontendForStmt>(loop_vars, snode, arch_,
                                                       for_loop_dec_.config);
  for_loop_dec_.reset();
  auto stmt = stmt_unique.get();
  this->insert(std::move(stmt_unique));
  this->create_scope(stmt->body, For);
}

void ASTBuilder::begin_frontend_struct_for_on_external_tensor(
    const ExprGroup &loop_vars,
    const Expr &external_tensor) {
  TI_WARN_IF(
      for_loop_dec_.config.strictly_serialized,
      "ti.loop_config(serialize=True) does not have effect on the struct for. "
      "The execution order is not guaranteed.");
  auto stmt_unique = std::make_unique<FrontendForStmt>(
      loop_vars, external_tensor, arch_, for_loop_dec_.config);
  for_loop_dec_.reset();
  auto stmt = stmt_unique.get();
  this->insert(std::move(stmt_unique));
  this->create_scope(stmt->body, For);
}

void ASTBuilder::begin_frontend_mesh_for(
    const Expr &i,
    const mesh::MeshPtr &mesh_ptr,
    const mesh::MeshElementType &element_type) {
  TI_WARN_IF(
      for_loop_dec_.config.strictly_serialized,
      "ti.loop_config(serialize=True) does not have effect on the mesh for. "
      "The execution order is not guaranteed.");
  auto stmt_unique = std::make_unique<FrontendForStmt>(
      ExprGroup(i), mesh_ptr, element_type, arch_, for_loop_dec_.config);
  for_loop_dec_.reset();
  auto stmt = stmt_unique.get();
  this->insert(std::move(stmt_unique));
  this->create_scope(stmt->body, For);
}

void ASTBuilder::begin_frontend_while(const Expr &cond) {
  auto stmt_unique = std::make_unique<FrontendWhileStmt>(cond);
  auto stmt = stmt_unique.get();
  this->insert(std::move(stmt_unique));
  this->create_scope(stmt->body, While);
}

void ASTBuilder::insert_break_stmt() {
  if (loop_state_stack_.back() == Outermost) {
    throw TaichiSyntaxError("Cannot break in the outermost loop");
  }
  this->insert(Stmt::make<FrontendBreakStmt>());
}

void ASTBuilder::insert_continue_stmt() {
  this->insert(Stmt::make<FrontendContinueStmt>());
}

void ASTBuilder::insert_expr_stmt(const Expr &val) {
  this->insert(Stmt::make<FrontendExprStmt>(val));
}

void ASTBuilder::insert_snode_activate(SNode *snode,
                                       const ExprGroup &expr_group) {
  ExprGroup expanded_group;
  expanded_group.exprs = this->expand_exprs(expr_group.exprs);
  this->insert(Stmt::make<FrontendSNodeOpStmt>(SNodeOpType::activate, snode,
                                               expanded_group));
}

void ASTBuilder::insert_snode_deactivate(SNode *snode,
                                         const ExprGroup &expr_group) {
  ExprGroup expanded_group;
  expanded_group.exprs = this->expand_exprs(expr_group.exprs);
  this->insert(Stmt::make<FrontendSNodeOpStmt>(SNodeOpType::deactivate, snode,
                                               expanded_group));
}

Expr ASTBuilder::snode_append(SNode *snode,
                              const ExprGroup &indices,
                              const std::vector<Expr> &vals) {
  ExprGroup expanded_exprs;
  expanded_exprs.exprs = this->expand_exprs(indices.exprs);
  std::vector<Expr> expanded_vals = this->expand_exprs(vals);
  return Expr::make<SNodeOpExpression>(snode, SNodeOpType::append,
                                       expanded_exprs, expanded_vals);
}

Expr ASTBuilder::snode_is_active(SNode *snode, const ExprGroup &indices) {
  ExprGroup expanded_exprs;
  expanded_exprs.exprs = this->expand_exprs(indices.exprs);
  return Expr::make<SNodeOpExpression>(snode, SNodeOpType::is_active,
                                       expanded_exprs);
}

Expr ASTBuilder::snode_length(SNode *snode, const ExprGroup &indices) {
  ExprGroup expanded_exprs;
  expanded_exprs.exprs = this->expand_exprs(indices.exprs);
  return Expr::make<SNodeOpExpression>(snode, SNodeOpType::length,
                                       expanded_exprs);
}

Expr ASTBuilder::snode_get_addr(SNode *snode, const ExprGroup &indices) {
  ExprGroup expanded_exprs;
  expanded_exprs.exprs = this->expand_exprs(indices.exprs);
  return Expr::make<SNodeOpExpression>(snode, SNodeOpType::get_addr,
                                       expanded_exprs);
}

std::vector<Expr> ASTBuilder::expand_exprs(const std::vector<Expr> &exprs) {
  if (exprs.size() == 0) {
    return exprs;
  }

  std::vector<Expr> expanded_exprs;
  for (auto expr : exprs) {
    TI_ASSERT_TYPE_CHECKED(expr);
    if (!expr->ret_type->is<TensorType>()) {
      expanded_exprs.push_back(expr);
    } else {
      // Expand TensorType expr
      /*
        Before:
          TensorType<4 x i32> index = Expr;

        After:
          TensorType<4 x i32>* id_expr = FrontendAllocaStmt(TensorType<4 x i32>)
          i32 ind0 = IndexExpression(id_expr, 0)
          i32 ind1 = IndexExpression(id_expr, 1)
          i32 ind2 = IndexExpression(id_expr, 2)
          i32 ind3 = IndexExpression(id_expr, 3)

          return {ind0, ind1, ind2, ind3}

      */
      auto tensor_type = expr->ret_type->cast<TensorType>();

      Expr id_expr;
      if (expr.is<IdExpression>()) {
        id_expr = expr;
      } else {
        id_expr = make_var(expr, expr->tb);
      }
      auto shape = tensor_type->get_shape();
      if (shape.size() == 1) {
        for (int i = 0; i < shape[0]; i++) {
          auto ind = Expr(std::make_shared<IndexExpression>(
              id_expr, ExprGroup(Expr(i)), expr->tb));
          ind.expr->ret_type = tensor_type->get_element_type();
          expanded_exprs.push_back(ind);
        }
      } else {
        TI_ASSERT(shape.size() == 2);
        for (int i = 0; i < shape[0]; i++) {
          for (int j = 0; j < shape[1]; j++) {
            auto ind = Expr(std::make_shared<IndexExpression>(
                id_expr, ExprGroup(Expr(i), Expr(j)), expr->tb));
            ind.expr->ret_type = tensor_type->get_element_type();
            expanded_exprs.push_back(ind);
          }
        }
      }
    }
  }

  return expanded_exprs;
}

Expr ASTBuilder::mesh_index_conversion(mesh::MeshPtr mesh_ptr,
                                       mesh::MeshElementType idx_type,
                                       const Expr &idx,
                                       mesh::ConvType &conv_type) {
  Expr expanded_idx;
  if (idx.is<IdExpression>() && idx.get_ret_type() == PrimitiveType::unknown) {
    expanded_idx = idx;
  } else {
    if (idx.expr->ret_type->is<TensorType>()) {
      TI_ASSERT(idx.expr->ret_type->cast<TensorType>()->get_num_elements() ==
                1);
    }
    expanded_idx = this->expand_exprs({idx})[0];
  }

  return Expr::make<MeshIndexConversionExpression>(mesh_ptr.ptr.get(), idx_type,
                                                   expanded_idx, conv_type);
}

void ASTBuilder::create_scope(std::unique_ptr<Block> &list, LoopType tp) {
  TI_ASSERT(list == nullptr);
  LoopState prev = loop_state_stack_.back();
  if (tp == NotLoop) {
    loop_state_stack_.push_back(prev);
  } else if (tp == For && stack_.size() == 1) {
    loop_state_stack_.push_back(Outermost);
  } else {
    loop_state_stack_.push_back(Inner);
  }
  list = std::make_unique<Block>();
  if (!stack_.empty()) {
    list->parent_stmt = get_last_stmt();
  }
  stack_.push_back(list.get());
}

void ASTBuilder::pop_scope() {
  stack_.pop_back();
  loop_state_stack_.pop_back();
}

Expr ASTBuilder::make_texture_op_expr(const TextureOpType &op,
                                      const Expr &texture_ptr,
                                      const ExprGroup &args) {
  ExprGroup expanded_args;
  expanded_args.exprs = this->expand_exprs(args.exprs);
  return Expr::make<TextureOpExpression>(op, texture_ptr, expanded_args);
}

Stmt *flatten_lvalue(Expr expr, Expression::FlattenContext *ctx) {
  expr->flatten(ctx);
  return expr->get_flattened_stmt();
}

Stmt *flatten_global_load(Stmt *ptr_stmt, Expression::FlattenContext *ctx) {
  ctx->push_back(std::make_unique<GlobalLoadStmt>(ptr_stmt));
  return ctx->back_stmt();
}

Stmt *flatten_local_load(Stmt *ptr_stmt, Expression::FlattenContext *ctx) {
  auto local_load = ctx->push_back<LocalLoadStmt>(ptr_stmt);
  local_load->ret_type = local_load->src->ret_type.ptr_removed();
  return local_load;
}

Stmt *flatten_rvalue(Expr ptr, Expression::FlattenContext *ctx) {
  ptr->flatten(ctx);
  Stmt *ptr_stmt = ptr->get_flattened_stmt();
  if (ptr.is<IdExpression>()) {
    if (ptr_stmt->is<AllocaStmt>()) {
      return flatten_local_load(ptr_stmt, ctx);
    }
  } else if (ptr.is<IndexExpression>()) {
    auto ix = ptr.cast<IndexExpression>();
    if (ix->is_local()) {
      return flatten_local_load(ptr_stmt, ctx);
    } else {
      return flatten_global_load(ptr_stmt, ctx);
    }
  } else if (ptr.is<ArgLoadExpression>() &&
             ptr.cast<ArgLoadExpression>()->is_ptr) {
    return flatten_global_load(ptr_stmt, ctx);
  }

  return ptr_stmt;
}

}  // namespace taichi::lang
