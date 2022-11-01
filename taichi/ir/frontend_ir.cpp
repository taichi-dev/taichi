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
  if (arch == Arch::cuda) {
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

void ArgLoadExpression::type_check(CompileConfig *) {
  TI_ASSERT_INFO(dt->is<PrimitiveType>() && dt != PrimitiveType::unknown,
                 "Invalid dt [{}] for ArgLoadExpression", dt->to_string());
  ret_type = dt;
}

void ArgLoadExpression::flatten(FlattenContext *ctx) {
  auto arg_load = std::make_unique<ArgLoadStmt>(arg_id, dt, is_ptr);
  ctx->push_back(std::move(arg_load));
  stmt = ctx->back_stmt();
}

void TexturePtrExpression::type_check(CompileConfig *config) {
}

void TexturePtrExpression::flatten(FlattenContext *ctx) {
  ctx->push_back<ArgLoadStmt>(arg_id, PrimitiveType::f32, true);
  ctx->push_back<TexturePtrStmt>(ctx->back_stmt(), num_dims, is_storage,
                                 num_channels, channel_format, lod);
  stmt = ctx->back_stmt();
}

void RandExpression::type_check(CompileConfig *) {
  TI_ASSERT_INFO(dt->is<PrimitiveType>() && dt != PrimitiveType::unknown,
                 "Invalid dt [{}] for RandExpression", dt->to_string());
  ret_type = dt;
}

void RandExpression::flatten(FlattenContext *ctx) {
  auto ran = std::make_unique<RandStmt>(dt);
  ctx->push_back(std::move(ran));
  stmt = ctx->back_stmt();
}

void UnaryOpExpression::type_check(CompileConfig *config) {
  TI_ASSERT_TYPE_CHECKED(operand);

  TI_ASSERT(config != nullptr);
  /*
    Dtype inference for both TensorType and PrimitiveType follow are essentially
    the same. Therefore we extract the primitive type to perform the type
    inference, and then reconstruct the TensorType once neccessary.
  */

  auto operand_primitive_type = operand->ret_type.get_element_type();
  auto ret_primitive_type = ret_type;

  if (config->real_matrix) {
    TI_ASSERT(operand_primitive_type->is<PrimitiveType>());

  } else if (!operand->ret_type->is<PrimitiveType>()) {
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
  flatten_rvalue(operand, ctx);
  auto unary = std::make_unique<UnaryOpStmt>(type, operand->stmt);
  if (is_cast()) {
    unary->cast_type = cast_type;
  }
  stmt = unary.get();
  stmt->tb = tb;
  ctx->push_back(std::move(unary));
}

Expr to_broadcast_tensor(const Expr &elt, const DataType &dt) {
  TI_ASSERT(dt->is<TensorType>());
  if (elt->ret_type == dt) {
    return elt;
  }
  auto tensor_type = dt->as<TensorType>();
  auto elt_type = tensor_type->get_element_type();
  TI_ASSERT_INFO(elt_type->is<PrimitiveType>(),
                 "Only primitive types are supported in Tensors, got {}",
                 elt_type->to_string());
  std::vector<Expr> broadcast_values(tensor_type->get_num_elements(), elt);
  return Expr::make<MatrixExpression>(broadcast_values,
                                      tensor_type->get_shape(), elt->ret_type);
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

void BinaryOpExpression::type_check(CompileConfig *config) {
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
      (lhs_type != PrimitiveType::i32 || rhs_type != PrimitiveType::i32) &&
      (!is_tensor_op || (lhs_type->cast<TensorType>()->get_element_type() !=
                             PrimitiveType::i32 ||
                         rhs_type->cast<TensorType>()->get_element_type() !=
                             PrimitiveType::i32)))
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
  flatten_rvalue(lhs, ctx);
  if (binary_is_logical(type)) {
    auto result = ctx->push_back<AllocaStmt>(ret_type);
    ctx->push_back<LocalStoreStmt>(result, lhs->stmt);
    auto cond = ctx->push_back<LocalLoadStmt>(result);
    auto if_stmt = ctx->push_back<IfStmt>(cond);

    FlattenContext rctx;
    rctx.current_block = ctx->current_block;
    flatten_rvalue(rhs, &rctx);
    rctx.push_back<LocalStoreStmt>(result, rhs->stmt);

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
  flatten_rvalue(rhs, ctx);
  ctx->push_back(std::make_unique<BinaryOpStmt>(type, lhs->stmt, rhs->stmt));
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
  flatten_rvalue(cond, ctx);
  auto if_stmt = ctx->push_back<IfStmt>(cond->stmt);

  Expression::FlattenContext lctx;
  lctx.current_block = ctx->current_block;
  flatten_rvalue(true_val, &lctx);
  lctx.push_back<LocalStoreStmt>(result, true_val->stmt);

  Expression::FlattenContext rctx;
  rctx.current_block = ctx->current_block;
  flatten_rvalue(false_val, &rctx);
  rctx.push_back<LocalStoreStmt>(result, false_val->stmt);

  auto true_block = std::make_unique<Block>();
  true_block->set_statements(std::move(lctx.stmts));
  if_stmt->set_true_statements(std::move(true_block));

  auto false_block = std::make_unique<Block>();
  false_block->set_statements(std::move(rctx.stmts));
  if_stmt->set_false_statements(std::move(false_block));

  ctx->push_back<LocalLoadStmt>(result);
  return;
}

void TernaryOpExpression::type_check(CompileConfig *config) {
  TI_ASSERT_TYPE_CHECKED(op1);
  TI_ASSERT_TYPE_CHECKED(op2);
  TI_ASSERT_TYPE_CHECKED(op3);
  auto op1_type = op1->ret_type;
  auto op2_type = op2->ret_type;
  auto op3_type = op3->ret_type;

  auto error = [&]() {
    throw TaichiTypeError(
        fmt::format("unsupported operand type(s) for '{}': '{}', '{}' and '{}'",
                    ternary_type_name(type), op1->ret_type->to_string(),
                    op2->ret_type->to_string(), op3->ret_type->to_string()));
  };

  bool is_valid = true;
  bool is_tensor = false;
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
    ret_type = TypeFactory::create_tensor_type(
        op2->ret_type->cast<TensorType>()->get_shape(), primitive_dtype);
  } else {
    ret_type = promoted_type(op2_type, op3_type);
  }
}

void TernaryOpExpression::flatten(FlattenContext *ctx) {
  // if (stmt)
  //  return;
  if (type == TernaryOpType::select) {
    flatten_rvalue(op1, ctx);
    flatten_rvalue(op2, ctx);
    flatten_rvalue(op3, ctx);
    ctx->push_back(
        std::make_unique<TernaryOpStmt>(type, op1->stmt, op2->stmt, op3->stmt));
  } else if (type == TernaryOpType::ifte) {
    make_ifte(ctx, ret_type, op1, op2, op3);
  }
  stmt = ctx->back_stmt();
  stmt->tb = tb;
  stmt->ret_type = ret_type;
}

void InternalFuncCallExpression::type_check(CompileConfig *) {
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
    flatten_rvalue(args[i], ctx);
    args_stmts[i] = args[i]->stmt;
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
  if (!get_compile_config()->real_matrix) {
    prim_dt = dt.get_element_type();
  }
  auto ptr = Stmt::make<ArgLoadStmt>(arg_id, prim_dt, /*is_ptr=*/true);

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
    flatten_rvalue(indices.exprs[i], ctx);
    Stmt *ind = indices.exprs[i]->stmt;
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
    flatten_rvalue(indices.exprs[i], ctx);
    Stmt *ind = indices.exprs[i]->stmt;
    index_stmts.push_back(ind);
  }
  flatten_lvalue(var, ctx);
  auto expr = var.cast<ExternalTensorExpression>();
  auto external_ptr_stmt = std::make_unique<ExternalPtrStmt>(
      expr->stmt, index_stmts, expr->dt.get_shape(), expr->element_dim);
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
                                        const Expr &var,
                                        const ExprGroup &indices,
                                        const std::vector<int> &shape,
                                        int stride) {
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
      flatten_rvalue(indices[i], ctx);
      Stmt *shape_stmt = ctx->push_back<ConstStmt>(TypedConstant(shape[i]));
      Stmt *mul_stmt = ctx->push_back<BinaryOpStmt>(BinaryOpType::mul,
                                                    offset_stmt, shape_stmt);
      offset_stmt = ctx->push_back<BinaryOpStmt>(BinaryOpType::add, mul_stmt,
                                                 indices[i]->stmt);
    }
  } else {
    int offset = 0;
    for (int i = 0; i < (int)indices.size(); ++i) {
      offset =
          offset * shape[i] + indices[i].cast<ConstExpression>()->val.val_int();
    }
    offset_stmt = ctx->push_back<ConstStmt>(TypedConstant(offset));
  }
  if (stride != 1) {
    Stmt *stride_stmt = ctx->push_back<ConstStmt>(TypedConstant(stride));
    offset_stmt = ctx->push_back<BinaryOpStmt>(BinaryOpType::mul, offset_stmt,
                                               stride_stmt);
  }
  return ctx->push_back<MatrixPtrStmt>(var->stmt, offset_stmt);
}

Stmt *make_tensor_access(Expression::FlattenContext *ctx,
                         Expr var,
                         const std::vector<ExprGroup> &indices_group,
                         DataType ret_type,
                         std::vector<int> shape,
                         int stride) {
  flatten_lvalue(var, ctx);
  if (!var->is_lvalue()) {
    auto alloca_stmt = ctx->push_back<AllocaStmt>(var->ret_type);
    ctx->push_back<LocalStoreStmt>(alloca_stmt, var->stmt);
    var->stmt = alloca_stmt;
  }
  if (is_tensor(ret_type)) {
    std::vector<Stmt *> stmts;
    for (auto &indices : indices_group) {
      stmts.push_back(
          make_tensor_access_single_element(ctx, var, indices, shape, stride));
    }
    return ctx->push_back<MatrixOfMatrixPtrStmt>(stmts, ret_type);
  }
  return make_tensor_access_single_element(ctx, var, indices_group[0], shape,
                                           stride);
}

void MatrixExpression::type_check(CompileConfig *config) {
  // TODO: typecheck matrix
  for (auto &arg : elements) {
    TI_ASSERT_TYPE_CHECKED(arg);
  }
  ret_type = dt;
}

void MatrixExpression::flatten(FlattenContext *ctx) {
  TI_ASSERT(this->dt->is<TensorType>());
  std::vector<Stmt *> values;
  for (auto &elt : elements) {
    flatten_rvalue(elt, ctx);
    values.push_back(elt->stmt);
  }
  stmt = ctx->push_back<MatrixInitStmt>(values);
  stmt->ret_type = this->dt;
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

void IndexExpression::type_check(CompileConfig *) {
  // TODO: Change to type-based solution
  // Currently, dimension compatibility check happens in Python
  TI_ASSERT(indices_group.size() == std::accumulate(begin(ret_shape),
                                                    end(ret_shape), 1,
                                                    std::multiplies<>()));
  if (!ret_shape.empty()) {
    TI_ASSERT_INFO(is_tensor(), "Slice or swizzle can only apply on matrices");
    auto element_type = var->ret_type->as<TensorType>()->get_element_type();
    ret_type = TypeFactory::create_tensor_type(ret_shape, element_type);
  } else if (is_field()) {  // field
    ret_type = var.cast<FieldExpression>()->dt->get_compute_type();
  } else if (is_matrix_field()) {
    auto matrix_field_expr = var.cast<MatrixFieldExpression>();
    ret_type = TypeFactory::create_tensor_type(matrix_field_expr->element_shape,
                                               matrix_field_expr->fields[0]
                                                   .cast<FieldExpression>()
                                                   ->dt->get_compute_type());
  } else if (is_ndarray()) {  // ndarray
    auto external_tensor_expr = var.cast<ExternalTensorExpression>();
    int total_dim = external_tensor_expr->dim;
    int index_dim = indices_group[0].exprs.size();

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
      TI_ERROR("Expected {} indices, but got {}.", shape.size(),
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
                           var->ret_type->cast<TensorType>()->get_shape(), 1);
  } else {
    throw TaichiTypeError(
        "Invalid IndexExpression: the source is not among field, ndarray or "
        "local tensor");
  }
  stmt->tb = tb;
}

void StrideExpression::type_check(CompileConfig *) {
  // This is an ugly hack for global tensors
  if (var.is<IndexExpression>() &&
      var.cast<IndexExpression>()->var.is<FieldExpression>())
    ret_type = var->ret_type;
  else
    throw TaichiTypeError(
        "Invalid StrideExpression: The source being indexed must be an element "
        "of a field");
}

void StrideExpression::flatten(FlattenContext *ctx) {
  stmt = make_tensor_access(ctx, var, {indices}, ret_type, shape, stride);
}

void RangeAssumptionExpression::type_check(CompileConfig *) {
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
  flatten_rvalue(input, ctx);
  flatten_rvalue(base, ctx);
  ctx->push_back(
      Stmt::make<RangeAssumptionStmt>(input->stmt, base->stmt, low, high));
  stmt = ctx->back_stmt();
}

void LoopUniqueExpression::type_check(CompileConfig *) {
  TI_ASSERT_TYPE_CHECKED(input);
  if (!input->ret_type->is<PrimitiveType>())
    throw TaichiTypeError(
        fmt::format("unsupported operand type(s) for 'loop_unique': '{}'",
                    input->ret_type->to_string()));
  ret_type = input->ret_type;
}

void LoopUniqueExpression::flatten(FlattenContext *ctx) {
  flatten_rvalue(input, ctx);
  ctx->push_back(Stmt::make<LoopUniqueStmt>(input->stmt, covers));
  stmt = ctx->back_stmt();
}

void IdExpression::flatten(FlattenContext *ctx) {
  stmt = ctx->current_block->lookup_var(id);
  if (!ret_type->is_primitive(PrimitiveTypeID::unknown)) {
    stmt->ret_type = ret_type;
  }
}

void AtomicOpExpression::type_check(CompileConfig *config) {
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
  flatten_rvalue(val, ctx);
  auto src_val = val->stmt;
  if (dest.is<IdExpression>()) {  // local variable
    // emit local store stmt
    auto alloca = ctx->current_block->lookup_var(dest.cast<IdExpression>()->id);
    ctx->push_back<AtomicOpStmt>(op_type, alloca, src_val);
  } else {
    TI_ASSERT(dest.is<IndexExpression>() || dest.is<StrideExpression>() ||
              (dest.is<ArgLoadExpression>() &&
               dest.cast<ArgLoadExpression>()->is_ptr));
    flatten_lvalue(dest, ctx);
    ctx->push_back<AtomicOpStmt>(op_type, dest->stmt, src_val);
  }
  stmt = ctx->back_stmt();
  stmt->ret_type = stmt->as<AtomicOpStmt>()->dest->ret_type;
  stmt->tb = tb;
}

void SNodeOpExpression::type_check(CompileConfig *) {
  if (op_type == SNodeOpType::get_addr) {
    ret_type = PrimitiveType::u64;
  } else {
    ret_type = PrimitiveType::i32;
  }
}

void SNodeOpExpression::flatten(FlattenContext *ctx) {
  std::vector<Stmt *> indices_stmt;
  for (int i = 0; i < (int)indices.size(); i++) {
    flatten_rvalue(indices[i], ctx);
    indices_stmt.push_back(indices[i]->stmt);
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
    flatten_rvalue(value, ctx);

    auto alloca = ctx->push_back<AllocaStmt>(PrimitiveType::i32);
    auto addr =
        ctx->push_back<SNodeOpStmt>(SNodeOpType::allocate, snode, ptr, alloca);
    auto ch_addr = ctx->push_back<GetChStmt>(addr, snode, 0);
    ctx->push_back<GlobalStoreStmt>(ch_addr, value->stmt);
    ctx->push_back<LocalLoadStmt>(alloca);
    TI_ERROR_IF(snode->type != SNodeType::dynamic,
                "ti.append only works on dynamic nodes.");
    TI_ERROR_IF(snode->ch.size() != 1,
                "ti.append only works on single-child dynamic nodes.");
  }
  stmt = ctx->back_stmt();
}

void TextureOpExpression::type_check(CompileConfig *config) {
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
  flatten_rvalue(texture_ptr, ctx);
  std::vector<Stmt *> arg_stmts;
  for (Expr &arg : args.exprs) {
    flatten_rvalue(arg, ctx);
    arg_stmts.push_back(arg->stmt);
  }
  ctx->push_back<TextureOpStmt>(op, texture_ptr->stmt, arg_stmts);
  stmt = ctx->back_stmt();
}

void ConstExpression::type_check(CompileConfig *) {
  TI_ASSERT_INFO(
      val.dt->is<PrimitiveType>() && val.dt != PrimitiveType::unknown,
      "Invalid dt [{}] for ConstExpression", val.dt->to_string());
  ret_type = val.dt;
}

void ConstExpression::flatten(FlattenContext *ctx) {
  ctx->push_back(Stmt::make<ConstStmt>(val));
  stmt = ctx->back_stmt();
}

void ExternalTensorShapeAlongAxisExpression::type_check(CompileConfig *) {
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

void FuncCallExpression::type_check(CompileConfig *) {
  for (auto &arg : args.exprs) {
    TI_ASSERT_TYPE_CHECKED(arg);
    // no arg type compatibility check for now due to lack of specification
  }
  TI_ASSERT_INFO(func->rets.size() <= 1,
                 "Too many (> 1) return values for FuncCallExpression");
  if (func->rets.size() == 1) {
    ret_type = func->rets[0].dt;
  }
}

void FuncCallExpression::flatten(FlattenContext *ctx) {
  std::vector<Stmt *> stmt_args;
  for (auto &arg : args.exprs) {
    flatten_rvalue(arg, ctx);
    stmt_args.push_back(arg->stmt);
  }
  ctx->push_back<FuncCallStmt>(func, stmt_args);
  stmt = ctx->back_stmt();
}

// Mesh related.

void MeshPatchIndexExpression::flatten(FlattenContext *ctx) {
  auto pid_stmt = std::make_unique<MeshPatchIndexStmt>();
  ctx->push_back(std::move(pid_stmt));
  stmt = ctx->back_stmt();
}

void MeshPatchIndexExpression::type_check(CompileConfig *) {
  ret_type = PrimitiveType::i32;
}

void MeshRelationAccessExpression::type_check(CompileConfig *) {
  ret_type = PrimitiveType::i32;
}

void MeshRelationAccessExpression::flatten(FlattenContext *ctx) {
  flatten_rvalue(mesh_idx, ctx);
  if (neighbor_idx) {
    flatten_rvalue(neighbor_idx, ctx);
    ctx->push_back<MeshRelationAccessStmt>(mesh, mesh_idx->stmt, to_type,
                                           neighbor_idx->stmt);
  } else {
    ctx->push_back<MeshRelationAccessStmt>(mesh, mesh_idx->stmt, to_type);
  }
  stmt = ctx->back_stmt();
}

void MeshIndexConversionExpression::type_check(CompileConfig *) {
  ret_type = PrimitiveType::i32;
}

void MeshIndexConversionExpression::flatten(FlattenContext *ctx) {
  flatten_rvalue(idx, ctx);
  ctx->push_back<MeshIndexConversionStmt>(mesh, idx_type, idx->stmt, conv_type);
  stmt = ctx->back_stmt();
}

void ReferenceExpression::type_check(CompileConfig *) {
  ret_type = var->ret_type;
}

void ReferenceExpression::flatten(FlattenContext *ctx) {
  flatten_lvalue(var, ctx);
  ctx->push_back<ReferenceStmt>(var->stmt);
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
  TI_ERROR_IF(arch_ != Arch::cuda && !arch_is_cpu(arch_),
              "ti.thread_idx() is only available in cuda or cpu context.");
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
  this->insert(Stmt::make<FrontendReturnStmt>(group));
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

Expr ASTBuilder::make_matrix_expr(const std::vector<int> &shape,
                                  const DataType &dt,
                                  const std::vector<Expr> &elements) {
  auto mat = Expr(std::make_shared<MatrixExpression>(elements, shape, dt));
  return mat;
}

Expr ASTBuilder::expr_alloca_local_tensor(const std::vector<int> &shape,
                                          const DataType &element_type,
                                          const ExprGroup &elements,
                                          std::string tb) {
  auto var = Expr(std::make_shared<IdExpression>(get_next_id()));
  this->insert(std::make_unique<FrontendAllocaStmt>(
      std::static_pointer_cast<IdExpression>(var.expr)->id, shape,
      element_type));
  var->ret_type = this->get_last_stmt()->ret_type;
  for (int i = 0; i < (int)elements.exprs.size(); ++i) {
    ExprGroup reversed_indices;
    int linearized_index = i;
    for (int d = (int)shape.size() - 1; d >= 0; --d) {
      reversed_indices.push_back(
          Expr::make<ConstExpression, int32>(linearized_index % shape[d]));
      linearized_index /= shape[d];
    }
    ExprGroup indices;
    for (int d = 0; d < (int)shape.size(); ++d)
      indices.push_back(reversed_indices[(int)shape.size() - 1 - d]);
    this->insert(std::make_unique<FrontendAssignStmt>(
        Expr::make<IndexExpression>(var, indices, tb), elements.exprs[i]));
  }
  return var;
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
  this->insert(Stmt::make<FrontendSNodeOpStmt>(SNodeOpType::activate, snode,
                                               expr_group));
}

void ASTBuilder::insert_snode_deactivate(SNode *snode,
                                         const ExprGroup &expr_group) {
  this->insert(Stmt::make<FrontendSNodeOpStmt>(SNodeOpType::deactivate, snode,
                                               expr_group));
}

std::vector<Expr> ASTBuilder::expand_expr(const std::vector<Expr> &exprs) {
  TI_ASSERT(exprs.size() > 0);

  if (exprs.size() > 1) {
    return exprs;
  }

  Expr index_expr = exprs[0];
  TI_ASSERT_TYPE_CHECKED(index_expr);
  if (!index_expr->ret_type->is<TensorType>()) {
    return exprs;
  }

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
  std::vector<Expr> expanded_exprs;

  auto tensor_type = index_expr->ret_type->cast<TensorType>();

  Expr id_expr;
  if (index_expr.is<IdExpression>()) {
    id_expr = index_expr;
  } else {
    id_expr = make_var(index_expr, index_expr->tb);
  }
  auto shape = tensor_type->get_shape();
  if (shape.size() == 1) {
    for (int i = 0; i < shape[0]; i++) {
      auto ind = Expr(std::make_shared<IndexExpression>(
          id_expr, ExprGroup(Expr(i)), index_expr->tb));
      ind.expr->ret_type = tensor_type->get_element_type();
      expanded_exprs.push_back(ind);
    }
  } else {
    TI_ASSERT(shape.size() == 2);
    for (int i = 0; i < shape[0]; i++) {
      for (int j = 0; j < shape[1]; j++) {
        auto ind = Expr(std::make_shared<IndexExpression>(
            id_expr, ExprGroup(Expr(i), Expr(j)), index_expr->tb));
        ind.expr->ret_type = tensor_type->get_element_type();
        expanded_exprs.push_back(ind);
      }
    }
  }
  return expanded_exprs;
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

void flatten_lvalue(Expr expr, Expression::FlattenContext *ctx) {
  expr->flatten(ctx);
}

void flatten_global_load(Expr ptr, Expression::FlattenContext *ctx) {
  ctx->push_back(std::make_unique<GlobalLoadStmt>(ptr->stmt));
  ptr->stmt = ctx->back_stmt();
}

void flatten_local_load(Expr ptr, Expression::FlattenContext *ctx) {
  ctx->push_back<LocalLoadStmt>(ptr->stmt);
  ptr->stmt = ctx->back_stmt();
}

void flatten_rvalue(Expr ptr, Expression::FlattenContext *ctx) {
  ptr->flatten(ctx);
  if (ptr.is<IdExpression>()) {
    if (ptr->stmt->is<AllocaStmt>()) {
      flatten_local_load(ptr, ctx);
    }
  } else if (ptr.is<IndexExpression>()) {
    auto ix = ptr.cast<IndexExpression>();
    if (ix->is_local()) {
      flatten_local_load(ptr, ctx);
    } else {
      flatten_global_load(ptr, ctx);
    }
  } else if (ptr.is<StrideExpression>()) {
    flatten_global_load(ptr, ctx);
  } else if (ptr.is<ArgLoadExpression>() &&
             ptr.cast<ArgLoadExpression>()->is_ptr) {
    flatten_global_load(ptr, ctx);
  }
}

}  // namespace taichi::lang
