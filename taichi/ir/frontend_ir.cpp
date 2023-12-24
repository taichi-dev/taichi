#include "taichi/ir/frontend_ir.h"

#include "taichi/ir/expression_printer.h"
#include "taichi/ir/statements.h"
#include "taichi/program/program.h"
#include "taichi/common/exceptions.h"

#include <numeric>

namespace taichi::lang {

#define TI_ASSERT_TYPE_CHECKED(x)                                          \
  do {                                                                     \
    if (x->ret_type == PrimitiveType::unknown) {                           \
      ErrorEmitter(                                                        \
          TaichiTypeError(), x.expr.get(),                                 \
          fmt::format("[{}] was not type-checked",                         \
                      ExpressionHumanFriendlyPrinter::expr_to_string(x))); \
    }                                                                      \
  } while (false)

static bool is_primitive_or_tensor_type(DataType &type) {
  return type->is<PrimitiveType>() || type->is<TensorType>();
}

FrontendSNodeOpStmt::FrontendSNodeOpStmt(SNodeOpType op_type,
                                         SNode *snode,
                                         const ExprGroup &indices,
                                         const Expr &val,
                                         const DebugInfo &dbg_info)
    : Stmt(dbg_info),
      op_type(op_type),
      snode(snode),
      indices(indices),
      val(val) {
  if (val.expr != nullptr) {
    TI_ASSERT(op_type == SNodeOpType::append);
  } else {
    TI_ASSERT(op_type != SNodeOpType::append);
  }
}

FrontendReturnStmt::FrontendReturnStmt(const ExprGroup &group,
                                       const DebugInfo &dbg_info)
    : Stmt(dbg_info), values(group) {
}

FrontendAssignStmt::FrontendAssignStmt(const Expr &lhs,
                                       const Expr &rhs,
                                       const DebugInfo &dbg_info)
    : Stmt(dbg_info), lhs(lhs), rhs(rhs) {
  TI_ASSERT(lhs->is_lvalue());
  if (lhs.is<IdExpression>() && lhs->ret_type == PrimitiveType::unknown) {
    lhs.expr->ret_type =
        TypeFactory::get_instance().get_pointer_type(rhs.get_rvalue_type());
  }
}

FrontendIfStmt::FrontendIfStmt(const FrontendIfStmt &o)
    : Stmt(o.dbg_info),
      condition(o.condition),
      true_statements(o.true_statements->clone()),
      false_statements(o.false_statements->clone()) {
}

FrontendForStmt::FrontendForStmt(const ExprGroup &loop_vars,
                                 SNode *snode,
                                 Arch arch,
                                 const ForLoopConfig &config,
                                 const DebugInfo &dbg_info)
    : Stmt(dbg_info), snode(snode) {
  init_config(arch, config);
  init_loop_vars(loop_vars);
}

FrontendForStmt::FrontendForStmt(const ExprGroup &loop_vars,
                                 const Expr &external_tensor,
                                 Arch arch,
                                 const ForLoopConfig &config,
                                 const DebugInfo &dbg_info)
    : Stmt(dbg_info), external_tensor(external_tensor) {
  init_config(arch, config);
  init_loop_vars(loop_vars);
}

FrontendForStmt::FrontendForStmt(const ExprGroup &loop_vars,
                                 const mesh::MeshPtr &mesh,
                                 const mesh::MeshElementType &element_type,
                                 Arch arch,
                                 const ForLoopConfig &config,
                                 const DebugInfo &dbg_info)
    : Stmt(dbg_info), mesh(mesh.ptr.get()), element_type(element_type) {
  init_config(arch, config);
  init_loop_vars(loop_vars);
}

FrontendForStmt::FrontendForStmt(const Expr &loop_var,
                                 const Expr &begin,
                                 const Expr &end,
                                 Arch arch,
                                 const ForLoopConfig &config,
                                 const DebugInfo &dbg_info)
    : Stmt(dbg_info), begin(begin), end(end) {
  init_config(arch, config);
  add_loop_var(loop_var);
}

FrontendForStmt::FrontendForStmt(const FrontendForStmt &o)
    : Stmt(o.dbg_info),
      snode(o.snode),
      external_tensor(o.external_tensor),
      mesh(o.mesh),
      element_type(o.element_type),
      begin(o.begin),
      end(o.end),
      body(o.body->clone()),
      loop_var_ids(o.loop_var_ids),
      is_bit_vectorized(o.is_bit_vectorized),
      num_cpu_threads(o.num_cpu_threads),
      strictly_serialized(o.strictly_serialized),
      mem_access_opt(o.mem_access_opt),
      block_dim(o.block_dim) {
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
  loop_var.expr->ret_type =
      TypeFactory::get_instance().get_pointer_type(PrimitiveType::i32);
}

FrontendFuncDefStmt::FrontendFuncDefStmt(const FrontendFuncDefStmt &o)
    : funcid(o.funcid), body(o.body->clone()) {
}

FrontendWhileStmt::FrontendWhileStmt(const FrontendWhileStmt &o)
    : Stmt(o.dbg_info), cond(o.cond), body(o.body->clone()) {
}

void ArgLoadExpression::type_check(const CompileConfig *) {
  ret_type = dt;
  if (is_ptr) {
    ret_type = TypeFactory::get_instance().get_pointer_type(ret_type, false);
  }
  if (!create_load) {
    ret_type = TypeFactory::get_instance().get_pointer_type(ret_type, false);
  }
}

void ArgLoadExpression::flatten(FlattenContext *ctx) {
  auto arg_load = std::make_unique<ArgLoadStmt>(arg_id, dt, is_ptr, create_load,
                                                arg_depth, dbg_info);
  arg_load->ret_type = ret_type;
  ctx->push_back(std::move(arg_load));
  stmt = ctx->back_stmt();
}

void TexturePtrExpression::type_check(const CompileConfig *config) {
}

void TexturePtrExpression::flatten(FlattenContext *ctx) {
  ctx->push_back<ArgLoadStmt>(arg_id, PrimitiveType::f32, /*is_ptr=*/true,
                              /*create_load=*/true, /*arg_depth=*/arg_depth,
                              dbg_info);
  ctx->push_back<TexturePtrStmt>(ctx->back_stmt(), num_dims, is_storage, format,
                                 lod, dbg_info);
  stmt = ctx->back_stmt();
}

void RandExpression::type_check(const CompileConfig *) {
  if (!(dt->is<PrimitiveType>() && dt != PrimitiveType::unknown)) {
    ErrorEmitter(
        TaichiTypeError(), this,
        fmt::format("Invalid dt [{}] for RandExpression", dt->to_string()));
  }
  ret_type = dt;
}

void RandExpression::flatten(FlattenContext *ctx) {
  auto ran = std::make_unique<RandStmt>(dt, dbg_info);
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

  auto operand_type = operand.get_rvalue_type();
  auto operand_primitive_type = operand_type.get_element_type();
  auto ret_primitive_type = ret_type;

  if (!operand_primitive_type->is<PrimitiveType>()) {
    ErrorEmitter(TaichiTypeError(), this,
                 fmt::format("unsupported operand type(s) for '{}': '{}'",
                             unary_op_type_name(type),
                             operand_primitive_type->to_string()));
  }

  if ((type == UnaryOpType::round || type == UnaryOpType::floor ||
       type == UnaryOpType::ceil || is_trigonometric(type)) &&
      !is_real(operand_primitive_type))
    ErrorEmitter(
        TaichiTypeError(), this,
        fmt::format("'{}' takes real inputs only, however '{}' is provided",
                    unary_op_type_name(type),
                    operand_primitive_type->to_string()));

  if ((type == UnaryOpType::sqrt || type == UnaryOpType::exp ||
       type == UnaryOpType::log) &&
      !is_real(operand_primitive_type)) {
    ret_primitive_type = config->default_fp;
  } else {
    ret_primitive_type = is_cast() ? cast_type : operand_primitive_type;
  }

  if ((type == UnaryOpType::bit_not || type == UnaryOpType::logic_not) &&
      is_real(operand_primitive_type)) {
    ErrorEmitter(
        TaichiTypeError(), this,
        fmt::format("'{}' takes integral inputs only, however '{}' is provided",
                    unary_op_type_name(type),
                    operand_primitive_type->to_string()));
  }

  if (type == UnaryOpType::logic_not) {
    ret_primitive_type = PrimitiveType::u1;
  }

  if (type == UnaryOpType::frexp) {
    std::vector<AbstractDictionaryMember> elements;
    TI_ASSERT(operand_primitive_type->is_primitive(PrimitiveTypeID::f32) ||
              operand_primitive_type->is_primitive(PrimitiveTypeID::f64));
    elements.push_back({operand_primitive_type, "mantissa", 0});
    elements.push_back(
        {taichi::lang::TypeFactory::get_instance().get_primitive_int_type(
             32, /*is_signed=*/true),
         "exponent", (size_t)data_type_size(operand_primitive_type)});
    ret_type =
        taichi::lang::TypeFactory::get_instance().get_struct_type(elements);
    ret_type.set_is_pointer(true);
    return;
  }

  if (type == UnaryOpType::popcnt && is_real(operand_primitive_type)) {
    ErrorEmitter(
        TaichiTypeError(), this,
        fmt::format("'{}' takes integral inputs only, however '{}' is provided",
                    unary_op_type_name(type),
                    operand_primitive_type->to_string()));
  }

  if (operand_type->is<TensorType>()) {
    ret_type = taichi::lang::TypeFactory::get_instance().get_tensor_type(
        operand_type.get_shape(), ret_primitive_type);
  } else {
    TI_ASSERT(operand_type->is<PrimitiveType>());
    ret_type = ret_primitive_type;
  }
}

bool UnaryOpExpression::is_cast() const {
  return unary_op_is_cast(type);
}

void UnaryOpExpression::flatten(FlattenContext *ctx) {
  auto operand_stmt = flatten_rvalue(operand, ctx);
  auto unary =
      std::make_unique<UnaryOpStmt>(type, operand_stmt, operand_stmt->dbg_info);
  if (is_cast()) {
    unary->cast_type = cast_type;
  }
  stmt = unary.get();
  stmt->ret_type = ret_type;
  ctx->push_back(std::move(unary));
}

Expr to_broadcast_tensor(const Expr &elt, const DataType &dt) {
  auto elt_type = elt.get_rvalue_type();
  if (!elt_type->is<TensorType>() && !dt->is<TensorType>())
    return elt;

  if (elt_type->is<TensorType>() && dt->is<TensorType>()) {
    // Only tensor shape will be checked here, since the dtype will
    // be promoted later at irpass::type_check()
    if (elt_type.get_shape() != dt.get_shape()) {
      ErrorEmitter(TaichiTypeError(), elt.expr.get(),
                   "Cannot broadcast tensor to tensor");
    } else {
      return elt;
    }
  }

  auto tensor_type = dt->as<TensorType>();
  auto tensor_elt_type = tensor_type->get_element_type();
  if (!tensor_elt_type->is<PrimitiveType>()) {
    ErrorEmitter(
        TaichiTypeError(), elt.expr.get(),
        fmt::format("Only primitive types are supported in Tensors, got {}",
                    tensor_elt_type->to_string()));
  }
  std::vector<Expr> broadcast_values(tensor_type->get_num_elements(), elt);
  auto matrix_expr = Expr::make<MatrixExpression>(
      broadcast_values, tensor_type->get_shape(), elt_type, elt->dbg_info);
  matrix_expr->type_check(nullptr);
  return matrix_expr;
}

std::tuple<Expr, Expr> unify_binop_operands(const Expr &e1, const Expr &e2) {
  auto e1_type = e1.get_rvalue_type();
  auto e2_type = e2.get_rvalue_type();
  if (e1_type->is<PrimitiveType>() && e2_type->is<TensorType>()) {
    return std::tuple(to_broadcast_tensor(e1, e2_type), e2);
  } else if (e1_type->is<TensorType>() && e2_type->is<PrimitiveType>()) {
    return std::tuple(e1, to_broadcast_tensor(e2, e1_type));
  } else {
    return std::tuple(e1, e2);
  }
}

void BinaryOpExpression::type_check(const CompileConfig *config) {
  TI_ASSERT_TYPE_CHECKED(lhs);
  TI_ASSERT_TYPE_CHECKED(rhs);

  auto lhs_type = lhs.get_rvalue_type();
  auto rhs_type = rhs.get_rvalue_type();
  auto error = [&]() {
    throw TaichiTypeError(
        fmt::format("unsupported operand type(s) for '{}': '{}' and '{}'",
                    binary_op_type_symbol(type), lhs_type->to_string(),
                    rhs_type->to_string()));
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
    if (lhs_type == PrimitiveType::unknown)
      lhs.type_check(config);
    if (rhs_type == PrimitiveType::unknown)
      rhs.type_check(config);
    lhs_type = lhs.get_rvalue_type();
    rhs_type = rhs.get_rvalue_type();
    TI_ASSERT(lhs_type->is<TensorType>());
    TI_ASSERT(rhs_type->is<TensorType>());
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

  auto make_dt = [&is_tensor_op, lhs_type](DataType dt) {
    if (is_tensor_op) {
      return TypeFactory::create_tensor_type(
          lhs_type->cast<TensorType>()->get_shape(), dt);
    } else {
      return dt;
    }
  };

  if (binary_is_bitwise(type) && (!is_integral(lhs_type.get_element_type()) ||
                                  !is_integral(rhs_type.get_element_type())))
    error();
  if (binary_is_logical(type) && !(is_integral(lhs_type.get_element_type()) &&
                                   is_integral(rhs_type.get_element_type())))
    error();
  if (is_comparison(type)) {
    ret_type = make_dt(PrimitiveType::u1);
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

  auto lhs_type = lhs.get_rvalue_type();
  auto rhs_type = rhs.get_rvalue_type();

  if (binary_is_logical(type) && !is_tensor(lhs_type) && !is_tensor(rhs_type)) {
    auto result = ctx->push_back<AllocaStmt>(ret_type, dbg_info);
    ctx->push_back<LocalStoreStmt>(result, lhs_stmt, lhs_stmt->dbg_info);
    auto cond = ctx->push_back<LocalLoadStmt>(result, dbg_info);
    auto if_stmt = ctx->push_back<IfStmt>(cond, dbg_info);

    FlattenContext rctx;
    rctx.current_block = ctx->current_block;
    auto rhs_stmt = flatten_rvalue(rhs, &rctx);
    rctx.push_back<LocalStoreStmt>(result, rhs_stmt, rhs_stmt->dbg_info);

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

    auto ret = ctx->push_back<LocalLoadStmt>(result, dbg_info);
    stmt = ret;
    stmt->ret_type = ret_type;
    return;
  }
  auto rhs_stmt = flatten_rvalue(rhs, ctx);
  ctx->push_back(std::make_unique<BinaryOpStmt>(
      type, lhs_stmt, rhs_stmt, /*is_bit_vectorized=*/false, dbg_info));
  stmt = ctx->back_stmt();
  stmt->ret_type = ret_type;
}

void make_ifte(Expression::FlattenContext *ctx,
               DataType ret_type,
               Expr cond,
               Expr true_val,
               Expr false_val,
               const DebugInfo &dbg_info) {
  auto result = ctx->push_back<AllocaStmt>(ret_type, dbg_info);
  auto cond_stmt = flatten_rvalue(cond, ctx);
  auto if_stmt = ctx->push_back<IfStmt>(cond_stmt, cond->dbg_info);

  Expression::FlattenContext lctx;
  lctx.current_block = ctx->current_block;
  auto true_val_stmt = flatten_rvalue(true_val, &lctx);
  lctx.push_back<LocalStoreStmt>(result, true_val_stmt, true_val->dbg_info);

  Expression::FlattenContext rctx;
  rctx.current_block = ctx->current_block;
  auto false_val_stmt = flatten_rvalue(false_val, &rctx);
  rctx.push_back<LocalStoreStmt>(result, false_val_stmt, false_val->dbg_info);

  auto true_block = std::make_unique<Block>();
  true_block->set_statements(std::move(lctx.stmts));
  if_stmt->set_true_statements(std::move(true_block));

  auto false_block = std::make_unique<Block>();
  false_block->set_statements(std::move(rctx.stmts));
  if_stmt->set_false_statements(std::move(false_block));

  ctx->push_back<LocalLoadStmt>(result, dbg_info);
  return;
}

static std::tuple<Expr, Expr, Expr> unify_ternaryop_operands(const Expr &e1,
                                                             const Expr &e2,
                                                             const Expr &e3) {
  auto target_dtype = PrimitiveType::unknown;
  // Since we don't support broadcasting between two TensorTypes,
  // we can simply use the first TensorType's dtype as the target dtype.
  auto e1_type = e1.get_rvalue_type();
  auto e2_type = e2.get_rvalue_type();
  auto e3_type = e3.get_rvalue_type();
  if (e1_type->is<TensorType>()) {
    target_dtype = e1_type;
  } else if (e2_type->is<TensorType>()) {
    target_dtype = e2_type;
  } else if (e3_type->is<TensorType>()) {
    target_dtype = e3_type;
  }

  if (target_dtype == PrimitiveType::unknown) {
    return std::tuple(e1, e2, e3);
  }

  return std::tuple(e1, to_broadcast_tensor(e2, target_dtype),
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
  auto op1_type = op1.get_rvalue_type();
  auto op2_type = op2.get_rvalue_type();
  auto op3_type = op3.get_rvalue_type();

  auto error = [&]() {
    ErrorEmitter(
        TaichiTypeError(), this,
        fmt::format("unsupported operand type(s) for '{}': '{}', '{}' and '{}'",
                    ternary_type_name(type), op1_type->to_string(),
                    op2_type->to_string(), op3_type->to_string()));
  };
  std::vector<int> shape;
  if (op2_type->is<TensorType>() && op3_type->is<TensorType>()) {
    // valid
    is_tensor = true;
    if (op1_type->is<TensorType>() &&
        op1_type->cast<TensorType>()->get_shape() !=
            op2_type->cast<TensorType>()->get_shape()) {
      is_valid = false;
    }
    if (op2_type->cast<TensorType>()->get_shape() !=
        op3_type->cast<TensorType>()->get_shape()) {
      is_valid = false;
    }

    if (op1_type->is<TensorType>()) {
      op1_type = op1_type->cast<TensorType>()->get_element_type();
    }
    shape = op2_type->cast<TensorType>()->get_shape();
    op2_type = op2_type->cast<TensorType>()->get_element_type();
    op3_type = op3_type->cast<TensorType>()->get_element_type();

  } else if (op1_type->is<PrimitiveType>() && op2_type->is<PrimitiveType>() &&
             op3_type->is<PrimitiveType>()) {
    // valid
  } else {
    is_valid = false;
  }

  if (!is_integral(op1_type)) {
    is_valid = false;
  }
  if (!op2_type->is<PrimitiveType>() || !op3_type->is<PrimitiveType>()) {
    is_valid = false;
  }

  if (!is_valid)
    error();

  if (is_tensor) {
    auto primitive_dtype = promoted_type(op2_type, op3_type);
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
    ctx->push_back(std::make_unique<TernaryOpStmt>(type, op1_stmt, op2_stmt,
                                                   op3_stmt, dbg_info));
  } else if (type == TernaryOpType::ifte) {
    make_ifte(ctx, ret_type, op1, op2, op3, dbg_info);
  }
  stmt = ctx->back_stmt();
  stmt->ret_type = ret_type;
}

void InternalFuncCallExpression::type_check(const CompileConfig *) {
  std::vector<DataType> arg_types;
  for (auto &arg : args) {
    TI_ASSERT_TYPE_CHECKED(arg);
    arg_types.push_back(arg.get_rvalue_type());
  }
  ret_type = op->type_check(arg_types);
}

void InternalFuncCallExpression::flatten(FlattenContext *ctx) {
  stmt = op->flatten(ctx, args, ret_type);
  stmt->dbg_info = dbg_info;
}

void ExternalTensorExpression::flatten(FlattenContext *ctx) {
  auto type =
      TypeFactory::get_instance().get_ndarray_struct_type(dt, ndim, needs_grad);
  type = TypeFactory::get_instance().get_pointer_type((Type *)type);

  auto ptr = Stmt::make<ArgLoadStmt>(
      arg_id, type, /*is_ptr=*/true,
      /*create_load=*/false, /*arg_depth=*/arg_depth, /*dbg_info=*/dbg_info);

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
  ret_type.set_is_pointer(true);
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
  // FIXME: No need to make it negative since we only support AOS
  auto element_dim = -expr->dt.get_shape().size();
  auto external_ptr_stmt = std::make_unique<ExternalPtrStmt>(
      var_stmt, index_stmts, indices.size(), expr->dt.get_shape(),
      expr->is_grad, expr->boundary);
  if (expr->ndim - element_dim == indices.size()) {
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
                                        const DebugInfo &dbg_info) {
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
  return ctx->push_back<MatrixPtrStmt>(var_stmt, offset_stmt, dbg_info);
}

Stmt *make_tensor_access(Expression::FlattenContext *ctx,
                         Expr var,
                         const std::vector<ExprGroup> &indices_group,
                         DataType ret_type,
                         std::vector<int> shape,
                         const DebugInfo &dbg_info) {
  auto var_stmt = flatten_lvalue(var, ctx);
  if (!var->is_lvalue()) {
    auto alloca_stmt = ctx->push_back<AllocaStmt>(var.get_rvalue_type());
    ctx->push_back<LocalStoreStmt>(alloca_stmt, var_stmt);
    var_stmt = alloca_stmt;
  }

  bool is_shared_array =
      (var_stmt->is<AllocaStmt>() && var_stmt->as<AllocaStmt>()->is_shared);

  if (ret_type.ptr_removed()->is<TensorType>() && !is_shared_array) {
    std::vector<Stmt *> stmts;
    for (auto &indices : indices_group) {
      stmts.push_back(make_tensor_access_single_element(ctx, var_stmt, indices,
                                                        shape, dbg_info));
    }
    return ctx->push_back<MatrixOfMatrixPtrStmt>(stmts, ret_type);
  }
  return make_tensor_access_single_element(ctx, var_stmt, indices_group[0],
                                           shape, dbg_info);
}

void MatrixExpression::type_check(const CompileConfig *config) {
  auto tensor_type = dt->as<TensorType>();
  TI_ASSERT(tensor_type->get_num_elements() == elements.size());

  for (auto &arg : elements) {
    TI_ASSERT_TYPE_CHECKED(arg);
    if (arg.get_rvalue_type()->get_type() != tensor_type->get_element_type()) {
      arg = cast(arg, tensor_type->get_element_type());
      arg->type_check(config);
    }
  }
  ret_type = dt;
}

void MatrixExpression::flatten(FlattenContext *ctx) {
  TI_ASSERT(dt->is<TensorType>());
  std::vector<Stmt *> values;
  for (auto &elt : elements) {
    values.push_back(flatten_rvalue(elt, ctx));
  }
  stmt = ctx->push_back<MatrixInitStmt>(values);
  stmt->ret_type = dt;
}

IndexExpression::IndexExpression(const Expr &var,
                                 const ExprGroup &indices,
                                 const DebugInfo &dbg_info)
    : Expression(dbg_info), var(var), indices_group({indices}) {
}

IndexExpression::IndexExpression(const Expr &var,
                                 const std::vector<ExprGroup> &indices_group,
                                 const std::vector<int> &ret_shape,
                                 const DebugInfo &dbg_info)
    : Expression(dbg_info),
      var(var),
      indices_group(indices_group),
      ret_shape(ret_shape) {
  // IndexExpression with ret_shape is used for matrix slicing, where each entry
  // of ExprGroup is interpreted as a group of indices to return within each
  // axis. For example, mat[0, 3:5] has indices_group={0, [3, 4]}, where [3, 4]
  // means "m"-axis will return a TensorType with size of 2. In this case, we
  // should not expand indices_group due to its special semantics.
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
  return var->ret_type.ptr_removed()->is<TensorType>();
}

bool IndexExpression::is_local() const {
  return !is_global();
}

bool IndexExpression::is_global() const {
  if (var.is<IndexExpression>()) {
    // Special case: Pointer chasing. For example, if we are indexing into
    // tensor elements of fields / ndarrays, this index expr should be treated
    // as global.
    return var.cast<IndexExpression>()->is_global();
  }

  // Only Ndarray and Field comes outside from a kernel
  return is_field() || is_matrix_field() || is_ndarray();
}

static void field_validation(FieldExpression *field_expr, int index_dim) {
  TI_ASSERT(field_expr != nullptr);
  TI_ASSERT(field_expr->snode != nullptr);
  int field_dim = field_expr->snode->num_active_indices;

  if (field_dim != index_dim) {
    ErrorEmitter(
        TaichiIndexError(), field_expr,
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
  auto var_type = var.get_rvalue_type();
  if (has_slice) {
    if (!is_tensor()) {
      ErrorEmitter(TaichiTypeError(), this,
                   "Slice or swizzle can only apply on matrices");
    }
    auto element_type = var_type->as<TensorType>()->get_element_type();
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
    int ndim = external_tensor_expr->ndim;
    int element_dim = external_tensor_expr->dt.get_shape().size();
    int total_dim = ndim + element_dim;
    if (total_dim != index_dim + element_dim) {
      ErrorEmitter(
          TaichiIndexError(), this,
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
    auto tensor_type = var_type->as<TensorType>();
    auto shape = tensor_type->get_shape();
    if (indices_group[0].size() != shape.size()) {
      ErrorEmitter(TaichiIndexError(), this,
                   fmt::format("Expected {} indices, got {}.", shape.size(),
                               indices_group[0].size()));
    }
    ret_type = tensor_type->get_element_type();
  } else {
    ErrorEmitter(
        TaichiIndexError(), this,
        "Invalid IndexExpression: the source is not among field, ndarray or "
        "local tensor");
  }
  ret_type = TypeFactory::get_instance().get_pointer_type(ret_type);
  for (auto &indices : indices_group) {
    for (int i = 0; i < indices.exprs.size(); i++) {
      auto &expr = indices.exprs[i];
      TI_ASSERT_TYPE_CHECKED(expr);
      auto expr_type = expr.get_rvalue_type();
      if (!is_integral(expr_type))
        ErrorEmitter(TaichiTypeError(), this,
                     fmt::format("indices must be integers, however '{}' is "
                                 "provided as index {}",
                                 expr_type->to_string(), i));
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
    stmt = make_tensor_access(
        ctx, var, indices_group, ret_type,
        var->ret_type.ptr_removed()->as<TensorType>()->get_shape(), dbg_info);
  } else {
    ErrorEmitter(
        TaichiIndexError(), this,
        "Invalid IndexExpression: the source is not among field, ndarray or "
        "local tensor");
  }
  stmt->dbg_info = dbg_info;
}

void RangeAssumptionExpression::type_check(const CompileConfig *) {
  TI_ASSERT_TYPE_CHECKED(input);
  TI_ASSERT_TYPE_CHECKED(base);
  auto input_type = input.get_rvalue_type();
  auto base_type = base.get_rvalue_type();
  if (!input_type->is<PrimitiveType>() || !base_type->is<PrimitiveType>() ||
      input_type != base_type)
    ErrorEmitter(TaichiTypeError(), this,
                 fmt::format("unsupported operand type(s) for "
                             "'range_assumption': '{}' and '{}'",
                             input_type->to_string(), base_type->to_string()));
  ret_type = input_type;
}

void RangeAssumptionExpression::flatten(FlattenContext *ctx) {
  auto input_stmt = flatten_rvalue(input, ctx);
  auto base_stmt = flatten_rvalue(base, ctx);
  ctx->push_back(Stmt::make<RangeAssumptionStmt>(input_stmt, base_stmt, low,
                                                 high, dbg_info));
  stmt = ctx->back_stmt();
}

void LoopUniqueExpression::type_check(const CompileConfig *) {
  TI_ASSERT_TYPE_CHECKED(input);
  auto input_type = input.get_rvalue_type();

  if (!input_type->is<PrimitiveType>())
    ErrorEmitter(
        TaichiTypeError(), this,
        fmt::format("unsupported operand type(s) for 'loop_unique': '{}'",
                    input_type->to_string()));
  ret_type = input_type;
}

void LoopUniqueExpression::flatten(FlattenContext *ctx) {
  auto input_stmt = flatten_rvalue(input, ctx);
  ctx->push_back(Stmt::make<LoopUniqueStmt>(input_stmt, covers, dbg_info));
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
    ErrorEmitter(
        TaichiTypeError(), this,
        fmt::format(
            "unsupported operand type(s) for 'atomic_{}': '{}' and '{}'",
            atomic_op_type_name(op_type), dest->ret_type->to_string(),
            val->ret_type->to_string()));
  };

  // Broadcast val to dest if neccessary
  auto val_dtype = val.get_rvalue_type();
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
  if (val_dtype->is<TensorType>()) {
    val_dtype = val_dtype.get_element_type();
  }

  if (!val_dtype->is<PrimitiveType>()) {
    error();
  }

  if (is_quant(dest_dtype)) {
    ret_type = dest_dtype->get_compute_type();
  } else if (dest_dtype->is<PrimitiveType>() || dest_dtype->is<TensorType>()) {
    ret_type = dest_dtype;
  } else {
    error();
  }

  auto const &ret_element_type = ret_type.get_element_type();
  if (ret_element_type != val_dtype) {
    auto promoted = promoted_type(ret_element_type, val_dtype);
    if (ret_element_type != promoted) {
      ErrorEmitter(
          TaichiCastWarning(), this,
          fmt::format("Atomic {} may lose precision: {} <- {}",
                      atomic_op_type_name(op_type),
                      ret_element_type->to_string(), val_dtype->to_string()));
    }
  }
}

void AtomicOpExpression::flatten(FlattenContext *ctx) {
  TI_ASSERT(dest.expr->is_lvalue());
  // replace atomic sub with negative atomic add
  if (op_type == AtomicOpType::sub) {
    if (val->ret_type != ret_type) {
      val.set(Expr::make<UnaryOpExpression>(UnaryOpType::cast_value, val,
                                            ret_type, val->dbg_info));
    }

    val.set(
        Expr::make<UnaryOpExpression>(UnaryOpType::neg, val, val->dbg_info));
    op_type = AtomicOpType::add;
  }
  // expand rhs
  auto val_stmt = flatten_rvalue(val, ctx);
  auto dest_stmt = flatten_lvalue(dest, ctx);
  stmt = ctx->push_back<AtomicOpStmt>(op_type, dest_stmt, val_stmt, dbg_info);
  stmt->ret_type = stmt->as<AtomicOpStmt>()->dest->ret_type;
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
  } else if (op_type == SNodeOpType::is_active) {
    ret_type = PrimitiveType::u1;
  } else {
    ret_type = PrimitiveType::i32;
  }
  if (op_type == SNodeOpType::append) {
    TI_ASSERT(snode->ch.size() == values.size());
    for (int i = 0; i < values.size(); i++) {
      TI_ASSERT_TYPE_CHECKED(values[i]);
      auto &dst_type = snode->ch[i]->dt;
      auto value_type = values[i].get_rvalue_type();
      auto promoted = promoted_type(dst_type, value_type);
      if (dst_type != promoted) {
        ErrorEmitter(
            TaichiCastWarning(), this,
            fmt::format("Append may lose precision: {} <- {}",
                        dst_type->to_string(), value_type->to_string()));
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
  auto ptr = ctx->push_back<GlobalPtrStmt>(snode, indices_stmt, true,
                                           is_cell_access, dbg_info);
  if (op_type == SNodeOpType::is_active) {
    if (!(snode->type == SNodeType::pointer || snode->type == SNodeType::hash ||
          snode->type == SNodeType::bitmasked)) {
      ErrorEmitter(
          TaichiTypeError(), this,
          "ti.is_active only works on pointer, hash or bitmasked nodes.");
    }
    ctx->push_back<SNodeOpStmt>(SNodeOpType::is_active, snode, ptr, nullptr,
                                dbg_info);
  } else if (op_type == SNodeOpType::length) {
    ctx->push_back<SNodeOpStmt>(SNodeOpType::length, snode, ptr, nullptr,
                                dbg_info);
  } else if (op_type == SNodeOpType::get_addr) {
    ctx->push_back<SNodeOpStmt>(SNodeOpType::get_addr, snode, ptr, nullptr,
                                dbg_info);
  } else if (op_type == SNodeOpType::append) {
    auto alloca = ctx->push_back<AllocaStmt>(PrimitiveType::i32, dbg_info);
    auto addr = ctx->push_back<SNodeOpStmt>(SNodeOpType::allocate, snode, ptr,
                                            alloca, dbg_info);
    for (int i = 0; i < values.size(); i++) {
      auto value_stmt = flatten_rvalue(values[i], ctx);
      auto ch_addr = ctx->push_back<GetChStmt>(
          addr, snode, i, /*is_bit_vectorized = */ false, dbg_info);
      ctx->push_back<GlobalStoreStmt>(ch_addr, value_stmt, dbg_info);
    }
    ctx->push_back<LocalLoadStmt>(alloca, dbg_info);
    if (snode->type != SNodeType::dynamic) {
      ErrorEmitter(TaichiTypeError(), this,
                   "ti.append only works on dynamic nodes.");
    }
  }
  stmt = ctx->back_stmt();
}

TextureOpExpression::TextureOpExpression(TextureOpType op,
                                         Expr texture_ptr,
                                         const ExprGroup &args,
                                         const DebugInfo &dbg_info)
    : Expression(dbg_info), op(op), texture_ptr(texture_ptr), args(args) {
}

void TextureOpExpression::type_check(const CompileConfig *config) {
  TI_ASSERT(texture_ptr.is<TexturePtrExpression>());
  auto ptr = texture_ptr.cast<TexturePtrExpression>();
  if (op == TextureOpType::kSampleLod) {
    // UV, Lod
    if (args.size() != ptr->num_dims + 1) {
      ErrorEmitter(TaichiTypeError(), this,
                   fmt::format("Invalid number of args for sample_lod Texture "
                               "op with a {}-dimension texture",
                               ptr->num_dims));
    }
    for (int i = 0; i < ptr->num_dims; i++) {
      TI_ASSERT_TYPE_CHECKED(args[i]);
      auto arg_type = args[i].get_rvalue_type();
      if (arg_type != PrimitiveType::f32) {
        ErrorEmitter(
            TaichiTypeError(), this,
            fmt::format("Invalid type for texture sample_lod: '{}', all "
                        "arguments must be f32",
                        arg_type->to_string()));
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
      auto arg_type = args[i].get_rvalue_type();
      if (arg_type != PrimitiveType::i32) {
        ErrorEmitter(
            TaichiTypeError(), this,
            fmt::format("Invalid type for texture fetch_texel: '{}', all "
                        "arguments must be i32",
                        arg_type->to_string()));
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
      auto arg_type = args[i].get_rvalue_type();
      if (arg_type != PrimitiveType::i32) {
        ErrorEmitter(TaichiTypeError(), this,
                     fmt::format("Invalid type for texture load: '{}', all "
                                 "arguments must be i32",
                                 arg_type->to_string()));
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
      auto arg_type = args[i].get_rvalue_type();
      if (arg_type != PrimitiveType::i32) {
        ErrorEmitter(TaichiTypeError(), this,
                     fmt::format("Invalid type for texture load: '{}', index "
                                 "arguments must be i32",
                                 arg_type->to_string()));
      }
    }
    for (int i = ptr->num_dims; i < ptr->num_dims + 4; i++) {
      TI_ASSERT_TYPE_CHECKED(args[i]);
      auto arg_type = args[i].get_rvalue_type();
      if (arg_type != PrimitiveType::f32) {
        ErrorEmitter(TaichiTypeError(), this,
                     fmt::format("Invalid type for texture load: '{}', value "
                                 "arguments must be f32",
                                 arg_type->to_string()));
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
  ctx->push_back<TextureOpStmt>(op, texture_ptr_stmt, arg_stmts, dbg_info);
  stmt = ctx->back_stmt();
}

void ConstExpression::type_check(const CompileConfig *) {
  if (!(val.dt->is<PrimitiveType>() && val.dt != PrimitiveType::unknown)) {
    ErrorEmitter(TaichiTypeError(), this,
                 fmt::format("Invalid dt [{}] for ConstExpression",
                             val.dt->to_string()));
  }
  ret_type = val.dt;
}

void ConstExpression::flatten(FlattenContext *ctx) {
  ctx->push_back(Stmt::make<ConstStmt>(val, dbg_info));
  stmt = ctx->back_stmt();
}

void ExternalTensorShapeAlongAxisExpression::type_check(const CompileConfig *) {
  if (!(ptr.is<ExternalTensorExpression>() || ptr.is<TexturePtrExpression>())) {
    ErrorEmitter(
        TaichiTypeError(), this,
        fmt::format(
            "Invalid ptr [{}] for ExternalTensorShapeAlongAxisExpression",
            ExpressionHumanFriendlyPrinter::expr_to_string(ptr)));
  }
  ret_type = PrimitiveType::i32;
}

void ExternalTensorShapeAlongAxisExpression::flatten(FlattenContext *ctx) {
  auto temp = ptr.cast<ExternalTensorExpression>();
  TI_ASSERT(0 <= axis && axis < temp->ndim);
  ctx->push_back<ExternalTensorShapeAlongAxisStmt>(axis, temp->arg_id,
                                                   dbg_info);
  stmt = ctx->back_stmt();
}

void ExternalTensorBasePtrExpression::type_check(const CompileConfig *) {
  TI_ASSERT_INFO(ptr.is<ExternalTensorExpression>(),
                 "Invalid ptr [{}] for ExternalTensorBasePtrExpression",
                 ExpressionHumanFriendlyPrinter::expr_to_string(ptr));
  ret_type = ptr.cast<ExternalTensorExpression>()->dt.get_element_type();
  ret_type.set_is_pointer(true);
}

void ExternalTensorBasePtrExpression::flatten(FlattenContext *ctx) {
  auto tensor = ptr.cast<ExternalTensorExpression>();
  ctx->push_back<ExternalTensorBasePtrStmt>(tensor->arg_id, is_grad, dbg_info);
  stmt = ctx->back_stmt();
  stmt->ret_type = ret_type;
}

void GetElementExpression::type_check(const CompileConfig *config) {
  TI_ASSERT_TYPE_CHECKED(src);
  auto src_type = src->ret_type;
  if (!src_type->is<PointerType>()) {
    ErrorEmitter(
        TaichiTypeError(), this,
        fmt::format("Invalid src [{}] for GetElementExpression",
                    ExpressionHumanFriendlyPrinter::expr_to_string(src)));
  }

  ret_type = src_type.ptr_removed()->as<StructType>()->get_element_type(index);
}

void GetElementExpression::flatten(FlattenContext *ctx) {
  ctx->push_back<GetElementStmt>(flatten_rvalue(src, ctx), index, dbg_info);
  stmt = ctx->back_stmt();
}
// Mesh related.

void MeshPatchIndexExpression::flatten(FlattenContext *ctx) {
  auto pid_stmt = std::make_unique<MeshPatchIndexStmt>(dbg_info);
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
                                           neighbor_idx_stmt, dbg_info);
  } else {
    ctx->push_back<MeshRelationAccessStmt>(mesh, mesh_idx_stmt, to_type,
                                           dbg_info);
  }
  stmt = ctx->back_stmt();
}

MeshIndexConversionExpression::MeshIndexConversionExpression(
    mesh::Mesh *mesh,
    mesh::MeshElementType idx_type,
    const Expr idx,
    mesh::ConvType conv_type,
    const DebugInfo &dbg_info)
    : Expression(dbg_info),
      mesh(mesh),
      idx_type(idx_type),
      idx(idx),
      conv_type(conv_type) {
}

void MeshIndexConversionExpression::type_check(const CompileConfig *) {
  ret_type = PrimitiveType::i32;
}

void MeshIndexConversionExpression::flatten(FlattenContext *ctx) {
  auto idx_stmt = flatten_rvalue(idx, ctx);
  ctx->push_back<MeshIndexConversionStmt>(mesh, idx_type, idx_stmt, conv_type,
                                          dbg_info);
  stmt = ctx->back_stmt();
}

void ReferenceExpression::type_check(const CompileConfig *) {
  ret_type = TypeFactory::get_instance().get_pointer_type(var->ret_type);
}

void ReferenceExpression::flatten(FlattenContext *ctx) {
  auto var_stmt = flatten_lvalue(var, ctx);
  ctx->push_back<ReferenceStmt>(var_stmt, dbg_info);
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
                                   const DebugInfo &dbg_info) {
  // Inside a kernel or a function
  // Create an assignment in the IR
  if (lhs.expr == nullptr) {
    lhs.set(rhs);
  } else if (lhs.expr->is_lvalue()) {
    auto stmt = std::make_unique<FrontendAssignStmt>(lhs, rhs, dbg_info);
    this->insert(std::move(stmt));

  } else {
    ErrorEmitter(
        TaichiRuntimeError(), lhs.expr.get(),
        fmt::format("Cannot assign to non-lvalue: {}",
                    ExpressionHumanFriendlyPrinter::expr_to_string(lhs)));
  }
}

Expr ASTBuilder::make_var(const Expr &x, const DebugInfo &dbg_info) {
  auto var = this->expr_alloca(dbg_info);
  this->insert_assignment(var, x, dbg_info);
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
  auto loop = stack_.size() ? stack_.back()->parent_stmt() : nullptr;
  TI_ERROR_IF(
      arch_ != Arch::cuda && !arch_is_cpu(arch_) && arch_ != Arch::amdgpu,
      "ti.thread_idx() is only available in cuda or cpu or amdgpu context.");
  if (loop != nullptr) {
    auto i = stack_.size() - 1;
    while (!(loop->is<FrontendForStmt>())) {
      loop = i > 0 ? stack_[--i]->parent_stmt() : nullptr;
      if (loop == nullptr)
        break;
    }
  }
  TI_ERROR_IF(!(loop && loop->is<FrontendForStmt>()),
              "ti.thread_idx() is only valid within loops.");
  return Expr::make<InternalFuncCallExpression>(
      Operations::get(InternalOp::linear_thread_idx), std::vector<Expr>{});
}

Expr ASTBuilder::insert_patch_idx_expr(const DebugInfo &dbg_info) {
  auto loop = stack_.size() ? stack_.back()->parent_stmt() : nullptr;
  if (loop != nullptr) {
    auto i = stack_.size() - 1;
    while (!(loop->is<FrontendForStmt>())) {
      loop = i > 0 ? stack_[--i]->parent_stmt() : nullptr;
      if (loop == nullptr)
        break;
    }
  }
  TI_ERROR_IF(!(loop && loop->is<FrontendForStmt>() &&
                loop->as<FrontendForStmt>()->mesh),
              "ti.mesh_patch_idx() is only valid within mesh-for loops.");
  return Expr::make<MeshPatchIndexExpression>(dbg_info);
}

void ASTBuilder::create_kernel_exprgroup_return(const ExprGroup &group,
                                                const DebugInfo &dbg_info) {
  auto expanded_exprs = this->expand_exprs(group.exprs);
  ExprGroup expanded_expr_group;
  expanded_expr_group.exprs = std::move(expanded_exprs);
  this->insert(Stmt::make<FrontendReturnStmt>(expanded_expr_group, dbg_info));
}

void ASTBuilder::create_print(
    std::vector<std::variant<Expr, std::string>> contents,
    std::vector<std::optional<std::string>> formats,
    const DebugInfo &dbg_info) {
  this->insert(
      std::make_unique<FrontendPrintStmt>(contents, formats, dbg_info));
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

void ASTBuilder::begin_frontend_if(const Expr &cond,
                                   const DebugInfo &dbg_info) {
  auto stmt_tmp = std::make_unique<FrontendIfStmt>(cond, dbg_info);
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
                                           const ExprGroup &outputs,
                                           const DebugInfo &dbg_info) {
  auto stmt = Stmt::make<FrontendExternalFuncStmt>(
      (void *)func_addr, source, filename, funcname, args.exprs, outputs.exprs,
      dbg_info);
  this->insert(std::move(stmt));
}

Expr ASTBuilder::expr_alloca(const DebugInfo &dbg_info) {
  auto var = Expr(std::make_shared<IdExpression>(get_next_id()));
  this->insert(std::make_unique<FrontendAllocaStmt>(
      std::static_pointer_cast<IdExpression>(var.expr)->id,
      PrimitiveType::unknown, dbg_info));
  return var;
}

std::optional<Expr> ASTBuilder::insert_func_call(Function *func,
                                                 const ExprGroup &args,
                                                 const DebugInfo &dbg_info) {
  ExprGroup expanded_args;
  expanded_args.exprs = this->expand_exprs(args.exprs);
  if (!func->rets.empty()) {
    auto var = Expr(std::make_shared<IdExpression>(get_next_id()));
    this->insert(std::make_unique<FrontendFuncCallStmt>(
        func, expanded_args,
        std::static_pointer_cast<IdExpression>(var.expr)->id, dbg_info));
    var.expr->ret_type = func->ret_type;
    var.expr->ret_type.set_is_pointer(true);
    return var;
  } else {
    this->insert(std::make_unique<FrontendFuncCallStmt>(
        func, expanded_args, /*id=*/std::nullopt, dbg_info));
    return std::nullopt;
  }
}

Expr ASTBuilder::make_matrix_expr(const std::vector<int> &shape,
                                  const DataType &dt,
                                  const std::vector<Expr> &elements,
                                  const DebugInfo &dbg_info) {
  /*
    Since we have both "shape" and "element_type" in MatrixExpression,
    we should flatten all the elements and disallow recursive TensorType in
    element Expr
  */
  TI_ASSERT(dt->is<PrimitiveType>());
  auto expanded_elements = this->expand_exprs(elements);
  auto mat = Expr(std::make_shared<MatrixExpression>(expanded_elements, shape,
                                                     dt, dbg_info));
  return mat;
}

Expr ASTBuilder::expr_alloca_shared_array(const std::vector<int> &shape,
                                          const DataType &element_type,
                                          const DebugInfo &dbg_info) {
  auto var = Expr(std::make_shared<IdExpression>(get_next_id()));
  this->insert(std::make_unique<FrontendAllocaStmt>(
      std::static_pointer_cast<IdExpression>(var.expr)->id, shape, element_type,
      true, dbg_info));
  var->ret_type = this->get_last_stmt()->ret_type;
  return var;
}

void ASTBuilder::expr_assign(const Expr &lhs,
                             const Expr &rhs,
                             const DebugInfo &dbg_info) {
  TI_ASSERT(lhs->is_lvalue());
  auto stmt = std::make_unique<FrontendAssignStmt>(lhs, rhs, dbg_info);
  this->insert(std::move(stmt));
}

Expr ASTBuilder::expr_subscript(const Expr &expr,
                                const ExprGroup &indices,
                                const DebugInfo &dbg_info) {
  TI_ASSERT(expr.is<FieldExpression>() || expr.is<MatrixFieldExpression>() ||
            expr.is<ExternalTensorExpression>() ||
            is_tensor(expr.expr->ret_type.ptr_removed()));

  // IndexExpression without ret_shape is used for matrix indexing,
  // where each entry of ExprGroup is interpreted as indexing into a specific
  // axis. For example, mat[3, 4] has indices_group={[3, 4]}, where [3, 4]
  // corresponds to "n"-axis and "m"-axis of the matrix. Therefore we expand
  // indices_group={[3, 4]} into {3, 4} to avoid TensorType in indices.
  std::vector<Expr> expanded_indices = this->expand_exprs(indices.exprs);
  auto expanded_expr_group = ExprGroup();
  expanded_expr_group.exprs = expanded_indices;

  return Expr::make<IndexExpression>(expr, expanded_expr_group, dbg_info);
}

void ASTBuilder::create_assert_stmt(const Expr &cond,
                                    const std::string &msg,
                                    const std::vector<Expr> &args,
                                    const DebugInfo &dbg_info) {
  auto stmt_unique =
      std::make_unique<FrontendAssertStmt>(cond, msg, args, dbg_info);
  this->insert(std::move(stmt_unique));
}

void ASTBuilder::begin_frontend_range_for(const Expr &i,
                                          const Expr &s,
                                          const Expr &e,
                                          const DebugInfo &dbg_info) {
  auto stmt_unique = std::make_unique<FrontendForStmt>(
      i, s, e, arch_, for_loop_dec_.config, dbg_info);
  auto stmt = stmt_unique.get();
  this->insert(std::move(stmt_unique));
  this->create_scope(stmt->body,
                     for_loop_dec_.config.strictly_serialized ? While : For);
  for_loop_dec_.reset();
}

void ASTBuilder::begin_frontend_struct_for_on_snode(const ExprGroup &loop_vars,
                                                    SNode *snode,
                                                    const DebugInfo &dbg_info) {
  TI_WARN_IF(
      for_loop_dec_.config.strictly_serialized,
      "ti.loop_config(serialize=True) does not have effect on the struct for. "
      "The execution order is not guaranteed.");
  auto stmt_unique = std::make_unique<FrontendForStmt>(
      loop_vars, snode, arch_, for_loop_dec_.config, dbg_info);
  for_loop_dec_.reset();
  auto stmt = stmt_unique.get();
  this->insert(std::move(stmt_unique));
  this->create_scope(stmt->body, For);
}

void ASTBuilder::begin_frontend_struct_for_on_external_tensor(
    const ExprGroup &loop_vars,
    const Expr &external_tensor,
    const DebugInfo &dbg_info) {
  TI_WARN_IF(
      for_loop_dec_.config.strictly_serialized,
      "ti.loop_config(serialize=True) does not have effect on the struct for. "
      "The execution order is not guaranteed.");
  auto stmt_unique = std::make_unique<FrontendForStmt>(
      loop_vars, external_tensor, arch_, for_loop_dec_.config, dbg_info);
  for_loop_dec_.reset();
  auto stmt = stmt_unique.get();
  this->insert(std::move(stmt_unique));
  this->create_scope(stmt->body, For);
}

void ASTBuilder::begin_frontend_mesh_for(
    const Expr &i,
    const mesh::MeshPtr &mesh_ptr,
    const mesh::MeshElementType &element_type,
    const DebugInfo &dbg_info) {
  TI_WARN_IF(
      for_loop_dec_.config.strictly_serialized,
      "ti.loop_config(serialize=True) does not have effect on the mesh for. "
      "The execution order is not guaranteed.");
  auto stmt_unique =
      std::make_unique<FrontendForStmt>(ExprGroup(i), mesh_ptr, element_type,
                                        arch_, for_loop_dec_.config, dbg_info);
  for_loop_dec_.reset();
  auto stmt = stmt_unique.get();
  this->insert(std::move(stmt_unique));
  this->create_scope(stmt->body, For);
}

void ASTBuilder::begin_frontend_while(const Expr &cond,
                                      const DebugInfo &dbg_info) {
  auto stmt_unique = std::make_unique<FrontendWhileStmt>(cond, dbg_info);
  auto stmt = stmt_unique.get();
  this->insert(std::move(stmt_unique));
  this->create_scope(stmt->body, While);
}

void ASTBuilder::insert_break_stmt(const DebugInfo &dbg_info) {
  if (loop_state_stack_.back() == Outermost) {
    throw TaichiSyntaxError("Cannot break in the outermost loop");
  }
  this->insert(Stmt::make<FrontendBreakStmt>(dbg_info));
}

void ASTBuilder::insert_continue_stmt(const DebugInfo &dbg_info) {
  this->insert(Stmt::make<FrontendContinueStmt>(dbg_info));
}

void ASTBuilder::insert_expr_stmt(const Expr &val) {
  this->insert(Stmt::make<FrontendExprStmt>(val));
}

void ASTBuilder::insert_snode_activate(SNode *snode,
                                       const ExprGroup &expr_group,
                                       const DebugInfo &dbg_info) {
  ExprGroup expanded_group;
  expanded_group.exprs = this->expand_exprs(expr_group.exprs);
  this->insert(Stmt::make<FrontendSNodeOpStmt>(
      SNodeOpType::activate, snode, expanded_group,
      /*val = */ Expr(std::shared_ptr<Expression>(nullptr)), dbg_info));
}

void ASTBuilder::insert_snode_deactivate(SNode *snode,
                                         const ExprGroup &expr_group,
                                         const DebugInfo &dbg_info) {
  ExprGroup expanded_group;
  expanded_group.exprs = this->expand_exprs(expr_group.exprs);
  this->insert(Stmt::make<FrontendSNodeOpStmt>(
      SNodeOpType::deactivate, snode, expanded_group,
      /*val = */ Expr(std::shared_ptr<Expression>(nullptr)), dbg_info));
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

    auto expand_tensor_or_scalar = [&](const Expr &expr) {
      if (!expr->ret_type.ptr_removed()->is<TensorType>()) {
        expanded_exprs.push_back(expr);
      } else {
        // Expand TensorType expr
        // clang-format off
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
        // clang-format on
        auto tensor_type = expr->ret_type.ptr_removed()->cast<TensorType>();

        Expr id_expr;
        if (expr.is<IdExpression>()) {
          id_expr = expr;
        } else {
          id_expr = make_var(expr, expr->dbg_info);
        }
        auto shape = tensor_type->get_shape();
        if (shape.size() == 1) {
          for (int i = 0; i < shape[0]; i++) {
            auto ind = Expr(std::make_shared<IndexExpression>(
                id_expr, ExprGroup(Expr(i)), expr->dbg_info));
            ind->type_check(nullptr);
            expanded_exprs.push_back(ind);
          }
        } else {
          TI_ASSERT(shape.size() == 2);
          for (int i = 0; i < shape[0]; i++) {
            for (int j = 0; j < shape[1]; j++) {
              auto ind = Expr(std::make_shared<IndexExpression>(
                  id_expr, ExprGroup(Expr(i), Expr(j)), expr->dbg_info));
              ind->type_check(nullptr);
              expanded_exprs.push_back(ind);
            }
          }
        }
      }
    };

    std::function<void(const Expr &, const StructType *, std::vector<int> &)>
        expand_struct = [&](const Expr &expr, const StructType *struct_type,
                            std::vector<int> &indices) {
          auto num_elem = struct_type->elements().size();
          for (int i = 0; i < num_elem; i++) {
            indices.push_back(i);
            auto element_type = struct_type->get_element_type({i});
            if (auto element_struct_type = element_type->cast<StructType>()) {
              expand_struct(expr, element_struct_type, indices);
            } else {
              auto elem = Expr(std::make_shared<GetElementExpression>(
                  expr, indices, expr->dbg_info));
              elem.expr->ret_type = element_type;
              expand_tensor_or_scalar(elem);
            }
            indices.pop_back();
          }
        };
    auto type = expr->ret_type.ptr_removed();
    if (auto struct_type = type->cast<StructType>()) {
      std::vector<int> indices;
      expand_struct(expr, struct_type, indices);
    } else {
      expand_tensor_or_scalar(expr);
    }
  }
  return expanded_exprs;
}

Expr ASTBuilder::mesh_index_conversion(mesh::MeshPtr mesh_ptr,
                                       mesh::MeshElementType idx_type,
                                       const Expr &idx,
                                       mesh::ConvType &conv_type,
                                       const DebugInfo &dbg_info) {
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

  return Expr::make<MeshIndexConversionExpression>(
      mesh_ptr.ptr.get(), idx_type, expanded_idx, conv_type, dbg_info);
}

void ASTBuilder::create_scope(std::unique_ptr<Block> &list, LoopType tp) {
  TI_ASSERT(list == nullptr);
  LoopState prev = loop_state_stack_.back();
  if (tp == NotLoop) {
    loop_state_stack_.push_back(prev);
  } else if (tp == For && stack_.size() == 1 && is_kernel_) {
    loop_state_stack_.push_back(Outermost);
  } else {
    loop_state_stack_.push_back(Inner);
  }
  list = std::make_unique<Block>();
  if (!stack_.empty()) {
    list->set_parent_stmt(get_last_stmt());
  }
  stack_.push_back(list.get());
}

void ASTBuilder::pop_scope() {
  stack_.pop_back();
  loop_state_stack_.pop_back();
}

Expr ASTBuilder::make_texture_op_expr(const TextureOpType &op,
                                      const Expr &texture_ptr,
                                      const ExprGroup &args,
                                      const DebugInfo &dbg_info) {
  ExprGroup expanded_args;
  expanded_args.exprs = this->expand_exprs(args.exprs);
  return Expr::make<TextureOpExpression>(op, texture_ptr, expanded_args,
                                         dbg_info);
}

Stmt *flatten_lvalue(Expr expr, Expression::FlattenContext *ctx) {
  expr->flatten(ctx);
  Stmt *ptr_stmt = expr->get_flattened_stmt();
  ptr_stmt->dbg_info = expr->dbg_info;
  return ptr_stmt;
}

Stmt *flatten_global_load(Stmt *ptr_stmt, Expression::FlattenContext *ctx) {
  auto load_stmt =
      std::make_unique<GlobalLoadStmt>(ptr_stmt, ptr_stmt->dbg_info);
  auto pointee_type = load_stmt->src->ret_type.ptr_removed();
  load_stmt->ret_type = pointee_type->get_compute_type();
  ctx->push_back(std::move(load_stmt));
  return ctx->back_stmt();
}

Stmt *flatten_local_load(Stmt *ptr_stmt, Expression::FlattenContext *ctx) {
  auto local_load = ctx->push_back<LocalLoadStmt>(ptr_stmt, ptr_stmt->dbg_info);
  local_load->ret_type = local_load->src->ret_type.ptr_removed();
  return local_load;
}

Stmt *flatten_rvalue(Expr ptr, Expression::FlattenContext *ctx) {
  ptr->flatten(ctx);
  Stmt *ptr_stmt = ptr->get_flattened_stmt();
  ptr_stmt->dbg_info = ptr->dbg_info;
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
