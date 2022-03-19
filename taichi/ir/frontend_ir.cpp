#include "taichi/ir/frontend_ir.h"

#include "taichi/ir/statements.h"
#include "taichi/program/program.h"
#include "taichi/common/exceptions.h"

TLANG_NAMESPACE_BEGIN

#define TI_ASSERT_TYPE_CHECKED(x)                       \
  TI_ASSERT_INFO(x->ret_type != PrimitiveType::unknown, \
                 "[{}] was not type-checked", x.serialize())

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

IRNode *FrontendContext::root() {
  return static_cast<IRNode *>(root_node_.get());
}

FrontendForStmt::FrontendForStmt(const ExprGroup &loop_var,
                                 const Expr &global_var,
                                 Arch arch,
                                 const ForLoopConfig &config)
    : global_var(global_var),
      bit_vectorize(config.bit_vectorize),
      num_cpu_threads(config.num_cpu_threads),
      strictly_serialized(config.strictly_serialized),
      mem_access_opt(config.mem_access_opt),
      block_dim(config.block_dim) {
  if (arch == Arch::cuda) {
    this->num_cpu_threads = 1;
    TI_ASSERT(this->block_dim <= taichi_max_gpu_block_dim);
  } else {
    // cpu
    if (this->num_cpu_threads == 0)
      this->num_cpu_threads = std::thread::hardware_concurrency();
  }
  loop_var_id.resize(loop_var.size());
  for (int i = 0; i < (int)loop_var.size(); i++) {
    loop_var_id[i] = loop_var[i].cast<IdExpression>()->id;
    loop_var[i].expr->ret_type = PrimitiveType::i32;
  }
}

FrontendForStmt::FrontendForStmt(const ExprGroup &loop_var,
                                 const mesh::MeshPtr &mesh,
                                 const mesh::MeshElementType &element_type,
                                 Arch arch,
                                 const ForLoopConfig &config)
    : bit_vectorize(config.bit_vectorize),
      num_cpu_threads(config.num_cpu_threads),
      mem_access_opt(config.mem_access_opt),
      block_dim(config.block_dim),
      mesh_for(true),
      mesh(mesh.ptr.get()),
      element_type(element_type) {
  if (arch == Arch::cuda) {
    this->num_cpu_threads = 1;
    TI_ASSERT(this->block_dim <= taichi_max_gpu_block_dim);
  } else {
    // cpu
    if (this->num_cpu_threads == 0)
      this->num_cpu_threads = std::thread::hardware_concurrency();
  }
  loop_var_id.resize(loop_var.size());
  for (int i = 0; i < (int)loop_var.size(); i++) {
    loop_var_id[i] = loop_var[i].cast<IdExpression>()->id;
  }
}

FrontendContext::FrontendContext(Arch arch) {
  root_node_ = std::make_unique<Block>();
  current_builder_ = std::make_unique<ASTBuilder>(root_node_.get(), arch);
}

FrontendForStmt::FrontendForStmt(const Expr &loop_var,
                                 const Expr &begin,
                                 const Expr &end,
                                 Arch arch,
                                 const ForLoopConfig &config)
    : begin(begin),
      end(end),
      bit_vectorize(config.bit_vectorize),
      num_cpu_threads(config.num_cpu_threads),
      strictly_serialized(config.strictly_serialized),
      mem_access_opt(config.mem_access_opt),
      block_dim(config.block_dim) {
  if (arch == Arch::cuda) {
    this->num_cpu_threads = 1;
  } else {
    if (this->num_cpu_threads == 0)
      this->num_cpu_threads = std::thread::hardware_concurrency();
  }
  loop_var_id.resize(1);
  loop_var_id[0] = loop_var.cast<IdExpression>()->id;
  loop_var.expr->ret_type = PrimitiveType::i32;
}

void ArgLoadExpression::type_check(CompileConfig *) {
  TI_ASSERT_INFO(dt->is<PrimitiveType>() && dt != PrimitiveType::unknown,
                 "Invalid dt [{}] for ArgLoadExpression", dt->to_string());
  ret_type = dt;
}

void ArgLoadExpression::flatten(FlattenContext *ctx) {
  auto arg_load = std::make_unique<ArgLoadStmt>(arg_id, dt);
  ctx->push_back(std::move(arg_load));
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

void UnaryOpExpression::serialize(std::ostream &ss) {
  ss << '(';
  if (is_cast()) {
    ss << (type == UnaryOpType::cast_value ? "" : "reinterpret_");
    ss << unary_op_type_name(type);
    ss << '<' << data_type_name(cast_type) << "> ";
  } else {
    ss << unary_op_type_name(type) << ' ';
  }
  operand->serialize(ss);
  ss << ')';
}

void UnaryOpExpression::type_check(CompileConfig *) {
  TI_ASSERT_TYPE_CHECKED(operand);
  if (!operand->ret_type->is<PrimitiveType>())
    throw TaichiTypeError(
        fmt::format("unsupported operand type(s) for '{}': '{}'",
                    unary_op_type_name(type), operand->ret_type->to_string()));
  if ((type == UnaryOpType::round || type == UnaryOpType::floor ||
       type == UnaryOpType::ceil || is_trigonometric(type)) &&
      !is_real(operand->ret_type))
    throw TaichiTypeError(
        fmt::format("'{}' takes real inputs only, however '{}' is provided",
                    unary_op_type_name(type), operand->ret_type->to_string()));
  ret_type = is_cast() ? cast_type : operand->ret_type;
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
  if (!lhs_type->is<PrimitiveType>() || !rhs_type->is<PrimitiveType>())
    error();
  if (binary_is_bitwise(type) &&
      (!is_integral(lhs_type) || !is_integral(rhs_type)))
    error();
  if (binary_is_logical(type) &&
      (lhs_type != PrimitiveType::i32 || rhs_type != PrimitiveType::i32))
    error();
  if (is_comparison(type) || binary_is_logical(type)) {
    ret_type = PrimitiveType::i32;
    return;
  }
  if (type == BinaryOpType::truediv) {
    auto default_fp = config->default_fp;
    if (!is_real(lhs_type)) {
      lhs_type = default_fp;
    }
    if (!is_real(rhs_type)) {
      rhs_type = default_fp;
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
    auto cond = ctx->push_back<LocalLoadStmt>(LocalAddress(result, 0));
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

    auto ret = ctx->push_back<LocalLoadStmt>(LocalAddress(result, 0));
    ret->tb = tb;
    stmt = ret;
    return;
  }
  flatten_rvalue(rhs, ctx);
  ctx->push_back(std::make_unique<BinaryOpStmt>(type, lhs->stmt, rhs->stmt));
  ctx->stmts.back()->tb = tb;
  stmt = ctx->back_stmt();
}

void TernaryOpExpression::type_check(CompileConfig *) {
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
  if (!is_integral(op1_type) || !op2_type->is<PrimitiveType>() ||
      !op3_type->is<PrimitiveType>())
    error();
  ret_type = promoted_type(op2_type, op3_type);
}

void TernaryOpExpression::flatten(FlattenContext *ctx) {
  // if (stmt)
  //  return;
  flatten_rvalue(op1, ctx);
  flatten_rvalue(op2, ctx);
  flatten_rvalue(op3, ctx);
  ctx->push_back(
      std::make_unique<TernaryOpStmt>(type, op1->stmt, op2->stmt, op3->stmt));
  stmt = ctx->back_stmt();
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
  ctx->push_back<InternalFuncStmt>(func_name, args_stmts);
  stmt = ctx->back_stmt();
}

void ExternalTensorExpression::flatten(FlattenContext *ctx) {
  auto ptr = Stmt::make<ArgLoadStmt>(arg_id, dt, /*is_ptr=*/true);
  ctx->push_back(std::move(ptr));
  stmt = ctx->back_stmt();
}

void GlobalVariableExpression::flatten(FlattenContext *ctx) {
  TI_ASSERT(snode->num_active_indices == 0);
  auto ptr = Stmt::make<GlobalPtrStmt>(LaneAttribute<SNode *>(snode),
                                       std::vector<Stmt *>());
  ctx->push_back(std::move(ptr));
}

void GlobalPtrExpression::type_check(CompileConfig *) {
  // Currently, dimension compatibility check happens in Python
  if (snode != nullptr) {
    ret_type = snode->dt;
  } else if (var.is<GlobalVariableExpression>()) {
    ret_type =
        var.cast<GlobalVariableExpression>()->snode->dt->get_compute_type();
  } else if (var.is<ExternalTensorExpression>()) {
    for (int i = 0; i < indices.exprs.size(); i++) {
      auto &expr = indices.exprs[i];
      TI_ASSERT_TYPE_CHECKED(expr);
      if (!is_integral(expr->ret_type))
        throw TaichiTypeError(
            fmt::format("indices must be integers, however '{}' is "
                        "provided as index {}",
                        expr->ret_type->to_string(), i));
    }
    ret_type = var.cast<ExternalTensorExpression>()->dt;
  } else {
    TI_ERROR("Invalid GlobalPtrExpression");
  }
}

void GlobalPtrExpression::serialize(std::ostream &ss) {
  if (snode) {
    ss << snode->get_node_type_name_hinted();
  } else {
    var.serialize(ss);
  }
  ss << '[';
  for (int i = 0; i < (int)indices.size(); i++) {
    indices.exprs[i]->serialize(ss);
    if (i + 1 < (int)indices.size())
      ss << ", ";
  }
  ss << ']';
}

void GlobalPtrExpression::flatten(FlattenContext *ctx) {
  std::vector<Stmt *> index_stmts;
  std::vector<int> offsets;
  SNode *snode = nullptr;
  if (this->snode != nullptr) {
    snode = this->snode;
  }
  if (bool(var) && var.is<GlobalVariableExpression>()) {
    snode = var.cast<GlobalVariableExpression>()->snode;
    offsets = snode->index_offsets;
  }
  for (int i = 0; i < (int)indices.size(); i++) {
    flatten_rvalue(indices.exprs[i], ctx);
    Stmt *ind = indices.exprs[i]->stmt;
    if (!offsets.empty()) {
      // Subtract offsets from indices so that new indices are
      // within [0, +inf)
      auto offset = ctx->push_back<ConstStmt>(TypedConstant(offsets[i]));
      ind = ctx->push_back<BinaryOpStmt>(BinaryOpType::sub, ind, offset);
    }
    index_stmts.push_back(ind);
  }
  if (snode) {
    ctx->push_back(std::make_unique<GlobalPtrStmt>(snode, index_stmts));
  } else {
    TI_ASSERT(var.is<ExternalTensorExpression>());
    flatten_lvalue(var, ctx);
    auto expr = var.cast<ExternalTensorExpression>();
    ctx->push_back(std::make_unique<ExternalPtrStmt>(
        expr->stmt, index_stmts, expr->element_shape, expr->element_dim));
  }
  stmt = ctx->back_stmt();
}

void TensorElementExpression::type_check(CompileConfig *) {
  std::string invalid_msg{
      "Invalid TensorElementExpression: the source is neither a local tensor "
      "nor a global tensor field"};
  if (is_local_tensor()) {
    TI_ASSERT_INFO(var->ret_type->is<TensorType>(), invalid_msg);
    ret_type = var->ret_type->cast<TensorType>()->get_element_type();
  } else if (is_global_tensor()) {
    TI_ASSERT_INFO(
        var.is<GlobalPtrExpression>() &&
            var.cast<GlobalPtrExpression>()->var.is<GlobalVariableExpression>(),
        invalid_msg);
    ret_type = var.cast<GlobalPtrExpression>()
                   ->var.cast<GlobalVariableExpression>()
                   ->snode->dt;
  } else {
    TI_ERROR(invalid_msg);
  }
}

bool TensorElementExpression::is_local_tensor() const {
  return var.is<IdExpression>();
}

bool TensorElementExpression::is_global_tensor() const {
  return var.is<GlobalPtrExpression>();
}

void TensorElementExpression::flatten(FlattenContext *ctx) {
  flatten_lvalue(var, ctx);
  Stmt *offset_stmt = ctx->push_back<ConstStmt>(TypedConstant(0));
  for (int i = 0; i < (int)shape.size(); ++i) {
    flatten_rvalue(indices[i], ctx);
    Stmt *shape_stmt = ctx->push_back<ConstStmt>(TypedConstant(shape[i]));
    Stmt *mul_stmt = ctx->push_back<BinaryOpStmt>(BinaryOpType::mul,
                                                  offset_stmt, shape_stmt);
    offset_stmt = ctx->push_back<BinaryOpStmt>(BinaryOpType::add, mul_stmt,
                                               indices[i]->stmt);
  }
  Stmt *stride_stmt = ctx->push_back<ConstStmt>(TypedConstant(stride));
  offset_stmt =
      ctx->push_back<BinaryOpStmt>(BinaryOpType::mul, offset_stmt, stride_stmt);
  stmt = ctx->push_back<PtrOffsetStmt>(var->stmt, offset_stmt);
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

void LoopUniqueExpression::serialize(std::ostream &ss) {
  ss << "loop_unique(";
  input.serialize(ss);
  for (int i = 0; i < covers.size(); i++) {
    if (i == 0)
      ss << ", covers=[";
    ss << covers[i]->get_node_type_name_hinted();
    if (i == (int)covers.size() - 1)
      ss << ']';
    else
      ss << ", ";
  }
  ss << ')';
}

void LoopUniqueExpression::flatten(FlattenContext *ctx) {
  flatten_rvalue(input, ctx);
  ctx->push_back(Stmt::make<LoopUniqueStmt>(input->stmt, covers));
  stmt = ctx->back_stmt();
}

void IdExpression::flatten(FlattenContext *ctx) {
  stmt = ctx->current_block->lookup_var(id);
}

void AtomicOpExpression::type_check(CompileConfig *) {
  TI_ASSERT_TYPE_CHECKED(dest);
  TI_ASSERT_TYPE_CHECKED(val);
  auto error = [&]() {
    throw TaichiTypeError(fmt::format(
        "unsupported operand type(s) for 'atomic_{}': '{}' and '{}'",
        atomic_op_type_name(op_type), dest->ret_type->to_string(),
        val->ret_type->to_string()));
  };
  if (!val->ret_type->is<PrimitiveType>())
    error();
  if (auto cit = dest->ret_type->cast<CustomIntType>()) {
    ret_type = cit->get_compute_type();
  } else if (auto cft = dest->ret_type->cast<CustomFloatType>()) {
    ret_type = cft->get_compute_type();
  } else if (dest->ret_type->is<PrimitiveType>()) {
    ret_type = dest->ret_type;
  } else {
    error();
  }
}

void AtomicOpExpression::serialize(std::ostream &ss) {
  if (op_type == AtomicOpType::add) {
    ss << "atomic_add(";
  } else if (op_type == AtomicOpType::sub) {
    ss << "atomic_sub(";
  } else if (op_type == AtomicOpType::min) {
    ss << "atomic_min(";
  } else if (op_type == AtomicOpType::max) {
    ss << "atomic_max(";
  } else if (op_type == AtomicOpType::bit_and) {
    ss << "atomic_bit_and(";
  } else if (op_type == AtomicOpType::bit_or) {
    ss << "atomic_bit_or(";
  } else if (op_type == AtomicOpType::bit_xor) {
    ss << "atomic_bit_xor(";
  } else {
    // min/max not supported in the LLVM backend yet.
    TI_NOT_IMPLEMENTED;
  }
  dest.serialize(ss);
  ss << ", ";
  val.serialize(ss);
  ss << ")";
}

void AtomicOpExpression::flatten(FlattenContext *ctx) {
  // replace atomic sub with negative atomic add
  if (op_type == AtomicOpType::sub) {
    val.set(Expr::make<UnaryOpExpression>(UnaryOpType::neg, val));
    op_type = AtomicOpType::add;
  }
  // expand rhs
  auto expr = val;
  flatten_rvalue(expr, ctx);
  if (dest.is<IdExpression>()) {  // local variable
    // emit local store stmt
    auto alloca = ctx->current_block->lookup_var(dest.cast<IdExpression>()->id);
    ctx->push_back<AtomicOpStmt>(op_type, alloca, expr->stmt);
  } else {
    TI_ASSERT(dest.is<GlobalPtrExpression>() ||
              dest.is<TensorElementExpression>());
    flatten_lvalue(dest, ctx);
    ctx->push_back<AtomicOpStmt>(op_type, dest->stmt, expr->stmt);
  }
  stmt = ctx->back_stmt();
}

void SNodeOpExpression::type_check(CompileConfig *) {
  if (op_type == SNodeOpType::get_addr) {
    ret_type = PrimitiveType::u64;
  } else {
    ret_type = PrimitiveType::i32;
  }
}

void SNodeOpExpression::serialize(std::ostream &ss) {
  ss << snode_op_type_name(op_type);
  ss << '(';
  ss << snode->get_node_type_name_hinted() << ", [";
  indices.serialize(ss);
  ss << "]";
  if (value.expr) {
    ss << ' ';
    value.serialize(ss);
  }
  ss << ')';
}

void SNodeOpExpression::flatten(FlattenContext *ctx) {
  std::vector<Stmt *> indices_stmt;
  for (int i = 0; i < (int)indices.size(); i++) {
    flatten_rvalue(indices[i], ctx);
    indices_stmt.push_back(indices[i]->stmt);
  }
  auto ptr = ctx->push_back<GlobalPtrStmt>(snode, indices_stmt);
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
    ctx->push_back<SNodeOpStmt>(SNodeOpType::append, snode, ptr, value->stmt);
    TI_ERROR_IF(snode->type != SNodeType::dynamic,
                "ti.append only works on dynamic nodes.");
    TI_ERROR_IF(snode->ch.size() != 1,
                "ti.append only works on single-child dynamic nodes.");
    TI_ERROR_IF(data_type_size(snode->ch[0]->dt) != 4,
                "ti.append only works on i32/f32 nodes.");
  }
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
  TI_ASSERT_INFO(ptr.is<ExternalTensorExpression>(),
                 "Invalid ptr [{}] for ExternalTensorShapeAlongAxisExpression",
                 ptr.serialize());
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

void FuncCallExpression::serialize(std::ostream &ss) {
  ss << "func_call(\"" << func->func_key.get_full_name() << "\", ";
  args.serialize(ss);
  ss << ')';
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

void ASTBuilder::insert_assignment(Expr &lhs, const Expr &rhs) {
  // Inside a kernel or a function
  // Create an assignment in the IR
  if (lhs.expr == nullptr) {
    lhs.set(rhs);
  } else if (lhs.expr->is_lvalue()) {
    this->insert(std::make_unique<FrontendAssignStmt>(lhs, rhs));
  } else {
    TI_ERROR("Cannot assign to non-lvalue: {}", lhs.serialize());
  }
}

Expr ASTBuilder::make_var(const Expr &x) {
  auto var = Expr(std::make_shared<IdExpression>());
  this->insert(std::make_unique<FrontendAllocaStmt>(
      std::static_pointer_cast<IdExpression>(var.expr)->id,
      PrimitiveType::unknown));
  this->insert_assignment(var, x);
  return var;
}

void ASTBuilder::insert_for(const Expr &s,
                            const Expr &e,
                            const std::function<void(Expr)> &func) {
  auto i = Expr(std::make_shared<IdExpression>());
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
  return Expr::make<InternalFuncCallExpression>("linear_thread_idx",
                                                std::vector<Expr>{});
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
                loop->as<FrontendForStmt>()->mesh_for),
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
  auto var = Expr(std::make_shared<IdExpression>());
  this->insert(std::make_unique<FrontendAllocaStmt>(
      std::static_pointer_cast<IdExpression>(var.expr)->id,
      PrimitiveType::unknown));
  return var;
}

Expr ASTBuilder::expr_alloca_local_tensor(const std::vector<int> &shape,
                                          const DataType &element_type,
                                          const ExprGroup &elements) {
  auto var = Expr(std::make_shared<IdExpression>());
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
        Expr::make<TensorElementExpression>(var, indices, shape, 1),
        elements.exprs[i]));
  }
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

void ASTBuilder::begin_frontend_struct_for(const ExprGroup &loop_vars,
                                           const Expr &global) {
  auto stmt_unique = std::make_unique<FrontendForStmt>(loop_vars, global, arch_,
                                                       for_loop_dec_.config);
  for_loop_dec_.reset();
  auto stmt = stmt_unique.get();
  this->insert(std::move(stmt_unique));
  this->create_scope(stmt->body, For);
}

void ASTBuilder::begin_frontend_mesh_for(
    const Expr &i,
    const mesh::MeshPtr &mesh_ptr,
    const mesh::MeshElementType &element_type) {
  auto stmt_unique = std::make_unique<FrontendForStmt>(
      i, mesh_ptr, element_type, arch_, for_loop_dec_.config);
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
  ctx->push_back<LocalLoadStmt>(LocalAddress(ptr->stmt, 0));
  ptr->stmt = ctx->back_stmt();
}

void flatten_rvalue(Expr ptr, Expression::FlattenContext *ctx) {
  ptr->flatten(ctx);
  if (ptr.is<IdExpression>()) {
    if (ptr->stmt->is<AllocaStmt>()) {
      flatten_local_load(ptr, ctx);
    }
  } else if (ptr.is<GlobalPtrExpression>()) {
    flatten_global_load(ptr, ctx);
  } else if (ptr.is<GlobalVariableExpression>()) {
    TI_ASSERT(ptr.cast<GlobalVariableExpression>()->snode->num_active_indices ==
              0);
    flatten_global_load(ptr[ExprGroup()], ctx);
  } else if (ptr.is<TensorElementExpression>()) {
    auto tensor_ptr = ptr.cast<TensorElementExpression>();
    if (tensor_ptr->is_global_tensor())
      flatten_global_load(ptr, ctx);
    else if (tensor_ptr->is_local_tensor())
      flatten_local_load(ptr, ctx);
    else {
      TI_NOT_IMPLEMENTED
    }
  }
}

TLANG_NAMESPACE_END
