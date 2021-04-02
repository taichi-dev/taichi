#include "taichi/ir/frontend_ir.h"

#include "taichi/ir/statements.h"
#include "taichi/program/program.h"

TLANG_NAMESPACE_BEGIN

FrontendSNodeOpStmt::FrontendSNodeOpStmt(SNodeOpType op_type,
                                         SNode *snode,
                                         const ExprGroup &indices,
                                         const Expr &val)
    : op_type(op_type), snode(snode), indices(indices.loaded()), val(val) {
  if (val.expr != nullptr) {
    TI_ASSERT(op_type == SNodeOpType::append);
    this->val.set(load_if_ptr(val));
  } else {
    TI_ASSERT(op_type != SNodeOpType::append);
  }
}

FrontendAssignStmt::FrontendAssignStmt(const Expr &lhs, const Expr &rhs)
    : lhs(lhs), rhs(rhs) {
  TI_ASSERT(lhs->is_lvalue());
}

IRNode *FrontendContext::root() {
  return static_cast<IRNode *>(root_node.get());
}

FrontendForStmt::FrontendForStmt(const ExprGroup &loop_var,
                                 const Expr &global_var)
    : global_var(global_var) {
  vectorize = dec.vectorize;
  bit_vectorize = dec.bit_vectorize;
  num_cpu_threads = dec.num_cpu_threads;
  strictly_serialized = dec.strictly_serialized;
  block_dim = dec.block_dim;
  auto cfg = get_current_program().config;
  if (cfg.arch == Arch::cuda) {
    vectorize = 1;
    num_cpu_threads = 1;
    TI_ASSERT(block_dim <= taichi_max_gpu_block_dim);
  } else {
    // cpu
    if (num_cpu_threads == 0)
      num_cpu_threads = std::thread::hardware_concurrency();
  }
  mem_access_opt = dec.mem_access_opt;
  dec.reset();
  if (vectorize == -1)
    vectorize = 1;

  loop_var_id.resize(loop_var.size());
  for (int i = 0; i < (int)loop_var.size(); i++) {
    loop_var_id[i] = loop_var[i].cast<IdExpression>()->id;
  }
}

DecoratorRecorder dec;

FrontendContext::FrontendContext() {
  root_node = std::make_unique<Block>();
  current_builder = std::make_unique<ASTBuilder>(root_node.get());
}

FrontendForStmt::FrontendForStmt(const Expr &loop_var,
                                 const Expr &begin,
                                 const Expr &end)
    : begin(begin), end(end) {
  vectorize = dec.vectorize;
  bit_vectorize = dec.bit_vectorize;
  num_cpu_threads = dec.num_cpu_threads;
  strictly_serialized = dec.strictly_serialized;
  block_dim = dec.block_dim;
  auto cfg = get_current_program().config;
  if (cfg.arch == Arch::cuda) {
    vectorize = 1;
    num_cpu_threads = 1;
  } else {
    if (num_cpu_threads == 0)
      num_cpu_threads = std::thread::hardware_concurrency();
  }
  mem_access_opt = dec.mem_access_opt;
  dec.reset();
  if (vectorize == -1)
    vectorize = 1;
  loop_var_id.resize(1);
  loop_var_id[0] = loop_var.cast<IdExpression>()->id;
}

void ArgLoadExpression::flatten(FlattenContext *ctx) {
  auto arg_load = std::make_unique<ArgLoadStmt>(arg_id, dt);
  ctx->push_back(std::move(arg_load));
  stmt = ctx->back_stmt();
}

void RandExpression::flatten(FlattenContext *ctx) {
  auto ran = std::make_unique<RandStmt>(dt);
  ctx->push_back(std::move(ran));
  stmt = ctx->back_stmt();
}

std::string UnaryOpExpression::serialize() {
  if (is_cast()) {
    std::string reint = type == UnaryOpType::cast_value ? "" : "reinterpret_";
    return fmt::format("({}{}<{}> {})", reint, unary_op_type_name(type),
                       data_type_name(cast_type), operand->serialize());
  } else {
    return fmt::format("({} {})", unary_op_type_name(type),
                       operand->serialize());
  }
}

bool UnaryOpExpression::is_cast() const {
  return unary_op_is_cast(type);
}

void UnaryOpExpression::flatten(FlattenContext *ctx) {
  operand->flatten(ctx);
  auto unary = std::make_unique<UnaryOpStmt>(type, operand->stmt);
  if (is_cast()) {
    unary->cast_type = cast_type;
  }
  stmt = unary.get();
  stmt->tb = tb;
  ctx->push_back(std::move(unary));
}

void BinaryOpExpression::flatten(FlattenContext *ctx) {
  // if (stmt)
  //  return;
  lhs->flatten(ctx);
  rhs->flatten(ctx);
  ctx->push_back(std::make_unique<BinaryOpStmt>(type, lhs->stmt, rhs->stmt));
  ctx->stmts.back()->tb = tb;
  stmt = ctx->back_stmt();
}

void TernaryOpExpression::flatten(FlattenContext *ctx) {
  // if (stmt)
  //  return;
  op1->flatten(ctx);
  op2->flatten(ctx);
  op3->flatten(ctx);
  ctx->push_back(
      std::make_unique<TernaryOpStmt>(type, op1->stmt, op2->stmt, op3->stmt));
  stmt = ctx->back_stmt();
}

void ExternalFuncCallExpression::flatten(FlattenContext *ctx) {
  std::vector<Stmt *> arg_statements, output_statements;
  for (auto &s : args) {
    s->flatten(ctx);
    arg_statements.push_back(s->stmt);
  }
  for (auto &s : outputs) {
    output_statements.push_back(s.cast<IdExpression>()->flatten_noload(ctx));
  }
  ctx->push_back(std::make_unique<ExternalFuncCallStmt>(
      func, source, arg_statements, output_statements));
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

std::string GlobalPtrExpression::serialize() {
  std::string s = fmt::format(
      "{}[", snode ? snode->get_node_type_name_hinted() : var.serialize());
  for (int i = 0; i < (int)indices.size(); i++) {
    s += indices.exprs[i]->serialize();
    if (i + 1 < (int)indices.size())
      s += ", ";
  }
  s += "]";
  return s;
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
    indices.exprs[i]->flatten(ctx);
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
    var->flatten(ctx);
    ctx->push_back(std::make_unique<ExternalPtrStmt>(
        var.cast<ExternalTensorExpression>()->stmt, index_stmts));
  }
  stmt = ctx->back_stmt();
}

void RangeAssumptionExpression::flatten(FlattenContext *ctx) {
  input->flatten(ctx);
  base->flatten(ctx);
  ctx->push_back(
      Stmt::make<RangeAssumptionStmt>(input->stmt, base->stmt, low, high));
  stmt = ctx->back_stmt();
}

std::string LoopUniqueExpression::serialize() {
  std::string result = "loop_unique(" + input->serialize();
  for (int i = 0; i < covers.size(); i++) {
    if (i == 0)
      result += ", covers=[";
    result += covers[i]->get_node_type_name_hinted();
    if (i == (int)covers.size() - 1)
      result += "]";
    else
      result += ", ";
  }
  result += ")";
  return result;
}

void LoopUniqueExpression::flatten(FlattenContext *ctx) {
  input->flatten(ctx);
  ctx->push_back(Stmt::make<LoopUniqueStmt>(input->stmt, covers));
  stmt = ctx->back_stmt();
}

void IdExpression::flatten(FlattenContext *ctx) {
  auto var_stmt = ctx->current_block->lookup_var(id);
  if (var_stmt->is<AllocaStmt>()) {
    ctx->push_back(std::make_unique<LocalLoadStmt>(LocalAddress(var_stmt, 0)));
    stmt = ctx->back_stmt();
  } else {
    // The loop index may have a coordinate offset.
    TI_ASSERT(var_stmt->is<LoopIndexStmt>() || var_stmt->is<BinaryOpStmt>());
    stmt = var_stmt;
  }
}

std::string AtomicOpExpression::serialize() {
  if (op_type == AtomicOpType::add) {
    return fmt::format("atomic_add({}, {})", dest.serialize(), val.serialize());
  } else if (op_type == AtomicOpType::sub) {
    return fmt::format("atomic_sub({}, {})", dest.serialize(), val.serialize());
  } else if (op_type == AtomicOpType::min) {
    return fmt::format("atomic_min({}, {})", dest.serialize(), val.serialize());
  } else if (op_type == AtomicOpType::max) {
    return fmt::format("atomic_max({}, {})", dest.serialize(), val.serialize());
  } else if (op_type == AtomicOpType::bit_and) {
    return fmt::format("atomic_bit_and({}, {})", dest.serialize(),
                       val.serialize());
  } else if (op_type == AtomicOpType::bit_or) {
    return fmt::format("atomic_bit_or({}, {})", dest.serialize(),
                       val.serialize());
  } else if (op_type == AtomicOpType::bit_xor) {
    return fmt::format("atomic_bit_xor({}, {})", dest.serialize(),
                       val.serialize());
  } else {
    // min/max not supported in the LLVM backend yet.
    TI_NOT_IMPLEMENTED;
  }
}

void AtomicOpExpression::flatten(FlattenContext *ctx) {
  // replace atomic sub with negative atomic add
  if (op_type == AtomicOpType::sub) {
    val.set(Expr::make<UnaryOpExpression>(UnaryOpType::neg, val));
    op_type = AtomicOpType::add;
  }
  // expand rhs
  auto expr = val;
  expr->flatten(ctx);
  if (dest.is<IdExpression>()) {  // local variable
    // emit local store stmt
    auto alloca = ctx->current_block->lookup_var(dest.cast<IdExpression>()->id);
    ctx->push_back<AtomicOpStmt>(op_type, alloca, expr->stmt);
  } else {  // global variable
    TI_ASSERT(dest.is<GlobalPtrExpression>());
    auto global_ptr = dest.cast<GlobalPtrExpression>();
    global_ptr->flatten(ctx);
    ctx->push_back<AtomicOpStmt>(op_type, ctx->back_stmt(), expr->stmt);
  }
  stmt = ctx->back_stmt();
}

std::string SNodeOpExpression::serialize() {
  if (value.expr) {
    return fmt::format("{}({}, [{}], {})", snode_op_type_name(op_type),
                       snode->get_node_type_name_hinted(), indices.serialize(),
                       value.serialize());
  } else {
    return fmt::format("{}({}, [{}])", snode_op_type_name(op_type),
                       snode->get_node_type_name_hinted(), indices.serialize());
  }
}

void SNodeOpExpression::flatten(FlattenContext *ctx) {
  std::vector<Stmt *> indices_stmt;
  for (int i = 0; i < (int)indices.size(); i++) {
    indices[i]->flatten(ctx);
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
  } else if (op_type == SNodeOpType::append) {
    value->flatten(ctx);
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

void GlobalLoadExpression::flatten(FlattenContext *ctx) {
  ptr->flatten(ctx);
  ctx->push_back(std::make_unique<GlobalLoadStmt>(ptr->stmt));
  stmt = ctx->back_stmt();
}

void ConstExpression::flatten(FlattenContext *ctx) {
  ctx->push_back(Stmt::make<ConstStmt>(val));
  stmt = ctx->back_stmt();
}

void ExternalTensorShapeAlongAxisExpression::flatten(FlattenContext *ctx) {
  auto temp = ptr.cast<ExternalTensorExpression>();
  TI_ASSERT(0 <= axis && axis < temp->dim);
  ctx->push_back<ExternalTensorShapeAlongAxisStmt>(axis, temp->arg_id);
  stmt = ctx->back_stmt();
}

Block *ASTBuilder::current_block() {
  if (stack.empty())
    return nullptr;
  else
    return stack.back();
}

Stmt *ASTBuilder::get_last_stmt() {
  TI_ASSERT(!stack.empty());
  return stack.back()->back();
}

void ASTBuilder::insert(std::unique_ptr<Stmt> &&stmt, int location) {
  TI_ASSERT(!stack.empty());
  stack.back()->insert(std::move(stmt), location);
}

void ASTBuilder::stop_gradient(SNode *snode) {
  TI_ASSERT(!stack.empty());
  stack.back()->stop_gradients.push_back(snode);
}

std::unique_ptr<ASTBuilder::ScopeGuard> ASTBuilder::create_scope(
    std::unique_ptr<Block> &list) {
  TI_ASSERT(list == nullptr);
  list = std::make_unique<Block>();
  if (!stack.empty()) {
    list->parent_stmt = get_last_stmt();
  }
  return std::make_unique<ScopeGuard>(this, list.get());
}

ASTBuilder &current_ast_builder() {
  return context->builder();
}

std::unique_ptr<FrontendContext> context;

TLANG_NAMESPACE_END
