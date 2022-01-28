#include "taichi/ir/operation.h"

namespace taichi {
namespace lang {

DataTypeExpression::DataTypeExpression(DataType ty) : type(ty) {
}

void DataTypeExpression::unify(Solutions &solutions, DataType ty) const {
  TI_ASSERT(type == ty);
}

DataType DataTypeExpression::resolve(const Solutions &solutions) const {
  return type;
}

std::string DataTypeExpression::to_string() const {
  return type.to_string();
}

TyvarTypeExpression::TyvarTypeExpression(const Identifier &id) : id(id) {
}

void TyvarTypeExpression::unify(Solutions &solutions, DataType ty) const {
  if (solutions.find(id) == solutions.end()) {
    solutions[id] = ty;
  } else {
    DataTypeExpression(solutions[id]).unify(solutions, ty);
  }
}

DataType TyvarTypeExpression::resolve(const Solutions &solutions) const {
  if (solutions.find(id) == solutions.end()) {
    TI_ERROR("Cannot determine type variable {}", id.name());
  } else {
    return solutions.at(id);
  }
}

std::string TyvarTypeExpression::to_string() const {
  return id.name();
}

CommonTypeExpression::CommonTypeExpression(TypeExpr lhs, TypeExpr rhs)
    : lhs(lhs), rhs(rhs) {
}

void CommonTypeExpression::unify(Solutions &solutions, DataType ty) const {
  TI_ASSERT(resolve(solutions) == ty);
}

DataType CommonTypeExpression::resolve(const Solutions &solutions) const {
  DataType l = lhs->resolve(solutions), r = rhs->resolve(solutions);
  return promoted_type(l->get_compute_type(), r->get_compute_type());
}

std::string CommonTypeExpression::to_string() const {
  return fmt::format("{} | {}", lhs->to_string(), rhs->to_string());
}

std::shared_ptr<DataTypeExpression> TypeSpec::dt(DataType dt) {
  std::cerr << (dt == DataType(nullptr));
  return std::make_shared<DataTypeExpression>(dt);
}

Tyvar TypeSpec::var(const Identifier &id) {
  return std::make_shared<TyvarTypeExpression>(id);
}

Tyvar TypeSpec::var(const std::string &name) {
  return std::make_shared<TyvarTypeExpression>(Identifier(name));
}

std::shared_ptr<CommonTypeExpression> TypeSpec::lub(TypeExpr lhs,
                                                    TypeExpr rhs) {
  return std::make_shared<CommonTypeExpression>(lhs, rhs);
}

DynamicTrait::DynamicTrait(const std::string &description,
                           const std::function<TraitPred> &func)
    : description_(description), func_(func) {
}

std::string DynamicTrait::describe() const {
  return description_;
}

bool DynamicTrait::has_type(DataType ty) const {
  return func_(ty);
}

std::vector<Stmt *> get_all_stmts(const std::vector<Expr> &exprs) {
  std::vector<Stmt *> stmts;
  stmts.reserve(exprs.size());
  for (auto &expr : exprs) {
    stmts.push_back(expr->stmt);
  }
  return stmts;
}

Operation::Constraint::Constraint(Tyvar tyvar) : tyvar(tyvar) {
}
Operation::Constraint::Constraint(Tyvar tyvar, Trait *trait)
    : tyvar(tyvar), traits{trait} {
}
Operation::Constraint::Constraint(Tyvar tyvar,
                                  const std::vector<Trait *> &traits)
    : tyvar(tyvar), traits(traits) {
}

Operation::Param::Param(TypeExpr ty) : type_expr(ty) {
}
Operation::Param::Param(ValueType vt, TypeExpr ty)
    : value_type(vt), type_expr(ty) {
}

Operation::Operation(const std::string &name,
                     const std::vector<Constraint> &constraints,
                     const std::vector<Param> &params,
                     TypeExpr result)
    : name(name), constraints(constraints), params(params), result(result) {
}

Stmt *Operation::lower(Expression::FlattenContext *ctx,
                       std::vector<Expr> &args) const {
  for (int i = 0; i < args.size(); i++) {
    if (params[i].value_type == LValue) {
      flatten_lvalue(args[i], ctx);
    } else if (params[i].value_type == RValue) {
      flatten_rvalue(args[i], ctx);
    }
  }
  return flatten(ctx, args);
}

DataType Operation::check(const std::vector<Expr> &args) const {
  TypeExpression::Solutions solutions;
  TI_ASSERT(args.size() == params.size());
  for (int i = 0; i < args.size(); i++) {
    params[i].type_expr->unify(solutions, args[i]->ret_type);
  }
  for (auto &constraint : constraints) {
    if (solutions.find(constraint.tyvar->id) == solutions.end()) {
      TI_ERROR("Type variable {} cannot be determined in operation {}",
               constraint.tyvar->id.name(), name);
    }
    for (auto &trait : constraint.traits) {
      if (!trait->has_type(solutions[constraint.tyvar->id])) {
        TI_ERROR("{} is not {}, which is required by operation {}",
                 constraint.tyvar->id.name(), trait->describe(), name);
      }
    }
  }
  return result->resolve(solutions);
}

Stmt *InternalCallOperation::flatten(Expression::FlattenContext *ctx,
                                     std::vector<Expr> &args) const {
  return ctx->push_back<InternalFuncStmt>(internal_call_name_,
                                          get_all_stmts(args));
}

std::vector<Operation::Param> make_real_params_from_dt(
    const std::vector<DataType> &params) {
  std::vector<Operation::Param> real_params;
  real_params.reserve(params.size());
  for (auto dt : params) {
    real_params.emplace_back(TypeSpec::dt(dt));
  }
  return real_params;
}

InternalCallOperation::InternalCallOperation(
    const std::string &name,
    const std::string &internal_name,
    const std::vector<DataType> &params,
    DataType result)
    : Operation(name,
                std::vector<Constraint>(),
                make_real_params_from_dt(params),
                TypeSpec::dt(result)),
      internal_call_name_(internal_name) {
}

Stmt *DynamicOperation::flatten(Expression::FlattenContext *ctx,
                                std::vector<Expr> &args) const {
  return func_(ctx, args);
}

DynamicOperation::DynamicOperation(const std::string &name,
                                   const std::vector<Constraint> &tyvars,
                                   const std::vector<Param> &params,
                                   TypeExpr result,
                                   const std::function<DynOp> &func)
    : Operation(name, tyvars, params, result), func_(func) {
}

StaticTraits::StaticTraits() {
  primitive = new DynamicTrait(
      "Primitive", [](DataType ty) { return ty->is<PrimitiveType>(); });

  custom = new DynamicTrait("Custom", is_custom_type);

  scalar = new DynamicTrait("Scalar", [this](DataType ty) {
    return primitive->has_type(ty) || custom->has_type(ty);
  });

  real = new DynamicTrait("Real", is_real);

  integral = new DynamicTrait("Integral", is_integral);
}

InternalOps::InternalOps() {
  thread_index = new InternalCallOperation("thread_index", "linear_thread_idx",
                                           {}, PrimitiveType::i32);

  insert_triplet =
      new InternalCallOperation("insert_triplet", "insert_triplet",
                                {PrimitiveType::u64, PrimitiveType::i32,
                                 PrimitiveType::i32, PrimitiveType::f32},
                                PrimitiveType::i32);

  do_nothing = new InternalCallOperation("do_nothing", "do_nothing", {},
                                         PrimitiveType::i32);

  refresh_counter = new InternalCallOperation(
      "refresh_counter", "refresh_counter", {}, PrimitiveType::i32);
}

TestInternalOps::TestInternalOps() {
  test_active_mask = new InternalCallOperation(
      "test_active_mask", "test_active_mask", {}, PrimitiveType::i32);

  test_shfl = new InternalCallOperation("test_shfl", "test_shfl", {},
                                        PrimitiveType::i32);

  test_stack = new InternalCallOperation("test_stack", "test_stack", {},
                                         PrimitiveType::i32);

  test_list_manager = new InternalCallOperation(
      "test_list_manager", "test_list_manager", {}, PrimitiveType::i32);

  test_node_allocator = new InternalCallOperation(
      "test_node_allocator", "test_node_allocator", {}, PrimitiveType::i32);

  test_node_allocator_gc_cpu = new InternalCallOperation(
      "test_node_allocator_gc_cpu", "test_node_allocator_gc_cpu", {},
      PrimitiveType::i32);

  test_internal_func_args = new InternalCallOperation(
      "test_internal_func_args", "test_internal_func_args",
      {PrimitiveType::f32, PrimitiveType::f32, PrimitiveType::i32},
      PrimitiveType::i32);
}

}  // namespace lang
}  // namespace taichi
