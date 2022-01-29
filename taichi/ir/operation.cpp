#include "taichi/ir/operation.h"
#include "taichi/ir/frontend_ir.h"

namespace taichi {
namespace lang {

#define TI_ASSERT_TYPE_CHECKED(x)                       \
  TI_ASSERT_INFO(x->ret_type != PrimitiveType::unknown, \
                 "[{}] was not type-checked", x.serialize())

DataTypeExpression::DataTypeExpression(DataType ty) : type(ty) {
}

DataTypeExpression::DatatypeMismatch::DatatypeMismatch(DataType expected,
                                                       DataType actual)
    : expected(expected), actual(actual) {
}

std::string DataTypeExpression::DatatypeMismatch::to_string() const {
  return fmt::format("the actual type `{}` is not the expected type `{}`",
                     actual.to_string(), expected.to_string());
}

void DataTypeExpression::unify(Solutions &solutions, DataType ty) const {
  if (type != ty)
    throw DatatypeMismatch(type, ty);
}

DataType DataTypeExpression::resolve(const Solutions &solutions) const {
  return type;
}

std::string DataTypeExpression::to_string() const {
  return type.to_string();
}

TyvarTypeExpression::TyvarTypeExpression(const Identifier &id) : id(id) {
}

TyvarTypeExpression::TyvarUnsolvable::TyvarUnsolvable(const std::string &name)
    : name(name) {
}

std::string TyvarTypeExpression::TyvarUnsolvable::to_string() const {
  return fmt::format("`{}` cannot be determined", name);
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
    throw TyvarUnsolvable(id.name());
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
  DataTypeExpression(resolve(solutions)).unify(solutions, ty);
}

DataType CommonTypeExpression::resolve(const Solutions &solutions) const {
  DataType l = lhs->resolve(solutions), r = rhs->resolve(solutions);
  return promoted_type(l->get_compute_type(), r->get_compute_type());
}

std::string CommonTypeExpression::to_string() const {
  return fmt::format("{} | {}", lhs->to_string(), rhs->to_string());
}

ComputeTypeExpression::ComputeTypeExpression(TypeExpr ty) : ty(ty) {
}

void ComputeTypeExpression::unify(Solutions &solutions, DataType ty) const {
  return DataTypeExpression(resolve(solutions)).unify(solutions, ty);
}

DataType ComputeTypeExpression::resolve(const Solutions &solutions) const {
  auto ct = ty->resolve(solutions);
  return ct->get_compute_type();
}

std::string ComputeTypeExpression::to_string() const {
  return fmt::format("compute({})", ty->to_string());
}

std::shared_ptr<DataTypeExpression> TypeSpec::dt(DataType dt) {
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

std::shared_ptr<ComputeTypeExpression> TypeSpec::comp(TypeExpr ty) {
  return std::make_shared<ComputeTypeExpression>(ty);
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

std::vector<Stmt *> get_all_stmts(const std::vector<Expr> &exprs,
                                  Expression::FlattenContext *ctx) {
  std::vector<Stmt *> stmts;
  stmts.reserve(exprs.size());
  for (auto expr : exprs) {
    stmts.push_back(flatten_rvalue(expr, ctx));
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

Operation::Operation(const std::string &name,
                     const std::vector<Constraint> &constraints,
                     const std::vector<TypeExpr> &params,
                     TypeExpr result)
    : name(name), constraints(constraints), params(params), result(result) {
}

DataType Operation::check(const std::vector<Expr> &args) const {
#define TYPE_ERROR(x, xs...)                                \
  throw TaichiTypeError(fmt::format("In a call to `{}`: " x \
                                    " (signature: {})",     \
                                    name, xs, signature_string()))
  TypeExpression::Solutions solutions;
  if (args.size() != params.size()) {
    TYPE_ERROR("The argument list length is incorrect (expected {}, actual {})",
               params.size(), args.size());
  }
  for (int i = 0; i < args.size(); i++) {
    TI_ASSERT_TYPE_CHECKED(args[i]);
    try {
      params[i]->unify(solutions, args[i]->ret_type);
    } catch (TypeExpression::UnifyFailure &e) {
      TYPE_ERROR("For the {}th argument, {}", i + 1, e.to_string());
    }
  }
  for (auto &constraint : constraints) {
    if (solutions.find(constraint.tyvar->id) == solutions.end()) {
      TYPE_ERROR("`{}` cannot be determined", constraint.tyvar->id.name());
    }
    for (auto &trait : constraint.traits) {
      if (!trait->has_type(solutions[constraint.tyvar->id])) {
        TYPE_ERROR("`{}` needs to be {}, but {} is not",
                   constraint.tyvar->id.name(), trait->describe(),
                   solutions[constraint.tyvar->id].to_string());
      }
    }
  }
  return result->resolve(solutions);
#undef TYPE_ERROR
}

std::string Operation::signature_string() const {
  std::string sig(name);
  if (!constraints.empty()) {
    sig += "<";
    for (int i = 0; i < constraints.size(); i++) {
      auto &con = constraints[i];
      sig += con.tyvar->to_string();
      if (!con.traits.empty()) {
        sig += " : ";
        for (int j = 0; j < con.traits.size(); j++) {
          sig += con.traits[j]->describe();
          if (j != con.traits.size() - 1)
            sig += " + ";
        }
      }
      if (i != constraints.size() - 1)
        sig += ", ";
    }
    sig += ">";
  }
  sig += "(";
  for (int i = 0; i < params.size(); i++) {
    sig += params[i]->to_string();
    if (i != params.size() - 1)
      sig += ", ";
  }
  sig += ") -> ";
  sig += result->to_string();
  return sig;
}

Expr Operation::call(const std::vector<Expr> &args) {
  return Expr::make<CallExpression>(this, args);
}

Stmt *DynamicOperation::flatten(Expression::FlattenContext *ctx,
                                std::vector<Expr> &args) const {
  return func_(ctx, args);
}

DynamicOperation::DynamicOperation(const std::string &name,
                                   const std::vector<Constraint> &tyvars,
                                   const std::vector<TypeExpr> &params,
                                   TypeExpr result,
                                   const std::function<DynOp> &func)
    : Operation(name, tyvars, params, result), func_(func) {
}

DynamicOperation::DynamicOperation(const std::string &name,
                                   Operation *temp,
                                   const std::function<DynOp> &func)
    : DynamicOperation(name,
                       temp->constraints,
                       temp->params,
                       temp->result,
                       func) {
}

}  // namespace lang
}  // namespace taichi
