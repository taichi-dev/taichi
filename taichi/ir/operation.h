#pragma once

#include "taichi/ir/ir.h"
#include "taichi/ir/type.h"
#include "taichi/ir/expression.h"

namespace taichi {
namespace lang {

// a "type expression" that has type variables and type-level functions.
class TypeExpression {
 public:
  // A Solution is a mapping of all tyvars in a type expression to concrete
  // datatypes.
  using Solutions = std::map<Identifier, DataType>;

  class UnifyFailure {
   public:
    virtual std::string to_string() const = 0;
  };

  // Equates the type expr and the given datatype, and solve the concrete type
  // of tyvars along the way.
  virtual void unify(Solutions &solutions, DataType ty) const = 0;

  // Resolve a type expr to a concrete datatype with a set of solutions.
  virtual DataType resolve(const Solutions &solutions) const = 0;

  virtual std::string to_string() const = 0;
};

using TypeExpr = std::shared_ptr<TypeExpression>;

// Any datatype is trivially a type expression.
class DataTypeExpression : public TypeExpression {
  class DatatypeMismatch : public UnifyFailure {
   public:
    const DataType expected, actual;
    std::string to_string() const override;
    DatatypeMismatch(DataType expected, DataType actual);
  };

 public:
  const DataType type;
  DataTypeExpression(DataType ty);

  void unify(Solutions &solutions, DataType ty) const override;
  DataType resolve(const Solutions &solutions) const override;
  std::string to_string() const override;
};

class TyvarTypeExpression;
using Tyvar = std::shared_ptr<TyvarTypeExpression>;

// The class of tyvars.
class TyvarTypeExpression : public TypeExpression {
  class TyvarUnsolvable : public UnifyFailure {
   public:
    const std::string name;
    std::string to_string() const override;
    TyvarUnsolvable(const std::string &name);
  };

 public:
  const Identifier id;
  TyvarTypeExpression(const Identifier &id);

  void unify(Solutions &solutions, DataType ty) const override;
  DataType resolve(const Solutions &solutions) const override;
  std::string to_string() const override;
};

// A type-level function that returns the std::common_type of two primitive
// datatypes.
class CommonTypeExpression : public TypeExpression {
 public:
  const TypeExpr lhs;
  const TypeExpr rhs;
  CommonTypeExpression(TypeExpr lhs, TypeExpr rhs);

  void unify(Solutions &solutions, DataType ty) const override;
  DataType resolve(const Solutions &solutions) const override;
  std::string to_string() const override;
};

// A type-level function that returns the compute type of a custom scalar type.
class ComputeTypeExpression : public TypeExpression {
 public:
  const TypeExpr ty;
  ComputeTypeExpression(TypeExpr ty);

  void unify(Solutions &solutions, DataType ty) const override;
  DataType resolve(const Solutions &solutions) const override;
  std::string to_string() const override;
};

// A convenient builder class for type expressions. Preferably use this instead
// of std::make_shared for TypeExpressions.
class TypeSpec {
 public:
  static std::shared_ptr<DataTypeExpression> dt(DataType dt);
  static Tyvar var(const Identifier &id);
  static Tyvar var(const std::string &name);
  static std::shared_ptr<CommonTypeExpression> lub(TypeExpr lhs, TypeExpr rhs);
  static std::shared_ptr<ComputeTypeExpression> comp(TypeExpr ty);
};

// A trait is a predicate on types.
class Trait {
 public:
  virtual bool has_type(DataType ty) const = 0;
  virtual std::string describe() const = 0;
};

class DynamicTrait : public Trait {
 private:
  using TraitPred = bool(DataType ty);

  const std::string description_;
  const std::function<TraitPred> func_;

 public:
  DynamicTrait(const std::string &description,
               const std::function<TraitPred> &func);

  std::string describe() const override;
  bool has_type(DataType ty) const override;
};

Stmt *flatten_lvalue(Expr expr, Expression::FlattenContext *ctx);

Stmt *flatten_rvalue(Expr expr, Expression::FlattenContext *ctx);

std::vector<Stmt *> get_all_stmts(const std::vector<Expr> &exprs,
                                  Expression::FlattenContext *ctx);

class Operation {
 public:
  // A Constraint describes one tyvar involved in the param and return types
  // and also defines what traits should the tyvar satisfy.
  class Constraint {
   public:
    const Tyvar tyvar;
    const std::vector<Trait *> traits;

    Constraint(Tyvar tyvar);
    Constraint(Tyvar tyvar, Trait *trait);
    Constraint(Tyvar tyvar, const std::vector<Trait *> &traits);
  };

  const std::string name;
  const std::vector<Constraint> constraints;
  const std::vector<TypeExpr> params;
  const TypeExpr result;

  Operation(const std::string &name,
            const std::vector<Constraint> &constraints,
            const std::vector<TypeExpr> &params,
            TypeExpr result);

  DataType check(const std::vector<Expr> &args) const;
  std::string signature_string() const;

  virtual Stmt *flatten(Expression::FlattenContext *ctx,
                        std::vector<Expr> &args) const = 0;

  Expr call(const std::vector<Expr> &args);
  template <typename... Args>
  Expr call(const Args &... args) {
    return call({args...});
  }
};

class DynamicOperation : public Operation {
 public:
  using DynOp = Stmt *(Expression::FlattenContext *ctx,
                       std::vector<Expr> &args);

  DynamicOperation(const std::string &name,
                   const std::vector<Constraint> &tyvars,
                   const std::vector<TypeExpr> &params,
                   TypeExpr result,
                   const std::function<DynOp> &func);
  DynamicOperation(const std::string &name,
                   Operation *temp,
                   const std::function<DynOp> &func);

  Stmt *flatten(Expression::FlattenContext *ctx,
                std::vector<Expr> &args) const override;

 private:
  const std::function<DynOp> func_;
};

}  // namespace lang
}  // namespace taichi
