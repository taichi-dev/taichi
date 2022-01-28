#pragma once

#include <string>
#include <vector>

#include "taichi/ir/type.h"
#include "taichi/ir/expression.h"
#include "taichi/ir/statements.h"

namespace taichi {
namespace lang {

// a "type expression" that has type variables and type-level functions.
class TypeExpression {
 public:
  // A Solution is a mapping of all tyvars in a type expression to concrete
  // datatypes.
  using Solutions = std::map<Identifier, DataType>;

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
 public:
  const DataType type;
  DataTypeExpression(DataType ty);

  void unify(Solutions &solutions, DataType ty) const override;
  DataType resolve(const Solutions &solutions) const override;
  std::string to_string() const override;
};

// The class of tyvars.
class TyvarTypeExpression : public TypeExpression {
 public:
  const Identifier id;
  TyvarTypeExpression(const Identifier &id);

  void unify(Solutions &solutions, DataType ty) const override;
  DataType resolve(const Solutions &solutions) const override;
  std::string to_string() const override;
};

using Tyvar = std::shared_ptr<TyvarTypeExpression>;

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

// A convenient builder class for type expressions. Preferably use this instead
// of std::make_shared for TypeExpressions.
class TypeSpec {
 public:
  static std::shared_ptr<DataTypeExpression> dt(DataType dt);
  static Tyvar var(const Identifier &id);
  static Tyvar var(const std::string &name);
  static std::shared_ptr<CommonTypeExpression> lub(TypeExpr lhs, TypeExpr rhs);
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

void flatten_lvalue(Expr expr, Expression::FlattenContext *ctx);

void flatten_rvalue(Expr expr, Expression::FlattenContext *ctx);

std::vector<Stmt *> get_all_stmts(const std::vector<Expr> &exprs);

class Operation {
  virtual Stmt *flatten(Expression::FlattenContext *ctx,
                        std::vector<Expr> &args) const = 0;

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

  enum ValueType {
    LValue,
    RValue,
  };

  // A Param can be either an lvalue or an rvalue.
  class Param {
   public:
    const ValueType value_type = RValue;
    const TypeExpr type_expr;

    Param(TypeExpr ty);
    Param(ValueType vt, TypeExpr ty);
  };

  const std::string name;
  const std::vector<Constraint> constraints;
  const std::vector<Param> params;
  const TypeExpr result;

  Operation(const std::string &name,
            const std::vector<Constraint> &constraints,
            const std::vector<Param> &params,
            TypeExpr result);

  Stmt *lower(Expression::FlattenContext *ctx, std::vector<Expr> &args) const;
  DataType check(const std::vector<Expr> &args) const;
};

// This class corresponds to InternalFuncStmt.
class InternalCallOperation : public Operation {
  const std::string internal_call_name_;

  Stmt *flatten(Expression::FlattenContext *ctx,
                std::vector<Expr> &args) const override;

 public:
  InternalCallOperation(const std::string &name,
                        const std::string &internal_name,
                        const std::vector<DataType> &params,
                        DataType result);
};

class DynamicOperation : public Operation {
  using DynOp = Stmt *(Expression::FlattenContext *ctx,
                       const std::vector<Expr> &stmts);
  const std::function<DynOp> func_;

  Stmt *flatten(Expression::FlattenContext *ctx,
                std::vector<Expr> &stmts) const override;

 public:
  DynamicOperation(const std::string &name,
                   const std::vector<Constraint> &tyVars,
                   const std::vector<Param> &params,
                   TypeExpr result,
                   const std::function<DynOp> &func);
};

class StaticTraits {
  StaticTraits();
  static inline std::unique_ptr<StaticTraits> instance_;

 public:
  Trait *primitive;
  Trait *custom;
  Trait *scalar;
  Trait *real;
  Trait *integral;

  static const StaticTraits &get() {
    if (!instance_)
      instance_ = std::make_unique<StaticTraits>(StaticTraits());
    return *instance_;
  }
};

class InternalOps {
  InternalOps();
  static inline std::unique_ptr<InternalOps> instance_;

 public:
  Operation *thread_index;
  Operation *insert_triplet;
  Operation *do_nothing;
  Operation *refresh_counter;

  static const InternalOps &get() {
    if (!instance_)
      instance_ = std::make_unique<InternalOps>(InternalOps());
    return *instance_;
  }
};

class TestInternalOps {
  TestInternalOps();
  static inline std::unique_ptr<TestInternalOps> instance_;

 public:
  Operation *test_stack;
  Operation *test_active_mask;
  Operation *test_shfl;
  Operation *test_list_manager;
  Operation *test_node_allocator;
  Operation *test_node_allocator_gc_cpu;
  Operation *test_internal_func_args;

  static const TestInternalOps &get() {
    if (!instance_)
      instance_ = std::make_unique<TestInternalOps>(TestInternalOps());
    return *instance_;
  }
};

}  // namespace lang
}  // namespace taichi
