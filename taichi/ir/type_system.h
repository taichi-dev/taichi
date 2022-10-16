#pragma once

#include "taichi/ir/expression.h"

namespace taichi {
namespace lang {

class TypeSystemError {
 public:
  virtual ~TypeSystemError() = default;
  virtual std::string to_string() const = 0;
};

class TyVar;

class TypeExpression {
 public:
  virtual ~TypeExpression() = default;
  virtual void unify(int pos,
                     DataType dt,
                     std::map<Identifier, DataType> &solutions) const = 0;
  virtual DataType resolve(
      std::map<Identifier, DataType> const &solutions) const = 0;
  virtual std::string to_string() const = 0;
};

using TypeExpr = std::shared_ptr<TypeExpression>;

class TyVar : public TypeExpression {
  const Identifier name;

 public:
  TyVar(Identifier id) : name(id) {
  }
  void unify(int pos,
             DataType dt,
             std::map<Identifier, DataType> &solutions) const override;
  DataType resolve(
      std::map<Identifier, DataType> const &solutions) const override;
  std::string to_string() const override;
};

class TyLub : public TypeExpression {
  const TypeExpr lhs, rhs;

 public:
  TyLub(TypeExpr lhs, TypeExpr rhs) : lhs(lhs), rhs(rhs) {
  }
  void unify(int pos,
             DataType dt,
             std::map<Identifier, DataType> &solutions) const override;
  DataType resolve(
      std::map<Identifier, DataType> const &solutions) const override;
  std::string to_string() const override;
};

class TyCompute : public TypeExpression {
  const TypeExpr exp;

 public:
  TyCompute(TypeExpr exp) : exp(exp) {
  }
  void unify(int pos,
             DataType dt,
             std::map<Identifier, DataType> &solutions) const override;
  DataType resolve(
      std::map<Identifier, DataType> const &solutions) const override;
  std::string to_string() const override;
};

class TyMono : public TypeExpression {
  const DataType monotype;

 public:
  TyMono(DataType dt) : monotype(dt) {
  }
  void unify(int pos,
             DataType dt,
             std::map<Identifier, DataType> &solutions) const override;
  DataType resolve(
      std::map<Identifier, DataType> const &solutions) const override;
  std::string to_string() const override;
};

class TyVarMismatch : public TypeSystemError {
  const Identifier var;
  const DataType original, conflicting;

 public:
  TyVarMismatch(Identifier var, DataType original, DataType conflicting)
      : var(var), original(original), conflicting(conflicting) {
  }
  std::string to_string() const override;
};

class TypeMismatch : public TypeSystemError {
  int position;
  const DataType param, arg;

 public:
  TypeMismatch(int pos, DataType param, DataType arg)
      : position(pos), param(param), arg(arg) {
  }
  std::string to_string() const override;
};

class TyVarUnsolved : public TypeSystemError {
  const Identifier var;

 public:
  TyVarUnsolved(Identifier var) : var(var) {
  }
  std::string to_string() const override;
};

class Trait {
 public:
  virtual ~Trait() = default;
  virtual bool validate(const DataType dt) const = 0;
  virtual std::string to_string() const = 0;
};

class DynamicTrait : public Trait {
 private:
  std::string name;
  std::function<bool(const DataType dt)> impl;

 public:
  DynamicTrait(std::string name, std::function<bool(const DataType dt)> impl)
      : name(name), impl(impl) {
  }
  bool validate(const DataType dt) const override;
  std::string to_string() const override;
};

class TraitMismatch : public TypeSystemError {
  const DataType dt;
  const Trait *trait;

 public:
  TraitMismatch(const DataType dt, const Trait *trait) : dt(dt), trait(trait) {
  }
  std::string to_string() const override;
};

class ArgLengthMismatch : public TypeSystemError {
  const int param, arg;

 public:
  ArgLengthMismatch(int param, int arg) : param(param), arg(arg) {
  }
  std::string to_string() const override;
};

class Constraint {
 public:
  const std::shared_ptr<TyVar> tyvar;
  const Trait *trait;
  Constraint(const std::shared_ptr<TyVar> tyvar, const Trait *trait)
      : tyvar(tyvar), trait(trait) {
  }
};

class Signature {
  const std::vector<Constraint> constraints;
  const std::vector<TypeExpr> parameters;
  const TypeExpr ret_type;

 public:
  Signature(std::vector<Constraint> constraints,
            std::vector<TypeExpr> parameters,
            TypeExpr ret_type)
      : constraints(constraints), parameters(parameters), ret_type(ret_type) {
  }
  Signature(std::vector<TypeExpr> parameters, TypeExpr ret_type)
      : parameters(parameters), ret_type(ret_type) {
  }
  Signature(TypeExpr ret_type) : ret_type(ret_type) {
  }
  DataType type_check(std::vector<DataType> arguments) const;
};

class StaticTraits {
 public:
  StaticTraits();
  static const StaticTraits *get();
  const Trait *real;
  const Trait *integral;
  const Trait *primitive;
  const Trait *scalar;
};

class Operation {
 public:
  const std::string name;
  const Signature sig;

  Operation(std::string name, Signature sig) : name(name), sig(sig) {
  }
  virtual ~Operation() = default;

  DataType type_check(std::vector<DataType> arg_types) const;
  virtual Stmt *flatten(Expression::FlattenContext *ctx,
                        std::vector<Expr> args,
                        DataType ret_type) const = 0;
};

class InternalOps {
 public:
  InternalOps();
  static const InternalOps *get();
#define PER_INTERNAL_OP(x) const Operation *x;
#include "taichi/inc/internal_ops.inc.h"
#undef PER_INTERNAL_OP
};

class InternalTestOps {};

}  // namespace lang
}  // namespace taichi