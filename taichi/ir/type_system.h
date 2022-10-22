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
  const Identifier name_;

 public:
  explicit TyVar(Identifier id) : name_(id) {
  }
  void unify(int pos,
             DataType dt,
             std::map<Identifier, DataType> &solutions) const override;
  DataType resolve(
      std::map<Identifier, DataType> const &solutions) const override;
  std::string to_string() const override;
};

class TyLub : public TypeExpression {
  const TypeExpr lhs_, rhs_;

 public:
  explicit TyLub(TypeExpr lhs, TypeExpr rhs) : lhs_(lhs), rhs_(rhs) {
  }
  void unify(int pos,
             DataType dt,
             std::map<Identifier, DataType> &solutions) const override;
  DataType resolve(
      std::map<Identifier, DataType> const &solutions) const override;
  std::string to_string() const override;
};

class TyCompute : public TypeExpression {
  const TypeExpr exp_;

 public:
  explicit TyCompute(TypeExpr exp) : exp_(exp) {
  }
  void unify(int pos,
             DataType dt,
             std::map<Identifier, DataType> &solutions) const override;
  DataType resolve(
      std::map<Identifier, DataType> const &solutions) const override;
  std::string to_string() const override;
};

class TyMono : public TypeExpression {
  const DataType monotype_;

 public:
  explicit TyMono(DataType dt) : monotype_(dt) {
  }
  void unify(int pos,
             DataType dt,
             std::map<Identifier, DataType> &solutions) const override;
  DataType resolve(
      std::map<Identifier, DataType> const &solutions) const override;
  std::string to_string() const override;
};

class TyVarMismatch : public TypeSystemError {
  const Identifier var_;
  const DataType original_, conflicting_;

 public:
  explicit TyVarMismatch(Identifier var,
                         DataType original,
                         DataType conflicting)
      : var_(var), original_(original), conflicting_(conflicting) {
  }
  std::string to_string() const override;
};

class TypeMismatch : public TypeSystemError {
  int position_;
  const DataType param_, arg_;

 public:
  explicit TypeMismatch(int pos, DataType param, DataType arg)
      : position_(pos), param_(param), arg_(arg) {
  }
  std::string to_string() const override;
};

class TyVarUnsolved : public TypeSystemError {
  const Identifier var_;

 public:
  explicit TyVarUnsolved(Identifier var) : var_(var) {
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
  std::string name_;
  std::function<bool(const DataType dt)> impl_;

 public:
  explicit DynamicTrait(std::string name,
                        std::function<bool(const DataType dt)> impl)
      : name_(name), impl_(impl) {
  }
  bool validate(const DataType dt) const override;
  std::string to_string() const override;
};

class TraitMismatch : public TypeSystemError {
  const DataType dt_;
  const Trait *trait_;

 public:
  explicit TraitMismatch(const DataType dt, const Trait *trait)
      : dt_(dt), trait_(trait) {
  }
  std::string to_string() const override;
};

class ArgLengthMismatch : public TypeSystemError {
  const int param_, arg_;

 public:
  explicit ArgLengthMismatch(int param, int arg) : param_(param), arg_(arg) {
  }
  std::string to_string() const override;
};

class Constraint {
 public:
  const std::shared_ptr<TyVar> tyvar;
  const Trait *trait;
  explicit Constraint(const std::shared_ptr<TyVar> tyvar, const Trait *trait)
      : tyvar(tyvar), trait(trait) {
  }
};

class Signature {
  const std::vector<Constraint> constraints_;
  const std::vector<TypeExpr> parameters_;
  const TypeExpr ret_type_;

 public:
  explicit Signature(std::vector<Constraint> constraints,
                     std::vector<TypeExpr> parameters,
                     TypeExpr ret_type)
      : constraints_(constraints),
        parameters_(parameters),
        ret_type_(ret_type) {
  }
  explicit Signature(std::vector<TypeExpr> parameters, TypeExpr ret_type)
      : parameters_(parameters), ret_type_(ret_type) {
  }
  explicit Signature(TypeExpr ret_type) : ret_type_(ret_type) {
  }
  DataType type_check(std::vector<DataType> arguments) const;
};

class StaticTraits {
 public:
  explicit StaticTraits();
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

  explicit Operation(std::string name, Signature sig) : name(name), sig(sig) {
  }
  virtual ~Operation() = default;

  DataType type_check(std::vector<DataType> arg_types) const;
  virtual Stmt *flatten(Expression::FlattenContext *ctx,
                        std::vector<Expr> args,
                        DataType ret_type) const = 0;
};

class InternalOps {
 public:
  explicit InternalOps();
  static const InternalOps *get();
#define PER_INTERNAL_OP(x) const Operation *x;
#include "taichi/inc/internal_ops.inc.h"
#undef PER_INTERNAL_OP
};

}  // namespace lang
}  // namespace taichi