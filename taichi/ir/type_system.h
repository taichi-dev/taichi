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
  explicit TyVar(const Identifier &id) : name_(id) {
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
  explicit TyVarMismatch(const Identifier &var,
                         DataType original,
                         DataType conflicting)
      : var_(var), original_(original), conflicting_(conflicting) {
  }
  std::string to_string() const override;
};

class TypeMismatch : public TypeSystemError {
  const int position_;
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
  explicit TyVarUnsolved(const Identifier &var) : var_(var) {
  }
  std::string to_string() const override;
};

class Trait {
 public:
  virtual ~Trait() = default;
  virtual bool validate(DataType dt) const = 0;
  virtual std::string to_string() const = 0;
};

class DynamicTrait : public Trait {
 private:
  const std::string name_;
  const std::function<bool(DataType dt)> impl_;

 public:
  explicit DynamicTrait(const std::string &name,
                        const std::function<bool(DataType dt)> &impl)
      : name_(name), impl_(impl) {
  }
  bool validate(DataType dt) const override;
  std::string to_string() const override;
};

class TraitMismatch : public TypeSystemError {
  const DataType dt_;
  Trait const *trait_;

 public:
  explicit TraitMismatch(DataType dt, Trait *trait) : dt_(dt), trait_(trait) {
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
  Trait *trait;
  explicit Constraint(std::shared_ptr<TyVar> tyvar, Trait *trait)
      : tyvar(tyvar), trait(trait) {
  }
};

class Signature {
  const std::vector<Constraint> constraints_;
  const std::vector<TypeExpr> parameters_;
  const TypeExpr ret_type_;

 public:
  explicit Signature(const std::vector<Constraint> &constraints,
                     const std::vector<TypeExpr> &parameters,
                     TypeExpr ret_type)
      : constraints_(constraints),
        parameters_(parameters),
        ret_type_(ret_type) {
  }
  explicit Signature(const std::vector<TypeExpr> &parameters, TypeExpr ret_type)
      : parameters_(parameters), ret_type_(ret_type) {
  }
  explicit Signature(TypeExpr ret_type) : ret_type_(ret_type) {
  }
  DataType type_check(const std::vector<DataType> &arguments) const;
};

enum class StaticTraitID {
  real,
  integral,
  primitive,
  scalar,
};

class StaticTraits {
 public:
  static Trait *get(StaticTraitID traitId);

 private:
  inline static std::map<StaticTraitID, std::unique_ptr<Trait>> traits_;
  static void init_traits();
};

class Operation {
 public:
  const std::string name;
  const Signature sig;

  explicit Operation(const std::string &name, const Signature &sig)
      : name(name), sig(sig) {
  }
  virtual ~Operation() = default;

  DataType type_check(const std::vector<DataType> &arg_types) const;
  virtual Stmt *flatten(Expression::FlattenContext *ctx,
                        const std::vector<Expr> &args,
                        DataType ret_type) const = 0;
};

enum class InternalOp {
#define PER_INTERNAL_OP(x) x,
#include "taichi/inc/internal_ops.inc.h"
#undef PER_INTERNAL_OP
};

class Operations {
 public:
  static Operation *get(InternalOp opcode);

 private:
  inline static std::map<InternalOp, std::unique_ptr<Operation>> internals_;
  static void init_internals();
};

}  // namespace lang
}  // namespace taichi
