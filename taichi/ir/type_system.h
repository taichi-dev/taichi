#pragma once

#include "taichi/ir/expression.h"

// This file includes the typechecking mechanism for frontend internal ops.
// Each internal op is given a type signature with argument types and a
// return type, and each type may contain type variables that are constrained
// by 'traits', i.e. predicates over types such as "integer type", "real number
// type", etc.
//
// These type signatures are used in a typechecking pass of the frontend AST,
// and any type mismatch will generate principled error messages.

namespace taichi::lang {

// Type errors that arise in the typechecking process.
class TypeSystemError {
 public:
  virtual ~TypeSystemError() = default;
  virtual std::string to_string() const = 0;
};

class TyVar;

// TypeExpression are expressions of types that are used in operation
// signatures. They may contain type variables, "common type" of two type
// expressions, or the "compute type" of a certain type expression.
class TypeExpression {
 public:
  virtual ~TypeExpression() = default;
  // In the typechecking process, we will match the type of argument
  // (a DataType) against the type of parameter (a TypeExpression).
  // Since TypeExpressions contain type variables, we will assign concrete
  // DataTypes to these variables in the process. The solution set of this
  // process is this type; the int in the pair is the parameter position
  // where the type expression is first solved.
  using Solutions = std::map<Identifier, std::pair<DataType, int>>;
  // Unification is the process of matching a DataType against a TypeExpression,
  // and generating Solutions along the way. If unification is impossible,
  // a TypeSystemError will be thrown detailing the error.
  virtual void unify(int pos, DataType dt, Solutions &solutions) const = 0;
  // Resolution is the process of substituting type variables within a
  // TypeExpression, to get a result DataType.
  virtual DataType resolve(Solutions const &solutions) const = 0;
  virtual std::string to_string() const = 0;
  virtual bool contains_tyvar(const TyVar &tyvar) const = 0;
};

// A pointer to a TypeExpression.
using TypeExpr = std::shared_ptr<TypeExpression>;

// A type variable.
class TyVar : public TypeExpression {
  const Identifier name_;

 public:
  explicit TyVar(const Identifier &id) : name_(id) {
  }
  void unify(int pos, DataType dt, Solutions &solutions) const override;
  DataType resolve(Solutions const &solutions) const override;
  std::string to_string() const override;
  bool contains_tyvar(const TyVar &tyvar) const override;
};

// The "common type" of two types according to the C++ type system. This uses
// the promoted_type() function in type_factory.cpp.
//
// Since promoted_type() is not injective, we cannot solve unification
// equations of the form
//
//   promoted_type(TyVar1, TyVar2) === dt;
//
// and this will instead result in an error.
class TyLub : public TypeExpression {
  const TypeExpr lhs_, rhs_;

 public:
  explicit TyLub(TypeExpr lhs, TypeExpr rhs) : lhs_(lhs), rhs_(rhs) {
  }
  void unify(int pos, DataType dt, Solutions &solutions) const override;
  DataType resolve(Solutions const &solutions) const override;
  std::string to_string() const override;
  bool contains_tyvar(const TyVar &tyvar) const override;
};

// The "compute type" of a certain type. This uses Type::get_compute_type()
// under the hood.
//
// Since get_compute_type() is not injective, we cannot solve unification
// equations of the form
//
//   TyVar.get_compute_type() === dt;
//
// and this will instead result in an error.
class TyCompute : public TypeExpression {
  const TypeExpr exp_;

 public:
  explicit TyCompute(TypeExpr exp) : exp_(exp) {
  }
  void unify(int pos, DataType dt, Solutions &solutions) const override;
  DataType resolve(Solutions const &solutions) const override;
  std::string to_string() const override;
  bool contains_tyvar(const TyVar &tyvar) const override;
};

// A "mono-type", i.e. DataType embedded into a type expression.
class TyMono : public TypeExpression {
  const DataType monotype_;

 public:
  explicit TyMono(DataType dt) : monotype_(dt) {
  }
  void unify(int pos, DataType dt, Solutions &solutions) const override;
  DataType resolve(Solutions const &solutions) const override;
  std::string to_string() const override;
  bool contains_tyvar(const TyVar &tyvar) const override;
};

// Type error: a type variable is solved to two different DataTypes.
class TyVarMismatch : public TypeSystemError {
  const int solved_position_, current_position_;
  const DataType original_, conflicting_;

 public:
  explicit TyVarMismatch(int solved_position,
                         int current_position,
                         DataType original,
                         DataType conflicting)
      : solved_position_(solved_position),
        current_position_(current_position),
        original_(original),
        conflicting_(conflicting) {
  }
  std::string to_string() const override;
};

// Type error: a TyMono is unified with a different datatype.
class TypeMismatch : public TypeSystemError {
  const int position_;
  const DataType param_, arg_;

 public:
  explicit TypeMismatch(int pos, DataType param, DataType arg)
      : position_(pos), param_(param), arg_(arg) {
  }
  std::string to_string() const override;
};

// Impossible: a type variable is not solved. This implies that the type
// signature of an operation is malformed.
class TyVarUnsolved : public TypeSystemError {
  const Identifier var_;

 public:
  explicit TyVarUnsolved(const Identifier &var) : var_(var) {
  }
  std::string to_string() const override;
};

// A trait i.e. a predicate over types. This can be used to constrain type
// variables in type signatures.
class Trait {
 public:
  virtual ~Trait() = default;
  virtual bool validate(DataType dt) const = 0;
  virtual std::string to_string() const = 0;
};

// You can construct a trait via a function (DataType) -> bool.
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

// A constraint on a type variable. This states that the DataType that the
// variable is resolved to must satisfy a certain trait.
class Constraint {
 public:
  const std::shared_ptr<TyVar> tyvar;
  Trait *const trait;
  explicit Constraint(std::shared_ptr<TyVar> tyvar, Trait *trait)
      : tyvar(tyvar), trait(trait) {
  }
};

// Type error: the type of argument does not satisfy the trait required. E.g.
// passing in a f32 while the trait requires an integer type.
class TraitMismatch : public TypeSystemError {
  const int occurrence_;
  const DataType dt_;
  const Constraint constraint_;

 public:
  explicit TraitMismatch(int occurrence, DataType dt, Constraint constraint)
      : occurrence_(occurrence), dt_(dt), constraint_(constraint) {
  }
  std::string to_string() const override;
};

// Impossible: the number of arguments does not match the length of the
// parameter list. This often implies some error in the python glue code.
class ArgLengthMismatch : public TypeSystemError {
  const int param_, arg_;

 public:
  explicit ArgLengthMismatch(int param, int arg) : param_(param), arg_(arg) {
  }
  std::string to_string() const override;
};

// A type signature for an operation. This consists of 3 parts:
//
// - Constraints over type variables
// - Parameter list, each parameter being annotated with a type expression
// - The return type as a type expression.
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
  // Check a list of argument types against the type expression. If this fails,
  // some TypeSystemError will be raised.
  DataType type_check(const std::vector<DataType> &arguments) const;
};

enum class StaticTraitID {
  real,       // Real number types, i.e. all float types.
  integral,   // Integer types, including the custom integer types.
  primitive,  // Primitive types. Only ixx, uxx, and fxx types are included.
  scalar,     // Scalar types. This includes all real and integral types.
};

// Static traits are a set of predefined traits that are often used.
class StaticTraits {
 public:
  // Get a certain static trait from its ID.
  static Trait *get(StaticTraitID traitId);

 private:
  inline static std::map<StaticTraitID, std::unique_ptr<Trait>> traits_;
  static void init_traits();
};

// An internal operation. This consists of a display name (used in serialization
// and error messages), a type signature, and a flattening function used in
// frontend-to-IR passes.
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

// Internal operation IDs.
enum class InternalOp {
#define PER_INTERNAL_OP(x) x,
#include "taichi/inc/internal_ops.inc.h"
#undef PER_INTERNAL_OP
};

// The set of internal operations. Any new operation should be defined here.
class Operations {
 public:
  // Get an internal operation from its ID.
  static Operation *get(InternalOp opcode);

 private:
  inline static std::map<InternalOp, std::unique_ptr<Operation>> internals_;
  static void init_internals();
};

}  // namespace taichi::lang
