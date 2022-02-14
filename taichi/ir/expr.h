#pragma once

#include "taichi/util/str.h"
#include "taichi/ir/type_utils.h"

TLANG_NAMESPACE_BEGIN

struct CompileConfig;
class Expression;
class Identifier;
class ExprGroup;
class SNode;

class Expr {
 public:
  std::shared_ptr<Expression> expr;
  bool const_value;
  bool atomic;

  Expr() {
    const_value = false;
    atomic = false;
  }

  Expr(int32 x);

  Expr(int64 x);

  Expr(float32 x);

  Expr(float64 x);

  Expr(std::shared_ptr<Expression> expr) : Expr() {
    this->expr = expr;
  }

  Expr(const Expr &o) : Expr() {
    set(o);
    const_value = o.const_value;
  }

  Expr(Expr &&o) : Expr() {
    set(o);
    const_value = o.const_value;
    atomic = o.atomic;
  }

  Expr(const Identifier &id);

  void set(const Expr &o) {
    expr = o.expr;
  }

  operator bool() const {
    return expr.get() != nullptr;
  }

  Expression *operator->() {
    return expr.get();
  }

  Expression const *operator->() const {
    return expr.get();
  }

  template <typename T>
  std::shared_ptr<T> cast() const {
    TI_ASSERT(expr != nullptr);
    return std::dynamic_pointer_cast<T>(expr);
  }

  template <typename T>
  bool is() const {
    return cast<T>() != nullptr;
  }

  // FIXME: We really should disable it completely,
  // but we can't. This is because the usage of
  // std::variant<Expr, std::string> in FrontendPrintStmt.
  Expr &operator=(const Expr &o);

  Expr operator[](const ExprGroup &indices) const;

  std::string serialize() const;
  void serialize(std::ostream &ss) const;

  Expr operator!();

  Expr eval() const;

  template <typename T, typename... Args>
  static Expr make(Args &&... args) {
    return Expr(std::make_shared<T>(std::forward<Args>(args)...));
  }

  Expr parent() const;

  SNode *snode() const;

  void declare(DataType dt);

  // traceback for type checking error message
  void set_tb(const std::string &tb);

  void set_grad(const Expr &o);

  void set_attribute(const std::string &key, const std::string &value);

  std::string get_attribute(const std::string &key) const;

  DataType get_ret_type() const;

  void type_check(CompileConfig *config);
};

Expr select(const Expr &cond, const Expr &true_val, const Expr &false_val);

Expr operator-(const Expr &expr);

Expr operator~(const Expr &expr);

// Value cast
Expr cast(const Expr &input, DataType dt);

template <typename T>
Expr cast(const Expr &input) {
  return taichi::lang::cast(input, get_data_type<T>());
}

Expr bit_cast(const Expr &input, DataType dt);

template <typename T>
Expr bit_cast(const Expr &input) {
  return taichi::lang::bit_cast(input, get_data_type<T>());
}

TLANG_NAMESPACE_END
