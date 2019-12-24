#pragma once
#include <taichi/tlang_util.h>

TLANG_NAMESPACE_BEGIN

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

  Expr(Identifier id);

  void set(const Expr &o) {
    expr = o.expr;
  }

  Expression *operator->() {
    return expr.get();
  }

  Expression const *operator->() const {
    return expr.get();
  }

  template <typename T>
  Handle<T> cast() const {
    TC_ASSERT(expr != nullptr);
    return std::dynamic_pointer_cast<T>(expr);
  }

  template <typename T>
  bool is() const {
    return cast<T>() != nullptr;
  }

  Expr &operator=(const Expr &o);

  Expr operator[](ExprGroup) const;

  std::string serialize() const;

  void *evaluate_addr(int i, int j, int k, int l);

  template <typename... Indices>
  void *val_tmp(DataType dt, Indices... indices);

  template <typename T, typename... Indices>
  T &val(Indices... indices);

  template <typename T, typename... Indices>
  void set_val(const T &v, Indices... indices) {
    val<T, Indices...>(indices...) = v;
  }

  void operator+=(const Expr &o);
  void operator-=(const Expr &o);
  void operator*=(const Expr &o);
  void operator/=(const Expr &o);
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
};


TLANG_NAMESPACE_END


