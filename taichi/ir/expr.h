#pragma once

#include "taichi/util/str.h"
#include "taichi/ir/type_utils.h"

namespace taichi::lang {

struct CompileConfig;
class Expression;
class Identifier;
class ExprGroup;
class SNode;
class ASTBuilder;

class Expr {
 public:
  std::shared_ptr<Expression> expr;
  bool const_value;
  bool atomic;

  Expr() {
    const_value = false;
    atomic = false;
  }

  explicit Expr(uint1 x);

  explicit Expr(int16 x);

  explicit Expr(int32 x);

  explicit Expr(int64 x);

  explicit Expr(float32 x);

  explicit Expr(float64 x);

  explicit Expr(std::shared_ptr<Expression> expr) : Expr() {
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

  explicit Expr(const Identifier &id);

  void set(const Expr &o) {
    expr = o.expr;
  }

  // NOLINTNEXTLINE(google-explicit-constructor)
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

  template <typename T, typename... Args>
  static Expr make(Args &&...args) {
    return Expr(std::make_shared<T>(std::forward<Args>(args)...));
  }

  SNode *snode() const;

  // debug info, contains traceback for type checking error message
  void set_dbg_info(const DebugInfo &dbg_info);

  const std::string &get_tb() const;

  void set_adjoint(const Expr &o);

  void set_dual(const Expr &o);

  void set_adjoint_checkbit(const Expr &o);

  DataType get_ret_type() const;

  DataType get_rvalue_type() const;

  void type_check(const CompileConfig *config);
};

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

// like Expr::Expr, but allows to explicitly specify the type
template <typename T>
Expr value(const T &val) {
  return Expr(val);
}

Expr expr_rand(DataType dt);

template <typename T>
Expr expr_rand() {
  return taichi::lang::expr_rand(get_data_type<T>());
}

Expr assume_range(const Expr &expr,
                  const Expr &base,
                  int low,
                  int high,
                  const DebugInfo &dbg_info = DebugInfo());

Expr loop_unique(const Expr &input,
                 const std::vector<SNode *> &covers,
                 const DebugInfo &dbg_info = DebugInfo());

Expr expr_field(Expr id_expr, DataType dt);

Expr expr_matrix_field(const std::vector<Expr> &fields,
                       const std::vector<int> &element_shape);

}  // namespace taichi::lang
