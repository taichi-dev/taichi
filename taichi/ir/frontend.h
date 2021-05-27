// Frontend constructs
#pragma once

#include "taichi/common/core.h"
#include "taichi/common/dict.h"
#include "taichi/ir/frontend_ir.h"

namespace taichi {
static_assert(
    sizeof(real) == sizeof(float32),
    "Please build the taichi compiler with single precision (TI_USE_DOUBLE=0)");
namespace math {
inline int maximum(int a) {
  return a;
}
}  // namespace math
}  // namespace taichi
#include <set>
#if defined(TI_PLATFORM_UNIX)
#include <dlfcn.h>
#endif

#include "taichi/lang_util.h"
#include "taichi/program/program.h"

TLANG_NAMESPACE_BEGIN

void layout(const std::function<void()> &body);

inline Kernel &kernel(const std::function<void()> &body) {
  return get_current_program().kernel(body);
}

inline void kernel_name(std::string name) {
  get_current_program().get_current_kernel().name = name;
}

template <typename T>
inline void declare_var(Expr &a) {
  current_ast_builder().insert(std::make_unique<FrontendAllocaStmt>(
      std::static_pointer_cast<IdExpression>(a.expr)->id, get_data_type<T>()));
}

inline void declare_unnamed_var(Expr &a, DataType dt) {
  auto id = Identifier();
  auto a_ = Expr::make<IdExpression>(id);

  current_ast_builder().insert(std::make_unique<FrontendAllocaStmt>(id, dt));

  if (a.expr) {
    a_ = a;  // assign
  }

  a.set(a_);
}

inline void declare_var(Expr &a) {
  current_ast_builder().insert(std::make_unique<FrontendAllocaStmt>(
      std::static_pointer_cast<IdExpression>(a.expr)->id,
      PrimitiveType::unknown));
}

inline void set_ambient(Expr expr_, float32 val) {
  auto expr = expr_.cast<GlobalVariableExpression>();
  expr->ambient_value = TypedConstant(val);
  expr->has_ambient = true;
}

inline void set_ambient(Expr expr_, int32 val) {
  auto expr = expr_.cast<GlobalVariableExpression>();
  expr->ambient_value = TypedConstant(val);
  expr->has_ambient = true;
}

Expr global_new(Expr id_expr, DataType dt);

Expr global_new(DataType dt, std::string name = "");

template <typename T>
Expr Rand() {
  return Expr::make<RandExpression>(get_data_type<T>());
}

template <typename T>
T Eval(const T &t) {
  return t.eval();
}

Expr copy(const Expr &expr);

template <typename... indices>
std::vector<Index> Indices(indices... ind) {
  auto ind_vec = std::vector<int>({ind...});
  std::vector<Index> ret;
  for (auto in : ind_vec) {
    ret.push_back(Index(in));
  }
  return ret;
}

inline Expr Atomic(Expr dest) {
  // NOTE: dest must be passed by value so that the original
  // expr will not be modified into an atomic one.
  dest.atomic = true;
  return dest;
}

// expr_group are indices
inline void Activate(SNode *snode, const ExprGroup &expr_group) {
  current_ast_builder().insert(Stmt::make<FrontendSNodeOpStmt>(
      SNodeOpType::activate, snode, expr_group));
}

inline void Activate(const Expr &expr, const ExprGroup &expr_group) {
  return Activate(expr.snode(), expr_group);
}

inline void Deactivate(SNode *snode, const ExprGroup &expr_group) {
  current_ast_builder().insert(Stmt::make<FrontendSNodeOpStmt>(
      SNodeOpType::deactivate, snode, expr_group));
}

inline Expr Append(SNode *snode, const ExprGroup &indices, const Expr &val) {
  return Expr::make<SNodeOpExpression>(snode, SNodeOpType::append, indices,
                                       val);
}

inline Expr Append(const Expr &expr,
                   const ExprGroup &indices,
                   const Expr &val) {
  return Append(expr.snode(), indices, val);
}

inline void InsertAssert(const std::string &text, const Expr &cond) {
  current_ast_builder().insert(Stmt::make<FrontendAssertStmt>(cond, text));
}

inline void Clear(SNode *snode, const ExprGroup &indices) {
  current_ast_builder().insert(
      Stmt::make<FrontendSNodeOpStmt>(SNodeOpType::clear, snode, indices));
}

inline Expr is_active(SNode *snode, const ExprGroup &indices) {
  return Expr::make<SNodeOpExpression>(snode, SNodeOpType::is_active, indices);
}

inline void Clear(const Expr &expr, const ExprGroup &indices) {
  return Clear(expr.snode(), indices);
}

inline Expr Length(SNode *snode, const ExprGroup &indices) {
  return Expr::make<SNodeOpExpression>(snode, SNodeOpType::length, indices);
}

inline Expr Length(const Expr &expr, const ExprGroup &indices) {
  return Length(expr.snode(), indices);
}

inline Expr AssumeInRange(const Expr &expr,
                          const Expr &base,
                          int low,
                          int high) {
  return Expr::make<RangeAssumptionExpression>(expr, base, low, high);
}

inline Expr LoopUnique(const Expr &input, const std::vector<SNode *> &covers) {
  return Expr::make<LoopUniqueExpression>(load_if_ptr(input), covers);
}

void insert_snode_access_flag(SNodeAccessFlag v, const Expr &field);

// Begin: legacy frontend constructs

class If {
 public:
  FrontendIfStmt *stmt;

  explicit If(const Expr &cond);

  If(const Expr &cond, const std::function<void()> &func);

  If &Then(const std::function<void()> &func);

  If &Else(const std::function<void()> &func);
};

class For {
 public:
  For(const Expr &i,
      const Expr &s,
      const Expr &e,
      const std::function<void()> &func);

  For(const ExprGroup &i,
      const Expr &global,
      const std::function<void()> &func);

  For(const Expr &s, const Expr &e, const std::function<void(Expr)> &func);
};

class While {
 public:
  While(const Expr &cond, const std::function<void()> &func);
};

// End: legacy frontend constructs

TLANG_NAMESPACE_END

TI_NAMESPACE_BEGIN
inline Dict parse_param(std::vector<std::string> cli_param) {
  Dict dict;
  for (auto &s : cli_param) {
    auto div = s.find('=');
    if (div == std::string::npos) {
      TI_INFO("CLI parameter format: key=value, e.g. file_name=test.bin.");
      exit(-1);
    }
    dict.set(s.substr(0, div), s.substr(div + 1));
  }
  TI_P(dict);
  return dict;
}

TI_NAMESPACE_END
