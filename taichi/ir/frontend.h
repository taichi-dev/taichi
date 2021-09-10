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

template <typename... AX>
std::vector<Axis> Axes(AX... axes) {
  auto ax_vec = std::vector<int>({axes...});
  std::vector<Axis> ret;
  for (auto ax : ax_vec) {
    ret.push_back(Axis(ax));
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

void reset_snode_access_flag();

// Begin: legacy frontend constructs

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

// End: legacy frontend constructs

TLANG_NAMESPACE_END
