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

template <typename... AX>
std::vector<Axis> make_axes(AX... axes) {
  auto ax_vec = std::vector<int>({axes...});
  std::vector<Axis> ret;
  for (auto ax : ax_vec) {
    ret.push_back(Axis(ax));
  }
  return ret;
}

// expr_group are indices
inline pStmt snode_activate(SNode *snode, const ExprGroup &expr_group) {
  return Stmt::make<FrontendSNodeOpStmt>(SNodeOpType::activate, snode,
                                         expr_group);
}

inline pStmt snode_deactivate(SNode *snode, const ExprGroup &expr_group) {
  return Stmt::make<FrontendSNodeOpStmt>(SNodeOpType::deactivate, snode,
                                         expr_group);
}

inline Expr snode_append(SNode *snode,
                         const ExprGroup &indices,
                         const Expr &val) {
  return Expr::make<SNodeOpExpression>(snode, SNodeOpType::append, indices,
                                       val);
}

inline Expr snode_append(const Expr &expr,
                         const ExprGroup &indices,
                         const Expr &val) {
  return snode_append(expr.snode(), indices, val);
}

inline Expr snode_is_active(SNode *snode, const ExprGroup &indices) {
  return Expr::make<SNodeOpExpression>(snode, SNodeOpType::is_active, indices);
}

inline Expr snode_length(SNode *snode, const ExprGroup &indices) {
  return Expr::make<SNodeOpExpression>(snode, SNodeOpType::length, indices);
}

inline Expr snode_get_addr(SNode *snode, const ExprGroup &indices) {
  return Expr::make<SNodeOpExpression>(snode, SNodeOpType::get_addr, indices);
}

inline Expr snode_length(const Expr &expr, const ExprGroup &indices) {
  return snode_length(expr.snode(), indices);
}

inline Expr assume_range(const Expr &expr,
                         const Expr &base,
                         int low,
                         int high) {
  return Expr::make<RangeAssumptionExpression>(expr, base, low, high);
}

inline Expr loop_unique(const Expr &input, const std::vector<SNode *> &covers) {
  return Expr::make<LoopUniqueExpression>(input, covers);
}
TLANG_NAMESPACE_END
