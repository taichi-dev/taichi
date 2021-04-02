#pragma once

#include <memory>
#include <unordered_map>

#include "taichi/ir/snode.h"
#include "taichi/ir/frontend_ir.h"

// This file groups the set of helpers that need the Expr associated with a
// given SNode. Expr is part of the frontend, which somehow depends on the
// files inside "taichi/program". To make SNode fairly low-level and depend
// on less, we thus move SNode-Expr related utils away from SNode itself.
namespace taichi {
namespace lang {

using SNodeGlobalVarExprMap =
    std::unordered_map<const SNode *,
                       std::shared_ptr<GlobalVariableExpression>>;

void place_child(Expr *expr_arg,
                 const std::vector<int> &offset,
                 SNode *parent,
                 SNodeGlobalVarExprMap *snode_to_exprs);

void make_lazy_grad(SNode *snode, SNodeGlobalVarExprMap *snode_to_exprs);

}  // namespace lang
}  // namespace taichi
