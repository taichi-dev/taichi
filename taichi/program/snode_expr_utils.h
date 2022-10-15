#pragma once

#include <memory>
#include <unordered_map>
#include <functional>
#include <vector>

// This file groups the set of helpers that need the Expr associated with a
// given SNode. Expr is part of the frontend, which somehow depends on the
// files inside "taichi/program". To make SNode fairly low-level and depend
// on less, we thus move SNode-Expr related utils away from SNode itself.
namespace taichi::lang {
class Expr;
class SNode;
class FieldExpression;

using SNodeFieldMap =
    std::unordered_map<const SNode *, std::shared_ptr<FieldExpression>>;

void place_child(Expr *expr_arg,
                 const std::vector<int> &offset,
                 int id_in_bit_struct,
                 SNode *parent,
                 SNodeFieldMap *snode_to_exprs);

void make_lazy_place(SNode *snode,
                     SNodeFieldMap *snode_to_fields,
                     const std::function<void(std::unique_ptr<SNode> &,
                                              std::vector<Expr> &)> &collect);

}  // namespace taichi::lang
