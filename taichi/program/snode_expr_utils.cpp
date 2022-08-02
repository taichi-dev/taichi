#include "taichi/program/snode_expr_utils.h"
#include "taichi/ir/snode.h"
#include "taichi/ir/frontend_ir.h"

namespace taichi {
namespace lang {

namespace {

class GradInfoImpl final : public SNode::GradInfoProvider {
 public:
  explicit GradInfoImpl(GlobalVariableExpression *glb_var) : glb_var_(glb_var) {
  }

  bool is_primal() const override {
    return glb_var_->is_primal;
  }

  SNode *adjoint_snode() const override {
    auto &adj = glb_var_->adjoint;
    if (adj.expr == nullptr) {
      return nullptr;
    }
    return adj.snode();
  }

  SNode *dual_snode() const override {
    auto &dual = glb_var_->dual;
    if (dual.expr == nullptr) {
      return nullptr;
    }
    return dual.snode();
  }

  SNode *adjoint_visited_snode() const override {
    auto &adjoint_visited = glb_var_->adjoint_visited;
    if (adjoint_visited.expr == nullptr) {
      return nullptr;
    }
    return adjoint_visited.snode();
  }

 private:
  GlobalVariableExpression *glb_var_;
};

}  // namespace

void place_child(Expr *expr_arg,
                 const std::vector<int> &offset,
                 SNode *parent,
                 SNodeGlobalVarExprMap *snode_to_exprs) {
  if (parent->type == SNodeType::root) {
    // never directly place to root
    auto &ds = parent->dense(std::vector<Axis>(), {}, false);
    place_child(expr_arg, offset, &ds, snode_to_exprs);
  } else {
    TI_ASSERT(expr_arg->is<GlobalVariableExpression>());
    auto glb_var_expr = expr_arg->cast<GlobalVariableExpression>();
    TI_ERROR_IF(glb_var_expr->snode != nullptr,
                "This variable has been placed.");
    auto &child = parent->insert_children(SNodeType::place);
    glb_var_expr->set_snode(&child);
    if (glb_var_expr->name == "") {
      child.name = glb_var_expr->ident.raw_name();
    } else {
      child.name = glb_var_expr->name;
    }
    if (glb_var_expr->has_ambient) {
      glb_var_expr->snode->has_ambient = true;
      glb_var_expr->snode->ambient_val = glb_var_expr->ambient_value;
    }
    glb_var_expr->snode->grad_info =
        std::make_unique<GradInfoImpl>(glb_var_expr.get());
    (*snode_to_exprs)[glb_var_expr->snode] = glb_var_expr;
    child.dt = glb_var_expr->dt;
    if (parent->bit_struct_type_builder) {
      child.id_in_bit_struct =
          parent->bit_struct_type_builder->add_member(child.dt);
    }
    if (!offset.empty())
      child.set_index_offsets(offset);
  }
}

void make_lazy_place(
    SNode *snode,
    SNodeGlobalVarExprMap *snode_to_exprs,
    const std::function<void(std::unique_ptr<SNode> &,
                             std::vector<Expr> &,
                             SNodeGlobalVarExprMap *snode_to_exprs)> &collect) {
  if (snode->type == SNodeType::place)
    return;
  for (auto &c : snode->ch) {
    make_lazy_place(c.get(), snode_to_exprs, collect);
  }
  std::vector<Expr> new_places;
  for (auto &c : snode->ch) {
    collect(c, new_places, snode_to_exprs);
  }
  for (auto p : new_places) {
    place_child(&p, /*offset=*/{}, snode, snode_to_exprs);
  }
}

}  // namespace lang
}  // namespace taichi
