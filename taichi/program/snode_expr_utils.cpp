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

  SNode *adjoint_flag_snode() const override {
    auto &adjoint_flag = glb_var_->adjoint_flag;
    if (adjoint_flag.expr == nullptr) {
      return nullptr;
    }
    return adjoint_flag.snode();
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
    SNode *new_exp_snode = nullptr;
    if (auto qflt = glb_var_expr->dt->cast<QuantFloatType>()) {
      auto exp = qflt->get_exponent_type();
      // Non-empty exponent type. First create a place SNode for the
      // exponent value.
      if (parent->placing_shared_exp &&
          parent->currently_placing_exp_snode != nullptr) {
        // Reuse existing exponent
        TI_ASSERT_INFO(parent->currently_placing_exp_snode_dtype == exp,
                       "QuantFloatTypes with shared exponents must have "
                       "exactly the same exponent type.");
        new_exp_snode = parent->currently_placing_exp_snode;
      } else {
        auto &exp_node = parent->insert_children(SNodeType::place);
        exp_node.dt = exp;
        std::tie(exp_node.id_in_bit_struct, exp_node.bit_offset) =
            parent->bit_struct_type_builder->add_member(exp);
        exp_node.name = glb_var_expr->ident.raw_name() + "_exp";
        new_exp_snode = &exp_node;
        if (parent->placing_shared_exp) {
          parent->currently_placing_exp_snode = new_exp_snode;
          parent->currently_placing_exp_snode_dtype = exp;
        }
      }
    }
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
      std::tie(child.id_in_bit_struct, child.bit_offset) =
          parent->bit_struct_type_builder->add_member(child.dt);
      if (parent->placing_shared_exp) {
        child.owns_shared_exponent = true;
        parent->bit_struct_type_builder->set_member_owns_shared_exponent(
            child.id_in_bit_struct);
      }
      if (new_exp_snode) {
        child.exp_snode = new_exp_snode;
        parent->bit_struct_type_builder->set_member_exponent(
            child.id_in_bit_struct, new_exp_snode->id_in_bit_struct);
        new_exp_snode->exponent_users.push_back(&child);
        parent->bit_struct_type_builder->add_member_exponent_user(
            new_exp_snode->id_in_bit_struct, child.id_in_bit_struct);
      }
    }
    if (!offset.empty())
      child.set_index_offsets(offset);
  }
}

void make_lazy_grad(SNode *snode,
                    SNodeGlobalVarExprMap *snode_to_exprs,
                    bool is_adjoint,
                    bool is_dual,
                    bool is_adjoint_flag) {
  if (snode->type == SNodeType::place)
    return;
  for (auto &c : snode->ch) {
    make_lazy_grad(c.get(), snode_to_exprs, is_adjoint, is_dual,
                   is_adjoint_flag);
  }
  std::vector<Expr> new_grads;
  for (auto &c : snode->ch) {
    bool collected_for_adjoint = false;
    if (is_adjoint) {
      if (c->type == SNodeType::place && c->is_primal() && is_real(c->dt) &&
          !c->has_adjoint()) {
        new_grads.push_back(snode_to_exprs->at(c.get())->adjoint);
        collected_for_adjoint = true;
      }
    }
    if (is_adjoint_flag) {
      // Only allocate adjoint_flag for field with adjoint or ready for
      // allocating adjoint
      if (c->type == SNodeType::place && c->is_primal() && is_real(c->dt) &&
          (c->has_adjoint() || collected_for_adjoint)) {
        new_grads.push_back(snode_to_exprs->at(c.get())->adjoint_flag);
      }
    }
    if (is_dual) {
      if (c->type == SNodeType::place && c->is_primal() && is_real(c->dt) &&
          !c->has_dual()) {
        new_grads.push_back(snode_to_exprs->at(c.get())->dual);
      }
    }
  }
  for (auto p : new_grads) {
    place_child(&p, /*offset=*/{}, snode, snode_to_exprs);
  }
}

}  // namespace lang
}  // namespace taichi
