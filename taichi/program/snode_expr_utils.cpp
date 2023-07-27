#include "taichi/program/snode_expr_utils.h"
#include "taichi/ir/snode.h"
#include "taichi/ir/frontend_ir.h"

namespace taichi::lang {

namespace {

class GradInfoImpl final : public SNode::GradInfoProvider {
 public:
  explicit GradInfoImpl(FieldExpression *field) : field_(field) {
  }

  bool is_primal() const override {
    return field_->snode_grad_type == SNodeGradType::kPrimal;
  }

  SNodeGradType get_snode_grad_type() const override {
    return field_->snode_grad_type;
  }

  SNode *adjoint_snode() const override {
    auto &adj = field_->adjoint;
    if (adj.expr == nullptr) {
      return nullptr;
    }
    return adj.snode();
  }

  SNode *dual_snode() const override {
    auto &dual = field_->dual;
    if (dual.expr == nullptr) {
      return nullptr;
    }
    return dual.snode();
  }

  SNode *adjoint_checkbit_snode() const override {
    auto &adjoint_checkbit = field_->adjoint_checkbit;
    if (adjoint_checkbit.expr == nullptr) {
      return nullptr;
    }
    return adjoint_checkbit.snode();
  }

 private:
  FieldExpression *field_;
};

}  // namespace

void place_child(Expr *expr_arg,
                 const std::vector<int> &offset,
                 int id_in_bit_struct,
                 SNode *parent,
                 SNodeFieldMap *snode_to_exprs) {
  if (parent->type == SNodeType::root) {
    // never directly place to root
    auto &ds = parent->dense(std::vector<Axis>(), {});
    place_child(expr_arg, offset, id_in_bit_struct, &ds, snode_to_exprs);
  } else {
    TI_ASSERT(expr_arg->is<FieldExpression>());
    auto field = expr_arg->cast<FieldExpression>();
    TI_ERROR_IF(field->snode != nullptr, "This variable has been placed.");
    auto &child = parent->insert_children(SNodeType::place);
    field->set_snode(&child);
    if (field->name == "") {
      child.name = field->ident.raw_name();
    } else {
      child.name = field->name;
    }
    if (field->has_ambient) {
      field->snode->has_ambient = true;
      field->snode->ambient_val = field->ambient_value;
    }
    field->snode->grad_info = std::make_unique<GradInfoImpl>(field.get());
    (*snode_to_exprs)[field->snode] = field;
    child.dt = field->dt;
    child.id_in_bit_struct = id_in_bit_struct;
    if (!offset.empty())
      child.set_index_offsets(offset);
  }
}

void make_lazy_place(SNode *snode,
                     SNodeFieldMap *snode_to_fields,
                     const std::function<void(std::unique_ptr<SNode> &,
                                              std::vector<Expr> &)> &collect) {
  if (snode->type == SNodeType::place)
    return;
  for (auto &c : snode->ch) {
    make_lazy_place(c.get(), snode_to_fields, collect);
  }
  std::vector<Expr> new_places;
  for (auto &c : snode->ch) {
    collect(c, new_places);
  }
  for (auto p : new_places) {
    place_child(&p, /*offset=*/{}, -1, snode, snode_to_fields);
  }
}

}  // namespace taichi::lang
