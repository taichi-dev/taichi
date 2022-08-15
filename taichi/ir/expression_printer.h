#pragma once

#include "taichi/ir/expr.h"
#include "taichi/ir/expression.h"
#include "taichi/ir/frontend_ir.h"
#include "taichi/program/program.h"
#include "taichi/analysis/offline_cache_util.h"

namespace taichi {
namespace lang {

class ExpressionPrinter : public ExpressionVisitor {
 public:
  ExpressionPrinter(std::ostream *os = nullptr) : os_(os) {
  }

  void set_ostream(std::ostream *os) {
    os_ = os;
  }

  std::ostream *get_ostream() {
    return os_;
  }

 private:
  std::ostream *os_{nullptr};
};

class ExpressionHumanFriendlyPrinter : public ExpressionPrinter {
 public:
  explicit ExpressionHumanFriendlyPrinter(std::ostream *os = nullptr)
      : ExpressionPrinter(os) {
  }

  void visit(ExprGroup &expr_group) override {
    emit_vector(expr_group.exprs);
  }

  void visit(ArgLoadExpression *expr) override {
    emit(
        fmt::format("arg[{}] (dt={})", expr->arg_id, data_type_name(expr->dt)));
  }

  void visit(TexturePtrExpression *expr) override {
    emit(fmt::format("(Texture *)(arg[{}])", expr->arg_id));
  }

  void visit(TextureOpExpression *expr) override {
    emit(fmt::format("texture_{}(", texture_op_type_name(expr->op)));
    visit(expr->args);
    emit(")");
  }

  void visit(RandExpression *expr) override {
    emit(fmt::format("rand<{}>()", data_type_name(expr->dt)));
  }

  void visit(UnaryOpExpression *expr) override {
    emit('(');
    if (expr->is_cast()) {
      emit(expr->type == UnaryOpType::cast_value ? "" : "reinterpret_");
      emit(unary_op_type_name(expr->type));
      emit('<', data_type_name(expr->cast_type), "> ");
    } else {
      emit(unary_op_type_name(expr->type), ' ');
    }
    expr->operand->accept(this);
    emit(')');
  }

  void visit(BinaryOpExpression *expr) override {
    emit('(');
    expr->lhs->accept(this);
    emit(' ', binary_op_type_symbol(expr->type), ' ');
    expr->rhs->accept(this);
    emit(')');
  }

  void visit(TernaryOpExpression *expr) override {
    emit(ternary_type_name(expr->type), '(');
    expr->op1->accept(this);
    emit(' ');
    expr->op2->accept(this);
    emit(' ');
    expr->op3->accept(this);
    emit(')');
  }

  void visit(InternalFuncCallExpression *expr) override {
    emit("internal call ", expr->func_name, '(');
    if (expr->with_runtime_context) {
      emit("runtime, ");
    }
    emit_vector(expr->args);
    emit(')');
  }

  void visit(ExternalTensorExpression *expr) override {
    emit(fmt::format("{}d_ext_arr (element_dim={}, dt={})", expr->dim,
                     expr->element_dim, expr->dt->to_string()));
  }

  void visit(GlobalVariableExpression *expr) override {
    emit("#", expr->ident.name());
    if (expr->snode) {
      emit(
          fmt::format(" (snode={})", expr->snode->get_node_type_name_hinted()));
    } else {
      emit(fmt::format(" (dt={})", expr->dt->to_string()));
    }
  }

  void visit(MatrixExpression *expr) override {
    emit('[');
    emit_vector(expr->elements);
    emit(']');
    emit(fmt::format(" (dt={})", expr->dt->to_string()));
  }

  void visit(IndexExpression *expr) override {
    expr->var->accept(this);
    emit('[');
    emit_vector(expr->indices.exprs);
    emit(']');
  }

  void visit(StrideExpression *expr) override {
    expr->var->accept(this);
    emit('[');
    emit_vector(expr->indices.exprs);
    emit("] (");
    emit_vector(expr->shape);
    emit(", stride = ", expr->stride);
    emit(')');
  }

  void visit(RangeAssumptionExpression *expr) override {
    emit("assume_in_range({");
    expr->base->accept(this);
    emit(fmt::format("{:+d}", expr->low), " <= (");
    expr->input->accept(this);
    emit(")  < ");
    expr->base->accept(this);
    emit(fmt::format("{:+d})", expr->high));
  }

  void visit(LoopUniqueExpression *expr) override {
    emit("loop_unique(");
    expr->input->accept(this);
    if (!expr->covers.empty()) {
      emit(", covers=[");
      emit_vector(expr->covers);
      emit(']');
    }
    emit(')');
  }

  void visit(IdExpression *expr) override {
    emit(expr->id.name());
  }

  void visit(AtomicOpExpression *expr) override {
    const auto op_type = (std::size_t)expr->op_type;
    constexpr const char *names_table[] = {
        "atomic_add",     "atomic_sub",    "atomic_min",     "atomic_max",
        "atomic_bit_and", "atomic_bit_or", "atomic_bit_xor",
    };
    if (op_type > std::size(names_table)) {
      // min/max not supported in the LLVM backend yet.
      TI_NOT_IMPLEMENTED;
    }
    emit(names_table[op_type], '(');
    expr->dest->accept(this);
    emit(", ");
    expr->val->accept(this);
    emit(")");
  }

  void visit(SNodeOpExpression *expr) override {
    emit(snode_op_type_name(expr->op_type));
    emit('(', expr->snode->get_node_type_name_hinted(), ", [");
    emit_vector(expr->indices.exprs);
    emit("]");
    if (expr->value.expr) {
      emit(' ');
      expr->value->accept(this);
    }
    emit(')');
  }

  void visit(ConstExpression *expr) override {
    emit(expr->val.stringify());
  }

  void visit(ExternalTensorShapeAlongAxisExpression *expr) override {
    emit("external_tensor_shape_along_axis(");
    expr->ptr->accept(this);
    emit(", ", expr->axis, ')');
  }

  void visit(FuncCallExpression *expr) override {
    emit("func_call(\"", expr->func->func_key.get_full_name(), "\", ");
    emit_vector(expr->args.exprs);
    emit(')');
  }

  void visit(MeshPatchIndexExpression *expr) override {
    emit("mesh_patch_idx()");
  }

  void visit(MeshRelationAccessExpression *expr) override {
    if (expr->neighbor_idx) {
      emit("mesh_relation_access(");
      expr->mesh_idx->accept(this);
      emit(", ", mesh::element_type_name(expr->to_type), '[');
      expr->neighbor_idx->accept(this);
      emit("])");
    } else {
      emit("mesh_relation_size(");
      expr->mesh_idx->accept(this);
      emit(", ", mesh::element_type_name(expr->to_type), ')');
    }
  }

  void visit(MeshIndexConversionExpression *expr) override {
    emit("mesh_index_conversion(", mesh::conv_type_name(expr->conv_type), ", ",
         mesh::element_type_name(expr->idx_type), ", ");
    expr->idx->accept(this);
    emit(")");
  }

  void visit(ReferenceExpression *expr) override {
    emit("ref(");
    expr->var->accept(this);
    emit(")");
  }

  static std::string expr_to_string(Expr &expr) {
    std::ostringstream oss;
    ExpressionHumanFriendlyPrinter printer(&oss);
    expr->accept(&printer);
    return oss.str();
  }

 protected:
  template <typename... Args>
  void emit(Args &&...args) {
    TI_ASSERT(this->get_ostream());
    (*this->get_ostream() << ... << std::forward<Args>(args));
  }

  template <typename T>
  void emit_vector(std::vector<T> &v) {
    if (!v.empty()) {
      emit_element(v[0]);
      const auto size = v.size();
      for (std::size_t i = 1; i < size; ++i) {
        emit(", ");
        emit_element(v[i]);
      }
    }
  }

  template <typename D>
  void emit_element(D &&e) {
    using T =
        typename std::remove_cv<typename std::remove_reference<D>::type>::type;
    if constexpr (std::is_same_v<T, Expr>) {
      e->accept(this);
    } else if constexpr (std::is_same_v<T, SNode *>) {
      emit(e->get_node_type_name_hinted());
    } else {
      emit(std::forward<D>(e));
    }
  }
};

}  // namespace lang
}  // namespace taichi
