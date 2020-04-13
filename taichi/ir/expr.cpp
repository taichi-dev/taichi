#include "expr.h"
#include "ir.h"
#include "taichi/program/program.h"

TLANG_NAMESPACE_BEGIN

std::string Expr::serialize() const {
  TI_ASSERT(expr);
  return expr->serialize();
}

void Expr::set_tb(const std::string &tb) {
  expr->tb = tb;
}

void Expr::set_attribute(const std::string &key, const std::string &value) {
  expr->set_attribute(key, value);
}

std::string Expr::get_attribute(const std::string &key) const {
  return expr->get_attribute(key);
}

Expr select(const Expr &cond, const Expr &true_val, const Expr &false_val) {
  return Expr::make<TernaryOpExpression>(TernaryOpType::select, cond, true_val,
                                         false_val);
}

Expr operator-(const Expr &expr) {
  return Expr::make<UnaryOpExpression>(UnaryOpType::neg, expr);
}

Expr operator~(const Expr &expr) {
  return Expr::make<UnaryOpExpression>(UnaryOpType::bit_not, expr);
}

Expr cast(const Expr &input, DataType dt) {
  auto ret = std::make_shared<UnaryOpExpression>(UnaryOpType::cast, input);
  ret->cast_type = dt;
  ret->cast_by_value = true;
  return Expr(ret);
}

Expr bit_cast(const Expr &input, DataType dt) {
  auto ret = std::make_shared<UnaryOpExpression>(UnaryOpType::cast, input);
  ret->cast_type = dt;
  ret->cast_by_value = false;
  return Expr(ret);
}

Expr Expr::operator[](const ExprGroup &indices) const {
  TI_ASSERT(is<GlobalVariableExpression>() || is<ExternalTensorExpression>());
  return Expr::make<GlobalPtrExpression>(*this, indices.loaded());
}

Expr &Expr::operator=(const Expr &o) {
  if (get_current_program().current_kernel) {
    if (expr == nullptr) {
      set(o.eval());
    } else if (expr->is_lvalue()) {
      current_ast_builder().insert(std::make_unique<FrontendAssignStmt>(
          ptr_if_global(*this), load_if_ptr(o)));
    } else {
      // set(o.eval());
      TI_ERROR("Cannot assign to non-lvalue: {}", serialize());
    }
  } else {
    set(o);
  }
  return *this;
}

Expr Expr::parent() const {
  TI_ASSERT(is<GlobalVariableExpression>());
  return Expr::make<GlobalVariableExpression>(
      cast<GlobalVariableExpression>()->snode->parent);
}

SNode *Expr::snode() const {
  TI_ASSERT(is<GlobalVariableExpression>());
  return cast<GlobalVariableExpression>()->snode;
}

Expr Expr::operator!() {
  return Expr::make<UnaryOpExpression>(UnaryOpType::logic_not, expr);
}

void Expr::declare(DataType dt) {
  set(Expr::make<GlobalVariableExpression>(dt, Identifier()));
}

void Expr::set_grad(const Expr &o) {
  this->cast<GlobalVariableExpression>()->adjoint.set(o);
}

TLANG_NAMESPACE_END
