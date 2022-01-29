#include "taichi/ir/expr.h"

#include "taichi/ir/frontend_ir.h"
#include "taichi/ir/ir.h"
#include "taichi/ir/operation_impl.h"
#include "taichi/program/program.h"

TLANG_NAMESPACE_BEGIN

void Expr::serialize(std::ostream &ss) const {
  TI_ASSERT(expr);
  expr->serialize(ss);
}

std::string Expr::serialize() const {
  std::stringstream ss;
  serialize(ss);
  return ss.str();
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

DataType Expr::get_ret_type() const {
  return expr->ret_type;
}

void Expr::type_check() {
  expr->type_check();
}

Expr select(const Expr &cond, const Expr &true_val, const Expr &false_val) {
  return InternalOps::get().select->call(cond, true_val, false_val);
}

Expr operator-(const Expr &expr) {
  return InternalOps::get().neg->call(expr);
}

Expr operator~(const Expr &expr) {
  return InternalOps::get().bit_not->call(expr);
}

Expr cast(const Expr &input, DataType dt) {
  return Expr::make<CastExpression>(UnaryOpType::cast_value, input, dt);
}

Expr bit_cast(const Expr &input, DataType dt) {
  return Expr::make<CastExpression>(UnaryOpType::cast_bits, input, dt);
}

Expr Expr::operator[](const ExprGroup &indices) const {
  TI_ASSERT(is<GlobalVariableExpression>() || is<ExternalTensorExpression>());
  return Expr::make<GlobalPtrExpression>(*this, indices);
}

Expr &Expr::operator=(const Expr &o) {
  set(o);
  return *this;
}

Expr Expr::parent() const {
  TI_ASSERT_INFO(is<GlobalVariableExpression>(),
                 "Cannot get snode parent of non-global variables.");
  return Expr::make<GlobalVariableExpression>(
      cast<GlobalVariableExpression>()->snode->parent);
}

SNode *Expr::snode() const {
  TI_ASSERT_INFO(is<GlobalVariableExpression>(),
                 "Cannot get snode of non-global variables.");
  return cast<GlobalVariableExpression>()->snode;
}

Expr Expr::operator!() {
  return InternalOps::get().logic_not->call(expr);
}

void Expr::declare(DataType dt) {
  set(Expr::make<GlobalVariableExpression>(dt, Identifier()));
}

void Expr::set_grad(const Expr &o) {
  this->cast<GlobalVariableExpression>()->adjoint.set(o);
}

Expr::Expr(int32 x) : Expr() {
  expr = std::make_shared<ConstExpression>(x);
}

Expr::Expr(int64 x) : Expr() {
  expr = std::make_shared<ConstExpression>(x);
}

Expr::Expr(float32 x) : Expr() {
  expr = std::make_shared<ConstExpression>(x);
}

Expr::Expr(float64 x) : Expr() {
  expr = std::make_shared<ConstExpression>(x);
}

Expr::Expr(const Identifier &id) : Expr() {
  expr = std::make_shared<IdExpression>(id);
}

TLANG_NAMESPACE_END
