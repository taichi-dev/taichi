#include "taichi/ir/expr.h"

#include "taichi/ir/frontend_ir.h"
#include "taichi/ir/ir.h"
#include "taichi/program/program.h"

namespace taichi::lang {

void Expr::set_tb(const std::string &tb) {
  expr->tb = tb;
}

DataType Expr::get_ret_type() const {
  return expr->ret_type;
}

void Expr::type_check(CompileConfig *config) {
  expr->type_check(config);
}

Expr cast(const Expr &input, DataType dt) {
  return Expr::make<UnaryOpExpression>(UnaryOpType::cast_value, input, dt);
}

Expr bit_cast(const Expr &input, DataType dt) {
  return Expr::make<UnaryOpExpression>(UnaryOpType::cast_bits, input, dt);
}

Expr Expr::operator[](const ExprGroup &indices) const {
  TI_ASSERT(is<FieldExpression>() || is<MatrixFieldExpression>() ||
            is<ExternalTensorExpression>() || is_tensor(expr->ret_type));
  return Expr::make<IndexExpression>(*this, indices);
}

Expr &Expr::operator=(const Expr &o) {
  set(o);
  return *this;
}

SNode *Expr::snode() const {
  TI_ASSERT_INFO(is<FieldExpression>(),
                 "Cannot get snode of non-field expressions.");
  return cast<FieldExpression>()->snode;
}

void Expr::set_adjoint(const Expr &o) {
  this->cast<FieldExpression>()->adjoint.set(o);
}

void Expr::set_dual(const Expr &o) {
  this->cast<FieldExpression>()->dual.set(o);
}

void Expr::set_adjoint_checkbit(const Expr &o) {
  this->cast<FieldExpression>()->adjoint_checkbit.set(o);
}

Expr::Expr(int16 x) : Expr() {
  expr = std::make_shared<ConstExpression>(PrimitiveType::i16, x);
}

Expr::Expr(int32 x) : Expr() {
  expr = std::make_shared<ConstExpression>(PrimitiveType::i32, x);
}

Expr::Expr(int64 x) : Expr() {
  expr = std::make_shared<ConstExpression>(PrimitiveType::i64, x);
}

Expr::Expr(float32 x) : Expr() {
  expr = std::make_shared<ConstExpression>(PrimitiveType::f32, x);
}

Expr::Expr(float64 x) : Expr() {
  expr = std::make_shared<ConstExpression>(PrimitiveType::f64, x);
}

Expr::Expr(const Identifier &id) : Expr() {
  expr = std::make_shared<IdExpression>(id);
}

Expr expr_rand(DataType dt) {
  return Expr::make<RandExpression>(dt);
}

Expr snode_append(SNode *snode, const ExprGroup &indices, const Expr &val) {
  return Expr::make<SNodeOpExpression>(snode, SNodeOpType::append, indices,
                                       val);
}

Expr snode_is_active(SNode *snode, const ExprGroup &indices) {
  return Expr::make<SNodeOpExpression>(snode, SNodeOpType::is_active, indices);
}

Expr snode_length(SNode *snode, const ExprGroup &indices) {
  return Expr::make<SNodeOpExpression>(snode, SNodeOpType::length, indices);
}

Expr snode_get_addr(SNode *snode, const ExprGroup &indices) {
  return Expr::make<SNodeOpExpression>(snode, SNodeOpType::get_addr, indices);
}

Expr assume_range(const Expr &expr, const Expr &base, int low, int high) {
  return Expr::make<RangeAssumptionExpression>(expr, base, low, high);
}

Expr loop_unique(const Expr &input, const std::vector<SNode *> &covers) {
  return Expr::make<LoopUniqueExpression>(input, covers);
}

Expr expr_field(Expr id_expr, DataType dt) {
  TI_ASSERT(id_expr.is<IdExpression>());
  auto ret = Expr(
      std::make_shared<FieldExpression>(dt, id_expr.cast<IdExpression>()->id));
  return ret;
}

Expr expr_matrix_field(const std::vector<Expr> &fields,
                       const std::vector<int> &element_shape) {
  return Expr::make<MatrixFieldExpression>(fields, element_shape);
}

}  // namespace taichi::lang
