#include "taichi/ir/expr.h"

#include "taichi/ir/frontend_ir.h"
#include "taichi/ir/ir.h"
#include "taichi/program/program.h"

namespace taichi::lang {

void Expr::set_dbg_info(const DebugInfo &dbg_info) {
  expr->dbg_info = dbg_info;
}

const std::string &Expr::get_tb() const {
  return expr->get_tb();
}

DataType Expr::get_ret_type() const {
  return expr->ret_type;
}

DataType Expr::get_rvalue_type() const {
  if (auto argload = cast<ArgLoadExpression>()) {
    if (argload->is_ptr) {
      return argload->ret_type.ptr_removed();
    }
    return argload->ret_type;
  }
  if (auto id = cast<IdExpression>()) {
    return id->ret_type.ptr_removed();
  }
  if (auto index_expr = cast<IndexExpression>()) {
    return index_expr->ret_type.ptr_removed();
  }
  if (auto unary = cast<UnaryOpExpression>()) {
    if (unary->type == UnaryOpType::frexp) {
      return unary->ret_type.ptr_removed();
    }
    return unary->ret_type;
  }
  if (auto texture_op = cast<TextureOpExpression>()) {
    if (texture_op->op == TextureOpType::kStore) {
      return texture_op->ret_type.ptr_removed();
    }
    return texture_op->ret_type;
  }
  return expr->ret_type;
}

void Expr::type_check(const CompileConfig *config) {
  expr->type_check(config);
}

Expr cast(const Expr &input, DataType dt) {
  return Expr::make<UnaryOpExpression>(UnaryOpType::cast_value, input, dt);
}

Expr bit_cast(const Expr &input, DataType dt) {
  return Expr::make<UnaryOpExpression>(UnaryOpType::cast_bits, input, dt);
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

Expr::Expr(uint1 x) : Expr() {
  expr = std::make_shared<ConstExpression>(PrimitiveType::u1, x);
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

Expr assume_range(const Expr &expr,
                  const Expr &base,
                  int low,
                  int high,
                  const DebugInfo &dbg_info) {
  return Expr::make<RangeAssumptionExpression>(expr, base, low, high, dbg_info);
}

Expr loop_unique(const Expr &input,
                 const std::vector<SNode *> &covers,
                 const DebugInfo &dbg_info) {
  return Expr::make<LoopUniqueExpression>(input, covers, dbg_info);
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
