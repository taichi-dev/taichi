#include "taichi/ir/ir_builder.h"
#include "taichi/ir/statements.h"

TLANG_NAMESPACE_BEGIN

IRBuilder::IRBuilder() {
  root_ = std::make_unique<Block>();
  insert_point_.block = root_->as<Block>();
  insert_point_.position = 0;
}

Stmt *IRBuilder::insert(std::unique_ptr<Stmt> &&stmt) {
  return insert_point_.block->insert(std::move(stmt), insert_point_.position++);
}

Stmt *IRBuilder::get_int32(int32 value) {
  return insert(
      Stmt::make<ConstStmt>(LaneAttribute<TypedConstant>(TypedConstant(
          TypeFactory::get_instance().get_primitive_type(PrimitiveTypeID::i32),
          value))));
}

Stmt *IRBuilder::get_int64(int64 value) {
  return insert(
      Stmt::make<ConstStmt>(LaneAttribute<TypedConstant>(TypedConstant(
          TypeFactory::get_instance().get_primitive_type(PrimitiveTypeID::i64),
          value))));
}

Stmt *IRBuilder::get_float32(float32 value) {
  return insert(
      Stmt::make<ConstStmt>(LaneAttribute<TypedConstant>(TypedConstant(
          TypeFactory::get_instance().get_primitive_type(PrimitiveTypeID::f32),
          value))));
}

Stmt *IRBuilder::get_float64(float64 value) {
  return insert(
      Stmt::make<ConstStmt>(LaneAttribute<TypedConstant>(TypedConstant(
          TypeFactory::get_instance().get_primitive_type(PrimitiveTypeID::f64),
          value))));
}

Stmt *IRBuilder::create_arg_load(int arg_id, DataType dt, bool is_ptr) {
  return insert(Stmt::make<ArgLoadStmt>(arg_id, dt, is_ptr));
}

Stmt *IRBuilder::create_return(Stmt *value) {
  return insert(Stmt::make<KernelReturnStmt>(value));
}

Stmt *IRBuilder::create_cast(Stmt *value, DataType output_type) {
  auto &&result = Stmt::make_typed<UnaryOpStmt>(UnaryOpType::cast_value, value);
  result->cast_type = output_type;
  return insert(std::move(result));
}

Stmt *IRBuilder::create_bit_cast(Stmt *value, DataType output_type) {
  auto &&result = Stmt::make_typed<UnaryOpStmt>(UnaryOpType::cast_bits, value);
  result->cast_type = output_type;
  return insert(std::move(result));
}

Stmt *IRBuilder::create_neg(Stmt *value) {
  return insert(Stmt::make<UnaryOpStmt>(UnaryOpType::neg, value));
}

Stmt *IRBuilder::create_not(Stmt *value) {
  return insert(Stmt::make<UnaryOpStmt>(UnaryOpType::bit_not, value));
}

Stmt *IRBuilder::create_logical_not(Stmt *value) {
  return insert(Stmt::make<UnaryOpStmt>(UnaryOpType::logic_not, value));
}

Stmt *IRBuilder::create_floor(Stmt *value) {
  return insert(Stmt::make<UnaryOpStmt>(UnaryOpType::floor, value));
}

Stmt *IRBuilder::create_ceil(Stmt *value) {
  return insert(Stmt::make<UnaryOpStmt>(UnaryOpType::ceil, value));
}

Stmt *IRBuilder::create_abs(Stmt *value) {
  return insert(Stmt::make<UnaryOpStmt>(UnaryOpType::abs, value));
}

Stmt *IRBuilder::create_sgn(Stmt *value) {
  return insert(Stmt::make<UnaryOpStmt>(UnaryOpType::sgn, value));
}

Stmt *IRBuilder::create_sqrt(Stmt *value) {
  return insert(Stmt::make<UnaryOpStmt>(UnaryOpType::sqrt, value));
}

Stmt *IRBuilder::create_rsqrt(Stmt *value) {
  return insert(Stmt::make<UnaryOpStmt>(UnaryOpType::rsqrt, value));
}

Stmt *IRBuilder::create_sin(Stmt *value) {
  return insert(Stmt::make<UnaryOpStmt>(UnaryOpType::sin, value));
}

Stmt *IRBuilder::create_asin(Stmt *value) {
  return insert(Stmt::make<UnaryOpStmt>(UnaryOpType::asin, value));
}

Stmt *IRBuilder::create_cos(Stmt *value) {
  return insert(Stmt::make<UnaryOpStmt>(UnaryOpType::cos, value));
}

Stmt *IRBuilder::create_acos(Stmt *value) {
  return insert(Stmt::make<UnaryOpStmt>(UnaryOpType::acos, value));
}

Stmt *IRBuilder::create_tan(Stmt *value) {
  return insert(Stmt::make<UnaryOpStmt>(UnaryOpType::tan, value));
}

Stmt *IRBuilder::create_tanh(Stmt *value) {
  return insert(Stmt::make<UnaryOpStmt>(UnaryOpType::tanh, value));
}

Stmt *IRBuilder::create_exp(Stmt *value) {
  return insert(Stmt::make<UnaryOpStmt>(UnaryOpType::exp, value));
}

Stmt *IRBuilder::create_log(Stmt *value) {
  return insert(Stmt::make<UnaryOpStmt>(UnaryOpType::log, value));
}

Stmt *IRBuilder::create_add(Stmt *l, Stmt *r) {
  return insert(Stmt::make<BinaryOpStmt>(BinaryOpType::add, l, r));
}

Stmt *IRBuilder::create_sub(Stmt *l, Stmt *r) {
  return insert(Stmt::make<BinaryOpStmt>(BinaryOpType::sub, l, r));
}

Stmt *IRBuilder::create_mul(Stmt *l, Stmt *r) {
  return insert(Stmt::make<BinaryOpStmt>(BinaryOpType::mul, l, r));
}

Stmt *IRBuilder::create_div(Stmt *l, Stmt *r) {
  return insert(Stmt::make<BinaryOpStmt>(BinaryOpType::div, l, r));
}

Stmt *IRBuilder::create_floordiv(Stmt *l, Stmt *r) {
  return insert(Stmt::make<BinaryOpStmt>(BinaryOpType::floordiv, l, r));
}

Stmt *IRBuilder::create_truediv(Stmt *l, Stmt *r) {
  return insert(Stmt::make<BinaryOpStmt>(BinaryOpType::truediv, l, r));
}

Stmt *IRBuilder::create_mod(Stmt *l, Stmt *r) {
  return insert(Stmt::make<BinaryOpStmt>(BinaryOpType::mod, l, r));
}

Stmt *IRBuilder::create_max(Stmt *l, Stmt *r) {
  return insert(Stmt::make<BinaryOpStmt>(BinaryOpType::max, l, r));
}

Stmt *IRBuilder::create_min(Stmt *l, Stmt *r) {
  return insert(Stmt::make<BinaryOpStmt>(BinaryOpType::min, l, r));
}

Stmt *IRBuilder::create_atan2(Stmt *l, Stmt *r) {
  return insert(Stmt::make<BinaryOpStmt>(BinaryOpType::atan2, l, r));
}

Stmt *IRBuilder::create_pow(Stmt *l, Stmt *r) {
  return insert(Stmt::make<BinaryOpStmt>(BinaryOpType::pow, l, r));
}

Stmt *IRBuilder::create_and(Stmt *l, Stmt *r) {
  return insert(Stmt::make<BinaryOpStmt>(BinaryOpType::bit_and, l, r));
}

Stmt *IRBuilder::create_or(Stmt *l, Stmt *r) {
  return insert(Stmt::make<BinaryOpStmt>(BinaryOpType::bit_or, l, r));
}

Stmt *IRBuilder::create_xor(Stmt *l, Stmt *r) {
  return insert(Stmt::make<BinaryOpStmt>(BinaryOpType::bit_xor, l, r));
}

Stmt *IRBuilder::create_shl(Stmt *l, Stmt *r) {
  return insert(Stmt::make<BinaryOpStmt>(BinaryOpType::bit_shl, l, r));
}

Stmt *IRBuilder::create_shr(Stmt *l, Stmt *r) {
  return insert(Stmt::make<BinaryOpStmt>(BinaryOpType::bit_shr, l, r));
}

Stmt *IRBuilder::create_sar(Stmt *l, Stmt *r) {
  return insert(Stmt::make<BinaryOpStmt>(BinaryOpType::bit_sar, l, r));
}

Stmt *IRBuilder::create_cmp_lt(Stmt *l, Stmt *r) {
  return insert(Stmt::make<BinaryOpStmt>(BinaryOpType::cmp_lt, l, r));
}

Stmt *IRBuilder::create_cmp_le(Stmt *l, Stmt *r) {
  return insert(Stmt::make<BinaryOpStmt>(BinaryOpType::cmp_le, l, r));
}

Stmt *IRBuilder::create_cmp_gt(Stmt *l, Stmt *r) {
  return insert(Stmt::make<BinaryOpStmt>(BinaryOpType::cmp_gt, l, r));
}

Stmt *IRBuilder::create_cmp_ge(Stmt *l, Stmt *r) {
  return insert(Stmt::make<BinaryOpStmt>(BinaryOpType::cmp_ge, l, r));
}

Stmt *IRBuilder::create_cmp_eq(Stmt *l, Stmt *r) {
  return insert(Stmt::make<BinaryOpStmt>(BinaryOpType::cmp_eq, l, r));
}

Stmt *IRBuilder::create_cmp_ne(Stmt *l, Stmt *r) {
  return insert(Stmt::make<BinaryOpStmt>(BinaryOpType::cmp_ne, l, r));
}

Stmt *IRBuilder::create_select(Stmt *cond,
                               Stmt *true_result,
                               Stmt *false_result) {
  return insert(Stmt::make<TernaryOpStmt>(TernaryOpType::select, cond,
                                          true_result, false_result));
}

TLANG_NAMESPACE_END
