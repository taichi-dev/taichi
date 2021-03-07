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

Stmt *IRBuilder::get_argument(int arg_id, DataType dt, bool is_ptr) {
  return insert(Stmt::make<ArgLoadStmt>(arg_id, dt, is_ptr));
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

template <typename... Args>
Stmt *IRBuilder::create_print(Args &&... args) {
  return insert(Stmt::make<PrintStmt>(std::forward(args)));
}

TLANG_NAMESPACE_END
