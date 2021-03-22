#include "taichi/ir/ir_builder.h"
#include "taichi/ir/statements.h"
#include "taichi/common/logging.h"

TLANG_NAMESPACE_BEGIN

IRBuilder::IRBuilder() {
  reset();
}

void IRBuilder::reset() {
  root_ = std::make_unique<Block>();
  insert_point_.block = root_->as<Block>();
  insert_point_.position = 0;
}

std::unique_ptr<IRNode> IRBuilder::extract_ir() {
  auto &&result = std::move(root_);
  reset();
  return std::move(result);
}

Stmt *IRBuilder::insert(std::unique_ptr<Stmt> &&stmt) {
  return insert(std::move(stmt), &insert_point_);
}

Stmt *IRBuilder::insert(std::unique_ptr<Stmt> &&stmt,
                        InsertPoint *insert_point) {
  return insert_point->block->insert(std::move(stmt), insert_point->position++);
}

void IRBuilder::set_insertion_point(InsertPoint new_insert_point) {
  insert_point_ = new_insert_point;
}

void IRBuilder::set_insertion_point_to_after(Stmt *stmt) {
  set_insertion_point({stmt->parent, stmt->parent->locate(stmt) + 1});
}

void IRBuilder::set_insertion_point_to_before(Stmt *stmt) {
  set_insertion_point({stmt->parent, stmt->parent->locate(stmt)});
}

void IRBuilder::set_insertion_point_to_loop_begin(Stmt *loop) {
  if (auto range_for = loop->cast<RangeForStmt>()) {
    set_insertion_point({range_for->body.get(), 0});
  } else if (auto struct_for = loop->cast<StructForStmt>()) {
    set_insertion_point({struct_for->body.get(), 0});
  } else if (auto while_stmt = loop->cast<WhileStmt>()) {
    set_insertion_point({while_stmt->body.get(), 0});
  } else {
    TI_ERROR("Statement {} is not a loop.", loop->name());
  }
}

void IRBuilder::set_insertion_point_to_true_branch(Stmt *if_stmt) {
  TI_ASSERT(if_stmt->is<IfStmt>());
  set_insertion_point({if_stmt->as<IfStmt>()->true_statements.get(), 0});
}

void IRBuilder::set_insertion_point_to_false_branch(Stmt *if_stmt) {
  TI_ASSERT(if_stmt->is<IfStmt>());
  set_insertion_point({if_stmt->as<IfStmt>()->false_statements.get(), 0});
}

Stmt *IRBuilder::create_range_for(Stmt *begin,
                                  Stmt *end,
                                  int vectorize,
                                  int bit_vectorize,
                                  int parallelize,
                                  int block_dim,
                                  bool strictly_serialized) {
  return insert(Stmt::make<RangeForStmt>(begin, end, std::make_unique<Block>(),
                                         vectorize, bit_vectorize, parallelize,
                                         block_dim, strictly_serialized));
}

Stmt *IRBuilder::create_struct_for(SNode *snode,
                                   int vectorize,
                                   int bit_vectorize,
                                   int parallelize,
                                   int block_dim) {
  return insert(Stmt::make<StructForStmt>(snode, std::make_unique<Block>(),
                                          vectorize, bit_vectorize, parallelize,
                                          block_dim));
}

Stmt *IRBuilder::create_while_true() {
  return insert(Stmt::make<WhileStmt>(std::make_unique<Block>()));
}

Stmt *IRBuilder::create_if(Stmt *cond) {
  auto &&result = Stmt::make_typed<IfStmt>(cond);
  result->set_true_statements(std::make_unique<Block>());
  result->set_false_statements(std::make_unique<Block>());
  return insert(std::move(result));
}

Stmt *IRBuilder::create_break() {
  return insert(Stmt::make<WhileControlStmt>(nullptr, get_int32(0)));
}

Stmt *IRBuilder::create_continue() {
  return insert(Stmt::make<ContinueStmt>());
}

Stmt *IRBuilder::get_loop_index(Stmt *loop, int index) {
  return insert(Stmt::make<LoopIndexStmt>(loop, index));
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

Stmt *IRBuilder::create_local_var(DataType dt) {
  return insert(Stmt::make<AllocaStmt>(dt));
}

TLANG_NAMESPACE_END
