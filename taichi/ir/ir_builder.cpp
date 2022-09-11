#include "taichi/ir/ir_builder.h"
#include "taichi/ir/statements.h"
#include "taichi/common/logging.h"

TLANG_NAMESPACE_BEGIN

namespace {

inline bool stmt_location_did_not_change(Stmt *stmt, int location) {
  return location >= 0 && location < stmt->parent->size() &&
         stmt->parent->statements[location].get() == stmt;
}

}  // namespace

IRBuilder::IRBuilder() {
  reset();
}

void IRBuilder::reset() {
  root_ = std::make_unique<Block>();
  insert_point_.block = root_->as<Block>();
  insert_point_.position = 0;
}

std::unique_ptr<Block> IRBuilder::extract_ir() {
  auto result = std::move(root_);
  reset();
  return result;
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

void IRBuilder::set_insertion_point_to_true_branch(IfStmt *if_stmt) {
  if (!if_stmt->true_statements)
    if_stmt->set_true_statements(std::make_unique<Block>());
  set_insertion_point({if_stmt->true_statements.get(), 0});
}

void IRBuilder::set_insertion_point_to_false_branch(IfStmt *if_stmt) {
  if (!if_stmt->false_statements)
    if_stmt->set_false_statements(std::make_unique<Block>());
  set_insertion_point({if_stmt->false_statements.get(), 0});
}

IRBuilder::LoopGuard::~LoopGuard() {
  if (stmt_location_did_not_change(loop_, location_)) {
    // faster than set_insertion_point_to_after()
    builder_.set_insertion_point({loop_->parent, location_ + 1});
  } else {
    builder_.set_insertion_point_to_after(loop_);
  }
}

IRBuilder::IfGuard::IfGuard(IRBuilder &builder,
                            IfStmt *if_stmt,
                            bool true_branch)
    : builder_(builder), if_stmt_(if_stmt) {
  location_ = (int)if_stmt_->parent->size() - 1;
  if (true_branch) {
    builder_.set_insertion_point_to_true_branch(if_stmt_);
  } else {
    builder_.set_insertion_point_to_false_branch(if_stmt_);
  }
}

IRBuilder::IfGuard::~IfGuard() {
  if (stmt_location_did_not_change(if_stmt_, location_)) {
    // faster than set_insertion_point_to_after()
    builder_.set_insertion_point({if_stmt_->parent, location_ + 1});
  } else {
    builder_.set_insertion_point_to_after(if_stmt_);
  }
}

RangeForStmt *IRBuilder::create_range_for(Stmt *begin,
                                          Stmt *end,
                                          bool is_bit_vectorized,
                                          int num_cpu_threads,
                                          int block_dim,
                                          bool strictly_serialized) {
  return insert(Stmt::make_typed<RangeForStmt>(
      begin, end, std::make_unique<Block>(), is_bit_vectorized, num_cpu_threads,
      block_dim, strictly_serialized));
}

StructForStmt *IRBuilder::create_struct_for(SNode *snode,
                                            bool is_bit_vectorized,
                                            int num_cpu_threads,
                                            int block_dim) {
  return insert(Stmt::make_typed<StructForStmt>(
      snode, std::make_unique<Block>(), is_bit_vectorized, num_cpu_threads,
      block_dim));
}

MeshForStmt *IRBuilder::create_mesh_for(mesh::Mesh *mesh,
                                        mesh::MeshElementType element_type,
                                        bool is_bit_vectorized,
                                        int num_cpu_threads,
                                        int block_dim) {
  return insert(Stmt::make_typed<MeshForStmt>(
      mesh, element_type, std::make_unique<Block>(), is_bit_vectorized,
      num_cpu_threads, block_dim));
}

WhileStmt *IRBuilder::create_while_true() {
  return insert(Stmt::make_typed<WhileStmt>(std::make_unique<Block>()));
}

IfStmt *IRBuilder::create_if(Stmt *cond) {
  return insert(Stmt::make_typed<IfStmt>(cond));
}

WhileControlStmt *IRBuilder::create_break() {
  return insert(Stmt::make_typed<WhileControlStmt>(nullptr, get_int32(0)));
}

ContinueStmt *IRBuilder::create_continue() {
  return insert(Stmt::make_typed<ContinueStmt>());
}

FuncCallStmt *IRBuilder::create_func_call(Function *func,
                                          const std::vector<Stmt *> &args) {
  return insert(Stmt::make_typed<FuncCallStmt>(func, args));
}

LoopIndexStmt *IRBuilder::get_loop_index(Stmt *loop, int index) {
  return insert(Stmt::make_typed<LoopIndexStmt>(loop, index));
}

ConstStmt *IRBuilder::get_int32(int32 value) {
  return insert(Stmt::make_typed<ConstStmt>(TypedConstant(
      TypeFactory::get_instance().get_primitive_type(PrimitiveTypeID::i32),
      value)));
}

ConstStmt *IRBuilder::get_int64(int64 value) {
  return insert(Stmt::make_typed<ConstStmt>(TypedConstant(
      TypeFactory::get_instance().get_primitive_type(PrimitiveTypeID::i64),
      value)));
}

ConstStmt *IRBuilder::get_uint32(uint32 value) {
  return insert(Stmt::make_typed<ConstStmt>(TypedConstant(
      TypeFactory::get_instance().get_primitive_type(PrimitiveTypeID::u32),
      value)));
}

ConstStmt *IRBuilder::get_uint64(uint64 value) {
  return insert(Stmt::make_typed<ConstStmt>(TypedConstant(
      TypeFactory::get_instance().get_primitive_type(PrimitiveTypeID::u64),
      value)));
}

ConstStmt *IRBuilder::get_float32(float32 value) {
  return insert(Stmt::make_typed<ConstStmt>(TypedConstant(
      TypeFactory::get_instance().get_primitive_type(PrimitiveTypeID::f32),
      value)));
}

ConstStmt *IRBuilder::get_float64(float64 value) {
  return insert(Stmt::make_typed<ConstStmt>(TypedConstant(
      TypeFactory::get_instance().get_primitive_type(PrimitiveTypeID::f64),
      value)));
}

RandStmt *IRBuilder::create_rand(DataType value_type) {
  return insert(Stmt::make_typed<RandStmt>(value_type));
}

ArgLoadStmt *IRBuilder::create_arg_load(int arg_id, DataType dt, bool is_ptr) {
  return insert(Stmt::make_typed<ArgLoadStmt>(arg_id, dt, is_ptr));
}

ReturnStmt *IRBuilder::create_return(Stmt *value) {
  return insert(Stmt::make_typed<ReturnStmt>(value));
}

UnaryOpStmt *IRBuilder::create_cast(Stmt *value, DataType output_type) {
  auto &&result = Stmt::make_typed<UnaryOpStmt>(UnaryOpType::cast_value, value);
  result->cast_type = output_type;
  return insert(std::move(result));
}

UnaryOpStmt *IRBuilder::create_bit_cast(Stmt *value, DataType output_type) {
  auto &&result = Stmt::make_typed<UnaryOpStmt>(UnaryOpType::cast_bits, value);
  result->cast_type = output_type;
  return insert(std::move(result));
}

UnaryOpStmt *IRBuilder::create_neg(Stmt *value) {
  return insert(Stmt::make_typed<UnaryOpStmt>(UnaryOpType::neg, value));
}

UnaryOpStmt *IRBuilder::create_not(Stmt *value) {
  return insert(Stmt::make_typed<UnaryOpStmt>(UnaryOpType::bit_not, value));
}

UnaryOpStmt *IRBuilder::create_logical_not(Stmt *value) {
  return insert(Stmt::make_typed<UnaryOpStmt>(UnaryOpType::logic_not, value));
}

UnaryOpStmt *IRBuilder::create_round(Stmt *value) {
  return insert(Stmt::make_typed<UnaryOpStmt>(UnaryOpType::round, value));
}

UnaryOpStmt *IRBuilder::create_floor(Stmt *value) {
  return insert(Stmt::make_typed<UnaryOpStmt>(UnaryOpType::floor, value));
}

UnaryOpStmt *IRBuilder::create_ceil(Stmt *value) {
  return insert(Stmt::make_typed<UnaryOpStmt>(UnaryOpType::ceil, value));
}

UnaryOpStmt *IRBuilder::create_abs(Stmt *value) {
  return insert(Stmt::make_typed<UnaryOpStmt>(UnaryOpType::abs, value));
}

UnaryOpStmt *IRBuilder::create_sgn(Stmt *value) {
  return insert(Stmt::make_typed<UnaryOpStmt>(UnaryOpType::sgn, value));
}

UnaryOpStmt *IRBuilder::create_sqrt(Stmt *value) {
  return insert(Stmt::make_typed<UnaryOpStmt>(UnaryOpType::sqrt, value));
}

UnaryOpStmt *IRBuilder::create_rsqrt(Stmt *value) {
  return insert(Stmt::make_typed<UnaryOpStmt>(UnaryOpType::rsqrt, value));
}

UnaryOpStmt *IRBuilder::create_sin(Stmt *value) {
  return insert(Stmt::make_typed<UnaryOpStmt>(UnaryOpType::sin, value));
}

UnaryOpStmt *IRBuilder::create_asin(Stmt *value) {
  return insert(Stmt::make_typed<UnaryOpStmt>(UnaryOpType::asin, value));
}

UnaryOpStmt *IRBuilder::create_cos(Stmt *value) {
  return insert(Stmt::make_typed<UnaryOpStmt>(UnaryOpType::cos, value));
}

UnaryOpStmt *IRBuilder::create_acos(Stmt *value) {
  return insert(Stmt::make_typed<UnaryOpStmt>(UnaryOpType::acos, value));
}

UnaryOpStmt *IRBuilder::create_tan(Stmt *value) {
  return insert(Stmt::make_typed<UnaryOpStmt>(UnaryOpType::tan, value));
}

UnaryOpStmt *IRBuilder::create_tanh(Stmt *value) {
  return insert(Stmt::make_typed<UnaryOpStmt>(UnaryOpType::tanh, value));
}

UnaryOpStmt *IRBuilder::create_exp(Stmt *value) {
  return insert(Stmt::make_typed<UnaryOpStmt>(UnaryOpType::exp, value));
}

UnaryOpStmt *IRBuilder::create_log(Stmt *value) {
  return insert(Stmt::make_typed<UnaryOpStmt>(UnaryOpType::log, value));
}

BinaryOpStmt *IRBuilder::create_add(Stmt *l, Stmt *r) {
  return insert(Stmt::make_typed<BinaryOpStmt>(BinaryOpType::add, l, r));
}

BinaryOpStmt *IRBuilder::create_sub(Stmt *l, Stmt *r) {
  return insert(Stmt::make_typed<BinaryOpStmt>(BinaryOpType::sub, l, r));
}

BinaryOpStmt *IRBuilder::create_mul(Stmt *l, Stmt *r) {
  return insert(Stmt::make_typed<BinaryOpStmt>(BinaryOpType::mul, l, r));
}

BinaryOpStmt *IRBuilder::create_div(Stmt *l, Stmt *r) {
  return insert(Stmt::make_typed<BinaryOpStmt>(BinaryOpType::div, l, r));
}

BinaryOpStmt *IRBuilder::create_floordiv(Stmt *l, Stmt *r) {
  return insert(Stmt::make_typed<BinaryOpStmt>(BinaryOpType::floordiv, l, r));
}

BinaryOpStmt *IRBuilder::create_truediv(Stmt *l, Stmt *r) {
  return insert(Stmt::make_typed<BinaryOpStmt>(BinaryOpType::truediv, l, r));
}

BinaryOpStmt *IRBuilder::create_mod(Stmt *l, Stmt *r) {
  return insert(Stmt::make_typed<BinaryOpStmt>(BinaryOpType::mod, l, r));
}

BinaryOpStmt *IRBuilder::create_max(Stmt *l, Stmt *r) {
  return insert(Stmt::make_typed<BinaryOpStmt>(BinaryOpType::max, l, r));
}

BinaryOpStmt *IRBuilder::create_min(Stmt *l, Stmt *r) {
  return insert(Stmt::make_typed<BinaryOpStmt>(BinaryOpType::min, l, r));
}

BinaryOpStmt *IRBuilder::create_atan2(Stmt *l, Stmt *r) {
  return insert(Stmt::make_typed<BinaryOpStmt>(BinaryOpType::atan2, l, r));
}

BinaryOpStmt *IRBuilder::create_pow(Stmt *l, Stmt *r) {
  return insert(Stmt::make_typed<BinaryOpStmt>(BinaryOpType::pow, l, r));
}

BinaryOpStmt *IRBuilder::create_and(Stmt *l, Stmt *r) {
  return insert(Stmt::make_typed<BinaryOpStmt>(BinaryOpType::bit_and, l, r));
}

BinaryOpStmt *IRBuilder::create_or(Stmt *l, Stmt *r) {
  return insert(Stmt::make_typed<BinaryOpStmt>(BinaryOpType::bit_or, l, r));
}

BinaryOpStmt *IRBuilder::create_xor(Stmt *l, Stmt *r) {
  return insert(Stmt::make_typed<BinaryOpStmt>(BinaryOpType::bit_xor, l, r));
}

BinaryOpStmt *IRBuilder::create_shl(Stmt *l, Stmt *r) {
  return insert(Stmt::make_typed<BinaryOpStmt>(BinaryOpType::bit_shl, l, r));
}

BinaryOpStmt *IRBuilder::create_shr(Stmt *l, Stmt *r) {
  return insert(Stmt::make_typed<BinaryOpStmt>(BinaryOpType::bit_shr, l, r));
}

BinaryOpStmt *IRBuilder::create_sar(Stmt *l, Stmt *r) {
  return insert(Stmt::make_typed<BinaryOpStmt>(BinaryOpType::bit_sar, l, r));
}

BinaryOpStmt *IRBuilder::create_cmp_lt(Stmt *l, Stmt *r) {
  return insert(Stmt::make_typed<BinaryOpStmt>(BinaryOpType::cmp_lt, l, r));
}

BinaryOpStmt *IRBuilder::create_cmp_le(Stmt *l, Stmt *r) {
  return insert(Stmt::make_typed<BinaryOpStmt>(BinaryOpType::cmp_le, l, r));
}

BinaryOpStmt *IRBuilder::create_cmp_gt(Stmt *l, Stmt *r) {
  return insert(Stmt::make_typed<BinaryOpStmt>(BinaryOpType::cmp_gt, l, r));
}

BinaryOpStmt *IRBuilder::create_cmp_ge(Stmt *l, Stmt *r) {
  return insert(Stmt::make_typed<BinaryOpStmt>(BinaryOpType::cmp_ge, l, r));
}

BinaryOpStmt *IRBuilder::create_cmp_eq(Stmt *l, Stmt *r) {
  return insert(Stmt::make_typed<BinaryOpStmt>(BinaryOpType::cmp_eq, l, r));
}

BinaryOpStmt *IRBuilder::create_cmp_ne(Stmt *l, Stmt *r) {
  return insert(Stmt::make_typed<BinaryOpStmt>(BinaryOpType::cmp_ne, l, r));
}

AtomicOpStmt *IRBuilder::create_atomic_add(Stmt *dest, Stmt *val) {
  return insert(Stmt::make_typed<AtomicOpStmt>(AtomicOpType::add, dest, val));
}

AtomicOpStmt *IRBuilder::create_atomic_sub(Stmt *dest, Stmt *val) {
  return insert(Stmt::make_typed<AtomicOpStmt>(AtomicOpType::sub, dest, val));
}

AtomicOpStmt *IRBuilder::create_atomic_max(Stmt *dest, Stmt *val) {
  return insert(Stmt::make_typed<AtomicOpStmt>(AtomicOpType::max, dest, val));
}

AtomicOpStmt *IRBuilder::create_atomic_min(Stmt *dest, Stmt *val) {
  return insert(Stmt::make_typed<AtomicOpStmt>(AtomicOpType::min, dest, val));
}

AtomicOpStmt *IRBuilder::create_atomic_and(Stmt *dest, Stmt *val) {
  return insert(
      Stmt::make_typed<AtomicOpStmt>(AtomicOpType::bit_and, dest, val));
}

AtomicOpStmt *IRBuilder::create_atomic_or(Stmt *dest, Stmt *val) {
  return insert(
      Stmt::make_typed<AtomicOpStmt>(AtomicOpType::bit_or, dest, val));
}

AtomicOpStmt *IRBuilder::create_atomic_xor(Stmt *dest, Stmt *val) {
  return insert(
      Stmt::make_typed<AtomicOpStmt>(AtomicOpType::bit_xor, dest, val));
}

TernaryOpStmt *IRBuilder::create_select(Stmt *cond,
                                        Stmt *true_result,
                                        Stmt *false_result) {
  return insert(Stmt::make_typed<TernaryOpStmt>(TernaryOpType::select, cond,
                                                true_result, false_result));
}

AllocaStmt *IRBuilder::create_local_var(DataType dt) {
  return insert(Stmt::make_typed<AllocaStmt>(dt));
}

LocalLoadStmt *IRBuilder::create_local_load(AllocaStmt *ptr) {
  return insert(Stmt::make_typed<LocalLoadStmt>(ptr));
}

void IRBuilder::create_local_store(AllocaStmt *ptr, Stmt *data) {
  insert(Stmt::make_typed<LocalStoreStmt>(ptr, data));
}

GlobalPtrStmt *IRBuilder::create_global_ptr(
    SNode *snode,
    const std::vector<Stmt *> &indices) {
  return insert(Stmt::make_typed<GlobalPtrStmt>(snode, indices));
}

ExternalPtrStmt *IRBuilder::create_external_ptr(
    ArgLoadStmt *ptr,
    const std::vector<Stmt *> &indices) {
  return insert(
      Stmt::make_typed<ExternalPtrStmt>(ptr, indices, std::vector<int>(), 0));
}

AdStackAllocaStmt *IRBuilder::create_ad_stack(const DataType &dt,
                                              std::size_t max_size) {
  return insert(Stmt::make_typed<AdStackAllocaStmt>(dt, max_size));
}

void IRBuilder::ad_stack_push(AdStackAllocaStmt *stack, Stmt *val) {
  insert(Stmt::make_typed<AdStackPushStmt>(stack, val));
}

void IRBuilder::ad_stack_pop(AdStackAllocaStmt *stack) {
  insert(Stmt::make_typed<AdStackPopStmt>(stack));
}

AdStackLoadTopStmt *IRBuilder::ad_stack_load_top(AdStackAllocaStmt *stack) {
  return insert(Stmt::make_typed<AdStackLoadTopStmt>(stack));
}

AdStackLoadTopAdjStmt *IRBuilder::ad_stack_load_top_adjoint(
    AdStackAllocaStmt *stack) {
  return insert(Stmt::make_typed<AdStackLoadTopAdjStmt>(stack));
}

void IRBuilder::ad_stack_accumulate_adjoint(AdStackAllocaStmt *stack,
                                            Stmt *val) {
  insert(Stmt::make_typed<AdStackAccAdjointStmt>(stack, val));
}

// Mesh related.

MeshRelationAccessStmt *IRBuilder::get_relation_size(
    mesh::Mesh *mesh,
    Stmt *mesh_idx,
    mesh::MeshElementType to_type) {
  return insert(
      Stmt::make_typed<MeshRelationAccessStmt>(mesh, mesh_idx, to_type));
}

MeshRelationAccessStmt *IRBuilder::get_relation_access(
    mesh::Mesh *mesh,
    Stmt *mesh_idx,
    mesh::MeshElementType to_type,
    Stmt *neighbor_idx) {
  return insert(Stmt::make_typed<MeshRelationAccessStmt>(
      mesh, mesh_idx, to_type, neighbor_idx));
}

MeshIndexConversionStmt *IRBuilder::get_index_conversion(
    mesh::Mesh *mesh,
    mesh::MeshElementType idx_type,
    Stmt *idx,
    mesh::ConvType conv_type) {
  return insert(Stmt::make_typed<MeshIndexConversionStmt>(mesh, idx_type, idx,
                                                          conv_type));
}

MeshPatchIndexStmt *IRBuilder::get_patch_index() {
  return insert(Stmt::make_typed<MeshPatchIndexStmt>());
}

TLANG_NAMESPACE_END
