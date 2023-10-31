#include "taichi/ir/type_system.h"
#include "taichi/ir/statements.h"
#include "taichi/ir/frontend_ir.h"

namespace taichi::lang {

void TyVar::unify(int pos, DataType dt, Solutions &solutions) const {
  if (solutions.find(name_) != solutions.end()) {
    if (solutions[name_].first != dt) {
      throw TyVarMismatch(solutions[name_].second, pos, solutions[name_].first,
                          dt);
    }
  } else {
    solutions[name_] = std::make_pair(dt, pos);
  }
}

DataType TyVar::resolve(const Solutions &solutions) const {
  if (solutions.find(name_) == solutions.end()) {
    throw TyVarUnsolved(name_);
  } else {
    return solutions.at(name_).first;
  }
}

std::string TyVar::to_string() const {
  return name_.name();
}

bool TyVar::contains_tyvar(const TyVar &tyvar) const {
  return name_ == tyvar.name_;
}

void TyLub::unify(int pos, DataType dt, Solutions &solutions) const {
  TyMono(resolve(solutions)).unify(pos, dt, solutions);
}

DataType TyLub::resolve(const Solutions &solutions) const {
  return promoted_type(lhs_->resolve(solutions)->get_compute_type(),
                       rhs_->resolve(solutions)->get_compute_type());
}

std::string TyLub::to_string() const {
  return lhs_->to_string() + " | " + rhs_->to_string();
}

bool TyLub::contains_tyvar(const TyVar &tyvar) const {
  return lhs_->contains_tyvar(tyvar) || rhs_->contains_tyvar(tyvar);
}

void TyCompute::unify(int pos, DataType dt, Solutions &solutions) const {
  TyMono(resolve(solutions)).unify(pos, dt, solutions);
}

DataType TyCompute::resolve(const Solutions &solutions) const {
  return exp_->resolve(solutions)->get_compute_type();
}

std::string TyCompute::to_string() const {
  return "comp(" + exp_->to_string() + ")";
}

bool TyCompute::contains_tyvar(const TyVar &tyvar) const {
  return exp_->contains_tyvar(tyvar);
}

void TyMono::unify(int pos, DataType dt, Solutions &solutions) const {
  if (monotype_ != dt) {
    throw TypeMismatch(pos, monotype_, dt);
  }
}

DataType TyMono::resolve(const Solutions &solutions) const {
  return monotype_;
}

std::string TyMono::to_string() const {
  return monotype_.to_string();
}

bool TyMono::contains_tyvar(const TyVar &tyvar) const {
  return false;
}

std::string TyVarMismatch::to_string() const {
  return "argument #" + std::to_string(solved_position_ + 1) + " and #" +
         std::to_string(current_position_ + 1) +
         " should be the same type, but they are different: " +
         original_.to_string() + " and " + conflicting_.to_string();
}

std::string TypeMismatch::to_string() const {
  return "expected " + param_.to_string() + " for argument #" +
         std::to_string(position_ + 1) + ", but got " + arg_.to_string();
}

std::string TyVarUnsolved::to_string() const {
  return "cannot infer the type variable " + var_.name() +
         ". this is not supposed to happen; please report this as a bug";
}

std::string TraitMismatch::to_string() const {
  return "the inferred type of argument #" + std::to_string(occurrence_ + 1) +
         " is " + dt_.to_string() + ", which is not a " +
         constraint_.trait->to_string();
}

std::string ArgLengthMismatch::to_string() const {
  return std::to_string(arg_) + " arguments were passed in but expected " +
         std::to_string(param_) +
         ". this is not supposed to happen; please report this as a bug";
}

DataType Signature::type_check(const std::vector<DataType> &arguments) const {
  if (parameters_.size() != arguments.size()) {
    throw ArgLengthMismatch(parameters_.size(), arguments.size());
  }
  TypeExpression::Solutions solutions;
  for (int i = 0; i < parameters_.size(); i++) {
    parameters_[i]->unify(i, arguments[i], solutions);
  }
  for (auto &c : constraints_) {
    auto dt = c.tyvar->resolve(solutions);
    if (!c.trait->validate(dt)) {
      int occurrence = -1;
      for (int i = 0; i < parameters_.size(); i++) {
        if (parameters_[i]->contains_tyvar(*c.tyvar)) {
          occurrence = i;
          break;
        }
      }
      throw TraitMismatch(occurrence, dt, c);
    }
  }
  return ret_type_->resolve(solutions);
}

DataType Operation::type_check(const std::vector<DataType> &arg_types) const {
  try {
    return sig.type_check(arg_types);
  } catch (TypeSystemError &err) {
    std::string msg;
    msg += "In a call to the operation `" + name + "`:\n";
    msg += "  " + err.to_string();
    msg += "  (called with argument types ";
    for (int i = 0; i < arg_types.size(); i++) {
      msg += arg_types[i].to_string();
      if (i != arg_types.size() - 1)
        msg += ", ";
      else
        msg += ")\n";
    }
    throw TaichiTypeError(msg);
  }
}

namespace {

int var_counter_ = 0;

#define PRIM(dt) \
  DataType dt =  \
      TypeFactory::get_instance().get_primitive_type(PrimitiveTypeID::dt);

PRIM(i32)
PRIM(i64)
PRIM(f32)
PRIM(f64)
PRIM(u32)
PRIM(u64)
DataType i32_void = i32;
DataType f32_ptr = TypeFactory::get_instance().get_pointer_type(f32, false);

#undef PRIM

Trait *Real = StaticTraits::get(StaticTraitID::real);
Trait *Integral = StaticTraits::get(StaticTraitID::integral);
Trait *Primitive = StaticTraits::get(StaticTraitID::primitive);
Trait *Scalar = StaticTraits::get(StaticTraitID::scalar);

[[maybe_unused]] Constraint operator<(std::shared_ptr<TyVar> tyvar,
                                      Trait *trait) {
  return Constraint(tyvar, trait);
}

std::shared_ptr<TyMono> operator!(DataType dt) {
  return std::make_shared<TyMono>(dt);
}

[[maybe_unused]] std::shared_ptr<TyLub> operator|(TypeExpr lhs, TypeExpr rhs) {
  return std::make_shared<TyLub>(lhs, rhs);
}

std::shared_ptr<TyVar> tyvar(std::string name) {
  return std::make_shared<TyVar>(Identifier(var_counter_++, name));
}

[[maybe_unused]] std::shared_ptr<TyCompute> comp(TypeExpr ty) {
  return std::make_shared<TyCompute>(ty);
}

std::vector<TypeExpr> type_exprs_from_dts(const std::vector<DataType> &params) {
  std::vector<TypeExpr> exprs;
  for (auto dt : params) {
    exprs.push_back(std::make_shared<TyMono>(dt));
  }
  return exprs;
}

std::vector<Stmt *> get_all_rvalues(const std::vector<Expr> &args,
                                    Expression::FlattenContext *ctx) {
  std::vector<Stmt *> stmts;
  for (auto arg : args) {
    stmts.push_back(flatten_rvalue(arg, ctx));
  }
  return stmts;
}

};  // namespace

bool DynamicTrait::validate(const DataType dt) const {
  return impl_(dt);
}

std::string DynamicTrait::to_string() const {
  return name_;
}

Trait *StaticTraits::get(StaticTraitID traitId) {
  if (traits_.empty()) {
    init_traits();
  }
  return traits_[traitId].get();
}

void StaticTraits::init_traits() {
  traits_[StaticTraitID::real] =
      std::make_unique<DynamicTrait>("Real", is_real);
  traits_[StaticTraitID::integral] =
      std::make_unique<DynamicTrait>("Integral", is_integral);
  traits_[StaticTraitID::primitive] = std::make_unique<DynamicTrait>(
      "Primitive", [](DataType dt) { return dt->is<PrimitiveType>(); });
  traits_[StaticTraitID::scalar] = std::make_unique<DynamicTrait>(
      "Scalar", [](DataType dt) { return is_real(dt) || is_integral(dt); });
}

class InternalCallOperation : public Operation {
  const std::string internal_call_name_;
  const bool with_runtime_context_;

 public:
  InternalCallOperation(const std::string &internal_name,
                        const std::vector<DataType> &params,
                        DataType result,
                        bool with_runtime_context)
      : Operation(internal_name,
                  Signature(type_exprs_from_dts(params), !result)),
        internal_call_name_(internal_name),
        with_runtime_context_(with_runtime_context) {
  }
  InternalCallOperation(const std::string &internal_name,
                        Signature sig,
                        bool with_runtime_context)
      : Operation(internal_name, sig),
        internal_call_name_(internal_name),
        with_runtime_context_(with_runtime_context) {
  }

  Stmt *flatten(Expression::FlattenContext *ctx,
                const std::vector<Expr> &args,
                DataType ret_type) const override {
    auto rargs = get_all_rvalues(args, ctx);
    return ctx->push_back<InternalFuncStmt>(
        internal_call_name_, rargs, (Type *)ret_type, with_runtime_context_);
  }
};

Operation *Operations::get(InternalOp opcode) {
  if (internals_.empty()) {
    init_internals();
  }
  return internals_[opcode].get();
}

void Operations::init_internals() {
#define PLAIN_OP(name, ret, ctx, ...)                                     \
  internals_[InternalOp::name] = std::make_unique<InternalCallOperation>( \
      #name, std::vector<DataType>{__VA_ARGS__}, ret, ctx)
#define POLY_OP(name, ctx, sig)  \
  internals_[InternalOp::name] = \
      std::make_unique<InternalCallOperation>(#name, sig, ctx)

#define COMPOSITE_EXTRACT(n) \
  PLAIN_OP(composite_extract_##n, f32, false, f32_ptr);
  COMPOSITE_EXTRACT(0);
  COMPOSITE_EXTRACT(1);
  COMPOSITE_EXTRACT(2);
  COMPOSITE_EXTRACT(3);
#undef COMPOSITE_EXTRACT

#define INSERT_TRIPLET(dt) \
  PLAIN_OP(insert_triplet_##dt, i32_void, true, u64, i32, i32, dt);
  INSERT_TRIPLET(f32);
  INSERT_TRIPLET(f64);
#undef INSERT_TRIPLET

  PLAIN_OP(linear_thread_idx, i32, true);
  PLAIN_OP(test_stack, i32_void, true);
  PLAIN_OP(test_active_mask, i32_void, true);
  PLAIN_OP(test_shfl, i32_void, true);
  PLAIN_OP(test_list_manager, i32_void, true);
  PLAIN_OP(test_node_allocator, i32_void, true);
  PLAIN_OP(test_node_allocator_gc_cpu, i32_void, true);
  PLAIN_OP(do_nothing, i32_void, true);
  PLAIN_OP(refresh_counter, i32_void, true);
  PLAIN_OP(test_internal_func_args, i32, true, f32, f32, i32);

  // CUDA ops:
  // block_barrier, grid_memfence, cuda_all_sync, cuda_any_sync, cuda_uni_sync,
  // cuda_ballot, cuda_shfl_sync, cuda_shfl_up_sync, cuda_shfl_down_sync,
  // cuda_shfl_xor_sync, cuda_match_any_sync, cuda_match_all_sync,
  // cuda_active_mask, warp_barrier

#define CUDA_VOTE_SYNC(name) \
  PLAIN_OP(cuda_##name##_sync_i32, i32, false, u32, i32)
#define CUDA_SHFL_SYNC(name, dt) \
  PLAIN_OP(cuda_##name##_sync_##dt, dt, false, u32, dt, i32, i32)
#define CUDA_MATCH_SYNC(name, dt) \
  PLAIN_OP(cuda_match_##name##_sync_##dt, u32, false, u32, dt)

  PLAIN_OP(block_barrier, i32_void, false);
  PLAIN_OP(block_barrier_and_i32, i32, false, i32);
  PLAIN_OP(block_barrier_or_i32, i32, false, i32);
  PLAIN_OP(block_barrier_count_i32, i32, false, i32);
  PLAIN_OP(grid_memfence, i32_void, false);
  CUDA_VOTE_SYNC(all);
  CUDA_VOTE_SYNC(any);
  CUDA_VOTE_SYNC(uni);
  PLAIN_OP(cuda_ballot_i32, i32, false, i32);
  CUDA_SHFL_SYNC(shfl, i32);
  CUDA_SHFL_SYNC(shfl_up, i32);
  CUDA_SHFL_SYNC(shfl_down, i32);
  CUDA_SHFL_SYNC(shfl_xor, i32);
  CUDA_SHFL_SYNC(shfl, f32);
  CUDA_SHFL_SYNC(shfl_up, f32);
  CUDA_SHFL_SYNC(shfl_down, f32);
  CUDA_MATCH_SYNC(any, i32);
  CUDA_MATCH_SYNC(all, i32);
  PLAIN_OP(cuda_active_mask, u32, false);
  PLAIN_OP(warp_barrier, i32_void, false, u32);

#undef CUDA_MATCH_SYNC
#undef CUDA_SHFL_SYNC
#undef CUDA_VOTE_SYNC

  // Vulkan ops:
  // workgroupBarrier, workgroupMemoryBarrier, localInvocationId,
  // vkGlobalThreadIdx, subgroupBarrier, subgroupMemoryBarrier, subgroupElect,
  // subgroupBroadcast, subgroupSize, subgroupInvocationId, subgroupAdd,
  // subgroupMul, subgroupMin, subgroupMax, subgroupAnd, subgroupOr,
  // subgroupXor, subgroupInclusiveAdd, subgroupInclusiveMul,
  // subgroupInclusiveMin, subgroupInclusiveMax, subgroupInclusiveAnd,
  // subgroupInclusiveOr, subgroupInclusiveXor

  auto ValueT = tyvar("ValueT");

  PLAIN_OP(workgroupBarrier, i32_void, false);
  PLAIN_OP(workgroupMemoryBarrier, i32_void, false);
  PLAIN_OP(localInvocationId, i32, false);
  PLAIN_OP(vkGlobalThreadIdx, i32, false);
  PLAIN_OP(subgroupBarrier, i32_void, false);
  PLAIN_OP(subgroupMemoryBarrier, i32_void, false);
  PLAIN_OP(subgroupElect, i32, false);
  POLY_OP(subgroupBroadcast, false, Signature({}, {ValueT, !u32}, ValueT));
  PLAIN_OP(subgroupSize, i32, false);
  PLAIN_OP(subgroupInvocationId, i32, false);
  POLY_OP(subgroupAdd, false, Signature({}, {ValueT}, ValueT));
  POLY_OP(subgroupMul, false, Signature({}, {ValueT}, ValueT));
  POLY_OP(subgroupMin, false, Signature({}, {ValueT}, ValueT));
  POLY_OP(subgroupMax, false, Signature({}, {ValueT}, ValueT));
  POLY_OP(subgroupAnd, false, Signature({}, {ValueT}, ValueT));
  POLY_OP(subgroupOr, false, Signature({}, {ValueT}, ValueT));
  POLY_OP(subgroupXor, false, Signature({}, {ValueT}, ValueT));
  POLY_OP(subgroupInclusiveAdd, false, Signature({}, {ValueT}, ValueT));
  POLY_OP(subgroupInclusiveMul, false, Signature({}, {ValueT}, ValueT));
  POLY_OP(subgroupInclusiveMin, false, Signature({}, {ValueT}, ValueT));
  POLY_OP(subgroupInclusiveMax, false, Signature({}, {ValueT}, ValueT));
  POLY_OP(subgroupInclusiveAnd, false, Signature({}, {ValueT}, ValueT));
  POLY_OP(subgroupInclusiveOr, false, Signature({}, {ValueT}, ValueT));
  POLY_OP(subgroupInclusiveXor, false, Signature({}, {ValueT}, ValueT));

#undef POLY_OP
#undef PLAIN_OP
}

}  // namespace taichi::lang
