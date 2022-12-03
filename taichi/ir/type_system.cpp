#include "taichi/ir/type_system.h"
#include "taichi/ir/statements.h"
#include "taichi/ir/frontend_ir.h"

namespace taichi::lang {

void TyVar::unify(int pos,
                  DataType dt,
                  std::map<Identifier, DataType> &solutions) const {
  if (solutions.find(name_) != solutions.end()) {
    if (solutions[name_] != dt) {
      throw TyVarMismatch(name_, solutions[name_], dt);
    }
  } else {
    solutions[name_] = dt;
  }
}

DataType TyVar::resolve(const std::map<Identifier, DataType> &solutions) const {
  if (solutions.find(name_) == solutions.end()) {
    throw TyVarUnsolved(name_);
  } else {
    return solutions.at(name_);
  }
}

std::string TyVar::to_string() const {
  return name_.name();
}

void TyLub::unify(int pos,
                  DataType dt,
                  std::map<Identifier, DataType> &solutions) const {
  TyMono(resolve(solutions)).unify(pos, dt, solutions);
}

DataType TyLub::resolve(const std::map<Identifier, DataType> &solutions) const {
  return promoted_type(lhs_->resolve(solutions)->get_compute_type(),
                       rhs_->resolve(solutions)->get_compute_type());
}

std::string TyLub::to_string() const {
  return lhs_->to_string() + " | " + rhs_->to_string();
}

void TyCompute::unify(int pos,
                      DataType dt,
                      std::map<Identifier, DataType> &solutions) const {
  TyMono(resolve(solutions)).unify(pos, dt, solutions);
}

DataType TyCompute::resolve(
    const std::map<Identifier, DataType> &solutions) const {
  return exp_->resolve(solutions)->get_compute_type();
}

std::string TyCompute::to_string() const {
  return "comp(" + exp_->to_string() + ")";
}

void TyMono::unify(int pos,
                   DataType dt,
                   std::map<Identifier, DataType> &solutions) const {
  if (monotype_ != dt) {
    throw TypeMismatch(pos, monotype_, dt);
  }
}

DataType TyMono::resolve(
    const std::map<Identifier, DataType> &solutions) const {
  return monotype_;
}

std::string TyMono::to_string() const {
  return monotype_.to_string();
}

std::string TyVarMismatch::to_string() const {
  return "expected " + original_.to_string() + ", but got " +
         conflicting_.to_string();
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
  return "the argument type " + dt_.to_string() + " is not " +
         trait_->to_string();
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
  std::map<Identifier, DataType> solutions;
  for (int i = 0; i < parameters_.size(); i++) {
    parameters_[i]->unify(i, arguments[i], solutions);
  }
  for (auto &c : constraints_) {
    auto dt = c.tyvar->resolve(solutions);
    if (!c.trait->validate(dt)) {
      throw TraitMismatch(dt, c.trait);
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

namespace TypeExprBuilder {

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
DataType i0 = i32;
DataType f32_ptr = TypeFactory::get_instance().get_pointer_type(f32, false);

#undef PRIM

Trait *Real = StaticTraits::get(StaticTraitID::real);
Trait *Integral = StaticTraits::get(StaticTraitID::integral);
Trait *Primitive = StaticTraits::get(StaticTraitID::primitive);
Trait *Scalar = StaticTraits::get(StaticTraitID::scalar);

Constraint operator<(std::shared_ptr<TyVar> tyvar, Trait *trait) {
  return Constraint(tyvar, trait);
}

std::shared_ptr<TyMono> operator!(DataType dt) {
  return std::make_shared<TyMono>(dt);
}

std::shared_ptr<TyLub> operator|(TypeExpr lhs, TypeExpr rhs) {
  return std::make_shared<TyLub>(lhs, rhs);
}

std::shared_ptr<TyVar> tyvar(std::string name) {
  return std::make_shared<TyVar>(Identifier(var_counter_++, name));
}

std::shared_ptr<TyCompute> comp(TypeExpr ty) {
  return std::make_shared<TyCompute>(ty);
}
};  // namespace TypeExprBuilder

using namespace TypeExprBuilder;

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
    flatten_rvalue(arg, ctx);
    stmts.push_back(arg->stmt);
  }
  return stmts;
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
  PLAIN_OP(insert_triplet_##dt, i0, true, u64, i32, i32, dt);
  INSERT_TRIPLET(f32);
  INSERT_TRIPLET(f64);
#undef INSERT_TRIPLET

  PLAIN_OP(linear_thread_idx, i32, true);
  PLAIN_OP(test_stack, i0, true);
  PLAIN_OP(test_active_mask, i0, true);
  PLAIN_OP(test_shfl, i0, true);
  PLAIN_OP(test_list_manager, i0, true);
  PLAIN_OP(test_node_allocator, i0, true);
  PLAIN_OP(test_node_allocator_gc_cpu, i0, true);
  PLAIN_OP(do_nothing, i0, true);
  PLAIN_OP(refresh_counter, i0, true);
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

  PLAIN_OP(block_barrier, i0, false);
  PLAIN_OP(grid_memfence, i0, false);
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
  PLAIN_OP(warp_barrier, i0, false, u32);

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

  auto lhs = tyvar("lhs"), rhs = tyvar("rhs");

  PLAIN_OP(workgroupBarrier, i0, false);
  PLAIN_OP(workgroupMemoryBarrier, i0, false);
  PLAIN_OP(localInvocationId, i32, false);
  PLAIN_OP(vkGlobalThreadIdx, i32, false);
  PLAIN_OP(subgroupBarrier, i0, false);
  PLAIN_OP(subgroupMemoryBarrier, i0, false);
  PLAIN_OP(subgroupElect, i32, false);
  POLY_OP(subgroupBroadcast, false, Signature({}, {lhs}, lhs));
  PLAIN_OP(subgroupSize, i32, false);
  PLAIN_OP(subgroupInvocationId, i32, false);
  POLY_OP(subgroupAdd, false, Signature({}, {lhs}, lhs));
  POLY_OP(subgroupMul, false, Signature({}, {lhs}, lhs));
  POLY_OP(subgroupMin, false, Signature({}, {lhs}, lhs));
  POLY_OP(subgroupMax, false, Signature({}, {lhs}, lhs));
  POLY_OP(subgroupAnd, false, Signature({}, {lhs}, lhs));
  POLY_OP(subgroupOr, false, Signature({}, {lhs}, lhs));
  POLY_OP(subgroupXor, false, Signature({}, {lhs}, lhs));
  POLY_OP(subgroupInclusiveAdd, false, Signature({}, {lhs}, lhs));
  POLY_OP(subgroupInclusiveMul, false, Signature({}, {lhs}, lhs));
  POLY_OP(subgroupInclusiveMin, false, Signature({}, {lhs}, lhs));
  POLY_OP(subgroupInclusiveMax, false, Signature({}, {lhs}, lhs));
  POLY_OP(subgroupInclusiveAnd, false, Signature({}, {lhs}, lhs));
  POLY_OP(subgroupInclusiveOr, false, Signature({}, {lhs}, lhs));
  POLY_OP(subgroupInclusiveXor, false, Signature({}, {lhs}, lhs));

#undef POLY_OP
#undef PLAIN_OP
}

}  // namespace taichi::lang
