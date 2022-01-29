#include "taichi/ir/operation_impl.h"
#include "taichi/ir/frontend_ir.h"
#include "taichi/ir/statements.h"

namespace taichi {
namespace lang {

std::vector<TypeExpr> make_real_params_from_dt(
    const std::vector<DataType> &params) {
  std::vector<TypeExpr> real_params;
  real_params.reserve(params.size());
  for (auto dt : params) {
    real_params.emplace_back(TypeSpec::dt(dt));
  }
  return real_params;
}

// This class corresponds to InternalFuncStmt.
class InternalCallOperation : public Operation {
  const std::string internal_call_name_;

 public:
  InternalCallOperation(const std::string &name,
                        const std::string &internal_name,
                        const std::vector<DataType> &params,
                        DataType result)
      : Operation(name,
                  std::vector<Constraint>(),
                  make_real_params_from_dt(params),
                  TypeSpec::dt(result)),
        internal_call_name_(internal_name) {
  }

  Stmt *flatten(Expression::FlattenContext *ctx,
                std::vector<Expr> &args) const override {
    return ctx->push_back<InternalFuncStmt>(internal_call_name_,
                                            get_all_stmts(args, ctx));
  }
};

class BinaryPrimOperation : public Operation {
 protected:
  const BinaryOpType op_type;

 public:
  static inline const Tyvar L = TypeSpec::var("LHS"), R = TypeSpec::var("RHS");
  BinaryPrimOperation(const std::string &name,
                      BinaryOpType op_type,
                      const std::vector<Trait *> &common_traits,
                      TypeExpr ret_type)
      : Operation(name,
                  {{L, common_traits}, {R, common_traits}},
                  {L, R},
                  ret_type),
        op_type(op_type) {
  }

  Stmt *flatten(Expression::FlattenContext *ctx,
                std::vector<Expr> &args) const override {
    return ctx->push_back<BinaryOpStmt>(op_type, flatten_rvalue(args[0], ctx),
                                        flatten_rvalue(args[1], ctx));
  }

  static Operation *arith(const std::string &name, BinaryOpType op_type) {
    return new BinaryPrimOperation(
        name, op_type, {StaticTraits::get().primitive}, TypeSpec::lub(L, R));
  }

  static Operation *bitwise(const std::string &name, BinaryOpType op_type) {
    return new BinaryPrimOperation(
        name, op_type, {StaticTraits::get().integral}, TypeSpec::lub(L, R));
  }

  static Operation *compare(const std::string &name, BinaryOpType op_type) {
    return new BinaryPrimOperation(name, op_type,
                                   {StaticTraits::get().primitive},
                                   TypeSpec::dt(PrimitiveType::i32));
  }
};

class UnaryPrimOperation : public Operation {
 protected:
  const UnaryOpType op_type;

 public:
  static inline const Tyvar T = TypeSpec::var("operandType");
  UnaryPrimOperation(const std::string &name,
                     UnaryOpType op_type,
                     const std::vector<Trait *> &traits,
                     TypeExpr ret_type)
      : Operation(name, {{T, traits}}, {T}, ret_type), op_type(op_type) {
  }

  Stmt *flatten(Expression::FlattenContext *ctx,
                std::vector<Expr> &args) const override {
    return ctx->push_back<UnaryOpStmt>(op_type, flatten_rvalue(args[0], ctx));
  }

  static Operation *real(const std::string &name, UnaryOpType op_type) {
    return new UnaryPrimOperation(name, op_type, {StaticTraits::get().real}, T);
  }
};

class AtomicOperation : public Operation {
 protected:
  const AtomicOpType op_type;

 public:
  static inline const Tyvar Dest = TypeSpec::var("destType"),
                            Val = TypeSpec::var("valType");
  AtomicOperation(const std::string &name,
                  AtomicOpType op_type,
                  Trait *extra_trait)
      : Operation(name,
                  {{Dest, {StaticTraits::get().scalar, extra_trait}},
                   {Val, {StaticTraits::get().primitive, extra_trait}}},
                  {Dest, Val},
                  TypeSpec::comp(Dest)),
        op_type(op_type) {
  }

  AtomicOperation(const std::string &name, AtomicOpType op_type)
      : Operation(name,
                  {{Dest, StaticTraits::get().scalar},
                   {Val, StaticTraits::get().primitive}},
                  {Dest, Val},
                  TypeSpec::comp(Dest)),
        op_type(op_type) {
  }

  Stmt *flatten(Expression::FlattenContext *ctx,
                std::vector<Expr> &args) const override = 0;
};

class AtomicPrimOperation : public AtomicOperation {
 public:
  AtomicPrimOperation(const std::string &name,
                      AtomicOpType op_type,
                      Trait *extra_trait)
      : AtomicOperation(name, op_type, extra_trait) {
  }

  AtomicPrimOperation(const std::string &name, AtomicOpType op_type)
      : AtomicOperation(name, op_type) {
  }

  Stmt *flatten(Expression::FlattenContext *ctx,
                std::vector<Expr> &args) const override {
    auto val = flatten_rvalue(args[1], ctx);
    auto dest = args[0];
    if (dest.is<IdExpression>()) {  // local variable
      // emit local store stmt
      auto alloca =
          ctx->current_block->lookup_var(dest.cast<IdExpression>()->id);
      return ctx->push_back<AtomicOpStmt>(op_type, alloca, val);
    } else {
      TI_ASSERT(dest.is<GlobalPtrExpression>() ||
                dest.is<TensorElementExpression>());
      return ctx->push_back<AtomicOpStmt>(op_type, flatten_lvalue(dest, ctx),
                                          val);
    }
  }

  static Operation *arith(const std::string &name, AtomicOpType op_type) {
    return new AtomicPrimOperation(name, op_type);
  }

  static Operation *bitwise(const std::string &name, AtomicOpType op_type) {
    return new AtomicPrimOperation(name, op_type, StaticTraits::get().integral);
  }
};

void StaticTraits::init() {
  primitive = new DynamicTrait(
      "Primitive", [](DataType ty) { return ty->is<PrimitiveType>(); });

  custom = new DynamicTrait("Custom", is_custom_type);

  scalar = new DynamicTrait("Scalar", [this](DataType ty) {
    return primitive->has_type(ty) || custom->has_type(ty);
  });

  real = new DynamicTrait("Real", is_real);

  integral = new DynamicTrait("Integral", is_integral);
}

void InternalOps::init() {
  thread_index = new InternalCallOperation("thread_index", "linear_thread_idx",
                                           {}, PrimitiveType::i32);

  insert_triplet =
      new InternalCallOperation("insert_triplet", "insert_triplet",
                                {PrimitiveType::u64, PrimitiveType::i32,
                                 PrimitiveType::i32, PrimitiveType::f32},
                                PrimitiveType::i32);

  do_nothing = new InternalCallOperation("do_nothing", "do_nothing", {},
                                         PrimitiveType::i32);

  refresh_counter = new InternalCallOperation(
      "refresh_counter", "refresh_counter", {}, PrimitiveType::i32);

  max = BinaryPrimOperation::arith("max", BinaryOpType::max);
  min = BinaryPrimOperation::arith("min", BinaryOpType::min);
  add = BinaryPrimOperation::arith("operator +", BinaryOpType::add);
  sub = BinaryPrimOperation::arith("operator -", BinaryOpType::sub);
  mul = BinaryPrimOperation::arith("operator *", BinaryOpType::mul);
  div = BinaryPrimOperation::arith("operator /", BinaryOpType::div);
  truediv = BinaryPrimOperation::arith("truediv", BinaryOpType::truediv);
  floordiv = BinaryPrimOperation::arith("operator //", BinaryOpType::floordiv);
  pow = BinaryPrimOperation::arith("operator **", BinaryOpType::pow);

  bit_and = BinaryPrimOperation::bitwise("operator &", BinaryOpType::bit_and);
  bit_or = BinaryPrimOperation::bitwise("operator |", BinaryOpType::bit_or);
  bit_xor = BinaryPrimOperation::bitwise("operator ^", BinaryOpType::bit_xor);
  bit_shl = BinaryPrimOperation::bitwise("operator <<", BinaryOpType::bit_shl);
  bit_shr = BinaryPrimOperation::bitwise("operator >>", BinaryOpType::bit_shr);
  bit_sar = BinaryPrimOperation::bitwise("operator >>", BinaryOpType::bit_sar);

  cmp_lt = BinaryPrimOperation::compare("operator <", BinaryOpType::cmp_lt);
  cmp_le = BinaryPrimOperation::compare("operator <=", BinaryOpType::cmp_le);
  cmp_gt = BinaryPrimOperation::compare("operator >", BinaryOpType::cmp_gt);
  cmp_ge = BinaryPrimOperation::compare("operator >=", BinaryOpType::cmp_ge);
  cmp_eq = BinaryPrimOperation::compare("operator ==", BinaryOpType::cmp_eq);
  cmp_ne = BinaryPrimOperation::compare("operator !=", BinaryOpType::cmp_ne);

  atan2 = new BinaryPrimOperation(
      "atan2", BinaryOpType::atan2, {StaticTraits::get().real},
      TypeSpec::lub(BinaryPrimOperation::L, BinaryPrimOperation::R));
  mod = new BinaryPrimOperation(
      "mod", BinaryOpType::mod, {StaticTraits::get().integral},
      TypeSpec::lub(BinaryPrimOperation::L, BinaryPrimOperation::R));

  floor = UnaryPrimOperation::real("floor", UnaryOpType::floor);
  ceil = UnaryPrimOperation::real("ceil", UnaryOpType::ceil);
  round = UnaryPrimOperation::real("round", UnaryOpType::round);

  sin = UnaryPrimOperation::real("sin", UnaryOpType::sin);
  asin = UnaryPrimOperation::real("asin", UnaryOpType::asin);
  cos = UnaryPrimOperation::real("cos", UnaryOpType::cos);
  acos = UnaryPrimOperation::real("acos", UnaryOpType::acos);
  tan = UnaryPrimOperation::real("tan", UnaryOpType::tan);
  tanh = UnaryPrimOperation::real("tanh", UnaryOpType::tanh);

  exp = UnaryPrimOperation::real("exp", UnaryOpType::exp);
  log = UnaryPrimOperation::real("log", UnaryOpType::log);
  sqrt = UnaryPrimOperation::real("sqrt", UnaryOpType::sqrt);
  rsqrt = UnaryPrimOperation::real("rsqrt", UnaryOpType::rsqrt);
  sgn = UnaryPrimOperation::real("sgn", UnaryOpType::sgn);

  neg = new UnaryPrimOperation("operator -", UnaryOpType::neg,
                               {StaticTraits::get().primitive},
                               UnaryPrimOperation::T);
  abs = new UnaryPrimOperation("abs", UnaryOpType::abs,
                               {StaticTraits::get().primitive},
                               UnaryPrimOperation::T);

  bit_not = new UnaryPrimOperation("operator ~", UnaryOpType::bit_not,
                                   {StaticTraits::get().integral},
                                   UnaryPrimOperation::T);
  logic_not = new UnaryPrimOperation("operator not", UnaryOpType::logic_not,
                                     {StaticTraits::get().integral},
                                     UnaryPrimOperation::T);

  atomic_add = AtomicPrimOperation::arith("operator +=", AtomicOpType::add);
  atomic_max = AtomicPrimOperation::arith("atomic_max", AtomicOpType::max);
  atomic_min = AtomicPrimOperation::arith("atomic_min", AtomicOpType::min);
  atomic_sub = new DynamicOperation(
      "operator -=", atomic_add,
      [this](Expression::FlattenContext *ctx, std::vector<Expr> &args) {
        args[1].set(
            Expr::make<CallExpression>(neg, std::vector<Expr>{args[1]}));
        args[1].type_check();
        return atomic_add->flatten(ctx, args);
      });

  atomic_bit_and =
      AtomicPrimOperation::bitwise("operator &=", AtomicOpType::bit_and);
  atomic_bit_or =
      AtomicPrimOperation::bitwise("operator |=", AtomicOpType::bit_or);
  atomic_bit_xor =
      AtomicPrimOperation::bitwise("operator ^=", AtomicOpType::bit_xor);

  {
    auto Res = TypeSpec::var("Res");
    select = new DynamicOperation(
        "select", {{Res, StaticTraits::get().primitive}},
        {TypeSpec::dt(PrimitiveType::i32), Res, Res}, Res,
        [](Expression::FlattenContext *ctx, std::vector<Expr> &args) {
          return ctx->push_back(std::make_unique<TernaryOpStmt>(
              TernaryOpType::select, flatten_rvalue(args[0], ctx),
              flatten_rvalue(args[1], ctx), flatten_rvalue(args[2], ctx)));
        });
  }
}

void TestInternalOps::init() {
  test_active_mask = new InternalCallOperation(
      "test_active_mask", "test_active_mask", {}, PrimitiveType::i32);

  test_shfl = new InternalCallOperation("test_shfl", "test_shfl", {},
                                        PrimitiveType::i32);

  test_stack = new InternalCallOperation("test_stack", "test_stack", {},
                                         PrimitiveType::i32);

  test_list_manager = new InternalCallOperation(
      "test_list_manager", "test_list_manager", {}, PrimitiveType::i32);

  test_node_allocator = new InternalCallOperation(
      "test_node_allocator", "test_node_allocator", {}, PrimitiveType::i32);

  test_node_allocator_gc_cpu = new InternalCallOperation(
      "test_node_allocator_gc_cpu", "test_node_allocator_gc_cpu", {},
      PrimitiveType::i32);

  test_internal_func_args = new InternalCallOperation(
      "test_internal_func_args", "test_internal_func_args",
      {PrimitiveType::f32, PrimitiveType::f32, PrimitiveType::i32},
      PrimitiveType::i32);
}

#define SINGLETON(cls)                          \
  const cls &cls::get() {                       \
    if (!instance_) {                           \
      instance_ = std::make_unique<cls>(cls()); \
      instance_->init();                        \
    }                                           \
    return *instance_;                          \
  }

SINGLETON(StaticTraits)
SINGLETON(InternalOps)
SINGLETON(TestInternalOps)

#undef SINGLETON

}  // namespace lang
}  // namespace taichi
