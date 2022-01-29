#include "gtest/gtest.h"

#include "taichi/ir/frontend_ir.h"
#include "taichi/ir/operation_impl.h"
#include "taichi/program/program.h"

namespace taichi {
namespace lang {

TEST(FrontendTypeInference, Const) {
  auto const_i64 = Expr::make<ConstExpression, int64>(1LL << 63);
  const_i64->type_check();
  EXPECT_EQ(const_i64->ret_type, PrimitiveType::i64);
}

TEST(FrontendTypeInference, ArgLoad) {
  auto arg_load_u64 = Expr::make<ArgLoadExpression>(2, PrimitiveType::u64);
  arg_load_u64->type_check();
  EXPECT_EQ(arg_load_u64->ret_type, PrimitiveType::u64);
}

TEST(FrontendTypeInference, Rand) {
  auto rand_f16 = Expr::make<RandExpression>(PrimitiveType::f16);
  rand_f16->type_check();
  EXPECT_EQ(rand_f16->ret_type, PrimitiveType::f16);
}

TEST(FrontendTypeInference, Id) {
  auto prog = std::make_unique<Program>(Arch::x64);
  auto func = []() {};
  auto kernel = std::make_unique<Kernel>(*prog, func, "fake_kernel");
  Callable::CurrentCallableGuard _(kernel->program, kernel.get());
  auto const_i32 = Expr::make<ConstExpression, int32>(-(1 << 20));
  const_i32->type_check();
  auto id_i32 = prog->current_ast_builder()->make_var(const_i32);
  EXPECT_EQ(id_i32->ret_type, PrimitiveType::i32);
}

TEST(FrontendTypeInference, BinaryOp) {
  auto prog = std::make_unique<Program>(Arch::x64);
  auto const_i32 = Expr::make<ConstExpression, int32>(-(1 << 20));
  const_i32->type_check();
  auto const_f32 = Expr::make<ConstExpression, float64>(5.0);
  const_f32->type_check();
  auto truediv_f64 = expr_truediv(const_i32, const_f32);
  truediv_f64->type_check();
  EXPECT_EQ(truediv_f64->ret_type, PrimitiveType::f64);
}

TEST(FrontendTypeInference, UnaryOp) {
  auto const_i16 = Expr::make<ConstExpression, int16>(-(1 << 10));
  const_i16->type_check();
  EXPECT_EQ(const_i16->ret_type, PrimitiveType::i16);
  auto cast_i8 = cast(const_i16, PrimitiveType::i8);
  cast_i8->type_check();
  EXPECT_EQ(cast_i8->ret_type, PrimitiveType::i8);
  auto bit_not_i16 = ~const_i16;
  bit_not_i16->type_check();
  EXPECT_EQ(bit_not_i16->ret_type, PrimitiveType::i16);
}

TEST(FrontendTypeInference, TernaryOp) {
  auto const_i32 = Expr::make<ConstExpression, int32>(-(1 << 10));
  const_i32->type_check();
  EXPECT_EQ(const_i32->ret_type, PrimitiveType::i32);
  auto cast_i8 = cast(const_i32, PrimitiveType::i8);
  cast_i8->type_check();
  EXPECT_EQ(cast_i8->ret_type, PrimitiveType::i8);
  auto const_i8 = Expr::make<ConstExpression, int8>(5);
  const_i8->type_check();
  EXPECT_EQ(const_i8->ret_type, PrimitiveType::i8);
  auto ternary_i8 = expr_select(const_i32, cast_i8, const_i8);
  ternary_i8->type_check();
  EXPECT_EQ(ternary_i8->ret_type, PrimitiveType::i8);
}

TEST(FrontendTypeInference, GlobalPtr_GlobalVariable) {
  auto snode = std::make_unique<SNode>(0, SNodeType::root);
  snode->dt = PrimitiveType::u8;
  auto global_var = Expr::make<GlobalVariableExpression>(snode.get());
  auto index = Expr::make<ConstExpression, float32>(2);
  index->type_check();
  auto global_ptr =
      Expr::make<GlobalPtrExpression>(global_var, ExprGroup(index));
  global_ptr->type_check();
  EXPECT_EQ(global_ptr->ret_type, PrimitiveType::u8);
}

TEST(FrontendTypeInference, GlobalPtr_ExternalTensor) {
  auto index = Expr::make<ConstExpression, float32>(2);
  index->type_check();
  auto external_tensor =
      Expr::make<ExternalTensorExpression>(PrimitiveType::u16, 1, 0, 0);
  auto global_ptr =
      Expr::make<GlobalPtrExpression>(external_tensor, ExprGroup(index));
  EXPECT_THROW(global_ptr->type_check(), TaichiTypeError);
}

TEST(FrontendTypeInference, TensorElement) {
  auto prog = std::make_unique<Program>(Arch::x64);
  auto func = []() {};
  auto kernel = std::make_unique<Kernel>(*prog, func, "fake_kernel");
  Callable::CurrentCallableGuard _(kernel->program, kernel.get());
  const std::vector<int> shape{3};
  auto var = Expr(std::make_shared<IdExpression>());
  current_ast_builder().insert(std::make_unique<FrontendAllocaStmt>(
      std::static_pointer_cast<IdExpression>(var.expr)->id, shape,
      PrimitiveType::u32));
  var->ret_type = current_ast_builder().get_last_stmt()->ret_type;
  auto index = Expr::make<ConstExpression, int32>(2);
  index->type_check();
  auto tensor_element =
      Expr::make<TensorElementExpression>(var, ExprGroup(index), shape, 1);
  tensor_element->type_check();
  EXPECT_EQ(tensor_element->ret_type, PrimitiveType::u32);
}

TEST(FrontendTypeInference, AtomicOp) {
  auto const_i32 = Expr::make<ConstExpression, int32>(-(1 << 20));
  const_i32->type_check();
  auto const_f32 = Expr::make<ConstExpression, float32>(5.0);
  const_f32->type_check();
  auto atomic_add_i32 =
      InternalOps::get().atomic_add->call(const_i32, const_f32);
  atomic_add_i32->type_check();
  EXPECT_EQ(atomic_add_i32->ret_type, PrimitiveType::i32);
}

TEST(FrontendTypeInference, SNodeOp) {
  auto snode = std::make_unique<SNode>(0, SNodeType::root);
  snode->dt = PrimitiveType::u8;
  auto index = Expr::make<ConstExpression, int32>(2);
  index->type_check();
  auto snode_op = Expr::make<SNodeOpExpression>(
      snode.get(), SNodeOpType::get_addr, ExprGroup(index));
  snode_op->type_check();
  EXPECT_EQ(snode_op->ret_type, PrimitiveType::u64);
}

TEST(FrontendTypeInference, ExternalTensorShapeAlongAxis) {
  auto external_tensor =
      Expr::make<ExternalTensorExpression>(PrimitiveType::u64, 1, 0, 0);
  auto shape =
      Expr::make<ExternalTensorShapeAlongAxisExpression>(external_tensor, 0);
  shape->type_check();
  EXPECT_EQ(shape->ret_type, PrimitiveType::i32);
}

TEST(FrontendTypeInference, RangeAssumption) {
  auto const_f32_a = Expr::make<ConstExpression, float32>(5.0);
  const_f32_a->type_check();
  auto const_f32_b = Expr::make<ConstExpression, float32>(5.0);
  const_f32_b->type_check();
  auto valid =
      Expr::make<RangeAssumptionExpression>(const_f32_a, const_f32_b, 0, 1);
  valid->type_check();
  EXPECT_EQ(valid->ret_type, PrimitiveType::f32);
  auto const_f64 = Expr::make<ConstExpression, float64>(5.0);
  const_f64->type_check();
  auto invalid =
      Expr::make<RangeAssumptionExpression>(const_f32_a, const_f64, 0, 1);
  EXPECT_THROW(invalid->type_check(), TaichiTypeError);
}

TEST(FrontendTypeInference, LoopUnique) {
  auto const_i64 = Expr::make<ConstExpression, int64>(5);
  const_i64->type_check();
  auto loop_unique =
      Expr::make<LoopUniqueExpression>(const_i64, std::vector<SNode *>{});
  loop_unique->type_check();
  EXPECT_EQ(loop_unique->ret_type, PrimitiveType::i64);
}

TEST(FrontendTypeInference, InternalFuncCall) {
  auto internal_func_call = InternalOps::get().do_nothing->call();
  internal_func_call->type_check();
  EXPECT_EQ(internal_func_call->ret_type, PrimitiveType::i32);
}

}  // namespace lang
}  // namespace taichi
