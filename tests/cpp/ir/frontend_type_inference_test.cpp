#include "gtest/gtest.h"

#include "taichi/ir/frontend_ir.h"
#include "taichi/program/program.h"
#include "taichi/ir/expression_ops.h"
#include "taichi/program/compile_config.h"

namespace taichi::lang {

TEST(FrontendTypeInference, Const) {
  auto const_i64 = value<int64>(1LL << 63);
  const_i64->type_check(nullptr);
  EXPECT_EQ(const_i64->ret_type, PrimitiveType::i64);
}

TEST(FrontendTypeInference, ArgLoad) {
  auto arg_load_u64 =
      Expr::make<ArgLoadExpression>(std::vector<int>{2}, PrimitiveType::u64);
  arg_load_u64->type_check(nullptr);
  EXPECT_EQ(arg_load_u64->ret_type, PrimitiveType::u64);
}

TEST(FrontendTypeInference, Rand) {
  auto rand_f16 = expr_rand(PrimitiveType::f16);
  rand_f16->type_check(nullptr);
  EXPECT_EQ(rand_f16->ret_type, PrimitiveType::f16);
}

TEST(FrontendTypeInference, Id) {
  auto prog = std::make_unique<Program>(Arch::x64);
  auto func = []() {};
  auto kernel = std::make_unique<Kernel>(*prog, func, "fake_kernel");
  auto const_i32 = value<int32>(-(1 << 20));
  const_i32->type_check(nullptr);
  auto id_i32 =
      kernel->context->builder().make_var(const_i32, const_i32->dbg_info);
  EXPECT_EQ(id_i32->ret_type,
            DataType(TypeFactory::get_instance().get_pointer_type(
                PrimitiveType::i32)));
}

TEST(FrontendTypeInference, BinaryOp) {
  default_compile_config.default_fp = PrimitiveType::f64;
  auto prog = std::make_unique<Program>(Arch::x64);
  auto const_i32 = value<int32>(-(1 << 20));
  const_i32->type_check(nullptr);
  auto const_f32 = value<float32>(5.0);
  const_f32->type_check(nullptr);
  auto truediv_f64 = expr_truediv(const_i32, const_f32);
  truediv_f64->type_check(&prog->compile_config());
  EXPECT_EQ(truediv_f64->ret_type, PrimitiveType::f64);
}

TEST(FrontendTypeInference, UnaryOp) {
  default_compile_config.default_fp = PrimitiveType::f64;
  auto prog = std::make_unique<Program>(Arch::x64);
  auto const_i16 = value<int16>(-(1 << 10));

  CompileConfig dummy_config;
  const_i16->type_check(nullptr);
  EXPECT_EQ(const_i16->ret_type, PrimitiveType::i16);
  auto cast_i8 = cast(const_i16, PrimitiveType::i8);
  cast_i8->type_check(&dummy_config);
  EXPECT_EQ(cast_i8->ret_type, PrimitiveType::i8);
  auto bit_not_i16 = ~const_i16;
  bit_not_i16->type_check(&dummy_config);
  EXPECT_EQ(bit_not_i16->ret_type, PrimitiveType::i16);
  auto log_f64 = expr_log(const_i16);
  log_f64->type_check(&prog->compile_config());
  EXPECT_EQ(log_f64->ret_type, PrimitiveType::f64);
}

TEST(FrontendTypeInference, TernaryOp) {
  auto const_u1 = value<uint1>(true);
  const_u1->type_check(nullptr);
  EXPECT_EQ(const_u1->ret_type, PrimitiveType::u1);
  auto const_i32 = value<int32>(-(1 << 10));
  const_i32->type_check(nullptr);
  EXPECT_EQ(const_i32->ret_type, PrimitiveType::i32);
  auto cast_i8 = cast(const_i32, PrimitiveType::i8);

  CompileConfig dummy_config;
  cast_i8->type_check(&dummy_config);
  EXPECT_EQ(cast_i8->ret_type, PrimitiveType::i8);

  auto const_f32 = value<float32>(5.0);
  const_f32->type_check(nullptr);
  EXPECT_EQ(const_f32->ret_type, PrimitiveType::f32);
  auto ternary_f32 = expr_select(const_u1, cast_i8, const_f32);
  ternary_f32->type_check(nullptr);
  EXPECT_EQ(ternary_f32->ret_type, PrimitiveType::f32);
}

TEST(FrontendTypeInference, TernaryOp_NoBroadcast) {
  auto cond = value<uint1>(true);
  cond->type_check(nullptr);
  EXPECT_EQ(cond->ret_type, PrimitiveType::u1);

  auto const_3 = Expr::make<ConstExpression, int32>(3);
  auto const_5 = Expr::make<ConstExpression, int32>(5);
  std::vector<int> shape = {3, 1};

  std::vector<Expr> op2_element = {const_3, const_3, const_3};
  std::vector<Expr> op3_element = {const_5, const_5, const_5};

  auto op2 =
      Expr::make<MatrixExpression>(op2_element, shape, PrimitiveType::i32);
  op2->type_check(nullptr);
  auto op3 =
      Expr::make<MatrixExpression>(op2_element, shape, PrimitiveType::i32);
  op3->type_check(nullptr);

  auto ternary =
      Expr::make<TernaryOpExpression>(TernaryOpType::select, cond, op2, op3);
  ternary->type_check(nullptr);

  auto ternary_expr = ternary.cast<TernaryOpExpression>();
  auto cond_ret_type = ternary_expr->op1->ret_type;
  auto op2_ret_type = ternary_expr->op2->ret_type;
  auto op3_ret_type = ternary_expr->op3->ret_type;

  EXPECT_TRUE(op2_ret_type->is<TensorType>() &&
              op2_ret_type->cast<TensorType>()->get_shape() == shape);
  EXPECT_TRUE(op3_ret_type->is<TensorType>() &&
              op3_ret_type->cast<TensorType>()->get_shape() == shape);
  EXPECT_EQ(cond_ret_type, PrimitiveType::u1);
}

TEST(FrontendTypeInference, GlobalPtr_Field) {
  auto prog = std::make_unique<Program>(Arch::x64);
  auto func = []() {};
  auto kernel = std::make_unique<Kernel>(*prog, func, "fake_kernel");
  auto *ast_builder = &kernel->context->builder();

  auto global_var =
      Expr::make<FieldExpression>(PrimitiveType::u8, Identifier(0));
  SNode snode;
  snode.num_active_indices = 1;
  std::dynamic_pointer_cast<FieldExpression>(global_var.expr)
      ->set_snode(&snode);

  auto index = value<int32>(2);
  index->type_check(nullptr);
  auto global_ptr = ast_builder->expr_subscript(global_var, ExprGroup(index));
  global_ptr->type_check(nullptr);
  EXPECT_EQ(global_ptr->ret_type,
            DataType(TypeFactory::get_instance().get_pointer_type(
                PrimitiveType::u8)));
}

TEST(FrontendTypeInference, GlobalPtr_ExternalTensor) {
  auto prog = std::make_unique<Program>(Arch::x64);
  auto func = []() {};
  auto kernel = std::make_unique<Kernel>(*prog, func, "fake_kernel");
  auto *ast_builder = &kernel->context->builder();

  auto index = value<float32>(2);
  index->type_check(nullptr);
  auto external_tensor = Expr::make<ExternalTensorExpression>(
      PrimitiveType::u16, 1, std::vector<int>{0}, 0);
  auto global_ptr =
      ast_builder->expr_subscript(external_tensor, ExprGroup(index));
  EXPECT_THROW(global_ptr->type_check(nullptr), TaichiTypeError);
}

TEST(FrontendTypeInference, TensorElement) {
  auto prog = std::make_unique<Program>(Arch::x64);
  auto func = []() {};
  auto kernel = std::make_unique<Kernel>(*prog, func, "fake_kernel");
  auto *ast_builder = &kernel->context->builder();
  const std::vector<int> shape{3};
  auto var = Expr(std::make_shared<IdExpression>(ast_builder->get_next_id()));
  ast_builder->insert(std::make_unique<FrontendAllocaStmt>(
      std::static_pointer_cast<IdExpression>(var.expr)->id, shape,
      PrimitiveType::u32));
  var->ret_type = ast_builder->get_last_stmt()->ret_type;
  auto index = value<int32>(2);
  index->type_check(nullptr);
  auto tensor_element = Expr::make<IndexExpression>(var, ExprGroup(index));
  tensor_element->type_check(nullptr);
  EXPECT_EQ(tensor_element->ret_type,
            DataType(TypeFactory::get_instance().get_pointer_type(
                PrimitiveType::u32)));
}

TEST(FrontendTypeInference, AtomicOp) {
  auto const_i32 = value<int32>(-(1 << 20));
  const_i32->type_check(nullptr);
  auto const_f32 = value<float32>(5.0);
  const_f32->type_check(nullptr);
  auto atomic_add_i32 =
      Expr::make<AtomicOpExpression>(AtomicOpType::add, const_i32, const_f32);
  atomic_add_i32->type_check(nullptr);
  EXPECT_EQ(atomic_add_i32->ret_type, PrimitiveType::i32);
}

TEST(FrontendTypeInference, SNodeOp) {
  auto prog = std::make_unique<Program>(Arch::x64);
  auto func = []() {};
  auto kernel = std::make_unique<Kernel>(*prog, func, "fake_kernel");
  auto snode = std::make_unique<SNode>(0, SNodeType::root);
  snode->dt = PrimitiveType::u8;
  auto index = value<int32>(2);
  index->type_check(nullptr);
  auto snode_op =
      kernel->context->builder().snode_get_addr(snode.get(), ExprGroup(index));
  snode_op->type_check(nullptr);
  EXPECT_EQ(snode_op->ret_type, PrimitiveType::u64);
}

TEST(FrontendTypeInference, ExternalTensorShapeAlongAxis) {
  auto external_tensor = Expr::make<ExternalTensorExpression>(
      PrimitiveType::u64, 1, std::vector<int>{0}, 0);
  auto shape =
      Expr::make<ExternalTensorShapeAlongAxisExpression>(external_tensor, 0);
  shape->type_check(nullptr);
  EXPECT_EQ(shape->ret_type, PrimitiveType::i32);
}

TEST(FrontendTypeInference, RangeAssumption) {
  auto const_f32_a = value<float32>(5.0);
  const_f32_a->type_check(nullptr);
  auto const_f32_b = value<float32>(5.0);
  const_f32_b->type_check(nullptr);
  auto valid = assume_range(const_f32_a, const_f32_b, 0, 1);
  valid->type_check(nullptr);
  EXPECT_EQ(valid->ret_type, PrimitiveType::f32);
  auto const_f64 = value<float64>(5.0);
  const_f64->type_check(nullptr);
  auto invalid = assume_range(const_f32_a, const_f64, 0, 1);
  EXPECT_THROW(invalid->type_check(nullptr), TaichiTypeError);
}

TEST(FrontendTypeInference, LoopUnique) {
  auto const_i64 = value<int64>(5);
  const_i64->type_check(nullptr);
  auto lunique = loop_unique(const_i64, {});
  lunique->type_check(nullptr);
  EXPECT_EQ(lunique->ret_type, PrimitiveType::i64);
}

TEST(FrontendTypeInference, InternalFuncCall) {
  auto internal_func_call = Expr::make<InternalFuncCallExpression>(
      Operations::get(InternalOp::do_nothing), std::vector<Expr>{});
  internal_func_call->type_check(nullptr);
  EXPECT_EQ(internal_func_call->ret_type, PrimitiveType::i32);
}

TEST(FrontendTypeInference, TensorTypeUnification) {
  // lhs mat, rhs const
  auto element = Expr::make<ConstExpression, int32>(1);
  std::vector<Expr> elements = {element, element};
  std::vector<int> shape = {2, 1};
  auto mat = Expr::make<MatrixExpression>(elements, shape, PrimitiveType::i32);
  mat->type_check(nullptr);
  auto const_val = Expr::make<ConstExpression, int32>(2);
  const_val->type_check(nullptr);
  auto expr = Expr::make<BinaryOpExpression>(BinaryOpType::add, mat, const_val);
  expr->type_check(nullptr);
  auto binaryop_expr = expr.cast<BinaryOpExpression>();
  EXPECT_TRUE(binaryop_expr->rhs->ret_type->is<TensorType>());
  auto rhs_type = binaryop_expr->rhs->ret_type->cast<TensorType>();
  auto ret_type = binaryop_expr->ret_type;
  auto expected_shape = std::vector<int>({2, 1});
  EXPECT_TRUE(ret_type->is<TensorType>() &&
              ret_type->cast<TensorType>()->get_shape() == expected_shape);
  EXPECT_TRUE(rhs_type->get_shape() == expected_shape);

  // lhs const, rhs mat
  expr = Expr::make<BinaryOpExpression>(BinaryOpType::div, const_val, mat);
  expr->type_check(nullptr);
  binaryop_expr = expr.cast<BinaryOpExpression>();
  EXPECT_TRUE(binaryop_expr->rhs->ret_type->is<TensorType>());
  auto lhs_type = binaryop_expr->lhs->ret_type->cast<TensorType>();
  ret_type = binaryop_expr->ret_type;
  EXPECT_TRUE(ret_type->is<TensorType>() &&
              ret_type->cast<TensorType>()->get_shape() == expected_shape);
  EXPECT_TRUE(lhs_type->get_shape() == expected_shape);
}
}  // namespace taichi::lang
