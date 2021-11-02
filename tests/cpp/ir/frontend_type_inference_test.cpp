#include "gtest/gtest.h"

#include "taichi/ir/frontend_ir.h"
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
  auto id_i32 = Var(const_i32);
  EXPECT_EQ(id_i32->ret_type, PrimitiveType::i32);
}

TEST(FrontendTypeInference, BinaryOp) {
  auto prog = std::make_unique<Program>(Arch::x64);
  prog->config.default_fp = PrimitiveType::f64;
  auto const_i32 = Expr::make<ConstExpression, int32>(-(1 << 20));
  const_i32->type_check();
  auto const_f32 = Expr::make<ConstExpression, float32>(5.0);
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

}  // namespace lang
}  // namespace taichi
