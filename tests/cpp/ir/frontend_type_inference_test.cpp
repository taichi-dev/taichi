#include "gtest/gtest.h"

#include "taichi/ir/frontend_ir.h"
#include "tests/cpp/program/test_program.h"

namespace taichi {
namespace lang {

TEST(Type, BitTypesO) {
  auto i32 = TypeFactory::get_instance()
                 .get_primitive_type(PrimitiveTypeID::i32)
                 ->as<PrimitiveType>();
  auto ci5 = TypeFactory::get_instance().get_custom_int_type(5, true, i32);
  auto cu11 = TypeFactory::get_instance().get_custom_int_type(11, false, i32);
  auto u16 = TypeFactory::get_instance().get_primitive_int_type(16, false);

  auto bs =
      TypeFactory::get_instance().get_bit_struct_type(u16, {ci5, cu11}, {0, 5});

  EXPECT_EQ(bs->to_string(), "bs(ci5@0, cu11@5)");

  auto ci1 = TypeFactory::get_instance().get_custom_int_type(1, true, i32);
  auto ba = TypeFactory::get_instance().get_bit_array_type(i32, ci1, 32);

  EXPECT_EQ(ba->to_string(), "ba(ci1x32)");
}

TEST(FrontendTypeInference, Const) {
  auto const_i64 = Expr::make<ConstExpression, int64>(1LL << 63);
  EXPECT_EQ(const_i64->ret_type, PrimitiveType::i64);
}

TEST(FrontendTypeInference, ArgLoad) {
  auto arg_load_u64 = Expr::make<ArgLoadExpression>(2, PrimitiveType::u64);
  EXPECT_EQ(arg_load_u64->ret_type, PrimitiveType::u64);
}

TEST(FrontendTypeInference, Rand) {
  auto rand_f16 = Expr::make<RandExpression>(PrimitiveType::f16);
  EXPECT_EQ(rand_f16->ret_type, PrimitiveType::f16);
}

TEST(FrontendTypeInference, Id) {
  auto prog = std::make_unique<Program>(Arch::x64);
  auto func = []() {};
  auto kernel = std::make_unique<Kernel>(*prog, func, "fake_kernel");
  Callable::CurrentCallableGuard _(kernel->program, kernel.get());
  auto const_i32 = Expr::make<ConstExpression, int32>(-(1 << 20));
  auto id_i32 = Var(const_i32);
  EXPECT_EQ(id_i32->ret_type, PrimitiveType::i32);
}

TEST(FrontendTypeInference, BinaryOp) {
  auto prog = std::make_unique<Program>(Arch::x64);
  prog->config.default_fp = PrimitiveType::f64;
  auto const_i32 = Expr::make<ConstExpression, int32>(-(1 << 20));
  auto const_f32 = Expr::make<ConstExpression, float32>(5.0);
  auto truediv_f64 = expr_truediv(const_i32, const_f32);
  EXPECT_EQ(truediv_f64->ret_type, PrimitiveType::f64);
}

}  // namespace lang
}  // namespace taichi
