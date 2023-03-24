#include "gtest/gtest.h"

#include "taichi/ir/statements.h"
#include "taichi/ir/ir_builder.h"
#include "taichi/ir/transforms.h"
#include "tests/cpp/program/test_program.h"

namespace taichi::lang {

class ConstantFoldTest : public ::testing::Test {
 protected:
  void SetUp() override {
    tp_.setup();
  }

  TestProgram tp_;
};

TEST_F(ConstantFoldTest, Unary) {
  IRBuilder builder;

  auto *x = builder.create_arg_load(0, get_data_type<float>(), false);
  auto *sqrt = builder.create_sqrt(builder.get_uint32(10));
  auto *neg = builder.create_neg(sqrt);
  auto *round = builder.create_round(neg);
  auto *floor = builder.create_floor(round);
  auto *bit_cast = builder.create_bit_cast(floor, PrimitiveType::f32);
  auto *abs = builder.create_abs(bit_cast);
  auto *sin = builder.create_sin(abs);
  auto *asin = builder.create_asin(sin);
  auto *cos = builder.create_cos(asin);
  auto *acos = builder.create_acos(cos);
  auto *tan = builder.create_tan(acos);
  auto *tanh = builder.create_tanh(tan);
  auto *exp = builder.create_exp(tanh);
  auto *log = builder.create_log(exp);
  auto *rsqrt = builder.create_rsqrt(log);
  auto *ceil = builder.create_ceil(rsqrt);
  auto *cast_value = builder.create_cast(ceil, PrimitiveType::i32);
  auto *logical_not = builder.create_logical_not(cast_value);
  auto *bit_not = builder.create_not(logical_not);
  auto *result = builder.create_sub(x, bit_not);
  builder.create_return(result);

  auto ir = builder.extract_ir();
  ASSERT_TRUE(ir->is<Block>());
  auto *ir_block = ir->as<Block>();
  irpass::type_check(ir_block, CompileConfig());
  EXPECT_EQ(ir_block->size(), 25);

  irpass::constant_fold(ir_block, CompileConfig(), {tp_.prog()});
  irpass::die(ir_block);

  EXPECT_EQ(ir_block->size(), 4);
  EXPECT_EQ(ir_block->statements[0].get(), x);
  EXPECT_TRUE(ir_block->statements[1]->is<ConstStmt>());
  EXPECT_EQ(ir_block->statements[1]->as<ConstStmt>()->val.val_float(), -1);
}

TEST_F(ConstantFoldTest, Binary) {
  IRBuilder builder;

  auto *x = builder.create_arg_load(0, get_data_type<float>(), false);
  auto *one = builder.get_int32(1);
  auto *two = builder.get_int32(2);
  auto *mul = builder.create_mul(builder.get_int32(8), builder.get_int32(2));
  auto *add = builder.create_add(mul, one);
  auto *sub = builder.create_sub(add, two);
  auto *truediv = builder.create_truediv(sub, two);
  auto *floordiv = builder.create_floordiv(truediv, two);
  auto *div = builder.create_div(floordiv, sub);
  auto *ceil = builder.create_ceil(div);
  auto *cast = builder.create_cast(ceil, PrimitiveType::i32);
  auto *mod = builder.create_mod(cast, two);
  auto *max = builder.create_max(mod, two);
  auto *min = builder.create_min(max, one);
  auto *bit_and = builder.create_and(min, two);
  auto *bit_or = builder.create_or(bit_and, two);
  auto *bit_shl = builder.create_shl(bit_or, one);
  auto *bit_shr = builder.create_shr(bit_shl, one);
  auto *bit_sar = builder.create_sar(bit_shr, two);
  auto *cmp_lt = builder.create_cmp_lt(bit_sar, two);
  auto *cmp_gt = builder.create_cmp_gt(cmp_lt, one);
  auto *cmp_ge = builder.create_cmp_ge(cmp_gt, two);
  auto *cmp_le = builder.create_cmp_le(cmp_ge, one);
  auto *cmp_eq = builder.create_cmp_eq(cmp_le, one);
  auto *cmp_ne = builder.create_cmp_ne(cmp_eq, one);
  auto *atan2 = builder.create_atan2(cmp_ne, one);
  auto *pow = builder.create_pow(atan2, two);
  auto *pow_cast = builder.create_cast(pow, PrimitiveType::i32);
  auto *logical_or = builder.create_logical_or(pow_cast, two);
  auto *logical_and = builder.create_logical_or(logical_or, one);

  auto *result = builder.create_sub(x, logical_and);
  builder.create_return(result);

  auto ir = builder.extract_ir();
  ASSERT_TRUE(ir->is<Block>());
  auto *ir_block = ir->as<Block>();
  irpass::type_check(ir_block, CompileConfig());
  EXPECT_EQ(ir_block->size(), 41);

  irpass::constant_fold(ir_block, CompileConfig(), {tp_.prog()});
  irpass::die(ir_block);

  EXPECT_EQ(ir_block->size(), 4);
  EXPECT_EQ(ir_block->statements[0].get(), x);
  EXPECT_TRUE(ir_block->statements[1]->is<ConstStmt>());
  EXPECT_EQ(ir_block->statements[1]->as<ConstStmt>()->val.val_float(), 2);
}

TEST_F(ConstantFoldTest, BinaryCmp) {
  IRBuilder builder;

  auto *x = builder.create_arg_load(0, get_data_type<float>(), false);
  auto *one = builder.get_int32(1);
  auto *two = builder.get_int32(2);
  auto *cmp_lt = builder.create_cmp_lt(one, two);

  auto *result = builder.create_sub(x, cmp_lt);
  builder.create_return(result);

  auto ir = builder.extract_ir();
  ASSERT_TRUE(ir->is<Block>());
  auto *ir_block = ir->as<Block>();
  irpass::type_check(ir_block, CompileConfig());
  EXPECT_EQ(ir_block->size(), 7);

  irpass::constant_fold(ir_block, CompileConfig(), {tp_.prog()});
  irpass::die(ir_block);

  EXPECT_EQ(ir_block->size(), 4);
  EXPECT_EQ(ir_block->statements[0].get(), x);
  EXPECT_TRUE(ir_block->statements[1]->is<ConstStmt>());
  EXPECT_EQ(ir_block->statements[1]->as<ConstStmt>()->val.val_float(), 1);
}
}  // namespace taichi::lang
