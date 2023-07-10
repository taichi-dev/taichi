#include "gtest/gtest.h"

#include "taichi/ir/statements.h"
#include "taichi/ir/ir_builder.h"
#include "taichi/ir/transforms.h"
#include "tests/cpp/program/test_program.h"

namespace taichi::lang {

class ConstantFoldTest : public ::testing::Test {
 protected:
  void run_constant_fold() {
    ir = builder.extract_ir();
    ASSERT_TRUE(ir->is<Block>());
    auto *ir_block = ir->as<Block>();
    irpass::type_check(ir_block, CompileConfig());

    irpass::constant_fold(ir_block);
    irpass::die(ir_block);

    EXPECT_EQ(ir_block->size(), 4);
    EXPECT_TRUE(ir_block->statements[1]->is<ConstStmt>());
  }

  IRBuilder builder;
  std::unique_ptr<Block> ir;
};

TEST_F(ConstantFoldTest, UnaryNeg) {
  auto *x = builder.create_arg_load({0}, get_data_type<float>(), false, 0);
  auto *op = builder.get_int32(1);
  auto *out = builder.create_neg(op);
  auto *result = builder.create_sub(x, out);
  builder.create_return(result);

  run_constant_fold();

  EXPECT_EQ(ir->as<Block>()->statements[1]->as<ConstStmt>()->val.val_float(),
            -1);
}

TEST_F(ConstantFoldTest, UnarySqrt) {
  auto *x = builder.create_arg_load({0}, get_data_type<float>(), false, 0);
  auto *op = builder.get_int32(4);
  auto *out = builder.create_sqrt(op);
  auto *result = builder.create_sub(x, out);
  builder.create_return(result);

  run_constant_fold();

  EXPECT_EQ(ir->as<Block>()->statements[1]->as<ConstStmt>()->val.val_float(),
            2);
}

TEST_F(ConstantFoldTest, UnaryRound) {
  auto *x = builder.create_arg_load({0}, get_data_type<float>(), false, 0);
  auto *op = builder.get_float32(3.4);
  auto *out = builder.create_round(op);
  auto *result = builder.create_sub(x, out);
  builder.create_return(result);

  run_constant_fold();

  EXPECT_EQ(ir->as<Block>()->statements[1]->as<ConstStmt>()->val.val_float(),
            3);
}

TEST_F(ConstantFoldTest, UnaryFloor) {
  auto *x = builder.create_arg_load({0}, get_data_type<float>(), false, 0);
  auto *op = builder.get_float32(3.4);
  auto *out = builder.create_floor(op);
  auto *result = builder.create_sub(x, out);
  builder.create_return(result);

  run_constant_fold();

  EXPECT_EQ(ir->as<Block>()->statements[1]->as<ConstStmt>()->val.val_float(),
            3);
}

TEST_F(ConstantFoldTest, UnaryCeil) {
  auto *x = builder.create_arg_load({0}, get_data_type<float>(), false, 0);
  auto *op = builder.get_float32(3.4);
  auto *out = builder.create_ceil(op);
  auto *result = builder.create_sub(x, out);
  builder.create_return(result);

  run_constant_fold();

  EXPECT_EQ(ir->as<Block>()->statements[1]->as<ConstStmt>()->val.val_float(),
            4);
}

TEST_F(ConstantFoldTest, UnaryBitCast) {
  auto *x = builder.create_arg_load({0}, get_data_type<float>(), false, 0);
  auto *op = builder.get_int32(1);
  auto *out = builder.create_bit_cast(op, PrimitiveType::f32);
  auto *result = builder.create_sub(x, out);
  builder.create_return(result);

  run_constant_fold();

  EXPECT_NEAR(ir->as<Block>()->statements[1]->as<ConstStmt>()->val.val_float(),
              1.4013e-45, 1e-3);
}

TEST_F(ConstantFoldTest, UnaryAbs) {
  auto *x = builder.create_arg_load({0}, get_data_type<float>(), false, 0);
  auto *op = builder.get_float32(-3.4);
  auto *out = builder.create_abs(op);
  auto *result = builder.create_sub(x, out);
  builder.create_return(result);

  run_constant_fold();

  EXPECT_NEAR(ir->as<Block>()->statements[1]->as<ConstStmt>()->val.val_float(),
              3.4, 1e-4);
}

TEST_F(ConstantFoldTest, UnarySin) {
  auto *x = builder.create_arg_load({0}, get_data_type<float>(), false, 0);
  auto *op = builder.get_float32(1);
  auto *out = builder.create_sin(op);
  auto *result = builder.create_sub(x, out);
  builder.create_return(result);

  run_constant_fold();

  EXPECT_NEAR(ir->as<Block>()->statements[1]->as<ConstStmt>()->val.val_float(),
              0.84147, 1e-4);
}

TEST_F(ConstantFoldTest, UnaryAsin) {
  auto *x = builder.create_arg_load({0}, get_data_type<float>(), false, 0);
  auto *op = builder.get_float32(1);
  auto *out = builder.create_asin(op);
  auto *result = builder.create_sub(x, out);
  builder.create_return(result);

  run_constant_fold();

  EXPECT_NEAR(ir->as<Block>()->statements[1]->as<ConstStmt>()->val.val_float(),
              1.5708, 1e-4);
}

TEST_F(ConstantFoldTest, UnaryCos) {
  auto *x = builder.create_arg_load({0}, get_data_type<float>(), false, 0);
  auto *op = builder.get_float32(0.5);
  auto *out = builder.create_cos(op);
  auto *result = builder.create_sub(x, out);
  builder.create_return(result);

  run_constant_fold();

  EXPECT_NEAR(ir->as<Block>()->statements[1]->as<ConstStmt>()->val.val_float(),
              0.877582, 1e-4);
}

TEST_F(ConstantFoldTest, UnaryAcos) {
  auto *x = builder.create_arg_load({0}, get_data_type<float>(), false, 0);
  auto *op = builder.get_float32(0.5);
  auto *out = builder.create_acos(op);
  auto *result = builder.create_sub(x, out);
  builder.create_return(result);

  run_constant_fold();

  EXPECT_NEAR(ir->as<Block>()->statements[1]->as<ConstStmt>()->val.val_float(),
              1.0471976, 1e-4);
}

TEST_F(ConstantFoldTest, UnaryTan) {
  auto *x = builder.create_arg_load({0}, get_data_type<float>(), false, 0);
  auto *op = builder.get_float32(0.5);
  auto *out = builder.create_tan(op);
  auto *result = builder.create_sub(x, out);
  builder.create_return(result);

  run_constant_fold();

  EXPECT_NEAR(ir->as<Block>()->statements[1]->as<ConstStmt>()->val.val_float(),
              0.546302, 1e-4);
}

TEST_F(ConstantFoldTest, UnaryTanh) {
  auto *x = builder.create_arg_load({0}, get_data_type<float>(), false, 0);
  auto *op = builder.get_float32(0.5);
  auto *out = builder.create_tanh(op);
  auto *result = builder.create_sub(x, out);
  builder.create_return(result);

  run_constant_fold();

  EXPECT_NEAR(ir->as<Block>()->statements[1]->as<ConstStmt>()->val.val_float(),
              0.462117, 1e-4);
}

TEST_F(ConstantFoldTest, UnaryExp) {
  auto *x = builder.create_arg_load({0}, get_data_type<float>(), false, 0);
  auto *op = builder.get_float32(0.5);
  auto *out = builder.create_exp(op);
  auto *result = builder.create_sub(x, out);
  builder.create_return(result);

  run_constant_fold();

  EXPECT_NEAR(ir->as<Block>()->statements[1]->as<ConstStmt>()->val.val_float(),
              1.648721, 1e-4);
}

TEST_F(ConstantFoldTest, UnaryLog) {
  auto *x = builder.create_arg_load({0}, get_data_type<float>(), false, 0);
  auto *op = builder.get_float32(4);
  auto *out = builder.create_log(op);
  auto *result = builder.create_sub(x, out);
  builder.create_return(result);

  run_constant_fold();

  EXPECT_NEAR(ir->as<Block>()->statements[1]->as<ConstStmt>()->val.val_float(),
              1.38629, 1e-4);
}

TEST_F(ConstantFoldTest, UnaryBitNot) {
  auto *x = builder.create_arg_load({0}, get_data_type<float>(), false, 0);
  auto *op = builder.get_int32(1);
  auto *out = builder.create_not(op);
  auto *result = builder.create_sub(x, out);
  builder.create_return(result);

  run_constant_fold();

  EXPECT_EQ(ir->as<Block>()->statements[1]->as<ConstStmt>()->val.val_float(),
            -2);
}

TEST_F(ConstantFoldTest, UnaryLogicNot) {
  auto *x = builder.create_arg_load({0}, get_data_type<float>(), false, 0);
  auto *op = builder.get_int32(1);
  auto *out = builder.create_logical_not(op);
  auto *result = builder.create_sub(x, out);
  builder.create_return(result);

  run_constant_fold();

  EXPECT_EQ(ir->as<Block>()->statements[1]->as<ConstStmt>()->val.val_float(),
            0);
}

TEST_F(ConstantFoldTest, UnaryCastValue) {
  auto *x = builder.create_arg_load({0}, get_data_type<float>(), false, 0);
  auto *op = builder.get_int32(1);
  auto *out = builder.create_cast(op, PrimitiveType::f32);
  auto *result = builder.create_sub(x, out);
  builder.create_return(result);

  run_constant_fold();

  EXPECT_EQ(ir->as<Block>()->statements[1]->as<ConstStmt>()->val.val_float(),
            1);
}

TEST_F(ConstantFoldTest, UnaryRsqrt) {
  auto *x = builder.create_arg_load({0}, get_data_type<float>(), false, 0);
  auto *op = builder.get_float32(4);
  auto *out = builder.create_rsqrt(op);
  auto *result = builder.create_sub(x, out);
  builder.create_return(result);

  run_constant_fold();

  EXPECT_EQ(ir->as<Block>()->statements[1]->as<ConstStmt>()->val.val_float(),
            0.5);
}

TEST_F(ConstantFoldTest, BinaryMul) {
  auto *x = builder.create_arg_load({0}, get_data_type<float>(), false, 0);
  auto *lhs = builder.get_int32(1);
  auto *rhs = builder.get_int32(2);
  auto *out = builder.create_mul(lhs, rhs);
  auto *result = builder.create_sub(x, out);
  builder.create_return(result);

  run_constant_fold();

  EXPECT_EQ(ir->as<Block>()->statements[1]->as<ConstStmt>()->val.val_float(),
            2);
}

TEST_F(ConstantFoldTest, BinaryAdd) {
  auto *x = builder.create_arg_load({0}, get_data_type<float>(), false, 0);
  auto *one = builder.get_int32(1);
  auto *rhs = builder.get_int32(2);
  auto *out = builder.create_add(one, rhs);
  auto *result = builder.create_sub(x, out);
  builder.create_return(result);

  run_constant_fold();

  EXPECT_EQ(ir->as<Block>()->statements[1]->as<ConstStmt>()->val.val_float(),
            3);
}

TEST_F(ConstantFoldTest, BinarySub) {
  auto *x = builder.create_arg_load({0}, get_data_type<float>(), false, 0);
  auto *one = builder.get_int32(1);
  auto *rhs = builder.get_int32(2);
  auto *out = builder.create_sub(one, rhs);
  auto *result = builder.create_sub(x, out);
  builder.create_return(result);

  run_constant_fold();

  EXPECT_EQ(ir->as<Block>()->statements[1]->as<ConstStmt>()->val.val_float(),
            -1);
}

TEST_F(ConstantFoldTest, BinaryFloorDiv) {
  auto *x = builder.create_arg_load({0}, get_data_type<float>(), false, 0);
  auto *one = builder.get_float32(1);
  auto *rhs = builder.get_float32(2);
  auto *out = builder.create_floordiv(one, rhs);
  auto *result = builder.create_sub(x, out);
  builder.create_return(result);

  run_constant_fold();

  EXPECT_EQ(ir->as<Block>()->statements[1]->as<ConstStmt>()->val.val_float(),
            0);
}

TEST_F(ConstantFoldTest, BinaryDiv) {
  auto *x = builder.create_arg_load({0}, get_data_type<float>(), false, 0);
  auto *one = builder.get_float32(1);
  auto *rhs = builder.get_float32(2);
  auto *out = builder.create_div(one, rhs);
  auto *result = builder.create_sub(x, out);
  builder.create_return(result);

  run_constant_fold();

  EXPECT_EQ(ir->as<Block>()->statements[1]->as<ConstStmt>()->val.val_float(),
            0.5);
}

TEST_F(ConstantFoldTest, BinaryMod) {
  auto *x = builder.create_arg_load({0}, get_data_type<float>(), false, 0);
  auto *lhs = builder.get_int32(3);
  auto *rhs = builder.get_int32(2);
  auto *out = builder.create_mod(lhs, rhs);
  auto *result = builder.create_sub(x, out);
  builder.create_return(result);

  run_constant_fold();

  EXPECT_EQ(ir->as<Block>()->statements[1]->as<ConstStmt>()->val.val_float(),
            1);
}

TEST_F(ConstantFoldTest, BinaryMax) {
  auto *x = builder.create_arg_load({0}, get_data_type<float>(), false, 0);
  auto *lhs = builder.get_int32(3);
  auto *rhs = builder.get_int32(2);
  auto *out = builder.create_max(lhs, rhs);
  auto *result = builder.create_sub(x, out);
  builder.create_return(result);

  run_constant_fold();

  EXPECT_EQ(ir->as<Block>()->statements[1]->as<ConstStmt>()->val.val_float(),
            3);
}

TEST_F(ConstantFoldTest, BinaryMin) {
  auto *x = builder.create_arg_load({0}, get_data_type<float>(), false, 0);
  auto *lhs = builder.get_int32(3);
  auto *rhs = builder.get_int32(2);
  auto *out = builder.create_min(lhs, rhs);
  auto *result = builder.create_sub(x, out);
  builder.create_return(result);

  run_constant_fold();

  EXPECT_EQ(ir->as<Block>()->statements[1]->as<ConstStmt>()->val.val_float(),
            2);
}

TEST_F(ConstantFoldTest, BinaryBitAnd) {
  auto *x = builder.create_arg_load({0}, get_data_type<float>(), false, 0);
  auto *lhs = builder.get_int32(3);
  auto *rhs = builder.get_int32(2);
  auto *out = builder.create_and(lhs, rhs);
  auto *result = builder.create_sub(x, out);
  builder.create_return(result);

  run_constant_fold();

  EXPECT_EQ(ir->as<Block>()->statements[1]->as<ConstStmt>()->val.val_float(),
            2);
}

TEST_F(ConstantFoldTest, BinaryBitOr) {
  auto *x = builder.create_arg_load({0}, get_data_type<float>(), false, 0);
  auto *lhs = builder.get_int32(3);
  auto *rhs = builder.get_int32(2);
  auto *out = builder.create_or(lhs, rhs);
  auto *result = builder.create_sub(x, out);
  builder.create_return(result);

  run_constant_fold();

  EXPECT_EQ(ir->as<Block>()->statements[1]->as<ConstStmt>()->val.val_float(),
            3);
}

TEST_F(ConstantFoldTest, BinaryBitShl) {
  auto *x = builder.create_arg_load({0}, get_data_type<float>(), false, 0);
  auto *lhs = builder.get_int32(3);
  auto *rhs = builder.get_int32(2);
  auto *out = builder.create_shl(lhs, rhs);
  auto *result = builder.create_sub(x, out);
  builder.create_return(result);

  run_constant_fold();

  EXPECT_EQ(ir->as<Block>()->statements[1]->as<ConstStmt>()->val.val_float(),
            12);
}

TEST_F(ConstantFoldTest, BinaryBitShrInt32) {
  auto *x = builder.create_arg_load({0}, get_data_type<int>(), false, 0);
  auto *lhs = builder.get_int32(-1);
  auto *rhs = builder.get_uint32(2);
  auto *out = builder.create_shr(lhs, rhs);
  auto *result = builder.create_sub(x, out);
  builder.create_return(result);

  run_constant_fold();

  EXPECT_EQ(ir->as<Block>()->statements[1]->as<ConstStmt>()->val.val_int(),
            1073741823);
}

TEST_F(ConstantFoldTest, BinaryBitShrInt64) {
  auto *x = builder.create_arg_load({0}, get_data_type<int64_t>(), false, 0);
  auto *lhs = builder.get_int64(-1);
  auto *rhs = builder.get_uint32(2);
  auto *out = builder.create_shr(lhs, rhs);
  auto *result = builder.create_sub(x, out);
  builder.create_return(result);

  run_constant_fold();

  EXPECT_EQ(ir->as<Block>()->statements[1]->as<ConstStmt>()->val.val_int(),
            4611686018427387903);
}

TEST_F(ConstantFoldTest, BinaryBitSar) {
  auto *x = builder.create_arg_load({0}, get_data_type<int>(), false, 0);
  auto *lhs = builder.get_int32(-1);
  auto *rhs = builder.get_uint32(2);
  auto *out = builder.create_sar(lhs, rhs);
  auto *result = builder.create_sub(x, out);
  builder.create_return(result);

  run_constant_fold();

  EXPECT_EQ(ir->as<Block>()->statements[1]->as<ConstStmt>()->val.val_int(), -1);
}

TEST_F(ConstantFoldTest, BinaryCmpLt) {
  auto *x = builder.create_arg_load({0}, get_data_type<float>(), false, 0);
  auto *lhs = builder.get_int32(1);
  auto *rhs = builder.get_int32(2);
  auto *out = builder.create_cmp_lt(lhs, rhs);
  auto *result = builder.create_sub(x, out);
  builder.create_return(result);

  run_constant_fold();

  EXPECT_EQ(ir->as<Block>()->statements[1]->as<ConstStmt>()->val.val_float(),
            1);
}

TEST_F(ConstantFoldTest, BinaryCmpGt) {
  auto *x = builder.create_arg_load({0}, get_data_type<float>(), false, 0);
  auto *lhs = builder.get_int32(2);
  auto *rhs = builder.get_int32(2);
  auto *out = builder.create_cmp_gt(lhs, rhs);
  auto *result = builder.create_sub(x, out);
  builder.create_return(result);

  run_constant_fold();

  EXPECT_EQ(ir->as<Block>()->statements[1]->as<ConstStmt>()->val.val_float(),
            0);
}

TEST_F(ConstantFoldTest, BinaryCmpGe) {
  auto *x = builder.create_arg_load({0}, get_data_type<float>(), false, 0);
  auto *lhs = builder.get_int32(2);
  auto *rhs = builder.get_int32(2);
  auto *out = builder.create_cmp_ge(lhs, rhs);
  auto *result = builder.create_sub(x, out);
  builder.create_return(result);

  run_constant_fold();

  EXPECT_EQ(ir->as<Block>()->statements[1]->as<ConstStmt>()->val.val_float(),
            1);
}

TEST_F(ConstantFoldTest, BinaryCmpEq) {
  auto *x = builder.create_arg_load({0}, get_data_type<float>(), false, 0);
  auto *lhs = builder.get_int32(2);
  auto *rhs = builder.get_int32(2);
  auto *out = builder.create_cmp_eq(lhs, rhs);
  auto *result = builder.create_sub(x, out);
  builder.create_return(result);

  run_constant_fold();

  EXPECT_EQ(ir->as<Block>()->statements[1]->as<ConstStmt>()->val.val_float(),
            1);
}

TEST_F(ConstantFoldTest, BinaryCmpNes) {
  auto *x = builder.create_arg_load({0}, get_data_type<float>(), false, 0);
  auto *lhs = builder.get_int32(2);
  auto *rhs = builder.get_int32(2);
  auto *out = builder.create_cmp_ne(lhs, rhs);
  auto *result = builder.create_sub(x, out);
  builder.create_return(result);

  run_constant_fold();

  EXPECT_EQ(ir->as<Block>()->statements[1]->as<ConstStmt>()->val.val_float(),
            0);
}

TEST_F(ConstantFoldTest, BinaryPow) {
  auto *x = builder.create_arg_load({0}, get_data_type<float>(), false, 0);
  auto *lhs = builder.get_int32(2);
  auto *rhs = builder.get_int32(2);
  auto *out = builder.create_pow(lhs, rhs);
  auto *result = builder.create_sub(x, out);
  builder.create_return(result);

  run_constant_fold();

  EXPECT_EQ(ir->as<Block>()->statements[1]->as<ConstStmt>()->val.val_float(),
            4);
}

TEST_F(ConstantFoldTest, BinaryAtan2) {
  auto *x = builder.create_arg_load({0}, get_data_type<float>(), false, 0);
  auto *lhs = builder.get_int32(0);
  auto *rhs = builder.get_int32(2);
  auto *out = builder.create_atan2(lhs, rhs);
  auto *result = builder.create_sub(x, out);
  builder.create_return(result);

  run_constant_fold();

  EXPECT_EQ(ir->as<Block>()->statements[1]->as<ConstStmt>()->val.val_float(),
            0.);
}
}  // namespace taichi::lang
