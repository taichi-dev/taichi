#include "gtest/gtest.h"

#include "taichi/ir/statements.h"
#include "taichi/ir/ir_builder.h"
#include "taichi/ir/transforms.h"
#include "tests/cpp/program/test_program.h"

namespace taichi {
namespace lang {

class BinaryOpSimplifyTest : public ::testing::Test {
 protected:
  void SetUp() override {
    tp_.setup();
  }

  TestProgram tp_;
};

TEST_F(BinaryOpSimplifyTest, MultiplyPOT) {
  IRBuilder builder;
  // (x * 32) << 3
  auto *x = builder.create_arg_load(0, get_data_type<int>(), false);
  auto *product = builder.create_mul(x, builder.get_int32(32));
  auto *result = builder.create_shl(product, builder.get_int32(3));
  builder.create_return(result);
  auto ir = builder.extract_ir();
  ASSERT_TRUE(ir->is<Block>());
  auto *ir_block = ir->as<Block>();
  irpass::type_check(ir_block, CompileConfig());
  EXPECT_EQ(ir_block->size(), 6);

  irpass::alg_simp(ir_block, CompileConfig());
  // -> (x << 5) << 3
  irpass::binary_op_simplify(ir_block, CompileConfig());
  // -> x << (5 + 3)
  irpass::constant_fold(ir_block, CompileConfig(), {tp_.prog()});
  // -> x << 8
  irpass::die(ir_block);

  EXPECT_EQ(ir_block->size(), 4);
  EXPECT_EQ(ir_block->statements[0].get(), x);
  EXPECT_TRUE(ir_block->statements[1]->is<ConstStmt>());
  auto *const_stmt = ir_block->statements[1]->as<ConstStmt>();
  EXPECT_TRUE(is_integral(const_stmt->val.dt));
  EXPECT_TRUE(is_signed(const_stmt->val.dt));
  EXPECT_EQ(const_stmt->val.val_int(), 8);
  EXPECT_TRUE(ir_block->statements[2]->is<BinaryOpStmt>());
  auto *bin_op = ir_block->statements[2]->as<BinaryOpStmt>();
  EXPECT_EQ(bin_op->op_type, BinaryOpType::bit_shl);
  EXPECT_EQ(bin_op->rhs, const_stmt);
  EXPECT_TRUE(ir_block->statements[3]->is<ReturnStmt>());
  EXPECT_EQ(ir_block->statements[3]->as<ReturnStmt>()->values[0], bin_op);
}

TEST_F(BinaryOpSimplifyTest, ModPOT) {
  IRBuilder builder;
  // x % 8 in the Python frontend is transformed into:
  // x - x / 8 * 8
  auto *x = builder.create_arg_load(0, get_data_type<uint32>(), false);
  auto *division = builder.create_div(x, builder.get_uint32(8));
  auto *product = builder.create_mul(division, builder.get_uint32(8));
  auto *result = builder.create_sub(x, product);
  builder.create_return(result);
  auto ir = builder.extract_ir();
  ASSERT_TRUE(ir->is<Block>());
  auto *ir_block = ir->as<Block>();
  irpass::type_check(ir_block, CompileConfig());
  EXPECT_EQ(ir_block->size(), 7);

  // Eliminate the redundant constant 8
  irpass::whole_kernel_cse(ir_block);
  EXPECT_EQ(ir_block->size(), 6);

  // -> x - (x >> 3 << 3)
  irpass::alg_simp(ir_block, CompileConfig());
  irpass::whole_kernel_cse(ir_block);
  irpass::die(ir_block);
  EXPECT_EQ(ir_block->size(), 6);

  // -> x & (~(~7))
  irpass::binary_op_simplify(ir_block, CompileConfig());
  irpass::die(ir_block);
  EXPECT_EQ(ir_block->size(), 5);

  // -> x & 7
  irpass::constant_fold(ir_block, CompileConfig(), {tp_.prog()});
  irpass::die(ir_block);
  EXPECT_EQ(ir_block->size(), 4);
  EXPECT_EQ(ir_block->statements[0].get(), x);
  EXPECT_TRUE(ir_block->statements[1]->is<ConstStmt>());
  auto *const_stmt = ir_block->statements[1]->as<ConstStmt>();
  EXPECT_TRUE(is_integral(const_stmt->val.dt));
  EXPECT_TRUE(is_unsigned(const_stmt->val.dt));
  EXPECT_EQ(const_stmt->val.val_uint(), 7);
  EXPECT_TRUE(ir_block->statements[2]->is<BinaryOpStmt>());
  auto *bin_op = ir_block->statements[2]->as<BinaryOpStmt>();
  EXPECT_EQ(bin_op->op_type, BinaryOpType::bit_and);
  EXPECT_EQ(bin_op->rhs, const_stmt);
  EXPECT_TRUE(ir_block->statements[3]->is<ReturnStmt>());
  EXPECT_EQ(ir_block->statements[3]->as<ReturnStmt>()->values[0], bin_op);
}

}  // namespace lang
}  // namespace taichi
