#include "gtest/gtest.h"

#include "taichi/ir/statements.h"
#include "taichi/ir/ir_builder.h"
#include "taichi/ir/transforms.h"

namespace taichi {
namespace lang {

class BinaryOpSimplifyTest : public ::testing::Test {
 protected:
  void SetUp() override {
    prog_ = std::make_unique<Program>();
    prog_->materialize_layout();
  }

  std::unique_ptr<Program> prog_;
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
  irpass::constant_fold(ir_block, CompileConfig(), {prog_.get()});
  // -> x << 8
  irpass::die(ir_block);

  EXPECT_EQ(ir_block->size(), 4);
  EXPECT_EQ(ir_block->statements[0].get(), x);
  EXPECT_TRUE(ir_block->statements[1]->is<ConstStmt>());
  auto *const_stmt = ir_block->statements[1]->as<ConstStmt>();
  EXPECT_TRUE(is_integral(const_stmt->val[0].dt));
  EXPECT_TRUE(is_signed(const_stmt->val[0].dt));
  EXPECT_EQ(const_stmt->val[0].val_int(), 8);
  EXPECT_TRUE(ir_block->statements[2]->is<BinaryOpStmt>());
  auto *bin_op = ir_block->statements[2]->as<BinaryOpStmt>();
  EXPECT_EQ(bin_op->op_type, BinaryOpType::bit_shl);
  EXPECT_EQ(bin_op->rhs, const_stmt);
  EXPECT_TRUE(ir_block->statements[3]->is<ReturnStmt>());
  EXPECT_EQ(ir_block->statements[3]->as<ReturnStmt>()->value, bin_op);
}

}  // namespace lang
}  // namespace taichi
