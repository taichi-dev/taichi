#include "gtest/gtest.h"

#include "taichi/ir/statements.h"
#include "taichi/ir/ir_builder.h"
#include "taichi/ir/transforms.h"

namespace taichi {
namespace lang {

class DetermineAdStackSizeTest : public ::testing::Test {
 protected:
  void SetUp() override {
    prog_ = std::make_unique<Program>();
    prog_->materialize_runtime();
  }

  std::unique_ptr<Program> prog_;
};

TEST_F(DetermineAdStackSizeTest, Basic) {
  IRBuilder builder;
  auto *stack =
      builder.create_ad_stack(get_data_type<int>(), 0 /*adaptive size*/);
  builder.ad_stack_push(stack, builder.get_int32(1));
  builder.ad_stack_push(stack, builder.get_int32(2));
  builder.ad_stack_push(stack, builder.get_int32(3));
  builder.ad_stack_pop(stack);
  builder.ad_stack_pop(stack);
  builder.ad_stack_push(stack, builder.get_int32(4));
  builder.ad_stack_push(stack, builder.get_int32(5));
  builder.ad_stack_push(stack, builder.get_int32(6));
  // stack contains [1, 4, 5, 6] now
  builder.ad_stack_pop(stack);
  builder.ad_stack_pop(stack);
  builder.ad_stack_push(stack, builder.get_int32(7));

  auto *stack2 =
      builder.create_ad_stack(get_data_type<int>(), 0 /*adaptive size*/);
  builder.ad_stack_push(stack2, builder.get_int32(8));

  auto ir = builder.extract_ir();
  ASSERT_TRUE(ir->is<Block>());
  auto *ir_block = ir->as<Block>();
  irpass::type_check(ir_block, CompileConfig());
  EXPECT_EQ(ir_block->size(), 22);

  EXPECT_EQ(stack->max_size, 0);
  EXPECT_EQ(stack2->max_size, 0);
  irpass::determine_ad_stack_size(ir_block, CompileConfig());
  EXPECT_EQ(stack->max_size, 4);
  EXPECT_EQ(stack2->max_size, 1);
}

}  // namespace lang
}  // namespace taichi
