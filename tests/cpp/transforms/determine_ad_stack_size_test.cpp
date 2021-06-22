#include "gtest/gtest.h"

#include "taichi/ir/analysis.h"
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

TEST_F(DetermineAdStackSizeTest, Loop) {
  IRBuilder builder;
  auto *stack =
      builder.create_ad_stack(get_data_type<int>(), 0 /*adaptive size*/);
  auto *loop = builder.create_range_for(/*begin=*/builder.get_int32(0),
                                        /*end=*/builder.get_int32(10));
  {
    auto _ = builder.get_loop_guard(loop);
    builder.ad_stack_push(stack, builder.get_int32(1));
    builder.ad_stack_pop(stack);
  }

  auto ir = builder.extract_ir();
  ASSERT_TRUE(ir->is<Block>());
  auto *ir_block = ir->as<Block>();
  irpass::type_check(ir_block, CompileConfig());
  EXPECT_EQ(irpass::analysis::count_statements(ir_block), 7);

  EXPECT_EQ(stack->max_size, 0);
  irpass::determine_ad_stack_size(ir_block, CompileConfig());
  EXPECT_EQ(stack->max_size, 1);
}

TEST_F(DetermineAdStackSizeTest, LoopInfeasible) {
  IRBuilder builder;
  auto *stack =
      builder.create_ad_stack(get_data_type<int>(), 0 /*adaptive size*/);
  auto *loop = builder.create_range_for(/*begin=*/builder.get_int32(0),
                                        /*end=*/builder.get_int32(100));
  {
    auto _ = builder.get_loop_guard(loop);
    builder.ad_stack_push(stack, builder.get_int32(1));
  }

  auto ir = builder.extract_ir();
  ASSERT_TRUE(ir->is<Block>());
  auto *ir_block = ir->as<Block>();
  irpass::type_check(ir_block, CompileConfig());
  EXPECT_EQ(irpass::analysis::count_statements(ir_block), 6);

  CompileConfig config;
  constexpr int kMaxAdStackSize = 32;
  config.max_ad_stack_size = kMaxAdStackSize;
  EXPECT_EQ(stack->max_size, 0);
  // Should have a warning here (unable to determine capacity for autodiff
  // stacks).
  irpass::determine_ad_stack_size(ir_block, CompileConfig());
  EXPECT_EQ(stack->max_size, kMaxAdStackSize);
}

TEST_F(DetermineAdStackSizeTest, If) {
  constexpr int kCommonPushes = 1;
  constexpr int kTrueBranchPushes = 3;
  constexpr int kFalseBranchPushes = 4;

  // parameterize
  for (int has_true_branch = 0; has_true_branch <= 1; has_true_branch++) {
    for (int has_false_branch = 0; has_false_branch <= 1; has_false_branch++) {
      IRBuilder builder;
      auto *arg = builder.create_arg_load(0, get_data_type<int>(), false);
      auto *stack =
          builder.create_ad_stack(get_data_type<int>(), 0 /*adaptive size*/);
      auto *if_stmt = builder.create_if(arg);
      auto *one = builder.get_int32(1);
      for (int i = 1; i <= kCommonPushes; i++) {
        builder.ad_stack_push(stack, one);  // Make sure the stack is not unused
      }
      if (has_true_branch) {
        auto _ = builder.get_if_guard(if_stmt, true);
        for (int i = 1; i <= kTrueBranchPushes; i++) {
          builder.ad_stack_push(stack, one);
        }
      }
      if (has_false_branch) {
        auto _ = builder.get_if_guard(if_stmt, false);
        for (int i = 1; i <= kFalseBranchPushes; i++) {
          builder.ad_stack_push(stack, one);
        }
      }

      auto ir = builder.extract_ir();
      ASSERT_TRUE(ir->is<Block>());
      auto *ir_block = ir->as<Block>();
      irpass::type_check(ir_block, CompileConfig());
      EXPECT_EQ(irpass::analysis::count_statements(ir_block),
                4 + kCommonPushes + has_true_branch * kTrueBranchPushes +
                    has_false_branch * kFalseBranchPushes);

      EXPECT_EQ(stack->max_size, 0);
      irpass::determine_ad_stack_size(ir_block, CompileConfig());
      EXPECT_EQ(
          stack->max_size,
          kCommonPushes + std::max(has_true_branch * kTrueBranchPushes,
                                   has_false_branch * kFalseBranchPushes));
    }
  }
}

}  // namespace lang
}  // namespace taichi
