#include "gtest/gtest.h"

#include "taichi/ir/analysis.h"
#include "taichi/ir/ir_builder.h"
#include "taichi/ir/statements.h"

namespace taichi {
namespace lang {
namespace irpass {
namespace analysis {

TEST(ValueDiffPtrIndex, ConstI32) {
  IRBuilder builder;

  auto *const1 = builder.get_int32(42);
  auto *const2 = builder.get_int32(2);

  const auto diff = value_diff_ptr_index(const1, const2);
  EXPECT_TRUE(diff.is_diff_certain);
  EXPECT_EQ(diff.diff_range, 40);
}

TEST(ValueDiffPtrIndex, ConstF32) {
  IRBuilder builder;

  auto *const1 = builder.get_float32(1.0f);
  auto *const2 = builder.get_float32(1.0f);

  const auto diff = value_diff_ptr_index(const1, const2);
  // We don't check floating-point numbers since the pass is for pointer indices
  EXPECT_FALSE(diff.is_diff_certain);
}

TEST(ValueDiffPtrIndex, BinOp) {
  IRBuilder builder;

  auto *alloca = builder.create_local_var(PrimitiveType::i32);
  auto *load = builder.create_local_load(alloca);
  auto *const1 = builder.get_int32(1);
  auto *bin1 = builder.create_add(load, const1);
  auto *bin2 = builder.create_sub(load, const1);

  const auto diff = value_diff_ptr_index(bin1, bin2);
  EXPECT_TRUE(diff.is_diff_certain);
  EXPECT_EQ(diff.diff_range, 2);
}

TEST(ValueDiffPtrIndex, BinOpButDiff) {
  IRBuilder builder;

  auto *val1_1 = builder.get_int32(1);
  auto *val1_2 = builder.get_int32(1);
  auto *val2 = builder.get_int32(3);
  auto *bin1 = builder.create_add(val1_1, val2);
  auto *bin2 = builder.create_add(val1_2, val2);

  const auto diff = value_diff_ptr_index(bin1, bin2);
  EXPECT_FALSE(diff.is_diff_certain);
}

TEST(DiffRangeTest, Add) {
  IRBuilder builder;

  auto for_stmt = std::make_unique<OffloadedStmt>(
      /*task_type=*/OffloadedTaskType::struct_for,
      /*arch=*/Arch::x64);
  auto *loop_idx = builder.get_loop_index(for_stmt.get(), /*index=*/0);
  auto *c1 = builder.get_int32(4);
  auto *loop_idx2 = builder.create_add(loop_idx, c1);

  auto diff = value_diff_loop_index(loop_idx2, for_stmt.get(), /*index=*/0);

  EXPECT_TRUE(diff.related());
  EXPECT_EQ(diff.coeff, 1);
  EXPECT_EQ(diff.low, 4);
  EXPECT_EQ(diff.high, 5);
}

TEST(DiffRangeTest, Sub) {
  IRBuilder builder;

  auto for_stmt = std::make_unique<OffloadedStmt>(
      /*task_type=*/OffloadedTaskType::struct_for,
      /*arch=*/Arch::x64);
  auto *loop_idx = builder.get_loop_index(for_stmt.get(), /*index=*/0);
  auto *c1 = builder.get_int32(4);
  auto *loop_idx2 = builder.create_sub(loop_idx, c1);

  auto diff = value_diff_loop_index(loop_idx2, for_stmt.get(), /*index=*/0);

  EXPECT_TRUE(diff.related());
  EXPECT_EQ(diff.coeff, 1);
  EXPECT_EQ(diff.low, -4);
  EXPECT_EQ(diff.high, -3);
}

TEST(DiffRangeTest, Mul) {
  IRBuilder builder;

  auto for_stmt = std::make_unique<OffloadedStmt>(
      /*task_type=*/OffloadedTaskType::struct_for,
      /*arch=*/Arch::x64);
  auto *loop_idx = builder.get_loop_index(for_stmt.get(), /*index=*/0);
  auto *c1 = builder.get_int32(4);
  auto *loop_idx2 = builder.create_mul(loop_idx, c1);

  auto diff = value_diff_loop_index(loop_idx2, for_stmt.get(), /*index=*/0);

  EXPECT_TRUE(diff.related());
  EXPECT_EQ(diff.coeff, 4);
  EXPECT_EQ(diff.low, 0);
  EXPECT_EQ(diff.high, 1);
}

TEST(DiffRangeTest, Shl) {
  IRBuilder builder;

  auto for_stmt = std::make_unique<OffloadedStmt>(
      /*task_type=*/OffloadedTaskType::struct_for,
      /*arch=*/Arch::x64);
  auto *loop_idx = builder.get_loop_index(for_stmt.get(), /*index=*/0);
  auto *c1 = builder.get_int32(2);
  auto *loop_idx2 = builder.create_shl(loop_idx, c1);

  auto diff = value_diff_loop_index(loop_idx2, for_stmt.get(), /*index=*/0);

  EXPECT_TRUE(diff.related());
  EXPECT_EQ(diff.coeff, 4);
  EXPECT_EQ(diff.low, 0);
  EXPECT_EQ(diff.high, 1);
}

}  // namespace analysis
}  // namespace irpass
}  // namespace lang
}  // namespace taichi
