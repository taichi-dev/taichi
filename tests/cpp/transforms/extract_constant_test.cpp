#include "gtest/gtest.h"

#include "taichi/ir/analysis.h"
#include "taichi/ir/ir_builder.h"
#include "taichi/ir/statements.h"
#include "taichi/ir/transforms.h"
#include "taichi/program/program.h"

namespace taichi {
namespace lang {

class ExtractConstantTest : public ::testing::Test {
 protected:
  void SetUp() override {
    prog_ = std::make_unique<Program>();
    prog_->materialize_runtime();
  }

  std::unique_ptr<Program> prog_;
};

TEST_F(ExtractConstantTest, ExtractConstant) {
  IRBuilder builder;
  auto *for_stmt =
      builder.create_range_for(builder.get_int32(0), builder.get_int32(10));
  builder.set_insertion_point_to_loop_begin(for_stmt);
  auto *x = builder.create_local_var(get_data_type<int>());
  auto *x_v = builder.create_local_load(x);
  auto *sum = builder.create_add(x_v, builder.get_int32(1));

  auto ir = builder.extract_ir();
  ASSERT_TRUE(ir->is<Block>());
  auto *ir_block = ir->as<Block>();
  irpass::type_check(ir_block, CompileConfig());
  EXPECT_EQ(ir_block->size(), 3);
  irpass::extract_constant(ir_block, CompileConfig());
  EXPECT_EQ(ir_block->size(), 4);
}

}  // namespace lang
}  // namespace taichi
