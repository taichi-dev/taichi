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
  builder.create_add(x_v, builder.get_int32(1));

  auto ir = builder.extract_ir();
  ASSERT_TRUE(ir->is<Block>());
  auto *ir_block = ir->as<Block>();
  irpass::type_check(ir_block, CompileConfig());
  // Before:
  //   kernel {
  //     <i32> $0 = const [10]
  //     <i32> $1 = const [0]
  //     $2 : for in range($1, $0) (vectorize -1) (bit_vectorize -1)
  //     block_dim=adaptive {
  //       <i32> $3 = alloca
  //       $4 = local load [ [$3[0]]]
  //       <i32> $5 = const [1]
  //       $6 = add $4 $5
  //     }
  //   }
  EXPECT_EQ(ir_block->size(), 3);
  irpass::extract_constant(ir_block, CompileConfig());
  // After:
  //   kernel {
  //     <i32> $0 = const [1]
  //     <i32> $1 = const [10]
  //     <i32> $2 = const [0]
  //     $3 : for in range($2, $1) (vectorize -1) (bit_vectorize -1)
  //     block_dim=adaptive {
  //       <i32> $4 = alloca
  //       <i32> $5 = local load [ [$4[0]]]
  //       <i32> $6 = add $5 $0
  //     }
  //   }
  EXPECT_EQ(ir_block->size(), 4);
}

}  // namespace lang
}  // namespace taichi
