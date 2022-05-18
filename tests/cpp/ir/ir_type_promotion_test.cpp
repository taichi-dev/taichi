#include "gtest/gtest.h"

#include "taichi/ir/statements.h"
#include "taichi/ir/ir_builder.h"
#include "taichi/ir/transforms.h"
#include "tests/cpp/program/test_program.h"

namespace taichi {
namespace lang {

TEST(IRTypePromotionTest, ShiftOp) {
  IRBuilder builder;

  // (u8)x << (i32)1 -> (u8)res
  auto *lhs = builder.create_arg_load(0, get_data_type<uint8>(), false);
  builder.create_shl(lhs, builder.get_int32(1));
  auto ir = builder.extract_ir();

  ASSERT_TRUE(ir->is<Block>());
  auto *ir_block = ir->as<Block>();
  irpass::type_check(ir_block, CompileConfig());

  EXPECT_TRUE(ir_block->statements.back()->is<BinaryOpStmt>());
  auto *binary_stmt = ir_block->statements.back()->as<BinaryOpStmt>();

  auto ret_type = binary_stmt->ret_type;
  EXPECT_TRUE(ret_type->is_primitive(PrimitiveTypeID::u8));
}

}  // namespace lang
}  // namespace taichi
