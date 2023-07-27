#include "gtest/gtest.h"

#include "taichi/ir/statements.h"
#include "taichi/ir/ir_builder.h"
#include "taichi/ir/transforms.h"
#include "tests/cpp/program/test_program.h"

namespace taichi::lang {

TEST(IRTypePromotionTest, ShiftOp) {
  IRBuilder builder;

  // (u8)x << (i32)1 -> (u8)res
  auto *lhs = builder.create_arg_load({0}, get_data_type<uint8>(), false, 0);
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

TEST(IRPromotionTest, TensorType) {
  IRBuilder builder;

  auto *lhs_element = builder.get_int32(1);
  auto *lhs_mat = builder.create_matrix_init({lhs_element});
  lhs_mat->ret_type =
      TypeFactory::create_tensor_type({1, 1}, PrimitiveType::i32);
  auto *rhs_element = builder.get_float32(1);
  auto *rhs_mat = builder.create_matrix_init({rhs_element});
  rhs_mat->ret_type =
      TypeFactory::create_tensor_type({1, 1}, PrimitiveType::f32);
  builder.create_add(lhs_mat, rhs_mat);
  auto ir = builder.extract_ir();
  auto config = CompileConfig();
  auto *block = ir->as<Block>();
  irpass::type_check(block, config);

  EXPECT_TRUE(block->statements.back()->is<BinaryOpStmt>());
  auto stmt = block->statements.back()->as<BinaryOpStmt>();
  auto rhs_type = stmt->rhs->ret_type;
  auto ret_type = stmt->ret_type;

  EXPECT_TRUE(rhs_type->is<TensorType>() &&
              rhs_type->cast<TensorType>()->get_element_type()->is_primitive(
                  PrimitiveTypeID::f32));
  EXPECT_TRUE(ret_type->is<TensorType>() &&
              ret_type->cast<TensorType>()->get_element_type()->is_primitive(
                  PrimitiveTypeID::f32));
}
}  // namespace taichi::lang
