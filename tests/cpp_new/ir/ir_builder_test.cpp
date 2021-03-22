#include "gtest/gtest.h"

#include "taichi/ir/ir_builder.h"
#include "taichi/ir/statements.h"

namespace taichi {
namespace lang {

TEST(IRBuilder, Basic) {
  IRBuilder builder;
  auto *lhs = builder.get_int32(40);
  auto *rhs = builder.get_int32(2);
  auto *add = builder.create_add(lhs, rhs);
  ASSERT_TRUE(add->is<BinaryOpStmt>());
  auto *addc = add->cast<BinaryOpStmt>();
  EXPECT_EQ(addc->lhs, lhs);
  EXPECT_EQ(addc->rhs, rhs);
  EXPECT_EQ(addc->op_type, BinaryOpType::add);
}

TEST(IRBuilder, Print) {
  IRBuilder builder;
  auto *one = builder.get_int32(1);
  ASSERT_TRUE(one->is<ConstStmt>());
  std::string message = "message";
  auto *result = builder.create_print(one, message, one);
  ASSERT_TRUE(result->is<PrintStmt>());
  auto *print = result->cast<PrintStmt>();
  EXPECT_EQ(print->contents.size(), 3);
  ASSERT_TRUE(std::holds_alternative<Stmt *>(print->contents[0]));
  EXPECT_EQ(std::get<Stmt *>(print->contents[0]), one);
  ASSERT_TRUE(std::holds_alternative<std::string>(print->contents[1]));
  EXPECT_EQ(std::get<std::string>(print->contents[1]), message);
  ASSERT_TRUE(std::holds_alternative<Stmt *>(print->contents[2]));
  EXPECT_EQ(std::get<Stmt *>(print->contents[2]), one);
}

TEST(IRBuilder, RangeFor) {
  IRBuilder builder;
  auto *zero = builder.get_int32(0);
  auto *ten = builder.get_int32(10);
  auto *loop = builder.create_range_for(zero, ten);
  builder.set_insertion_point_to_loop_begin(loop);
  auto *index = builder.get_loop_index(loop, 0);
  builder.set_insertion_point_to_after(loop);
  auto *ret = builder.create_return(zero);
  EXPECT_EQ(zero->parent->size(), 4);
  ASSERT_TRUE(loop->is<RangeForStmt>());
  auto *loopc = loop->cast<RangeForStmt>();
  EXPECT_EQ(loopc->body->size(), 1);
  EXPECT_EQ(loopc->body->statements[0].get(), index);
}
}  // namespace lang
}  // namespace taichi
