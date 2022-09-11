#include "gtest/gtest.h"

#include "taichi/ir/ir.h"
#include "taichi/ir/statements.h"

namespace taichi {
namespace lang {
namespace {

std::unique_ptr<ConstStmt> make_const_i32(int32_t value) {
  return Stmt::make_typed<ConstStmt>(TypedConstant(
      TypeFactory::get_instance().get_primitive_type(PrimitiveTypeID::i32),
      value));
}

TEST(Block, Erase) {
  Block b;
  b.insert(make_const_i32(1));
  b.insert(make_const_i32(2));
  auto s3 = make_const_i32(3);
  auto s3_ptr = s3.get();
  b.insert(std::move(s3));
  b.erase(/*location=*/1);
  EXPECT_EQ(b.size(), 2);
  EXPECT_EQ(b.locate(s3_ptr), 1);
}

TEST(Block, EraseRange) {
  Block b;
  std::vector<ConstStmt *> stmt_ptrs;
  for (int i = 0; i < 5; ++i) {
    auto s = make_const_i32(i);
    stmt_ptrs.push_back(s.get());
    b.insert(std::move(s));
  }
  EXPECT_EQ(b.size(), 5);
  auto begin = b.find(stmt_ptrs[1]);
  auto end = b.find(stmt_ptrs[4]);
  b.erase_range(begin, end);
  EXPECT_EQ(b.size(), 2);
  EXPECT_EQ(b.locate(stmt_ptrs.front()), 0);
  EXPECT_EQ(b.locate(stmt_ptrs.back()), 1);
}

}  // namespace
}  // namespace lang
}  // namespace taichi
