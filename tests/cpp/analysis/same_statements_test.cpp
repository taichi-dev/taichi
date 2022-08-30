#include "gtest/gtest.h"

#include "taichi/ir/transforms.h"
#include "taichi/ir/analysis.h"
#include "taichi/ir/statements.h"

namespace taichi {
namespace lang {

TEST(SameStatements, TestSameBlock) {
  auto block = std::make_unique<Block>();

  auto global_load_addr =
      block->push_back<GlobalTemporaryStmt>(0, PrimitiveType::i32);
  auto global_load = block->push_back<GlobalLoadStmt>(global_load_addr);
  auto global_store_addr =
      block->push_back<GlobalTemporaryStmt>(4, PrimitiveType::i32);
  auto one = block->push_back<ConstStmt>(TypedConstant(1));
  auto if_stmt = block->push_back<IfStmt>(one)->as<IfStmt>();

  auto true_clause = std::make_unique<Block>();
  auto true_one = true_clause->push_back<ConstStmt>(TypedConstant(1));
  auto true_add = true_clause->push_back<BinaryOpStmt>(BinaryOpType::add,
                                                       global_load, true_one);
  auto true_store =
      true_clause->push_back<GlobalStoreStmt>(global_store_addr, true_add);
  if_stmt->set_true_statements(std::move(true_clause));

  auto false_clause = std::make_unique<Block>();
  auto false_one = false_clause->push_back<ConstStmt>(TypedConstant(1));
  auto false_add = false_clause->push_back<BinaryOpStmt>(
      BinaryOpType::add, global_load, false_one);
  auto false_store =
      false_clause->push_back<GlobalStoreStmt>(global_store_addr, false_add);
  if_stmt->set_false_statements(std::move(false_clause));

  irpass::type_check(block.get(), CompileConfig());
  irpass::re_id(block.get());
  EXPECT_EQ(block->size(), 5);

  EXPECT_TRUE(irpass::analysis::same_statements(true_one, false_one));

  EXPECT_TRUE(irpass::analysis::same_statements(true_one, one));

  EXPECT_TRUE(
      !irpass::analysis::same_statements(global_load_addr, global_store_addr));

  EXPECT_TRUE(irpass::analysis::same_statements(
      if_stmt->true_statements.get(), if_stmt->false_statements.get()));

  EXPECT_TRUE(!irpass::analysis::same_statements(true_store, false_store));
}

TEST(SameStatements, TestSameAssert) {
  auto block = std::make_unique<Block>();
  auto zero = block->push_back<ConstStmt>(TypedConstant(0));
  auto one = block->push_back<ConstStmt>(TypedConstant(1));
  auto assert_zero_a =
      block->push_back<AssertStmt>(zero, "a", std::vector<Stmt *>());
  auto assert_zero_a2 =
      block->push_back<AssertStmt>(zero, "a", std::vector<Stmt *>());
  auto assert_zero_b =
      block->push_back<AssertStmt>(zero, "b", std::vector<Stmt *>());
  auto assert_one_a =
      block->push_back<AssertStmt>(one, "a", std::vector<Stmt *>());
  auto assert_zero_a_one =
      block->push_back<AssertStmt>(zero, "a", std::vector<Stmt *>(1, one));
  auto assert_zero_a_one2 =
      block->push_back<AssertStmt>(zero, "a", std::vector<Stmt *>(1, one));
  auto assert_zero_a_zero =
      block->push_back<AssertStmt>(zero, "a", std::vector<Stmt *>(1, zero));
  auto assert_one_a_zero =
      block->push_back<AssertStmt>(one, "a", std::vector<Stmt *>(1, zero));

  irpass::type_check(block.get(), CompileConfig());
  irpass::re_id(block.get());
  EXPECT_EQ(block->size(), 10);
  EXPECT_TRUE(irpass::analysis::same_statements(assert_zero_a, assert_zero_a2));
  EXPECT_TRUE(!irpass::analysis::same_statements(assert_zero_a, assert_zero_b));
  EXPECT_TRUE(!irpass::analysis::same_statements(assert_zero_a, assert_one_a));
  EXPECT_TRUE(
      !irpass::analysis::same_statements(assert_zero_a, assert_zero_a_one));
  EXPECT_TRUE(
      irpass::analysis::same_statements(assert_zero_a_one, assert_zero_a_one2));
  EXPECT_TRUE(!irpass::analysis::same_statements(assert_zero_a_one,
                                                 assert_zero_a_zero));
  EXPECT_TRUE(
      !irpass::analysis::same_statements(assert_zero_a_one, assert_one_a_zero));
}

TEST(SameStatements, TestSameSnodeLookup) {
  auto block = std::make_unique<Block>();

  auto get_root = block->push_back<GetRootStmt>();
  auto zero = block->push_back<ConstStmt>(TypedConstant(0));
  SNode root(0, SNodeType::root);
  auto &child = root.insert_children(SNodeType::dense);
  auto lookup1 =
      block->push_back<SNodeLookupStmt>(&root, get_root, zero, false);
  auto lookup2 =
      block->push_back<SNodeLookupStmt>(&root, get_root, zero, false);
  auto lookup_activate =
      block->push_back<SNodeLookupStmt>(&root, get_root, zero, true);
  auto get_child = block->push_back<GetChStmt>(lookup_activate, 0);
  auto lookup_child =
      block->push_back<SNodeLookupStmt>(&child, get_child, zero, false);

  irpass::type_check(block.get(), CompileConfig());
  irpass::re_id(block.get());
  EXPECT_EQ(block->size(), 7);
  EXPECT_TRUE(irpass::analysis::same_statements(lookup1, lookup2));
  EXPECT_TRUE(!irpass::analysis::same_statements(lookup1, lookup_activate));
  EXPECT_TRUE(!irpass::analysis::same_statements(lookup1, lookup_child));
}

TEST(SameStatements, TestSameValue) {
  auto block = std::make_unique<Block>();
  auto one = block->push_back<ConstStmt>(TypedConstant(1));
  auto if_stmt = block->push_back<IfStmt>(one)->as<IfStmt>();

  auto true_clause = std::make_unique<Block>();
  auto true_rand = true_clause->push_back<RandStmt>(PrimitiveType::i32);
  auto true_add =
      true_clause->push_back<BinaryOpStmt>(BinaryOpType::add, one, true_rand);
  if_stmt->set_true_statements(std::move(true_clause));

  auto false_clause = std::make_unique<Block>();
  auto false_rand = false_clause->push_back<RandStmt>(PrimitiveType::i32);
  auto false_add =
      false_clause->push_back<BinaryOpStmt>(BinaryOpType::add, one, false_rand);
  if_stmt->set_false_statements(std::move(false_clause));

  irpass::type_check(block.get(), CompileConfig());
  irpass::re_id(block.get());
  EXPECT_EQ(irpass::analysis::count_statements(block.get()), 6);

  EXPECT_TRUE(irpass::analysis::same_statements(true_rand, false_rand));
  EXPECT_TRUE(!irpass::analysis::same_value(true_rand, false_rand));
  EXPECT_TRUE(irpass::analysis::same_statements(
      if_stmt->true_statements.get(), if_stmt->false_statements.get()));

  // They should be considered different if we don't check recursively.
  EXPECT_TRUE(
      !irpass::analysis::same_statements(true_add, false_add, std::nullopt));
  EXPECT_TRUE(irpass::analysis::same_statements(
      true_add, false_add, std::make_optional<std::unordered_map<int, int>>()));

  EXPECT_TRUE(!irpass::analysis::same_value(
      true_add, false_add, std::make_optional<std::unordered_map<int, int>>()));
}

TEST(SameStatements, TestSameLoopIndex) {
  auto block = std::make_unique<Block>();
  auto zero = block->push_back<ConstStmt>(TypedConstant(0));
  auto four = block->push_back<ConstStmt>(TypedConstant(4));
  auto range_for =
      block
          ->push_back<RangeForStmt>(zero, four, std::make_unique<Block>(), 1, 1,
                                    1, false)
          ->as<RangeForStmt>();
  auto loop_index_a = range_for->body->push_back<LoopIndexStmt>(range_for, 0);
  auto loop_index_b = range_for->body->push_back<LoopIndexStmt>(range_for, 0);

  irpass::type_check(block.get(), CompileConfig());
  irpass::re_id(block.get());

  EXPECT_TRUE(irpass::analysis::same_statements(loop_index_a, loop_index_b));
  EXPECT_TRUE(irpass::analysis::same_value(loop_index_a, loop_index_b));
}

}  // namespace lang
}  // namespace taichi
