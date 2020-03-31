#include <taichi/ir/frontend.h>
#include <taichi/common/testing.h>

TLANG_NAMESPACE_BEGIN

TI_TEST("same_statements") {
  auto block = std::make_unique<Block>();

  auto global_load_addr =
      block->push_back<GlobalTemporaryStmt>(0, VectorType(1, DataType::i32));
  auto global_load = block->push_back<GlobalLoadStmt>(global_load_addr);
  auto global_store_addr =
      block->push_back<GlobalTemporaryStmt>(4, VectorType(1, DataType::i32));
  auto one = block->push_back<ConstStmt>(TypedConstant(1));
  auto if_stmt = block->push_back<IfStmt>(one)->as<IfStmt>();

  auto true_clause = std::make_unique<Block>();
  auto true_one = true_clause->push_back<ConstStmt>(TypedConstant(1));
  auto true_add = true_clause->push_back<BinaryOpStmt>(
      BinaryOpType::add, global_load, true_one);
  auto true_store =
      true_clause->push_back<GlobalStoreStmt>(global_store_addr, true_add);
  if_stmt->true_statements = std::move(true_clause);

  auto false_clause = std::make_unique<Block>();
  auto false_one = false_clause->push_back<ConstStmt>(TypedConstant(1));
  auto false_add = false_clause->push_back<BinaryOpStmt>(
      BinaryOpType::add, global_load, false_one);
  auto false_store =
      false_clause->push_back<GlobalStoreStmt>(global_store_addr, false_add);
  if_stmt->false_statements = std::move(false_clause);

  irpass::typecheck(block.get());
  TI_CHECK(block->size() == 5);

  TI_CHECK(irpass::same_statements(true_one, false_one));

  TI_CHECK(irpass::same_statements(true_one, one));

  TI_CHECK(!irpass::same_statements(global_load_addr, global_store_addr));

  TI_CHECK(irpass::same_statements(
      if_stmt->true_statements.get(), if_stmt->false_statements.get()));

  TI_CHECK(!irpass::same_statements(true_store, false_store));
}

TLANG_NAMESPACE_END
