#include <taichi/lang.h>
#include <taichi/testing.h>

TLANG_NAMESPACE_BEGIN

// Basic tests within a basic block

TC_TEST("simplify_add_zero") {
  return; // uncomment to enable
  auto block = std::make_unique<Block>();

  auto global_load_addr =
      block->push_back<GlobalTemporaryStmt>(0, VectorType(1, DataType::i32));
  auto global_load = block->push_back<GlobalLoadStmt>(global_load_addr);
  auto zero = block->push_back<ConstStmt>(TypedConstant(0));
  auto add =
      block->push_back<BinaryOpStmt>(BinaryOpType::add, global_load, zero);
  auto global_store_addr =
      block->push_back<GlobalTemporaryStmt>(4, VectorType(1, DataType::i32));
  auto global_store = block->push_back<GlobalStoreStmt>(global_store_addr, add);

  irpass::typecheck(block.get());
  TC_CHECK(block->size() == 6);  // two addresses, one load, one store

  irpass::print(block.get());

  irpass::alg_simp(block.get());  // should eliminate add (TODO)
  irpass::die(block.get());       // should eliminate zero

  irpass::print(block.get());
  TC_CHECK(block->size() == 4);  // two addresses, one load, one store
  TC_CHECK((*block)[0]->is<GlobalTemporaryStmt>());
  // .. more tests, assuming instruction order not shuffled
}

TLANG_NAMESPACE_END
