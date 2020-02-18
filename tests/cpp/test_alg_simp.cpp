#include <taichi/lang.h>
#include <taichi/testing.h>

TLANG_NAMESPACE_BEGIN

// Basic tests within a basic block

TI_TEST("simplify_add_zero") {
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
  TI_CHECK(block->size() == 6);  // two addresses, one load, one store

  // irpass::print(block.get());

  irpass::alg_simp(block.get());  // should eliminate add
  irpass::die(block.get());       // should eliminate zero

  // irpass::print(block.get());
  TI_CHECK(block->size() == 4);  // two addresses, one load, one store
  TI_CHECK((*block)[0]->is<GlobalTemporaryStmt>());
  // .. more tests, assuming instruction order not shuffled
}

TI_TEST("simplify_multiply_one") {
  auto block = std::make_unique<Block>();

  auto global_load_addr =
      block->push_back<GlobalTemporaryStmt>(0, VectorType(1, DataType::f32));
  auto global_load = block->push_back<GlobalLoadStmt>(global_load_addr);
  auto one = block->push_back<ConstStmt>(TypedConstant(1.0f));
  auto mul1 =
      block->push_back<BinaryOpStmt>(BinaryOpType::mul, one, global_load);
  auto mul2 = block->push_back<BinaryOpStmt>(BinaryOpType::mul, mul1, one);
  auto zero = block->push_back<ConstStmt>(TypedConstant(0.0f));
  auto div = block->push_back<BinaryOpStmt>(BinaryOpType::div, zero, one);
  auto sub = block->push_back<BinaryOpStmt>(BinaryOpType::sub, mul2, div);
  auto global_store_addr =
      block->push_back<GlobalTemporaryStmt>(4, VectorType(1, DataType::f32));
  auto global_store = block->push_back<GlobalStoreStmt>(global_store_addr, sub);

  irpass::typecheck(block.get());
  TI_CHECK(block->size() == 10);  // two addresses, one load, one store

  // irpass::print(block.get());

  irpass::alg_simp(block.get());  // should eliminate mul, div, sub
  irpass::die(block.get());       // should eliminate zero

  // irpass::print(block.get());

  TI_CHECK(block->size() == 4);  // two addresses, one load, one store
  TI_CHECK((*block)[0]->is<GlobalTemporaryStmt>());
}

TLANG_NAMESPACE_END
