#include <taichi/lang.h>
#include <taichi/testing.h>
#include <taichi/program.h>

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
  TI_CHECK(block->size() == 6);

  // irpass::print(block.get());

  irpass::alg_simp(block.get(), CompileConfig());  // should eliminate add
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
  TI_CHECK(block->size() == 10);

  // irpass::print(block.get());

  irpass::alg_simp(block.get(), CompileConfig());  // should eliminate mul, div, sub
  irpass::die(block.get());       // should eliminate zero, one

  // irpass::print(block.get());

  TI_CHECK(block->size() == 4);  // two addresses, one load, one store
  TI_CHECK((*block)[0]->is<GlobalTemporaryStmt>());
}

TI_TEST("simplify_multiply_zero_fast_math") {
  auto block = std::make_unique<Block>();

  auto global_load_addr =
      block->push_back<GlobalTemporaryStmt>(0, VectorType(1, DataType::i32));
  auto global_load = block->push_back<GlobalLoadStmt>(global_load_addr);
  auto zero = block->push_back<ConstStmt>(TypedConstant(0));
  auto mul =
      block->push_back<BinaryOpStmt>(BinaryOpType::mul, global_load, zero);
  auto one = block->push_back<ConstStmt>(TypedConstant(1));
  auto add = block->push_back<BinaryOpStmt>(BinaryOpType::add, mul, one);
  auto global_store_addr =
      block->push_back<GlobalTemporaryStmt>(4, VectorType(1, DataType::i32));
  auto global_store = block->push_back<GlobalStoreStmt>(global_store_addr, add);

  irpass::typecheck(block.get());
  TI_CHECK(block->size() == 8);

  CompileConfig config_without_fast_math;
  config_without_fast_math.fast_math = false;
  irpass::alg_simp(block.get(), config_without_fast_math);  // should eliminate mul, add
  irpass::die(block.get());  // should eliminate zero, load

  TI_CHECK(block->size() == 4);  // two addresses, one one, one store
  TI_CHECK((*block)[0]->is<GlobalTemporaryStmt>());


  block = std::make_unique<Block>();

  global_load_addr =
      block->push_back<GlobalTemporaryStmt>(8, VectorType(1, DataType::f32));
  global_load = block->push_back<GlobalLoadStmt>(global_load_addr);
  zero = block->push_back<ConstStmt>(TypedConstant(0));
  mul = block->push_back<BinaryOpStmt>(BinaryOpType::mul, global_load, zero);
  one = block->push_back<ConstStmt>(TypedConstant(1));
  add = block->push_back<BinaryOpStmt>(BinaryOpType::add, mul, one);
  global_store_addr =
      block->push_back<GlobalTemporaryStmt>(12, VectorType(1, DataType::f32));
  global_store = block->push_back<GlobalStoreStmt>(global_store_addr, add);

  irpass::typecheck(block.get());  // insert 2 casts
  TI_CHECK(block->size() == 10);

  irpass::constant_fold(block.get());  // should change 2 casts into const
  irpass::alg_simp(block.get(), config_without_fast_math);  // should not eliminate
  irpass::die(block.get());  // should eliminate 2 const
  TI_CHECK(block->size() == 8);

  CompileConfig config_with_fast_math;
  config_with_fast_math.fast_math = true;
  irpass::alg_simp(block.get(), config_with_fast_math);  // should eliminate mul, add
  irpass::die(block.get());  // should eliminate zero, load

  TI_CHECK(block->size() == 4);  // two addresses, one one, one store
  TI_CHECK((*block)[0]->is<GlobalTemporaryStmt>());
}

TI_TEST("simplify_linearized_with_trivial_inputs") {
  auto block = std::make_unique<Block>();

  auto get_root = block->push_back<GetRootStmt>();
  auto linearized_empty =
      block->push_back<LinearizeStmt>(std::vector<Stmt*>(), std::vector<int>());
  SNode root(0, SNodeType::root);
  root.insert_children(SNodeType::dense);
  auto lookup =
      block->push_back<SNodeLookupStmt>(&root, get_root, linearized_empty, false, std::vector<Stmt*>());
  auto get_child = block->push_back<GetChStmt>(lookup, 0);
  auto zero = block->push_back<ConstStmt>(TypedConstant(0));
  auto linearized_zero =
      block->push_back<LinearizeStmt>(std::vector<Stmt*>(2, zero), std::vector<int>({8, 4}));
  auto lookup2 =
      block->push_back<SNodeLookupStmt>(&*root.ch[0], get_child, linearized_zero, true, std::vector<Stmt*>());

  irpass::typecheck(block.get());
  // irpass::print(block.get());
  TI_CHECK(block->size() == 7);

  irpass::simplify(block.get());  // should lower linearized
  // irpass::print(block.get());
  // TI_CHECK(block->size() == 8);

  irpass::typecheck(block.get());  // necessary here
  // irpass::print(block.get());

  irpass::constant_fold(block.get());
  irpass::alg_simp(block.get(), CompileConfig());
  irpass::die(block.get());  // should eliminate consts
  // irpass::print(block.get());
  TI_CHECK(block->size() == 5);  // get root, const 0, lookup, get child, lookup
}

TLANG_NAMESPACE_END
