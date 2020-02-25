#include <taichi/tlang.h>
#include <taichi/common/testing.h>

TLANG_NAMESPACE_BEGIN

// Basic tests within a basic block

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
      block->push_back<SNodeLookupStmt>(root.ch[0].get(), get_child, linearized_zero, true, std::vector<Stmt*>());

  irpass::typecheck(block.get());
  // irpass::print(block.get());
  TI_CHECK(block->size() == 7);

  irpass::simplify(block.get());  // should lower linearized
  // irpass::print(block.get());
  // TI_CHECK(block->size() == 11);  // not required to check size here

  irpass::constant_fold(block.get());
  irpass::alg_simp(block.get(), CompileConfig());
  irpass::die(block.get());  // should eliminate consts
  // irpass::print(block.get());
  TI_CHECK(block->size() == 5);  // get root, const 0, lookup, get child, lookup
}

TLANG_NAMESPACE_END
