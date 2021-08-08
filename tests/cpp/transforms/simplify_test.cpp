#include "gtest/gtest.h"

#include "taichi/ir/statements.h"
#include "taichi/ir/transforms.h"
#include "tests/cpp/program/test_program.h"

namespace taichi {
namespace lang {

// Basic tests within a basic block

TEST(Simplify, SimplifyLinearizedWithTrivialInputs) {
  TestProgram test_prog;
  test_prog.setup();

  auto block = std::make_unique<Block>();

  auto func = []() {};
  auto kernel =
      std::make_unique<Kernel>(*test_prog.prog(), func, "fake_kernel");
  block->kernel = kernel.get();

  auto get_root = block->push_back<GetRootStmt>();
  auto linearized_empty = block->push_back<LinearizeStmt>(std::vector<Stmt *>(),
                                                          std::vector<int>());
  SNode root(0, SNodeType::root);
  root.insert_children(SNodeType::dense);
  auto lookup = block->push_back<SNodeLookupStmt>(&root, get_root,
                                                  linearized_empty, false);
  auto get_child = block->push_back<GetChStmt>(lookup, 0);
  auto zero = block->push_back<ConstStmt>(TypedConstant(0));
  auto linearized_zero = block->push_back<LinearizeStmt>(
      std::vector<Stmt *>(2, zero), std::vector<int>({8, 4}));
  [[maybe_unused]] auto lookup2 = block->push_back<SNodeLookupStmt>(
      root.ch[0].get(), get_child, linearized_zero, true);

  irpass::type_check(block.get(), kernel->program->config);
  EXPECT_EQ(block->size(), 7);

  irpass::simplify(block.get(),
                   kernel->program->config);  // should lower linearized
  // EXPECT_EQ(block->size(), 11);  // not required to check size here

  irpass::constant_fold(block.get(), kernel->program->config,
                        {kernel->program});
  irpass::alg_simp(block.get(), kernel->program->config);
  irpass::die(block.get());  // should eliminate consts
  irpass::simplify(block.get(), kernel->program->config);
  irpass::whole_kernel_cse(block.get());
  if (kernel->program->config.advanced_optimization) {
    // get root, const 0, lookup, get child, lookup
    EXPECT_EQ(block->size(), 5);
  }
}

}  // namespace lang
}  // namespace taichi
