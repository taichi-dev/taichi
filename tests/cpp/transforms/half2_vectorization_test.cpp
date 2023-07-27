#include "gtest/gtest.h"

#include "taichi/ir/statements.h"
#include "taichi/ir/transforms.h"
#include "tests/cpp/program/test_program.h"

namespace taichi::lang {

TEST(Half2Vectorization, Ndarray) {
  // Basic tests within a basic block
  TestProgram test_prog;
  test_prog.setup();

  auto block = std::make_unique<Block>();

  auto func = []() {};
  auto kernel =
      std::make_unique<Kernel>(*test_prog.prog(), func, "fake_kernel");

  auto half2_type =
      TypeFactory::get_instance().create_tensor_type({2}, PrimitiveType::f16);

  auto argload_stmt = block->push_back<ArgLoadStmt>(
      std::vector<int>{0} /*arg_id*/, PrimitiveType::f16, /*is_ptr*/ true,
      /*create_load*/ false, /*arg_depth*/ 0);
  argload_stmt->ret_type = half2_type;
  auto const_0_stmt = block->push_back<ConstStmt>(TypedConstant(0));

  std::vector<Stmt *> external_ptr_indices0 = {const_0_stmt};
  auto external_ptr_stmt_0 =
      block->push_back<ExternalPtrStmt>(argload_stmt, external_ptr_indices0);
  external_ptr_stmt_0->ret_type = half2_type;
  external_ptr_stmt_0->ret_type.set_is_pointer(true);

  auto val_0_stmt = block->push_back<ConstStmt>(TypedConstant(10));
  auto val_1_stmt = block->push_back<ConstStmt>(TypedConstant(20));

  std::vector<Stmt *> values = {val_0_stmt, val_1_stmt};
  auto matrix_stmt = block->push_back<MatrixInitStmt>(values);
  matrix_stmt->ret_type = half2_type;

  auto atomic_stmt = block->push_back<AtomicOpStmt>(
      AtomicOpType::add, external_ptr_stmt_0, matrix_stmt);
  atomic_stmt->ret_type = half2_type;

  /*
    Before:
      <[Tensor (2) f16]> $0 = argaddr[0]
      <i32> $1 = const 0
      <*[Tensor (2) f16]> $2 = external_ptr $0, [$1] layout=AOS is_grad=false
      <i32> $3 = const 10
      <i32> $4 = const 20
      <[Tensor (2) f16]> $5 = [$3, $4]
      <[Tensor (2) f16]> $6 = atomic add($2, $5)
  */

  irpass::scalarize(block.get(), true /*half2_optimization_enabled*/);
  CompileConfig config;
  irpass::full_simplify(block.get(), config, {false, false});

  /*
    After:
      <[Tensor (2) f16]> $0 = argaddr[0]
      <i32> $1 = const 0
      <*[Tensor (2) f16]> $2 = external_ptr $0, [$1] layout=AOS is_grad=false
      <i32> $3 = const 10
      <i32> $4 = const 20
      <[Tensor (2) f16]> $5 = [$3, $4]
      <[Tensor (2) f16]> $6 = atomic add($2, $5)
  */
  EXPECT_EQ(block->size(), 7);

  // Check for scalarized statements
  EXPECT_EQ(block->statements[5]->is<MatrixInitStmt>(), true);
  EXPECT_EQ(block->statements[2]->is<ExternalPtrStmt>(), true);
  EXPECT_EQ(block->statements[6]->is<AtomicOpStmt>(), true);
}

TEST(Half2Vectorization, GlobalTemporary) {
  // Basic tests within a basic block
  TestProgram test_prog;
  test_prog.setup();

  auto block = std::make_unique<Block>();

  auto func = []() {};
  auto kernel =
      std::make_unique<Kernel>(*test_prog.prog(), func, "fake_kernel");
  auto half2_type =
      TypeFactory::get_instance().create_tensor_type({2}, PrimitiveType::f16);

  auto val_0_stmt = block->push_back<ConstStmt>(TypedConstant(10));
  auto val_1_stmt = block->push_back<ConstStmt>(TypedConstant(20));

  std::vector<Stmt *> values = {val_0_stmt, val_1_stmt};
  auto matrix_stmt = block->push_back<MatrixInitStmt>(values);
  matrix_stmt->ret_type = half2_type;

  auto global_temp_stmt_0 =
      block->push_back<GlobalTemporaryStmt>(0, half2_type);

  block->push_back<AtomicOpStmt>(AtomicOpType::add, global_temp_stmt_0,
                                 matrix_stmt);

  irpass::type_check(block.get(), CompileConfig());

  /*
    Before:
      <i32> $0 = const 10
      <f16> $1 = cast_value<f16> $0
      <i32> $2 = const 20
      <f16> $3 = cast_value<f16> $2
      <[Tensor (2) f16]> $4 = [$1, $3]
      <*[Tensor (2) f16]> $5 = global tmp var (offset = 0 B)
      <[Tensor (2) f16]> $6 = atomic add($5, $4)
  */

  irpass::scalarize(block.get(), true /*half2_optimization_enabled*/);
  CompileConfig config;
  irpass::full_simplify(block.get(), config, {false, false});
  /*
    After:
      <f16> $0 = const 10.0
      <f16> $1 = const 20.0
      <[Tensor (2) f16]> $2 = [$0, $1]
      <*[Tensor (2) f16]> $3 = global tmp var (offset = 0 B)
      <[Tensor (2) f16]> $4 = atomic add($3, $2)
  */
  EXPECT_EQ(block->size(), 5);

  // Check for scalarized statements
  EXPECT_EQ(block->statements[2]->is<MatrixInitStmt>(), true);
  EXPECT_EQ(block->statements[3]->is<GlobalTemporaryStmt>(), true);
  EXPECT_EQ(block->statements[4]->is<AtomicOpStmt>(), true);
}

TEST(Half2Vectorization, Field) {
  // Basic tests within a basic block
  TestProgram test_prog;
  test_prog.setup();

  auto block = std::make_unique<Block>();

  auto func = []() {};
  auto kernel =
      std::make_unique<Kernel>(*test_prog.prog(), func, "fake_kernel");

  auto get_root = block->push_back<GetRootStmt>();
  auto linearized_empty = block->push_back<LinearizeStmt>(std::vector<Stmt *>(),
                                                          std::vector<int>());
  SNode root(0, SNodeType::root);
  root.insert_children(SNodeType::place);
  root.insert_children(SNodeType::place);

  auto lookup = block->push_back<SNodeLookupStmt>(&root, get_root,
                                                  linearized_empty, false);

  auto get_ch_stmt_0 = block->push_back<GetChStmt>(lookup, 0);

  auto half2_type =
      TypeFactory::get_instance().create_tensor_type({2}, PrimitiveType::f16);
  get_ch_stmt_0->ret_type = half2_type;
  get_ch_stmt_0->ret_type.set_is_pointer(true);
  get_ch_stmt_0->as<GetChStmt>()->overrided_dtype = true;

  auto val_0_stmt = block->push_back<ConstStmt>(TypedConstant(10));
  auto val_1_stmt = block->push_back<ConstStmt>(TypedConstant(20));

  std::vector<Stmt *> values = {val_0_stmt, val_1_stmt};
  auto matrix_stmt = block->push_back<MatrixInitStmt>(values);
  matrix_stmt->ret_type = half2_type;

  block->push_back<AtomicOpStmt>(AtomicOpType::add, get_ch_stmt_0, matrix_stmt);

  irpass::type_check(block.get(), CompileConfig());
  /*
    Before:
      <*gen> $0 = get root nullptr
      <i32> $1 = linearized(ind {}, stride {})
      <*gen> $2 = [S1root][root]::lookup($0, $1) activate = false
      <*[Tensor (2) f16]> $3 = get child [S1root->S2place<gen>] $2
      <i32> $4 = const 10
      <f16> $5 = cast_value<f16> $4
      <i32> $6 = const 20
      <f16> $7 = cast_value<f16> $6
      <[Tensor (2) f16]> $8 = [$5, $7]
      <[Tensor (2) f16]> $9 = atomic add($3, $8)
  */

  irpass::scalarize(block.get(), true /*half2_optimization_enabled*/);

  CompileConfig config;
  irpass::full_simplify(block.get(), config, {false, false});
  /*
    After:
      <*gen> $0 = get root nullptr
      <i32> $1 = const 0
      <*gen> $2 = [S1root][root]::lookup($0, $1) activate = false
      <*[Tensor (2) f16]> $3 = get child [S1root->S2place<gen>] $2
      <f16> $4 = const 10.0
      <f16> $5 = const 20.0
      <[Tensor (2) f16]> $6 = [$4, $5]
      <[Tensor (2) f16]> $7 = atomic add($3, $6)
  */

  EXPECT_EQ(block->size(), 8);
  // Check for scalarized statements
  EXPECT_EQ(block->statements[6]->is<MatrixInitStmt>(), true);
  EXPECT_EQ(block->statements[3]->is<GetChStmt>(), true);
  EXPECT_EQ(block->statements[7]->is<AtomicOpStmt>(), true);
}

}  // namespace taichi::lang
