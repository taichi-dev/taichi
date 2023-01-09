#include "gtest/gtest.h"

#include "taichi/ir/statements.h"
#include "taichi/ir/transforms.h"
#include "tests/cpp/program/test_program.h"

namespace taichi::lang {

TEST(TensorType, eliminateImmutableLocalVars) {
  // Basic tests within a basic block
  TestProgram test_prog;
  test_prog.setup();

  auto block = std::make_unique<Block>();

  auto func = []() {};
  auto kernel =
      std::make_unique<Kernel>(*test_prog.prog(), func, "fake_kernel");

  auto &type_factory = TypeFactory::get_instance();

  /*
    Declare tensor types
  */
  Type *tensor2x2 = type_factory.get_tensor_type(
      {2, 2}, type_factory.get_primitive_type(PrimitiveTypeID::i32));

  /* Define initial IR

    %1 = AllocaStmt(TensorType(2x2, i32))
    %2 = ConstStmt(1)
    %3 = MatrixInitStmt({%2, %2, %2, %2})
    LocalStoreStmt(%1, %3)
    %4 = LocalLoadStmt(%1)
    %5 = LocalLoadStmt(%1)
    %6 = BinaryOpStmt(add, %4, %5)
  */
  Stmt *alloca_stmt = block->push_back<AllocaStmt>(tensor2x2);
  alloca_stmt->ret_type = tensor2x2;

  auto const_1_stmt = block->push_back<ConstStmt>(TypedConstant(1));
  std::vector<Stmt *> matrix_vals = {const_1_stmt, const_1_stmt, const_1_stmt,
                                     const_1_stmt};
  auto matrix_init_stmt = block->push_back<MatrixInitStmt>(matrix_vals);
  matrix_init_stmt->ret_type = tensor2x2;

  block->push_back<LocalStoreStmt>(alloca_stmt, matrix_init_stmt);

  auto load_stmt0 = block->push_back<LocalLoadStmt>(alloca_stmt);
  load_stmt0->ret_type = tensor2x2;

  auto load_stmt1 = block->push_back<LocalLoadStmt>(alloca_stmt);
  load_stmt1->ret_type = tensor2x2;

  auto bin_stmt =
      block->push_back<BinaryOpStmt>(BinaryOpType::add, load_stmt0, load_stmt1);
  bin_stmt->ret_type = tensor2x2;

  irpass::eliminate_immutable_local_vars(block.get());

  /* Transformed IR

    %1 = ConstStmt(1)
    %2 = MatrixInitStmt({%1, %1, %1, %1})
    %3 = BinaryOpStmt(add, %3, %3)
  */

  EXPECT_EQ(block->size(), 3);

  // Check for scalarized statements
  EXPECT_EQ(block->statements[0]->is<ConstStmt>(), true);
  EXPECT_EQ(block->statements[1]->is<MatrixInitStmt>(), true);
  EXPECT_EQ(block->statements[2]->is<BinaryOpStmt>(), true);
  EXPECT_EQ(block->statements[2]->ret_type == DataType(tensor2x2), true);
}

}  // namespace taichi::lang
