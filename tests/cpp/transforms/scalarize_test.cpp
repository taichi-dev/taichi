#include "gtest/gtest.h"

#include "taichi/ir/statements.h"
#include "taichi/ir/transforms.h"
#include "tests/cpp/program/test_program.h"

namespace taichi::lang {

TEST(Scalarize, ScalarizeGlobalStore) {
  // Basic tests within a basic block
  TestProgram test_prog;
  test_prog.setup();

  auto block = std::make_unique<Block>();

  auto func = []() {};
  auto kernel =
      std::make_unique<Kernel>(*test_prog.prog(), func, "fake_kernel");

  auto &type_factory = TypeFactory::get_instance();

  /*
    TensorType<4 x i32>* %1 = ExternalPtrStmt()
    TensorType<4 x i32>  %2 = MatrixInitStmt([1, 1, 2, 2])
    StoreStmt(%1, %2)
  */
  Type *tensor_type = type_factory.get_tensor_type(
      {2, 2}, type_factory.get_primitive_type(PrimitiveTypeID::i32));
  auto const_1_stmt = block->push_back<ConstStmt>(TypedConstant(1));
  auto const_2_stmt = block->push_back<ConstStmt>(TypedConstant(2));
  auto type =
      TypeFactory::get_instance().get_ndarray_struct_type(tensor_type, 1);

  auto argload_stmt = block->push_back<ArgLoadStmt>(
      std::vector<int>{0} /*arg_id*/, type, /*is_ptr*/ true,
      /*create_load*/ false, /*arg_depth*/ 0);

  std::vector<Stmt *> indices = {};
  Stmt *dest_stmt = block->push_back<ExternalPtrStmt>(
      argload_stmt, indices);  // fake ExternalPtrStmt

  dest_stmt->ret_type = type_factory.get_pointer_type(tensor_type);

  std::vector<Stmt *> matrix_init_vals = {const_1_stmt, const_1_stmt,
                                          const_2_stmt, const_2_stmt};
  auto matrix_init_stmt =
      block->push_back<MatrixInitStmt>(std::move(matrix_init_vals));
  matrix_init_stmt->ret_type = tensor_type;

  block->push_back<GlobalStoreStmt>(dest_stmt, matrix_init_stmt);

  irpass::scalarize(block.get());
  irpass::lower_matrix_ptr(block.get());
  irpass::die(block.get());

  EXPECT_EQ(block->size(), 2 /*const*/ + 1 /*argload*/ + 4 /*const*/ +
                               4 /*external_ptr*/ + 4 /*store*/);

  // Check for scalarized statements
  EXPECT_EQ(block->statements[3]->is<ConstStmt>(), true);
  EXPECT_EQ(block->statements[4]->is<ExternalPtrStmt>(), true);
  EXPECT_EQ(block->statements[5]->is<GlobalStoreStmt>(), true);

  EXPECT_EQ(block->statements[6]->is<ConstStmt>(), true);
  EXPECT_EQ(block->statements[7]->is<ExternalPtrStmt>(), true);
  EXPECT_EQ(block->statements[8]->is<GlobalStoreStmt>(), true);

  EXPECT_EQ(block->statements[9]->is<ConstStmt>(), true);
  EXPECT_EQ(block->statements[10]->is<ExternalPtrStmt>(), true);
  EXPECT_EQ(block->statements[11]->is<GlobalStoreStmt>(), true);

  EXPECT_EQ(block->statements[12]->is<ConstStmt>(), true);
  EXPECT_EQ(block->statements[13]->is<ExternalPtrStmt>(), true);
  EXPECT_EQ(block->statements[14]->is<GlobalStoreStmt>(), true);
}

TEST(Scalarize, ScalarizeGlobalLoad) {
  TestProgram test_prog;
  test_prog.setup();

  auto block = std::make_unique<Block>();

  auto func = []() {};
  auto kernel =
      std::make_unique<Kernel>(*test_prog.prog(), func, "fake_kernel");

  auto &type_factory = TypeFactory::get_instance();

  /*
    TensorType<4 x i32>* %1 = ExternalPtrStmt()
    TensorType<4 x i32>  %2 = LoadStmt(%1)
    StoreStmt(%1, %2)
  */
  Type *tensor_type = type_factory.get_tensor_type(
      {2, 2}, type_factory.get_primitive_type(PrimitiveTypeID::i32));
  auto type =
      TypeFactory::get_instance().get_ndarray_struct_type(tensor_type, 1);

  auto argload_stmt = block->push_back<ArgLoadStmt>(
      std::vector<int>{0} /*arg_id*/, type, /*is_ptr*/ true,
      /*create_load*/ false, /*arg_depth*/ 0);

  std::vector<Stmt *> indices = {};
  Stmt *src_stmt = block->push_back<ExternalPtrStmt>(
      argload_stmt, indices);  // fake ExternalPtrStmt
  src_stmt->ret_type = type_factory.get_pointer_type(tensor_type);

  auto load_stmt = block->push_back<GlobalLoadStmt>(src_stmt);

  // Without this GlobalStoreStmt, nothing survives irpass::die()
  block->push_back<GlobalStoreStmt>(src_stmt, load_stmt);

  irpass::scalarize(block.get());
  irpass::lower_matrix_ptr(block.get());
  irpass::die(block.get());

  EXPECT_EQ(block->size(), 1 /*argload*/ + 4 /*const*/ + 4 /*external_ptr*/ +
                               4 /*load*/ + 4 /*const*/ + 4 /*external_ptr*/ +
                               4 /*store*/);

  // Check for scalarized statements
  EXPECT_EQ(block->statements[1]->is<ConstStmt>(), true);
  EXPECT_EQ(block->statements[2]->is<ExternalPtrStmt>(), true);
  EXPECT_EQ(block->statements[3]->is<GlobalLoadStmt>(), true);

  EXPECT_EQ(block->statements[4]->is<ConstStmt>(), true);
  EXPECT_EQ(block->statements[5]->is<ExternalPtrStmt>(), true);
  EXPECT_EQ(block->statements[6]->is<GlobalLoadStmt>(), true);

  EXPECT_EQ(block->statements[7]->is<ConstStmt>(), true);
  EXPECT_EQ(block->statements[8]->is<ExternalPtrStmt>(), true);
  EXPECT_EQ(block->statements[9]->is<GlobalLoadStmt>(), true);

  EXPECT_EQ(block->statements[10]->is<ConstStmt>(), true);
  EXPECT_EQ(block->statements[11]->is<ExternalPtrStmt>(), true);
  EXPECT_EQ(block->statements[12]->is<GlobalLoadStmt>(), true);
}

TEST(Scalarize, ScalarizeLocalStore) {
  // Basic tests within a basic block
  TestProgram test_prog;
  test_prog.setup();

  auto block = std::make_unique<Block>();

  auto func = []() {};
  auto kernel =
      std::make_unique<Kernel>(*test_prog.prog(), func, "fake_kernel");

  auto &type_factory = TypeFactory::get_instance();

  /*
    TensorType<4 x i32>* %1 = AllocaStmt()
    TensorType<4 x i32>  %2 = MatrixInitStmt([1, 1, 2, 2])
    StoreStmt(%1, %2)
  */
  Type *tensor_type = type_factory.get_tensor_type(
      {2, 2}, type_factory.get_primitive_type(PrimitiveTypeID::i32));
  Stmt *dest_stmt = block->push_back<AllocaStmt>(tensor_type);
  dest_stmt->ret_type = type_factory.get_pointer_type(tensor_type);

  auto const_1_stmt = block->push_back<ConstStmt>(TypedConstant(1));
  auto const_2_stmt = block->push_back<ConstStmt>(TypedConstant(2));
  std::vector<Stmt *> matrix_init_vals = {const_1_stmt, const_1_stmt,
                                          const_2_stmt, const_2_stmt};
  auto matrix_init_stmt =
      block->push_back<MatrixInitStmt>(std::move(matrix_init_vals));
  matrix_init_stmt->ret_type = tensor_type;

  // LocalStoreStmt survives irpass::die()
  block->push_back<LocalStoreStmt>(dest_stmt, matrix_init_stmt);

  irpass::scalarize(block.get());
  irpass::die(block.get());

  EXPECT_EQ(block->size(), 2 /*const*/ + 4 /*alloca*/ + 4 /*store*/);

  // Check for scalarized statements
  EXPECT_EQ(block->statements[0]->is<AllocaStmt>(), true);
  EXPECT_EQ(block->statements[1]->is<AllocaStmt>(), true);
  EXPECT_EQ(block->statements[2]->is<AllocaStmt>(), true);
  EXPECT_EQ(block->statements[3]->is<AllocaStmt>(), true);

  EXPECT_EQ(block->statements[4]->is<ConstStmt>(), true);
  EXPECT_EQ(block->statements[5]->is<ConstStmt>(), true);

  EXPECT_EQ(block->statements[6]->is<LocalStoreStmt>(), true);
  EXPECT_EQ(block->statements[7]->is<LocalStoreStmt>(), true);
  EXPECT_EQ(block->statements[8]->is<LocalStoreStmt>(), true);
  EXPECT_EQ(block->statements[9]->is<LocalStoreStmt>(), true);
}

TEST(Scalarize, ScalarizeLocalLoad) {
  // Basic tests within a basic block
  TestProgram test_prog;
  test_prog.setup();

  auto block = std::make_unique<Block>();

  auto func = []() {};
  auto kernel =
      std::make_unique<Kernel>(*test_prog.prog(), func, "fake_kernel");

  auto &type_factory = TypeFactory::get_instance();

  /*
    TensorType<4 x i32>* %1 = AllocaStmt()
    LoadStmt(%1)
  */
  Type *tensor_type = type_factory.get_tensor_type(
      {2, 2}, type_factory.get_primitive_type(PrimitiveTypeID::i32));
  Stmt *src_stmt = block->push_back<AllocaStmt>(tensor_type);
  src_stmt->ret_type = type_factory.get_pointer_type(tensor_type);

  auto load_stmt = block->push_back<LocalLoadStmt>(src_stmt);

  // Without this GlobalStoreStmt, nothing survives irpass::die()
  block->push_back<GlobalStoreStmt>(src_stmt, load_stmt);

  irpass::scalarize(block.get());
  irpass::die(block.get());

  EXPECT_EQ(block->size(), 4 /*alloca*/ + 4 /*load*/ + 4 /*store*/);

  // Check for scalarized statements
  EXPECT_EQ(block->statements[0]->is<AllocaStmt>(), true);
  EXPECT_EQ(block->statements[1]->is<AllocaStmt>(), true);
  EXPECT_EQ(block->statements[2]->is<AllocaStmt>(), true);
  EXPECT_EQ(block->statements[3]->is<AllocaStmt>(), true);

  EXPECT_EQ(block->statements[4]->is<LocalLoadStmt>(), true);
  EXPECT_EQ(block->statements[5]->is<LocalLoadStmt>(), true);
  EXPECT_EQ(block->statements[6]->is<LocalLoadStmt>(), true);
  EXPECT_EQ(block->statements[7]->is<LocalLoadStmt>(), true);
}

}  // namespace taichi::lang
