#include "gtest/gtest.h"

#include "taichi/ir/statements.h"
#include "taichi/ir/transforms.h"
#include "tests/cpp/program/test_program.h"

namespace taichi {
namespace lang {

// Basic tests within a basic block
template <typename T>
void test_store_scalarize() {
  TestProgram test_prog;
  test_prog.setup();

  auto block = std::make_unique<Block>();

  auto func = []() {};
  auto kernel =
      std::make_unique<Kernel>(*test_prog.prog(), func, "fake_kernel");
  block->kernel = kernel.get();

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
  auto argload_stmt = block->push_back<ArgLoadStmt>(0 /*arg_id*/, tensor_type);

  Stmt *dest_stmt = nullptr;
  if (std::is_same<T, GlobalStoreStmt>::value) {
    std::vector<Stmt *> indices = {};
    dest_stmt = block->push_back<ExternalPtrStmt>(
        argload_stmt, indices);  // fake ExternalPtrStmt

  } else {
    dest_stmt = block->push_back<AllocaStmt>(tensor_type);
  }
  dest_stmt->ret_type = type_factory.get_pointer_type(tensor_type);

  std::vector<Stmt *> matrix_init_vals = {const_1_stmt, const_1_stmt,
                                          const_2_stmt, const_2_stmt};
  auto matrix_init_stmt =
      block->push_back<MatrixInitStmt>(std::move(matrix_init_vals));
  matrix_init_stmt->ret_type = tensor_type;

  block->push_back<T>(dest_stmt, matrix_init_stmt);

  irpass::scalarize(block.get());

  EXPECT_EQ(block->size(), 2 /*const*/ + 1 /*argload*/ + 1 /*external_ptr*/ +
                               1 /*matrix_init*/ + 4 /*const*/ +
                               4 /*ptroffset*/ + 4 /*store*/);

  // Check for scalarized statements
  EXPECT_EQ(block->statements[5]->is<ConstStmt>(), true);
  EXPECT_EQ(block->statements[6]->is<PtrOffsetStmt>(), true);
  EXPECT_EQ(block->statements[7]->is<T>(), true);

  EXPECT_EQ(block->statements[8]->is<ConstStmt>(), true);
  EXPECT_EQ(block->statements[9]->is<PtrOffsetStmt>(), true);
  EXPECT_EQ(block->statements[10]->is<T>(), true);

  EXPECT_EQ(block->statements[11]->is<ConstStmt>(), true);
  EXPECT_EQ(block->statements[12]->is<PtrOffsetStmt>(), true);
  EXPECT_EQ(block->statements[13]->is<T>(), true);

  EXPECT_EQ(block->statements[14]->is<ConstStmt>(), true);
  EXPECT_EQ(block->statements[15]->is<PtrOffsetStmt>(), true);
  EXPECT_EQ(block->statements[16]->is<T>(), true);
}

template <typename T>
void test_load_scalarize() {
  TestProgram test_prog;
  test_prog.setup();

  auto block = std::make_unique<Block>();

  auto func = []() {};
  auto kernel =
      std::make_unique<Kernel>(*test_prog.prog(), func, "fake_kernel");
  block->kernel = kernel.get();

  auto &type_factory = TypeFactory::get_instance();

  /*
    TensorType<4 x i32>* %1 = ExternalPtrStmt()
    TensorType<4 x i32>  %2 = LoadStmt(%1)
  */
  Type *tensor_type = type_factory.get_tensor_type(
      {2, 2}, type_factory.get_primitive_type(PrimitiveTypeID::i32));
  auto argload_stmt = block->push_back<ArgLoadStmt>(0 /*arg_id*/, tensor_type);

  std::vector<Stmt *> indices = {};
  Stmt *src_stmt = block->push_back<ExternalPtrStmt>(
      argload_stmt, indices);  // fake ExternalPtrStmt
  src_stmt->ret_type = type_factory.get_pointer_type(tensor_type);

  block->push_back<T>(src_stmt);

  irpass::scalarize(block.get());

  EXPECT_EQ(block->size(), 1 /*argload*/ + 1 /*external_ptr*/ + 4 /*const*/ +
                               4 /*ptroffset*/ + 4 /*load*/ +
                               1 /*matrix_init*/);

  // Check for scalarized statements
  EXPECT_EQ(block->statements[2]->is<ConstStmt>(), true);
  EXPECT_EQ(block->statements[3]->is<PtrOffsetStmt>(), true);
  EXPECT_EQ(block->statements[4]->is<T>(), true);

  EXPECT_EQ(block->statements[5]->is<ConstStmt>(), true);
  EXPECT_EQ(block->statements[6]->is<PtrOffsetStmt>(), true);
  EXPECT_EQ(block->statements[7]->is<T>(), true);

  EXPECT_EQ(block->statements[8]->is<ConstStmt>(), true);
  EXPECT_EQ(block->statements[9]->is<PtrOffsetStmt>(), true);
  EXPECT_EQ(block->statements[10]->is<T>(), true);

  EXPECT_EQ(block->statements[11]->is<ConstStmt>(), true);
  EXPECT_EQ(block->statements[12]->is<PtrOffsetStmt>(), true);
  EXPECT_EQ(block->statements[13]->is<T>(), true);

  EXPECT_EQ(block->statements[14]->is<MatrixInitStmt>(), true);
}

TEST(Scalarize, ScalarizeStore) {
  test_store_scalarize<GlobalStoreStmt>();
  test_store_scalarize<LocalStoreStmt>();
}

TEST(Scalarize, ScalarizeLoad) {
  test_load_scalarize<GlobalLoadStmt>();
  test_load_scalarize<LocalLoadStmt>();
}

}  // namespace lang
}  // namespace taichi
