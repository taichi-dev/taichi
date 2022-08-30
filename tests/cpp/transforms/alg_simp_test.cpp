#include "gtest/gtest.h"

#include "taichi/ir/statements.h"
#include "taichi/ir/transforms.h"
#include "tests/cpp/program/test_program.h"

namespace taichi {
namespace lang {

class AlgebraicSimplicationTest : public ::testing::Test {
 protected:
  void SetUp() override {
    tp_.setup();
  }

  Program &prog() {
    return *tp_.prog();
  }

  TestProgram tp_;
};

TEST_F(AlgebraicSimplicationTest, SimplifyAddZero) {
  auto block = std::make_unique<Block>();

  auto func = []() {};
  auto kernel = std::make_unique<Kernel>(prog(), func, "fake_kernel");
  block->kernel = kernel.get();

  auto global_load_addr =
      block->push_back<GlobalTemporaryStmt>(0, PrimitiveType::i32);
  auto global_load = block->push_back<GlobalLoadStmt>(global_load_addr);
  auto zero = block->push_back<ConstStmt>(TypedConstant(0));
  auto add =
      block->push_back<BinaryOpStmt>(BinaryOpType::add, global_load, zero);
  auto global_store_addr =
      block->push_back<GlobalTemporaryStmt>(4, PrimitiveType::i32);
  block->push_back<GlobalStoreStmt>(global_store_addr, add);

  irpass::type_check(block.get(), CompileConfig());
  EXPECT_EQ(block->size(), 6);

  irpass::alg_simp(block.get(), CompileConfig());  // should eliminate add
  irpass::die(block.get());                        // should eliminate zero

  EXPECT_EQ(block->size(), 4);  // two addresses, one load, one store
  EXPECT_TRUE((*block)[0]->is<GlobalTemporaryStmt>());
}

TEST_F(AlgebraicSimplicationTest, SimplifyMultiplyOne) {
  auto block = std::make_unique<Block>();

  auto func = []() {};
  auto kernel = std::make_unique<Kernel>(prog(), func, "fake_kernel");
  block->kernel = kernel.get();

  auto global_load_addr =
      block->push_back<GlobalTemporaryStmt>(0, PrimitiveType::f32);
  auto global_load = block->push_back<GlobalLoadStmt>(global_load_addr);
  auto one = block->push_back<ConstStmt>(TypedConstant(1.0f));
  auto mul1 =
      block->push_back<BinaryOpStmt>(BinaryOpType::mul, one, global_load);
  auto mul2 = block->push_back<BinaryOpStmt>(BinaryOpType::mul, mul1, one);
  auto zero = block->push_back<ConstStmt>(TypedConstant(0.0f));
  auto div = block->push_back<BinaryOpStmt>(BinaryOpType::div, zero, one);
  auto sub = block->push_back<BinaryOpStmt>(BinaryOpType::sub, mul2, div);
  auto global_store_addr =
      block->push_back<GlobalTemporaryStmt>(4, PrimitiveType::f32);
  [[maybe_unused]] auto global_store =
      block->push_back<GlobalStoreStmt>(global_store_addr, sub);

  irpass::type_check(block.get(), CompileConfig());
  EXPECT_EQ(block->size(), 10);

  irpass::alg_simp(block.get(),
                   CompileConfig());  // should eliminate mul, div, sub
  irpass::die(block.get());           // should eliminate zero, one

  EXPECT_EQ(block->size(), 4);  // two addresses, one load, one store
  EXPECT_TRUE((*block)[0]->is<GlobalTemporaryStmt>());
}

TEST_F(AlgebraicSimplicationTest, SimplifyMultiplyZeroFastMath) {
  auto block = std::make_unique<Block>();
  auto func = []() {};
  auto kernel = std::make_unique<Kernel>(prog(), func, "fake_kernel");
  block->kernel = kernel.get();

  auto global_load_addr =
      block->push_back<GlobalTemporaryStmt>(0, PrimitiveType::i32);
  auto global_load = block->push_back<GlobalLoadStmt>(global_load_addr);
  auto zero = block->push_back<ConstStmt>(TypedConstant(0));
  auto mul =
      block->push_back<BinaryOpStmt>(BinaryOpType::mul, global_load, zero);
  auto one = block->push_back<ConstStmt>(TypedConstant(1));
  auto add = block->push_back<BinaryOpStmt>(BinaryOpType::add, mul, one);
  auto global_store_addr =
      block->push_back<GlobalTemporaryStmt>(4, PrimitiveType::i32);
  [[maybe_unused]] auto global_store =
      block->push_back<GlobalStoreStmt>(global_store_addr, add);

  CompileConfig config_without_fast_math;
  config_without_fast_math.fast_math = false;
  kernel->program->config = config_without_fast_math;

  irpass::type_check(block.get(), config_without_fast_math);
  EXPECT_EQ(block->size(), 8);

  irpass::alg_simp(block.get(),
                   config_without_fast_math);  // should eliminate mul, add
  irpass::die(block.get());                    // should eliminate zero, load

  EXPECT_EQ(block->size(), 3);  // one address, one one, one store

  block = std::make_unique<Block>();
  block->kernel = kernel.get();

  global_load_addr =
      block->push_back<GlobalTemporaryStmt>(8, PrimitiveType::f32);
  global_load = block->push_back<GlobalLoadStmt>(global_load_addr);
  zero = block->push_back<ConstStmt>(TypedConstant(0));
  mul = block->push_back<BinaryOpStmt>(BinaryOpType::mul, global_load, zero);
  one = block->push_back<ConstStmt>(TypedConstant(1));
  add = block->push_back<BinaryOpStmt>(BinaryOpType::add, mul, one);
  global_store_addr =
      block->push_back<GlobalTemporaryStmt>(12, PrimitiveType::f32);
  global_store = block->push_back<GlobalStoreStmt>(global_store_addr, add);

  irpass::type_check(block.get(), config_without_fast_math);  // insert 2 casts
  EXPECT_EQ(block->size(), 10);

  irpass::constant_fold(block.get(), config_without_fast_math,
                        {kernel->program});  // should change 2 casts into const
  irpass::alg_simp(block.get(),
                   config_without_fast_math);  // should not eliminate
  irpass::die(block.get());                    // should eliminate 2 const
  EXPECT_EQ(block->size(), 8);

  CompileConfig config_with_fast_math;
  config_with_fast_math.fast_math = true;
  kernel->program->config = config_with_fast_math;

  irpass::alg_simp(block.get(),
                   config_with_fast_math);  // should eliminate mul, add
  irpass::die(block.get());                 // should eliminate zero, load

  EXPECT_EQ(block->size(), 3);  // one address, one one, one store
}

TEST_F(AlgebraicSimplicationTest, SimplifyAndMinusOne) {
  auto block = std::make_unique<Block>();

  auto global_load_addr =
      block->push_back<GlobalTemporaryStmt>(0, PrimitiveType::i32);
  auto global_load = block->push_back<GlobalLoadStmt>(global_load_addr);
  auto minus_one = block->push_back<ConstStmt>(TypedConstant(-1));
  auto and_result = block->push_back<BinaryOpStmt>(BinaryOpType::bit_and,
                                                   minus_one, global_load);
  auto global_store_addr =
      block->push_back<GlobalTemporaryStmt>(4, PrimitiveType::i32);
  block->push_back<GlobalStoreStmt>(global_store_addr, and_result);

  auto func = []() {};
  auto kernel = std::make_unique<Kernel>(prog(), func, "fake_kernel");
  block->kernel = kernel.get();
  irpass::type_check(block.get(), CompileConfig());
  EXPECT_EQ(block->size(), 6);

  irpass::alg_simp(block.get(), CompileConfig());  // should eliminate and
  irpass::die(block.get());                        // should eliminate zero

  EXPECT_EQ(block->size(), 4);  // two addresses, one load, one store
  EXPECT_TRUE((*block)[0]->is<GlobalTemporaryStmt>());
}

}  // namespace lang
}  // namespace taichi
