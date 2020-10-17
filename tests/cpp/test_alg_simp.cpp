#include "taichi/ir/frontend.h"
#include "taichi/ir/statements.h"
#include "taichi/ir/transforms.h"
#include "taichi/util/testing.h"

TLANG_NAMESPACE_BEGIN

// Basic tests within a basic block
TI_TEST("alg_simp") {
  SECTION("simplify_add_zero") {
    TI_TEST_PROGRAM;

    auto block = std::make_unique<Block>();

    auto func = []() {};
    auto kernel =
        std::make_unique<Kernel>(get_current_program(), func, "fake_kernel");
    block->kernel = kernel.get();

    auto global_load_addr = block->push_back<GlobalTemporaryStmt>(
        0, TypeFactory::create_vector_or_scalar_type(1, PrimitiveType::i32));
    auto global_load = block->push_back<GlobalLoadStmt>(global_load_addr);
    auto zero = block->push_back<ConstStmt>(TypedConstant(0));
    auto add =
        block->push_back<BinaryOpStmt>(BinaryOpType::add, global_load, zero);
    auto global_store_addr = block->push_back<GlobalTemporaryStmt>(
        4, TypeFactory::create_vector_or_scalar_type(1, PrimitiveType::i32));
    auto global_store =
        block->push_back<GlobalStoreStmt>(global_store_addr, add);

    irpass::type_check(block.get());
    TI_CHECK(block->size() == 6);

    // irpass::print(block.get());

    irpass::alg_simp(block.get());  // should eliminate add
    irpass::die(block.get());       // should eliminate zero

    // irpass::print(block.get());
    TI_CHECK(block->size() == 4);  // two addresses, one load, one store
    TI_CHECK((*block)[0]->is<GlobalTemporaryStmt>());
    // .. more tests, assuming instruction order not shuffled
  }

  SECTION("simplify_multiply_one") {
    TI_TEST_PROGRAM;

    auto block = std::make_unique<Block>();

    auto func = []() {};
    auto kernel =
        std::make_unique<Kernel>(get_current_program(), func, "fake_kernel");
    block->kernel = kernel.get();

    auto global_load_addr = block->push_back<GlobalTemporaryStmt>(
        0, TypeFactory::create_vector_or_scalar_type(1, PrimitiveType::f32));
    auto global_load = block->push_back<GlobalLoadStmt>(global_load_addr);
    auto one = block->push_back<ConstStmt>(TypedConstant(1.0f));
    auto mul1 =
        block->push_back<BinaryOpStmt>(BinaryOpType::mul, one, global_load);
    auto mul2 = block->push_back<BinaryOpStmt>(BinaryOpType::mul, mul1, one);
    auto zero = block->push_back<ConstStmt>(TypedConstant(0.0f));
    auto div = block->push_back<BinaryOpStmt>(BinaryOpType::div, zero, one);
    auto sub = block->push_back<BinaryOpStmt>(BinaryOpType::sub, mul2, div);
    auto global_store_addr = block->push_back<GlobalTemporaryStmt>(
        4, TypeFactory::create_vector_or_scalar_type(1, PrimitiveType::f32));
    auto global_store =
        block->push_back<GlobalStoreStmt>(global_store_addr, sub);

    irpass::type_check(block.get());
    TI_CHECK(block->size() == 10);

    // irpass::print(block.get());

    irpass::alg_simp(block.get());  // should eliminate mul, div, sub
    irpass::die(block.get());       // should eliminate zero, one

    // irpass::print(block.get());

    TI_CHECK(block->size() == 4);  // two addresses, one load, one store
    TI_CHECK((*block)[0]->is<GlobalTemporaryStmt>());
  }

  SECTION("simplify_multiply_zero_fast_math") {
    TI_TEST_PROGRAM;

    auto block = std::make_unique<Block>();
    auto func = []() {};
    auto kernel =
        std::make_unique<Kernel>(get_current_program(), func, "fake_kernel");
    block->kernel = kernel.get();

    auto global_load_addr = block->push_back<GlobalTemporaryStmt>(
        0, TypeFactory::create_vector_or_scalar_type(1, PrimitiveType::i32));
    auto global_load = block->push_back<GlobalLoadStmt>(global_load_addr);
    auto zero = block->push_back<ConstStmt>(TypedConstant(0));
    auto mul =
        block->push_back<BinaryOpStmt>(BinaryOpType::mul, global_load, zero);
    auto one = block->push_back<ConstStmt>(TypedConstant(1));
    auto add = block->push_back<BinaryOpStmt>(BinaryOpType::add, mul, one);
    auto global_store_addr = block->push_back<GlobalTemporaryStmt>(
        4, TypeFactory::create_vector_or_scalar_type(1, PrimitiveType::i32));
    auto global_store =
        block->push_back<GlobalStoreStmt>(global_store_addr, add);

    irpass::type_check(block.get());
    TI_CHECK(block->size() == 8);

    CompileConfig config_without_fast_math;
    config_without_fast_math.fast_math = false;
    kernel->program.config = config_without_fast_math;

    irpass::alg_simp(block.get());  // should eliminate mul, add
    irpass::die(block.get());       // should eliminate zero, load

    TI_CHECK(block->size() == 3);  // one address, one one, one store

    block = std::make_unique<Block>();
    block->kernel = kernel.get();

    global_load_addr = block->push_back<GlobalTemporaryStmt>(
        8, TypeFactory::create_vector_or_scalar_type(1, PrimitiveType::f32));
    global_load = block->push_back<GlobalLoadStmt>(global_load_addr);
    zero = block->push_back<ConstStmt>(TypedConstant(0));
    mul = block->push_back<BinaryOpStmt>(BinaryOpType::mul, global_load, zero);
    one = block->push_back<ConstStmt>(TypedConstant(1));
    add = block->push_back<BinaryOpStmt>(BinaryOpType::add, mul, one);
    global_store_addr = block->push_back<GlobalTemporaryStmt>(
        12, TypeFactory::create_vector_or_scalar_type(1, PrimitiveType::f32));
    global_store = block->push_back<GlobalStoreStmt>(global_store_addr, add);

    irpass::type_check(block.get());  // insert 2 casts
    TI_CHECK(block->size() == 10);

    irpass::constant_fold(block.get());  // should change 2 casts into const
    irpass::alg_simp(block.get());       // should not eliminate
    irpass::die(block.get());            // should eliminate 2 const
    TI_CHECK(block->size() == 8);

    CompileConfig config_with_fast_math;
    config_with_fast_math.fast_math = true;
    kernel->program.config = config_with_fast_math;

    irpass::alg_simp(block.get());  // should eliminate mul, add
    irpass::die(block.get());       // should eliminate zero, load

    TI_CHECK(block->size() == 3);  // one address, one one, one store
  }

  SECTION("simplify_and_minus_one") {
    TI_TEST_PROGRAM;

    auto block = std::make_unique<Block>();

    auto global_load_addr = block->push_back<GlobalTemporaryStmt>(
        0, TypeFactory::create_vector_or_scalar_type(1, PrimitiveType::i32));
    auto global_load = block->push_back<GlobalLoadStmt>(global_load_addr);
    auto minus_one = block->push_back<ConstStmt>(TypedConstant(-1));
    auto and_result = block->push_back<BinaryOpStmt>(BinaryOpType::bit_and,
                                                     minus_one, global_load);
    auto global_store_addr = block->push_back<GlobalTemporaryStmt>(
        4, TypeFactory::create_vector_or_scalar_type(1, PrimitiveType::i32));
    auto global_store =
        block->push_back<GlobalStoreStmt>(global_store_addr, and_result);

    auto func = []() {};
    auto kernel =
        std::make_unique<Kernel>(get_current_program(), func, "fake_kernel");
    block->kernel = kernel.get();
    irpass::type_check(block.get());
    TI_CHECK(block->size() == 6);

    irpass::alg_simp(block.get());  // should eliminate and
    irpass::die(block.get());       // should eliminate zero

    TI_CHECK(block->size() == 4);  // two addresses, one load, one store
    TI_CHECK((*block)[0]->is<GlobalTemporaryStmt>());
  }
}

TLANG_NAMESPACE_END
