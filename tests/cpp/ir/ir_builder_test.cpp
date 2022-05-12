#include "gtest/gtest.h"

#include "taichi/ir/ir_builder.h"
#include "taichi/ir/statements.h"
#include "tests/cpp/program/test_program.h"
#ifdef TI_WITH_VULKAN
#include "taichi/backends/vulkan/vulkan_loader.h"
#endif

namespace taichi {
namespace lang {

TEST(IRBuilder, Basic) {
  IRBuilder builder;
  auto *lhs = builder.get_int32(40);
  auto *rhs = builder.get_int32(2);
  auto *add = builder.create_add(lhs, rhs);
  ASSERT_TRUE(add->is<BinaryOpStmt>());
  auto *addc = add->cast<BinaryOpStmt>();
  EXPECT_EQ(addc->lhs, lhs);
  EXPECT_EQ(addc->rhs, rhs);
  EXPECT_EQ(addc->op_type, BinaryOpType::add);
  auto ir = builder.extract_ir();
  ASSERT_TRUE(ir->is<Block>());
  EXPECT_EQ(ir->as<Block>()->size(), 3);
}

TEST(IRBuilder, Print) {
  IRBuilder builder;
  auto *one = builder.get_int32(1);
  ASSERT_TRUE(one->is<ConstStmt>());
  std::string message = "message";
  auto *result = builder.create_print(one, message, one);
  ASSERT_TRUE(result->is<PrintStmt>());
  auto *print = result->cast<PrintStmt>();
  EXPECT_EQ(print->contents.size(), 3);
  ASSERT_TRUE(std::holds_alternative<Stmt *>(print->contents[0]));
  EXPECT_EQ(std::get<Stmt *>(print->contents[0]), one);
  ASSERT_TRUE(std::holds_alternative<std::string>(print->contents[1]));
  EXPECT_EQ(std::get<std::string>(print->contents[1]), message);
  ASSERT_TRUE(std::holds_alternative<Stmt *>(print->contents[2]));
  EXPECT_EQ(std::get<Stmt *>(print->contents[2]), one);
}

TEST(IRBuilder, RangeFor) {
  IRBuilder builder;
  auto *zero = builder.get_int32(0);
  auto *ten = builder.get_int32(10);
  auto *loop = builder.create_range_for(zero, ten);
  Stmt *index;
  {
    auto _ = builder.get_loop_guard(loop);
    index = builder.get_loop_index(loop, 0);
  }
  [[maybe_unused]] auto *ret = builder.create_return(zero);
  EXPECT_EQ(zero->parent->size(), 4);
  ASSERT_TRUE(loop->is<RangeForStmt>());
  auto *loopc = loop->cast<RangeForStmt>();
  EXPECT_EQ(loopc->body->size(), 1);
  EXPECT_EQ(loopc->body->statements[0].get(), index);
}

TEST(IRBuilder, LoopGuard) {
  IRBuilder builder;
  auto *zero = builder.get_int32(0);
  auto *ten = builder.get_int32(10);
  auto *loop = builder.create_range_for(zero, ten);
  Stmt *two;
  Stmt *one;
  Stmt *sum;
  {
    auto _ = builder.get_loop_guard(loop);
    one = builder.get_int32(1);
    builder.set_insertion_point_to_before(loop);
    two = builder.get_int32(2);
    builder.set_insertion_point_to_after(one);
    sum = builder.create_add(one, two);
  }
  // The insertion point should be after the loop now.
  auto *print = builder.create_print(two);
  EXPECT_EQ(zero->parent->size(), 5);
  EXPECT_EQ(zero->parent->statements[2].get(), two);
  EXPECT_EQ(zero->parent->statements[3].get(), loop);
  EXPECT_EQ(zero->parent->statements[4].get(), print);
  EXPECT_EQ(loop->body->size(), 2);
  EXPECT_EQ(loop->body->statements[0].get(), one);
  EXPECT_EQ(loop->body->statements[1].get(), sum);
}

TEST(IRBuilder, ExternalPtr) {
  TestProgram test_prog;
  test_prog.setup();

  IRBuilder builder;
  const int size = 10;
  auto array = std::make_unique<int[]>(size);
  array[0] = 2;
  array[2] = 40;
  auto *arg = builder.create_arg_load(/*arg_id=*/0, get_data_type<int>(),
                                      /*is_ptr=*/true);
  auto *zero = builder.get_int32(0);
  auto *one = builder.get_int32(1);
  auto *two = builder.get_int32(2);
  auto *a1ptr = builder.create_external_ptr(arg, {one});
  builder.create_global_store(a1ptr, one);  // a[1] = 1
  auto *a0 =
      builder.create_global_load(builder.create_external_ptr(arg, {zero}));
  auto *a2ptr = builder.create_external_ptr(arg, {two});
  auto *a2 = builder.create_global_load(a2ptr);
  auto *a0plusa2 = builder.create_add(a0, a2);
  builder.create_global_store(a2ptr, a0plusa2);  // a[2] = a[0] + a[2]
  auto block = builder.extract_ir();
  auto ker = std::make_unique<Kernel>(*test_prog.prog(), std::move(block));
  ker->insert_arg(get_data_type<int>(), /*is_array=*/true);
  auto launch_ctx = ker->make_launch_context();
  launch_ctx.set_arg_external_array(/*arg_id=*/0, (uint64)array.get(), size,
                                    /*is_device_allocation=*/false);
  (*ker)(launch_ctx);
  EXPECT_EQ(array[0], 2);
  EXPECT_EQ(array[1], 1);
  EXPECT_EQ(array[2], 42);
}

TEST(IRBuilder, Ndarray) {
  TestProgram test_prog;
#ifdef TI_WITH_VULKAN
  Arch arch = taichi::lang::vulkan::is_vulkan_api_available() ? Arch::vulkan : Arch::x64;
#else
  Arch arch = Arch::x64;
#endif
  test_prog.setup(arch);
  IRBuilder builder1;
  int size = 10;

  auto array = Ndarray(test_prog.prog(), PrimitiveType::i32, {size});
  array.write_int({0}, 2);
  array.write_int({2}, 40);
  {
    auto *arg = builder1.create_arg_load(/*arg_id=*/0, get_data_type<int>(),
                                         /*is_ptr=*/true);
    auto *zero = builder1.get_int32(0);
    auto *one = builder1.get_int32(1);
    auto *two = builder1.get_int32(2);
    auto *a1ptr = builder1.create_external_ptr(arg, {one});
    builder1.create_global_store(a1ptr, one);  // a[1] = 1
    auto *a0 =
        builder1.create_global_load(builder1.create_external_ptr(arg, {zero}));
    auto *a2ptr = builder1.create_external_ptr(arg, {two});
    auto *a2 = builder1.create_global_load(a2ptr);
    auto *a0plusa2 = builder1.create_add(a0, a2);
    builder1.create_global_store(a2ptr, a0plusa2);  // a[2] = a[0] + a[2]
  }
  auto block1 = builder1.extract_ir();
  auto ker1 = std::make_unique<Kernel>(*test_prog.prog(), std::move(block1), "ker1");
  ker1->insert_arg(get_data_type<int>(), /*is_array=*/true);
  auto launch_ctx1 = ker1->make_launch_context();
  launch_ctx1.set_arg_external_array(
      /*arg_id=*/0, array.get_device_allocation_ptr_as_int(), size,
      /*is_device_allocation=*/true);
  (*ker1)(launch_ctx1);
  EXPECT_EQ(array.read_int({0}), 2);
  EXPECT_EQ(array.read_int({1}), 1);
  EXPECT_EQ(array.read_int({2}), 42);

  IRBuilder builder2;
  {
    auto *arg0 = builder2.create_arg_load(/*arg_id=*/0, get_data_type<int>(),
                                          /*is_ptr=*/true);
    auto *arg1 = builder2.create_arg_load(/*arg_id=*/1, PrimitiveType::i32,
                                          /*is_ptr=*/false);
    auto *one = builder2.get_int32(1);
    auto *a1ptr = builder2.create_external_ptr(arg0, {one});
    builder2.create_global_store(a1ptr, arg1);  // a[1] = arg1
  }
  auto block2 = builder2.extract_ir();
  auto ker2 = std::make_unique<Kernel>(*test_prog.prog(), std::move(block2), "ker2");
  ker2->insert_arg(get_data_type<int>(), /*is_array=*/true);
  ker2->insert_arg(get_data_type<int>(), /*is_array=*/false);
  auto launch_ctx2 = ker2->make_launch_context();
  launch_ctx2.set_arg_external_array(
      /*arg_id=*/0, array.get_device_allocation_ptr_as_int(), size,
      /*is_device_allocation=*/true);
  launch_ctx2.set_arg_int(/*arg_id=*/1, 3);
  (*ker2)(launch_ctx2);
  EXPECT_EQ(array.read_int({0}), 2);
  EXPECT_EQ(array.read_int({1}), 3);
  EXPECT_EQ(array.read_int({2}), 42);
}
}  // namespace lang
}  // namespace taichi
