#include "gtest/gtest.h"

#include "taichi/ir/ir_builder.h"
#include "taichi/ir/statements.h"
#include "taichi/program/program.h"

namespace taichi {
namespace lang {

TEST(IRBuilder, RunSnode) {
  auto program = Program(arch_from_name("x64"));
  /*CompileConfig config_print_ir;
  config_print_ir.print_ir = true;
  prog_.config = config_print_ir;*/  // print_ir = True

  int n = 10;
  auto *pointer = &program.snode_root.get()->pointer(Index(0), n);
  auto *place = &pointer->insert_children(SNodeType::place);
  place->dt = PrimitiveType::i32;

  program.materialize_layout();

  std::unique_ptr<Kernel> kernel_init, kernel_return, kernel_ext;

  {
    IRBuilder builder;
    auto *zero = builder.get_int32(0);
    auto *n_stmt = builder.get_int32(n);
    auto *loop = builder.create_range_for(zero, n_stmt, 1, 0, 4);  // for index in range(0, n):
    {
      auto _ = builder.get_loop_guard(loop);
      auto *index = builder.get_loop_index(loop);
      auto *ptr = builder.create_global_ptr(place, {index});
      builder.create_global_store(ptr, index);  // place[index] = index
    }

    kernel_init = std::make_unique<Kernel>(program, builder.extract_ir(), "init");
  }

  {
    IRBuilder builder;
    auto *sum = builder.create_local_var(PrimitiveType::i32);
    auto *loop = builder.create_struct_for(pointer, 1, 0, 4); // for index in pointer:
    {
      auto _ = builder.get_loop_guard(loop);
      auto *index = builder.get_loop_index(loop);
      auto *sum_old = builder.create_local_load(sum);
      auto *place_index = builder.create_global_load(builder.create_global_ptr(place, {index}));
      builder.create_local_store(sum, builder.create_add(sum_old, place_index));
    }
    builder.create_return(builder.create_local_load(sum));

    kernel_return = std::make_unique<Kernel>(program, builder.extract_ir(), "return");
  }

  {
    IRBuilder builder;
    auto *loop = builder.create_struct_for(pointer, 1, 0, 4);
    {
      auto _ = builder.get_loop_guard(loop);
      auto *index = builder.get_loop_index(loop);
      auto *ext = builder.create_external_ptr(builder.create_arg_load(0, PrimitiveType::i32, true), {index});
      auto *place_index = builder.create_global_load(builder.create_global_ptr(place, {index}));
      builder.create_global_store(ext, place_index); // ext[i] = place[i]
    }

    kernel_ext = std::make_unique<Kernel>(program, builder.extract_ir(), "ext");
    kernel_ext->insert_arg(PrimitiveType::gen, true);
  }

  auto ctx_init = kernel_init->make_launch_context();
  auto ctx_return = kernel_return->make_launch_context();
  auto ctx_ext = kernel_ext->make_launch_context();
  int ext[n];
  ctx_ext.set_arg_external_array(0, taichi::uint64(ext), 0);

  (*kernel_init)(ctx_init);
  (*kernel_return)(ctx_return);
  EXPECT_EQ(program.fetch_result<int>(0), n * (n - 1) / 2);
  (*kernel_ext)(ctx_ext);
  for (int i = 0; i < n; i++) EXPECT_EQ(ext[i], i);
}

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
  auto prog = Program(arch_from_name("x64"));
  prog.materialize_layout();
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
  auto ker = std::make_unique<Kernel>(prog, std::move(block));
  ker->insert_arg(get_data_type<int>(), /*is_external_array=*/true);
  auto launch_ctx = ker->make_launch_context();
  launch_ctx.set_arg_external_array(/*arg_id=*/0, (uint64)array.get(), size);
  (*ker)(launch_ctx);
  EXPECT_EQ(array[0], 2);
  EXPECT_EQ(array[1], 1);
  EXPECT_EQ(array[2], 42);
}
}  // namespace lang
}  // namespace taichi
