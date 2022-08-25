#include "gtest/gtest.h"

#include "taichi/ir/analysis.h"
#include "taichi/ir/ir_builder.h"
#include "taichi/ir/statements.h"
#include "taichi/ir/transforms.h"
#include "taichi/program/program.h"

namespace taichi {
namespace lang {

class InliningTest : public ::testing::Test {
 protected:
  void SetUp() override {
    prog_ = std::make_unique<Program>();
    prog_->materialize_runtime();
  }

  std::unique_ptr<Program> prog_;
};

TEST_F(InliningTest, ArgLoadOfArgLoad) {
  IRBuilder builder;
  // def test_func(x: ti.i32) -> ti.i32:
  //     return x + 1
  auto *arg = builder.create_arg_load(/*arg_id=*/0, get_data_type<int>(),
                                      /*is_ptr=*/false);
  auto *sum = builder.create_add(arg, builder.get_int32(1));
  builder.create_return(sum);
  auto func_body = builder.extract_ir();
  EXPECT_TRUE(func_body->is<Block>());
  auto *func_block = func_body->as<Block>();
  EXPECT_EQ(func_block->size(), 4);

  auto *func = prog_->create_function(
      FunctionKey("test_func", /*func_id=*/0, /*instance_id=*/0));
  func->insert_scalar_arg(get_data_type<int>());
  func->insert_ret(get_data_type<int>());
  func->set_function_body(std::move(func_body));

  // def kernel(x: ti.i32) -> ti.i32:
  //     return test_func(x)
  auto *kernel_arg = builder.create_arg_load(/*arg_id=*/0, get_data_type<int>(),
                                             /*is_ptr=*/false);
  auto *func_call = builder.create_func_call(func, {kernel_arg});
  builder.create_return(func_call);
  auto kernel_body = builder.extract_ir();
  EXPECT_TRUE(kernel_body->is<Block>());
  auto *kernel_block = kernel_body->as<Block>();
  EXPECT_EQ(kernel_block->size(), 3);
  irpass::type_check(kernel_block, CompileConfig());

  irpass::inlining(kernel_block, CompileConfig(), {});
  irpass::full_simplify(kernel_block, CompileConfig(),
                        {false, false, prog_.get()});

  EXPECT_EQ(kernel_block->size(), 4);
  EXPECT_TRUE(irpass::analysis::same_statements(func_block, kernel_block));
}

}  // namespace lang
}  // namespace taichi
