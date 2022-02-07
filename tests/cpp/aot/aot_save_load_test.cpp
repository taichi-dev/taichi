#include "gtest/gtest.h"
#include "taichi/ir/ir_builder.h"
#include "taichi/ir/statements.h"
#include "taichi/program/program.h"
#ifdef TI_WITH_VULKAN
#include "taichi/backends/vulkan/aot_module_loader_impl.h"
#endif

using namespace taichi;
using namespace lang;

static void aot_save() {
  auto program = Program(Arch::vulkan);

  program.config.advanced_optimization = false;

  int n = 10;

  auto *root = new SNode(0, SNodeType::root);
  auto *pointer = &root->dense(Axis(0), n, false);
  auto *place = &pointer->insert_children(SNodeType::place);
  place->dt = PrimitiveType::i32;
  program.add_snode_tree(std::unique_ptr<SNode>(root), /*compile_only=*/true);

  auto aot_builder = program.make_aot_module_builder(Arch::vulkan);

  std::unique_ptr<Kernel> kernel_init, kernel_ret;

  {
    /*
    @ti.kernel
    def init():
      for index in range(n):
        place[index] = index
    */
    IRBuilder builder;
    auto *zero = builder.get_int32(0);
    auto *n_stmt = builder.get_int32(n);
    auto *loop = builder.create_range_for(zero, n_stmt, 1, 0, 4);
    {
      auto _ = builder.get_loop_guard(loop);
      auto *index = builder.get_loop_index(loop);
      auto *ptr = builder.create_global_ptr(place, {index});
      builder.create_global_store(ptr, index);
    }

    kernel_init =
        std::make_unique<Kernel>(program, builder.extract_ir(), "init");
  }

  {
    /*
    @ti.kernel
    def ret():
      sum = 0
      for index in place:
        sum = sum + place[index];
      return sum
    */
    IRBuilder builder;
    auto *sum = builder.create_local_var(PrimitiveType::i32);
    auto *loop = builder.create_struct_for(pointer, 1, 0, 4);
    {
      auto _ = builder.get_loop_guard(loop);
      auto *index = builder.get_loop_index(loop);
      auto *sum_old = builder.create_local_load(sum);
      auto *place_index =
          builder.create_global_load(builder.create_global_ptr(place, {index}));
      builder.create_local_store(sum, builder.create_add(sum_old, place_index));
    }
    builder.create_return(builder.create_local_load(sum));

    kernel_ret = std::make_unique<Kernel>(program, builder.extract_ir(), "ret");
    kernel_ret->insert_ret(PrimitiveType::i32);
  }

  aot_builder->add_field("place", place, true, place->dt, {n}, 1, 1);
  aot_builder->add("init", kernel_init.get());
  aot_builder->add("ret", kernel_ret.get());
  aot_builder->dump(".", "");
}

#ifdef TI_WITH_VULKAN
TEST(AotSaveLoad, Vulkan) {
  aot_save();

  vulkan::AotModuleLoaderImpl aot_loader(".");
  vulkan::VkRuntime::RegisterParams init_kernel, ret_kernel;

  auto ret = aot_loader.get_kernel("init", init_kernel);
  EXPECT_TRUE(ret);

  ret = aot_loader.get_kernel("ret", ret_kernel);
  EXPECT_TRUE(ret);

  ret = aot_loader.get_kernel("ret2", ret_kernel);
  EXPECT_FALSE(ret);

  auto root_size = aot_loader.get_root_size();
  EXPECT_EQ(root_size, 64);
}
#endif
