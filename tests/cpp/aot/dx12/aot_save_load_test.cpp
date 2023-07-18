#include "gtest/gtest.h"
#include "taichi/ir/ir_builder.h"
#include "taichi/ir/statements.h"
#include "taichi/inc/constants.h"
#include "taichi/program/program.h"
#include "tests/cpp/ir/ndarray_kernel.h"
#include "tests/cpp/program/test_program.h"
#include "taichi/aot/graph_data.h"
#include "taichi/program/graph_builder.h"
#include "taichi/runtime/dx12/aot_module_loader_impl.h"
#include "taichi/rhi/dx12/dx12_api.h"
#include "taichi/common/filesystem.hpp"

using namespace taichi;
using namespace lang;
namespace fs = std::filesystem;

[[maybe_unused]] static void aot_save(std::string &tmp_path) {
  default_compile_config.advanced_optimization = false;
  auto program = Program(Arch::dx12);

  int n = 10;

  auto *root = new SNode(0, SNodeType::root);
  auto *pointer = &root->dense(Axis(0), n);
  auto *place = &pointer->insert_children(SNodeType::place);
  place->dt = PrimitiveType::i32;
  program.add_snode_tree(std::unique_ptr<SNode>(root), /*compile_only=*/true);

  auto aot_builder = program.make_aot_module_builder(Arch::dx12, {});

  std::unique_ptr<Kernel> kernel_init, kernel_ret, kernel_simple_ret;

  {
    /*
    @ti.kernel
    def ret() -> ti.f32:
      sum = 0.2
      return sum
    */
    IRBuilder builder;
    auto *sum = builder.create_local_var(PrimitiveType::f32);
    builder.create_local_store(sum, builder.get_float32(0.2));
    builder.create_return(builder.create_local_load(sum));

    kernel_simple_ret =
        std::make_unique<Kernel>(program, builder.extract_ir(), "simple_ret");
    kernel_simple_ret->insert_ret(PrimitiveType::f32);
    kernel_simple_ret->finalize_rets();
  }

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
    kernel_ret->finalize_rets();
  }

  aot_builder->add("simple_ret", kernel_simple_ret.get());
  aot_builder->add_field("place", place, true, place->dt, {n}, 1, 1);
  aot_builder->add("init", kernel_init.get());
  aot_builder->add("ret", kernel_ret.get());
  aot_builder->dump(tmp_path, "");
}

#ifdef TI_WITH_DX12

TEST(AotSaveLoad, DX12) {
  if (!directx12::is_dx12_api_available()) {
    return;
  }

  fs::current_path(fs::temp_directory_path());
  fs::create_directory("dx12_aot");
  fs::permissions("dx12_aot", fs::perms::others_all, fs::perm_options::remove);

  std::string tmp_path = fs::temp_directory_path().append("dx12_aot").string();
  aot_save(tmp_path);

  // Run AOT module loader
  directx12::AotModuleParams mod_params;
  mod_params.module_path = tmp_path;

  std::unique_ptr<aot::Module> module =
      aot::Module::load(Arch::dx12, mod_params);
  EXPECT_TRUE(module);

  // Retrieve kernels/fields/etc from AOT module
  auto root_size = module->get_root_size();
  EXPECT_EQ(root_size, 1);

  auto init_kernel = module->get_kernel("init");
  EXPECT_TRUE(init_kernel);

  auto ret_kernel = module->get_kernel("ret");
  EXPECT_TRUE(ret_kernel);

  auto ret2_kernel = module->get_kernel("ret2");
  EXPECT_FALSE(ret2_kernel);

  // FIXME: test run kernels and check the result.

  fs::remove_all("dx12_aot");
}

#endif
