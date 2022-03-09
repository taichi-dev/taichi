#include "gtest/gtest.h"
#include "taichi/ir/ir_builder.h"
#include "taichi/ir/statements.h"
#include "taichi/program/program.h"
#ifdef TI_WITH_VULKAN
#include "taichi/backends/vulkan/aot_module_loader_impl.h"
#include "taichi/backends/device.h"
#include "taichi/backends/vulkan/vulkan_device.h"
#include "taichi/backends/vulkan/vulkan_device_creator.h"
#include "taichi/backends/vulkan/vulkan_loader.h"
#include "taichi/backends/vulkan/vulkan_utils.h"
#endif

using namespace taichi;
using namespace lang;

[[maybe_unused]] static void aot_save() {
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
  // Otherwise will segfault on macOS VM,
  // where Vulkan is installed but no devices are present
  if (!vulkan::is_vulkan_api_available()) {
    return;
  }

  aot_save();

  // API based on proposal https://github.com/taichi-dev/taichi/issues/3642
  // Initialize Vulkan program
  taichi::uint64 *result_buffer{nullptr};
  taichi::lang::RuntimeContext host_ctx;
  auto memory_pool =
      std::make_unique<taichi::lang::MemoryPool>(Arch::vulkan, nullptr);
  result_buffer = (taichi::uint64 *)memory_pool->allocate(
      sizeof(taichi::uint64) * taichi_result_buffer_entries, 8);

  // Create Taichi Device for computation
  lang::vulkan::VulkanDeviceCreator::Params evd_params;
  evd_params.api_version =
      taichi::lang::vulkan::VulkanEnvSettings::kApiVersion();
  auto embedded_device =
      std::make_unique<taichi::lang::vulkan::VulkanDeviceCreator>(evd_params);

  // Create Vulkan runtime
  vulkan::VkRuntime::Params params;
  params.host_result_buffer = result_buffer;
  params.device = embedded_device->device();
  auto vulkan_runtime =
      std::make_unique<taichi::lang::vulkan::VkRuntime>(std::move(params));

  // Run AOT module loader
  vulkan::AotModuleParams mod_params;
  mod_params.module_path = ".";
  mod_params.runtime = vulkan_runtime.get();

  std::unique_ptr<aot::Module> vk_module =
      aot::Module::load(".", Arch::vulkan, mod_params);
  EXPECT_TRUE(vk_module);

  // Retrieve kernels/fields/etc from AOT module
  auto root_size = vk_module->get_root_size();
  EXPECT_EQ(root_size, 64);
  vulkan_runtime->add_root_buffer(root_size);

  auto init_kernel = vk_module->get_kernel("init");
  EXPECT_TRUE(init_kernel);

  auto ret_kernel = vk_module->get_kernel("ret");
  EXPECT_TRUE(ret_kernel);

  auto ret2_kernel = vk_module->get_kernel("ret2");
  EXPECT_FALSE(ret2_kernel);

  // Run kernels
  init_kernel->launch(&host_ctx);
  ret_kernel->launch(&host_ctx);
  vulkan_runtime->synchronize();

  // Retrieve data
  auto x_field = vk_module->get_field("place");
  EXPECT_TRUE(x_field);

  std::vector<uint64_t> x(root_size);
  x_field->copy_to_host_buffer(&x[0]);
}
#endif
