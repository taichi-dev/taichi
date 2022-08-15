#include "gtest/gtest.h"
#include "taichi/ir/ir_builder.h"
#include "taichi/ir/statements.h"
#include "taichi/inc/constants.h"
#include "taichi/program/program.h"
#include "tests/cpp/program/test_program.h"
#include "taichi/aot/graph_data.h"
#include "taichi/program/graph_builder.h"
#include "taichi/runtime/gfx/aot_module_loader_impl.h"
#include "tests/cpp/aot/gfx/utils.h"
#ifdef TI_WITH_VULKAN
#include "taichi/rhi/device.h"
#include "taichi/rhi/vulkan/vulkan_device.h"
#include "taichi/rhi/vulkan/vulkan_device_creator.h"
#include "taichi/rhi/vulkan/vulkan_loader.h"
#include "taichi/rhi/vulkan/vulkan_utils.h"
#endif

using namespace taichi;
using namespace lang;

#ifdef TI_WITH_VULKAN

TEST(GfxAotTest, VulkanDenseField) {
  // Otherwise will segfault on macOS VM,
  // where Vulkan is installed but no devices are present
  if (!vulkan::is_vulkan_api_available()) {
    return;
  }

  // API based on proposal https://github.com/taichi-dev/taichi/issues/3642
  // Initialize Vulkan program
  taichi::uint64 *result_buffer{nullptr};
  taichi::lang::RuntimeContext host_ctx;
  auto memory_pool =
      std::make_unique<taichi::lang::MemoryPool>(Arch::vulkan, nullptr);
  result_buffer = (taichi::uint64 *)memory_pool->allocate(
      sizeof(taichi::uint64) * taichi_result_buffer_entries, 8);
  host_ctx.result_buffer = result_buffer;

  // Create Taichi Device for computation
  lang::vulkan::VulkanDeviceCreator::Params evd_params;
  evd_params.api_version = std::nullopt;
  auto embedded_device =
      std::make_unique<taichi::lang::vulkan::VulkanDeviceCreator>(evd_params);

  // Create Vulkan runtime
  gfx::GfxRuntime::Params params;
  params.host_result_buffer = result_buffer;
  params.device = embedded_device->device();
  auto vulkan_runtime =
      std::make_unique<taichi::lang::gfx::GfxRuntime>(std::move(params));

  // Run AOT module loader
  const auto folder_dir = getenv("TAICHI_AOT_FOLDER_PATH");
  std::stringstream ss;
  ss << folder_dir;
  gfx::AotModuleParams mod_params;
  mod_params.module_path = ss.str();
  mod_params.runtime = vulkan_runtime.get();

  std::unique_ptr<aot::Module> vk_module =
      aot::Module::load(Arch::vulkan, mod_params);
  EXPECT_TRUE(vk_module);

  // Retrieve kernels/fields/etc from AOT module
  auto root_size = vk_module->get_root_size();
  EXPECT_EQ(root_size, 64);
  vulkan_runtime->add_root_buffer(root_size);

  auto simple_ret_kernel = vk_module->get_kernel("simple_ret");
  EXPECT_TRUE(simple_ret_kernel);

  simple_ret_kernel->launch(&host_ctx);
  vulkan_runtime->synchronize();
  EXPECT_FLOAT_EQ(host_ctx.get_ret<float>(0), 0.2);

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
  auto x_field = vk_module->get_snode_tree("place");
  EXPECT_NE(x_field, nullptr);
}

TEST(GfxAotTest, VulkanNdarray) {
  // Otherwise will segfault on macOS VM,
  // where Vulkan is installed but no devices are present
  if (!vulkan::is_vulkan_api_available()) {
    return;
  }

  // save_ndarray_kernels(Arch::vulkan);

  // API based on proposal https://github.com/taichi-dev/taichi/issues/3642
  // Initialize Vulkan program
  taichi::uint64 *result_buffer{nullptr};
  taichi::lang::RuntimeContext host_ctx;
  auto memory_pool =
      std::make_unique<taichi::lang::MemoryPool>(Arch::vulkan, nullptr);
  result_buffer = (taichi::uint64 *)memory_pool->allocate(
      sizeof(taichi::uint64) * taichi_result_buffer_entries, 8);
  host_ctx.result_buffer = result_buffer;

  // Create Taichi Device for computation
  lang::vulkan::VulkanDeviceCreator::Params evd_params;
  evd_params.api_version = std::nullopt;
  auto embedded_device =
      std::make_unique<taichi::lang::vulkan::VulkanDeviceCreator>(evd_params);

  // Create Vulkan runtime
  gfx::GfxRuntime::Params params;
  params.host_result_buffer = result_buffer;
  params.device = embedded_device->device();
  auto vulkan_runtime =
      std::make_unique<taichi::lang::gfx::GfxRuntime>(std::move(params));

  // Run AOT module loader
  const auto folder_dir = getenv("TAICHI_AOT_FOLDER_PATH");
  std::stringstream ss;
  ss << folder_dir;
  gfx::AotModuleParams mod_params;
  mod_params.module_path = ss.str();
  mod_params.runtime = vulkan_runtime.get();

  std::unique_ptr<aot::Module> vk_module =
      aot::Module::load(Arch::vulkan, mod_params);
  EXPECT_TRUE(vk_module);

  // Retrieve kernels/fields/etc from AOT module
  auto root_size = vk_module->get_root_size();
  EXPECT_EQ(root_size, 0);
  vulkan_runtime->add_root_buffer(root_size);

  auto ker1 = vk_module->get_kernel("ker1");
  EXPECT_TRUE(ker1);

  const int size = 10;
  taichi::lang::Device::AllocParams alloc_params;
  alloc_params.host_write = true;
  alloc_params.host_read = true;
  alloc_params.size = size * sizeof(int);
  alloc_params.usage = taichi::lang::AllocUsage::Storage;
  DeviceAllocation devalloc_arr_ =
      embedded_device->device()->allocate_memory(alloc_params);
  Ndarray arr = Ndarray(devalloc_arr_, PrimitiveType::i32, {size});
  host_ctx.set_arg_ndarray(0, arr.get_device_allocation_ptr_as_int(),
                           arr.shape);
  int src[size] = {0};
  src[0] = 2;
  src[2] = 40;
  write_devalloc(devalloc_arr_, src, sizeof(src));
  ker1->launch(&host_ctx);
  vulkan_runtime->synchronize();

  int dst[size] = {33};
  load_devalloc(devalloc_arr_, dst, sizeof(dst));
  EXPECT_EQ(dst[0], 2);
  EXPECT_EQ(dst[1], 1);
  EXPECT_EQ(dst[2], 42);

  auto ker2 = vk_module->get_kernel("ker2");
  EXPECT_TRUE(ker2);

  host_ctx.set_arg(1, 3);
  ker2->launch(&host_ctx);
  vulkan_runtime->synchronize();
  load_devalloc(devalloc_arr_, dst, sizeof(dst));
  EXPECT_EQ(dst[0], 2);
  EXPECT_EQ(dst[1], 3);
  EXPECT_EQ(dst[2], 42);

  // Deallocate
  embedded_device->device()->dealloc_memory(devalloc_arr_);
}

#endif
