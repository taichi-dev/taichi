#include "gtest/gtest.h"

#include "taichi/rhi/vulkan/vulkan_device.h"
#include "taichi/rhi/vulkan/vulkan_device_creator.h"
#include "taichi/rhi/vulkan/vulkan_loader.h"

#include "taichi/runtime/program_impls/vulkan/vulkan_program.h"

// #include "tests/cpp/aot/gfx_utils.h"
#include "tests/cpp/backends/device_test_utils.h"
#include "taichi/system/memory_pool.h"

namespace taichi::lang {

TEST(VulkanDeviceTest, CreateDeviceAndAllocateMemory) {
  // Otherwise will segfault on macOS VM,
  // where Vulkan is installed but no devices are present
  if (!vulkan::is_vulkan_api_available()) {
    return;
  }
  // Create Taichi device for computation
  lang::vulkan::VulkanDeviceCreator::Params evd_params;
  evd_params.api_version = std::nullopt;
  auto embedded_device =
      std::make_unique<taichi::lang::vulkan::VulkanDeviceCreator>(evd_params);

  taichi::lang::vulkan::VulkanDevice *device_ =
      static_cast<taichi::lang::vulkan::VulkanDevice *>(
          embedded_device->device());
  EXPECT_NE(device_, nullptr);

  // Run memory allocation tests
  device_test_utils::test_memory_allocation(device_);
  device_test_utils::test_view_devalloc_as_ndarray(device_);
}

// TEST(VulkanDeviceTest, ViewAllocAsNdarray) {
//   // Otherwise will segfault on macOS VM,
//   // where Vulkan is installed but no devices are present
//   if (!vulkan::is_vulkan_api_available()) {
//     return;
//   }
//   // taichi::lang::vulkan::VulkanDevice *device_ =
//   //     static_cast<taichi::lang::vulkan::VulkanDevice *>(
//   //         create_vulkan_device());
//   // taichi::lang::vulkan::VulkanDevice* device_ = create_vulkan_device();
//   if (device_ == nullptr) {
//     return;
//   }
//   device_test_utils::test_view_devalloc_as_ndarray(device_);
// }

TEST(VulkanDeviceTest, CommandListTest) {
}

TEST(VulkanDeviceTest, MaterializeRuntimeTest) {
  // Otherwise will segfault on macOS VM,
  // where Vulkan is installed but no devices are present
  if (!vulkan::is_vulkan_api_available()) {
    return;
  }
  // Create Taichi device for computation
  lang::vulkan::VulkanDeviceCreator::Params evd_params;
  evd_params.api_version = std::nullopt;
  auto embedded_device =
      std::make_unique<taichi::lang::vulkan::VulkanDeviceCreator>(evd_params);

  taichi::lang::vulkan::VulkanDevice *device_ =
      static_cast<taichi::lang::vulkan::VulkanDevice *>(
          embedded_device->device());
  EXPECT_NE(device_, nullptr);
  std::unique_ptr<MemoryPool> pool =
      std::make_unique<MemoryPool>(Arch::vulkan, device_);
  std::unique_ptr<VulkanProgramImpl> program =
      std::make_unique<VulkanProgramImpl>(default_compile_config);
  uint64_t *result_buffer;
  program->materialize_runtime(pool.get(), nullptr, &result_buffer);

}


// TEST(Dx11ProgramTest, MaterializeRuntimeTest) {
//   std::unique_ptr<directx11::Dx11Device> device_ =
//       std::make_unique<directx11::Dx11Device>();
//   std::unique_ptr<MemoryPool> pool =
//       std::make_unique<MemoryPool>(Arch::dx11, device_.get());
//   std::unique_ptr<Dx11ProgramImpl> program =
//       std::make_unique<Dx11ProgramImpl>(default_compile_config);
//   /*
//   This test needs allocate_memory because of the call stack here:
//   Dx11ProgramImpl::materialize_runtime
//   - GfxRuntime::GfxRuntime
//      - GfxRuntime::init_buffers
//         - Dx11Device::allocate_memory_unique
//         - Dx11Device::get_compute_stream
//         - Dx11Stream::new_command_list
//         - Dx11Stream::buffer_fill
//         - Dx11Stream::submit_synced
//   */
//   uint64_t *result_buffer;
//   program->materialize_runtime(pool.get(), nullptr, &result_buffer);

//   TestProgram test_prog;
//   test_prog.setup();

//   IRBuilder builder;
//   auto *lhs = builder.get_int32(42);

//   auto block = builder.extract_ir();
//   test_prog.prog()->this_thread_config().arch = Arch::dx11;
//   auto ker = std::make_unique<Kernel>(*test_prog.prog(), std::move(block));
//   program->compile(ker.get(), nullptr);
// }

}  // namespace taichi::lang
