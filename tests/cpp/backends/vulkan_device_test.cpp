#include "gtest/gtest.h"

#ifdef TI_WITH_VULKAN

#include "taichi/rhi/vulkan/vulkan_loader.h"
#include "taichi/runtime/program_impls/vulkan/vulkan_program.h"

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

  // Run memory allocation tests
  device_test_utils::test_memory_allocation(embedded_device->device());
  device_test_utils::test_view_devalloc_as_ndarray(embedded_device->device());
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

  Device* device_ = embedded_device->device();

  // Materialize runtime
  std::unique_ptr<MemoryPool> pool =
      std::make_unique<MemoryPool>(Arch::vulkan, device_);
  std::unique_ptr<VulkanProgramImpl> program =
      std::make_unique<VulkanProgramImpl>(default_compile_config);
  uint64_t *result_buffer;
  program->materialize_runtime(pool.get(), nullptr, &result_buffer);

  device_test_utils::test_program(program.get(), Arch::vulkan);
}
}  // namespace taichi::lang

#endif