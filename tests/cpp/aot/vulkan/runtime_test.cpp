#include "gtest/gtest.h"
#define TI_RUNTIME_HOST
#include "taichi/common/core.h"
#include "taichi/program/ndarray.h"
#include "taichi/program/context.h"
#include "taichi/system/memory_pool.h"
#include "taichi/runtime/gfx/runtime.h"
#ifdef TI_WITH_VULKAN
#include "taichi/backends/device.h"
#include "taichi/backends/vulkan/vulkan_device.h"
#include "taichi/backends/vulkan/vulkan_device_creator.h"
#include "taichi/backends/vulkan/vulkan_loader.h"
#include "taichi/backends/vulkan/vulkan_utils.h"
#endif

using namespace taichi;
using namespace lang;

#ifdef TI_WITH_VULKAN
TEST(RuntimeTest, ViewDevAllocAsNdarray) {
  // Otherwise will segfault on macOS VM,
  // where Vulkan is installed but no devices are present
  if (!vulkan::is_vulkan_api_available()) {
    return;
  }

  // API based on proposal https://github.com/taichi-dev/taichi/issues/3642
  // Initialize Vulkan program
  taichi::uint64 *result_buffer{nullptr};
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
  taichi::lang::vulkan::VulkanDevice *device_ =
      static_cast<taichi::lang::vulkan::VulkanDevice *>(
          embedded_device->device());
  // Create Vulkan runtime
  gfx::GfxRuntime::Params params;
  params.host_result_buffer = result_buffer;
  params.device = device_;
  auto vulkan_runtime =
      std::make_unique<taichi::lang::gfx::GfxRuntime>(std::move(params));

  const int size = 40;
  taichi::lang::Device::AllocParams alloc_params;
  alloc_params.host_write = true;
  alloc_params.size = size * sizeof(int);
  alloc_params.usage = taichi::lang::AllocUsage::Storage;
  DeviceAllocation devalloc_arr_ = device_->allocate_memory(alloc_params);

  std::vector<int> element_shape = {4};
  auto arr1 = Ndarray(devalloc_arr_, PrimitiveType::i32, {10}, element_shape);
  EXPECT_TRUE(arr1.element_shape == element_shape);
  EXPECT_EQ(arr1.total_shape()[0], 10);
  EXPECT_EQ(arr1.total_shape()[1], 4);

  auto arr2 = Ndarray(devalloc_arr_, PrimitiveType::i32, {10}, element_shape,
                      ExternalArrayLayout::kSOA);
  EXPECT_TRUE(arr2.element_shape == element_shape);
  EXPECT_EQ(arr2.total_shape()[0], 4);
  EXPECT_EQ(arr2.total_shape()[1], 10);

  device_->dealloc_memory(devalloc_arr_);
}
#endif
