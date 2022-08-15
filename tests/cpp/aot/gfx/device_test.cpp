#include "gtest/gtest.h"
#define TI_RUNTIME_HOST
#include "taichi/common/core.h"
#include "taichi/program/ndarray.h"
#include "taichi/program/context.h"
#include "taichi/system/memory_pool.h"
#include "taichi/runtime/gfx/runtime.h"
#ifdef TI_WITH_VULKAN
#include "taichi/rhi/device.h"
#include "taichi/rhi/vulkan/vulkan_device.h"
#include "taichi/rhi/vulkan/vulkan_device_creator.h"
#include "taichi/rhi/vulkan/vulkan_loader.h"
#include "taichi/rhi/vulkan/vulkan_utils.h"
#endif
#ifdef TI_WITH_OPENGL
#include "taichi/rhi/opengl/opengl_api.h"
#endif

using namespace taichi;
using namespace lang;

void view_devalloc_as_ndarray(Device *device_) {
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

#ifdef TI_WITH_VULKAN
TEST(DeviceTest, ViewDevAllocAsNdarray) {
  // Otherwise will segfault on macOS VM,
  // where Vulkan is installed but no devices are present
  if (!vulkan::is_vulkan_api_available()) {
    return;
  }

  // Create Taichi Device for computation
  lang::vulkan::VulkanDeviceCreator::Params evd_params;
  evd_params.api_version = std::nullopt;
  auto embedded_device =
      std::make_unique<taichi::lang::vulkan::VulkanDeviceCreator>(evd_params);
  taichi::lang::vulkan::VulkanDevice *device_ =
      static_cast<taichi::lang::vulkan::VulkanDevice *>(
          embedded_device->device());

  view_devalloc_as_ndarray(device_);
}
#endif

#ifdef TI_WITH_OPENGL
TEST(DeviceTest, GLDevice) {
  if (!opengl::is_opengl_api_available()) {
    return;
  }

  auto device_ = taichi::lang::opengl::make_opengl_device();

  view_devalloc_as_ndarray(device_.get());
}
#endif
