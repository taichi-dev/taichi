#pragma once

#ifdef TI_WITH_VULKAN
#include "taichi/taichi_vulkan.h"
#include "taichi/runtime/vulkan/runtime.h"
#include "taichi/backends/vulkan/vulkan_device.h"
#include "taichi/backends/vulkan/vulkan_device_creator.h"

#include "taichi_core_impl.h"

class VulkanDevice;
class VulkanDeviceImported;
class VulkanDeviceOwned;
class VulkanContext;

class VulkanDevice : public Device {
 protected:
  VulkanDevice();

 public:
  taichi::lang::vulkan::VulkanDevice &get_vk();

  virtual Context *create_context() override final;
};
class VulkanDeviceImported : public VulkanDevice {
  taichi::lang::vulkan::VulkanDevice vk_device_;

 public:
  VulkanDeviceImported(
      const taichi::lang::vulkan::VulkanDevice::Params &params);

  virtual taichi::lang::Device &get() override final;
};
class VulkanDeviceOwned : public VulkanDevice {
  taichi::lang::vulkan::VulkanDeviceCreator vk_device_creator_;

 public:
  VulkanDeviceOwned();
  VulkanDeviceOwned(
      const taichi::lang::vulkan::VulkanDeviceCreator::Params &params);

  virtual taichi::lang::Device &get() override final;
};

class VulkanContext : public Context {
  // 32 is a magic number in `taichi/inc/constants.h`.
  std::array<uint64_t, 32> host_result_buffer_;
  taichi::lang::vulkan::VkRuntime vk_runtime_;

 public:
  VulkanContext(VulkanDevice &device);
  virtual ~VulkanContext() override final;

  taichi::lang::vulkan::VkRuntime &get_vk();
};

#endif  // TI_WITH_VULKAN
