#pragma once

#ifdef TI_WITH_VULKAN
#include "taichi/taichi_vulkan.h"
#include "taichi/runtime/vulkan/runtime.h"
#include "taichi/backends/vulkan/vulkan_device.h"
#include "taichi/backends/vulkan/vulkan_device_creator.h"

#include "taichi_core_impl.h"

class VulkanRuntime;
class VulkanRuntimeImported;
class VulkanRuntimeOwned;
class VulkanContext;

class VulkanRuntime : public Runtime {
 public:
  VulkanRuntime();

  taichi::lang::vulkan::VulkanDevice &get_vk();
  virtual taichi::lang::vulkan::VkRuntime &get_gfx_runtime() = 0;

  virtual TiAotModule load_aot_module(const char *module_path) override final;
  virtual void submit() override final;
  virtual void wait() override final;
};
class VulkanRuntimeImported : public VulkanRuntime {
  // A dirty workaround to ensure the device is fully initialized before
  // construction of `gfx_runtime_`.
  struct Workaround {
    taichi::lang::vulkan::VulkanDevice vk_device;
    Workaround(uint32_t api_version,
               const taichi::lang::vulkan::VulkanDevice::Params &params);
  } inner_;
  taichi::lang::vulkan::VkRuntime gfx_runtime_;

 public:
  VulkanRuntimeImported(
      uint32_t api_version,
      const taichi::lang::vulkan::VulkanDevice::Params &params);

  virtual taichi::lang::Device &get() override final;
  virtual taichi::lang::vulkan::VkRuntime &get_gfx_runtime() override final;
};
class VulkanRuntimeOwned : public VulkanRuntime {
  taichi::lang::vulkan::VulkanDeviceCreator vk_device_creator_;
  taichi::lang::vulkan::VkRuntime gfx_runtime_;

 public:
  VulkanRuntimeOwned();
  VulkanRuntimeOwned(
      const taichi::lang::vulkan::VulkanDeviceCreator::Params &params);

  virtual taichi::lang::Device &get() override final;
  virtual taichi::lang::vulkan::VkRuntime &get_gfx_runtime() override final;
};

#endif  // TI_WITH_VULKAN
