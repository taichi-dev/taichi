#pragma once

#define VK_NO_PROTOTYPES
#include "taichi/taichi_vulkan.h"
#include "taichi/rhi/vulkan/vulkan_device.h"
#include "taichi/rhi/vulkan/vulkan_device_creator.h"

#include "taichi_core_impl.h"
#include "taichi_gfx_impl.h"

class VulkanRuntime;
class VulkanRuntimeImported;
class VulkanRuntimeOwned;
class VulkanContext;

class VulkanRuntime : public GfxRuntime {
 public:
  VulkanRuntime();

  taichi::lang::vulkan::VulkanDevice &get_vk();
  virtual TiTexture allocate_texture(
      const taichi::lang::ImageParams &params) override final;
  virtual void free_texture(TiTexture texture) override final;
};
class VulkanRuntimeImported : public VulkanRuntime {
  // A dirty workaround to ensure the device is fully initialized before
  // construction of `gfx_runtime_`.
  struct Workaround {
    taichi::lang::vulkan::VulkanDevice vk_device;
    Workaround(uint32_t api_version,
               const taichi::lang::vulkan::VulkanDevice::Params &params);
  } inner_;
  taichi::lang::gfx::GfxRuntime gfx_runtime_;

 public:
  VulkanRuntimeImported(
      uint32_t api_version,
      const taichi::lang::vulkan::VulkanDevice::Params &params);

  virtual taichi::lang::Device &get() override final;
  virtual taichi::lang::gfx::GfxRuntime &get_gfx_runtime() override final;
};
class VulkanRuntimeOwned : public VulkanRuntime {
  taichi::lang::vulkan::VulkanDeviceCreator vk_device_creator_;
  taichi::lang::gfx::GfxRuntime gfx_runtime_;

 public:
  VulkanRuntimeOwned();
  VulkanRuntimeOwned(
      const taichi::lang::vulkan::VulkanDeviceCreator::Params &params);

  virtual taichi::lang::Device &get() override final;
  virtual taichi::lang::gfx::GfxRuntime &get_gfx_runtime() override final;
};
