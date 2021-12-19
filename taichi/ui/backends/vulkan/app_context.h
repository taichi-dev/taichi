#pragma once
#include "taichi/ui/common/app_config.h"
#include <memory>
#include "taichi/backends/vulkan/vulkan_device_creator.h"
#include "taichi/backends/vulkan/vulkan_loader.h"
#include "taichi/backends/vulkan/vulkan_device.h"
#include "taichi/ui/backends/vulkan/swap_chain.h"

TI_UI_NAMESPACE_BEGIN

namespace vulkan {

class AppContext {
 public:
  void init(GLFWwindow *glfw_window, const AppConfig &config);
  void cleanup();

  GLFWwindow *glfw_window() const;

  taichi::lang::vulkan::VulkanDevice &device();
  const taichi::lang::vulkan::VulkanDevice &device() const;
  bool requires_export_sharing() const;

  AppConfig config;

 private:
  std::unique_ptr<taichi::lang::vulkan::VulkanDeviceCreator>
      embedded_vulkan_device_{nullptr};

  // not owned
  taichi::lang::vulkan::VulkanDevice *vulkan_device_{nullptr};

  GLFWwindow *glfw_window_{nullptr};
};

}  // namespace vulkan

TI_UI_NAMESPACE_END
