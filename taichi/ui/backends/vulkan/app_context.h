#pragma once
#include "taichi/ui/common/app_config.h"
#include <memory>
#include "taichi/backends/vulkan/vulkan_device_creator.h"
#include "taichi/backends/vulkan/vulkan_loader.h"
#include "taichi/backends/vulkan/vulkan_device.h"
#include "taichi/ui/backends/vulkan/swap_chain.h"
#ifdef ANDROID
#include <android/native_window.h>
#endif

namespace taichi {
namespace lang {
class Program;
}  // namespace lang
}  // namespace taichi

TI_UI_NAMESPACE_BEGIN

#ifdef ANDROID
using TaichiWindow = ANativeWindow;
#else
using TaichiWindow = GLFWwindow;
#endif

namespace vulkan {

class TI_DLL_EXPORT AppContext {
 public:
  void init(lang::Program *prog, TaichiWindow *window, const AppConfig &config);
  void cleanup();

  TaichiWindow *taichi_window() const;
  lang::Program *prog() const;

  taichi::lang::vulkan::VulkanDevice &device();
  const taichi::lang::vulkan::VulkanDevice &device() const;
  bool requires_export_sharing() const;

  AppConfig config;

 private:
  std::unique_ptr<taichi::lang::vulkan::VulkanDeviceCreator>
      embedded_vulkan_device_{nullptr};

  // not owned
  taichi::lang::vulkan::VulkanDevice *vulkan_device_{nullptr};

  TaichiWindow *taichi_window_{nullptr};

  lang::Program *prog_{nullptr};
};

}  // namespace vulkan

TI_UI_NAMESPACE_END
