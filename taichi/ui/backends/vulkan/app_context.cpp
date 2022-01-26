#include "taichi/ui/utils/utils.h"
#include "taichi/ui/backends/vulkan/app_context.h"
#include "taichi/ui/backends/vulkan/swap_chain.h"
#include "taichi/program/program.h"

#include <string_view>

TI_UI_NAMESPACE_BEGIN

namespace vulkan {

using namespace taichi::lang::vulkan;
using namespace taichi::lang;

namespace {
std::vector<std::string> get_required_instance_extensions() {
#ifdef ANDROID
  std::vector<std::string> extensions;

  extensions.push_back(VK_KHR_SURFACE_EXTENSION_NAME);
  extensions.push_back(VK_KHR_ANDROID_SURFACE_EXTENSION_NAME);
  extensions.push_back(VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME);

  return extensions;
#else
  uint32_t glfw_ext_count = 0;
  const char **glfw_extensions;
  glfw_extensions = glfwGetRequiredInstanceExtensions(&glfw_ext_count);

  std::vector<std::string> extensions;

  for (int i = 0; i < glfw_ext_count; ++i) {
    extensions.push_back(glfw_extensions[i]);
  }

  // VulkanDeviceCreator will check that these are supported
  extensions.push_back(VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME);
  extensions.push_back(VK_KHR_EXTERNAL_MEMORY_CAPABILITIES_EXTENSION_NAME);
  extensions.push_back(VK_KHR_EXTERNAL_SEMAPHORE_CAPABILITIES_EXTENSION_NAME);
  extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);

  return extensions;
#endif
}

std::vector<std::string> get_required_device_extensions() {
  static std::vector<std::string> extensions {
    VK_KHR_SWAPCHAIN_EXTENSION_NAME,
#if !defined(ANDROID)
        VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME,
        VK_KHR_EXTERNAL_SEMAPHORE_EXTENSION_NAME,
#ifdef _WIN64
        VK_KHR_EXTERNAL_MEMORY_WIN32_EXTENSION_NAME,
        VK_KHR_EXTERNAL_SEMAPHORE_WIN32_EXTENSION_NAME,
#else
        VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME,
        VK_KHR_EXTERNAL_SEMAPHORE_FD_EXTENSION_NAME,
#endif
#endif
  };

  return extensions;
}
}  // namespace

void AppContext::init(Program *prog,
                      TaichiWindow *window,
                      const AppConfig &config) {
  taichi_window_ = window;
  prog_ = prog;
  this->config = config;

  // Create a Vulkan device if the original configuration is not for Vulkan or
  // there is no active current program (usage from external library for AOT
  // modules for example).
  if (config.ti_arch != Arch::vulkan || prog == nullptr) {
    VulkanDeviceCreator::Params evd_params;
    evd_params.additional_instance_extensions =
        get_required_instance_extensions();
    evd_params.additional_device_extensions = get_required_device_extensions();
    evd_params.is_for_ui = config.show_window;
    evd_params.surface_creator = [&](VkInstance instance) -> VkSurfaceKHR {
      VkSurfaceKHR surface = VK_NULL_HANDLE;
#ifdef ANDROID
      VkAndroidSurfaceCreateInfoKHR createInfo{
          .sType = VK_STRUCTURE_TYPE_ANDROID_SURFACE_CREATE_INFO_KHR,
          .pNext = nullptr,
          .flags = 0,
          .window = window};

      vkCreateAndroidSurfaceKHR(instance, &createInfo, nullptr, &surface);
#else
      if (glfwCreateWindowSurface(instance, window, nullptr, &surface) !=
          VK_SUCCESS) {
        throw std::runtime_error("failed to create window surface!");
      }
#endif
      return surface;
    };
    embedded_vulkan_device_ = std::make_unique<VulkanDeviceCreator>(evd_params);
  } else {
    vulkan_device_ = static_cast<VulkanDevice *>(prog->get_graphics_device());
  }
}

taichi::lang::vulkan::VulkanDevice &AppContext::device() {
  if (vulkan_device_) {
    return *vulkan_device_;
  }
  return *(embedded_vulkan_device_->device());
}

const taichi::lang::vulkan::VulkanDevice &AppContext::device() const {
  if (vulkan_device_) {
    return *vulkan_device_;
  }
  return *(embedded_vulkan_device_->device());
}

void AppContext::cleanup() {
  if (embedded_vulkan_device_) {
    embedded_vulkan_device_.reset();
  }
}

bool AppContext::requires_export_sharing() const {
  // only the cuda backends needs export_sharing to interop with vk
  // with other backends (e.g. vulkan backend on mac), turning export_sharing to
  // true leads to crashes
  // TODO: investigate this, and think of a more universal solution.
  return config.ti_arch == Arch::cuda;
}

TaichiWindow *AppContext::taichi_window() const {
  return taichi_window_;
}

lang::Program *AppContext::prog() const {
  return prog_;
}

}  // namespace vulkan

TI_UI_NAMESPACE_END
