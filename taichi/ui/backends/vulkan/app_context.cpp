#include "taichi/ui/utils/utils.h"
#include "taichi/ui/backends/vulkan/vulkan_utils.h"
#include "taichi/ui/backends/vulkan/app_context.h"
#include "taichi/ui/backends/vulkan/swap_chain.h"

#include <string_view>

TI_UI_NAMESPACE_BEGIN

namespace vulkan {

using namespace taichi::lang::vulkan;

namespace {
std::vector<std::string> get_required_instance_extensions() {
  uint32_t glfw_ext_count = 0;
  const char **glfw_extensions;
  glfw_extensions = glfwGetRequiredInstanceExtensions(&glfw_ext_count);

  std::vector<std::string> extensions;

  for (int i = 0; i < glfw_ext_count; ++i) {
    extensions.push_back(glfw_extensions[i]);
  }

  // EmbeddedVulkanDevice will check that these are supported
  extensions.push_back(VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME);
  extensions.push_back(VK_KHR_EXTERNAL_MEMORY_CAPABILITIES_EXTENSION_NAME);
  extensions.push_back(VK_KHR_EXTERNAL_SEMAPHORE_CAPABILITIES_EXTENSION_NAME);
  extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);

  return extensions;
}

std::vector<std::string> get_required_device_extensions() {
  static std::vector<std::string> extensions{
      VK_KHR_SWAPCHAIN_EXTENSION_NAME,
      VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME,
      VK_KHR_EXTERNAL_SEMAPHORE_EXTENSION_NAME,
#ifdef _WIN64
      VK_KHR_EXTERNAL_MEMORY_WIN32_EXTENSION_NAME,
      VK_KHR_EXTERNAL_SEMAPHORE_WIN32_EXTENSION_NAME,
#else
      VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME,
      VK_KHR_EXTERNAL_SEMAPHORE_FD_EXTENSION_NAME,
#endif
  };

  return extensions;
}
}  // namespace

void AppContext::init(GLFWwindow *glfw_window) {
  glfw_window_ = glfw_window;
  EmbeddedVulkanDevice::Params evd_params;
  evd_params.additional_instance_extensions =
      get_required_instance_extensions();
  evd_params.additional_device_extensions = get_required_device_extensions();
  evd_params.is_for_ui = true;
  evd_params.surface_creator = [&](VkInstance instance) -> VkSurfaceKHR {
    VkSurfaceKHR surface = VK_NULL_HANDLE;
    if (glfwCreateWindowSurface(instance, glfw_window, nullptr, &surface) !=
        VK_SUCCESS) {
      throw std::runtime_error("failed to create window surface!");
    }
    return surface;
  };
  vulkan_device_ = std::make_unique<EmbeddedVulkanDevice>(evd_params);

  swap_chain_.init(this, vulkan_device_->surface());

  create_render_passes();

  swap_chain_.create_framebuffers();
}

void AppContext::create_render_passes() {
  create_render_pass(render_pass_, swap_chain_.swap_chain_image_format(),
                     VK_IMAGE_LAYOUT_PRESENT_SRC_KHR, device(),
                     physical_device());
}

void AppContext::cleanup_swap_chain() {
  vkDestroyRenderPass(device(), render_pass_, nullptr);
  swap_chain_.cleanup_swap_chain();
}

void AppContext::cleanup() {
  swap_chain_.cleanup();
  vulkan_device_.reset();
}

void AppContext::recreate_swap_chain() {
  create_render_passes();
  swap_chain_.recreate_swap_chain();
}

int AppContext::get_swap_chain_size() {
  return swap_chain_.chain_size();
}

VkInstance AppContext::instance() const {
  return vulkan_device_->instance();
}

VkDevice AppContext::device() const {
  return vulkan_device_->device()->device();
}

VkPhysicalDevice AppContext::physical_device() const {
  return vulkan_device_->physical_device();
}

VulkanQueueFamilyIndices AppContext::queue_family_indices() const {
  return vulkan_device_->queue_family_indices();
}

VkQueue AppContext::graphics_queue() const {
  return vulkan_device_->device()->graphics_queue();
}

VkQueue AppContext::present_queue() const {
  return vulkan_device_->device()->present_queue();
}

VkCommandPool AppContext::command_pool() const {
  return vulkan_device_->device()->command_pool();
}

VkRenderPass AppContext::render_pass() const {
  return render_pass_;
}

const SwapChain &AppContext::swap_chain() const {
  return swap_chain_;
}
SwapChain &AppContext::swap_chain() {
  return swap_chain_;
}
GLFWwindow *AppContext::glfw_window() const {
  return glfw_window_;
}

}  // namespace vulkan

TI_UI_NAMESPACE_END
