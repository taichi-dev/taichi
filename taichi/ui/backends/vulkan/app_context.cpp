#include "taichi/ui/utils/utils.h"
#include "taichi/ui/backends/vulkan/vulkan_utils.h"
#include "taichi/ui/backends/vulkan/app_context.h"
#include "taichi/ui/backends/vulkan/swap_chain.h"

TI_UI_NAMESPACE_BEGIN

namespace vulkan {

using namespace taichi::lang::vulkan;

const std::vector<const char *> device_extensions = {
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

std::vector<const char *> get_required_instance_extensions() {
  uint32_t glfw_ext_count = 0;
  const char **glfw_extensions;
  glfw_extensions = glfwGetRequiredInstanceExtensions(&glfw_ext_count);

  std::vector<const char *> extensions(glfw_extensions,
                                       glfw_extensions + glfw_ext_count);

  extensions.push_back(VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME);
  extensions.push_back(VK_KHR_EXTERNAL_MEMORY_CAPABILITIES_EXTENSION_NAME);
  extensions.push_back(VK_KHR_EXTERNAL_SEMAPHORE_CAPABILITIES_EXTENSION_NAME);
  extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);

  return extensions;
}

void AppContext::init() {
  EmbeddedVulkanDevice::Params evd_params;
  evd_params.additional_instance_extensions =
      get_required_instance_extensions();
  evd_params.additional_device_extensions = device_extensions;
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

  swap_chain.app_context = this;
  swap_chain.surface = vulkan_device_->surface();

  swap_chain.create_swap_chain();
  swap_chain.create_image_views();
  swap_chain.create_depth_resources();
  swap_chain.create_sync_objects();

  create_render_passes();

  swap_chain.create_framebuffers();
}

void AppContext::create_render_passes() {
  create_render_pass(render_pass_, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR);
}

void AppContext::create_render_pass(VkRenderPass &render_pass,
                                    VkImageLayout final_color_layout) {
  VkFormat swap_chain_image_format = swap_chain.swap_chain_image_format;

  VkAttachmentDescription color_attachment{};
  color_attachment.format = swap_chain_image_format;
  color_attachment.samples = VK_SAMPLE_COUNT_1_BIT;
  color_attachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
  color_attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
  color_attachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
  color_attachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
  color_attachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  color_attachment.finalLayout = final_color_layout;

  VkAttachmentDescription depth_attachment{};
  depth_attachment.format = find_depth_format(physical_device());
  depth_attachment.samples = VK_SAMPLE_COUNT_1_BIT;
  depth_attachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
  depth_attachment.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
  depth_attachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
  depth_attachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
  depth_attachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  depth_attachment.finalLayout =
      VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

  VkAttachmentReference color_attachment_ref{};
  color_attachment_ref.attachment = 0;
  color_attachment_ref.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

  VkAttachmentReference depth_attachment_ref{};
  depth_attachment_ref.attachment = 1;
  depth_attachment_ref.layout =
      VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

  VkSubpassDescription subpass{};
  subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
  subpass.colorAttachmentCount = 1;
  subpass.pColorAttachments = &color_attachment_ref;
  subpass.pDepthStencilAttachment = &depth_attachment_ref;

  VkSubpassDependency dependency{};
  dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
  dependency.dstSubpass = 0;
  dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
                            VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
  dependency.srcAccessMask = 0;
  dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
                            VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
  dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT |
                             VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;

  std::array<VkAttachmentDescription, 2> attachments = {color_attachment,
                                                        depth_attachment};
  VkRenderPassCreateInfo render_pass_info{};
  render_pass_info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
  render_pass_info.attachmentCount = static_cast<uint32_t>(attachments.size());
  render_pass_info.pAttachments = attachments.data();
  render_pass_info.subpassCount = 1;
  render_pass_info.pSubpasses = &subpass;
  render_pass_info.dependencyCount = 1;
  render_pass_info.pDependencies = &dependency;

  if (vkCreateRenderPass(device(), &render_pass_info, nullptr, &render_pass) !=
      VK_SUCCESS) {
    throw std::runtime_error("failed to create render pass!");
  }
}

void AppContext::cleanup_swap_chain() {
  vkDestroyRenderPass(device(), render_pass_, nullptr);
  swap_chain.cleanup_swap_chain();
}

void AppContext::cleanup() {
  swap_chain.cleanup();
  vulkan_device_.reset();
}

void AppContext::recreate_swap_chain() {
  create_render_passes();
  swap_chain.recreate_swap_chain();
}

int AppContext::get_swap_chain_size() {
  return swap_chain.swap_chain_images.size();
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

}  // namespace vulkan

TI_UI_NAMESPACE_END
