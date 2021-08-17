#include "taichi/ui/utils/utils.h"
#include "taichi/ui/backends/vulkan/vulkan_utils.h"
#include "taichi/ui/backends/vulkan/app_context.h"
#include "taichi/ui/backends/vulkan/swap_chain.h"

TI_UI_NAMESPACE_BEGIN

namespace vulkan {

using namespace taichi::lang::vulkan;

void SwapChain::init(class AppContext *app_context, VkSurfaceKHR surface) {
  app_context_ = app_context;
  surface_ = surface;
  create_swap_chain();
  create_image_views();
  create_depth_resources();
  create_sync_objects();
}

void SwapChain::update_image_index() {
  vkWaitForFences(app_context_->device(), 1, &in_flight_scenes_[current_frame_],
                  VK_TRUE, UINT64_MAX);
  uint32_t image_index;
  vkAcquireNextImageKHR(app_context_->device(), swap_chain_, UINT64_MAX,
                        image_available_semaphores_[current_frame_],
                        VK_NULL_HANDLE, &image_index);
  curr_image_index_ = image_index;
}

uint32_t SwapChain::curr_image_index() {
  return curr_image_index_;
}

void SwapChain::cleanup_swap_chain() {
  vkDestroyImageView(app_context_->device(), depth_image_view_, nullptr);
  vkDestroyImage(app_context_->device(), depth_image_, nullptr);
  vkFreeMemory(app_context_->device(), depth_image_memory_, nullptr);

  for (auto framebuffer : swap_chain_framebuffers_) {
    vkDestroyFramebuffer(app_context_->device(), framebuffer, nullptr);
  }

  for (auto image_view : swap_chain_image_views_) {
    vkDestroyImageView(app_context_->device(), image_view, nullptr);
  }

  vkDestroySwapchainKHR(app_context_->device(), swap_chain_, nullptr);
}

void SwapChain::cleanup() {
  for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
    vkDestroySemaphore(app_context_->device(), render_finished_semaphores_[i],
                       nullptr);
    vkDestroySemaphore(app_context_->device(), image_available_semaphores_[i],
                       nullptr);
    vkDestroyFence(app_context_->device(), in_flight_scenes_[i], nullptr);
  }

  vkDestroySurfaceKHR(app_context_->instance(), surface_, nullptr);
}

void SwapChain::recreate_swap_chain() {
  create_swap_chain();
  create_image_views();

  create_depth_resources();
  create_framebuffers(render_pass_);

  images_in_flight_.resize(swap_chain_images_.size(), VK_NULL_HANDLE);
  requires_recreate_ = false;
}

void SwapChain::create_swap_chain() {
  SwapChainSupportDetails swap_chain_support =
      query_swap_chain_support(app_context_->physical_device(), surface_);

  VkSurfaceFormatKHR surface_format =
      choose_swap_surface_format(swap_chain_support.formats);
  VkPresentModeKHR present_mode =
      choose_swap_present_mode(swap_chain_support.present_modes);
  VkExtent2D extent = choose_swap_extent(swap_chain_support.capabilities);

  uint32_t image_count = swap_chain_support.capabilities.minImageCount + 1;
  if (swap_chain_support.capabilities.maxImageCount > 0 &&
      image_count > swap_chain_support.capabilities.maxImageCount) {
    image_count = swap_chain_support.capabilities.maxImageCount;
  }

  VkSwapchainCreateInfoKHR create_info{};
  create_info.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
  create_info.surface = surface_;

  create_info.minImageCount = image_count;
  create_info.imageFormat = surface_format.format;
  create_info.imageColorSpace = surface_format.colorSpace;
  create_info.imageExtent = extent;
  create_info.imageArrayLayers = 1;
  create_info.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;

  VulkanQueueFamilyIndices indices = app_context_->queue_family_indices();
  uint32_t queue_family_indices[] = {indices.graphics_family.value(),
                                     indices.present_family.value()};

  if (indices.graphics_family != indices.present_family) {
    create_info.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
    create_info.queueFamilyIndexCount = 2;
    create_info.pQueueFamilyIndices = queue_family_indices;
  } else {
    create_info.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
  }

  create_info.preTransform = swap_chain_support.capabilities.currentTransform;
  create_info.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
  create_info.presentMode = present_mode;
  create_info.clipped = VK_TRUE;

  if (vkCreateSwapchainKHR(app_context_->device(), &create_info, nullptr,
                           &swap_chain_) != VK_SUCCESS) {
    throw std::runtime_error("failed to create swap chain!");
  }

  vkGetSwapchainImagesKHR(app_context_->device(), swap_chain_, &image_count,
                          nullptr);
  swap_chain_images_.resize(image_count);
  vkGetSwapchainImagesKHR(app_context_->device(), swap_chain_, &image_count,
                          swap_chain_images_.data());

  swap_chain_image_format_ = surface_format.format;
  swap_chain_extent_ = extent;
}

void SwapChain::create_image_views() {
  swap_chain_image_views_.resize(swap_chain_images_.size());

  for (uint32_t i = 0; i < swap_chain_images_.size(); i++) {
    swap_chain_image_views_[i] =
        create_image_view(2, swap_chain_images_[i], swap_chain_image_format_,
                          VK_IMAGE_ASPECT_COLOR_BIT, app_context_->device());
  }
}

void SwapChain::create_framebuffers(VkRenderPass render_pass) {
  render_pass_ = render_pass;
  swap_chain_framebuffers_.resize(swap_chain_image_views_.size());

  for (size_t i = 0; i < swap_chain_image_views_.size(); i++) {
    std::array<VkImageView, 2> attachments = {swap_chain_image_views_[i],
                                              depth_image_view_};

    VkFramebufferCreateInfo framebuffer_info{};
    framebuffer_info.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
    framebuffer_info.renderPass = render_pass;
    framebuffer_info.attachmentCount =
        static_cast<uint32_t>(attachments.size());
    framebuffer_info.pAttachments = attachments.data();
    framebuffer_info.width = swap_chain_extent_.width;
    framebuffer_info.height = swap_chain_extent_.height;
    framebuffer_info.layers = 1;

    if (vkCreateFramebuffer(app_context_->device(), &framebuffer_info, nullptr,
                            &swap_chain_framebuffers_[i]) != VK_SUCCESS) {
      throw std::runtime_error("failed to create framebuffer!");
    }
  }
}

void SwapChain::create_depth_resources() {
  VkFormat depth_format = find_depth_format(app_context_->physical_device());

  create_image(
      2, swap_chain_extent_.width, swap_chain_extent_.height, 1, depth_format,
      VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
      VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, depth_image_, depth_image_memory_,
      app_context_->device(), app_context_->physical_device());
  depth_image_view_ =
      create_image_view(2, depth_image_, depth_format,
                        VK_IMAGE_ASPECT_DEPTH_BIT, app_context_->device());
}

void SwapChain::create_sync_objects() {
  image_available_semaphores_.resize(MAX_FRAMES_IN_FLIGHT);
  render_finished_semaphores_.resize(MAX_FRAMES_IN_FLIGHT);
  in_flight_scenes_.resize(MAX_FRAMES_IN_FLIGHT);
  images_in_flight_.resize(swap_chain_images_.size(), VK_NULL_HANDLE);

  VkSemaphoreCreateInfo semaphore_info{};
  semaphore_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

  VkFenceCreateInfo fenceInfo{};
  fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
  fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

  for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
    if (vkCreateSemaphore(app_context_->device(), &semaphore_info, nullptr,
                          &image_available_semaphores_[i]) != VK_SUCCESS ||
        vkCreateSemaphore(app_context_->device(), &semaphore_info, nullptr,
                          &render_finished_semaphores_[i]) != VK_SUCCESS ||
        vkCreateFence(app_context_->device(), &fenceInfo, nullptr,
                      &in_flight_scenes_[i]) != VK_SUCCESS) {
      throw std::runtime_error(
          "failed to create synchronization objects for a frame!");
    }
  }
}

void SwapChain::present_frame() {
  uint32_t image_index = curr_image_index_;

  VkSubmitInfo submit_info{};
  VkSemaphore signal_semaphores[] = {
      render_finished_semaphores_[current_frame_]};
  submit_info.signalSemaphoreCount = 1;
  submit_info.pSignalSemaphores = signal_semaphores;

  VkPresentInfoKHR present_info{};
  present_info.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;

  present_info.waitSemaphoreCount = 1;
  present_info.pWaitSemaphores = signal_semaphores;

  VkSwapchainKHR swap_chain_s[] = {swap_chain_};
  present_info.swapchainCount = 1;
  present_info.pSwapchains = swap_chain_s;

  present_info.pImageIndices = &image_index;

  VkResult result =
      vkQueuePresentKHR(app_context_->present_queue(), &present_info);

  if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR) {
    requires_recreate_ = true;
  } else if (result != VK_SUCCESS) {
    throw std::runtime_error("failed to present swap chain image!");
  }

  current_frame_ = (current_frame_ + 1) % MAX_FRAMES_IN_FLIGHT;

  vkDeviceWaitIdle(app_context_->device());
}

VkSurfaceFormatKHR SwapChain::choose_swap_surface_format(
    const std::vector<VkSurfaceFormatKHR> &available_formats) {
  for (const auto &available_format : available_formats) {
    if (available_format.format == VK_FORMAT_B8G8R8A8_UNORM &&
        available_format.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
      return available_format;
    }
  }

  return available_formats[0];
}

VkPresentModeKHR SwapChain::choose_swap_present_mode(
    const std::vector<VkPresentModeKHR> &available_present_modes) {
  if (app_context_->config.vsync) {
    for (const auto &available_present_mode : available_present_modes) {
      if (available_present_mode == VK_PRESENT_MODE_FIFO_KHR) {
        return available_present_mode;
      }
    }
  } else {
    for (const auto &available_present_mode : available_present_modes) {
      if (available_present_mode == VK_PRESENT_MODE_MAILBOX_KHR) {
        return available_present_mode;
      }
    }
    for (const auto &available_present_mode : available_present_modes) {
      if (available_present_mode == VK_PRESENT_MODE_IMMEDIATE_KHR) {
        return available_present_mode;
      }
    }
  }

  if (available_present_modes.size() == 0) {
    throw std::runtime_error("no avialble present modes");
  }

  return available_present_modes[0];
}

VkExtent2D SwapChain::choose_swap_extent(
    const VkSurfaceCapabilitiesKHR &capabilities) {
  if (capabilities.currentExtent.width != UINT32_MAX) {
    VkExtent2D result = capabilities.currentExtent;
    // printf("using currentExtent: %d %d\n",result .width,result .height);
    return result;
  } else {
    VkExtent2D actualExtent = {
        static_cast<uint32_t>(app_context_->config.width),
        static_cast<uint32_t>(app_context_->config.height)};

    actualExtent.width =
        std::clamp(actualExtent.width, capabilities.minImageExtent.width,
                   capabilities.maxImageExtent.width);
    actualExtent.height =
        std::clamp(actualExtent.height, capabilities.minImageExtent.height,
                   capabilities.maxImageExtent.height);

    // printf("%d %d\n",app_context_->config.width,app_context_->config.height);

    return actualExtent;
  }
}

bool SwapChain::requires_recreate() const {
  return requires_recreate_;
}

std::vector<VkFence> &SwapChain::in_flight_scenes() {
  return in_flight_scenes_;
}
VkExtent2D SwapChain::swap_chain_extent() const {
  return swap_chain_extent_;
}
std::vector<VkFence> &SwapChain::images_in_flight() {
  return images_in_flight_;
}

uint32_t SwapChain::current_frame() const {
  return current_frame_;
}

size_t SwapChain::chain_size() const {
  return swap_chain_images_.size();
}

const std::vector<VkSemaphore> &SwapChain::image_available_semaphores() const {
  return image_available_semaphores_;
}

const std::vector<VkSemaphore> &SwapChain::render_finished_semaphores() const {
  return render_finished_semaphores_;
}

VkFormat SwapChain::swap_chain_image_format() const {
  return swap_chain_image_format_;
}
const std::vector<VkFramebuffer> &SwapChain::swap_chain_framebuffers() const {
  return swap_chain_framebuffers_;
}

}  // namespace vulkan

TI_UI_NAMESPACE_END
