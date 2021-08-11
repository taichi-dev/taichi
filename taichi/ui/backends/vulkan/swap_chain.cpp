#include "taichi/ui/utils/utils.h"
#include "taichi/ui/backends/vulkan/vulkan_utils.h"
#include "taichi/ui/backends/vulkan/app_context.h"
#include "taichi/ui/backends/vulkan/swap_chain.h"

TI_UI_NAMESPACE_BEGIN

namespace vulkan {

using namespace taichi::lang::vulkan;

void SwapChain::update_image_index() {
  curr_image_index = get_image_index();
}

void SwapChain::cleanup_swap_chain() {
  vkDestroyImageView(app_context->device(), depth_image_view, nullptr);
  vkDestroyImage(app_context->device(), depth_image, nullptr);
  vkFreeMemory(app_context->device(), depth_image_memory, nullptr);

  for (auto framebuffer : swap_chain_framebuffers) {
    vkDestroyFramebuffer(app_context->device(), framebuffer, nullptr);
  }

  for (auto image_view : swap_chain_image_views) {
    vkDestroyImageView(app_context->device(), image_view, nullptr);
  }

  vkDestroySwapchainKHR(app_context->device(), swap_chain, nullptr);
}

void SwapChain::cleanup() {
  for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
    vkDestroySemaphore(app_context->device(), render_finished_semaphores[i],
                       nullptr);
    vkDestroySemaphore(app_context->device(), image_available_semaphores[i],
                       nullptr);
    vkDestroyFence(app_context->device(), in_flight_scenes[i], nullptr);
  }

  vkDestroySurfaceKHR(app_context->instance(), surface, nullptr);
}

void SwapChain::recreate_swap_chain() {
  create_swap_chain();
  create_image_views();

  create_depth_resources();
  create_framebuffers();

  images_in_flight.resize(swap_chain_images.size(), VK_NULL_HANDLE);
  requires_recreate = false;
}

void SwapChain::create_swap_chain() {
  SwapChainSupportDetails swap_chain_support =
      query_swap_chain_support(app_context->physical_device(), surface);

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
  create_info.surface = surface;

  create_info.minImageCount = image_count;
  create_info.imageFormat = surface_format.format;
  create_info.imageColorSpace = surface_format.colorSpace;
  create_info.imageExtent = extent;
  create_info.imageArrayLayers = 1;
  create_info.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;

  VulkanQueueFamilyIndices indices = app_context->queue_family_indices();
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

  if (vkCreateSwapchainKHR(app_context->device(), &create_info, nullptr,
                           &swap_chain) != VK_SUCCESS) {
    throw std::runtime_error("failed to create swap chain!");
  }

  vkGetSwapchainImagesKHR(app_context->device(), swap_chain, &image_count,
                          nullptr);
  swap_chain_images.resize(image_count);
  vkGetSwapchainImagesKHR(app_context->device(), swap_chain, &image_count,
                          swap_chain_images.data());

  swap_chain_image_format = surface_format.format;
  swap_chain_extent = extent;
}

void SwapChain::create_image_views() {
  swap_chain_image_views.resize(swap_chain_images.size());

  for (uint32_t i = 0; i < swap_chain_images.size(); i++) {
    swap_chain_image_views[i] =
        create_image_view(2, swap_chain_images[i], swap_chain_image_format,
                          VK_IMAGE_ASPECT_COLOR_BIT, app_context->device());
  }
}

void SwapChain::create_framebuffers() {
  swap_chain_framebuffers.resize(swap_chain_image_views.size());

  for (size_t i = 0; i < swap_chain_image_views.size(); i++) {
    std::array<VkImageView, 2> attachments = {swap_chain_image_views[i],
                                              depth_image_view};

    VkFramebufferCreateInfo framebuffer_info{};
    framebuffer_info.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
    framebuffer_info.renderPass = app_context->render_pass();
    framebuffer_info.attachmentCount =
        static_cast<uint32_t>(attachments.size());
    framebuffer_info.pAttachments = attachments.data();
    framebuffer_info.width = swap_chain_extent.width;
    framebuffer_info.height = swap_chain_extent.height;
    framebuffer_info.layers = 1;

    if (vkCreateFramebuffer(app_context->device(), &framebuffer_info, nullptr,
                            &swap_chain_framebuffers[i]) != VK_SUCCESS) {
      throw std::runtime_error("failed to create framebuffer!");
    }
  }
}

void SwapChain::create_depth_resources() {
  VkFormat depth_format = find_depth_format(app_context->physical_device());

  create_image(
      2, swap_chain_extent.width, swap_chain_extent.height, 1, depth_format,
      VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
      VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, depth_image, depth_image_memory,
      app_context->device(), app_context->physical_device());
  depth_image_view =
      create_image_view(2, depth_image, depth_format, VK_IMAGE_ASPECT_DEPTH_BIT,
                        app_context->device());
}

void SwapChain::create_sync_objects() {
  image_available_semaphores.resize(MAX_FRAMES_IN_FLIGHT);
  render_finished_semaphores.resize(MAX_FRAMES_IN_FLIGHT);
  in_flight_scenes.resize(MAX_FRAMES_IN_FLIGHT);
  images_in_flight.resize(swap_chain_images.size(), VK_NULL_HANDLE);

  VkSemaphoreCreateInfo semaphore_info{};
  semaphore_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

  VkFenceCreateInfo fenceInfo{};
  fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
  fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

  for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
    if (vkCreateSemaphore(app_context->device(), &semaphore_info, nullptr,
                          &image_available_semaphores[i]) != VK_SUCCESS ||
        vkCreateSemaphore(app_context->device(), &semaphore_info, nullptr,
                          &render_finished_semaphores[i]) != VK_SUCCESS ||
        vkCreateFence(app_context->device(), &fenceInfo, nullptr,
                      &in_flight_scenes[i]) != VK_SUCCESS) {
      throw std::runtime_error(
          "failed to create synchronization objects for a frame!");
    }
  }
}

uint32_t SwapChain::get_image_index() {
  vkWaitForFences(app_context->device(), 1, &in_flight_scenes[current_frame],
                  VK_TRUE, UINT64_MAX);
  uint32_t image_index;
  vkAcquireNextImageKHR(app_context->device(), swap_chain, UINT64_MAX,
                        image_available_semaphores[current_frame],
                        VK_NULL_HANDLE, &image_index);
  return image_index;
}

void SwapChain::present_frame() {
  uint32_t image_index = curr_image_index;

  VkSubmitInfo submit_info{};
  VkSemaphore signal_semaphores[] = {render_finished_semaphores[current_frame]};
  submit_info.signalSemaphoreCount = 1;
  submit_info.pSignalSemaphores = signal_semaphores;

  VkPresentInfoKHR present_info{};
  present_info.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;

  present_info.waitSemaphoreCount = 1;
  present_info.pWaitSemaphores = signal_semaphores;

  VkSwapchainKHR swap_chains[] = {swap_chain};
  present_info.swapchainCount = 1;
  present_info.pSwapchains = swap_chains;

  present_info.pImageIndices = &image_index;

  VkResult result =
      vkQueuePresentKHR(app_context->present_queue(), &present_info);

  if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR) {
    requires_recreate = true;
  } else if (result != VK_SUCCESS) {
    throw std::runtime_error("failed to present swap chain image!");
  }

  current_frame = (current_frame + 1) % MAX_FRAMES_IN_FLIGHT;

  vkDeviceWaitIdle(app_context->device());
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
  if (app_context->config.vsync) {
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
        static_cast<uint32_t>(app_context->config.width),
        static_cast<uint32_t>(app_context->config.height)};

    actualExtent.width =
        std::clamp(actualExtent.width, capabilities.minImageExtent.width,
                   capabilities.maxImageExtent.width);
    actualExtent.height =
        std::clamp(actualExtent.height, capabilities.minImageExtent.height,
                   capabilities.maxImageExtent.height);

    // printf("%d %d\n",app_context->config.width,app_context->config.height);

    return actualExtent;
  }
}
}  // namespace vulkan

TI_UI_NAMESPACE_END
