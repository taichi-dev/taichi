#include "taichi/rhi/vulkan/vulkan_surface.h"
#include "taichi/rhi/vulkan/vulkan_device.h"

namespace taichi::lang {
namespace vulkan {

VulkanSurface::VulkanSurface(VulkanDevice *device, const SurfaceConfig &config)
    : config_(config), device_(device) {
#ifdef ANDROID
  window_ = (ANativeWindow *)config.window_handle;
#else
  window_ = (GLFWwindow *)config.window_handle;
#endif
  if (window_) {
    if (config.native_surface_handle) {
      surface_ = (VkSurfaceKHR)config.native_surface_handle;
    } else {
#ifdef ANDROID
      VkAndroidSurfaceCreateInfoKHR createInfo{
          .sType = VK_STRUCTURE_TYPE_ANDROID_SURFACE_CREATE_INFO_KHR,
          .pNext = nullptr,
          .flags = 0,
          .window = window_};

      vkCreateAndroidSurfaceKHR(device->vk_instance(), &createInfo, nullptr,
                                &surface_);
#else
      glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
      VkResult err = glfwCreateWindowSurface(device->vk_instance(), window_,
                                             nullptr, &surface_);
      if (err) {
        TI_ERROR("Failed to create window surface ({})", err);
        return;
      }
#endif
    }

    create_swap_chain();

    image_available_ = vkapi::create_semaphore(device->vk_device(), 0);
  } else {
    ImageParams params = {ImageDimension::d2D,
                          BufferFormat::rgba8,
                          ImageLayout::present_src,
                          config.width,
                          config.height,
                          1,
                          false};
    // screenshot_image_ = device->create_image(params);
    swapchain_images_.push_back(device->create_image(params));
    swapchain_images_.push_back(device->create_image(params));
    width_ = config.width;
    height_ = config.height;
  }
}

VkPresentModeKHR choose_swap_present_mode(
    const std::vector<VkPresentModeKHR> &available_present_modes,
    bool vsync,
    bool adaptive) {
  if (vsync) {
    if (adaptive) {
      for (const auto &available_present_mode : available_present_modes) {
        if (available_present_mode == VK_PRESENT_MODE_FIFO_RELAXED_KHR) {
          return available_present_mode;
        }
      }
    }
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

void VulkanSurface::create_swap_chain() {
  auto choose_surface_format =
      [](const std::vector<VkSurfaceFormatKHR> &availableFormats) {
        for (const auto &availableFormat : availableFormats) {
          if (availableFormat.format == VK_FORMAT_B8G8R8A8_UNORM &&
              availableFormat.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
            return availableFormat;
          }
        }
        return availableFormats[0];
      };

  VkSurfaceCapabilitiesKHR capabilities;
  vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device_->vk_physical_device(),
                                            surface_, &capabilities);

  VkBool32 supported = false;
  vkGetPhysicalDeviceSurfaceSupportKHR(device_->vk_physical_device(),
                                       device_->graphics_queue_family_index(),
                                       surface_, &supported);

  if (!supported) {
    TI_ERROR("Selected queue does not support presenting");
    return;
  }

  uint32_t formatCount;
  vkGetPhysicalDeviceSurfaceFormatsKHR(device_->vk_physical_device(), surface_,
                                       &formatCount, nullptr);
  std::vector<VkSurfaceFormatKHR> surface_formats(formatCount);
  vkGetPhysicalDeviceSurfaceFormatsKHR(device_->vk_physical_device(), surface_,
                                       &formatCount, surface_formats.data());

  VkSurfaceFormatKHR surface_format = choose_surface_format(surface_formats);

  uint32_t present_mode_count;
  std::vector<VkPresentModeKHR> present_modes;
  vkGetPhysicalDeviceSurfacePresentModesKHR(
      device_->vk_physical_device(), surface_, &present_mode_count, nullptr);

  if (present_mode_count != 0) {
    present_modes.resize(present_mode_count);
    vkGetPhysicalDeviceSurfacePresentModesKHR(device_->vk_physical_device(),
                                              surface_, &present_mode_count,
                                              present_modes.data());
  }
  VkPresentModeKHR present_mode =
      choose_swap_present_mode(present_modes, config_.vsync, config_.adaptive);

  int width, height;
#ifdef ANDROID
  width = ANativeWindow_getWidth(window_);
  height = ANativeWindow_getHeight(window_);
#else
  glfwGetFramebufferSize(window_, &width, &height);
#endif

  VkExtent2D extent = {uint32_t(width), uint32_t(height)};
  extent.width =
      std::max(capabilities.minImageExtent.width,
               std::min(capabilities.maxImageExtent.width, extent.width));
  extent.height =
      std::max(capabilities.minImageExtent.height,
               std::min(capabilities.maxImageExtent.height, extent.height));
  TI_INFO("Creating suface of {}x{}", extent.width, extent.height);
  VkImageUsageFlags usage =
      VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;

  this->width_ = extent.width;
  this->height_ = extent.height;

  VkSwapchainCreateInfoKHR createInfo;
  createInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
  createInfo.pNext = nullptr;
  createInfo.flags = 0;
  createInfo.surface = surface_;
  createInfo.minImageCount = std::min<uint32_t>(capabilities.maxImageCount, 3);
  createInfo.imageFormat = surface_format.format;
  createInfo.imageColorSpace = surface_format.colorSpace;
  createInfo.imageExtent = extent;
  createInfo.imageArrayLayers = 1;
  createInfo.imageUsage = usage;
  createInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
  createInfo.queueFamilyIndexCount = 0;
  createInfo.pQueueFamilyIndices = nullptr;
  createInfo.preTransform = capabilities.currentTransform;
  createInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
  createInfo.presentMode = present_mode;
  createInfo.clipped = VK_TRUE;
  createInfo.oldSwapchain = VK_NULL_HANDLE;

  if (vkCreateSwapchainKHR(device_->vk_device(), &createInfo,
                           kNoVkAllocCallbacks, &swapchain_) != VK_SUCCESS) {
    TI_ERROR("Failed to create swapchain");
    return;
  }

  uint32_t num_images;
  vkGetSwapchainImagesKHR(device_->vk_device(), swapchain_, &num_images,
                          nullptr);
  std::vector<VkImage> swapchain_images(num_images);
  vkGetSwapchainImagesKHR(device_->vk_device(), swapchain_, &num_images,
                          swapchain_images.data());

  image_format_ = buffer_format_vk_to_ti(surface_format.format);

  for (VkImage img : swapchain_images) {
    vkapi::IVkImage image = vkapi::create_image(
        device_->vk_device(), img, surface_format.format, VK_IMAGE_TYPE_2D,
        VkExtent3D{uint32_t(width), uint32_t(height), 1}, 1u, 1u, usage);

    VkImageViewCreateInfo create_info{};
    create_info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    create_info.image = image->image;
    create_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
    create_info.format = image->format;
    create_info.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    create_info.subresourceRange.baseMipLevel = 0;
    create_info.subresourceRange.levelCount = 1;
    create_info.subresourceRange.baseArrayLayer = 0;
    create_info.subresourceRange.layerCount = 1;

    vkapi::IVkImageView view =
        vkapi::create_image_view(device_->vk_device(), image, &create_info);

    swapchain_images_.push_back(
        device_->import_vk_image(image, view, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR));
  }
}

void VulkanSurface::destroy_swap_chain() {
  for (auto alloc : swapchain_images_) {
    std::get<1>(device_->get_vk_image(alloc)) = nullptr;
    device_->destroy_image(alloc);
  }
  swapchain_images_.clear();
  vkDestroySwapchainKHR(device_->vk_device(), swapchain_, nullptr);
}

int VulkanSurface::get_image_count() {
  return swapchain_images_.size();
}

VulkanSurface::~VulkanSurface() {
  if (config_.window_handle) {
    destroy_swap_chain();
    image_available_ = nullptr;
    vkDestroySurfaceKHR(device_->vk_instance(), surface_, nullptr);
  } else {
    for (auto &img : swapchain_images_) {
      device_->destroy_image(img);
    }
    swapchain_images_.clear();
  }
  if (depth_buffer_ != kDeviceNullAllocation) {
    device_->dealloc_memory(depth_buffer_);
  }
  if (screenshot_buffer_ != kDeviceNullAllocation) {
    device_->dealloc_memory(screenshot_buffer_);
  }
}

void VulkanSurface::resize(uint32_t width, uint32_t height) {
  destroy_swap_chain();
  create_swap_chain();
}

std::pair<uint32_t, uint32_t> VulkanSurface::get_size() {
  return std::make_pair(width_, height_);
}

StreamSemaphore VulkanSurface::acquire_next_image() {
  if (!config_.window_handle) {
    image_index_ = (image_index_ + 1) % swapchain_images_.size();
    return nullptr;
  } else {
    vkAcquireNextImageKHR(device_->vk_device(), swapchain_, UINT64_MAX,
                          image_available_->semaphore, VK_NULL_HANDLE,
                          &image_index_);
    return std::make_shared<VulkanStreamSemaphoreObject>(image_available_);
  }
}

DeviceAllocation VulkanSurface::get_target_image() {
  return swapchain_images_[image_index_];
}

BufferFormat VulkanSurface::image_format() {
  return image_format_;
}

void VulkanSurface::present_image(
    const std::vector<StreamSemaphore> &wait_semaphores) {
  std::vector<VkSemaphore> vk_wait_semaphores;

  // Already transitioned to `present_src` at the end of the render pass.
  // device_->image_transition(get_target_image(),
  // ImageLayout::color_attachment,
  //                          ImageLayout::present_src);

  for (const StreamSemaphore &sema_ : wait_semaphores) {
    auto sema = std::static_pointer_cast<VulkanStreamSemaphoreObject>(sema_);
    vk_wait_semaphores.push_back(sema->vkapi_ref->semaphore);
  }

  VkPresentInfoKHR presentInfo{};
  presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
  presentInfo.waitSemaphoreCount = vk_wait_semaphores.size();
  presentInfo.pWaitSemaphores = vk_wait_semaphores.data();
  presentInfo.swapchainCount = 1;
  presentInfo.pSwapchains = &swapchain_;
  presentInfo.pImageIndices = &image_index_;
  presentInfo.pResults = nullptr;

  vkQueuePresentKHR(device_->graphics_queue(), &presentInfo);

  device_->wait_idle();
}

DeviceAllocation VulkanSurface::get_depth_data(DeviceAllocation &depth_alloc) {
  auto *stream = device_->get_graphics_stream();

  auto [w, h] = get_size();
  size_t size_bytes = w * h * 4;

  if (depth_buffer_ == kDeviceNullAllocation) {
    Device::AllocParams params{size_bytes, /*host_wrtie*/ false,
                               /*host_read*/ true, /*export_sharing*/ false,
                               AllocUsage::Uniform};
    depth_buffer_ = device_->allocate_memory(params);
  }

  std::unique_ptr<CommandList> cmd_list{nullptr};

  BufferImageCopyParams copy_params;
  copy_params.image_extent.x = w;
  copy_params.image_extent.y = h;
  copy_params.image_aspect_flag = VK_IMAGE_ASPECT_DEPTH_BIT;
  cmd_list = stream->new_command_list();
  cmd_list->image_transition(depth_alloc, ImageLayout::depth_attachment,
                             ImageLayout::transfer_src);
  cmd_list->image_to_buffer(depth_buffer_.get_ptr(), depth_alloc,
                            ImageLayout::transfer_src, copy_params);
  cmd_list->image_transition(depth_alloc, ImageLayout::transfer_src,
                             ImageLayout::depth_attachment);
  stream->submit_synced(cmd_list.get());

  return depth_buffer_;
}

DeviceAllocation VulkanSurface::get_image_data() {
  auto *stream = device_->get_graphics_stream();
  DeviceAllocation img_alloc = swapchain_images_[image_index_];
  auto [w, h] = get_size();
  size_t size_bytes = w * h * 4;

  /*
  if (screenshot_image_ == kDeviceNullAllocation) {
    ImageParams params = {ImageDimension::d2D,
                          BufferFormat::rgba8,
                          ImageLayout::transfer_dst,
                          w,
                          h,
                          1,
                          false};
    screenshot_image_ = device_->create_image(params);
  }
  */

  if (screenshot_buffer_ == kDeviceNullAllocation) {
    Device::AllocParams params{size_bytes, /*host_wrtie*/ false,
                               /*host_read*/ true, /*export_sharing*/ false,
                               AllocUsage::Uniform};
    screenshot_buffer_ = device_->allocate_memory(params);
  }

  std::unique_ptr<CommandList> cmd_list{nullptr};

  /*
  if (config_.window_handle) {
    // TODO: check if blit is supported, and use copy_image if not
    cmd_list = stream->new_command_list();
    cmd_list->blit_image(screenshot_image_, img_alloc,
                         ImageLayout::transfer_dst, ImageLayout::transfer_src,
                         {w, h, 1});
    cmd_list->image_transition(screenshot_image_, ImageLayout::transfer_dst,
                               ImageLayout::transfer_src);
    stream->submit_synced(cmd_list.get());
  }
  */

  BufferImageCopyParams copy_params;
  copy_params.image_extent.x = w;
  copy_params.image_extent.y = h;
  copy_params.image_aspect_flag = VK_IMAGE_ASPECT_COLOR_BIT;
  cmd_list = stream->new_command_list();
  cmd_list->image_transition(img_alloc, ImageLayout::present_src,
                             ImageLayout::transfer_src);
  // TODO: directly map the image to cpu memory
  cmd_list->image_to_buffer(screenshot_buffer_.get_ptr(), img_alloc,
                            ImageLayout::transfer_src, copy_params);
  cmd_list->image_transition(img_alloc, ImageLayout::transfer_src,
                             ImageLayout::present_src);
  /*
  if (config_.window_handle) {
    cmd_list->image_transition(screenshot_image_, ImageLayout::transfer_src,
                               ImageLayout::transfer_dst);
  }
  */
  stream->submit_synced(cmd_list.get());

  return screenshot_buffer_;
}

}  // namespace vulkan
}  // namespace taichi::lang
