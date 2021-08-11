#include "taichi/ui/backends/vulkan/vulkan_utils.h"

TI_UI_NAMESPACE_BEGIN

namespace vulkan {

SwapChainSupportDetails query_swap_chain_support(
    VkPhysicalDevice physical_device,
    VkSurfaceKHR surface) {
  SwapChainSupportDetails details;

  vkGetPhysicalDeviceSurfaceCapabilitiesKHR(physical_device, surface,
                                            &details.capabilities);

  uint32_t format_count;
  vkGetPhysicalDeviceSurfaceFormatsKHR(physical_device, surface, &format_count,
                                       nullptr);

  if (format_count != 0) {
    details.formats.resize(format_count);
    vkGetPhysicalDeviceSurfaceFormatsKHR(physical_device, surface,
                                         &format_count, details.formats.data());
  }

  uint32_t present_mode_count;
  vkGetPhysicalDeviceSurfacePresentModesKHR(physical_device, surface,
                                            &present_mode_count, nullptr);

  if (present_mode_count != 0) {
    details.present_modes.resize(present_mode_count);
    vkGetPhysicalDeviceSurfacePresentModesKHR(physical_device, surface,
                                              &present_mode_count,
                                              details.present_modes.data());
  }

  return details;
}

VkFormat find_supported_format(const std::vector<VkFormat> &candidates,
                               VkImageTiling tiling,
                               VkFormatFeatureFlags features,
                               VkPhysicalDevice physical_device) {
  for (VkFormat format : candidates) {
    VkFormatProperties props;
    vkGetPhysicalDeviceFormatProperties(physical_device, format, &props);

    if (tiling == VK_IMAGE_TILING_LINEAR &&
        (props.linearTilingFeatures & features) == features) {
      return format;
    } else if (tiling == VK_IMAGE_TILING_OPTIMAL &&
               (props.optimalTilingFeatures & features) == features) {
      return format;
    }
  }

  throw std::runtime_error("failed to find supported format!");
}

VkFormat find_depth_format(VkPhysicalDevice physical_device) {
  return find_supported_format(
      {VK_FORMAT_D32_SFLOAT, VK_FORMAT_D32_SFLOAT_S8_UINT,
       VK_FORMAT_D24_UNORM_S8_UINT},
      VK_IMAGE_TILING_OPTIMAL, VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT,
      physical_device);
}

VkCommandBuffer create_new_command_buffer(VkCommandPool command_pool,
                                          VkDevice device) {
  VkCommandBufferAllocateInfo alloc_info{};
  alloc_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  alloc_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  alloc_info.commandPool = command_pool;
  alloc_info.commandBufferCount = 1;

  VkCommandBuffer command_buffer;
  vkAllocateCommandBuffers(device, &alloc_info, &command_buffer);
  return command_buffer;
}

VkCommandBuffer begin_single_time_commands(VkCommandPool command_pool,
                                           VkDevice device) {
  VkCommandBuffer command_buffer =
      create_new_command_buffer(command_pool, device);

  VkCommandBufferBeginInfo begin_info{};
  begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

  vkBeginCommandBuffer(command_buffer, &begin_info);

  return command_buffer;
}

void end_single_time_commands(VkCommandBuffer command_buffer,
                              VkCommandPool command_pool,
                              VkDevice device,
                              VkQueue graphics_queue) {
  vkEndCommandBuffer(command_buffer);

  VkSubmitInfo submit_info{};
  submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  submit_info.commandBufferCount = 1;
  submit_info.pCommandBuffers = &command_buffer;

  vkQueueSubmit(graphics_queue, 1, &submit_info, VK_NULL_HANDLE);
  vkQueueWaitIdle(graphics_queue);

  vkFreeCommandBuffers(device, command_pool, 1, &command_buffer);
}

void copy_buffer(VkBuffer src_buffer,
                 VkBuffer dst_buffer,
                 VkDeviceSize size,
                 VkCommandPool command_pool,
                 VkDevice device,
                 VkQueue graphics_queue) {
  VkCommandBuffer command_buffer =
      begin_single_time_commands(command_pool, device);

  VkBufferCopy copy_region{};
  copy_region.size = size;
  vkCmdCopyBuffer(command_buffer, src_buffer, dst_buffer, 1, &copy_region);

  end_single_time_commands(command_buffer, command_pool, device,
                           graphics_queue);
}

void copy_buffer_to_image(VkBuffer buffer,
                          VkImage image,
                          uint32_t width,
                          uint32_t height,
                          VkCommandPool command_pool,
                          VkDevice device,
                          VkQueue graphics_queue) {
  VkCommandBuffer command_buffer =
      begin_single_time_commands(command_pool, device);

  VkBufferImageCopy region{};
  region.bufferOffset = 0;
  region.bufferRowLength = 0;
  region.bufferImageHeight = 0;
  region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
  region.imageSubresource.mipLevel = 0;
  region.imageSubresource.baseArrayLayer = 0;
  region.imageSubresource.layerCount = 1;
  region.imageOffset = {0, 0, 0};
  region.imageExtent = {width, height, 1};

  vkCmdCopyBufferToImage(command_buffer, buffer, image,
                         VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

  end_single_time_commands(command_buffer, command_pool, device,
                           graphics_queue);
}

uint32_t find_memory_type(uint32_t type_filter,
                          VkMemoryPropertyFlags properties,
                          VkPhysicalDevice physical_device) {
  VkPhysicalDeviceMemoryProperties mem_properties;
  vkGetPhysicalDeviceMemoryProperties(physical_device, &mem_properties);

  for (uint32_t i = 0; i < mem_properties.memoryTypeCount; i++) {
    if ((type_filter & (1 << i)) &&
        (mem_properties.memoryTypes[i].propertyFlags & properties) ==
            properties) {
      return i;
    }
  }

  throw std::runtime_error("failed to find suitable memory type!");
}

void alloc_device_memory(VkMemoryRequirements mem_requirements,
                         VkMemoryPropertyFlags properties,
                         VkDevice device,
                         VkPhysicalDevice physical_device,
                         VkDeviceMemory &mem) {
  VkMemoryAllocateInfo alloc_info{};
  alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  alloc_info.allocationSize = mem_requirements.size;
  alloc_info.memoryTypeIndex = find_memory_type(mem_requirements.memoryTypeBits,
                                                properties, physical_device);

  VkExportMemoryAllocateInfoKHR export_mem_alloc_info = {};
  export_mem_alloc_info.sType =
      VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO_KHR;
#ifdef _WIN64
  WindowsSecurityAttributes win_security_attribs;

  VkExportMemoryWin32HandleInfoKHR export_mem_win32_handle_info = {};
  export_mem_win32_handle_info.sType =
      VK_STRUCTURE_TYPE_EXPORT_MEMORY_WIN32_HANDLE_INFO_KHR;
  export_mem_win32_handle_info.pNext = NULL;
  export_mem_win32_handle_info.pAttributes = &win_security_attribs;
  export_mem_win32_handle_info.dwAccess =
      DXGI_SHARED_RESOURCE_READ | DXGI_SHARED_RESOURCE_WRITE;
  export_mem_win32_handle_info.name = (LPCWSTR)NULL;

  export_mem_alloc_info.pNext = &export_mem_win32_handle_info;
  export_mem_alloc_info.handleTypes =
      VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;
#else
  export_mem_alloc_info.pNext = NULL;
  export_mem_alloc_info.handleTypes =
      VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT_KHR;
#endif
  alloc_info.pNext = &export_mem_alloc_info;
  if (vkAllocateMemory(device, &alloc_info, nullptr, &mem) != VK_SUCCESS) {
    throw std::runtime_error("failed to allocate image memory!");
  }
}

void create_semaphore(VkSemaphore &result, VkDevice device) {
  VkSemaphoreCreateInfo semaphore_info = {};
  memset(&semaphore_info, 0, sizeof(semaphore_info));
  semaphore_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

  VkExportSemaphoreCreateInfoKHR export_semaphore_create_info = {};
  export_semaphore_create_info.sType =
      VK_STRUCTURE_TYPE_EXPORT_SEMAPHORE_CREATE_INFO_KHR;

#ifdef _WIN64
  WindowsSecurityAttributes win_security_attribs;

  VkExportSemaphoreWin32HandleInfoKHR export_semaphore_win32_handle_info = {};
  export_semaphore_win32_handle_info.sType =
      VK_STRUCTURE_TYPE_EXPORT_SEMAPHORE_WIN32_HANDLE_INFO_KHR;
  export_semaphore_win32_handle_info.pNext = NULL;
  export_semaphore_win32_handle_info.pAttributes = &win_security_attribs;
  export_semaphore_win32_handle_info.dwAccess =
      DXGI_SHARED_RESOURCE_READ | DXGI_SHARED_RESOURCE_WRITE;
  export_semaphore_win32_handle_info.name = (LPCWSTR)NULL;

  export_semaphore_create_info.pNext =
      IsWindows8OrGreater() ? &export_semaphore_win32_handle_info : NULL;
  export_semaphore_create_info.handleTypes =
      IsWindows8OrGreater()
          ? VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_BIT
          : VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_KMT_BIT;
#else
  export_semaphore_create_info.pNext = NULL;
  export_semaphore_create_info.handleTypes =
      VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT;
#endif
  semaphore_info.pNext = &export_semaphore_create_info;

  if (vkCreateSemaphore(device, &semaphore_info, nullptr, &result) !=
      VK_SUCCESS) {
    throw std::runtime_error(
        "failed to create synchronization objects for a CUDA-Vulkan!");
  }
}

void create_buffer(VkDeviceSize size,
                   VkBufferUsageFlags usage,
                   VkMemoryPropertyFlags properties,
                   VkBuffer &buffer,
                   VkDeviceMemory &buffer_mem,
                   VkDevice device,
                   VkPhysicalDevice physical_device) {
  VkBufferCreateInfo buffer_info{};
  buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  buffer_info.size = size;
  buffer_info.usage = usage;
  buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

  VkExternalMemoryBufferCreateInfo external_mem_buffer_create_info = {};
  external_mem_buffer_create_info.sType =
      VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO;
  external_mem_buffer_create_info.pNext = NULL;

#ifdef _WIN64
  external_mem_buffer_create_info.handleTypes =
      VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;
#else
  external_mem_buffer_create_info.handleTypes =
      VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT_KHR;
#endif

  buffer_info.pNext = &external_mem_buffer_create_info;

  if (vkCreateBuffer(device, &buffer_info, nullptr, &buffer) != VK_SUCCESS) {
    throw std::runtime_error("failed to create buffer!");
  }

  VkMemoryRequirements mem_requirements;
  vkGetBufferMemoryRequirements(device, buffer, &mem_requirements);

  alloc_device_memory(mem_requirements, properties, device, physical_device,
                      buffer_mem);

  vkBindBufferMemory(device, buffer, buffer_mem, 0);
}

VkShaderModule create_shader_module(const std::vector<char> &code,
                                    VkDevice device) {
  VkShaderModuleCreateInfo create_info{};
  create_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
  create_info.codeSize = code.size();
  create_info.pCode = reinterpret_cast<const uint32_t *>(code.data());

  VkShaderModule shader_module;
  if (vkCreateShaderModule(device, &create_info, nullptr, &shader_module) !=
      VK_SUCCESS) {
    throw std::runtime_error("failed to create shader module!");
  }

  return shader_module;
}

void create_image(int dim,
                  uint32_t width,
                  uint32_t height,
                  uint32_t depth,
                  VkFormat format,
                  VkImageTiling tiling,
                  VkImageUsageFlags usage,
                  VkMemoryPropertyFlags properties,
                  VkImage &image,
                  VkDeviceMemory &image_mem,
                  VkDevice device,
                  VkPhysicalDevice physical_device) {
  VkImageCreateInfo image_info{};
  image_info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
  if (dim == 1) {
    image_info.imageType = VK_IMAGE_TYPE_1D;
  } else if (dim == 2) {
    image_info.imageType = VK_IMAGE_TYPE_2D;
  } else if (dim == 3) {
    image_info.imageType = VK_IMAGE_TYPE_3D;
  } else {
    throw std::runtime_error("dim can only be 1 2 3");
  }
  image_info.extent.width = width;
  image_info.extent.height = height;
  image_info.extent.depth = depth;
  image_info.mipLevels = 1;
  image_info.arrayLayers = 1;
  image_info.format = format;
  image_info.tiling = tiling;
  image_info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  image_info.usage = usage;
  image_info.samples = VK_SAMPLE_COUNT_1_BIT;
  image_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

  VkExternalMemoryImageCreateInfo external_mem_image_create_info = {};
  external_mem_image_create_info.sType =
      VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO;
  external_mem_image_create_info.pNext = NULL;

#ifdef _WIN64
  external_mem_image_create_info.handleTypes =
      VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;
#else
  external_mem_image_create_info.handleTypes =
      VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT_KHR;
#endif

  image_info.pNext = &external_mem_image_create_info;

  if (vkCreateImage(device, &image_info, nullptr, &image) != VK_SUCCESS) {
    throw std::runtime_error("failed to create image!");
  }

  VkMemoryRequirements mem_requirements;
  vkGetImageMemoryRequirements(device, image, &mem_requirements);

  alloc_device_memory(mem_requirements, properties, device, physical_device,
                      image_mem);

  vkBindImageMemory(device, image, image_mem, 0);
}

VkImageView create_image_view(int dim,
                              VkImage image,
                              VkFormat format,
                              VkImageAspectFlags aspect_flags,
                              VkDevice device) {
  VkImageViewCreateInfo view_info{};
  view_info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
  view_info.image = image;
  if (dim == 1) {
    view_info.viewType = VK_IMAGE_VIEW_TYPE_1D;
  } else if (dim == 2) {
    view_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
  } else if (dim == 3) {
    view_info.viewType = VK_IMAGE_VIEW_TYPE_3D;
  } else {
    throw std::runtime_error("dim can only be 1 2 3");
  }

  view_info.format = format;
  view_info.subresourceRange.aspectMask = aspect_flags;
  view_info.subresourceRange.baseMipLevel = 0;
  view_info.subresourceRange.levelCount = 1;
  view_info.subresourceRange.baseArrayLayer = 0;
  view_info.subresourceRange.layerCount = 1;

  VkImageView image_view;
  if (vkCreateImageView(device, &view_info, nullptr, &image_view) !=
      VK_SUCCESS) {
    throw std::runtime_error("failed to create texture image view!");
  }

  return image_view;
}

void transition_image_layout(VkImage image,
                             VkFormat format,
                             VkImageLayout old_layout,
                             VkImageLayout new_layout,
                             VkCommandPool command_pool,
                             VkDevice device,
                             VkQueue graphics_queue) {
  VkCommandBuffer command_buffer =
      begin_single_time_commands(command_pool, device);

  VkImageMemoryBarrier barrier{};
  barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
  barrier.oldLayout = old_layout;
  barrier.newLayout = new_layout;
  barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  barrier.image = image;
  barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
  barrier.subresourceRange.baseMipLevel = 0;
  barrier.subresourceRange.levelCount = 1;
  barrier.subresourceRange.baseArrayLayer = 0;
  barrier.subresourceRange.layerCount = 1;

  VkPipelineStageFlags source_stage;
  VkPipelineStageFlags destination_stage;

  std::unordered_map<VkImageLayout, VkPipelineStageFlagBits> stages;
  stages[VK_IMAGE_LAYOUT_UNDEFINED] = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
  stages[VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL] = VK_PIPELINE_STAGE_TRANSFER_BIT;
  stages[VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL] =
      VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
  stages[VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL] =
      VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;

  std::unordered_map<VkImageLayout, VkAccessFlagBits> access;
  access[VK_IMAGE_LAYOUT_UNDEFINED] = (VkAccessFlagBits)0;
  access[VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL] = VK_ACCESS_TRANSFER_WRITE_BIT;
  access[VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL] = VK_ACCESS_SHADER_READ_BIT;
  access[VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL] =
      VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

  if (stages.find(old_layout) == stages.end() ||
      stages.find(new_layout) == stages.end()) {
    throw std::invalid_argument("unsupported layout transition!");
  }
  source_stage = stages.at(old_layout);
  destination_stage = stages.at(new_layout);

  if (access.find(old_layout) == access.end() ||
      access.find(new_layout) == access.end()) {
    throw std::invalid_argument("unsupported layout transition!");
  }
  barrier.srcAccessMask = access.at(old_layout);
  barrier.dstAccessMask = access.at(new_layout);

  vkCmdPipelineBarrier(command_buffer, source_stage, destination_stage, 0, 0,
                       nullptr, 0, nullptr, 1, &barrier);

  end_single_time_commands(command_buffer, command_pool, device,
                           graphics_queue);
}

}  // namespace vulkan

TI_UI_NAMESPACE_END
