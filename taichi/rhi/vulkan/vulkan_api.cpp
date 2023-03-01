#define VOLK_IMPLEMENTATION

#include "taichi/rhi/impl_support.h"

#include "taichi/rhi/vulkan/vulkan_api.h"
#include "taichi/rhi/vulkan/vulkan_loader.h"

namespace vkapi {

DeviceObjVkDescriptorSetLayout::~DeviceObjVkDescriptorSetLayout() {
  vkDestroyDescriptorSetLayout(device, layout, nullptr);
}

DeviceObjVkDescriptorPool::~DeviceObjVkDescriptorPool() {
  vkDestroyDescriptorPool(device, pool, nullptr);
}

DeviceObjVkDescriptorSet::~DeviceObjVkDescriptorSet() {
}

DeviceObjVkCommandPool::~DeviceObjVkCommandPool() {
  vkDestroyCommandPool(device, pool, nullptr);
}

DeviceObjVkCommandBuffer::~DeviceObjVkCommandBuffer() {
  if (this->level == VK_COMMAND_BUFFER_LEVEL_PRIMARY) {
    ref_pool->free_primary.push(buffer);
  } else {
    ref_pool->free_secondary.push(buffer);
  }
}

DeviceObjVkRenderPass::~DeviceObjVkRenderPass() {
  vkDestroyRenderPass(device, renderpass, nullptr);
}

DeviceObjVkPipelineLayout::~DeviceObjVkPipelineLayout() {
  vkDestroyPipelineLayout(device, layout, nullptr);
}

DeviceObjVkPipeline::~DeviceObjVkPipeline() {
  vkDestroyPipeline(device, pipeline, nullptr);
}

DeviceObjVkSampler::~DeviceObjVkSampler() {
  vkDestroySampler(device, sampler, nullptr);
}

DeviceObjVkImage::~DeviceObjVkImage() {
  if (allocation) {
    vmaDestroyImage(allocator, image, allocation);
  }
}

DeviceObjVkImageView::~DeviceObjVkImageView() {
  vkDestroyImageView(device, view, nullptr);
}

DeviceObjVkFramebuffer::~DeviceObjVkFramebuffer() {
  vkDestroyFramebuffer(device, framebuffer, nullptr);
}

DeviceObjVkSemaphore::~DeviceObjVkSemaphore() {
  vkDestroySemaphore(device, semaphore, nullptr);
}

DeviceObjVkFence::~DeviceObjVkFence() {
  vkDestroyFence(device, fence, nullptr);
}

DeviceObjVkPipelineCache::~DeviceObjVkPipelineCache() {
  vkDestroyPipelineCache(device, cache, nullptr);
}

DeviceObjVkBuffer::~DeviceObjVkBuffer() {
  if (allocation) {
    vmaDestroyBuffer(allocator, buffer, allocation);
  }
}

DeviceObjVkBufferView::~DeviceObjVkBufferView() {
  vkDestroyBufferView(device, view, nullptr);
}

DeviceObjVkAccelerationStructureKHR::~DeviceObjVkAccelerationStructureKHR() {
  PFN_vkDestroyAccelerationStructureKHR destroy_raytracing_pipeline_khr =
      PFN_vkDestroyAccelerationStructureKHR(vkGetInstanceProcAddr(
          taichi::lang::vulkan::VulkanLoader::instance().get_instance(),
          "vkDestroyAccelerationStructureKHR"));

  destroy_raytracing_pipeline_khr(device, accel, nullptr);
}
DeviceObjVkQueryPool::~DeviceObjVkQueryPool() {
  vkDestroyQueryPool(device, query_pool, nullptr);
}

IDeviceObj create_device_obj(VkDevice device) {
  IDeviceObj obj = std::make_shared<DeviceObj>();
  obj->device = device;
  return obj;
}

IVkSemaphore create_semaphore(VkDevice device,
                              VkSemaphoreCreateFlags flags,
                              void *pnext) {
  IVkSemaphore obj = std::make_shared<DeviceObjVkSemaphore>();
  obj->device = device;

  VkSemaphoreCreateInfo info{};
  info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
  info.pNext = pnext;
  info.flags = flags;

  VkResult res = vkCreateSemaphore(device, &info, nullptr, &obj->semaphore);
  BAIL_ON_VK_BAD_RESULT_NO_RETURN(res, "failed to create semaphore");
  return obj;
}

IVkFence create_fence(VkDevice device, VkFenceCreateFlags flags, void *pnext) {
  IVkFence obj = std::make_shared<DeviceObjVkFence>();
  obj->device = device;

  VkFenceCreateInfo info{};
  info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
  info.pNext = pnext;
  info.flags = flags;

  VkResult res = vkCreateFence(device, &info, nullptr, &obj->fence);
  BAIL_ON_VK_BAD_RESULT_NO_RETURN(res, "failed to create fence");
  return obj;
}

IVkDescriptorSetLayout create_descriptor_set_layout(
    VkDevice device,
    VkDescriptorSetLayoutCreateInfo *create_info) {
  IVkDescriptorSetLayout obj =
      std::make_shared<DeviceObjVkDescriptorSetLayout>();
  obj->device = device;
  VkResult res =
      vkCreateDescriptorSetLayout(device, create_info, nullptr, &obj->layout);
  BAIL_ON_VK_BAD_RESULT_NO_RETURN(res,
                                  "failed to create descriptor set layout");
  return obj;
}

IVkDescriptorPool create_descriptor_pool(
    VkDevice device,
    VkDescriptorPoolCreateInfo *create_info) {
  IVkDescriptorPool obj = std::make_shared<DeviceObjVkDescriptorPool>();
  obj->device = device;
  VkResult res =
      vkCreateDescriptorPool(device, create_info, nullptr, &obj->pool);
  if (res != VK_SUCCESS) {
    // All failure condition listed in spec are OOM
    return nullptr;
  }
  return obj;
}

IVkDescriptorSet allocate_descriptor_sets(IVkDescriptorPool pool,
                                          IVkDescriptorSetLayout layout,
                                          void *pnext) {
  IVkDescriptorSet obj = std::make_shared<DeviceObjVkDescriptorSet>();
  obj->device = pool->device;
  obj->ref_layout = layout;
  obj->ref_pool = pool;

  VkDescriptorSetAllocateInfo info{};
  info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
  info.pNext = pnext;
  info.descriptorPool = pool->pool;
  info.descriptorSetCount = 1;
  info.pSetLayouts = &layout->layout;

  if (vkAllocateDescriptorSets(pool->device, &info, &obj->set) ==
      VK_ERROR_OUT_OF_POOL_MEMORY) {
    return nullptr;
  }

  return obj;
}

IVkCommandPool create_command_pool(VkDevice device,
                                   VkCommandPoolCreateFlags flags,
                                   uint32_t queue_family_index) {
  IVkCommandPool obj = std::make_shared<DeviceObjVkCommandPool>();
  obj->device = device;
  obj->queue_family_index = queue_family_index;

  VkCommandPoolCreateInfo info{};
  info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
  info.pNext = nullptr;
  info.flags = flags;
  info.queueFamilyIndex = queue_family_index;

  VkResult res = vkCreateCommandPool(device, &info, nullptr, &obj->pool);
  BAIL_ON_VK_BAD_RESULT_NO_RETURN(res, "failed to create command pool");

  return obj;
}

IVkCommandBuffer allocate_command_buffer(IVkCommandPool pool,
                                         VkCommandBufferLevel level) {
  VkCommandBuffer cmdbuf{VK_NULL_HANDLE};

  if (level == VK_COMMAND_BUFFER_LEVEL_PRIMARY && pool->free_primary.size()) {
    cmdbuf = pool->free_primary.top();
    pool->free_primary.pop();
  } else if (level == VK_COMMAND_BUFFER_LEVEL_SECONDARY &&
             pool->free_secondary.size()) {
    cmdbuf = pool->free_secondary.top();
    pool->free_secondary.pop();
  } else {
    VkCommandBufferAllocateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    info.pNext = nullptr;
    info.commandPool = pool->pool;
    info.level = level;
    info.commandBufferCount = 1;

    VkResult res = vkAllocateCommandBuffers(pool->device, &info, &cmdbuf);
    if (res != VK_SUCCESS) {
      return nullptr;
    }
  }

  IVkCommandBuffer obj = std::make_shared<DeviceObjVkCommandBuffer>();
  obj->device = pool->device;
  obj->level = level;
  obj->ref_pool = pool;
  obj->buffer = cmdbuf;

  return obj;
}

IVkRenderPass create_render_pass(VkDevice device,
                                 VkRenderPassCreateInfo *create_info) {
  IVkRenderPass obj = std::make_shared<DeviceObjVkRenderPass>();
  obj->device = device;
  VkResult res =
      vkCreateRenderPass(device, create_info, nullptr, &obj->renderpass);
  BAIL_ON_VK_BAD_RESULT_NO_RETURN(res, "failed to create render pass");
  return obj;
}

IVkPipelineLayout create_pipeline_layout(
    VkDevice device,
    std::vector<IVkDescriptorSetLayout> &set_layouts,
    uint32_t push_constant_range_count,
    VkPushConstantRange *push_constant_ranges) {
  IVkPipelineLayout obj = std::make_shared<DeviceObjVkPipelineLayout>();
  obj->device = device;
  obj->ref_desc_layouts = set_layouts;

  std::vector<VkDescriptorSetLayout> layouts;
  layouts.reserve(set_layouts.size());
  for (auto &l : set_layouts) {
    layouts.push_back(l->layout);
  }

  VkPipelineLayoutCreateInfo info{};
  info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
  info.pNext = nullptr;
  info.setLayoutCount = uint32_t(layouts.size());
  info.pSetLayouts = layouts.data();
  info.pushConstantRangeCount = push_constant_range_count;
  info.pPushConstantRanges = push_constant_ranges;

  VkResult res = vkCreatePipelineLayout(device, &info, nullptr, &obj->layout);
  BAIL_ON_VK_BAD_RESULT_NO_RETURN(res, "failed to create pipeline layout");

  return obj;
}

IVkPipelineCache create_pipeline_cache(VkDevice device,
                                       VkPipelineCacheCreateFlags flags,
                                       size_t initial_size,
                                       const void *initial_data) {
  IVkPipelineCache obj = std::make_shared<DeviceObjVkPipelineCache>();
  obj->device = device;

  VkPipelineCacheCreateInfo info{};
  info.sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO;
  info.pNext = nullptr;
  info.flags = flags;
  info.initialDataSize = initial_size;
  info.pInitialData = initial_data;

  VkResult res = vkCreatePipelineCache(device, &info, nullptr, &obj->cache);
  BAIL_ON_VK_BAD_RESULT_NO_RETURN(res, "failed to create pipeline cache");

  return obj;
}

IVkPipeline create_compute_pipeline(VkDevice device,
                                    VkPipelineCreateFlags flags,
                                    VkPipelineShaderStageCreateInfo &stage,
                                    IVkPipelineLayout layout,
                                    IVkPipelineCache cache,
                                    IVkPipeline base_pipeline) {
  IVkPipeline obj = std::make_shared<DeviceObjVkPipeline>();
  obj->device = device;
  obj->ref_layout = layout;
  obj->ref_cache = cache;

  VkComputePipelineCreateInfo info{};
  info.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
  info.pNext = nullptr;
  info.flags = flags;
  info.stage = stage;
  info.layout = layout->layout;
  if (base_pipeline) {
    info.basePipelineHandle = base_pipeline->pipeline;
    info.basePipelineIndex = -1;
  } else {
    info.basePipelineHandle = VK_NULL_HANDLE;
    info.basePipelineIndex = 0;
  }

  VkResult res =
      vkCreateComputePipelines(device, cache ? cache->cache : VK_NULL_HANDLE, 1,
                               &info, nullptr, &obj->pipeline);
  RHI_THROW_UNLESS(res == VK_SUCCESS,
                   std::runtime_error("vkCreateComputePipelines failed"));

  return obj;
}

IVkPipeline create_graphics_pipeline(VkDevice device,
                                     VkGraphicsPipelineCreateInfo *create_info,
                                     IVkRenderPass renderpass,
                                     IVkPipelineLayout layout,
                                     IVkPipelineCache cache,
                                     IVkPipeline base_pipeline) {
  IVkPipeline obj = std::make_shared<DeviceObjVkPipeline>();
  obj->device = device;
  obj->ref_layout = layout;
  obj->ref_cache = cache;
  obj->ref_renderpass = renderpass;

  create_info->renderPass = renderpass->renderpass;
  create_info->layout = layout->layout;

  if (base_pipeline) {
    create_info->basePipelineHandle = base_pipeline->pipeline;
    create_info->basePipelineIndex = -1;
  } else {
    create_info->basePipelineHandle = VK_NULL_HANDLE;
    create_info->basePipelineIndex = 0;
  }

  VkResult res =
      vkCreateGraphicsPipelines(device, cache ? cache->cache : VK_NULL_HANDLE,
                                1, create_info, nullptr, &obj->pipeline);
  BAIL_ON_VK_BAD_RESULT_NO_RETURN(res, "failed to create graphics pipeline");

  return obj;
}

IVkPipeline create_graphics_pipeline_dynamic(
    VkDevice device,
    VkGraphicsPipelineCreateInfo *create_info,
    VkPipelineRenderingCreateInfoKHR *rendering_info,
    IVkPipelineLayout layout,
    IVkPipelineCache cache,
    IVkPipeline base_pipeline) {
  IVkPipeline obj = std::make_shared<DeviceObjVkPipeline>();
  obj->device = device;
  obj->ref_layout = layout;
  obj->ref_cache = cache;
  obj->ref_renderpass = nullptr;

  create_info->pNext = rendering_info;
  create_info->layout = layout->layout;

  if (base_pipeline) {
    create_info->basePipelineHandle = base_pipeline->pipeline;
    create_info->basePipelineIndex = -1;
  } else {
    create_info->basePipelineHandle = VK_NULL_HANDLE;
    create_info->basePipelineIndex = 0;
  }

  VkResult res =
      vkCreateGraphicsPipelines(device, cache ? cache->cache : VK_NULL_HANDLE,
                                1, create_info, nullptr, &obj->pipeline);
  BAIL_ON_VK_BAD_RESULT_NO_RETURN(res, "failed to create graphics pipeline");

  return obj;
}

IVkPipeline create_raytracing_pipeline(
    VkDevice device,
    VkRayTracingPipelineCreateInfoKHR *create_info,
    IVkPipelineLayout layout,
    std::vector<IVkPipeline> &pipeline_libraries,
    VkDeferredOperationKHR deferredOperation,
    IVkPipelineCache cache,
    IVkPipeline base_pipeline) {
  IVkPipeline obj = std::make_shared<DeviceObjVkPipeline>();
  obj->device = device;
  obj->ref_layout = layout;
  obj->ref_cache = cache;
  obj->ref_pipeline_libraries = pipeline_libraries;

  create_info->layout = layout->layout;

  if (base_pipeline) {
    create_info->basePipelineHandle = base_pipeline->pipeline;
    create_info->basePipelineIndex = -1;
  } else {
    create_info->basePipelineHandle = VK_NULL_HANDLE;
    create_info->basePipelineIndex = 0;
  }

  PFN_vkCreateRayTracingPipelinesKHR create_raytracing_pipeline_khr =
      PFN_vkCreateRayTracingPipelinesKHR(vkGetInstanceProcAddr(
          taichi::lang::vulkan::VulkanLoader::instance().get_instance(),
          "vkCreateRayTracingPipelinesKHR"));

  VkResult res = create_raytracing_pipeline_khr(
      device, deferredOperation, cache ? cache->cache : VK_NULL_HANDLE, 1,
      create_info, nullptr, &obj->pipeline);
  BAIL_ON_VK_BAD_RESULT_NO_RETURN(res, "failed to create raytracing pipeline");

  return obj;
}

IVkSampler create_sampler(VkDevice device, const VkSamplerCreateInfo &info) {
  IVkSampler sampler = std::make_shared<DeviceObjVkSampler>();
  sampler->device = device;

  BAIL_ON_VK_BAD_RESULT_NO_RETURN(
      vkCreateSampler(device, &info, nullptr, &sampler->sampler),
      "failed to create texture sampler!");

  return sampler;
}

IVkImage create_image(VkDevice device,
                      VmaAllocator allocator,
                      VkImageCreateInfo *image_info,
                      VmaAllocationCreateInfo *alloc_info) {
  IVkImage image = std::make_shared<DeviceObjVkImage>();
  image->device = device;
  image->allocator = allocator;
  image->format = image_info->format;
  image->type = image_info->imageType;
  image->width = image_info->extent.width;
  image->height = image_info->extent.height;
  image->depth = image_info->extent.depth;
  image->mip_levels = image_info->mipLevels;
  image->array_layers = image_info->arrayLayers;
  image->usage = image_info->usage;

  VkResult res = vmaCreateImage(allocator, image_info, alloc_info,
                                &image->image, &image->allocation, nullptr);
  BAIL_ON_VK_BAD_RESULT_NO_RETURN(res, "failed to create image");

  return image;
}

IVkImage create_image(VkDevice device,
                      VkImage image,
                      VkFormat format,
                      VkImageType type,
                      VkExtent3D extent,
                      uint32_t mip_levels,
                      uint32_t array_layers,
                      VkImageUsageFlags usage) {
  IVkImage obj = std::make_shared<DeviceObjVkImage>();
  obj->device = device;
  obj->image = image;
  obj->format = format;
  obj->type = type;
  obj->width = extent.width;
  obj->height = extent.height;
  obj->depth = extent.depth;
  obj->mip_levels = mip_levels;
  obj->array_layers = array_layers;
  obj->usage = usage;

  return obj;
}

IVkImageView create_image_view(VkDevice device,
                               IVkImage image,
                               VkImageViewCreateInfo *create_info) {
  IVkImageView view = std::make_shared<DeviceObjVkImageView>();
  view->device = device;
  view->ref_image = image;
  view->subresource_range = create_info->subresourceRange;
  view->type = create_info->viewType;

  create_info->image = image->image;

  VkResult res = vkCreateImageView(device, create_info, nullptr, &view->view);
  BAIL_ON_VK_BAD_RESULT_NO_RETURN(res, "failed to create image view");

  return view;
}

IVkFramebuffer create_framebuffer(VkFramebufferCreateFlags flags,
                                  IVkRenderPass renderpass,
                                  const std::vector<IVkImageView> &attachments,
                                  uint32_t width,
                                  uint32_t height,
                                  uint32_t layers,
                                  void *pnext) {
  IVkFramebuffer obj = std::make_shared<DeviceObjVkFramebuffer>();
  obj->device = renderpass->device;
  obj->ref_attachments = attachments;
  obj->ref_renderpass = renderpass;

  std::vector<VkImageView> views(attachments.size());
  for (int i = 0; i < attachments.size(); i++) {
    views[i] = attachments[i]->view;
  }

  VkFramebufferCreateInfo info{};
  info.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
  info.pNext = pnext;
  info.flags = flags;
  info.renderPass = renderpass->renderpass;
  info.attachmentCount = uint32_t(attachments.size());
  info.pAttachments = views.data();
  info.width = width;
  info.height = height;
  info.layers = layers;

  VkResult res = vkCreateFramebuffer(renderpass->device, &info, nullptr,
                                     &obj->framebuffer);
  BAIL_ON_VK_BAD_RESULT_NO_RETURN(res, "failed to create framebuffer");

  return obj;
}

IVkBuffer create_buffer(VkDevice device,
                        VmaAllocator allocator,
                        VkBufferCreateInfo *buffer_info,
                        VmaAllocationCreateInfo *alloc_info) {
  IVkBuffer buffer = std::make_shared<DeviceObjVkBuffer>();
  buffer->device = device;
  buffer->allocator = allocator;
  buffer->usage = buffer_info->usage;

  VkResult res = vmaCreateBuffer(allocator, buffer_info, alloc_info,
                                 &buffer->buffer, &buffer->allocation, nullptr);
  if (res == VK_ERROR_OUT_OF_DEVICE_MEMORY ||
      res == VK_ERROR_OUT_OF_HOST_MEMORY) {
    return nullptr;
  }
  BAIL_ON_VK_BAD_RESULT_NO_RETURN(res, "failed to create buffer");

  return buffer;
}

IVkBuffer create_buffer(VkDevice device,
                        VkBuffer buffer,
                        VkBufferUsageFlags usage) {
  IVkBuffer obj = std::make_shared<DeviceObjVkBuffer>();
  obj->device = device;
  obj->buffer = buffer;
  obj->usage = usage;

  return obj;
}

IVkBufferView create_buffer_view(IVkBuffer buffer,
                                 VkBufferViewCreateFlags flags,
                                 VkFormat format,
                                 VkDeviceSize offset,
                                 VkDeviceSize range) {
  IVkBufferView view = std::make_shared<DeviceObjVkBufferView>();
  view->device = buffer->device;
  view->ref_buffer = buffer;
  view->format = format;
  view->offset = offset;
  view->range = range;

  VkBufferViewCreateInfo info{};
  info.sType = VK_STRUCTURE_TYPE_BUFFER_VIEW_CREATE_INFO;
  info.pNext = nullptr;
  info.flags = flags;
  info.buffer = buffer->buffer;
  info.format = format;
  info.offset = offset;
  info.range = range;

  VkResult res =
      vkCreateBufferView(buffer->device, &info, nullptr, &view->view);
  BAIL_ON_VK_BAD_RESULT_NO_RETURN(res, "failed to create buffer view");

  return view;
}

IVkAccelerationStructureKHR create_acceleration_structure(
    VkAccelerationStructureCreateFlagsKHR flags,
    IVkBuffer buffer,
    VkDeviceSize offset,
    VkDeviceSize size,
    VkAccelerationStructureTypeKHR type) {
  IVkAccelerationStructureKHR obj =
      std::make_shared<DeviceObjVkAccelerationStructureKHR>();
  obj->device = buffer->device;
  obj->ref_buffer = buffer;
  obj->offset = offset;
  obj->size = size;
  obj->type = type;

  VkAccelerationStructureCreateInfoKHR info{};
  info.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR;
  info.pNext = nullptr;
  info.createFlags = flags;
  info.buffer = buffer->buffer;
  info.offset = offset;
  info.size = size;
  info.type = type;
  info.deviceAddress = 0;

  PFN_vkCreateAccelerationStructureKHR create_acceleration_structure_khr =
      PFN_vkCreateAccelerationStructureKHR(vkGetInstanceProcAddr(
          taichi::lang::vulkan::VulkanLoader::instance().get_instance(),
          "vkCreateAccelerationStructureKHR"));

  VkResult res = create_acceleration_structure_khr(buffer->device, &info,
                                                   nullptr, &obj->accel);
  BAIL_ON_VK_BAD_RESULT_NO_RETURN(res,
                                  "failed to create acceleration structure");

  return obj;
}

IVkQueryPool create_query_pool(VkDevice device) {
  VkQueryPoolCreateInfo info{};
  info.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
  info.pNext = nullptr;
  info.queryCount = 2;
  info.queryType = VK_QUERY_TYPE_TIMESTAMP;

  VkQueryPool query_pool;
  VkResult res = vkCreateQueryPool(device, &info, nullptr, &query_pool);
  BAIL_ON_VK_BAD_RESULT_NO_RETURN(res, "failed to create query pool");

  IVkQueryPool obj = std::make_shared<DeviceObjVkQueryPool>();
  obj->device = device;
  obj->query_pool = query_pool;

  return obj;
}

}  // namespace vkapi
