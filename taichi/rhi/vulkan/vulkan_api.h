#pragma once

#include "taichi/rhi/vulkan/vulkan_common.h"

#include <vk_mem_alloc.h>

#include <memory>
#include <vector>
#include <stack>
#include <unordered_map>

namespace vkapi {

struct DeviceObj {
  VkDevice device{VK_NULL_HANDLE};
  virtual ~DeviceObj() = default;
};
using IDeviceObj = std::shared_ptr<DeviceObj>;
IDeviceObj create_device_obj(VkDevice device);

// VkEvent
struct DeviceObjVkEvent : public DeviceObj {
  bool external{false};
  VkEvent event{VK_NULL_HANDLE};
  ~DeviceObjVkEvent() override;
};
using IVkEvent = std::shared_ptr<DeviceObjVkEvent>;
IVkEvent create_event(VkDevice device,
                      VkEventCreateFlags flags,
                      void *pnext = nullptr);

// VkSemaphore
struct DeviceObjVkSemaphore : public DeviceObj {
  VkSemaphore semaphore{VK_NULL_HANDLE};
  ~DeviceObjVkSemaphore() override;
};
using IVkSemaphore = std::shared_ptr<DeviceObjVkSemaphore>;
IVkSemaphore create_semaphore(VkDevice device,
                              VkSemaphoreCreateFlags flags,
                              void *pnext = nullptr);

// VkFence
struct DeviceObjVkFence : public DeviceObj {
  VkFence fence{VK_NULL_HANDLE};
  ~DeviceObjVkFence() override;
};
using IVkFence = std::shared_ptr<DeviceObjVkFence>;
IVkFence create_fence(VkDevice device,
                      VkFenceCreateFlags flags,
                      void *pnext = nullptr);

// VkDescriptorSetLayout
struct DeviceObjVkDescriptorSetLayout : public DeviceObj {
  VkDescriptorSetLayout layout{VK_NULL_HANDLE};
  ~DeviceObjVkDescriptorSetLayout() override;
};
using IVkDescriptorSetLayout = std::shared_ptr<DeviceObjVkDescriptorSetLayout>;
IVkDescriptorSetLayout create_descriptor_set_layout(
    VkDevice device,
    VkDescriptorSetLayoutCreateInfo *create_info);

// VkDescriptorPool
struct DeviceObjVkDescriptorPool : public DeviceObj {
  VkDescriptorPool pool{VK_NULL_HANDLE};
  // Can recycling of this actually be trivial?
  // std::unordered_multimap<VkDescriptorSetLayout, VkDescriptorSet> free_list;
  ~DeviceObjVkDescriptorPool() override;
};
using IVkDescriptorPool = std::shared_ptr<DeviceObjVkDescriptorPool>;
IVkDescriptorPool create_descriptor_pool(
    VkDevice device,
    VkDescriptorPoolCreateInfo *create_info);

// VkDescriptorSet
struct DeviceObjVkDescriptorSet : public DeviceObj {
  VkDescriptorSet set{VK_NULL_HANDLE};
  IVkDescriptorSetLayout ref_layout{nullptr};
  IVkDescriptorPool ref_pool{nullptr};
  std::unordered_map<uint32_t, IDeviceObj> ref_binding_objs;
  ~DeviceObjVkDescriptorSet() override;
};
using IVkDescriptorSet = std::shared_ptr<DeviceObjVkDescriptorSet>;
// Returns nullptr is pool is full
IVkDescriptorSet allocate_descriptor_sets(IVkDescriptorPool pool,
                                          IVkDescriptorSetLayout layout,
                                          void *pnext = nullptr);

// VkCommandPool
struct DeviceObjVkCommandPool : public DeviceObj {
  VkCommandPool pool{VK_NULL_HANDLE};
  uint32_t queue_family_index{0};
  std::stack<VkCommandBuffer> free_primary;
  std::stack<VkCommandBuffer> free_secondary;
  ~DeviceObjVkCommandPool() override;
};
using IVkCommandPool = std::shared_ptr<DeviceObjVkCommandPool>;
IVkCommandPool create_command_pool(VkDevice device,
                                   VkCommandPoolCreateFlags flags,
                                   uint32_t queue_family_index);

// VkCommandBuffer
// Should keep track of used objects in the ref_pool
struct DeviceObjVkCommandBuffer : public DeviceObj {
  VkCommandBuffer buffer{VK_NULL_HANDLE};
  VkCommandBufferLevel level{VK_COMMAND_BUFFER_LEVEL_PRIMARY};
  IVkCommandPool ref_pool{nullptr};
  std::vector<IDeviceObj> refs;
  ~DeviceObjVkCommandBuffer() override;
};
using IVkCommandBuffer = std::shared_ptr<DeviceObjVkCommandBuffer>;
IVkCommandBuffer allocate_command_buffer(
    IVkCommandPool pool,
    VkCommandBufferLevel level = VK_COMMAND_BUFFER_LEVEL_PRIMARY);

// VkRenderPass
struct DeviceObjVkRenderPass : public DeviceObj {
  VkRenderPass renderpass{VK_NULL_HANDLE};
  ~DeviceObjVkRenderPass() override;
};
using IVkRenderPass = std::shared_ptr<DeviceObjVkRenderPass>;
IVkRenderPass create_render_pass(VkDevice device,
                                 VkRenderPassCreateInfo *create_info);

// VkPipelineLayout
struct DeviceObjVkPipelineLayout : public DeviceObj {
  VkPipelineLayout layout{VK_NULL_HANDLE};
  std::vector<IVkDescriptorSetLayout> ref_desc_layouts;
  ~DeviceObjVkPipelineLayout() override;
};
using IVkPipelineLayout = std::shared_ptr<DeviceObjVkPipelineLayout>;
IVkPipelineLayout create_pipeline_layout(
    VkDevice device,
    std::vector<IVkDescriptorSetLayout> &set_layouts,
    uint32_t push_constant_range_count = 0,
    VkPushConstantRange *push_constant_ranges = nullptr);

// VkPipelineCache
struct DeviceObjVkPipelineCache : public DeviceObj {
  VkPipelineCache cache{VK_NULL_HANDLE};
  ~DeviceObjVkPipelineCache() override;
};
using IVkPipelineCache = std::shared_ptr<DeviceObjVkPipelineCache>;
IVkPipelineCache create_pipeline_cache(VkDevice device,
                                       VkPipelineCacheCreateFlags flags,
                                       size_t initial_size = 0,
                                       const void *initial_data = nullptr);

// VkPipeline
struct DeviceObjVkPipeline : public DeviceObj {
  VkPipeline pipeline{VK_NULL_HANDLE};
  IVkPipelineLayout ref_layout{nullptr};
  IVkRenderPass ref_renderpass{nullptr};
  IVkPipelineCache ref_cache{nullptr};
  std::vector<std::shared_ptr<DeviceObjVkPipeline>> ref_pipeline_libraries;
  ~DeviceObjVkPipeline() override;
};
using IVkPipeline = std::shared_ptr<DeviceObjVkPipeline>;
IVkPipeline create_compute_pipeline(VkDevice device,
                                    VkPipelineCreateFlags flags,
                                    VkPipelineShaderStageCreateInfo &stage,
                                    IVkPipelineLayout layout,
                                    IVkPipelineCache cache = nullptr,
                                    IVkPipeline base_pipeline = nullptr);
IVkPipeline create_graphics_pipeline(VkDevice device,
                                     VkGraphicsPipelineCreateInfo *create_info,
                                     IVkRenderPass renderpass,
                                     IVkPipelineLayout layout,
                                     IVkPipelineCache cache = nullptr,
                                     IVkPipeline base_pipeline = nullptr);
IVkPipeline create_raytracing_pipeline(
    VkDevice device,
    VkRayTracingPipelineCreateInfoKHR *create_info,
    IVkPipelineLayout layout,
    std::vector<IVkPipeline> &pipeline_libraries,
    VkDeferredOperationKHR deferredOperation = VK_NULL_HANDLE,
    IVkPipelineCache cache = nullptr,
    IVkPipeline base_pipeline = nullptr);

// VkImage
struct DeviceObjVkImage : public DeviceObj {
  VkImage image{VK_NULL_HANDLE};
  VkFormat format{VK_FORMAT_UNDEFINED};
  VkImageType type{VK_IMAGE_TYPE_2D};
  uint32_t width{1};
  uint32_t height{1};
  uint32_t depth{1};
  uint32_t mip_levels{1};
  uint32_t array_layers{1};
  VmaAllocator allocator{nullptr};
  VmaAllocation allocation{nullptr};
  ~DeviceObjVkImage() override;
};
using IVkImage = std::shared_ptr<DeviceObjVkImage>;
// Allocate image
IVkImage create_image(VkDevice device,
                      VmaAllocator allocator,
                      VkImageCreateInfo *image_info,
                      VmaAllocationCreateInfo *alloc_info);
// Importing external image
IVkImage create_image(VkDevice device, VkImage image);

// VkImageView
struct DeviceObjVkImageView : public DeviceObj {
  VkImageView view{VK_NULL_HANDLE};
  VkImageViewType type{VK_IMAGE_VIEW_TYPE_2D};
  VkImageSubresourceRange subresource_range{
      VK_IMAGE_ASPECT_COLOR_BIT | VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1};
  IVkImage ref_image{nullptr};
  ~DeviceObjVkImageView() override;
};
using IVkImageView = std::shared_ptr<DeviceObjVkImageView>;
IVkImageView create_image_view(VkDevice device,
                               IVkImage image,
                               VkImageViewCreateInfo *create_info);

// VkFramebuffer
struct DeviceObjVkFramebuffer : public DeviceObj {
  VkFramebuffer framebuffer{VK_NULL_HANDLE};
  uint32_t width{0};
  uint32_t height{0};
  uint32_t layers{1};
  std::vector<IVkImageView> ref_attachments;
  IVkRenderPass ref_renderpass{nullptr};
  ~DeviceObjVkFramebuffer() override;
};
using IVkFramebuffer = std::shared_ptr<DeviceObjVkFramebuffer>;
IVkFramebuffer create_framebuffer(VkFramebufferCreateFlags flags,
                                  IVkRenderPass renderpass,
                                  const std::vector<IVkImageView> &attachments,
                                  uint32_t width,
                                  uint32_t height,
                                  uint32_t layers = 1,
                                  void *pnext = nullptr);

// VkBuffer
struct DeviceObjVkBuffer : public DeviceObj {
  VkBuffer buffer{VK_NULL_HANDLE};
  size_t size{0};
  VkBufferUsageFlags usage{0};
  VmaAllocator allocator{nullptr};
  VmaAllocation allocation{nullptr};
  ~DeviceObjVkBuffer() override;
};
using IVkBuffer = std::shared_ptr<DeviceObjVkBuffer>;
// Allocate buffer
IVkBuffer create_buffer(VkDevice device,
                        VmaAllocator allocator,
                        VkBufferCreateInfo *buffer_info,
                        VmaAllocationCreateInfo *alloc_info);
// Importing external buffer
IVkBuffer create_buffer(VkDevice device,
                        VkBuffer buffer,
                        size_t size,
                        VkBufferUsageFlags usage);

// VkBufferView
struct DeviceObjVkBufferView : public DeviceObj {
  VkBufferView view{VK_NULL_HANDLE};
  VkFormat format{VK_FORMAT_UNDEFINED};
  VkDeviceSize offset{0};
  VkDeviceSize range{0};
  IVkBuffer ref_buffer{nullptr};
  ~DeviceObjVkBufferView() override;
};
using IVkBufferView = std::shared_ptr<DeviceObjVkBufferView>;
IVkBufferView create_buffer_view(IVkBuffer buffer,
                                 VkBufferViewCreateFlags flags,
                                 VkFormat format,
                                 VkDeviceSize offset,
                                 VkDeviceSize range);

// VkAccelerationStructureKHR
struct DeviceObjVkAccelerationStructureKHR : public DeviceObj {
  VkAccelerationStructureKHR accel{VK_NULL_HANDLE};
  VkAccelerationStructureTypeKHR type{
      VK_ACCELERATION_STRUCTURE_TYPE_GENERIC_KHR};
  VkDeviceSize offset{0};
  VkDeviceSize size{0};
  IVkBuffer ref_buffer{nullptr};
  ~DeviceObjVkAccelerationStructureKHR() override;
};
using IVkAccelerationStructureKHR =
    std::shared_ptr<DeviceObjVkAccelerationStructureKHR>;
IVkAccelerationStructureKHR create_acceleration_structure(
    VkAccelerationStructureCreateFlagsKHR flags,
    IVkBuffer buffer,
    VkDeviceSize offset,
    VkDeviceSize size,
    VkAccelerationStructureTypeKHR type);

// VkQueryPool
struct DeviceObjVkQueryPool : public DeviceObj {
  VkQueryPool query_pool{VK_NULL_HANDLE};
  ~DeviceObjVkQueryPool() override;
};
using IVkQueryPool = std::shared_ptr<DeviceObjVkQueryPool>;
IVkQueryPool create_query_pool(VkDevice device);

}  // namespace vkapi
