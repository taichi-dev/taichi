#include "taichi/rhi/vulkan/vulkan_device_creator.h"

#include <iostream>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <unordered_map>
#include <vector>
#include <array>
#include <set>

#include "taichi/rhi/vulkan/vulkan_utils.h"
#include "taichi/rhi/vulkan/vulkan_loader.h"
#include "taichi/rhi/vulkan/vulkan_device.h"
#include "taichi/rhi/vulkan/vulkan_event.h"
#include "taichi/rhi/vulkan/vulkan_stream.h"
#include "taichi/rhi/vulkan/vulkan_resource_binder.h"
#include "taichi/common/core.h"

namespace taichi::lang {
namespace vulkan {

struct VulkanDevice::ThreadLocalStreams {
  unordered_map<std::thread::id, std::unique_ptr<VulkanStream>> map;
};

VulkanDevice::VulkanDevice()
    : compute_streams_(std::make_unique<ThreadLocalStreams>()),
      graphics_streams_(std::make_unique<ThreadLocalStreams>()) {
  caps_.set(DeviceCapability::spirv_version, 0x10000);
}

void VulkanDevice::init_vulkan_structs(Params &params) {
  instance_ = params.instance;
  device_ = params.device;
  physical_device_ = params.physical_device;
  compute_queue_ = params.compute_queue;
  compute_queue_family_index_ = params.compute_queue_family_index;
  graphics_queue_ = params.graphics_queue;
  graphics_queue_family_index_ = params.graphics_queue_family_index;

  create_vma_allocator();
  new_descriptor_pool();
}

VulkanDevice::~VulkanDevice() {
  // Note: Ideally whoever allocated the buffer & image should be responsible
  // for deallocation as well.
  // These manual deallocations work as last resort for the case where we
  // have GGUI window whose lifetime is controlled by Python but
  // shares the same underlying VulkanDevice with Program. In an extreme
  // edge case when Python shuts down and program gets destructed before
  // GGUI Window, buffers and images allocated through GGUI window won't
  // be properly deallocated before VulkanDevice destruction. This isn't
  // the most proper fix but is less intrusive compared to other
  // approaches.
  for (auto &alloc : allocations_) {
    alloc.second.buffer.reset();
  }
  for (auto &alloc : image_allocations_) {
    alloc.second.image.reset();
  }
  allocations_.clear();
  image_allocations_.clear();

  vkDeviceWaitIdle(device_);

  desc_pool_ = nullptr;

  framebuffer_pools_.clear();
  renderpass_pools_.clear();

  vmaDestroyAllocator(allocator_);
  vmaDestroyAllocator(allocator_export_);
}

std::unique_ptr<Pipeline> VulkanDevice::create_pipeline(
    const PipelineSourceDesc &src,
    std::string name) {
  TI_ASSERT(src.type == PipelineSourceType::spirv_binary &&
            src.stage == PipelineStageType::compute);
  TI_ERROR_IF(src.data == nullptr || src.size == 0,
              "pipeline source cannot be empty");

  SpirvCodeView code;
  code.data = (uint32_t *)src.data;
  code.size = src.size;
  code.stage = VK_SHADER_STAGE_COMPUTE_BIT;

  VulkanPipeline::Params params;
  params.code = {code};
  params.device = this;
  params.name = name;

  return std::make_unique<VulkanPipeline>(params);
}

std::unique_ptr<DeviceEvent> VulkanDevice::create_event() {
  return std::unique_ptr<DeviceEvent>(
      new VulkanDeviceEvent(vkapi::create_event(device_, 0)));
}

// #define TI_VULKAN_DEBUG_ALLOCATIONS

DeviceAllocation VulkanDevice::allocate_memory(const AllocParams &params) {
  DeviceAllocation handle;

  handle.device = this;
  handle.alloc_id = alloc_cnt_++;

  allocations_[handle.alloc_id] = {};
  AllocationInternal &alloc = allocations_[handle.alloc_id];

  VkBufferCreateInfo buffer_info{};
  buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  buffer_info.pNext = nullptr;
  buffer_info.size = params.size;
  // FIXME: How to express this in a backend-neutral way?
  buffer_info.usage =
      VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
  if (params.usage && AllocUsage::Storage) {
    buffer_info.usage |= VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
  }
  if (params.usage && AllocUsage::Uniform) {
    buffer_info.usage |= VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
  }
  if (params.usage && AllocUsage::Vertex) {
    buffer_info.usage |= VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
  }
  if (params.usage && AllocUsage::Index) {
    buffer_info.usage |= VK_BUFFER_USAGE_INDEX_BUFFER_BIT;
  }

  uint32_t queue_family_indices[] = {compute_queue_family_index_,
                                     graphics_queue_family_index_};

  if (compute_queue_family_index_ == graphics_queue_family_index_) {
    buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  } else {
    buffer_info.sharingMode = VK_SHARING_MODE_CONCURRENT;
    buffer_info.queueFamilyIndexCount = 2;
    buffer_info.pQueueFamilyIndices = queue_family_indices;
  }

  VkExternalMemoryBufferCreateInfo external_mem_buffer_create_info = {};
  external_mem_buffer_create_info.sType =
      VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO;
  external_mem_buffer_create_info.pNext = nullptr;

#ifdef _WIN64
  external_mem_buffer_create_info.handleTypes =
      VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;
#else
  external_mem_buffer_create_info.handleTypes =
      VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT_KHR;
#endif

  bool export_sharing = params.export_sharing && vk_caps_.external_memory;

  VmaAllocationCreateInfo alloc_info{};
  if (export_sharing) {
    buffer_info.pNext = &external_mem_buffer_create_info;
  }
#ifdef __APPLE__
  // weird behavior on apple: these flags are needed even if either read or
  // write is required
  if (params.host_read || params.host_write) {
#else
  if (params.host_read && params.host_write) {
#endif  //__APPLE__
    // This should be the unified memory on integrated GPUs
    alloc_info.requiredFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                               VK_MEMORY_PROPERTY_HOST_CACHED_BIT;
    alloc_info.preferredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
#ifdef __APPLE__
    // weird behavior on apple: if coherent bit is not set, then the memory
    // writes between map() and unmap() cannot be seen by gpu
    alloc_info.preferredFlags |= VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
#endif  //__APPLE__
  } else if (params.host_read) {
    alloc_info.requiredFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;
    alloc_info.preferredFlags = VK_MEMORY_PROPERTY_HOST_CACHED_BIT;
  } else if (params.host_write) {
    alloc_info.requiredFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;
    alloc_info.preferredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
  } else {
    alloc_info.requiredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
  }

  if (caps_.get(DeviceCapability::spirv_has_physical_storage_buffer)) {
    buffer_info.usage |= VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT_KHR;
  }

  alloc.buffer = vkapi::create_buffer(
      device_, export_sharing ? allocator_export_ : allocator_, &buffer_info,
      &alloc_info);
  vmaGetAllocationInfo(alloc.buffer->allocator, alloc.buffer->allocation,
                       &alloc.alloc_info);

#ifdef TI_VULKAN_DEBUG_ALLOCATIONS
  TI_TRACE("Allocate VK buffer {}, alloc_id={}", (void *)alloc.buffer,
           handle.alloc_id);
#endif

  if (caps_.get(DeviceCapability::spirv_has_physical_storage_buffer)) {
    VkBufferDeviceAddressInfoKHR info{};
    info.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO_KHR;
    info.buffer = alloc.buffer->buffer;
    info.pNext = nullptr;
    alloc.addr = vkGetBufferDeviceAddressKHR(device_, &info);
  }

  return handle;
}

void VulkanDevice::dealloc_memory(DeviceAllocation handle) {
  auto map_pair = allocations_.find(handle.alloc_id);

  TI_ASSERT_INFO(map_pair != allocations_.end(),
                 "Invalid handle (double free?) {}", handle.alloc_id);

#ifdef TI_VULKAN_DEBUG_ALLOCATIONS
  AllocationInternal &alloc = map_pair->second;
  TI_TRACE("Dealloc VK buffer {}, alloc_id={}", (void *)alloc.buffer,
           handle.alloc_id);
#endif

  allocations_.erase(handle.alloc_id);
}

uint64_t VulkanDevice::get_memory_physical_pointer(DeviceAllocation handle) {
  const auto &alloc_int = allocations_.at(handle.alloc_id);
  return uint64_t(alloc_int.addr);
}

void *VulkanDevice::map_range(DevicePtr ptr, uint64_t size) {
  AllocationInternal &alloc_int = allocations_.at(ptr.alloc_id);

  TI_ASSERT_INFO(alloc_int.mapped == nullptr,
                 "Memory can not be mapped multiple times");

  if (alloc_int.buffer->allocator) {
    vmaMapMemory(alloc_int.buffer->allocator, alloc_int.buffer->allocation,
                 &alloc_int.mapped);
    alloc_int.mapped = (uint8_t *)(alloc_int.mapped) + ptr.offset;
  } else {
    vkMapMemory(device_, alloc_int.alloc_info.deviceMemory,
                alloc_int.alloc_info.offset + ptr.offset, size, 0,
                &alloc_int.mapped);
  }

  return alloc_int.mapped;
}

void *VulkanDevice::map(DeviceAllocation alloc) {
  AllocationInternal &alloc_int = allocations_.at(alloc.alloc_id);

  TI_ASSERT_INFO(alloc_int.mapped == nullptr,
                 "Memory can not be mapped multiple times");

  VkResult res;
  if (alloc_int.buffer->allocator) {
    res = vmaMapMemory(alloc_int.buffer->allocator,
                       alloc_int.buffer->allocation, &alloc_int.mapped);
  } else {
    res = vkMapMemory(device_, alloc_int.alloc_info.deviceMemory,
                      alloc_int.alloc_info.offset, alloc_int.alloc_info.size, 0,
                      &alloc_int.mapped);
  }
  if (alloc_int.mapped == nullptr || res == VK_ERROR_MEMORY_MAP_FAILED) {
    TI_ERROR(
        "cannot map memory, potentially because the memory is not "
        "accessible from the host: ensure your memory is allocated with "
        "`host_read=true` or `host_write=true` (or `host_access=true` in C++ "
        "wrapper)");
  }
  BAIL_ON_VK_BAD_RESULT(res, "failed to map memory for unknown reasons");

  return alloc_int.mapped;
}

void VulkanDevice::unmap(DevicePtr ptr) {
  AllocationInternal &alloc_int = allocations_.at(ptr.alloc_id);

  TI_ASSERT_INFO(alloc_int.mapped, "Memory is not mapped");

  if (alloc_int.buffer->allocator) {
    vmaUnmapMemory(alloc_int.buffer->allocator, alloc_int.buffer->allocation);
  } else {
    vkUnmapMemory(device_, alloc_int.alloc_info.deviceMemory);
  }

  alloc_int.mapped = nullptr;
}

void VulkanDevice::unmap(DeviceAllocation alloc) {
  AllocationInternal &alloc_int = allocations_.at(alloc.alloc_id);

  TI_ASSERT_INFO(alloc_int.mapped, "Memory is not mapped");

  if (alloc_int.buffer->allocator) {
    vmaUnmapMemory(alloc_int.buffer->allocator, alloc_int.buffer->allocation);
  } else {
    vkUnmapMemory(device_, alloc_int.alloc_info.deviceMemory);
  }

  alloc_int.mapped = nullptr;
}

void VulkanDevice::memcpy_internal(DevicePtr dst,
                                   DevicePtr src,
                                   uint64_t size) {
  // TODO: always create a queue specifically for transfer
  Stream *stream = get_compute_stream();
  std::unique_ptr<CommandList> cmd = stream->new_command_list();
  cmd->buffer_copy(dst, src, size);
  stream->submit_synced(cmd.get());
}

Stream *VulkanDevice::get_compute_stream() {
  auto tid = std::this_thread::get_id();
  auto &stream_map = compute_streams_->map;
  auto iter = stream_map.find(tid);
  if (iter == stream_map.end()) {
    stream_map[tid] = std::make_unique<VulkanStream>(
        *this, compute_queue_, compute_queue_family_index_);
    return stream_map.at(tid).get();
  }
  return iter->second.get();
}

Stream *VulkanDevice::get_graphics_stream() {
  auto tid = std::this_thread::get_id();
  auto &stream_map = graphics_streams_->map;
  auto iter = stream_map.find(tid);
  if (iter == stream_map.end()) {
    stream_map[tid] = std::make_unique<VulkanStream>(
        *this, graphics_queue_, graphics_queue_family_index_);
    return stream_map.at(tid).get();
  }
  return iter->second.get();
}

void VulkanDevice::wait_idle() {
  for (auto &[tid, stream] : compute_streams_->map) {
    stream->command_sync();
  }
  for (auto &[tid, stream] : graphics_streams_->map) {
    stream->command_sync();
  }
}

std::unique_ptr<Pipeline> VulkanDevice::create_raster_pipeline(
    const std::vector<PipelineSourceDesc> &src,
    const RasterParams &raster_params,
    const std::vector<VertexInputBinding> &vertex_inputs,
    const std::vector<VertexInputAttribute> &vertex_attrs,
    std::string name) {
  VulkanPipeline::Params params;
  params.code = {};
  params.device = this;
  params.name = name;

  for (auto src_desc : src) {
    SpirvCodeView &code = params.code.emplace_back();
    code.data = (uint32_t *)src_desc.data;
    code.size = src_desc.size;
    code.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    if (src_desc.stage == PipelineStageType::fragment) {
      code.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    } else if (src_desc.stage == PipelineStageType::vertex) {
      code.stage = VK_SHADER_STAGE_VERTEX_BIT;
    } else if (src_desc.stage == PipelineStageType::geometry) {
      code.stage = VK_SHADER_STAGE_GEOMETRY_BIT;
    } else if (src_desc.stage == PipelineStageType::tesselation_control) {
      code.stage = VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT;
    } else if (src_desc.stage == PipelineStageType::tesselation_eval) {
      code.stage = VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT;
    }
  }

  return std::make_unique<VulkanPipeline>(params, raster_params, vertex_inputs,
                                          vertex_attrs);
}

std::tuple<VkDeviceMemory, size_t, size_t>
VulkanDevice::get_vkmemory_offset_size(const DeviceAllocation &alloc) const {
  auto buffer_alloc = allocations_.find(alloc.alloc_id);
  if (buffer_alloc != allocations_.end()) {
    return std::make_tuple(buffer_alloc->second.alloc_info.deviceMemory,
                           buffer_alloc->second.alloc_info.offset,
                           buffer_alloc->second.alloc_info.size);
  } else {
    const ImageAllocInternal &image_alloc =
        image_allocations_.at(alloc.alloc_id);
    return std::make_tuple(image_alloc.alloc_info.deviceMemory,
                           image_alloc.alloc_info.offset,
                           image_alloc.alloc_info.size);
  }
}

vkapi::IVkBuffer VulkanDevice::get_vkbuffer(
    const DeviceAllocation &alloc) const {
  const AllocationInternal &alloc_int = allocations_.at(alloc.alloc_id);

  return alloc_int.buffer;
}

std::tuple<vkapi::IVkImage, vkapi::IVkImageView, VkFormat>
VulkanDevice::get_vk_image(const DeviceAllocation &alloc) const {
  const ImageAllocInternal &alloc_int = image_allocations_.at(alloc.alloc_id);

  return std::make_tuple(alloc_int.image, alloc_int.view,
                         alloc_int.image->format);
}

vkapi::IVkFramebuffer VulkanDevice::get_framebuffer(
    const VulkanFramebufferDesc &desc) {
  if (framebuffer_pools_.find(desc) != framebuffer_pools_.end()) {
    return framebuffer_pools_.at(desc);
  }

  vkapi::IVkFramebuffer framebuffer = vkapi::create_framebuffer(
      0, desc.renderpass, desc.attachments, desc.width, desc.height, 1);

  framebuffer_pools_.insert({desc, framebuffer});

  return framebuffer;
}

DeviceAllocation VulkanDevice::import_vkbuffer(vkapi::IVkBuffer buffer) {
  AllocationInternal alloc_int{};
  alloc_int.external = true;
  alloc_int.buffer = buffer;
  alloc_int.mapped = nullptr;
  if (caps_.get(DeviceCapability::spirv_has_physical_storage_buffer)) {
    VkBufferDeviceAddressInfoKHR info{};
    info.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
    info.buffer = buffer->buffer;
    info.pNext = nullptr;
    alloc_int.addr = vkGetBufferDeviceAddress(device_, &info);
  }

  DeviceAllocation alloc;
  alloc.device = this;
  alloc.alloc_id = alloc_cnt_++;

  allocations_[alloc.alloc_id] = alloc_int;

  return alloc;
}

DeviceAllocation VulkanDevice::import_vk_image(vkapi::IVkImage image,
                                               vkapi::IVkImageView view,
                                               VkImageLayout layout) {
  ImageAllocInternal alloc_int;
  alloc_int.external = true;
  alloc_int.image = image;
  alloc_int.view = view;
  alloc_int.view_lods.emplace_back(view);

  DeviceAllocation alloc;
  alloc.device = this;
  alloc.alloc_id = alloc_cnt_++;

  image_allocations_[alloc.alloc_id] = alloc_int;

  return alloc;
}

vkapi::IVkImageView VulkanDevice::get_vk_imageview(
    const DeviceAllocation &alloc) const {
  return std::get<1>(get_vk_image(alloc));
}

vkapi::IVkImageView VulkanDevice::get_vk_lod_imageview(
    const DeviceAllocation &alloc,
    int lod) const {
  return image_allocations_.at(alloc.alloc_id).view_lods[lod];
}

DeviceAllocation VulkanDevice::create_image(const ImageParams &params) {
  DeviceAllocation handle;
  handle.device = this;
  handle.alloc_id = alloc_cnt_++;

  image_allocations_[handle.alloc_id] = {};
  ImageAllocInternal &alloc = image_allocations_[handle.alloc_id];

  int num_mip_levels = 1;

  bool is_depth = params.format == BufferFormat::depth16 ||
                  params.format == BufferFormat::depth24stencil8 ||
                  params.format == BufferFormat::depth32f;

  VkImageCreateInfo image_info{};
  image_info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
  image_info.pNext = nullptr;
  if (params.dimension == ImageDimension::d1D) {
    image_info.imageType = VK_IMAGE_TYPE_1D;
  } else if (params.dimension == ImageDimension::d2D) {
    image_info.imageType = VK_IMAGE_TYPE_2D;
  } else if (params.dimension == ImageDimension::d3D) {
    image_info.imageType = VK_IMAGE_TYPE_3D;
  }
  image_info.extent.width = params.x;
  image_info.extent.height = params.y;
  image_info.extent.depth = params.z;
  image_info.mipLevels = num_mip_levels;
  image_info.arrayLayers = 1;
  image_info.format = buffer_format_ti_to_vk(params.format);
  image_info.tiling = VK_IMAGE_TILING_OPTIMAL;
  image_info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  image_info.usage =
      VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
  if (params.usage & ImageAllocUsage::Sampled) {
    image_info.usage |= VK_IMAGE_USAGE_SAMPLED_BIT;
  }

  if (is_depth) {
    if (params.usage & ImageAllocUsage::Storage) {
      image_info.usage |= VK_IMAGE_USAGE_STORAGE_BIT;
    }
    if (params.usage & ImageAllocUsage::Attachment) {
      image_info.usage |= VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
    }
  } else {
    if (params.usage & ImageAllocUsage::Storage) {
      image_info.usage |= VK_IMAGE_USAGE_STORAGE_BIT;
    }
    if (params.usage & ImageAllocUsage::Attachment) {
      image_info.usage |= VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    }
  }
  image_info.samples = VK_SAMPLE_COUNT_1_BIT;

  uint32_t queue_family_indices[] = {compute_queue_family_index_,
                                     graphics_queue_family_index_};

  if (compute_queue_family_index_ == graphics_queue_family_index_) {
    image_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  } else {
    image_info.sharingMode = VK_SHARING_MODE_CONCURRENT;
    image_info.queueFamilyIndexCount = 2;
    image_info.pQueueFamilyIndices = queue_family_indices;
  }

  bool export_sharing = params.export_sharing && vk_caps_.external_memory;

  VkExternalMemoryImageCreateInfo external_mem_image_create_info = {};
  if (export_sharing) {
    external_mem_image_create_info.sType =
        VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO;
    external_mem_image_create_info.pNext = nullptr;

#ifdef _WIN64
    external_mem_image_create_info.handleTypes =
        VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;
#else
    external_mem_image_create_info.handleTypes =
        VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT_KHR;
#endif
    image_info.pNext = &external_mem_image_create_info;
  }

  VmaAllocationCreateInfo alloc_info{};
  if (params.export_sharing) {
  }
  alloc_info.usage = VMA_MEMORY_USAGE_GPU_ONLY;

  alloc.image = vkapi::create_image(
      device_, export_sharing ? allocator_export_ : allocator_, &image_info,
      &alloc_info);
  vmaGetAllocationInfo(alloc.image->allocator, alloc.image->allocation,
                       &alloc.alloc_info);

  VkImageViewCreateInfo view_info{};
  view_info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
  view_info.pNext = nullptr;
  if (params.dimension == ImageDimension::d1D) {
    view_info.viewType = VK_IMAGE_VIEW_TYPE_1D;
  } else if (params.dimension == ImageDimension::d2D) {
    view_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
  } else if (params.dimension == ImageDimension::d3D) {
    view_info.viewType = VK_IMAGE_VIEW_TYPE_3D;
  }
  view_info.format = image_info.format;
  view_info.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
  view_info.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
  view_info.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
  view_info.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;
  view_info.subresourceRange.aspectMask =
      is_depth ? VK_IMAGE_ASPECT_DEPTH_BIT : VK_IMAGE_ASPECT_COLOR_BIT;
  view_info.subresourceRange.baseMipLevel = 0;
  view_info.subresourceRange.levelCount = num_mip_levels;
  view_info.subresourceRange.baseArrayLayer = 0;
  view_info.subresourceRange.layerCount = 1;

  alloc.view = vkapi::create_image_view(device_, alloc.image, &view_info);

  for (int i = 0; i < num_mip_levels; i++) {
    view_info.subresourceRange.baseMipLevel = i;
    view_info.subresourceRange.levelCount = 1;
    alloc.view_lods.push_back(
        vkapi::create_image_view(device_, alloc.image, &view_info));
  }

  if (params.initial_layout != ImageLayout::undefined) {
    image_transition(handle, ImageLayout::undefined, params.initial_layout);
  }

#ifdef TI_VULKAN_DEBUG_ALLOCATIONS
  TI_TRACE("Allocate VK image {}, alloc_id={}", (void *)alloc.image,
           handle.alloc_id);
#endif

  return handle;
}

void VulkanDevice::destroy_image(DeviceAllocation handle) {
  auto map_pair = image_allocations_.find(handle.alloc_id);

  TI_ASSERT_INFO(map_pair != image_allocations_.end(),
                 "Invalid handle (double free?) {}", handle.alloc_id);

  image_allocations_.erase(handle.alloc_id);
}

vkapi::IVkRenderPass VulkanDevice::get_renderpass(
    const VulkanRenderPassDesc &desc) {
  if (renderpass_pools_.find(desc) != renderpass_pools_.end()) {
    return renderpass_pools_.at(desc);
  }

  std::vector<VkAttachmentDescription> attachments;
  std::vector<VkAttachmentReference> color_attachments;

  VkAttachmentReference depth_attachment;

  uint32_t i = 0;
  for (auto [format, clear] : desc.color_attachments) {
    VkAttachmentDescription &description = attachments.emplace_back();
    description.flags = 0;
    description.format = format;
    description.samples = VK_SAMPLE_COUNT_1_BIT;
    description.loadOp =
        clear ? VK_ATTACHMENT_LOAD_OP_CLEAR : VK_ATTACHMENT_LOAD_OP_LOAD;
    description.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    description.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    description.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    description.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    description.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

    VkAttachmentReference &ref = color_attachments.emplace_back();
    ref.attachment = i;
    ref.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    i += 1;
  }

  if (desc.depth_attachment != VK_FORMAT_UNDEFINED) {
    VkAttachmentDescription &description = attachments.emplace_back();
    description.flags = 0;
    description.format = desc.depth_attachment;
    description.samples = VK_SAMPLE_COUNT_1_BIT;
    description.loadOp = desc.clear_depth ? VK_ATTACHMENT_LOAD_OP_CLEAR
                                          : VK_ATTACHMENT_LOAD_OP_LOAD;
    description.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    description.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    description.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    description.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    description.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

    depth_attachment.attachment = i;
    depth_attachment.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
  }

  VkSubpassDescription subpass{};
  subpass.flags = 0;
  subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
  subpass.inputAttachmentCount = 0;
  subpass.pInputAttachments = nullptr;
  subpass.colorAttachmentCount = color_attachments.size();
  subpass.pColorAttachments = color_attachments.data();
  subpass.pResolveAttachments = nullptr;
  subpass.pDepthStencilAttachment = desc.depth_attachment == VK_FORMAT_UNDEFINED
                                        ? nullptr
                                        : &depth_attachment;
  subpass.preserveAttachmentCount = 0;
  subpass.pPreserveAttachments = nullptr;

  VkRenderPassCreateInfo renderpass_info{};
  renderpass_info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
  renderpass_info.pNext = nullptr;
  renderpass_info.flags = 0;
  renderpass_info.attachmentCount = attachments.size();
  renderpass_info.pAttachments = attachments.data();
  renderpass_info.subpassCount = 1;
  renderpass_info.pSubpasses = &subpass;
  renderpass_info.dependencyCount = 0;
  renderpass_info.pDependencies = nullptr;

  vkapi::IVkRenderPass renderpass =
      vkapi::create_render_pass(device_, &renderpass_info);

  renderpass_pools_.insert({desc, renderpass});

  return renderpass;
}

vkapi::IVkDescriptorSetLayout VulkanDevice::get_desc_set_layout(
    VulkanResourceBinder::Set &set) {
  if (desc_set_layouts_.find(set) == desc_set_layouts_.end()) {
    std::vector<VkDescriptorSetLayoutBinding> bindings;
    for (auto &pair : set.bindings) {
      bindings.push_back(VkDescriptorSetLayoutBinding{
          /*binding=*/pair.first, pair.second.type, /*descriptorCount=*/1,
          VK_SHADER_STAGE_ALL,
          /*pImmutableSamplers=*/nullptr});
    }

    VkDescriptorSetLayoutCreateInfo create_info{};
    create_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    create_info.pNext = nullptr;
    create_info.flags = 0;
    create_info.bindingCount = bindings.size();
    create_info.pBindings = bindings.data();

    auto layout = vkapi::create_descriptor_set_layout(device_, &create_info);
    desc_set_layouts_[set] = layout;

    return layout;
  } else {
    return desc_set_layouts_.at(set);
  }
}

vkapi::IVkDescriptorSet VulkanDevice::alloc_desc_set(
    vkapi::IVkDescriptorSetLayout layout) {
  // TODO: Currently we assume the calling code has called get_desc_set_layout
  // before allocating a desc set. Either we should guard against this or
  // maintain this assumption in other parts of the VulkanBackend
  vkapi::IVkDescriptorSet set =
      vkapi::allocate_descriptor_sets(desc_pool_, layout);

  if (set == nullptr) {
    new_descriptor_pool();
    set = vkapi::allocate_descriptor_sets(desc_pool_, layout);
  }

  return set;
}

void VulkanDevice::create_vma_allocator() {
  VmaAllocatorCreateInfo allocatorInfo = {};
  allocatorInfo.vulkanApiVersion = vk_caps_.vk_api_version;
  allocatorInfo.physicalDevice = physical_device_;
  allocatorInfo.device = device_;
  allocatorInfo.instance = instance_;

  VolkDeviceTable table;
  VmaVulkanFunctions vk_vma_functions{nullptr};

  volkLoadDeviceTable(&table, device_);
  vk_vma_functions.vkGetPhysicalDeviceProperties =
      PFN_vkGetPhysicalDeviceProperties(vkGetInstanceProcAddr(
          volkGetLoadedInstance(), "vkGetPhysicalDeviceProperties"));
  vk_vma_functions.vkGetPhysicalDeviceMemoryProperties =
      PFN_vkGetPhysicalDeviceMemoryProperties(vkGetInstanceProcAddr(
          volkGetLoadedInstance(), "vkGetPhysicalDeviceMemoryProperties"));
  vk_vma_functions.vkAllocateMemory = table.vkAllocateMemory;
  vk_vma_functions.vkFreeMemory = table.vkFreeMemory;
  vk_vma_functions.vkMapMemory = table.vkMapMemory;
  vk_vma_functions.vkUnmapMemory = table.vkUnmapMemory;
  vk_vma_functions.vkFlushMappedMemoryRanges = table.vkFlushMappedMemoryRanges;
  vk_vma_functions.vkInvalidateMappedMemoryRanges =
      table.vkInvalidateMappedMemoryRanges;
  vk_vma_functions.vkBindBufferMemory = table.vkBindBufferMemory;
  vk_vma_functions.vkBindImageMemory = table.vkBindImageMemory;
  vk_vma_functions.vkGetBufferMemoryRequirements =
      table.vkGetBufferMemoryRequirements;
  vk_vma_functions.vkGetImageMemoryRequirements =
      table.vkGetImageMemoryRequirements;
  vk_vma_functions.vkCreateBuffer = table.vkCreateBuffer;
  vk_vma_functions.vkDestroyBuffer = table.vkDestroyBuffer;
  vk_vma_functions.vkCreateImage = table.vkCreateImage;
  vk_vma_functions.vkDestroyImage = table.vkDestroyImage;
  vk_vma_functions.vkCmdCopyBuffer = table.vkCmdCopyBuffer;
  vk_vma_functions.vkGetBufferMemoryRequirements2KHR =
      table.vkGetBufferMemoryRequirements2KHR;
  vk_vma_functions.vkGetImageMemoryRequirements2KHR =
      table.vkGetImageMemoryRequirements2KHR;
  vk_vma_functions.vkBindBufferMemory2KHR = table.vkBindBufferMemory2KHR;
  vk_vma_functions.vkBindImageMemory2KHR = table.vkBindImageMemory2KHR;
  vk_vma_functions.vkGetPhysicalDeviceMemoryProperties2KHR =
      (PFN_vkGetPhysicalDeviceMemoryProperties2KHR)(std::max(
          vkGetInstanceProcAddr(volkGetLoadedInstance(),
                                "vkGetPhysicalDeviceMemoryProperties2KHR"),
          vkGetInstanceProcAddr(volkGetLoadedInstance(),
                                "vkGetPhysicalDeviceMemoryProperties2")));
  vk_vma_functions.vkGetDeviceBufferMemoryRequirements =
      table.vkGetDeviceBufferMemoryRequirements;
  vk_vma_functions.vkGetDeviceImageMemoryRequirements =
      table.vkGetDeviceImageMemoryRequirements;

  allocatorInfo.pVulkanFunctions = &vk_vma_functions;

  if (caps_.get(DeviceCapability::spirv_has_physical_storage_buffer)) {
    allocatorInfo.flags |= VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT;
  }

  vmaCreateAllocator(&allocatorInfo, &allocator_);

  VkPhysicalDeviceMemoryProperties properties;
  vkGetPhysicalDeviceMemoryProperties(physical_device_, &properties);

  std::vector<VkExternalMemoryHandleTypeFlags> flags(
      properties.memoryTypeCount);

  for (int i = 0; i < properties.memoryTypeCount; i++) {
    auto flag = properties.memoryTypes[i].propertyFlags;
    if (flag & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) {
#ifdef _WIN64
      flags[i] = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;
#else
      flags[i] = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
#endif
    } else {
      flags[i] = 0;
    }
  }

  allocatorInfo.pTypeExternalMemoryHandleTypes = flags.data();

  vmaCreateAllocator(&allocatorInfo, &allocator_export_);
}

void VulkanDevice::new_descriptor_pool() {
  std::vector<VkDescriptorPoolSize> pool_sizes{
      {VK_DESCRIPTOR_TYPE_SAMPLER, 64},
      {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 256},
      {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 256},
      {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 256},
      {VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER, 256},
      {VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER, 256},
      {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 256},
      {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 512},
      {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 128},
      {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC, 128},
      {VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT, 128}};
  VkDescriptorPoolCreateInfo pool_info = {};
  pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
  pool_info.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
  pool_info.maxSets = 64;
  pool_info.poolSizeCount = pool_sizes.size();
  pool_info.pPoolSizes = pool_sizes.data();
  desc_pool_ = vkapi::create_descriptor_pool(device_, &pool_info);
}

}  // namespace vulkan
}  // namespace taichi::lang
