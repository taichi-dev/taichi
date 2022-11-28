#pragma once

#include "taichi/rhi/vulkan/vulkan_api.h"
// FIXME: (penguinliong) Remove this.
#include "taichi/rhi/vulkan/vulkan_pipeline.h"
// FIXME: (penguinliong) Remove this.
#include "taichi/rhi/vulkan/vulkan_command_list.h"

#include <memory>
#include <optional>

#include <taichi/rhi/vulkan/vulkan_utils.h>
#include <taichi/common/ref_counted_pool.h>

namespace taichi::lang {
namespace vulkan {

using std::unordered_map;

class VulkanDevice;
class VulkanResourceBinder;
class VulkanStream;

struct RenderPassDescHasher {
  std::size_t operator()(const VulkanRenderPassDesc &desc) const {
    // TODO: Come up with a better hash
    size_t hash = 0;
    for (auto pair : desc.color_attachments) {
      hash ^= (size_t(pair.first) + pair.second);
      hash = (hash << 3) || (hash >> 61);
    }
    hash ^= (size_t(desc.depth_attachment) + desc.clear_depth);
    hash = (hash << 3) || (hash >> 61);
    return hash;
  }
};

struct FramebufferDescHasher {
  std::size_t operator()(const VulkanFramebufferDesc &desc) const {
    size_t hash = 0;
    for (auto view : desc.attachments) {
      hash ^= size_t(view->view);
      hash = (hash << 3) || (hash >> 61);
    }
    hash ^= desc.width;
    hash ^= desc.height;
    hash ^= size_t(desc.renderpass->renderpass);
    return hash;
  }
};

struct VulkanCapabilities {
  uint32_t vk_api_version;
  bool physical_device_features2;
  bool external_memory;
  bool wide_line;
  bool surface;
  bool present;
};

class TI_DLL_EXPORT VulkanDevice : public GraphicsDevice {
 public:
  struct Params {
    PFN_vkGetInstanceProcAddr get_proc_addr{nullptr};
    VkInstance instance;
    VkPhysicalDevice physical_device;
    VkDevice device;
    VkQueue compute_queue;
    uint32_t compute_queue_family_index;
    VkQueue graphics_queue;
    uint32_t graphics_queue_family_index;
  };

  VulkanDevice();
  void init_vulkan_structs(Params &params);
  ~VulkanDevice() override;

  Arch arch() const override {
    return Arch::vulkan;
  }

  std::unique_ptr<Pipeline> create_pipeline(
      const PipelineSourceDesc &src,
      std::string name = "Pipeline") override;
  std::unique_ptr<DeviceEvent> create_event() override;

  DeviceAllocation allocate_memory(const AllocParams &params) override;
  void dealloc_memory(DeviceAllocation handle) override;

  uint64_t get_memory_physical_pointer(DeviceAllocation handle) override;

  // Mapping can fail and will return nullptr
  void *map_range(DevicePtr ptr, uint64_t size) override;
  void *map(DeviceAllocation alloc) override;

  void unmap(DevicePtr ptr) override;
  void unmap(DeviceAllocation alloc) override;

  // Strictly intra device copy
  void memcpy_internal(DevicePtr dst, DevicePtr src, uint64_t size) override;

  Stream *get_compute_stream() override;
  Stream *get_graphics_stream() override;

  void wait_idle() override;

  std::unique_ptr<Pipeline> create_raster_pipeline(
      const std::vector<PipelineSourceDesc> &src,
      const RasterParams &raster_params,
      const std::vector<VertexInputBinding> &vertex_inputs,
      const std::vector<VertexInputAttribute> &vertex_attrs,
      std::string name = "Pipeline") override;

  DeviceAllocation create_image(const ImageParams &params) override;
  void destroy_image(DeviceAllocation handle) override;

  // Vulkan specific functions
  VkInstance vk_instance() const {
    return instance_;
  }

  VkDevice vk_device() const {
    return device_;
  }

  VkPhysicalDevice vk_physical_device() const {
    return physical_device_;
  }

  uint32_t compute_queue_family_index() const {
    return compute_queue_family_index_;
  }

  uint32_t graphics_queue_family_index() const {
    return graphics_queue_family_index_;
  }

  VkQueue graphics_queue() const {
    return graphics_queue_;
  }

  VkQueue compute_queue() const {
    return compute_queue_;
  }

  std::tuple<VkDeviceMemory, size_t, size_t> get_vkmemory_offset_size(
      const DeviceAllocation &alloc) const;

  vkapi::IVkBuffer get_vkbuffer(const DeviceAllocation &alloc) const;

  std::tuple<vkapi::IVkImage, vkapi::IVkImageView, VkFormat> get_vk_image(
      const DeviceAllocation &alloc) const;

  DeviceAllocation import_vkbuffer(vkapi::IVkBuffer buffer);

  DeviceAllocation import_vk_image(vkapi::IVkImage image,
                                   vkapi::IVkImageView view,
                                   VkImageLayout layout);

  vkapi::IVkImageView get_vk_imageview(const DeviceAllocation &alloc) const;

  vkapi::IVkImageView get_vk_lod_imageview(const DeviceAllocation &alloc,
                                           int lod) const;

  vkapi::IVkRenderPass get_renderpass(const VulkanRenderPassDesc &desc);

  vkapi::IVkFramebuffer get_framebuffer(const VulkanFramebufferDesc &desc);

  vkapi::IVkDescriptorSetLayout get_desc_set_layout(
      VulkanResourceBinder::Set &set);
  vkapi::IVkDescriptorSet alloc_desc_set(vkapi::IVkDescriptorSetLayout layout);

  inline void set_current_caps(DeviceCapabilityConfig &&caps) {
    caps_ = std::move(caps);
  }
  const DeviceCapabilityConfig &get_current_caps() const override {
    return caps_;
  }

  constexpr VulkanCapabilities &vk_caps() {
    return vk_caps_;
  }
  constexpr const VulkanCapabilities &vk_caps() const {
    return vk_caps_;
  }

 private:
  friend VulkanSurface;

  void create_vma_allocator();
  void new_descriptor_pool();

  DeviceCapabilityConfig caps_;
  VulkanCapabilities vk_caps_;

  VkInstance instance_;
  VkDevice device_;
  VkPhysicalDevice physical_device_;
  VmaAllocator allocator_;
  VmaAllocator allocator_export_{nullptr};

  VkQueue compute_queue_;
  uint32_t compute_queue_family_index_;

  VkQueue graphics_queue_;
  uint32_t graphics_queue_family_index_;

  struct ThreadLocalStreams;
  std::unique_ptr<ThreadLocalStreams> compute_streams_{nullptr};
  std::unique_ptr<ThreadLocalStreams> graphics_streams_{nullptr};

  // Memory allocation
  struct AllocationInternal {
    bool external{false};
    VmaAllocationInfo alloc_info;
    vkapi::IVkBuffer buffer;
    void *mapped{nullptr};
    VkDeviceAddress addr{0};
  };

  unordered_map<uint32_t, AllocationInternal> allocations_;

  uint32_t alloc_cnt_ = 0;

  // Images / Image views
  struct ImageAllocInternal {
    bool external{false};
    VmaAllocationInfo alloc_info;
    vkapi::IVkImage image;
    vkapi::IVkImageView view;
    std::vector<vkapi::IVkImageView> view_lods;
  };

  unordered_map<uint32_t, ImageAllocInternal> image_allocations_;

  // Renderpass
  unordered_map<VulkanRenderPassDesc,
                vkapi::IVkRenderPass,
                RenderPassDescHasher>
      renderpass_pools_;
  unordered_map<VulkanFramebufferDesc,
                vkapi::IVkFramebuffer,
                FramebufferDescHasher>
      framebuffer_pools_;

  // Descriptors / Layouts / Pools
  unordered_map<VulkanResourceBinder::Set,
                vkapi::IVkDescriptorSetLayout,
                VulkanResourceBinder::SetLayoutHasher>
      desc_set_layouts_;
  vkapi::IVkDescriptorPool desc_pool_{nullptr};
};

}  // namespace vulkan
}  // namespace taichi::lang
